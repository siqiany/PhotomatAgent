"""Workflow state machine and application operations for unified VASP."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from collections.abc import Mapping
from typing import Any, cast

from photomatagent.errors import ToolExecutionError

from photomatagent.scientific.applications.vasp.unified.approvals import (
    ApprovalKind,
    ApprovalReceiptStore,
    pending_decision,
)
from photomatagent.scientific.applications.vasp.unified.executors import (
    CollectionResult,
    OperationResult,
    PreflightResult,
    RecoveryResult,
    ReportResult,
    ServiceResult,
    StatusResult,
    SubmissionResult,
)
from photomatagent.scientific.applications.vasp.unified.fingerprints import (
    execution_fingerprint,
    scientific_fingerprint,
)
from photomatagent.scientific.applications.vasp.unified.models import (
    MolecularScientificSpec,
    PeriodicScientificSpec,
    ReportRequest,
    UnifiedStage,
    UnifiedVaspManifest,
    UnifiedVaspRequest,
    VaspWorkflowKind,
    WorkflowEvent,
    WorkflowState,
)
from photomatagent.scientific.applications.vasp.unified.repository import (
    ManifestRepository,
)
from photomatagent.scientific.applications.vasp.unified.recovery import (
    ValidatedRestartProof,
    _verified_restart_proof,
)
from photomatagent.scientific.applications.vasp.unified.resources import (
    ResourceAuthorizationService,
    ResourceDecisionState,
    VaspResourcePlanner,
)
from photomatagent.scientific.applications.vasp.unified.router import (
    UnifiedVaspRouter,
)
from photomatagent.scientific.remote.registry import (
    JobLifecycleState,
    derive_request_id,
)

_ALLOWED_TRANSITIONS: dict[WorkflowState, set[WorkflowState]] = {
    WorkflowState.PLANNED: {
        WorkflowState.PREPARED,
        WorkflowState.FAILED,
        WorkflowState.AWAITING_RESOURCE_CONFIRMATION,
        WorkflowState.AWAITING_SCIENTIFIC_CONFIRMATION,
    },
    WorkflowState.PREPARED: {
        WorkflowState.PREFLIGHTED,
        WorkflowState.FAILED,
        WorkflowState.AWAITING_RESOURCE_CONFIRMATION,
        WorkflowState.AWAITING_SCIENTIFIC_CONFIRMATION,
    },
    WorkflowState.PREFLIGHTED: {
        WorkflowState.SUBMITTED,
        WorkflowState.FAILED,
        WorkflowState.AWAITING_RESOURCE_CONFIRMATION,
        WorkflowState.AWAITING_SCIENTIFIC_CONFIRMATION,
    },
    WorkflowState.SUBMITTED: {
        WorkflowState.RUNNING,
        # A short job can finish between the submit and the first status query,
        # so a scheduler-confirmed completion is reachable without an observed
        # RUNNING transition. Without this edge a finished job would be
        # reported as FAILED (or, before this fix, as RUNNING forever).
        WorkflowState.SCHEDULER_COMPLETED,
        WorkflowState.RECONCILING,
        WorkflowState.FAILED,
        WorkflowState.AWAITING_RESOURCE_CONFIRMATION,
        WorkflowState.AWAITING_SCIENTIFIC_CONFIRMATION,
    },
    WorkflowState.RUNNING: {
        WorkflowState.SCHEDULER_COMPLETED,
        WorkflowState.RECONCILING,
        WorkflowState.FAILED,
        WorkflowState.AWAITING_RESOURCE_CONFIRMATION,
        WorkflowState.AWAITING_SCIENTIFIC_CONFIRMATION,
    },
    WorkflowState.RECONCILING: {
        WorkflowState.SUBMITTED,
        WorkflowState.RUNNING,
        WorkflowState.SCHEDULER_COMPLETED,
        WorkflowState.FAILED,
        WorkflowState.AWAITING_RESOURCE_CONFIRMATION,
        WorkflowState.AWAITING_SCIENTIFIC_CONFIRMATION,
    },
    WorkflowState.AWAITING_RESOURCE_CONFIRMATION: {
        WorkflowState.PREFLIGHTED,
        WorkflowState.SUBMITTED,
        WorkflowState.PREPARED,
        WorkflowState.FAILED,
    },
    WorkflowState.AWAITING_SCIENTIFIC_CONFIRMATION: {
        WorkflowState.PREFLIGHTED,
        WorkflowState.SUBMITTED,
        WorkflowState.PREPARED,
        WorkflowState.FAILED,
    },
    WorkflowState.SCHEDULER_COMPLETED: {
        WorkflowState.VALIDATED,
        WorkflowState.VALIDATION_FAILED,
        WorkflowState.PREFLIGHTED,
        WorkflowState.FAILED,
    },
    WorkflowState.VALIDATED: {
        WorkflowState.PREFLIGHTED,
        WorkflowState.FAILED,
    },
    WorkflowState.VALIDATION_FAILED: set(),
    WorkflowState.FAILED: set(),
}


_TERMINAL_FAILED_STATES = {
    "FAILED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "CANCELLED",
    "NODE_FAIL",
}

# Lifecycle states that prove a stage's scheduler work finished successfully.
# ``COLLECTED``/``VALIDATED`` also imply a completed scheduler run, but they
# are handled by ``collect`` (scientific validation), never by ``status``.
_SCHEDULER_SUCCESS_STATES = {
    "COMPLETED",
    "COLLECTED",
    "VALIDATED",
}

# Lifecycle states that carry no scheduler decision at all: the stage was
# never submitted, or reconciliation could not find its job. They must not be
# treated as either failure or success.
_UNDECIDED_STATES = {
    "NOT_FOUND",
    "UNKNOWN",
    "UNKNOWN_RECONCILIATION_REQUIRED",
    "SUBMITTING",
    "",
}

# Explicit, shared raw-status -> WorkflowState mapping.  Unknown strings are
# never treated as scheduler success; they become RECONCILING with a
# diagnostic so a later submit cannot silently create a second attempt.
_STATUS_MAP: dict[str, WorkflowState] = {
    "PLANNED": WorkflowState.PLANNED,
    "PREPARED": WorkflowState.PREPARED,
    "PREFLIGHT_PASSED": WorkflowState.PREFLIGHTED,
    "PREFLIGHTED": WorkflowState.PREFLIGHTED,
    "SUBMITTED": WorkflowState.SUBMITTED,
    "PENDING": WorkflowState.SUBMITTED,
    "RUNNING": WorkflowState.RUNNING,
    "COMPLETED": WorkflowState.SCHEDULER_COMPLETED,
    "COLLECTED": WorkflowState.SCHEDULER_COMPLETED,
    "SCHEDULER_COMPLETED": WorkflowState.SCHEDULER_COMPLETED,
    "VALIDATED": WorkflowState.VALIDATED,
    "VALIDATION_FAILED": WorkflowState.VALIDATION_FAILED,
    "FAILED": WorkflowState.FAILED,
    "TIMEOUT": WorkflowState.FAILED,
    "OUT_OF_MEMORY": WorkflowState.FAILED,
    "CANCELLED": WorkflowState.FAILED,
    "NODE_FAIL": WorkflowState.FAILED,
}

_RECONCILING_RAW_STATES = {
    "UNKNOWN",
    "UNKNOWN_RECONCILIATION_REQUIRED",
    "SUBMITTING",
}
_TERMINAL_FAILURE_RAW_STATES = {
    "FAILED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "CANCELLED",
    "NODE_FAIL",
}


def _normalized_stage_state(value: object) -> str:
    return str(value or "").strip().upper()


def _map_observed_stage_state(
    raw: object,
    *,
    current_state: WorkflowState,
    has_request: bool,
) -> tuple[WorkflowState, str | None]:
    """Map one executor/status observation without upgrading unknown states.

    ``NOT_FOUND`` is the normal state for a stage that was never submitted.
    If the manifest already has a request id for that stage, the missing
    registry record is instead an ambiguity that requires reconciliation.
    """
    normalized = _normalized_stage_state(raw)
    if normalized in {"", "NOT_FOUND"}:
        if has_request:
            return (
                WorkflowState.RECONCILING,
                "submitted stage has no registry record; reconcile required",
            )
        if current_state in {WorkflowState.PREPARED, WorkflowState.PREFLIGHTED}:
            return current_state, None
        return WorkflowState.PLANNED, None
    if normalized in _RECONCILING_RAW_STATES:
        return (
            WorkflowState.RECONCILING,
            f"stage state {raw!r} is not a scheduler completion; reconcile required",
        )
    mapped = _STATUS_MAP.get(normalized)
    if mapped is not None:
        return mapped, None
    return (
        WorkflowState.RECONCILING,
        f"unrecognized stage state {raw!r}; reconcile required",
    )


def _observed_reconciliation_required(
    manifest: UnifiedVaspManifest,
    stage_states: Mapping[str, object],
) -> bool:
    """True when an observation must go through reconcile, not a new submit."""
    for name, raw in stage_states.items():
        stage = next((item for item in manifest.stages if item.name == name), None)
        if stage is None:
            continue
        mapped, _ = _map_observed_stage_state(
            raw,
            current_state=stage.state,
            has_request=stage.request_id is not None,
        )
        if mapped is WorkflowState.RECONCILING:
            return True
    return False


def _confirmed_terminal_failure(stage_states: dict[str, str]) -> bool:
    """True when every submitted stage ended in a scheduler-confirmed failure.

    A stage that reconcile could not find (``NOT_FOUND``) never created a
    usable job and does not block a fresh attempt. At least one stage must
    actually have been submitted, otherwise there is nothing to reset.

    This whole-workflow predicate is retained for compatibility; recovery now
    targets the specific stage that is blocking progress (see
    ``_stage_recovery_plan``), because a completed upstream must never hide a
    failed downstream stage.
    """
    submitted = [
        state for state in stage_states.values() if state != "NOT_FOUND"
    ]
    if not submitted:
        return False
    return all(state in _TERMINAL_FAILED_STATES for state in submitted)


# Recovery actions for one target stage.  Deliberately explicit so a caller
# never has to interpret a bare "RECONCILING" string.
RECOVERY_NONE = "NONE"
RECOVERY_READY_FOR_NEW_ATTEMPT = "READY_FOR_NEW_ATTEMPT"
RECOVERY_COLLECT_DIAGNOSTICS = "COLLECT_DIAGNOSTICS"
RECOVERY_COLLECT_UPSTREAM = "COLLECT_UPSTREAM"
RECOVERY_WAIT = "WAIT"
RECOVERY_RECONCILE = "RECONCILE"

# Why a stage has no usable registry record.  Only a stage that can be proven
# never to have been submitted may be treated as a fresh attempt; a bound
# stage whose record is missing (or whose registry query failed) must stay in
# reconciliation, because its remote job identity is unknown.
BINDING_NEVER_SUBMITTED = "NEVER_SUBMITTED"
BINDING_BOUND_RECORD_MISSING = "BOUND_BUT_RECORD_MISSING"
BINDING_REGISTRY_QUERY_FAILED = "REGISTRY_QUERY_FAILED"
BINDING_RECORD_FOUND = "RECORD_FOUND"

# Sentinel raw state for a stage that is bound to an execution identity but
# whose registry record cannot be read.
_RECORD_MISSING_STATE = "<record-missing>"

_RECOVERY_ACTIVE_STATES = {"SUBMITTED", "PENDING", "RUNNING"}
_RECOVERY_SUCCESS_STATES = {"COMPLETED", "COLLECTED", "VALIDATED"}
_RECOVERY_VALIDATION_STATES = {"VALIDATION_FAILED"}
# Registry states that prove a stage passed deterministic scientific
# validation.  A mere scheduler COMPLETED/COLLECTED is not one of them.
_RECORD_VALIDATED_STATES = {"VALIDATED"}


def _stage_is_bound(stage: UnifiedStage) -> bool:
    """True when a stage was already bound to an execution identity.

    A bound stage has a persisted request id or an execution fingerprint, or
    its workflow state proves an attempt left the local planning phase.  Such
    a stage can never be reclassified as "never submitted" from a missing
    registry record.
    """
    return bool(
        stage.request_id
        or stage.execution_fingerprint
        or stage.state
        in {
            WorkflowState.PREFLIGHTED,
            WorkflowState.SUBMITTED,
            WorkflowState.RUNNING,
            WorkflowState.RECONCILING,
            WorkflowState.SCHEDULER_COMPLETED,
            WorkflowState.VALIDATION_FAILED,
            WorkflowState.VALIDATED,
            WorkflowState.FAILED,
        }
    )


def _effective_stage_state(
    manifest_state: WorkflowState, record_state: str
) -> WorkflowState:
    """Reconcile the manifest's stage state with its registry record.

    A restart can leave the manifest at PLANNED while the registry already
    proves the stage advanced, so the registry fact wins for the states that
    change recovery behaviour.  An explicit manifest VALIDATED is never
    downgraded by an ordinary scheduler COMPLETED observation.
    """
    if manifest_state is WorkflowState.VALIDATED:
        return manifest_state
    if record_state in _RECORD_VALIDATED_STATES:
        return WorkflowState.VALIDATED
    if record_state in _SCHEDULER_SUCCESS_STATES:
        return WorkflowState.SCHEDULER_COMPLETED
    if record_state in _TERMINAL_FAILED_STATES:
        return WorkflowState.FAILED
    if record_state in _RECOVERY_ACTIVE_STATES:
        return WorkflowState.RUNNING
    if record_state in _RECOVERY_VALIDATION_STATES:
        return WorkflowState.VALIDATION_FAILED
    if record_state in _RECONCILING_RAW_STATES:
        return WorkflowState.RECONCILING
    return manifest_state


def _recovery_plan_fields(plan: dict[str, Any]) -> dict[str, Any]:
    """Flatten a stage recovery plan into caller-visible result data."""
    return {
        "target_stage": plan.get("target_stage"),
        "recovery_action": plan.get("recovery_action"),
        "blocked_reason": plan.get("blocked_reason"),
        "next_action": plan.get("next_action"),
        "diagnostics_required": bool(plan.get("diagnostics_required")),
        "submitted_before": bool(plan.get("submitted_before")),
    }


def _stage_target_recovery(
    manifest: UnifiedVaspManifest,
    result: RecoveryResult,
    service: "UnifiedVaspService",
) -> bool:
    """True when a stage-targeted terminal failure should reset the workflow.

    Requires reconcile to have produced real per-stage lifecycle facts (an
    executor without them keeps the legacy whole-workflow contract) and the
    target stage to be the one blocking progress with a scheduler-confirmed
    terminal failure.  A completed upstream no longer vetoes this reset.
    """
    if not result.stage_states:
        return False
    plan = service._stage_recovery_plan(manifest)
    if plan.get("recovery_action") != RECOVERY_READY_FOR_NEW_ATTEMPT:
        return False
    target = plan.get("target_stage")
    if not isinstance(target, str) or not target:
        return False
    return result.stage_states.get(target) in _TERMINAL_FAILED_STATES




_COLLECT_TRANSITIONS: dict[WorkflowState, set[WorkflowState]] = {
    # Collection can be reached while a stage is still active (the scheduler
    # finished but no status query ran first), so the validation outcomes are
    # allowed from the whole active range, not only from SCHEDULER_COMPLETED.
    WorkflowState.PLANNED: {WorkflowState.VALIDATION_FAILED, WorkflowState.RECONCILING},
    WorkflowState.PREPARED: {WorkflowState.VALIDATION_FAILED, WorkflowState.RECONCILING},
    WorkflowState.PREFLIGHTED: {
        WorkflowState.VALIDATED,
        WorkflowState.VALIDATION_FAILED,
        WorkflowState.PREFLIGHTED,
        WorkflowState.RUNNING,
        WorkflowState.RECONCILING,
    },
    WorkflowState.SUBMITTED: {
        WorkflowState.VALIDATED,
        WorkflowState.SCHEDULER_COMPLETED,
        WorkflowState.VALIDATION_FAILED,
        WorkflowState.RUNNING,
        WorkflowState.RECONCILING,
        # A study parent whose child stage progressed returns to PREFLIGHTED so
        # the next declared stage can be authorized (see ``collect``).
        WorkflowState.PREFLIGHTED,
    },
    WorkflowState.RUNNING: {
        WorkflowState.VALIDATED,
        WorkflowState.SCHEDULER_COMPLETED,
        WorkflowState.VALIDATION_FAILED,
        WorkflowState.RECONCILING,
    },
    WorkflowState.RECONCILING: {
        WorkflowState.VALIDATED,
        WorkflowState.SCHEDULER_COMPLETED,
        WorkflowState.VALIDATION_FAILED,
        WorkflowState.RUNNING,
        WorkflowState.RECONCILING,
    },
    WorkflowState.SCHEDULER_COMPLETED: {
        WorkflowState.VALIDATED,
        WorkflowState.VALIDATION_FAILED,
        WorkflowState.SCHEDULER_COMPLETED,
        WorkflowState.RECONCILING,
    },
    WorkflowState.VALIDATED: {
        WorkflowState.VALIDATED,
        WorkflowState.SCHEDULER_COMPLETED,
        WorkflowState.VALIDATION_FAILED,
        WorkflowState.PREFLIGHTED,
        WorkflowState.RECONCILING,
    },
    WorkflowState.VALIDATION_FAILED: {
        WorkflowState.VALIDATED,
        WorkflowState.VALIDATION_FAILED,
        WorkflowState.SCHEDULER_COMPLETED,
        WorkflowState.RECONCILING,
    },
    WorkflowState.FAILED: {WorkflowState.VALIDATION_FAILED},
}


def _collect_transition_allowed(
    current: WorkflowState, target: WorkflowState
) -> bool:
    return target in _COLLECT_TRANSITIONS.get(current, set())


def _next_workflow_state(
    manifest: UnifiedVaspManifest,
    stage_states: dict[str, str],
) -> WorkflowState | None:
    """Translate observed stage states into the next workflow state.

    ``status`` and ``collect`` share one translation so a scheduler-confirmed
    completion can never be reported as ``RUNNING`` (which would make
    ``vasp.wait`` poll until the heat death of the cluster). Only stages that
    have an observation contribute; stages that were never submitted stay
    ``PLANNED``/``NOT_FOUND`` and cannot veto a completion.

    Returns ``None`` when nothing observable changed.
    """
    if not stage_states:
        return None
    states = [
        state for state in stage_states.values() if state not in _UNDECIDED_STATES
    ]
    if any(state in _TERMINAL_FAILED_STATES for state in states):
        return WorkflowState.FAILED
    if _observed_reconciliation_required(manifest, stage_states):
        return WorkflowState.RECONCILING
    if not states:
        return None
    if all(state in _SCHEDULER_SUCCESS_STATES for state in states):
        if manifest.state is not WorkflowState.VALIDATED:
            return WorkflowState.SCHEDULER_COMPLETED
        return None
    # Something is still active (SUBMITTED/PENDING/RUNNING) and nothing failed.
    return WorkflowState.RUNNING


def _dependency_errors(
    manifest: UnifiedVaspManifest, stage: UnifiedStage
) -> list[str]:
    """Return explicit reasons why a stage's dependencies are not ready."""
    errors: list[str] = []
    for dependency in stage.depends_on:
        candidate = next(
            (item for item in manifest.stages if item.name == dependency), None
        )
        if candidate is None:
            errors.append(
                f"required dependency {dependency!r} is missing from the manifest"
            )
        elif candidate.state is not WorkflowState.VALIDATED:
            errors.append(
                f"required dependency {dependency!r} is "
                f"{candidate.state.value}, not VALIDATED"
            )
    return errors


def _dependency_satisfied(
    manifest: UnifiedVaspManifest, stage: UnifiedStage
) -> bool:
    return not _dependency_errors(manifest, stage)


def _has_progressable_stage(manifest: UnifiedVaspManifest) -> bool:
    """True when a not-yet-run stage has all of its dependencies satisfied."""
    return any(
        stage.state is WorkflowState.PLANNED and _dependency_satisfied(manifest, stage)
        for stage in manifest.stages
    )


class UnifiedVaspService:
    """Narrow application service behind the public vasp.* deferred tools."""

    def __init__(
        self,
        repository: ManifestRepository,
        approvals: ApprovalReceiptStore,
        router: UnifiedVaspRouter,
        *,
        resources: ResourceAuthorizationService | None = None,
        planner: VaspResourcePlanner | None = None,
    ) -> None:
        self.repository = repository
        self.approvals = approvals
        self.router = router
        self.resources = resources or ResourceAuthorizationService(approvals)
        self.planner = planner or VaspResourcePlanner()

    def capabilities(self, workflow_kind: str | None = None) -> dict[str, Any]:
        """Return bounded configuration metadata without touching a backend."""
        supported = [kind.value for kind in VaspWorkflowKind]
        if workflow_kind is not None and workflow_kind not in supported:
            raise ValueError(f"unknown workflow kind {workflow_kind!r}")
        return {
            "ok": True,
            "workflow_kinds": supported,
            "filter": workflow_kind,
            "service": "unified_vasp",
        }

    async def wait(self, workflow_id: str) -> ServiceResult:
        """One bounded transport wait observation; callers own no poll loop."""
        return await self.status(workflow_id)

    # -- plan --------------------------------------------------------------

    def plan(self, request: UnifiedVaspRequest) -> UnifiedVaspManifest:
        manifest = self.repository.create(request)
        stages = self._initial_stages(manifest)
        updated = manifest.model_copy(update={"stages": stages})
        updated.scientific_fingerprint = scientific_fingerprint(
            updated.scientific_spec, updated.stages
        )
        self.repository.save(updated, expected_revision=0)
        return self.repository.load(updated.workflow_id)

    @staticmethod
    def _initial_stages(
        manifest: UnifiedVaspManifest,
    ) -> list[UnifiedStage]:
        kind = manifest.workflow_kind
        if kind == VaspWorkflowKind.PERIODIC:
            from photomatagent.scientific.applications.vasp.profiles import (
                periodic_stage_specs,
            )

            spec = cast(PeriodicScientificSpec, manifest.scientific_spec)
            return [
                UnifiedStage(
                    name=stage_spec.name,
                    depends_on=list(stage_spec.depends_on),
                    required_outputs=list(stage_spec.required_outputs),
                )
                for stage_spec in periodic_stage_specs(spec.profile)
            ]
        if kind == VaspWorkflowKind.MOLECULAR:
            return [
                UnifiedStage(
                    name=stage.name.value,
                    depends_on=(
                        [stage.depends_on.value]
                        if stage.depends_on is not None
                        else []
                    ),
                )
                for stage in cast(
                    MolecularScientificSpec, manifest.scientific_spec
                ).workflow.stages
            ]
        return [UnifiedStage(name="study")]

    # -- operations ---------------------------------------------------------

    async def prepare(self, workflow_id: str) -> ServiceResult:
        manifest = self._load(workflow_id)
        try:
            executor = self._executor(manifest)
        except RuntimeError as exc:
            return ServiceResult(
                ok=False,
                workflow_id=workflow_id,
                state=manifest.state,
                errors=[str(exc)],
            )
        self._transition(manifest, WorkflowState.PREPARED)
        try:
            result = await executor.prepare(manifest)
        except Exception as exc:
            return self._finish(
                manifest,
                WorkflowState.FAILED,
                OperationResult(
                    ok=False,
                    errors=[f"child prepare failed: {str(exc)[:240]}"],
                    evidence_gaps=["child prepare raised an error"],
                ),
            )
        if not result.ok:
            return self._finish(manifest, WorkflowState.FAILED, result)
        return self._save_and_result(manifest, result)

    async def preflight(self, workflow_id: str) -> ServiceResult:
        manifest = self._load(workflow_id)
        if manifest.state not in {
            WorkflowState.PLANNED,
            WorkflowState.PREPARED,
            WorkflowState.AWAITING_RESOURCE_CONFIRMATION,
            WorkflowState.AWAITING_SCIENTIFIC_CONFIRMATION,
            WorkflowState.SCHEDULER_COMPLETED,
            WorkflowState.VALIDATED,
        }:
            raise ValueError(
                f"preflight is not allowed from {manifest.state.value}"
            )
        self._transition(manifest, WorkflowState.PREFLIGHTED)
        try:
            executor = self._executor(manifest)
            result = await executor.preflight(manifest)
        except Exception as exc:
            return self._finish(
                manifest,
                WorkflowState.FAILED,
                PreflightResult(
                    ok=False,
                    passed=False,
                    errors=[f"child preflight failed: {str(exc)[:240]}"],
                    evidence_gaps=["child preflight raised an error"],
                ),
            )
        if not result.ok or not result.passed:
            manifest.state = WorkflowState.FAILED
            return self._save_and_result(manifest, result)
        # Preflight only the next dependency-satisfied stage.  This keeps
        # stage state authoritative in the child manifest and permits a
        # later stage to advance after collect/validation of its predecessor.
        for item in manifest.stages:
            if item.state is not WorkflowState.PLANNED:
                continue
            if _dependency_satisfied(manifest, item):
                item.state = WorkflowState.PREFLIGHTED
                break
        return self._save_and_result(manifest, result)

    async def submit(
        self, workflow_id: str, stage: str | None = None
    ) -> ServiceResult:
        manifest = self._load(workflow_id)
        if stage is None:
            target = self._next_actionable_stage(manifest)
            if target is None:
                # Keep the single-stage study/orchestration cursor and active
                # duplicate semantics; only an all-validated workflow has no
                # legal next action to submit.
                target = self._stage(manifest, None)
                if target.state is WorkflowState.VALIDATED:
                    pending = [
                        item.name
                        for item in manifest.stages
                        if item.state
                        not in {
                            WorkflowState.VALIDATED,
                            WorkflowState.FAILED,
                            WorkflowState.VALIDATION_FAILED,
                        }
                    ]
                    return ServiceResult(
                        ok=True,
                        workflow_id=workflow_id,
                        state=manifest.state,
                        data={
                            "duplicate": True,
                            "blocked_reason": (
                                "no dependency-ready stage to submit"
                            ),
                            "pending_stages": pending,
                        },
                        errors=[],
                    )
        else:
            target = self._stage(manifest, stage)
        dependency_errors = _dependency_errors(manifest, target)
        if dependency_errors:
            blocked_reason = "; ".join(dependency_errors)
            return ServiceResult(
                ok=False,
                workflow_id=workflow_id,
                state=manifest.state,
                errors=dependency_errors,
                data={
                    "blocked_reason": blocked_reason,
                    "pending_stages": list(target.depends_on),
                },
            )
        progression = (
            manifest.state is WorkflowState.VALIDATED
            and target.state in {WorkflowState.PLANNED, WorkflowState.PREFLIGHTED}
            and _dependency_satisfied(manifest, target)
        )
        if progression:
            manifest.state = WorkflowState.PREFLIGHTED
            target.state = WorkflowState.PREFLIGHTED
        elif manifest.state not in {
            WorkflowState.PREFLIGHTED,
            WorkflowState.AWAITING_RESOURCE_CONFIRMATION,
            WorkflowState.AWAITING_SCIENTIFIC_CONFIRMATION,
        }:
            if manifest.state in {
                WorkflowState.SUBMITTED,
                WorkflowState.RUNNING,
                WorkflowState.RECONCILING,
                WorkflowState.SCHEDULER_COMPLETED,
            }:
                # A whole-workflow "duplicate" is only true when the target
                # stage itself owns a live or completed job.  An executor with
                # a queryable registry that has no record for this stage means
                # the stage was never submitted and still needs preflight (the
                # historical iteration-9 false positive); an executor without
                # a registry keeps the legacy whole-workflow contract.
                if not self._executor_has_registry(manifest):
                    return ServiceResult(
                        ok=True,
                        workflow_id=workflow_id,
                        state=manifest.state,
                        data={
                            "duplicate": True,
                            "target_stage": target.name,
                            "registry_available": False,
                        },
                        errors=[],
                    )
                target_record = self._target_stage_record(manifest, target)
                recorded_state = (
                    target_record.state.value
                    if target_record is not None
                    and isinstance(
                        getattr(target_record, "state", None), JobLifecycleState
                    )
                    else ""
                )
                if target_record is None or not target_record.job_id:
                    if target_record is None and not target.request_id:
                        # The target stage was never bound to a request/job:
                        # whole-workflow progress from an earlier stage must
                        # not be reported as a duplicate.  Fall through so the
                        # normal submit path can preflight and authorize it.
                        pass
                    else:
                        return ServiceResult(
                            ok=False,
                            workflow_id=workflow_id,
                            state=manifest.state,
                            errors=[
                                f"stage {target.name!r} has no recorded job; "
                                "preflight is required before submitting"
                            ],
                            data={
                                "duplicate": False,
                                "needs_preflight": True,
                                "target_stage": target.name,
                                "blocked_reason": (
                                    f"workflow state {manifest.state.value} but "
                                    f"stage {target.name!r} has no job identity"
                                ),
                                "next_action": (
                                    "PREFLIGHT: this stage was never submitted; "
                                    "run vasp.preflight and resubmit through the "
                                    "normal gates"
                                ),
                            },
                        )
                    duplicate_target = False
                else:
                    duplicate_target = True
                if duplicate_target:
                    return ServiceResult(
                        ok=True,
                        workflow_id=workflow_id,
                        state=manifest.state,
                        data={
                            "duplicate": True,
                            "target_stage": target.name,
                            "recorded_stage_state": recorded_state,
                        },
                        errors=[],
                    )
            else:
                raise ValueError(
                    "submit is only allowed after a passing preflight; "
                    f"current state is {manifest.state.value}"
                )
        # ``target`` was resolved before the state gate so progression can
        # distinguish a new dependency-satisfied stage from an active retry.
        try:
            executor = self._executor(manifest)
        except RuntimeError as exc:
            return ServiceResult(
                ok=False,
                workflow_id=workflow_id,
                state=manifest.state,
                errors=[str(exc)],
            )
        prepare_stage_inputs = getattr(executor, "prepare_stage_inputs", None)
        if callable(prepare_stage_inputs):
            prepared_inputs = await prepare_stage_inputs(manifest, target)
            if not prepared_inputs.ok:
                blocked_errors = list(prepared_inputs.errors) or [
                    "stage input preparation failed"
                ]
                return ServiceResult(
                    ok=False,
                    workflow_id=workflow_id,
                    state=manifest.state,
                    errors=blocked_errors,
                    evidence_gaps=list(prepared_inputs.evidence_gaps),
                    data={
                        "blocked_reason": "; ".join(blocked_errors),
                        "pending_stages": [target.name],
                    },
                )
            attempt_inputs = (
                prepared_inputs.data.get("attempt_inputs")
                if isinstance(prepared_inputs.data, dict)
                else None
            )
            if isinstance(attempt_inputs, dict) and attempt_inputs:
                target.attempt_inputs = {
                    **target.attempt_inputs,
                    **attempt_inputs,
                }
                manifest = self._save(
                    manifest, expected_revision=manifest.revision
                )
                target = self._stage(manifest, target.name)
        resource = self.planner.recommend(manifest, target)
        manifest = self._persist_execution_identity(
            manifest, target, resource, target.attempt_inputs or None
        )
        target = self._stage(manifest, target.name)
        decision = self.resources.decide(manifest, target, resource)
        if decision.state is ResourceDecisionState.DENIED:
            return ServiceResult(
                ok=False,
                workflow_id=workflow_id,
                state=manifest.state,
                errors=list(decision.reasons),
                pending_decision=decision.pending_decision,
            )
        if decision.state is ResourceDecisionState.NEEDS_CONFIRMATION:
            manifest.state = WorkflowState.AWAITING_RESOURCE_CONFIRMATION
            manifest = self._save(manifest, expected_revision=manifest.revision)
            target = self._stage(manifest, target.name)
            # A decision must bind the revision that visibly awaits approval.
            # Rebuild it after that state change instead of trusting the
            # provisional decision generated before persistence.
            decision = self.resources.decide(manifest, target, resource)
            return ServiceResult(
                ok=False,
                workflow_id=workflow_id,
                state=manifest.state,
                errors=["resource confirmation required"],
                pending_decision=decision.pending_decision,
            )
        assert decision.effective is not None
        try:
            executor = self._executor(manifest)
            submission = await executor.submit(manifest, target, decision.effective)
        except Exception as exc:
            target.state = WorkflowState.FAILED
            return self._finish(
                manifest,
                WorkflowState.FAILED,
                SubmissionResult(
                    ok=False,
                    request_id=target.request_id or "unknown",
                    errors=[f"child submission failed: {str(exc)[:240]}"],
                    evidence_gaps=["child submission raised an error"],
                ),
            )
        if submission.needs_reconciliation:
            manifest.state = WorkflowState.RECONCILING
        elif submission.pending_decision is not None:
            manifest.state = (
                WorkflowState.AWAITING_RESOURCE_CONFIRMATION
                if submission.pending_decision.kind.value == "resource"
                else WorkflowState.AWAITING_SCIENTIFIC_CONFIRMATION
            )
        elif submission.ok:
            manifest.state = WorkflowState.SUBMITTED
            target.request_id = submission.request_id
            target.state = WorkflowState.SUBMITTED
        else:
            manifest.state = WorkflowState.FAILED
        return self._finish(manifest, manifest.state, submission)

    async def status(self, workflow_id: str) -> ServiceResult:
        manifest = self._load(workflow_id)
        try:
            executor = self._executor(manifest)
            result = await executor.status(manifest)
        except Exception as exc:
            manifest.state = WorkflowState.FAILED
            return self._finish(
                manifest,
                WorkflowState.FAILED,
                StatusResult(
                    ok=False,
                    query_failed=True,
                    errors=[f"child status failed: {str(exc)[:240]}"],
                    evidence_gaps=["child status raised an error"],
                ),
            )
        mapping_diagnostics = self._apply_observed_stage_states(
            manifest, result.stage_states
        )
        next_state = _next_workflow_state(manifest, result.stage_states)
        if next_state is not None and next_state is not manifest.state:
            self._apply_observed_state(manifest, next_state, result.stage_states)
        if mapping_diagnostics:
            result = result.model_copy(
                update={
                    "data": {
                        **result.data,
                        "state_mapping_diagnostics": mapping_diagnostics,
                    }
                }
            )
        return self._save_and_result(manifest, result)

    def _apply_observed_state(
        self,
        manifest: UnifiedVaspManifest,
        target: WorkflowState,
        stage_states: dict[str, str],
    ) -> None:
        """Apply an observed state when the transition is legal.

        A status query can also observe an already-terminal workflow (e.g. the
        job finished before validation, or a repeat query after failure). An
        illegal transition must never be raised from ``status``: the workflow
        already holds scheduler evidence, so only an explicit contradiction
        escalates to ``FAILED``.
        """
        if target in _ALLOWED_TRANSITIONS.get(manifest.state, set()):
            manifest.state = target
            return
        if target is WorkflowState.RUNNING and manifest.state in _TERMINAL_FAILED_STATES:
            return
        if manifest.state in {
            WorkflowState.SCHEDULER_COMPLETED,
            WorkflowState.VALIDATED,
            WorkflowState.VALIDATION_FAILED,
            WorkflowState.FAILED,
        }:
            return
        previous = manifest.state
        manifest.state = WorkflowState.FAILED
        manifest.events.append(
            WorkflowEvent(
                event_type="illegal_state_observation",
                timestamp=datetime.now(timezone.utc),
                details={
                    "observed": dict(stage_states),
                    "implied_state": target.value,
                    "previous_state": previous.value,
                },
            )
        )

    def _apply_observed_stage_states(
        self,
        manifest: UnifiedVaspManifest,
        stage_states: Mapping[str, object],
    ) -> list[str]:
        """Apply status/collect observations stage-by-stage.

        A validated stage is not downgraded by an ordinary scheduler
        ``COMPLETED`` re-observation.  Missing records for previously submitted
        stages, unknown states, and unrecognized strings are recorded as
        reconciliation diagnostics and mapped to ``RECONCILING`` rather than a
        scheduler-success state.
        """
        diagnostics: list[str] = []
        for name, raw in stage_states.items():
            item = next(
                (stage for stage in manifest.stages if stage.name == name),
                None,
            )
            if item is None:
                continue
            mapped, reason = _map_observed_stage_state(
                raw,
                current_state=item.state,
                has_request=item.request_id is not None,
            )
            if reason is not None:
                diagnostics.append(f"{name}: {reason}")
            if item.state is WorkflowState.VALIDATED:
                if mapped is WorkflowState.SCHEDULER_COMPLETED:
                    continue
                if mapped is WorkflowState.PLANNED and item.request_id is None:
                    continue
            item.state = mapped
        return diagnostics

    def _stage_recovery_plan(
        self, manifest: UnifiedVaspManifest
    ) -> dict[str, Any]:
        """Classify recovery around the stage that is actually blocking.

        Walks the dependency DAG and returns the first stage that is neither
        validated nor blocked only by an unvalidated upstream.  The plan
        separates four cases that used to collapse into one bare
        ``RECONCILING``:

        * the stage was never submitted -> a fresh attempt is possible;
        * the stage is active -> wait, never submit a second job;
        * the stage has a scheduler-confirmed terminal failure -> diagnose and
          only then allow a new attempt through the normal gates;
        * the stage's own state is unknown -> reconcile before any decision.
        """
        try:
            executor = self._executor(manifest)
        except Exception as exc:
            return {
                "target_stage": None,
                "recovery_action": RECOVERY_NONE,
                "blocked_reason": f"executor unavailable: {type(exc).__name__}: {exc}",
                "next_action": "STOP: no executor is available for this workflow",
                "submitted_before": False,
                "diagnostics_required": False,
            }
        for stage in manifest.stages:
            bound = _stage_is_bound(stage)
            query_failed = False
            record = None
            try:
                record = executor.session.registry.get(  # type: ignore[attr-defined]
                    self._stage_request_id(manifest, stage, executor)
                )
            except Exception:
                query_failed = True
                record = None
            if query_failed:
                binding = BINDING_REGISTRY_QUERY_FAILED
            elif record is None:
                binding = (
                    BINDING_BOUND_RECORD_MISSING
                    if bound
                    else BINDING_NEVER_SUBMITTED
                )
            else:
                binding = BINDING_RECORD_FOUND
            submitted_before = bool(
                record is not None
                and (record.job_id is not None or record.remote_directory)
            )
            raw_state = (
                record.state.value
                if record is not None
                and isinstance(getattr(record, "state", None), JobLifecycleState)
                else ""
            )
            if not raw_state and bound:
                # Bound identity but no readable record: the stage state is
                # the only local fact, and it is not proof of a fresh start.
                raw_state = _RECORD_MISSING_STATE
            effective_state = _effective_stage_state(stage.state, raw_state)
            if effective_state is WorkflowState.VALIDATED:
                continue
            unmet = [
                dependency
                for dependency in stage.depends_on
                if not self._dependency_is_validated(manifest, dependency)
            ]
            if unmet:
                # An unvalidated dependency: the blocking stage is upstream.
                continue
            if binding in {
                BINDING_BOUND_RECORD_MISSING,
                BINDING_REGISTRY_QUERY_FAILED,
                BINDING_RECORD_FOUND,
            } and raw_state in {
                "",
                _RECORD_MISSING_STATE,
            }:
                # The stage is bound to an execution identity (or the registry
                # query itself failed), so "never submitted" cannot be proven.
                # A new attempt is never authorized from an unknown identity.
                reason = (
                    "the registry query failed, so no job identity can be "
                    "confirmed"
                    if binding == BINDING_REGISTRY_QUERY_FAILED
                    else (
                        f"stage {stage.name!r} is already bound to request "
                        f"{stage.request_id!r} but its registry record is "
                        "missing"
                    )
                )
                return {
                    "target_stage": stage.name,
                    "recovery_action": RECOVERY_RECONCILE,
                    "record_binding": binding,
                    "blocked_reason": (
                        f"{reason}; its remote job identity is UNKNOWN"
                    ),
                    "next_action": (
                        f"RECONCILE: resolve the {stage.name!r} job identity "
                        "before any resubmit; never prepare a fresh attempt "
                        "from an unconfirmed identity"
                    ),
                    "submitted_before": True,
                    "diagnostics_required": False,
                    "stage_state": raw_state,
                }
            if raw_state in _TERMINAL_FAILED_STATES:
                return {
                    "target_stage": stage.name,
                    "recovery_action": RECOVERY_READY_FOR_NEW_ATTEMPT,
                    "record_binding": binding,
                    "blocked_reason": (
                        f"stage {stage.name!r} ended in scheduler state "
                        f"{raw_state}; its upstream dependencies are validated"
                    ),
                    "next_action": (
                        f"COLLECT_DIAGNOSTICS then READY_FOR_NEW_ATTEMPT: read "
                        f"the {stage.name!r} failure logs if available, then "
                        "submit a new attempt through the normal preflight and "
                        "approval gates"
                    ),
                    "submitted_before": submitted_before,
                    "diagnostics_required": True,
                    "stage_state": raw_state,
                }
            if raw_state in _RECOVERY_ACTIVE_STATES:
                return {
                    "target_stage": stage.name,
                    "recovery_action": RECOVERY_WAIT,
                    "record_binding": binding,
                    "blocked_reason": (
                        f"stage {stage.name!r} is {raw_state}; a second job "
                        "must never be submitted"
                    ),
                    "next_action": f"WAIT: poll {stage.name!r} status",
                    "submitted_before": submitted_before,
                    "diagnostics_required": False,
                    "stage_state": raw_state,
                }
            if raw_state in _RECOVERY_VALIDATION_STATES:
                return {
                    "target_stage": stage.name,
                    "recovery_action": RECOVERY_COLLECT_DIAGNOSTICS,
                    "record_binding": binding,
                    "blocked_reason": (
                        f"stage {stage.name!r} results failed scientific "
                        "validation"
                    ),
                    "next_action": (
                        f"COLLECT_DIAGNOSTICS: the {stage.name!r} outputs are "
                        "present but did not validate; failure diagnostics are "
                        "not evidence and a new attempt needs the normal gates"
                    ),
                    "submitted_before": submitted_before,
                    "diagnostics_required": True,
                    "stage_state": raw_state,
                }
            if raw_state in _RECOVERY_SUCCESS_STATES:
                return {
                    "target_stage": stage.name,
                    "recovery_action": RECOVERY_COLLECT_UPSTREAM,
                    "record_binding": binding,
                    "blocked_reason": (
                        f"stage {stage.name!r} finished at the scheduler but "
                        "has not been scientifically collected/validated"
                    ),
                    "next_action": (
                        f"COLLECT: validate {stage.name!r} before any "
                        "downstream stage can start"
                    ),
                    "submitted_before": submitted_before,
                    "diagnostics_required": False,
                    "stage_state": raw_state,
                }
            if submitted_before and raw_state in _UNDECIDED_STATES:
                return {
                    "target_stage": stage.name,
                    "recovery_action": RECOVERY_RECONCILE,
                    "blocked_reason": (
                        f"stage {stage.name!r} was submitted but its scheduler "
                        f"state is {raw_state or 'unknown'}"
                    ),
                    "next_action": (
                        f"RECONCILE: resolve the {stage.name!r} job identity "
                        "before any resubmit"
                    ),
                    "submitted_before": True,
                    "diagnostics_required": False,
                    "stage_state": raw_state,
                }
            return {
                "target_stage": stage.name,
                "recovery_action": RECOVERY_READY_FOR_NEW_ATTEMPT,
                "record_binding": binding,
                "blocked_reason": (
                    f"stage {stage.name!r} has never been submitted and its "
                    "dependencies are validated"
                ),
                "next_action": (
                    f"SUBMIT: {stage.name!r} is dependency-ready; go through "
                    "preflight and resource approval"
                ),
                "submitted_before": False,
                "diagnostics_required": False,
                "stage_state": raw_state,
            }
        return {
            "target_stage": None,
            "recovery_action": RECOVERY_NONE,
            "record_binding": BINDING_RECORD_FOUND,
            "blocked_reason": "every stage is validated",
            "next_action": "NONE: no recovery action is required",
            "submitted_before": False,
            "diagnostics_required": False,
            "stage_state": "",
        }

    def _dependency_is_validated(
        self, manifest: UnifiedVaspManifest, dependency: str
    ) -> bool:
        """True only when an upstream dependency is scientifically validated.

        A stage that merely finished at the scheduler (or whose record is
        COMPLETED) is explicitly *not* a validated dependency, so a downstream
        stage is never treated as ready to attempt on unvalidated upstream.
        """
        candidate = next(
            (item for item in manifest.stages if item.name == dependency), None
        )
        if candidate is None:
            return False
        if candidate.state is WorkflowState.VALIDATED:
            return True
        record = self._target_stage_record(manifest, candidate)
        if record is None:
            return False
        record_state = (
            record.state.value
            if isinstance(getattr(record, "state", None), JobLifecycleState)
            else ""
        )
        return (
            _effective_stage_state(candidate.state, record_state)
            is WorkflowState.VALIDATED
        )

    def _executor_has_registry(self, manifest: UnifiedVaspManifest) -> bool:
        """True when the executor exposes a queryable job registry.

        Executors without one (lightweight fakes, molecular backends) keep the
        legacy whole-workflow duplicate contract: the service cannot prove a
        stage was never submitted, so it must not claim it needs preflight.
        """
        try:
            executor = self._executor(manifest)
        except Exception:
            return False
        registry = getattr(getattr(executor, "session", None), "registry", None)
        return registry is not None and hasattr(registry, "get")

    def _target_stage_record(
        self, manifest: UnifiedVaspManifest, stage: UnifiedStage
    ) -> Any | None:
        """Return the registry record owned by one stage, if any."""
        try:
            executor = self._executor(manifest)
            registry = executor.session.registry  # type: ignore[attr-defined]
        except Exception:
            return None
        request_id = self._stage_request_id(manifest, stage, executor)
        if not request_id:
            return None
        try:
            return registry.get(request_id)
        except Exception:
            return None

    def _stage_request_id(
        self,
        manifest: UnifiedVaspManifest,
        stage: UnifiedStage,
        executor: Any | None = None,
    ) -> str:
        """Best-effort request identity for one stage without submitting.

        Prefers the executor's own derivation so the identity matches exactly
        what a submit would use (including attempt inputs); falls back to the
        manifest-bound execution fingerprint.
        """
        if stage.request_id:
            return stage.request_id
        derive = getattr(executor, "_request_id", None) if executor else None
        if callable(derive):
            try:
                candidate = derive(manifest, stage)
            except Exception:
                candidate = ""
            if candidate:
                return str(candidate)
        execution = stage.execution_fingerprint or manifest.execution_fingerprint
        if not execution:
            return ""
        return derive_request_id(
            f"periodic:{manifest.workflow_id}", stage.name, execution
        )

    async def resume(self, workflow_id: str) -> ServiceResult:
        manifest = self._load(workflow_id)
        try:
            executor = self._executor(manifest)
            result = await executor.reconcile(manifest)
        except Exception as exc:
            return self._finish(
                manifest,
                WorkflowState.FAILED,
                RecoveryResult(
                    ok=False,
                    action="STOP",
                    errors=[f"child resume failed: {str(exc)[:240]}"],
                    evidence_gaps=["child resume raised an error"],
                ),
            )
        action = result.action
        if action == "RECONCILE":
            manifest.state = WorkflowState.RECONCILING
        elif action == "NEEDS_RESOURCE_CONFIRMATION":
            manifest.state = WorkflowState.AWAITING_RESOURCE_CONFIRMATION
            manifest = self._save(manifest, expected_revision=manifest.revision)
            target = self._stage(manifest, None)
            resource = self.planner.recommend(manifest, target)
            manifest = self._persist_execution_identity(manifest, target, resource)
            target = self._stage(manifest, target.name)
            pending = result.pending_decision or pending_decision(
                manifest=manifest,
                kind=ApprovalKind.RESOURCE,
                summary=f"Resource confirmation required for recovery stage {target.name}",
                stage=target.name,
                resource_proposal=resource.model_dump(mode="json"),
            )
            self.approvals.record_pending(pending)
            return ServiceResult(
                ok=False,
                workflow_id=manifest.workflow_id,
                state=manifest.state,
                data=result.data,
                artifacts=result.artifacts,
                errors=list(result.errors) or ["resource confirmation required"],
                evidence_gaps=result.evidence_gaps,
                pending_decision=pending,
            )
        elif action == "NEEDS_SCIENTIFIC_CONFIRMATION":
            manifest.state = WorkflowState.AWAITING_SCIENTIFIC_CONFIRMATION
            manifest = self._save(manifest, expected_revision=manifest.revision)
            target = self._stage(manifest, None)
            resource = self.planner.recommend(manifest, target)
            manifest = self._persist_execution_identity(manifest, target, resource)
            target = self._stage(manifest, target.name)
            pending = result.pending_decision or pending_decision(
                manifest=manifest,
                kind=ApprovalKind.SCIENTIFIC,
                summary="Scientific confirmation required for recovery",
                stage=target.name,
            )
            self.approvals.record_pending(pending)
            return ServiceResult(
                ok=False,
                workflow_id=manifest.workflow_id,
                state=manifest.state,
                data=result.data,
                artifacts=result.artifacts,
                errors=list(result.errors) or ["scientific confirmation required"],
                evidence_gaps=result.evidence_gaps,
                pending_decision=pending,
            )
        elif action == "STOP" or not result.ok:
            manifest.state = WorkflowState.FAILED
        elif action == "AUTO_RESUME":
            if result.pending_decision is not None and self.approvals.valid_receipt(
                result.pending_decision, manifest
            ) is None:
                manifest.state = (
                    WorkflowState.AWAITING_RESOURCE_CONFIRMATION
                    if result.pending_decision.kind.value == "resource"
                    else WorkflowState.AWAITING_SCIENTIFIC_CONFIRMATION
                )
                self.approvals.record_pending(result.pending_decision)
            elif result.contcar_restart:
                restart_proof = self._validated_restart_artifact_proof(result)
                if (
                    restart_proof is None
                    or not result.restart_structural_validation
                    or result.scientific_fingerprint != manifest.scientific_fingerprint
                ):
                    manifest.state = WorkflowState.FAILED
                    data = dict(result.data)
                    data["recovery_rejected"] = "CONTCAR continuation lacks validated unchanged restart provenance"
                    result = result.model_copy(update={"ok": False, "data": data})
                else:
                    target = self._stage(manifest, None)
                    target.attempt_inputs = {
                        "restart_artifact_sha256": restart_proof.artifact_hash,
                        "restart_structural_validation": (
                            result.restart_structural_validation.model_dump(mode="json")
                        ),
                    }
                    resource = self.planner.recommend(manifest, target)
                    manifest = self._persist_execution_identity(
                        manifest, target, resource, target.attempt_inputs
                    )
                    manifest.state = WorkflowState.RUNNING
            elif _stage_target_recovery(manifest, result, self):
                # Stage-targeted recovery: reset the workflow so the target
                # stage can be diagnosed and re-attempted, without requiring
                # every other submitted stage to have failed too.
                plan = self._stage_recovery_plan(manifest)
                manifest.state = WorkflowState.PREFLIGHTED
                data = dict(result.data)
                data["reset"] = True
                data["reset_reason"] = (
                    "stage-targeted scheduler-confirmed terminal failure; "
                    "workflow reset to PREFLIGHTED for a fresh attempt "
                    "(vasp.submit)"
                )
                data.update(_recovery_plan_fields(plan))
                result = result.model_copy(update={"data": data})
            elif _confirmed_terminal_failure(result.stage_states):
                # Scheduler-confirmed failure: reset the workflow back to
                # PREFLIGHTED so the agent can submit again. The next
                # vasp.submit forces a new attempt (new request_id with a
                # parent pointer) instead of returning the terminal duplicate.
                manifest.state = WorkflowState.PREFLIGHTED
                data = dict(result.data)
                data["reset"] = True
                data["reset_reason"] = (
                    "scheduler-confirmed terminal failure; workflow reset "
                    "to PREFLIGHTED for a fresh attempt (vasp.submit)"
                )
                result = result.model_copy(update={"data": data})
            elif any(
                state in {"PENDING", "RUNNING", "SUBMITTED"}
                for state in result.stage_states.values()
            ):
                manifest.state = WorkflowState.RUNNING
                data = dict(result.data)
                data.update(
                    _recovery_plan_fields(self._stage_recovery_plan(manifest))
                )
                result = result.model_copy(update={"data": data})
            else:
                # A bare executor string is not scheduler evidence. Keep the
                # workflow in reconciliation until facts support an allowed
                # automatic transition or an exact approval is present.
                manifest.state = WorkflowState.RECONCILING
                data = dict(result.data)
                data.update(
                    _recovery_plan_fields(self._stage_recovery_plan(manifest))
                )
                result = result.model_copy(update={"data": data})
        else:
            manifest.state = WorkflowState.FAILED
        return self._save_and_result(manifest, result)

    async def collect(self, workflow_id: str) -> ServiceResult:
        manifest = self._load(workflow_id)
        try:
            executor = self._executor(manifest)
            result = await executor.collect(manifest)
        except Exception as exc:
            manifest.state = WorkflowState.FAILED
            return self._finish(
                manifest,
                WorkflowState.FAILED,
                CollectionResult(
                    ok=False,
                    errors=[f"child collect failed: {str(exc)[:240]}"],
                    evidence_gaps=["child collect raised an error"],
                ),
            )
        observed = getattr(result, "stage_states", {}) or {}
        mapping_diagnostics = self._apply_observed_stage_states(manifest, observed)
        if mapping_diagnostics:
            result = result.model_copy(
                update={
                    "data": {
                        **getattr(result, "data", {}),
                        "state_mapping_diagnostics": mapping_diagnostics,
                    }
                }
            )
        raw_states = [_normalized_stage_state(value) for value in observed.values()]
        # Only a hard execution failure or a deterministic stage validation
        # failure short-circuits the decision.  Missing downstream stages are
        # pending work, not a global validation failure.
        has_failed = any(value in _TERMINAL_FAILURE_RAW_STATES for value in raw_states)
        has_validation_failed = any(
            value == WorkflowState.VALIDATION_FAILED.value for value in raw_states
        )
        has_progressed = any(
            value
            in {
                WorkflowState.VALIDATED.value,
                WorkflowState.SCHEDULER_COMPLETED.value,
                "COMPLETED",
                "COLLECTED",
            }
            for value in raw_states
        )
        all_required_validated = bool(manifest.stages) and all(
            _normalized_stage_state(observed.get(stage.name))
            == WorkflowState.VALIDATED.value
            for stage in manifest.stages
        )
        if not observed:
            # Legacy executors that predate per-stage stage_states may still
            # declare workflow-level validation.  Executors that do report
            # stage_states must prove every required stage.
            all_required_validated = bool(result.validated)
        reconciliation_required = _observed_reconciliation_required(
            manifest, observed
        )
        next_state: WorkflowState
        if reconciliation_required:
            next_state = WorkflowState.RECONCILING
        elif has_failed or has_validation_failed:
            next_state = WorkflowState.VALIDATION_FAILED
        elif result.ok and result.validated and all_required_validated:
            next_state = WorkflowState.VALIDATED
        elif result.ok and has_progressed and _has_progressable_stage(manifest):
            # A dependency stage may have been collected while later declared
            # stages are still pending.  Progression uses
            # SCHEDULER_COMPLETED, never global VALIDATED; preflight then
            # authorizes the next dependency-ready stage.
            next_state = WorkflowState.SCHEDULER_COMPLETED
        elif (
            manifest.workflow_kind is VaspWorkflowKind.STUDY
            and not result.errors
            and not has_failed
        ):
            # A study child may have collected or validated one dependent stage
            # while its remaining declared tasks still need to run.  This is a
            # progression state only, never a parent scientific validation.
            next_state = WorkflowState.PREFLIGHTED
        elif result.ok and has_progressed:
            next_state = WorkflowState.SCHEDULER_COMPLETED
        elif result.ok:
            next_state = WorkflowState.RUNNING
        else:
            next_state = WorkflowState.VALIDATION_FAILED
        if next_state is not manifest.state:
            if _collect_transition_allowed(manifest.state, next_state):
                manifest.state = next_state
            else:
                previous = manifest.state
                manifest.state = WorkflowState.FAILED
                errors = list(getattr(result, "errors", []) or [])
                errors.append(
                    "illegal VASP state transition after collect: "
                    f"{previous.value} -> {next_state.value}"
                )
                result = result.model_copy(
                    update={"ok": False, "errors": errors}
                )
        data = dict(getattr(result, "data", {}) or {})
        if "pending_stages" not in data:
            data["pending_stages"] = [
                item.name
                for item in manifest.stages
                if item.state
                not in {
                    WorkflowState.VALIDATED,
                    WorkflowState.FAILED,
                    WorkflowState.VALIDATION_FAILED,
                }
            ]
        if "validated_stages" not in data:
            data["validated_stages"] = [
                item.name
                for item in manifest.stages
                if item.state is WorkflowState.VALIDATED
            ]
        if "next_stage" not in data:
            actionable = self._next_actionable_stage(manifest)
            if actionable is not None:
                data["next_stage"] = actionable.name
        if "blocked_reason" not in data:
            result_errors = list(getattr(result, "errors", []) or [])
            if result_errors:
                data["blocked_reason"] = "; ".join(
                    str(error) for error in result_errors[:3]
                )
            elif data.get("pending_stages"):
                data["blocked_reason"] = (
                    "pending: " + ", ".join(data["pending_stages"])
                )
        result = result.model_copy(update={"data": data})
        return self._save_and_result(manifest, result)

    async def report(
        self, workflow_id: str, request: ReportRequest
    ) -> ServiceResult:
        manifest = self._load(workflow_id)
        try:
            executor = self._executor(manifest)
            result = await executor.report(manifest, request)
        except Exception as exc:
            return self._finish(
                manifest,
                WorkflowState.FAILED,
                ReportResult(
                    ok=False,
                    report_kind=request.kind,
                    errors=[f"child report failed: {str(exc)[:240]}"],
                    evidence_gaps=["child report raised an error"],
                ),
            )
        return self._save_and_result(manifest, result)

    # -- internals ----------------------------------------------------------

    def _load(self, workflow_id: str) -> UnifiedVaspManifest:
        return self.repository.load(workflow_id)

    def load_manifest(self, workflow_id: str) -> UnifiedVaspManifest:
        """Read a child manifest for orchestrators that already own IDs."""
        return self.repository.load(workflow_id)

    def _executor(self, manifest: UnifiedVaspManifest):
        try:
            return self.router.executor_for(manifest.workflow_kind)
        except RuntimeError as exc:
            raise RuntimeError(str(exc)) from exc

    def _next_actionable_stage(
        self, manifest: UnifiedVaspManifest
    ) -> UnifiedStage | None:
        """Return the next dependency-ready, not-yet-validated stage."""
        for item in manifest.stages:
            if item.state is WorkflowState.PREFLIGHTED and _dependency_satisfied(
                manifest, item
            ):
                return item
        for item in manifest.stages:
            if item.state is WorkflowState.PLANNED and _dependency_satisfied(
                manifest, item
            ):
                return item
        return None

    def _stage(
        self, manifest: UnifiedVaspManifest, stage: str | None
    ) -> UnifiedStage:
        if stage is not None:
            for item in manifest.stages:
                if item.name == stage:
                    return item
            raise ValueError(f"unknown stage {stage!r}")
        if not manifest.stages:
            raise ValueError("manifest has no stages")
        # Prefer an already-preflighted, dependency-ready stage (the cursor
        # created by the latest preflight call).  Only then look for a
        # dependency-ready PLANNED stage; never fall back to stages[0] just
        # because it is first when a later stage is the legal next action.
        for item in manifest.stages:
            if item.state is WorkflowState.PREFLIGHTED:
                return item
        for item in manifest.stages:
            if item.state is WorkflowState.PLANNED and _dependency_satisfied(
                manifest, item
            ):
                return item
        for item in manifest.stages:
            if item.state is not WorkflowState.VALIDATED:
                return item
        return manifest.stages[0]

    def _transition(
        self, manifest: UnifiedVaspManifest, target: WorkflowState
    ) -> None:
        allowed = _ALLOWED_TRANSITIONS.get(manifest.state, set())
        if target not in allowed:
            raise ValueError(
                f"illegal VASP state transition: "
                f"{manifest.state.value} -> {target.value}"
            )
        manifest.state = target

    def _save(
        self, manifest: UnifiedVaspManifest, *, expected_revision: int
    ) -> UnifiedVaspManifest:
        return self.repository.save(manifest, expected_revision)

    def _persist_execution_identity(
        self,
        manifest: UnifiedVaspManifest,
        stage: UnifiedStage,
        resource: Any,
        attempt_inputs: dict[str, Any] | None = None,
    ) -> UnifiedVaspManifest:
        """Persist the exact execution identity before resource authorization.

        The scientific fingerprint remains intentionally unchanged by resource
        selection. Revision-checked persistence makes the identity authoritative
        before a receipt can be created or an executor can be reached.
        """
        identity = execution_fingerprint(
            manifest.scientific_fingerprint,
            resource,
            stage,
            attempt_inputs,
        )
        if stage.execution_fingerprint == identity:
            return manifest
        # This is the semantic authorization version, intentionally separate
        # from ordinary manifest revision.  Only execution-relevant changes
        # to this exact stage invalidate its receipts.
        stage.decision_epoch += 1
        stage.execution_fingerprint = identity
        manifest.execution_fingerprint = identity
        return self._save(manifest, expected_revision=manifest.revision)

    def _validated_restart_artifact_proof(
        self, result: RecoveryResult
    ) -> ValidatedRestartProof | None:
        """Resolve containment then obtain the opaque verified restart proof."""
        if not result.restart_artifact_path:
            return None
        try:
            artifact = self.repository.resolve_workspace_artifact(
                result.restart_artifact_path
            )
        except (OSError, ValueError, ToolExecutionError):
            return None
        return _verified_restart_proof(
            artifact,
            result.restart_artifact_sha256,
            result.restart_structural_validation,
        )

    def _save_and_result(
        self,
        manifest: UnifiedVaspManifest,
        result: Any = None,
    ) -> ServiceResult:
        saved = self._save(
            manifest,
            expected_revision=manifest.revision,
        )
        if result is None:
            result_data: dict[str, Any] = {}
            errors: list[str] = []
        else:
            result_data = result.data
            errors = list(result.errors)
        return ServiceResult(
            ok=bool(getattr(result, "ok", True))
            and bool(getattr(result, "passed", True))
            and not errors,
            workflow_id=saved.workflow_id,
            state=saved.state,
            data=result_data,
            artifacts=list(getattr(result, "artifacts", [])),
            errors=errors,
            evidence_gaps=list(getattr(result, "evidence_gaps", [])),
            evidence=list(getattr(result, "evidence", [])),
            pending_decision=getattr(result, "pending_decision", None),
        )

    def _finish(
        self,
        manifest: UnifiedVaspManifest,
        state: WorkflowState,
        result: Any,
    ) -> ServiceResult:
        manifest.state = state
        return self._save_and_result(manifest, result)
