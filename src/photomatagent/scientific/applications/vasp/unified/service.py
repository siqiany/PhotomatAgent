"""Workflow state machine and application operations for unified VASP."""

from __future__ import annotations

import re
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
}


def _confirmed_terminal_failure(stage_states: dict[str, str]) -> bool:
    """True when every submitted stage ended in a scheduler-confirmed failure.

    A stage that reconcile could not find (``NOT_FOUND``) never created a
    usable job and does not block a fresh attempt. At least one stage must
    actually have been submitted, otherwise there is nothing to reset.
    """
    submitted = [
        state for state in stage_states.values() if state != "NOT_FOUND"
    ]
    if not submitted:
        return False
    return all(state in _TERMINAL_FAILED_STATES for state in submitted)


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
                get_profile,
            )

            spec = cast(PeriodicScientificSpec, manifest.scientific_spec)
            profile = get_profile(spec.profile)
            return [UnifiedStage(name=name) for name in profile.stages]
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
            if all(
                next((candidate for candidate in manifest.stages if candidate.name == dependency), None)
                is not None
                and next(candidate for candidate in manifest.stages if candidate.name == dependency).state
                in {WorkflowState.SCHEDULER_COMPLETED, WorkflowState.VALIDATED}
                for dependency in item.depends_on
            ):
                item.state = WorkflowState.PREFLIGHTED
                break
        return self._save_and_result(manifest, result)

    async def submit(
        self, workflow_id: str, stage: str | None = None
    ) -> ServiceResult:
        manifest = self._load(workflow_id)
        target = self._stage(manifest, stage)
        progression = (
            manifest.state is WorkflowState.VALIDATED
            and target.state in {WorkflowState.PLANNED, WorkflowState.PREFLIGHTED}
            and all(
                next((candidate for candidate in manifest.stages if candidate.name == dependency), None)
                is not None
                and next(candidate for candidate in manifest.stages if candidate.name == dependency).state
                in {WorkflowState.SCHEDULER_COMPLETED, WorkflowState.VALIDATED}
                for dependency in target.depends_on
            )
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
                return ServiceResult(
                    ok=True,
                    workflow_id=workflow_id,
                    state=manifest.state,
                    data={"duplicate": True},
                    errors=[],
                )
            raise ValueError(
                "submit is only allowed after a passing preflight; "
                f"current state is {manifest.state.value}"
            )
        # ``target`` was resolved before the state gate so progression can
        # distinguish a new dependency-satisfied stage from an active retry.
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
        if not result.query_failed:
            if _confirmed_terminal_failure(result.stage_states):
                manifest.state = WorkflowState.FAILED
            else:
                manifest.state = WorkflowState.RUNNING
        return self._save_and_result(manifest, result)

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
            else:
                # A bare executor string is not scheduler evidence. Keep the
                # workflow in reconciliation until facts support an allowed
                # automatic transition or an exact approval is present.
                manifest.state = WorkflowState.RECONCILING
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
        for name, state in getattr(result, "stage_states", {}).items():
            for item in manifest.stages:
                if item.name == name:
                    try:
                        item.state = WorkflowState(state)
                    except ValueError:
                        item.state = WorkflowState.SCHEDULER_COMPLETED
        if result.ok and result.validated:
            manifest.state = WorkflowState.VALIDATED
        elif (
            manifest.workflow_kind is not VaspWorkflowKind.STUDY
            and result.ok
            and not result.errors
            and any(
            state == WorkflowState.VALIDATED.value
            for state in getattr(result, "stage_states", {}).values()
            )
        ):
            # A dependency stage may be collected while later WorkflowSpec
            # stages are still pending.  Keep the child at the progression
            # sentinel so preflight can authorize the next stage; the result
            # remains non-validated and carries explicit evidence gaps.
            manifest.state = WorkflowState.VALIDATED
        elif (
            manifest.workflow_kind is VaspWorkflowKind.STUDY
            and not result.errors
            and not any(
                state in {WorkflowState.FAILED.value, WorkflowState.VALIDATION_FAILED.value}
                for state in getattr(result, "stage_states", {}).values()
            )
        ):
            # A study child may have validated one dependency stage while its
            # remaining declared stages still need to run.  This is a
            # progression state only, never a parent scientific validation.
            manifest.state = WorkflowState.PREFLIGHTED
        elif result.ok:
            manifest.state = WorkflowState.SCHEDULER_COMPLETED
        else:
            manifest.state = WorkflowState.VALIDATION_FAILED
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
