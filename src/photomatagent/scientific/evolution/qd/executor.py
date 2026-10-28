"""Run a frozen QD trial through the authoritative AgentRuntime.

This adapter constructs a new runtime and a deliberately narrow registry for
each trial. It collects runtime observations, seals expected output files, and
then delegates qualification to the host-owned ContractValidator.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import AsyncGenerator, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from photomatagent.logging.event_logger import EventLogger
from photomatagent.models.base import ModelProvider
from photomatagent.redaction import redact_secrets
from photomatagent.runtime.budget import BudgetState
from photomatagent.runtime.context import ContextBuilder
from photomatagent.runtime.events import (
    LoopCompleted, QDGenomeMismatch, QDTrialCollected, QDTrialFailed, QDTrialStarted,
    RuntimeEvent, TextDelta, ToolApprovalRequired, ToolCallArgumentsDelta,
    ToolCallCompleted, ToolCompleted, ToolFailed, ToolPermissionDenied,
    ToolRequested, parse_event,
)
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import ApprovalHandler, PermissionPolicy, default_permission_policy
from photomatagent.scientific.evolution.qd.budget import BudgetLedger
from photomatagent.scientific.evolution.qd.models import (
    ArtifactDigest, EvaluationContract, TrialOutcome, TrialRecord,
    ValidationRecord, WorkflowGenome, canonical_genome_sha256,
)
from photomatagent.scientific.evolution.qd.validation import CHECKER_VERSION, ContractValidator, canonical_contract_sha256
from photomatagent.scientific.state import ScientificState
from photomatagent.skills.loader import SkillLoader
from photomatagent.tools.base import Tool
from photomatagent.tools.factory import create_default_registry
from photomatagent.workspace import Workspace


ServiceToolFactory = Callable[[TrialRecord, WorkflowGenome, Workspace], Mapping[str, Tool]]


def _json_tree_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (actual.keys() == expected.keys()
                and all(_json_tree_equal(actual[key], value) for key, value in expected.items()))
    if isinstance(expected, list):
        return (len(actual) == len(expected)
                and all(_json_tree_equal(left, right) for left, right in zip(actual, expected, strict=True)))
    return bool(actual == expected)


@dataclass(frozen=True, slots=True)
class QDExecutionContext:
    trial: TrialRecord
    attempt_workspace: Workspace
    genome: WorkflowGenome
    budget_handle: BudgetLedger | None
    service_refs: Mapping[str, Tool]
    runtime_environment_id: str | None = None
    protected_outputs: frozenset[str] = frozenset()
    workflow_gate: QDWorkflowGate = field(init=False)

    def __post_init__(self) -> None:
        if self.trial.genome_id != self.genome.genome_id:
            raise ValueError("GENOME_MISMATCH")
        if self.attempt_workspace.root.name != self.trial.trial_id:
            raise ValueError("QD_ATTEMPT_MISMATCH")
        object.__setattr__(self, "service_refs", MappingProxyType(dict(self.service_refs)))
        object.__setattr__(self, "workflow_gate", QDWorkflowGate(
            self.genome, environment_id=self.runtime_environment_id,
        ))


def _binding_steps(genome: WorkflowGenome, name: str,
                   arguments: Mapping[str, Any], *, environment_id: str | None = None) -> tuple[str, ...]:
    matches: list[str] = []
    script_steps = [step for step in genome.steps if step.kind == "script"]
    for step in genome.steps:
        parameters = step.model_dump(mode="json")["parameters"]
        if step.kind == "registered_tool":
            if step.binding_id == name and _json_tree_equal(dict(arguments), parameters):
                matches.append(step.step_id)
            continue
        if name != "script_run" or step.binding_id != name:
            continue
        script = parameters.get("script_path")
        if script is None:
            if len(genome.script_refs) != 1 or len(script_steps) != 1:
                continue
            script = genome.script_refs[0].path
        if script not in {ref.path for ref in genome.script_refs}:
            continue
        expected = {
            **parameters, "script_path": script,
            "environment_id": environment_id or genome.environment_ref,
        }
        if _json_tree_equal(dict(arguments), expected):
            matches.append(step.step_id)
    return tuple(matches)


def matches_frozen_binding(
    genome: WorkflowGenome, name: str, arguments: Mapping[str, Any], *,
    environment_id: str | None = None,
) -> bool:
    """Compare a model request to exactly one host-frozen step binding."""
    return len(_binding_steps(genome, name, arguments, environment_id=environment_id)) == 1


class QDWorkflowGate:
    """One-use DAG gate shared by the registered service tools of a trial."""

    def __init__(self, genome: WorkflowGenome, *, environment_id: str | None = None) -> None:
        self.genome = genome
        self.environment_id = environment_id
        self._steps = {step.step_id: step for step in genome.steps}
        self.attempted: set[str] = set()
        self.completed: set[str] = set()
        self.mismatch = False

    def begin(self, name: str, arguments: Mapping[str, Any]) -> str | None:
        matches = _binding_steps(
            self.genome, name, arguments, environment_id=self.environment_id,
        )
        if len(matches) != 1:
            self.mismatch = True
            return None
        step_id = matches[0]
        step = self._steps[step_id]
        if step_id in self.attempted or not set(step.depends_on).issubset(self.completed):
            self.mismatch = True
            return None
        self.attempted.add(step_id)
        return step_id

    def finish(self, step_id: str, success: bool) -> None:
        if success:
            self.completed.add(step_id)

    @property
    def all_complete(self) -> bool:
        return not self.mismatch and self.completed == set(self._steps)


def _frozen_files_match(workspace: Workspace, attempt: Workspace,
                        trial: TrialRecord, genome: WorkflowGenome) -> bool:
    frozen: list[tuple[Path, ArtifactDigest]] = [(attempt.root, ref) for ref in genome.script_refs]
    for ref in trial.input_digests:
        if not ref.path.startswith(trial.attempt_dir + "/"):
            return False
        frozen.append((workspace.root, ref))
    for root, ref in frozen:
        path = root
        for part in Path(ref.path).parts:
            path /= part
            if path.is_symlink():
                return False
        if not path.is_file() or path.stat().st_size != ref.size_bytes:
            return False
        if hashlib.sha256(path.read_bytes()).hexdigest() != ref.sha256:
            return False
    return True


def _contract_output_paths(workspace: Workspace, trial: TrialRecord,
                           contract: EvaluationContract) -> tuple[str, ...]:
    paths: set[str] = set()
    for check in contract.checks:
        relative = check.parameters.get("artifact_path")
        if relative is None:
            continue
        if (not isinstance(relative, str) or relative.startswith("/")
                or "\\" in relative or any(part in ("", ".", "..") for part in relative.split("/"))):
            raise ValueError("CONTRACT_ARTIFACT_PATH_INVALID")
        if relative.startswith(trial.attempt_dir + "/"):
            pass
        elif relative.startswith("attempts/"):
            raise ValueError("CONTRACT_ARTIFACT_PATH_INVALID")
        else:
            relative = f"{trial.attempt_dir}/{relative}"
        current = workspace.root
        for part in Path(relative).parts:
            current /= part
            if current.is_symlink():
                raise ValueError("CONTRACT_ARTIFACT_PATH_INVALID")
        if not workspace.contains(current.resolve(strict=False)):
            raise ValueError("CONTRACT_ARTIFACT_PATH_INVALID")
        paths.add(relative)
    return tuple(sorted(paths))


def _output_refs(workspace: Workspace, paths: tuple[str, ...], max_bytes: int) -> tuple[ArtifactDigest, ...]:
    refs: list[ArtifactDigest] = []
    for relative in paths:
        current = workspace.root
        unsafe = False
        for part in Path(relative).parts:
            current /= part
            if current.is_symlink():
                unsafe = True
                break
        if unsafe or not current.is_file() or current.stat().st_size > max_bytes:
            continue
        content = current.read_bytes()
        try:
            def reject_constant(value: str) -> None:
                raise ValueError(f"non-JSON constant: {value}")

            json.loads(content.decode("utf-8"), parse_constant=reject_constant)
            media_type = "application/json"
        except (UnicodeDecodeError, ValueError, RecursionError):
            media_type = "application/octet-stream"
        refs.append(ArtifactDigest(path=relative, sha256=hashlib.sha256(content).hexdigest(),
                                   size_bytes=len(content), media_type=media_type))
    return tuple(refs)


def _bounded_for_log(event: RuntimeEvent) -> RuntimeEvent:
    """Keep persisted model text and tool arguments bounded without altering execution."""
    if len(event.model_dump_json()) <= 16_000:
        return event
    updates: dict[str, Any] = {}
    if isinstance(event, ToolCallArgumentsDelta):
        updates["delta"] = "[bounded]"
    elif isinstance(event, ToolCallCompleted):
        updates["arguments"] = {}
    elif isinstance(event, (ToolRequested, ToolApprovalRequired)):
        updates["arguments"] = {}
    elif isinstance(event, TextDelta):
        updates["text"] = event.text[:1000]
    elif isinstance(event, ToolCompleted):
        updates.update(output=event.output[:1000], truncated=True)
    elif isinstance(event, ToolFailed):
        updates.update(error=event.error[:1000], truncated=True)
    candidate = event.model_copy(update=updates)
    if len(candidate.model_dump_json()) <= 16_000:
        return candidate
    for key, value in event.__dict__.items():
        if key in {"kind", "schema_version", "timestamp", "session_id", "run_id"}:
            continue
        if isinstance(value, str):
            updates[key] = value[:1000]
        elif isinstance(value, dict):
            updates[key] = {}
        elif isinstance(value, list):
            updates[key] = []
    return event.model_copy(update=updates)


class TrialExecutor:
    """One trial per fresh runtime, with no direct model-requested Tool.execute calls."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        model: ModelProvider,
        validator: ContractValidator,
        service_tool_factory: ServiceToolFactory,
        permission_policy: PermissionPolicy | None = None,
        approval_handler: ApprovalHandler | None = None,
        budget_handle: BudgetLedger | None = None,
        runtime_environment_id: str | None = None,
        max_iterations: int = 8,
        max_trace_events: int = 256,
        max_output_bytes: int = 1_000_000,
    ) -> None:
        if type(validator) is not ContractValidator:
            raise TypeError("QD_VALIDATOR_UNTRUSTED")
        if max_iterations <= 0 or max_trace_events < 2 or max_output_bytes <= 0:
            raise ValueError("QD_EXECUTOR_LIMIT_INVALID")
        self.workspace = workspace
        self.model = model
        self.validator = validator
        self.service_tool_factory = service_tool_factory
        self.permission_policy = permission_policy or default_permission_policy()
        self.approval_handler = approval_handler
        self.budget_handle = budget_handle
        self.runtime_environment_id = runtime_environment_id
        self.max_iterations = max_iterations
        self.max_trace_events = max_trace_events
        self.max_output_bytes = max_output_bytes
        self.last_events: tuple[RuntimeEvent, ...] = ()

    async def execute(self, *, trial: TrialRecord, genome: WorkflowGenome,
                      contract: EvaluationContract, token_budget: int | None = None,
                      model_call_budget: int | None = None) -> TrialOutcome:
        if trial.status != "RUNNING":
            raise ValueError("TRIAL_NOT_RUNNING")
        if not trial.attempt_dir.startswith("tmp/qd/") or Path(trial.attempt_dir).name != trial.trial_id:
            raise ValueError("QD_ATTEMPT_MISMATCH")
        attempt_path = self.workspace.resolve(trial.attempt_dir, must_exist=False)
        current = self.workspace.root
        for part in Path(trial.attempt_dir).parts:
            current /= part
            if current.is_symlink():
                raise ValueError("QD_ATTEMPT_SYMLINK")
        attempt_path.mkdir(parents=True, exist_ok=True)
        attempt = Workspace(attempt_path)
        sessions_root = self.workspace.root / ".photomatagent" / "sessions"
        if any(path.is_symlink() for path in (
            self.workspace.root / ".photomatagent", sessions_root,
            sessions_root / trial.session_id,
        )):
            raise ValueError("SESSION_PATH_UNSAFE")
        if (sessions_root / trial.session_id).exists():
            raise ValueError("SESSION_ALREADY_USED")
        trace_ref = f".photomatagent/sessions/{trial.session_id}/events.jsonl"
        if trial.runtime_trace_ref and trial.runtime_trace_ref != trace_ref:
            raise ValueError("QD_TRACE_MISMATCH")
        logger = EventLogger(sessions_root, session_id=trial.session_id)
        events: list[RuntimeEvent] = []
        current_trial = trial.model_copy(update={"runtime_trace_ref": trace_ref})

        async def record(event: RuntimeEvent, *, terminal: bool = False) -> bool:
            limit = self.max_trace_events if terminal else self.max_trace_events - 1
            if len(events) >= limit:
                return False
            event.session_id = trial.session_id
            bounded = _bounded_for_log(event)
            payload = redact_secrets(bounded.model_dump(mode="json"))
            if not isinstance(payload, dict):
                raise ValueError("QD_TRACE_REDACTION_INVALID")
            safe = parse_event(payload)
            if len(safe.model_dump_json()) > 16_000:
                payload = redact_secrets(_bounded_for_log(safe).model_dump(mode="json"))
                if not isinstance(payload, dict):
                    raise ValueError("QD_TRACE_REDACTION_INVALID")
                safe = parse_event(payload)
            events.append(safe)
            await logger.log(safe)
            return True

        async def failed(reason: str, *, mismatch: bool = False) -> TrialOutcome:
            recorded_cost = dict(current_trial.cost)
            recorded_cost.setdefault("model_tokens", 0.0)
            recorded_cost.setdefault("runtime_seconds", 0.0)
            record_trial = current_trial.model_copy(update={"status": "FAILED", "cost": recorded_cost})
            event: RuntimeEvent = (QDGenomeMismatch(event_id=trial.event_id, trial_id=trial.trial_id,
                                    genome_id=trial.genome_id) if mismatch else
                                    QDTrialFailed(event_id=trial.event_id, trial_id=trial.trial_id,
                                                  genome_id=trial.genome_id, reason_code=reason))
            await record(event, terminal=True)
            self.last_events = tuple(events)
            return TrialOutcome(trial=record_trial, validation=None, reason_code=reason)

        async def interrupted(reason: str) -> TrialOutcome:
            recorded_cost = dict(current_trial.cost)
            recorded_cost.setdefault("model_tokens", 0.0)
            recorded_cost.setdefault("runtime_seconds", 0.0)
            record_trial = current_trial.model_copy(update={"status": "INTERRUPTED", "cost": recorded_cost})
            await record(QDTrialFailed(event_id=trial.event_id, trial_id=trial.trial_id,
                                       genome_id=trial.genome_id, reason_code=reason), terminal=True)
            self.last_events = tuple(events)
            return TrialOutcome(trial=record_trial, validation=None, reason_code=reason)

        if (trial.genome_id != genome.genome_id or genome.content_sha256 != canonical_genome_sha256(genome)
                or genome.task_family != contract.task_family or trial.contract_sha256 != canonical_contract_sha256(contract)
                or trial.seed_set != contract.seed_set
                or not _frozen_files_match(self.workspace, attempt, trial, genome)):
            return await failed("GENOME_MISMATCH", mismatch=True)

        try:
            contract_paths = _contract_output_paths(self.workspace, trial, contract)
        except ValueError:
            return await failed("CONTRACT_ARTIFACT_PATH_INVALID")
        if any((self.workspace.root / path).exists() for path in contract_paths):
            return await failed("OUTPUT_ALREADY_EXISTS")
        service_refs = self.service_tool_factory(trial, genome, attempt)
        output_paths = frozenset(path[len(trial.attempt_dir) + 1:] for path in contract_paths)
        input_paths = frozenset(ref.path[len(trial.attempt_dir) + 1:] for ref in trial.input_digests)
        context = QDExecutionContext(trial=trial, attempt_workspace=attempt, genome=genome,
                                     budget_handle=self.budget_handle, service_refs=service_refs,
                                     runtime_environment_id=self.runtime_environment_id,
                                     protected_outputs=output_paths | input_paths)
        registry = create_default_registry(ScientificState(), attempt, qd_context=context)
        runtime = AgentRuntime(
            model=self.model, tools=registry, workspace=attempt, scientific_state=ScientificState(),
            context_builder=ContextBuilder(SkillLoader(attempt.root / ".photomatagent" / "fresh-context" / "skills")),
            permission_policy=self.permission_policy, approval_handler=self.approval_handler,
            budget=BudgetState(
                max_iterations=min(self.max_iterations, model_call_budget)
                if model_call_budget is not None else self.max_iterations,
                max_total_tokens=token_budget,
            ),
            session_id=trial.session_id,
        )
        await record(QDTrialStarted(event_id=trial.event_id, trial_id=trial.trial_id,
                                    genome_id=trial.genome_id))
        current_trial = current_trial.model_copy(update={"status": "RUNNING"})
        mismatch = False
        denied = False
        completed = 0
        tool_failed = False
        overflow = False
        completed_reason: str | None = None
        execution_started_at = time.monotonic()
        service_names = frozenset(service_refs)
        workflow_steps = []
        for step in genome.steps:
            parameters = step.model_dump(mode="json")["parameters"]
            if step.kind == "script":
                script = parameters.get("script_path")
                if script is None and len(genome.script_refs) == 1:
                    script = genome.script_refs[0].path
                if not isinstance(script, str) or script not in {ref.path for ref in genome.script_refs}:
                    return await failed("GENOME_MISMATCH", mismatch=True)
                parameters = {**parameters, "script_path": script,
                              "environment_id": self.runtime_environment_id}
            workflow_steps.append({
                "step_id": step.step_id, "kind": step.kind,
                "binding_id": step.binding_id, "depends_on": step.depends_on,
                "input_roles": step.input_roles, "output_roles": step.output_roles,
                "parameters": parameters,
            })
        instruction = (
            f"Execute frozen workflow {genome.genome_id} for trial {trial.trial_id}. "
            "Follow this host-provided, immutable execution manifest exactly. "
            "Call each step only after its dependencies complete; use the listed registered tool, "
            "script path, input references, arguments, and trusted environment ID. "
            "Do not invent or alter parameters. Manifest: "
            + json.dumps({
                "genome_id": genome.genome_id, "content_sha256": genome.content_sha256,
                "entrypoint": genome.entrypoint, "steps": workflow_steps,
                "scripts": [ref.model_dump(mode="json") for ref in genome.script_refs],
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + " Report execution observations; scientific qualification is checked separately."
        )

        def snapshot_runtime() -> None:
            nonlocal current_trial
            try:
                refs = _output_refs(self.workspace, contract_paths, self.max_output_bytes)
            except (OSError, ValueError):
                refs = ()
            current_trial = current_trial.model_copy(update={
                "output_refs": refs,
                "cost": {"model_tokens": float(runtime.budget.total_tokens),
                         "model_calls": float(runtime.budget.model_calls),
                         "runtime_seconds": max(0.0, time.monotonic() - execution_started_at)},
            })

        try:
            stream = cast(AsyncGenerator[RuntimeEvent, None], runtime.run(instruction))
            try:
                async for event in stream:
                    if not await record(event):
                        overflow = True
                        break
                    effective_name = getattr(event, "underlying_tool", None) or getattr(event, "tool_name", "")
                    if isinstance(event, ToolRequested) and effective_name in service_names:
                        if not matches_frozen_binding(
                            genome, effective_name, event.arguments,
                            environment_id=context.runtime_environment_id,
                        ):
                            mismatch = True
                    elif isinstance(event, ToolCompleted) and effective_name in service_names:
                        completed += 1
                    elif isinstance(event, ToolPermissionDenied) and effective_name in service_names:
                        denied = True
                    elif isinstance(event, ToolFailed) and effective_name in service_names:
                        tool_failed = True
                        if "GENOME_MISMATCH" in event.error:
                            mismatch = True
                    elif isinstance(event, LoopCompleted):
                        completed_reason = event.reason
            finally:
                await stream.aclose()
        except Exception:
            snapshot_runtime()
            return await interrupted("RUNTIME_FAILED")
        snapshot_runtime()
        if context.workflow_gate.mismatch or not _frozen_files_match(self.workspace, attempt, trial, genome):
            mismatch = True
        if mismatch:
            return await failed("GENOME_MISMATCH", mismatch=True)
        if denied:
            return await failed("PERMISSION_DENIED")
        if tool_failed:
            return await failed("TOOL_EXECUTION_FAILED")
        if overflow:
            return await failed("TRACE_LIMIT_EXCEEDED")
        if completed_reason != "final_response":
            return await failed("RUNTIME_INCOMPLETE")
        if completed == 0:
            return await failed("WORKFLOW_NOT_EXECUTED")
        if not context.workflow_gate.all_complete:
            return await failed("GENOME_MISMATCH", mismatch=True)
        collected = current_trial.model_copy(update={"status": "COLLECTED"})
        try:
            validation = self.validator.validate(trial=collected, contract=contract)
        except Exception:
            return await failed("CHECK_UNAVAILABLE")
        try:
            validation = ValidationRecord.model_validate(validation.model_dump(mode="json"))
            bundle_payload = [ref.model_dump(mode="json") for ref in
                              sorted(collected.output_refs, key=lambda item: item.path)]
            bundle_sha = hashlib.sha256(json.dumps(bundle_payload, sort_keys=True,
                                    separators=(",", ":"), ensure_ascii=False,
                                    allow_nan=False).encode("utf-8")).hexdigest()
        except (AttributeError, TypeError, ValueError):
            return await failed("CHECK_RESULT_MISMATCH")
        if (validation.trial_id != collected.trial_id
                or validation.genome_sha256 != genome.content_sha256
                or validation.bundle_sha256 != bundle_sha
                or validation.contract_sha256 != collected.contract_sha256
                or validation.suite_sha256 != contract.evaluation_suite_sha256
                or validation.checker_version != CHECKER_VERSION
                or dict(validation.cost) != dict(collected.cost)):
            return await failed("CHECK_RESULT_MISMATCH")
        await record(QDTrialCollected(event_id=trial.event_id, trial_id=trial.trial_id,
                                      genome_id=trial.genome_id,
                                      output_count=len(collected.output_refs)), terminal=True)
        self.last_events = tuple(events)
        return TrialOutcome(trial=collected.model_copy(update={"status": "VALIDATED"}),
                            validation=validation)

    async def execute_acquired(
        self, *, trial: TrialRecord, genome: WorkflowGenome,
        contract: EvaluationContract, token_budget: int | None = None,
    ) -> TrialOutcome:
        """Run a newly acquired, already-probed Genome only under its frozen lock.

        Acquisition probes do not qualify a scientific result. This adapter
        verifies the frozen Genome/contract/environment bindings, then sends
        the independent scored Trial through the same authoritative runtime.
        """
        if (genome.genome_id != trial.genome_id
                or genome.content_sha256 != canonical_genome_sha256(genome)
                or contract.task_family != genome.task_family
                or canonical_contract_sha256(contract) != trial.contract_sha256):
            return TrialOutcome(
                trial=trial.model_copy(update={"status": "FAILED"}),
                validation=None, reason_code="GENOME_MISMATCH",
            )
        if genome.environment_ref != trial.environment_sha256:
            return TrialOutcome(
                trial=trial.model_copy(update={"status": "FAILED"}),
                validation=None, reason_code="GENOME_ENVIRONMENT_MISMATCH",
            )
        return await self.execute(
            trial=trial, genome=genome, contract=contract, token_budget=token_budget,
        )
