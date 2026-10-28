"""Trusted CLI assembly for persisted QD tasks."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from datetime import datetime, timezone
from pydantic import BaseModel, ConfigDict, Field
from photomatagent.capability_acquisition.sandbox import SandboxBackend
from photomatagent.scientific.evolution.qd.config import EvolutionBudget
from typing import Any

from photomatagent.capability_acquisition.container_backend import PodmanSandboxBackend
from photomatagent.capability_acquisition.models import EnvironmentRecord
from photomatagent.capability_acquisition.sandbox import SandboxService
from photomatagent.capability_acquisition.tools import ScriptRunTool
from photomatagent.config import DotEnvConfig, read_preferred_config
from photomatagent.models.base import ModelProvider
from photomatagent.models.factory import create_provider
from photomatagent.runtime.permissions import (
    AllowAllPolicy, AskPolicy, AutoApproveHandler, DenyAllPolicy, DenyHandler,
    PermissionPolicy,
)
from photomatagent.scientific.evolution.qd.archive import QDArchive
from photomatagent.scientific.evolution.qd.budget import BudgetLedger
from photomatagent.scientific.evolution.qd.config import QDConfig
from photomatagent.scientific.evolution.qd.coordinator import EvolutionCoordinator
from photomatagent.scientific.evolution.qd.executor import TrialExecutor
from photomatagent.scientific.evolution.qd.models import (
    ArtifactDigest, EvaluationContract, EventOutcome, RevisionIssue, TrialRecord,
    WorkflowGenome,
)
from photomatagent.scientific.evolution.qd.operators import ModelProposalEngine
from photomatagent.scientific.evolution.qd.store import QDStore
from photomatagent.scientific.evolution.qd.validation import ContractValidator
from photomatagent.tools.base import Tool, ToolResult
from photomatagent.workspace import Workspace


def resolve_environment_id(
    genome: WorkflowGenome, *, configured_id: str | None, record: EnvironmentRecord | None,
) -> str:
    """Map a frozen environment digest to a registered executable ID fail-closed."""
    if (
        not configured_id
        or record is None
        or record.status != "READY"
        or record.environment_id != configured_id
        or record.environment_sha256 != genome.environment_ref
    ):
        raise ValueError("QD_ENVIRONMENT_MISMATCH")
    return record.environment_id


class QDTaskRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    task_id: str
    status: str = "READY"
    contract: dict[str, Any]
    baseline_genome: dict[str, Any]
    issues: tuple[dict[str, Any], ...] = ()
    budget: EvolutionBudget = Field(default_factory=EvolutionBudget)
    environment_id: str | None = None
    model_provider: str | None = None
    model_id: str | None = None
    mutable_parameters: tuple[dict[str, Any], ...] = ()
    event_outcome: dict[str, Any] | None = None
    approval_mode: str = "ask"
    revision: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class QDRuntimeAdapter:
    """Assemble existing QD services behind one trusted CLI boundary."""

    def __init__(
        self, *, workspace: Workspace, task: Any, config: QDConfig,
        model: ModelProvider | None = None, backend: SandboxBackend | None = None,
        environment: EnvironmentRecord | None = None, seed: int = 0,
        cancel_event: asyncio.Event | None = None, proposal_engine: Any | None = None,
    ) -> None:
        self.workspace = workspace
        self.task = task
        self.config = config
        self.model = model or _configured_model(workspace.root)
        self.backend = backend
        self.cancel_event = cancel_event
        self.environment = environment
        self.seed = seed
        self.proposal_engine = proposal_engine
        if not config.enabled:
            raise ValueError("QD_DISABLED")

    async def run_event(self) -> EventOutcome:
        contract = EvaluationContract.model_validate(self.task.contract)
        baseline = WorkflowGenome.model_validate(self.task.baseline_genome)
        issues = tuple(RevisionIssue.model_validate(item) for item in self.task.issues)
        storage = _workspace_path(self.workspace, self.config.environment_storage)
        backend = self.backend or PodmanSandboxBackend(storage, executable=self.config.sandbox)
        if self.task.environment_id is None:
            raise ValueError("QD_ENVIRONMENT_NOT_CONFIGURED")
        environment = self.environment or backend.restore_environment(self.task.environment_id)  # type: ignore[attr-defined]
        environment_id = resolve_environment_id(
            baseline, configured_id=self.task.environment_id, record=environment,
        )

        store_ref: dict[str, QDStore] = {}

        def trial_factory(event_id: str, genome_id: str) -> TrialRecord:
            genome = store_ref["store"].load_genome(genome_id)
            return _prepare_trial(self.workspace, event_id, genome, contract)

        store = QDStore(self.workspace, trial_factory=trial_factory)
        store_ref["store"] = store
        ledger = BudgetLedger(self.workspace)
        validator = ContractValidator(self.workspace, genome_loader=self._load_genome(store))
        policy, approval_handler = _approval(self.task.approval_mode)
        rules = _mutable_allowlist(self.task.mutable_parameters)
        proposal_engine = self.proposal_engine or ModelProposalEngine(self.model, mutable_parameters=rules)

        def service_factory(trial: TrialRecord, genome: WorkflowGenome, attempt: Workspace) -> Mapping[str, Tool]:
            if any(step.kind != "script" or step.binding_id != "script_run" for step in genome.steps):
                raise ValueError("QD_SERVICE_BINDING_UNAVAILABLE")
            scripts = {ref.path: ref.sha256 for ref in genome.script_refs}
            input_bindings: dict[str, str] = {}
            for step in genome.steps:
                values = step.parameters.get("input_refs", {})
                if isinstance(values, Mapping):
                    input_bindings.update({str(key): str(value) for key, value in values.items()})
            sandbox = SandboxService(
                workspace=attempt.root, trial_id=trial.trial_id,
                genome_sha256=genome.content_sha256, scripts=scripts,
                input_bindings=input_bindings, backend=backend,
            )
            # Historical environments without a frozen plan still execute, but cannot
            # become cross-task local method candidates.
            plan = None
            plan_path = self.workspace.root / ".photomatagent" / "capability-plans" / f"{environment.plan_sha256}.json"
            if plan_path.is_file() and not plan_path.is_symlink():
                import json
                from datetime import datetime, timedelta, timezone
                from photomatagent.capability_acquisition.models import InstallPlan
                raw = json.loads(plan_path.read_text())
                raw["expires_at"] = datetime.now(timezone.utc) + timedelta(days=1)
                plan = InstallPlan.model_validate(raw)
            sandbox.register_environment(environment, plan=plan)
            delegate = ScriptRunTool(sandbox)
            declared = _declared_output_paths(trial, contract)
            return {"script_run": _QDArtifactScriptTool(delegate, sandbox, attempt, declared)}

        executor = TrialExecutor(
            self.workspace, model=self.model, validator=validator,
            service_tool_factory=service_factory,
            permission_policy=policy, approval_handler=approval_handler,
            budget_handle=ledger, runtime_environment_id=environment_id,
        )
        methods = {step.binding_id: step.kind for step in baseline.steps}
        archive = QDArchive(
            store, contract, method_class_by_binding=methods,
        )
        coordinator = EvolutionCoordinator(
            store=store, ledger=ledger, proposal_engine=proposal_engine,
            executor=executor, validator=validator, archive=archive,
            contract=contract, baseline=baseline, issues=issues,
            budget=self.task.budget, seed=self.seed,
            is_solved=lambda validation: validation.qualification == "PASS",
            cancel_event=self.cancel_event,
        )
        return await coordinator.run_event(self.task.task_id)

    @staticmethod
    def _load_genome(store: QDStore):
        def load(genome_id: str) -> WorkflowGenome | None:
            try:
                return store.load_genome(genome_id)
            except (OSError, ValueError):
                return None
        return load


def _configured_model(workspace: Path) -> ModelProvider:
    config = read_preferred_config(DotEnvConfig(workspace))
    if config is None or config.provider == "fake" or not config.model:
        raise ValueError("QD_MODEL_NOT_CONFIGURED")
    if config.api_key_env is None:
        raise ValueError("QD_MODEL_CREDENTIALS_NOT_CONFIGURED")
    return create_provider(config.provider, config.model)


def _workspace_path(workspace: Workspace, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("QD_PATH_INVALID")
    resolved = workspace.resolve(value, must_exist=False)
    current = workspace.root
    for part in path.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("QD_PATH_UNSAFE")
    return resolved


def _approval(mode: str) -> tuple[PermissionPolicy, Any]:
    if mode == "auto":
        return AllowAllPolicy(), AutoApproveHandler()
    if mode == "deny":
        return DenyAllPolicy(), DenyHandler()
    if mode == "ask":
        from photomatagent.cli.prompt import make_approval_handler, make_prompt_session
        return AskPolicy(), make_approval_handler(make_prompt_session())
    raise ValueError("QD_APPROVAL_MODE_INVALID")


def _mutable_allowlist(raw_rules: tuple[dict[str, Any], ...]) -> Mapping[Any, Any]:
    result: dict[tuple[str, str], dict[str, tuple[Any, ...]]] = {}
    for raw in raw_rules:
        step_id, binding_id = raw.get("step_id"), raw.get("binding_id")
        parameters = raw.get("parameters")
        if not isinstance(step_id, str) or not isinstance(binding_id, str) or not isinstance(parameters, dict):
            raise ValueError("PARAMETER_POLICY_INVALID")
        result[(step_id, binding_id)] = {
            key: tuple(tuple(choice) if key == "argv" and isinstance(choice, list) else choice
                       for choice in values)
            for key, values in parameters.items()
        }
    return result


def _prepare_trial(
    workspace: Workspace, event_id: str, genome: WorkflowGenome,
    contract: EvaluationContract,
) -> TrialRecord:
    identity = hashlib.sha256(f"{event_id}\0{genome.content_sha256}".encode()).hexdigest()[:32]
    trial_id = f"trial_{identity}"
    session_id = f"qd_{identity}"
    attempt_dir = f"tmp/qd/{trial_id}"
    attempt_root = _workspace_path(workspace, attempt_dir)
    if attempt_root.exists():
        if attempt_root.is_symlink() or not attempt_root.is_dir():
            raise ValueError("QD_ATTEMPT_PATH_UNSAFE")
    else:
        attempt_root.mkdir(parents=True)
    input_paths: set[str] = set()
    for step in genome.steps:
        refs = step.parameters.get("input_refs", {})
        if isinstance(refs, Mapping):
            for path in refs.values():
                if not isinstance(path, str):
                    raise ValueError("QD_INPUT_PATH_INVALID")
                input_paths.add(path)
    input_digests: list[ArtifactDigest] = []
    frozen = [(ref.path, ref.sha256, ref.size_bytes, ref.media_type) for ref in genome.script_refs]
    for source_path in sorted(input_paths):
        source_ref = _source_digest(workspace, source_path)
        frozen.append((source_path, source_ref.sha256, source_ref.size_bytes, source_ref.media_type))
        input_digests.append(source_ref.model_copy(update={"path": f"{attempt_dir}/{source_path}"}))
    for relative, digest, size, _media in frozen:
        source = _source_file(workspace, relative)
        payload = source.read_bytes()
        if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("QD_INPUT_CHANGED")
        target = attempt_root / relative
        _write_frozen_file(attempt_root, target, payload, digest)
    return TrialRecord(
        trial_id=trial_id, event_id=event_id, genome_id=genome.genome_id,
        contract_sha256=_contract_digest(contract), input_digests=tuple(input_digests),
        environment_sha256=genome.environment_ref, seed_set=contract.seed_set,
        status="RESERVED", session_id=session_id, attempt_dir=attempt_dir,
        runtime_trace_ref=f".photomatagent/sessions/{session_id}/events.jsonl",
        cost={"model_tokens": 0.0, "runtime_seconds": 0.0}, output_refs=(),
    )


def _source_file(workspace: Workspace, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("QD_INPUT_PATH_INVALID")
    current = workspace.root
    for part in path.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("QD_INPUT_PATH_UNSAFE")
    resolved = workspace.resolve(value, must_exist=True)
    if not resolved.is_file():
        raise ValueError("QD_INPUT_PATH_INVALID")
    return resolved


def _source_digest(workspace: Workspace, value: str) -> ArtifactDigest:
    source = _source_file(workspace, value)
    payload = source.read_bytes()
    return ArtifactDigest(
        path=value, sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload), media_type="application/octet-stream",
    )


def _write_frozen_file(root: Path, target: Path, payload: bytes, digest: str) -> None:
    current = root
    for part in target.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("QD_ATTEMPT_PATH_UNSAFE")
    if target.exists():
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError("QD_FROZEN_INPUT_CHANGED")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".qd-input-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _contract_digest(contract: EvaluationContract) -> str:
    from photomatagent.scientific.evolution.qd.validation import canonical_contract_sha256
    return canonical_contract_sha256(contract)


def _declared_output_paths(trial: TrialRecord, contract: EvaluationContract) -> frozenset[str]:
    result: set[str] = set()
    for check in contract.checks:
        path = check.parameters.get("artifact_path")
        if isinstance(path, str):
            result.add(path[len(trial.attempt_dir) + 1:] if path.startswith(trial.attempt_dir + "/") else path)
    return frozenset(result)


class _QDArtifactScriptTool(Tool):
    """Translate verified sandbox artifacts into the contract-owned trial workspace."""

    def __init__(
        self, delegate: ScriptRunTool, sandbox: SandboxService, attempt: Workspace,
        declared_paths: frozenset[str],
    ) -> None:
        self.delegate = delegate
        self.sandbox = sandbox
        self.attempt = attempt
        self.declared_paths = declared_paths
        for field in ("name", "description", "short_description", "input_schema", "exposure", "namespace", "source", "tags", "searchable", "cost_class"):
            setattr(self, field, getattr(delegate, field))

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        result = await self.delegate.execute(arguments)
        if result.is_error:
            return result
        try:
            payload = json.loads(result.output)
            execution_id = payload["execution_id"]
            payload["archived_artifacts"] = self.sandbox.archive_artifacts(
                execution_id, destination=self.attempt, declared_paths=self.declared_paths,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return ToolResult(output=f"QD_ARTIFACT_ARCHIVE_FAILED: {exc}", is_error=True)
        return ToolResult(output=json.dumps(payload, ensure_ascii=False, separators=(",", ":")), data=result.data)
