"""Compile task-owned, bounded scientific revisions into the existing QD loop."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from photomatagent.application.errors import ServiceError
from photomatagent.application.models import BudgetSpec, WorkbenchRevisionPlan
from photomatagent.application.skills import CheckerSpec
from photomatagent.capability_acquisition.models import (
    EnvironmentRecord,
    ExecutionRecord,
    ExecutionRequest,
    InstallPlan,
)
from photomatagent.capability_acquisition.sandbox import SandboxService, _sha256_file
from photomatagent.scientific.evolution.qd.config import (
    EvolutionBudget,
    MutableParameterRule,
    QDConfig,
)
from photomatagent.scientific.evolution.qd.models import (
    AcceptanceCheck,
    ArtifactDigest,
    EvaluationContract,
    RevisionIssue,
    WorkflowGenome,
)
from photomatagent.scientific.evolution.qd.runtime_adapter import (
    QDRuntimeAdapter,
    QDTaskRecord,
    _mutable_allowlist,
)
from photomatagent.scientific.evolution.qd.operators import (
    SeededRandomParameterProposalEngine,
)
from photomatagent.scientific.evolution.qd.validation import canonical_contract_sha256
from photomatagent.workspace import Workspace


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


class QDRevisionOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    execution_id: StrictStr = Field(pattern=r"^exec_[0-9a-f]{32}$")
    checker: CheckerSpec
    search_mode: Literal["directed", "random"] = "directed"
    argv_choices: list[list[StrictStr]] = Field(min_length=2, max_length=32)
    max_trials: StrictInt = Field(default=6, ge=1, le=32)
    seed: StrictInt = Field(default=0, ge=0, le=2**31 - 1)

    @model_validator(mode="after")
    def bounded_choices(self) -> QDRevisionOptions:
        if len({tuple(choice) for choice in self.argv_choices}) < 2:
            raise ValueError("QD_PARAMETER_SPACE_UNSUPPORTED")
        if not self.checker.field.strip():
            raise ValueError("QD_CHECKER_FIELD_REQUIRED")
        if self.checker.checker_id == "json_numeric_bounds":
            if self.checker.minimum is None and self.checker.maximum is None:
                raise ValueError("QD_CHECKER_BOUNDS_REQUIRED")
            if (
                self.checker.minimum is not None
                and self.checker.maximum is not None
                and self.checker.minimum > self.checker.maximum
            ):
                raise ValueError("QD_CHECKER_BOUNDS_INVALID")
        MutableParameterRule(
            step_id="execute",
            binding_id="script_run",
            parameters={"argv": self.argv_choices},
        )
        return self


class QDStrategyService:
    """Trusted host operations reserve runs; all proposed tools remain runtime-owned."""

    def __init__(
        self,
        *,
        store: Any,
        manager: Any,
        workspace: Path,
        backend_factory: Any = None,
        model_factory: Any = None,
    ) -> None:
        self.store, self.manager = store, manager
        self.workspace = Workspace(workspace)
        self.backend_factory, self.model_factory = backend_factory, model_factory
        manager.register_host_operation("qd_revision", self.execute)

    def compile_revision(
        self,
        plan: WorkbenchRevisionPlan,
        options: QDRevisionOptions,
        budget: BudgetSpec,
    ) -> dict[str, object]:
        try:
            root = self.workspace.resolve(
                ".photomatagent/executions/" + options.execution_id, must_exist=True
            )
            for parent in (root, *root.parents):
                if parent == self.workspace.root:
                    break
                if parent.is_symlink():
                    raise ValueError("SOURCE_PATH_UNSAFE")
            provenance = json.loads((root / "request.json").read_text())
            request = ExecutionRequest.model_validate(provenance["request"])
            record = ExecutionRecord.model_validate_json(
                (root / "record.json").read_text()
            )
            environment = EnvironmentRecord.model_validate(provenance["environment"])
            checker = options.checker
            raw_plan = dict(provenance["plan"] or {})
            raw_plan["expires_at"] = datetime.now(timezone.utc) + timedelta(days=1)
            install_plan = InstallPlan.model_validate(raw_plan)
            if (
                request.trial_id != plan.source_run_id
                or request.workspace_root != str(self.workspace.root)
                or not request.script_path.startswith(f"tmp/workbench/{plan.task_id}/")
                or request.execution_id != record.execution_id
                or request.script_digest != record.script_sha256
                or environment.environment_id != request.environment_id
                or environment.status != "READY"
                or environment.environment_sha256 != record.environment_sha256
                or environment.plan_sha256 != install_plan.plan_sha256
            ):
                raise ValueError("SOURCE_EXECUTION_BINDING_MISMATCH")
            if record.state != "COMPLETED" or record.exit_code != 0 or record.timed_out:
                raise ValueError("SOURCE_EXECUTION_NOT_COMPLETE")
            if list(request.argv) not in options.argv_choices:
                raise ValueError("QD_BASELINE_ARGV_NOT_IN_SPACE")
            if not any(
                artifact.path == checker.output_path for artifact in record.artifacts
            ):
                raise ValueError("QD_OUTPUT_NOT_IN_SOURCE")
            service = SandboxService(
                workspace=self.workspace.root,
                trial_id=request.trial_id,
                genome_sha256=request.genome_sha256,
                scripts={},
            )
            service._requests[request.execution_id] = request
            service._records[request.execution_id] = record
            inputs, input_sha = service.execution_inputs(request.execution_id)
            script = self.workspace.resolve(
                str((root / "script.py").relative_to(self.workspace.root))
            )
            if script.is_symlink() or _sha256_file(script) != request.script_digest:
                raise ValueError("SOURCE_SCRIPT_CHANGED")
            output = checker.output_path
            if (
                output.startswith("/")
                or "\\" in output
                or any(p in ("", ".", "..") for p in output.split("/"))
            ):
                raise ValueError("QD_OUTPUT_PATH_UNSUPPORTED")
            if (
                checker.checker_id == "json_numeric_sum"
                and checker.input_key not in inputs
            ):
                raise ValueError("QD_CHECKER_INPUT_UNAVAILABLE")
            if (
                checker.checker_id == "json_numeric_bounds"
                and checker.minimum is None
                and checker.maximum is None
            ):
                raise ValueError("QD_CHECKER_BOUNDS_REQUIRED")
            task = self.store.task(plan.task_id)
            goal_sha = digest(task.goal)
            binding = {
                "goal_sha256": goal_sha,
                "source_run_id": plan.source_run_id,
                "source_digest": plan.source_digest,
                "execution_id": request.execution_id,
                "script_sha256": request.script_digest,
                "input_sha256": input_sha,
                "environment_sha256": environment.environment_sha256,
                "install_plan_sha256": install_plan.plan_sha256,
                "checker": checker.model_dump(),
                "argv_choices": options.argv_choices,
                "seed": options.seed,
                "search_mode": options.search_mode,
            }
            identity = digest(binding)
            script_ref = ArtifactDigest(
                path=script.relative_to(self.workspace.root).as_posix(),
                sha256=request.script_digest,
                size_bytes=script.stat().st_size,
                media_type="text/x-python",
            )
            input_refs = {
                key: path.relative_to(self.workspace.root).as_posix()
                for key, path in inputs.items()
            }
            contract = EvaluationContract(
                contract_id="contract_" + identity[:32],
                task_family="workbench_script",
                version="1",
                input_schema={"goal_sha256": goal_sha, "input_sha256": input_sha},
                output_schema={},
                checks=(
                    AcceptanceCheck(
                        check_id="scientific_output",
                        kind="registered_checker",
                        artifact_role="result",
                        parameters={
                            "artifact_path": output,
                            "checker": checker.model_dump(),
                            "input_refs": input_refs,
                        },
                    ),
                ),
                quality_metric="scientific_output",
                quality_direction="maximize",
                quality_tolerance=0.0,
                evaluation_suite_sha256=identity,
                seed_set=(options.seed,),
                descriptor_config={},
            )
            genome = WorkflowGenome.from_behavior(
                genome_id="baseline_" + identity[:32],
                task_family=contract.task_family,
                entrypoint="execute",
                steps=(
                    {
                        "step_id": "execute",
                        "kind": "script",
                        "binding_id": "script_run",
                        "input_roles": list(inputs),
                        "output_roles": ["result"],
                        "parameters": {
                            "script_path": script_ref.path,
                            "input_refs": input_refs,
                            "argv": list(request.argv),
                        },
                    },
                ),
                script_refs=(script_ref,),
                environment_ref=environment.environment_sha256,
                applicability=(task.goal,),
                recovery=(
                    "Keep goal, inputs, script, dependencies and checker frozen.",
                ),
            )
            actionable = tuple(
                item
                for item in plan.issues
                if item.status == "CORRECTION"
                and item.cannot_process_reason is None
                and (item.actions or item.acceptance_test)
            )
            issues = tuple(
                RevisionIssue(
                    issue_id=item.issue_id,
                    source_kind="human",
                    source_ref=plan.plan_id,
                    artifact_sha256=plan.source_digest,
                    status="open",
                    severity="high",
                    step_ids=("execute",),
                    requested_actions=(
                        item.problem,
                        *item.actions,
                        *((item.acceptance_test,) if item.acceptance_test else ()),
                    ),
                    preserve=(task.goal, "Frozen checker and input contract"),
                    acceptance_check_ids=("scientific_output",),
                )
                for item in actionable
            )
            if not issues:
                raise ValueError("QD_NO_ACTIONABLE_FEEDBACK")
            qd_task = QDTaskRecord(
                task_id="event_"
                + digest(
                    {
                        "plan": plan.plan_id,
                        "binding": binding,
                        "budget": budget.model_dump(),
                    }
                )[:32],
                contract=contract.model_dump(mode="json"),
                baseline_genome=genome.model_dump(mode="json"),
                issues=tuple(issue.model_dump(mode="json") for issue in issues),
                environment_id=environment.environment_id,
                approval_mode="auto",
                mutable_parameters=(
                    {
                        "step_id": "execute",
                        "binding_id": "script_run",
                        "parameters": {"argv": options.argv_choices},
                    },
                ),
                budget=EvolutionBudget(
                    max_trials=options.max_trials,
                    max_proposals=max(1, options.max_trials * 2),
                    max_model_calls=budget.max_model_calls,
                    max_model_tokens=budget.max_tokens,
                    max_wall_seconds=float(budget.wall_seconds),
                ),
            )
            return {
                "operation": "qd_revision",
                "plan_id": plan.plan_id,
                "task": qd_task.model_dump(mode="json"),
                "environment": environment.model_dump(mode="json"),
                "binding": binding,
                "binding_sha256": identity,
                "seed": options.seed,
                "contract_sha256": canonical_contract_sha256(contract),
            }
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise ServiceError(
                "QD_UNSUPPORTED",
                f"QD 无法编译此来源：{exc}",
                status_code=409,
                next_action="保留原反馈，选择普通反馈修订；或补充可信检查器与有界参数空间。",
            ) from exc

    async def execute(self, run: Any) -> None:
        from photomatagent.models.factory import create_provider

        operation = run.request.host_operation
        if (
            operation is None
            or operation.get("operation") != "qd_revision"
            or run.request.strategy != "qd"
        ):
            raise ValueError("QD_HOST_OPERATION_INVALID")
        binding = operation["binding"]
        if (
            digest(binding) != operation["binding_sha256"]
            or digest(self.store.task(run.task_id).goal) != binding["goal_sha256"]
        ):
            raise ValueError("QD_FROZEN_GOAL_CHANGED")
        task = QDTaskRecord.model_validate(operation["task"])
        environment = EnvironmentRecord.model_validate(operation["environment"])
        options = self.manager.runtime_options
        model = (self.model_factory or create_provider)(
            str(options.get("provider", "fake")), options.get("model")
        )
        try:
            await self._execute_qd_run(
                run, operation, binding, task, environment, model
            )
        finally:
            close = getattr(model, "aclose", None) or getattr(model, "close", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result

    async def _execute_qd_run(
        self,
        run: Any,
        operation: dict[str, Any],
        binding: dict[str, Any],
        task: QDTaskRecord,
        environment: EnvironmentRecord,
        model: Any,
    ) -> None:
        from photomatagent.capability_acquisition.local_backend import LocalDevBackend
        from photomatagent.capability_acquisition.container_backend import (
            PodmanSandboxBackend,
        )
        from photomatagent.scientific.evolution.qd.store import QDStore
        from photomatagent.scientific.evolution.qd.budget import BudgetLedger

        service = self

        class ObservedModel:
            def __init__(self) -> None:
                self.provider = model.provider
                self.model = model.model

            async def stream(self, request: Any) -> Any:
                service.store.append_event_for_run(
                    run.run_id,
                    "model_request_started",
                    {"cost_settlement": "UNSETTLED", "phase": "qd"},
                )
                async for event in model.stream(request):
                    from photomatagent.models.types import ModelCompleted

                    if isinstance(event, ModelCompleted):
                        usage = event.response.usage
                        service.store.append_event_for_run(
                            run.run_id,
                            "model_response_completed",
                            {
                                "cost_settlement": "KNOWN",
                                "phase": "qd",
                                "usage": {
                                    "input_tokens": usage.input_tokens,
                                    "output_tokens": usage.output_tokens,
                                    "total_tokens": usage.resolved_total_tokens,
                                },
                            },
                        )
                    yield event

        try:
            storage = (
                self.workspace.root / ".photomatagent" / "capabilities" / run.task_id
            )
            backend = (
                self.backend_factory(storage, environment.backend)
                if self.backend_factory
                else (
                    LocalDevBackend(storage)
                    if environment.backend == "local_dev"
                    else PodmanSandboxBackend(storage)
                )
            )
            rules = _mutable_allowlist(task.mutable_parameters)
            proposal_engine = (
                SeededRandomParameterProposalEngine(mutable_parameters=rules)
                if binding["search_mode"] == "random"
                else None
            )
            self.store.append_event_for_run(
                run.run_id,
                "qd_compiled",
                {
                    "goal_sha256": binding["goal_sha256"],
                    "source_run_id": binding["source_run_id"],
                    "execution_id": binding["execution_id"],
                    "contract_sha256": operation["contract_sha256"],
                    "candidate_space": {
                        "argv_choices": binding["argv_choices"],
                        "seed": operation["seed"],
                        "search_mode": binding["search_mode"],
                        "sampling_rule": (
                            "ModelProposalEngine single-step mutation constrained by the frozen argv allowlist"
                            if binding["search_mode"] == "directed"
                            else "host-seeded uniform sampling without replacement from the frozen argv allowlist"
                        ),
                    },
                },
            )
            adapter = QDRuntimeAdapter(
                workspace=self.workspace,
                task=task,
                config=QDConfig(enabled=True),
                model=ObservedModel(),
                backend=backend,
                environment=environment,
                seed=int(operation["seed"]),
                proposal_engine=proposal_engine,
            )
            async with asyncio.timeout(run.request.budget.wall_seconds):
                outcome = await adapter.run_event()
            self.store.append_event_for_run(
                run.run_id, "qd_completed", outcome.model_dump(mode="json")
            )
            self.store.append_event_for_run(
                run.run_id,
                "loop_completed",
                {
                    "reason": "completed"
                    if outcome.status == "SOLVED"
                    else "qd_" + outcome.status.lower()
                },
            )
        finally:
            qd_store = QDStore(
                self.workspace,
                trial_factory=lambda event, genome: (_ for _ in ()).throw(
                    ValueError("READ_ONLY")
                ),
            )
            trial_records = qd_store.trials_for_event(task.task_id)
            trial_by_genome = {
                trial.genome_id: trial.trial_id for trial in trial_records
            }
            lineage = _proposal_lineage(
                qd_store, task.task_id, task.budget.max_proposals, trial_by_genome
            )
            for trial in trial_records:
                genome = qd_store.load_genome(trial.genome_id)
                self.store.link_source(
                    run.task_id, "qd_trial", trial.trial_id, genome.content_sha256
                )
                validation = None
                try:
                    validation = qd_store.load_current_validation(trial.trial_id)
                except Exception:
                    pass
                self.store.append_event_for_run(
                    run.run_id,
                    "qd_trial",
                    {
                        "trial_id": trial.trial_id,
                        "genome_id": trial.genome_id,
                        "status": trial.status,
                        "qualification": validation.qualification
                        if validation
                        else "UNKNOWN",
                        "outcomes": [
                            o.model_dump(mode="json") for o in validation.outcomes
                        ]
                        if validation
                        else [],
                        "output_refs": [
                            o.model_dump(mode="json") for o in trial.output_refs
                        ],
                        "cost": dict(trial.cost),
                        **lineage.get(
                            trial.genome_id,
                            {
                                "parent_ids": [],
                                "parent_trial_ids": [],
                                "changed_fields": [],
                                "feedback_refs": [],
                                "operator": "host_baseline",
                            },
                        ),
                        "failure_evidence": [
                            {
                                "check_id": outcome.check_id,
                                "verdict": outcome.verdict,
                                "reason_code": outcome.reason_code,
                            }
                            for outcome in validation.outcomes
                            if outcome.verdict != "PASS"
                        ]
                        if validation
                        else [{"code": "VALIDATION_MISSING"}],
                    },
                )
                for ref in trial.output_refs:
                    artifact_id = (
                        "artifact_"
                        + digest(
                            {
                                "run": run.run_id,
                                "trial": trial.trial_id,
                                "path": ref.path,
                            }
                        )[:32]
                    )
                    try:
                        self.store.artifact(artifact_id)
                    except KeyError:
                        self.store.register_artifact(
                            artifact_id=artifact_id,
                            task_id=run.task_id,
                            run_id=run.run_id,
                            path=ref.path,
                            filename=Path(ref.path).name,
                            sha256=ref.sha256,
                            media_type=ref.media_type,
                            size_bytes=ref.size_bytes,
                        )
            ledger = BudgetLedger(self.workspace).snapshot(scope=f"qd_{task.task_id}_")
            self.store.append_event_for_run(run.run_id, "qd_cost", ledger.to_dict())


def _proposal_lineage(
    store: Any, event_id: str, max_batches: int, trial_by_genome: dict[str, str]
) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for batch_index in range(max_batches):
        batch = store.load_proposal_batch(event_id, batch_index)
        if batch is None:
            break
        proposal_batch, genomes, _parents, _elapsed = batch
        for proposal in proposal_batch.proposals:
            child = genomes.get(proposal.genome_id)
            if child is None:
                continue
            changed: list[str] = []
            parent_ids = list(proposal.parent_ids)
            if parent_ids:
                try:
                    parent = store.load_genome(parent_ids[0])
                    old_steps = {step.step_id: step for step in parent.steps}
                    for step in child.steps:
                        old = old_steps.get(step.step_id)
                        if old is None:
                            changed.append(f"{step.step_id}.added")
                            continue
                        keys = set(old.parameters) | set(step.parameters)
                        changed.extend(
                            f"{step.step_id}.{key}"
                            for key in sorted(keys)
                            if old.parameters.get(key) != step.parameters.get(key)
                        )
                except (OSError, ValueError):
                    changed = []
            records[proposal.genome_id] = {
                "parent_ids": parent_ids,
                "parent_trial_ids": [
                    trial_by_genome[parent]
                    for parent in parent_ids
                    if parent in trial_by_genome
                ],
                "changed_fields": changed,
                "feedback_refs": list(proposal.feedback_refs),
                "operator": proposal.operator,
                "change_summary": proposal.change_summary,
                "model_provenance": dict(proposal.model_provenance),
            }
    return records
