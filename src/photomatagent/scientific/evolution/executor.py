"""Application orchestration for one persisted scientific evolution episode."""

from __future__ import annotations

import inspect
import json
import os
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from photomatagent.errors import ToolExecutionError
from photomatagent.logging.event_logger import EventLogger
from photomatagent.redaction import redact_text
from photomatagent.runtime.budget import BudgetState
from photomatagent.runtime.context import ContextBuilder
from photomatagent.runtime.context_engine import (
    ContextEngine,
    ContextEngineConfig,
    ProviderContextSummarizer,
)
from photomatagent.runtime.events import RuntimeEvent
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import (
    AllowAllPolicy,
    DenyAllPolicy,
    PolicyRule,
    SwitchablePermissionPolicy,
    default_permission_policy,
)
from photomatagent.scientific.evolution.artifacts import (
    EpisodeArtifactCollector,
    materialize_primary_result,
    sha256_file,
)
from photomatagent.scientific.evolution.comparison import evaluate_machine_acceptance
from photomatagent.scientific.evolution.events import bounded_summary
from photomatagent.scientific.evolution.models import (
    ArtifactRef,
    CostSnapshot,
    EpisodeRecord,
    EvolutionTask,
    RevisionPlan,
    new_episode_owner_token,
)
from photomatagent.scientific.evolution.revision import format_revision_instruction
from photomatagent.scientific.evolution.service import (
    EvolutionOperationConflict,
    EvolutionService,
    MutationResult,
)
from photomatagent.scientific.evolution.store import EvaluationLease, EvolutionStore
from photomatagent.scientific.loop import (
    ScientificJudge,
    ScientificLoopConfig,
    ScientificLoopController,
    ScientificLoopSummary,
)
from photomatagent.scientific.state import ScientificState
from photomatagent.skills.loader import SkillLoader
from photomatagent.scientific.backends.mock import MockBackend
from photomatagent.workspace import Workspace
from photomatagent.tools.bridges import ToolCallBridge, ToolDescribeTool, ToolSearchTool
from photomatagent.tools.base import Tool
from photomatagent.tools.calculator import CalculatorTool
from photomatagent.tools.echo import EchoTool
from photomatagent.tools.factory import create_default_registry
from photomatagent.tools.mock_calculation import MockCalculationTool
from photomatagent.tools.registry import ToolRegistry
from photomatagent.tools.scientific_state_inspect import ScientificStateInspectTool
from photomatagent.tools.surface import ToolCatalog, ToolSurfaceConfig, ToolSurfacePlanner

EventSink = Callable[[RuntimeEvent], Awaitable[None] | None]
_REVISION_INSTRUCTION_MAX_CHARS = 12_000
_FRESH_STRATEGY_GUIDANCE = {
    "STATIC": "Use the fixed baseline procedure without learned task history.",
    "EVIDENCE_FIRST": "Prioritize independently obtained structured evidence.",
    "DIVERSITY_FIRST": "Prioritize diverse independently checked candidates.",
    "UNCERTAINTY_FIRST": "Prioritize resolving the largest scientific uncertainty.",
}
_FRESH_TOOL_TYPES: dict[str, type[object]] = {
    "echo": EchoTool,
    "calculator": CalculatorTool,
    "scientific_state_inspect": ScientificStateInspectTool,
    "mock.run_calculation": MockCalculationTool,
    "tool_search": ToolSearchTool,
    "tool_describe": ToolDescribeTool,
    "tool_call": ToolCallBridge,
}
_FRESH_TOOL_INSTANCE_FIELDS: dict[str, set[str]] = {
    "echo": set(),
    "calculator": set(),
    "scientific_state_inspect": {"_state"},
    "mock.run_calculation": {"backend"},
    "tool_search": {"catalog", "default_limit", "max_limit"},
    "tool_describe": {"catalog"},
    "tool_call": set(),
}
_TOOL_CONTRACT_FIELDS = (
    "name",
    "description",
    "short_description",
    "exposure",
    "namespace",
    "source",
    "tags",
    "searchable",
    "cost_class",
)


def _class_tool_contract(tool_type: type[object]) -> tuple[object, ...]:
    schema = json.dumps(
        getattr(tool_type, "input_schema"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return (
        *(getattr(tool_type, field) for field in _TOOL_CONTRACT_FIELDS),
        schema,
        getattr(tool_type, "execute"),
        getattr(tool_type, "tool_metadata"),
    )


_FRESH_TOOL_CONTRACTS = {
    name: _class_tool_contract(tool_type)
    for name, tool_type in _FRESH_TOOL_TYPES.items()
}


class RuntimeFactory(Protocol):
    def __call__(self, **kwargs: Any) -> AgentRuntime: ...


def _trusted_fresh_permission_components(
    supplied: AgentRuntime,
) -> tuple[SwitchablePermissionPolicy, object | None]:
    policy = supplied.permission_policy
    if (
        type(policy) is not SwitchablePermissionPolicy
        or set(vars(policy)) != {"_base", "_settings", "_session_allow_all"}
        or policy._settings is not None  # type: ignore[attr-defined]
        or policy._session_allow_all  # type: ignore[attr-defined]
    ):
        raise ValueError("fresh runtime permission wrapper is not trusted")
    base = policy._base  # type: ignore[attr-defined]
    if type(base) is PolicyRule:
        expected = default_permission_policy()
        if (
            set(vars(base)) != {"_rules", "_default"}
            or base._rules != expected._rules  # type: ignore[attr-defined]
            or base._default != expected._default  # type: ignore[attr-defined]
        ):
            raise ValueError("fresh runtime permission rules were modified")
        cloned_base = default_permission_policy()
        from photomatagent.cli.prompt import CLIApprovalHandler

        handler = supplied._approval_handler
        if (
            handler is None
            or type(handler) is not CLIApprovalHandler
            or set(vars(handler)) != {"session"}
        ):
            raise ValueError("fresh ask approval handler is not trusted")
        cloned_handler: object | None = CLIApprovalHandler(handler.session)
    elif type(base) is AllowAllPolicy:
        if vars(base):
            raise ValueError("fresh runtime base permission policy was modified")
        cloned_base = AllowAllPolicy()
        cloned_handler = None
    elif type(base) is DenyAllPolicy:
        if vars(base):
            raise ValueError("fresh runtime base permission policy was modified")
        cloned_base = DenyAllPolicy()
        cloned_handler = None
    else:
        raise ValueError("fresh runtime base permission policy is not trusted")
    if type(base) is not PolicyRule and supplied._approval_handler is not None:
        raise ValueError("fresh non-interactive policy has an approval handler")
    return SwitchablePermissionPolicy(cloned_base, settings=None), cloned_handler


def _build_trusted_fresh_runtime(
    supplied: AgentRuntime,
    *,
    workspace_root: Path,
    application_approval_root: Path,
) -> AgentRuntime:
    """Rebuild the authoritative runtime without any supplier-owned tools."""

    if type(supplied) is not AgentRuntime:
        raise TypeError("runtime_factory must return an exact AgentRuntime")
    if type(supplied.workspace) is not Workspace or supplied.workspace.root != workspace_root:
        raise ValueError("runtime_factory did not bind the evaluation workspace")
    if not supplied._fresh_approval:
        raise ValueError("runtime_factory disabled fresh approval isolation")
    if supplied._application_approval_root != application_approval_root:
        raise ValueError("runtime_factory did not bind the evaluation approval namespace")
    if supplied.conversation_state.messages:
        raise ValueError("runtime_factory returned a reused conversation")
    if type(supplied.budget) is not BudgetState or any(
        value != 0 for value in supplied.budget.snapshot().values()
    ):
        raise ValueError("runtime_factory returned a reused budget")
    if supplied.scientific_state != ScientificState():
        raise ValueError("runtime_factory returned reused scientific state")
    if not isinstance(supplied.session_id, str) or not supplied.session_id:
        raise ValueError("runtime_factory returned an invalid session ID")

    permission, approval_handler = _trusted_fresh_permission_components(supplied)
    workspace = Workspace(workspace_root)
    scientific = ScientificState()
    registry = create_default_registry(
        scientific,
        workspace,
        application_approval_root=application_approval_root,
        evaluation_isolation=True,
    )
    return AgentRuntime(
        model=supplied._model,
        tools=registry,
        workspace=workspace,
        scientific_state=scientific,
        permission_policy=permission,
        budget=BudgetState(max_iterations=supplied.budget.max_iterations),
        approval_handler=approval_handler,  # type: ignore[arg-type]
        session_id=supplied.session_id,
        fresh_approval=True,
        application_approval_root=application_approval_root,
    )


@dataclass(frozen=True, slots=True)
class EpisodeExecutionResult:
    """The durable completion produced by one executor invocation."""

    episode: EpisodeRecord
    runtime_session_id: str
    scientific_state_path: str
    scientific_summary: ScientificLoopSummary
    artifact: ArtifactRef
    cost: CostSnapshot


class ScientificEpisodeExecutor:
    """Run one reserved episode through the authoritative scientific loop.

    This class owns lifecycle orchestration only. Tool execution remains wholly
    inside ``AgentRuntime`` through the ``ScientificLoopController`` it wraps.
    """

    def __init__(
        self,
        store: EvolutionStore,
        *,
        event_logger: EventLogger | None = None,
    ) -> None:
        self.store = store
        self.event_logger = event_logger
        self.service = EvolutionService(store)

    async def execute(
        self,
        *,
        task: EvolutionTask,
        episode: EpisodeRecord,
        runtime: AgentRuntime,
        config: ScientificLoopConfig,
        revision: RevisionPlan | None = None,
        judge: ScientificJudge | None = None,
        on_event: EventSink | None = None,
        owner_token: str | None = None,
    ) -> EpisodeExecutionResult:
        if episode.execution_mode != "FRESH_EVALUATION":
            self.service.reconcile(task.evolution_id)
        self._validate_execution(
            task=task,
            episode=episode,
            runtime=runtime,
            owner_token=owner_token,
        )
        runtime_session_id = self._runtime_session_id(runtime)
        self._validate_unused_session(
            runtime_session_id,
            evolution_id=task.evolution_id,
            version=episode.version,
        )
        event_log_path = self._event_log_path()
        persisted_revision = self._persisted_revision(task, episode, revision)
        instruction = self._instruction(task, episode, persisted_revision)
        collector = EpisodeArtifactCollector()
        started_at = time.monotonic()
        runtime_logs_events = self._runtime_uses_event_logger(runtime)
        try:
            if episode.execution_mode == "FRESH_EVALUATION":
                if owner_token is None:
                    raise ValueError("fresh evaluation requires an owner token")
                running_result = self.service.mark_evaluation_running(
                    task.evolution_id,
                    episode.version,
                    owner_token=owner_token,
                    runtime_session_id=runtime_session_id,
                    event_log_path=event_log_path,
                )
            else:
                running_result = self.service.mark_episode_running(
                    task.evolution_id,
                    episode.version,
                    owner_token=owner_token,
                    runtime_session_id=runtime_session_id,
                    event_log_path=event_log_path,
                )
            running = running_result.entity
            controller = ScientificLoopController(
                target=episode.target_snapshot.to_target_spec(),
                runtime=runtime,
                config=config,
                judge=judge,
                event_sinks=[],
                session_id=runtime_session_id,
            )
            await self._publish(running_result, on_event)
            async for event in controller.run(goal=instruction):
                collector.observe(event)
                if self.event_logger is not None and (
                    not runtime_logs_events or event.run_id == controller.run_id
                ):
                    await self.event_logger.log(event)
                await self._deliver(on_event, event)
            if controller.summary is None:
                raise RuntimeError("scientific loop ended without a summary")
            artifact = materialize_primary_result(
                workspace=runtime.workspace,
                evolution_id=task.evolution_id,
                version=episode.version,
                conversation=runtime.conversation_state,
                collector=collector,
            )
            if episode.execution_mode == "FRESH_EVALUATION":
                artifact = self._copy_evaluation_artifact(
                    task.evolution_id,
                    episode.version,
                    runtime.workspace,
                    artifact,
                )
                state_path = self.store.write_evaluation_scientific_state(
                    task.evolution_id,
                    episode.version,
                    runtime.scientific_state,
                )
            else:
                state_path = self.store.write_scientific_state(
                    task.evolution_id,
                    episode.version,
                    runtime.scientific_state,
                )
            scientific_state_path = self.store.workspace.relative(state_path)
            cost = self._cost_snapshot(runtime, started_at)
            completion = running.model_copy(
                update={
                    "scientific_state_path": scientific_state_path,
                    "summary": controller.summary,
                    "artifact": artifact,
                    "cost": cost,
                }
            )
            if persisted_revision is not None:
                completion = completion.model_copy(
                    update={
                        "acceptance_results": evaluate_machine_acceptance(
                            plan=persisted_revision,
                            episode=completion,
                            state=runtime.scientific_state,
                        )
                    }
                )
            if episode.execution_mode == "FRESH_EVALUATION":
                if owner_token is None:
                    raise ValueError("fresh evaluation requires an owner token")
                completed_result = self.service.complete_evaluation(
                    task.evolution_id,
                    episode.version,
                    result=completion,
                    owner_token=owner_token,
                )
            else:
                completed_result = self.service.complete_episode(
                    task.evolution_id,
                    episode.version,
                    result=completion,
                    owner_token=owner_token,
                )
            comparison_result = None
            if persisted_revision is not None and episode.parent_version is not None:
                comparison_result = self.service.compare(
                    task.evolution_id,
                    episode.parent_version,
                    episode.version,
                )
            await self._publish(completed_result, on_event)
            if comparison_result is not None:
                await self._publish(comparison_result, on_event)
        except BaseException as exc:
            await self._reconcile_exception(
                task=task,
                episode=episode,
                error=exc,
                on_event=on_event,
                owner_token=owner_token,
            )
            raise
        completed = completed_result.entity
        return EpisodeExecutionResult(
            episode=completed,
            runtime_session_id=runtime_session_id,
            scientific_state_path=scientific_state_path,
            scientific_summary=controller.summary,
            artifact=artifact,
            cost=cost,
        )

    @staticmethod
    def _validate_canonical_fresh_tool(name: str, tool: object) -> None:
        expected_type = _FRESH_TOOL_TYPES[name]
        if type(tool) is not expected_type:
            raise ValueError(
                "fresh runtime tool implementations are not the trusted allowlist"
            )
        if set(vars(tool)) != _FRESH_TOOL_INSTANCE_FIELDS[name]:
            raise ValueError(f"fresh canonical tool contract was modified: {name}")
        execute = getattr(tool, "execute")
        actual_contract = (
            *(getattr(tool, field) for field in _TOOL_CONTRACT_FIELDS),
            json.dumps(
                getattr(tool, "input_schema"),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
            getattr(execute, "__func__", execute),
            getattr(
                getattr(tool, "tool_metadata"),
                "__func__",
                getattr(tool, "tool_metadata"),
            ),
        )
        if (
            getattr(execute, "__self__", None) is not tool
            or getattr(getattr(tool, "tool_metadata"), "__self__", None) is not tool
            or getattr(type(tool), "tool_metadata") is not Tool.tool_metadata
            or actual_contract != _FRESH_TOOL_CONTRACTS[name]
        ):
            raise ValueError(f"fresh canonical tool contract was modified: {name}")

    def _validate_execution(
        self,
        *,
        task: EvolutionTask,
        episode: EpisodeRecord,
        runtime: AgentRuntime,
        owner_token: str | None,
    ) -> None:
        if episode.execution_mode == "FRESH_EVALUATION":
            if type(runtime) is not AgentRuntime:
                raise ValueError("fresh runtime implementation is not trusted")
            if type(runtime.workspace) is not Workspace:
                raise ValueError("fresh runtime workspace implementation is not trusted")
            if episode.evaluation_workspace_path is None:
                raise ValueError("fresh evaluation workspace provenance is missing")
            expected_workspace = self.service.validate_evaluation_workspace(episode)
            if runtime.workspace.root != expected_workspace:
                raise ValueError("fresh runtime must use its isolated evaluation workspace")
            if type(runtime._tools) is not ToolRegistry:
                raise ValueError("fresh runtime registry implementation is not trusted")
            if runtime._tools.sealed:
                raise ValueError("fresh runtime registry was sealed before attestation")
            context_builder = runtime._context_builder
            if (
                type(context_builder) is not ContextBuilder
                or set(vars(context_builder)) != {"skill_loader"}
            ):
                raise ValueError("fresh context builder is not trusted")
            try:
                trusted_skills_dir = runtime.workspace.resolve(
                    ".photomatagent/fresh-context/skills",
                    must_exist=True,
                )
            except (OSError, ValueError, ToolExecutionError) as exc:
                raise ValueError("fresh context skill binding is not trusted") from exc
            skill_loader = context_builder.skill_loader
            sources = skill_loader.sources if type(skill_loader) is SkillLoader else []
            if (
                type(skill_loader) is not SkillLoader
                or set(vars(skill_loader))
                != {"skills_dir", "_explicit_skills_dir", "diagnostics", "_sources"}
                or skill_loader.skills_dir != trusted_skills_dir
                or not skill_loader._explicit_skills_dir
                or skill_loader.diagnostics
                or len(sources) != 1
                or sources[0].path != trusted_skills_dir
                or any(trusted_skills_dir.iterdir())
            ):
                raise ValueError("fresh context skill binding is not trusted")
            context_engine = runtime._context_engine
            if type(context_engine) is not ContextEngine:
                raise ValueError("fresh context engine implementation is not trusted")
            if context_engine.snapshot() != {
                "compaction_state": None,
                "compacted_message_count": 0,
                "compaction_count": 0,
            }:
                raise ValueError("fresh runtime requires a blank context snapshot")
            if context_engine.config != ContextEngineConfig():
                raise ValueError("fresh context engine configuration is not trusted")
            summarizer = context_engine.summarizer
            if (
                type(summarizer) is not ProviderContextSummarizer
                or summarizer.provider is not runtime._model
                or summarizer.last_usage is not None
            ):
                raise ValueError("fresh context summarizer binding is not trusted")
            tool_surface = runtime._tool_surface
            if (
                type(tool_surface) is not ToolSurfacePlanner
                or tool_surface.registry is not runtime._tools
                or type(tool_surface.catalog) is not ToolCatalog
                or tool_surface.catalog._registry is not runtime._tools
                or tool_surface.config != ToolSurfaceConfig()
            ):
                raise ValueError("fresh tool-surface planner binding is not trusted")
            if runtime._model_context_limit != context_engine.config.context_limit_tokens:
                raise ValueError("fresh context limit binding is not trusted")
            actual_tools = {tool.name: tool for tool in runtime._tools.list_tools()}
            if set(actual_tools) != set(_FRESH_TOOL_TYPES):
                raise ValueError(
                    "fresh runtime tool implementations are not the trusted allowlist"
                )
            for name, tool in actual_tools.items():
                self._validate_canonical_fresh_tool(name, tool)
            state_tool = actual_tools["scientific_state_inspect"]
            if state_tool._state is not runtime.scientific_state:  # type: ignore[attr-defined]
                raise ValueError("fresh state tool is not bound to the runtime state")
            mock_tool = actual_tools["mock.run_calculation"]
            if type(mock_tool.backend) is not MockBackend:  # type: ignore[attr-defined]
                raise ValueError("fresh mock tool backend is not trusted")
            search_tool = actual_tools["tool_search"]
            describe_tool = actual_tools["tool_describe"]
            if (
                type(search_tool.catalog) is not ToolCatalog  # type: ignore[attr-defined]
                or search_tool.catalog is not describe_tool.catalog  # type: ignore[attr-defined]
                or search_tool.catalog._registry is not runtime._tools  # type: ignore[attr-defined]
            ):
                raise ValueError("fresh tool catalog binding is not trusted")
            policy = runtime.permission_policy
            if type(policy) is not SwitchablePermissionPolicy:
                raise ValueError("fresh runtime permission wrapper is not trusted")
            if policy._settings is not None or policy._session_allow_all:  # type: ignore[attr-defined]
                raise ValueError("fresh runtime cannot inherit approval settings")
            base = policy._base  # type: ignore[attr-defined]
            if type(base) is PolicyRule:
                expected_policy = default_permission_policy()
                if (
                    base._rules != expected_policy._rules  # type: ignore[attr-defined]
                    or base._default != expected_policy._default  # type: ignore[attr-defined]
                ):
                    raise ValueError("fresh runtime permission rules were modified")
            elif type(base) not in {AllowAllPolicy, DenyAllPolicy}:
                raise ValueError("fresh runtime base permission policy is not trusted")
            if type(base) is PolicyRule:
                from photomatagent.cli.prompt import CLIApprovalHandler

                handler = runtime._approval_handler
                if handler is None or type(handler) is not CLIApprovalHandler:
                    raise ValueError("fresh ask approval handler is not trusted")
            elif runtime._approval_handler is not None:
                raise ValueError("fresh non-interactive policy has an approval handler")
            if not runtime._fresh_approval:
                raise ValueError("fresh runtime approval isolation is disabled")
            expected_approval_root = expected_workspace / (
                f".photomatagent/evolution-approvals/{task.evolution_id}/"
                f"{episode.version}_{episode.episode_id}"
            )
            if runtime._application_approval_root != expected_approval_root:
                raise ValueError("fresh runtime approval namespace is not episode-scoped")
        elif runtime.workspace.root != self.store.workspace.root:
            raise ValueError("runtime and evolution store must use the same workspace")
        if runtime.conversation_state.messages:
            raise ValueError("evolution episodes require a fresh runtime conversation")
        if any(value != 0 for value in runtime.budget.snapshot().values()):
            raise ValueError("evolution episodes require a fresh runtime budget")
        if (
            episode.execution_mode == "FRESH_EVALUATION"
            and runtime.scientific_state != ScientificState()
        ):
            raise ValueError("fresh evaluation requires a blank ScientificState")
        if task.evolution_id != episode.evolution_id:
            raise ValueError("task and episode belong to different evolution tasks")
        stored_task = self.store.load_task(task.evolution_id)
        stored_episode = (
            self.store.load_evaluation_episode(task.evolution_id, episode.version)
            if episode.execution_mode == "FRESH_EVALUATION"
            else self.store.load_episode(task.evolution_id, episode.version)
        )
        if episode.execution_mode == "FRESH_EVALUATION":
            if stored_task.current_evaluation_version != episode.version:
                raise ValueError("evaluation is not the current reserved evaluation")
        elif stored_task.current_version != episode.version:
            raise ValueError("episode is not the task's current reserved version")
        if stored_episode != episode or stored_episode.status != "RESERVED":
            raise ValueError("executor requires the exact persisted RESERVED episode")
        if (
            stored_episode.owner_token is not None
            and stored_episode.owner_token != owner_token
        ):
            raise ValueError("executor owner token does not match the reserved episode")
        if task.goal != stored_task.goal or task.target != stored_task.target:
            raise ValueError("task does not match the persisted immutable task snapshot")
        if episode.execution_mode == "FRESH_EVALUATION":
            runtime._tools.seal()

    def _runtime_session_id(self, runtime: AgentRuntime) -> str:
        runtime_id = runtime.session_id
        logger_id = self.event_logger.session_id if self.event_logger is not None else None
        if not isinstance(runtime_id, str) or not runtime_id:
            raise ValueError("runtime session ID is unavailable")
        if logger_id is not None and logger_id != runtime_id:
            raise ValueError("event logger and runtime session IDs do not match")
        return runtime_id

    def _event_log_path(self) -> str | None:
        if self.event_logger is None:
            return None
        try:
            path = self.store.workspace.resolve(
                str(self.event_logger.events_path),
                must_exist=True,
            )
        except (OSError, ValueError, ToolExecutionError) as exc:
            raise ValueError("event log path must resolve inside the workspace") from exc
        if not path.is_file():
            raise ValueError("event log path must be a regular workspace file")
        return self.store.workspace.relative(path)

    def _validate_unused_session(
        self,
        session_id: str,
        *,
        evolution_id: str,
        version: str,
    ) -> None:
        for existing_task in self.store.list_tasks():
            if existing_task.current_version is None:
                continue
            latest = int(existing_task.current_version[1:])
            for number in range(1, latest + 1):
                existing_version = f"v{number:03d}"
                existing = self.store.load_episode(
                    existing_task.evolution_id,
                    existing_version,
                )
                if (
                    existing.runtime_session_id == session_id
                    and (existing.evolution_id, existing.version)
                    != (evolution_id, version)
                ):
                    raise ValueError(
                        "runtime session is already attributed to another episode"
                    )
            for number in range(1, len(existing_task.evaluation_episode_ids) + 1):
                existing = self.store.load_evaluation_episode(
                    existing_task.evolution_id,
                    f"v{number:03d}",
                )
                if (
                    existing.runtime_session_id == session_id
                    and (existing.evolution_id, existing.version)
                    != (evolution_id, version)
                ):
                    raise ValueError(
                        "runtime session is already attributed to another evaluation"
                    )

    def _persisted_revision(
        self,
        task: EvolutionTask,
        episode: EpisodeRecord,
        supplied: RevisionPlan | None,
    ) -> RevisionPlan | None:
        if episode.execution_mode == "FRESH_EVALUATION":
            if supplied is not None or episode.revision_plan_id is not None:
                raise ValueError("fresh evaluation must not receive a revision")
            if episode.applied_feedback_id is not None or episode.strategy_id is None:
                raise ValueError("fresh evaluation provenance is not isolated")
            strategy = self.store.load_strategy(
                episode.evolution_id,
                episode.strategy_id,
            )
            if (
                strategy.strategy_id != episode.strategy_id
                or strategy.arm != episode.strategy_arm
                or strategy.strategy_sha256 != episode.strategy_sha256
                or strategy.cutoff_at != episode.strategy_cutoff_at
                or strategy.strategy_sha256 is None
                or strategy.cutoff_at is None
            ):
                raise ValueError(
                    "fresh evaluation strategy does not match its frozen snapshot"
                )
            return None
        if episode.revision_plan_id is None:
            if supplied is not None:
                raise ValueError("initial episode must not receive a revision")
            return None
        persisted = self.store.load_revision(
            episode.evolution_id,
            episode.revision_plan_id,
        )
        if supplied is None or supplied != persisted:
            raise ValueError("supplied revision does not equal the persisted RevisionPlan")
        if (
            not persisted.confirmed
            or persisted.has_blocking_ambiguity
            or persisted.evolution_id != episode.evolution_id
            or persisted.source_version != episode.parent_version
            or persisted.feedback_id != episode.applied_feedback_id
        ):
            raise ValueError("persisted RevisionPlan does not match the reserved episode")
        if episode.strategy_id is None:
            raise ValueError("revised episode has no persisted strategy")
        strategy = self.store.load_strategy(episode.evolution_id, episode.strategy_id)
        from photomatagent.scientific.evolution.strategy import FixedStrategySelector

        canonical = FixedStrategySelector().select(task, persisted)
        if (
            strategy != canonical
            or strategy.arm != episode.strategy_arm
            or strategy.parameters.get("revision_id") != persisted.revision_id
        ):
            raise ValueError("persisted strategy does not match the reserved episode")
        return persisted

    def _runtime_uses_event_logger(self, runtime: AgentRuntime) -> bool:
        if self.event_logger is None:
            return False
        expected_path = self.event_logger.events_path.resolve()
        for sink in runtime._event_sinks:
            owner = getattr(sink, "__self__", None)
            if (
                isinstance(owner, EventLogger)
                and owner.events_path.resolve() == expected_path
            ):
                return True
        return False

    async def _publish(
        self,
        result: MutationResult[object],
        on_event: EventSink | None,
    ) -> None:
        evolution_id = getattr(result.entity, "evolution_id", None)
        if isinstance(evolution_id, str):
            self.store.flush_event_outbox(evolution_id)
            self.store.append_events(
                evolution_id,
                result.events,
                idempotency_scope="durable-outbox",
            )
        for event in result.events:
            if self.event_logger is not None:
                await self.event_logger.log(event)
            await self._deliver(on_event, event)

    @staticmethod
    async def _deliver(sink: EventSink | None, event: RuntimeEvent) -> None:
        if sink is None:
            return
        pending = sink(event)
        if inspect.isawaitable(pending):
            await pending

    def _instruction(
        self,
        task: EvolutionTask,
        episode: EpisodeRecord,
        revision: RevisionPlan | None,
    ) -> str:
        if episode.execution_mode == "FRESH_EVALUATION":
            guidance = _FRESH_STRATEGY_GUIDANCE[episode.strategy_arm]
            return (
                f"{task.goal}\n\n"
                f"Frozen evaluation strategy: {episode.strategy_arm}. {guidance}"
            )
        if revision is None:
            return task.goal
        revision_text = format_revision_instruction(
            revision,
            strategy=episode.strategy_arm,
        )
        return (
            f"{task.goal}\n\n"
            "--- Confirmed structured revision contract ---\n"
            f"{revision_text[:_REVISION_INSTRUCTION_MAX_CHARS]}"
        )

    @staticmethod
    def _cost_snapshot(runtime: AgentRuntime, started_at: float) -> CostSnapshot:
        budget = runtime.budget
        return CostSnapshot(
            input_tokens=budget.input_tokens,
            output_tokens=budget.output_tokens,
            tool_calls=budget.tool_calls,
            runtime_seconds=max(0.0, time.monotonic() - started_at),
            hpc_cost=budget.compute_cost,
        )

    @staticmethod
    def _bounded_error(exc: BaseException) -> str:
        return bounded_summary(redact_text(f"{type(exc).__name__}: {exc}"))

    async def _reconcile_exception(
        self,
        *,
        task: EvolutionTask,
        episode: EpisodeRecord,
        error: BaseException,
        on_event: EventSink | None,
        owner_token: str | None,
    ) -> None:
        recovery_error: BaseException | None = None
        for _ in range(2):
            try:
                persisted = (
                    self.store.load_evaluation_episode(
                        task.evolution_id,
                        episode.version,
                    )
                    if episode.execution_mode == "FRESH_EVALUATION"
                    else self.store.load_episode(task.evolution_id, episode.version)
                )
                persisted_task = self.store.load_task(task.evolution_id)
                if episode.execution_mode == "FRESH_EVALUATION":
                    if persisted.status == "COMPLETED":
                        return
                    if persisted.status == "FAILED":
                        return
                    if owner_token is None:
                        raise ValueError("fresh evaluation requires an owner token")
                    failed = self.service.fail_evaluation(
                        task.evolution_id,
                        episode.version,
                        self._bounded_error(error),
                        owner_token=owner_token,
                    )
                    await self._publish(failed, on_event)
                    return
                if persisted.status == "COMPLETED":
                    if persisted_task.status == "RUNNING":
                        reconciled = self.service.complete_episode(
                            task.evolution_id,
                            episode.version,
                            result=persisted,
                            owner_token=owner_token,
                        )
                        await self._publish(reconciled, on_event)
                    return
                if persisted.status == "FAILED":
                    if persisted_task.status == "RUNNING":
                        reconciled = self.service.fail_episode(
                            task.evolution_id,
                            episode.version,
                            persisted.error or self._bounded_error(error),
                            owner_token=owner_token,
                        )
                        await self._publish(reconciled, on_event)
                    return
                if persisted.status in {"RESERVED", "RUNNING"}:
                    failed = self.service.fail_episode(
                        task.evolution_id,
                        episode.version,
                        self._bounded_error(error),
                        owner_token=owner_token,
                    )
                    await self._publish(failed, on_event)
                    return
            except BaseException as exc:
                recovery_error = exc
                continue
        if recovery_error is not None:
            safe_recovery_error = redact_text(
                f"{type(recovery_error).__name__}: {recovery_error}"
            )
            error.add_note(
                "episode terminal-state reconciliation also failed: "
                f"{safe_recovery_error}"
            )

    def _copy_evaluation_artifact(
        self,
        evolution_id: str,
        version: str,
        isolated_workspace: Workspace,
        artifact: ArtifactRef,
    ) -> ArtifactRef:
        source = isolated_workspace.resolve(artifact.path, must_exist=True)
        if not source.is_file() or source.is_symlink():
            raise ValueError("isolated evaluation result must be a regular file")
        destination_relative = (
            f"user_output/{evolution_id}/evaluations/{version}/result.md"
        )
        destination = self.store.workspace.resolve(
            destination_relative,
            must_exist=False,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".result.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle, source.open("rb") as reader:
                descriptor = -1
                for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise ValueError("evaluation result already exists") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)
        return ArtifactRef(
            path=destination_relative,
            media_type=artifact.media_type,
            size_bytes=destination.stat().st_size,
            sha256=sha256_file(destination),
        )


async def run_fresh_evaluation(
    *,
    service: EvolutionService,
    task: EvolutionTask,
    strategy_id: str,
    executor: ScientificEpisodeExecutor,
    runtime_factory: RuntimeFactory,
    config: ScientificLoopConfig | None = None,
    judge: ScientificJudge | None = None,
    on_event: EventSink | None = None,
    owner_token: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    _lease: EvaluationLease | None = None,
) -> EpisodeExecutionResult:
    """Claim and run an isolated evaluation from only task + frozen strategy."""

    if _lease is None:
        with service.store.evaluation_lease(task.evolution_id) as held_lease:
            return await run_fresh_evaluation(
                service=service,
                task=task,
                strategy_id=strategy_id,
                executor=executor,
                runtime_factory=runtime_factory,
                config=config,
                judge=judge,
                on_event=on_event,
                owner_token=owner_token,
                provider=provider,
                model=model,
                _lease=held_lease,
            )

    authoritative = service.get(task.evolution_id)
    if (
        task.goal != authoritative.goal
        or task.target != authoritative.target
        or task.input_sha256 != authoritative.input_sha256
        or task.task_group_id != authoritative.task_group_id
    ):
        raise ValueError("task does not match the immutable persisted task input")
    token = owner_token or new_episode_owner_token()
    claim = service.claim_fresh_evaluation(
        task.evolution_id,
        strategy_id=strategy_id,
        owner_token=token,
        provider=provider,
        model=model,
        lease=_lease,
        reclaim_reserved_owner=True,
    )
    service.store.flush_event_outbox(task.evolution_id)
    service.store.append_events(
        task.evolution_id,
        claim.events,
        idempotency_scope="durable-outbox",
    )
    approval_root = Path(".photomatagent/evolution-approvals") / task.evolution_id / (
        f"{claim.episode.version}_{claim.episode.episode_id}"
    )
    if claim.episode.evaluation_workspace_path is None:
        raise ValueError("evaluation workspace provenance is missing")
    evaluation_workspace = service.validate_evaluation_workspace(claim.episode)
    try:
        supplied_runtime = runtime_factory(
            workspace_root=evaluation_workspace,
            scientific_state=ScientificState(),
            fresh_approval=True,
            application_approval_root=approval_root,
        )
        if not service.store.has_active_evaluation_lease(
            _lease,
            task.evolution_id,
        ):
            raise EvolutionOperationConflict(
                "fresh evaluation lost its execution lease"
            )
        evaluation_workspace = service.validate_evaluation_workspace(claim.episode)
        runtime = _build_trusted_fresh_runtime(
            supplied_runtime,
            workspace_root=evaluation_workspace,
            application_approval_root=evaluation_workspace / approval_root,
        )
        await executor._publish(
            MutationResult(claim.episode, claim.events),
            on_event,
        )
        return await executor.execute(
            task=claim.task,
            episode=claim.episode,
            runtime=runtime,
            config=config or ScientificLoopConfig(),
            judge=judge,
            on_event=on_event,
            owner_token=claim.owner_token,
        )
    except BaseException as exc:
        try:
            persisted = service.store.load_evaluation_episode(
                task.evolution_id,
                claim.episode.version,
            )
            if persisted.status in {"RESERVED", "RUNNING"}:
                failed = service.fail_evaluation(
                    task.evolution_id,
                    claim.episode.version,
                    ScientificEpisodeExecutor._bounded_error(exc),
                    owner_token=claim.owner_token,
                )
                await executor._publish(failed, on_event)
        except BaseException as recovery_exc:
            exc.add_note(
                "fresh-evaluation recovery failed: "
                f"{ScientificEpisodeExecutor._bounded_error(recovery_exc)}"
            )
        raise


__all__ = [
    "EpisodeExecutionResult",
    "EventSink",
    "ScientificEpisodeExecutor",
    "RuntimeFactory",
    "run_fresh_evaluation",
]
