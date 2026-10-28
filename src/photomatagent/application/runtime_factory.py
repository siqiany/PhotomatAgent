"""UI-neutral assembly of the PhotomatAgent runtime.

The application layer owns construction of the model loop and its one tool
registry.  Entry points supply only their interaction policy and event sinks;
no UI dependency belongs in this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any

from photomatagent.logging.event_logger import EventLogger
from photomatagent.models.factory import create_provider
from photomatagent.runtime.budget import BudgetState
from photomatagent.runtime.context_config import resolve_context_config
from photomatagent.runtime.context_engine import ContextEngineConfig
from photomatagent.runtime.loop import AgentRuntime, EventSink
from photomatagent.runtime.state import ConversationState
from photomatagent.runtime.permissions import (
    ApprovalHandler,
    ApprovalSettings,
    PermissionPolicy,
    PermissionDecision,
    PermissionResult,
    SwitchablePermissionPolicy,
    default_permission_policy,
)
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.factory import create_default_registry
from photomatagent.workspace import Workspace

if TYPE_CHECKING:
    from photomatagent.capability_acquisition.sandbox import SandboxService


@dataclass(frozen=True)
class InteractionStrategy:
    """Interaction hooks supplied by an entry point.

    Approval is deliberately a protocol-level callback.  The application
    factory never creates a prompt or imports a console implementation.
    """

    permission_policy: PermissionPolicy | None = None
    approval_handler: ApprovalHandler | None = None
    fresh_approval: bool = False
    application_approval_root: Path | str | None = None


@dataclass(frozen=True)
class RuntimeResources:
    """Objects constructed for one runtime and closed on its execution loop."""

    owners: tuple[object, ...]

    async def aclose(self) -> None:
        seen: set[int] = set()
        for owner in self.owners:
            if id(owner) in seen:
                continue
            seen.add(id(owner))
            close = (
                getattr(owner, "aclose", None)
                or getattr(owner, "close_all", None)
                or getattr(owner, "close", None)
            )
            if not callable(close):
                continue
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                # One failed owner must not strand the remaining resources or queue.
                continue


class _WorkbenchCapabilityPolicy(PermissionPolicy):
    """Allow the task-local acquisition loop without granting broad host tools."""

    def __init__(self, task_id: str, fallback: PermissionPolicy) -> None:
        self._prefix = Path("tmp") / "workbench" / task_id
        self._fallback = fallback

    async def check(self, tool_name: str, arguments: dict[str, object]) -> PermissionResult:
        if tool_name in {"capability_search", "capability_inspect", "capability_install", "script_run", "skill_run"}:
            return PermissionResult(PermissionDecision.ALLOW, "task-local capability workflow")
        if tool_name in {"write", "edit"}:
            path = arguments.get("path")
            if isinstance(path, str):
                relative = Path(path)
                if (not relative.is_absolute() and ".." not in relative.parts
                        and relative.parts[:len(self._prefix.parts)] == self._prefix.parts):
                    return PermissionResult(PermissionDecision.ALLOW, "task-local workbench file")
        return await self._fallback.check(tool_name, arguments)


class RuntimeFactory:
    """Build an :class:`AgentRuntime` with the canonical tool registry."""

    @staticmethod
    async def build_owned(**kwargs: Any) -> tuple[AgentRuntime, RuntimeResources]:
        """Return the runtime with explicit owners for background execution."""
        created_owners: list[object] = []
        try:
            runtime, _logger = RuntimeFactory.build(
                **kwargs, resource_owners=created_owners
            )
        except BaseException:
            await RuntimeResources(tuple(created_owners)).aclose()
            raise
        return runtime, RuntimeResources(tuple(created_owners))

    @staticmethod
    def build(
        *,
        provider: str = "fake",
        model: str | None = None,
        workspace: Workspace | Path | str | None = None,
        workspace_root: Workspace | Path | str | None = None,
        session_dir: Path | str | None = None,
        session_id: str | None = None,
        budget: BudgetState | None = None,
        max_iterations: int = 10000,
        event_sinks: Sequence[EventSink] | None = None,
        event_sink: EventSink | None = None,
        execution_services: Mapping[str, object] | None = None,
        capability_sandbox_service: SandboxService | None = None,
        capability_task_id: str | None = None,
        capability_run_id: str | None = None,
        interaction: InteractionStrategy | None = None,
        permission_policy: PermissionPolicy | None = None,
        approval_handler: ApprovalHandler | None = None,
        scientific_state: ScientificState | None = None,
        conversation_state: ConversationState | None = None,
        log_events: bool = True,
        context_engine_config: ContextEngineConfig | None = None,
        context_limit_tokens: int | None = None,
        compact_trigger_tokens: int | None = None,
        evaluation_isolation: bool = False,
        resource_owners: list[object] | None = None,
    ) -> tuple[AgentRuntime, EventLogger | None]:
        """Construct the runtime and optional session event logger.

        ``execution_services`` is intentionally a narrow injection point for
        registered scientific services.  Unknown keys are ignored by the
        registry layer rather than treated as model authority.
        """
        if workspace is not None and workspace_root is not None:
            raise ValueError("pass workspace or workspace_root, not both")
        supplied_workspace = workspace if workspace is not None else workspace_root
        boundary = (
            supplied_workspace
            if isinstance(supplied_workspace, Workspace)
            else Workspace(supplied_workspace or Path.cwd())
        )
        scientific = scientific_state if scientific_state is not None else ScientificState()
        strategy = interaction or InteractionStrategy()
        approval_root = strategy.application_approval_root
        resolved_approval_root = (
            boundary.resolve(str(approval_root), must_exist=False)
            if approval_root is not None
            else None
        )
        services = dict(execution_services or {})
        sandbox = services.get("capability_sandbox_service", capability_sandbox_service)
        if sandbox is None and capability_task_id is not None and capability_run_id is not None:
            import hashlib
            import re
            from photomatagent.capability_acquisition.local_backend import LocalDevBackend
            from photomatagent.capability_acquisition.sandbox import SandboxService

            if not re.fullmatch(r"task_[0-9a-f]{32}", capability_task_id) or not re.fullmatch(r"run_[0-9a-f]{32}", capability_run_id):
                raise ValueError("INVALID_WORKBENCH_ID")
            sandbox = SandboxService(
                workspace=boundary.root, trial_id=capability_run_id,
                genome_sha256=hashlib.sha256(capability_run_id.encode()).hexdigest(),
                scripts={}, dynamic_script_prefix=f"tmp/workbench/{capability_task_id}",
                backend=LocalDevBackend(boundary.root / ".photomatagent" / "capabilities" / capability_task_id),
            )
        registry = create_default_registry(
            scientific,
            boundary,
            application_approval_root=resolved_approval_root,
            evaluation_isolation=evaluation_isolation,
            capability_sandbox_service=sandbox,  # type: ignore[arg-type]
            resource_owners=resource_owners,
            local_skill_task_id=capability_task_id,
            local_skill_run_id=capability_run_id,
            local_skill_service=services.get("local_skill_service"),
        )
        model_provider = create_provider(provider, model)
        if resource_owners is not None:
            resource_owners.append(model_provider)
        effective_budget = budget or BudgetState(max_iterations=max_iterations)
        if evaluation_isolation:
            effective_context_config = ContextEngineConfig()
        elif context_engine_config is not None:
            effective_context_config = context_engine_config
        else:
            effective_context_config = resolve_context_config(
                context_limit_tokens=context_limit_tokens,
                compact_trigger_tokens=compact_trigger_tokens,
            )
            if capability_task_id is not None:
                effective_context_config = effective_context_config.model_copy(
                    update={"output_recovery_attempts": 1}
                )

        policy = strategy.permission_policy or permission_policy or default_permission_policy()
        if capability_task_id is not None and strategy.permission_policy is None and permission_policy is None:
            policy = _WorkbenchCapabilityPolicy(capability_task_id, policy)
        settings = (
            None
            if strategy.fresh_approval
            else ApprovalSettings(boundary.root)
        )
        policy = SwitchablePermissionPolicy(policy, settings=settings)
        sinks = list(event_sinks or ())
        if event_sink is not None:
            sinks.append(event_sink)
        sessions_base = (
            Path(session_dir)
            if session_dir is not None and Path(session_dir).is_absolute()
            else boundary.resolve(str(session_dir or ".photomatagent/sessions"), must_exist=False)
        )
        logger = (
            EventLogger(sessions_base, session_id=session_id)
            if log_events
            else None
        )
        if logger is not None:
            sinks.append(logger.log)
        runtime = AgentRuntime(
            model=model_provider,
            tools=registry,
            workspace=boundary,
            scientific_state=scientific,
            conversation_state=conversation_state,
            permission_policy=policy,
            budget=effective_budget,
            approval_handler=strategy.approval_handler or approval_handler,
            event_sinks=sinks,
            session_id=logger.session_id if logger else session_id,
            fresh_approval=strategy.fresh_approval,
            application_approval_root=resolved_approval_root,
            context_engine_config=effective_context_config,
        )
        return runtime, logger


__all__ = ["InteractionStrategy", "RuntimeFactory", "RuntimeResources"]
