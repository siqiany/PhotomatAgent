"""Interactive CLI assembly and event consumption."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from prompt_toolkit.shortcuts import PromptSession
from rich.console import Console

from photomatagent.cli.prompt import make_approval_handler, make_prompt_session
from photomatagent.cli.render import ChatRenderer
from photomatagent.logging.event_logger import EventLogger, default_sessions_dir
from photomatagent.models.factory import create_provider
from photomatagent.runtime.budget import BudgetState
from photomatagent.runtime.loop import AgentRuntime, EventSink
from photomatagent.runtime.permissions import (
    ApprovalSettings,
    AllowAllPolicy,
    DenyAllPolicy,
    PermissionPolicy,
    SwitchablePermissionPolicy,
    default_permission_policy,
)
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.factory import create_default_registry
from photomatagent.workspace import Workspace

ApprovalMode = Literal["ask", "auto", "deny"]


def build_runtime(
    *,
    provider: str = "fake",
    model: str | None = None,
    workspace_root: Path | str | None = None,
    approval: ApprovalMode = "ask",
    max_iterations: int = 25,
    prompt_session: PromptSession | None = None,
    session_dir: Path | str | None = None,
    log_events: bool = True,
) -> tuple[AgentRuntime, EventLogger | None]:
    workspace = Workspace(workspace_root or Path.cwd())
    scientific = ScientificState()
    registry = create_default_registry(scientific, workspace)
    model_provider = create_provider(provider, model)
    budget = BudgetState(max_iterations=max_iterations)

    policy: PermissionPolicy
    approval_handler = None
    if approval == "ask":
        policy = default_permission_policy()
        approval_handler = make_approval_handler(prompt_session or make_prompt_session())
    elif approval == "auto":
        policy = AllowAllPolicy()
    else:
        policy = DenyAllPolicy()

    sinks: list[EventSink] = []
    logger = EventLogger(session_dir or default_sessions_dir()) if log_events else None
    if logger is not None:
        sinks.append(logger.log)

    policy = SwitchablePermissionPolicy(
        policy, settings=ApprovalSettings(workspace.root)
    )
    runtime = AgentRuntime(
        model=model_provider,
        tools=registry,
        workspace=workspace,
        scientific_state=scientific,
        permission_policy=policy,
        budget=budget,
        approval_handler=approval_handler,
        event_sinks=sinks,
        session_id=logger.session_id if logger else None,
    )
    return runtime, logger


async def run_goal(console: Console, runtime: AgentRuntime, goal: str) -> None:
    renderer = ChatRenderer(console)
    try:
        async for event in runtime.run(goal):
            renderer.handle(event)
    except Exception:
        # Runtime has already emitted ProviderFailed/LoopFailed. The CLI keeps
        # the session alive and does not expose an SDK traceback by default.
        pass
    finally:
        renderer.flush_agent_text()


async def run_interactive_chat(
    console: Console, runtime: AgentRuntime, session: PromptSession
) -> None:
    from photomatagent.cli.commands import ChatCommandRouter

    commands = ChatCommandRouter(console, runtime, runtime.workspace)
    console.print(
        "[dim]Type your research goal or [/][bold]/help[/][dim] for commands; "
        "[/][bold]/exit[/][dim] quits.[/]"
    )
    while True:
        try:
            goal = await session.prompt_async("❯ ")
        except (EOFError, KeyboardInterrupt):
            break
        normalized = goal.strip().lower()
        if normalized in {"/exit", "/quit", "exit", "quit"}:
            break
        if goal.strip().startswith("/"):
            await commands.execute(goal.strip())
            continue
        if goal.strip():
            await run_goal(console, runtime, goal)


async def run_chat(
    *,
    provider: str = "fake",
    model: str | None = None,
    workspace_root: Path | str | None = None,
    approval: ApprovalMode = "ask",
    max_iterations: int = 25,
    log_events: bool = True,
    goal: str | None = None,
) -> None:
    console = Console()
    prompt_session = make_prompt_session() if goal is None or approval == "ask" else None
    runtime, logger = build_runtime(
        provider=provider,
        model=model,
        workspace_root=workspace_root,
        approval=approval,
        max_iterations=max_iterations,
        prompt_session=prompt_session,
        log_events=log_events,
    )
    if goal is not None:
        await run_goal(console, runtime, goal)
    else:
        assert prompt_session is not None
        await run_interactive_chat(console, runtime, prompt_session)
    if logger is not None:
        console.print(f"[dim]events logged: {logger.events_path}[/]")
