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
from photomatagent.sessions.store import SessionSnapshot
from photomatagent.tools.factory import create_default_registry
from photomatagent.workspace import Workspace

ApprovalMode = Literal["ask", "auto", "deny"]


def build_runtime(
    *,
    provider: str = "fake",
    model: str | None = None,
    workspace_root: Path | str | None = None,
    approval: ApprovalMode = "ask",
    max_iterations: int = 10000,
    prompt_session: PromptSession | None = None,
    session_dir: Path | str | None = None,
    session_id: str | None = None,
    log_events: bool = True,
    scientific_state: ScientificState | None = None,
    fresh_approval: bool = False,
    application_approval_root: Path | str | None = None,
    evaluation_isolation: bool = False,
) -> tuple[AgentRuntime, EventLogger | None]:
    workspace = Workspace(workspace_root or Path.cwd())
    scientific = (
        scientific_state if scientific_state is not None else ScientificState()
    )
    resolved_approval_root = (
        workspace.resolve(str(application_approval_root), must_exist=False)
        if application_approval_root is not None
        else None
    )
    registry = create_default_registry(
        scientific,
        workspace,
        application_approval_root=resolved_approval_root,
        evaluation_isolation=evaluation_isolation,
    )
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
    logger = (
        EventLogger(session_dir or default_sessions_dir(), session_id=session_id)
        if log_events
        else None
    )
    if logger is not None:
        sinks.append(logger.log)

    policy = SwitchablePermissionPolicy(
        policy,
        settings=None if fresh_approval else ApprovalSettings(workspace.root),
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
        fresh_approval=fresh_approval,
        application_approval_root=resolved_approval_root,
    )
    return runtime, logger


async def run_goal(
    console: Console,
    runtime: AgentRuntime,
    goal: str,
    *,
    session_dir: Path | str | None = None,
) -> None:
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
        if session_dir is not None:
            from photomatagent.sessions.store import (
                save_session_snapshot,
                snapshot_path,
            )

            save_session_snapshot(
                session_dir,
                conversation=runtime.conversation_state,
                scientific=runtime.scientific_state,
                engine=runtime.context_engine.snapshot(),
            )
            console.print(f"[dim]session state saved: {snapshot_path(session_dir)}[/]")


async def run_interactive_chat(
    console: Console,
    runtime: AgentRuntime,
    session: PromptSession,
    *,
    logger: EventLogger | None = None,
) -> None:
    from photomatagent.cli.commands import ChatCommandRouter

    commands = ChatCommandRouter(
        console,
        runtime,
        runtime.workspace,
        logger=logger,
        sessions_dir=(logger.session_dir.parent if logger is not None else None),
    )
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
            await run_goal(
                console,
                runtime,
                goal,
                session_dir=logger.session_dir if logger is not None else None,
            )


def _load_resume_snapshot(
    resume: str,
    console: Console,
    *,
    sessions_dir: Path | str | None = None,
) -> tuple[Path, SessionSnapshot]:
    """Resolve a resume target and load its snapshot."""
    from photomatagent.observability.trace import resolve_session_path
    from photomatagent.sessions.store import load_session_snapshot, session_is_resumable

    session_dir = resolve_session_path(resume, sessions_dir)
    if not session_is_resumable(session_dir):
        console.print(
            f"[red]session {session_dir.name} has no resumable state "
            "(session_state.json). It can only be replayed.[/]"
        )
        raise FileNotFoundError(f"session snapshot not found in {session_dir}")
    snapshot = load_session_snapshot(session_dir)
    console.print(
        f"[dim]已加载历史 session {session_dir.name}："
        f"{len(snapshot.conversation.messages)} 条消息，"
        f"{len(snapshot.scientific.evidence)} 条证据。可直接继续追问。[/]"
    )
    return session_dir, snapshot


async def run_chat(
    *,
    provider: str = "fake",
    model: str | None = None,
    workspace_root: Path | str | None = None,
    approval: ApprovalMode = "ask",
    max_iterations: int = 10000,
    log_events: bool = True,
    goal: str | None = None,
    resume: str | None = None,
    sessions_dir: Path | str | None = None,
) -> None:
    console = Console()
    prompt_session = make_prompt_session() if goal is None or approval == "ask" else None
    resume_session_dir: Path | None = None
    if resume is not None:
        resume_session_dir, snapshot = _load_resume_snapshot(
            resume, console, sessions_dir=sessions_dir
        )
    sessions_base = resume_session_dir.parent if resume_session_dir is not None else None
    resumed_session_id = resume_session_dir.name if resume_session_dir is not None else None
    runtime, logger = build_runtime(
        provider=provider,
        model=model,
        workspace_root=workspace_root,
        approval=approval,
        max_iterations=max_iterations,
        prompt_session=prompt_session,
        session_dir=sessions_base or sessions_dir,
        session_id=resumed_session_id,
        log_events=log_events,
    )
    if resume is not None:
        assert resume_session_dir is not None
        runtime.restore_session(snapshot)
    if goal is not None:
        await run_goal(
            console, runtime, goal, session_dir=logger.session_dir if logger else None
        )
    else:
        assert prompt_session is not None
        await run_interactive_chat(
            console,
            runtime,
            prompt_session,
            logger=logger,
        )
    if logger is not None:
        console.print(f"[dim]events logged: {logger.events_path}[/]")
