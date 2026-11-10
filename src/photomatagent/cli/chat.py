"""Interactive CLI assembly and event consumption."""

from __future__ import annotations

import asyncio
import signal
from pathlib import Path
from typing import Literal, Any
from collections.abc import Coroutine

from prompt_toolkit.shortcuts import PromptSession
from rich.console import Console

from photomatagent.application.runtime_factory import (
    InteractionStrategy,
    RuntimeFactory,
)
from photomatagent.cli.prompt import make_approval_handler, make_prompt_session
from photomatagent.cli.render import ChatRenderer
from photomatagent.logging.event_logger import EventLogger
from photomatagent.runtime.context_engine import ContextEngineConfig
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import (
    AllowAllPolicy,
    DenyAllPolicy,
    PermissionPolicy,
    default_permission_policy,
)
from photomatagent.scientific.state import ScientificState
from photomatagent.sessions.store import SessionSnapshot
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
    context_engine_config: ContextEngineConfig | None = None,
    context_limit_tokens: int | None = None,
    compact_trigger_tokens: int | None = None,
) -> tuple[AgentRuntime, EventLogger | None]:
    policy: PermissionPolicy
    approval_handler = None
    if approval == "ask":
        policy = default_permission_policy()
        approval_handler = make_approval_handler(prompt_session or make_prompt_session())
    elif approval == "auto":
        policy = AllowAllPolicy()
    else:
        policy = DenyAllPolicy()
    return RuntimeFactory.build(
        provider=provider,
        model=model,
        workspace=workspace_root,
        session_dir=session_dir,
        session_id=session_id,
        budget=None,
        max_iterations=max_iterations,
        interaction=InteractionStrategy(
            permission_policy=policy,
            approval_handler=approval_handler,
            fresh_approval=fresh_approval,
            application_approval_root=application_approval_root,
        ),
        scientific_state=scientific_state,
        log_events=log_events,
        evaluation_isolation=evaluation_isolation,
        context_engine_config=context_engine_config,
        context_limit_tokens=context_limit_tokens,
        compact_trigger_tokens=compact_trigger_tokens,
    )


async def run_goal(
    console: Console,
    runtime: AgentRuntime,
    goal: str,
    *,
    session_dir: Path | str | None = None,
    intent: Literal["auto", "supplement", "replace"] = "auto",
    image_paths: tuple[str, ...] = (),
) -> None:
    renderer = ChatRenderer(
        console, pause_summary_provider=runtime.pause_summary
    )
    try:
        events = runtime.run(goal, intent=intent, image_paths=image_paths) if image_paths else runtime.run(goal, intent=intent)
        async for event in events:
            renderer.handle(event)
    except asyncio.CancelledError:
        console.print("[yellow]本轮已中止，已有历史会保存。输入纠偏意见继续；/goal 明确更换目标。[/]")
    except Exception as exc:
        if image_paths:
            console.print(f"图像输入失败（{getattr(exc, 'code', type(exc).__name__)}）", markup=False)
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
    resume_task: str | None = None,
) -> None:
    from photomatagent.cli.commands import ChatCommandRouter

    commands = ChatCommandRouter(
        console,
        runtime,
        runtime.workspace,
        logger=logger,
        sessions_dir=(logger.session_dir.parent if logger is not None else None),
        prompt_session=session,
    )
    console.print(
        "[dim]Type your research goal or [/][bold]/help[/][dim] for commands; "
        "[/][bold]/exit[/][dim] quits. /research 自动研究与专家返修；/make skill 提炼历史经验。\n"
        "普通后续输入补充原目标；/goal 更换目标。运行中 Ctrl+C 中止当前轮并返回输入。[/]"
    )
    try:
        if resume_task is not None:
            await commands.execute(f"/research --task {resume_task}")
        while True:
            try:
                prompt = ("专家评审 ❯ " if commands.research_mode.task_id else "研究问题 ❯ ") if commands.research_mode is not None else "❯ "
                goal = await session.prompt_async(prompt)
            except (EOFError, KeyboardInterrupt):
                break
            normalized = goal.strip().lower()
            if normalized in {"/exit", "/quit", "exit", "quit"}:
                break
            try:
                if goal.strip().startswith("/goal "):
                    if commands.research_mode is not None:
                        console.print("科研任务原目标保持冻结；请 /back 后使用 /goal 开始新目标。")
                        continue
                    text = goal.strip()[6:].strip()
                    intent: Literal["auto", "supplement", "replace"] = "replace"
                elif goal.strip().startswith("/"):
                    await _interruptible(commands.execute(goal.strip()))
                    continue
                else:
                    text = goal.strip()
                    intent = "supplement"
                if not text:
                    continue
                if commands.research_mode is not None:
                    await _interruptible(commands.research_mode.submit(text))
                else:
                    await _interruptible(run_goal(console, runtime, text, intent=intent,
                        session_dir=logger.session_dir if logger is not None else None))
            except asyncio.CancelledError:
                if commands.research_mode is not None:
                    await commands.research_mode.stop()
                else:
                    console.print("已中止当前操作，可继续输入。")
            except Exception as exc:
                console.print(f"操作未完成：{exc}", markup=False)
    finally:
        from photomatagent.application.runtime_factory import RuntimeResources
        await RuntimeResources((runtime.model_provider,)).aclose()


async def _interruptible(operation: Coroutine[Any, Any, Any]) -> Any:
    """SIGINT cancels only the foreground operation, keeping chat alive."""
    task = asyncio.create_task(operation)
    previous = signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGINT, lambda *_: task.cancel())
        return await task
    finally:
        signal.signal(signal.SIGINT, previous)


def _load_resume_snapshot(
    resume: str,
    console: Console,
    *,
    sessions_dir: Path | str | None = None,
) -> tuple[Path, SessionSnapshot]:
    """Resolve a resume target and load its snapshot."""
    from photomatagent.observability.trace import resolve_session_path
    from photomatagent.sessions.store import (
        load_session_snapshot,
        migration_diagnostic_codes,
        session_is_resumable,
    )

    session_dir = resolve_session_path(resume, sessions_dir)
    if not session_is_resumable(session_dir):
        console.print(
            f"[red]session {session_dir.name} has no resumable state "
            "(session_state.json). It can only be replayed.[/]"
        )
        raise FileNotFoundError(f"session snapshot not found in {session_dir}")
    snapshot = load_session_snapshot(session_dir)
    diagnostic_codes = migration_diagnostic_codes(snapshot)
    diagnostic_text = (
        f" 迁移诊断：{','.join(diagnostic_codes)}。"
        if diagnostic_codes
        else ""
    )
    console.print(
        f"[dim]已加载历史 session {session_dir.name}："
        f"{len(snapshot.conversation.messages)} 条消息，"
        f"{len(snapshot.scientific.evidence)} 条证据。可直接继续追问。"
        f"{diagnostic_text}[/]"
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
    context_limit_tokens: int | None = None,
    compact_trigger_tokens: int | None = None,
    context_engine_config: ContextEngineConfig | None = None,
) -> None:
    console = Console()
    workspace = Workspace(workspace_root or Path.cwd())
    configured_sessions_dir = (
        Path(sessions_dir) if sessions_dir is not None and Path(sessions_dir).is_absolute()
        else workspace.resolve(str(sessions_dir or ".photomatagent/sessions"), must_exist=False)
    )
    prompt_session = make_prompt_session() if goal is None or approval == "ask" else None
    resume_session_dir: Path | None = None
    resume_task: str | None = None
    if resume is not None:
        from photomatagent.observability.trace import resolve_session_path
        from photomatagent.sessions.binding import managed_task
        candidate = resolve_session_path(resume, configured_sessions_dir)
        resume_task = managed_task(candidate)
        if resume_task is None:
            resume_session_dir, snapshot = _load_resume_snapshot(
                resume, console, sessions_dir=configured_sessions_dir
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
        session_dir=sessions_base or configured_sessions_dir,
        session_id=resumed_session_id,
        log_events=log_events,
        context_engine_config=context_engine_config,
        context_limit_tokens=context_limit_tokens,
        compact_trigger_tokens=compact_trigger_tokens,
    )
    if resume_session_dir is not None:
        assert resume_session_dir is not None
        runtime.restore_session(snapshot)
    try:
        if goal is not None:
            if resume_task is not None:
                from photomatagent.cli.research_mode import ResearchMode
                mode = ResearchMode(console, prompt_session or make_prompt_session())
                await mode.select(["--task", resume_task])
                await mode.submit(goal)
            else:
                await run_goal(
                    console, runtime, goal, session_dir=logger.session_dir if logger else None
                )
        else:
            assert prompt_session is not None
            await run_interactive_chat(
                console, runtime, prompt_session, logger=logger, resume_task=resume_task,
            )
        if logger is not None:
            console.print(f"[dim]events logged: {logger.events_path}[/]")
    finally:
        from photomatagent.application.runtime_factory import RuntimeResources
        await RuntimeResources((runtime.model_provider, logger)).aclose()
