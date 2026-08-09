"""photomatagent chat: interactive event consumer for the Agent Runtime."""

from __future__ import annotations

from typing import Literal

from prompt_toolkit.shortcuts import PromptSession
from rich.console import Console

from photomatagent.cli.prompt import make_approval_handler, make_prompt_session
from photomatagent.cli.render import ChatRenderer
from photomatagent.logging.event_logger import EventLogger, default_sessions_dir
from photomatagent.models.fake import FakeModelProvider
from photomatagent.runtime.budget import BudgetState
from photomatagent.runtime.loop import AgentRuntime, EventSink
from photomatagent.runtime.permissions import AllowAllPolicy, AskPolicy, DenyAllPolicy, PermissionPolicy
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.factory import create_default_registry

ApprovalMode = Literal["ask", "auto", "deny"]


def build_runtime(
    *,
    approval: ApprovalMode = "ask",
    max_iterations: int = 10,
    session_dir=None,
    log_events: bool = True,
) -> tuple[AgentRuntime, EventLogger | None]:
    """Assemble a runtime wired for the CLI (fake model, default tools)."""
    scientific = ScientificState()
    registry = create_default_registry(scientific)
    model = FakeModelProvider(auto=True)
    budget = BudgetState(max_iterations=max_iterations)

    policy: PermissionPolicy
    approval_handler = None
    if approval == "ask":
        policy = AskPolicy()
        approval_handler = make_approval_handler(make_prompt_session())
    elif approval == "auto":
        policy = AllowAllPolicy()
    else:
        policy = DenyAllPolicy()

    sinks: list[EventSink] = []
    logger = None
    if log_events:
        logger = EventLogger(session_dir or default_sessions_dir())
        sinks.append(logger.log)

    runtime = AgentRuntime(
        model=model,
        tools=registry,
        scientific_state=scientific,
        permission_policy=policy,
        budget=budget,
        approval_handler=approval_handler,
        event_sinks=sinks,
    )
    return runtime, logger


async def run_goal(console: Console, runtime: AgentRuntime, goal: str) -> None:
    renderer = ChatRenderer(console)
    async for event in runtime.run(goal):
        renderer.handle(event)
    renderer.flush_agent_text()


async def run_interactive_chat(console: Console, runtime: AgentRuntime) -> None:
    session = make_prompt_session()
    console.print("[dim]Type your research goal. [/][bold]/exit[/][dim] to quit.[/]")
    while True:
        try:
            goal = await session.prompt_async("❯ ")
        except (EOFError, KeyboardInterrupt):
            break
        if goal.strip().lower() in {"/exit", "/quit", "exit", "quit"}:
            break
        if not goal.strip():
            continue
        await run_goal(console, runtime, goal)


async def run_chat(
    *,
    approval: ApprovalMode = "ask",
    max_iterations: int = 10,
    log_events: bool = True,
    goal: str | None = None,
) -> None:
    console = Console()
    runtime, logger = build_runtime(
        approval=approval, max_iterations=max_iterations, log_events=log_events
    )
    if goal is not None:
        await run_goal(console, runtime, goal)
    else:
        await run_interactive_chat(console, runtime)
    if logger is not None:
        console.print(f"[dim]events logged: {logger.events_path}[/]")
