"""Typer commands for persistent expert-feedback evolution tasks.

The commands in this module are deliberately a thin application layer.  They
compose the existing runtime and scientific-loop factories; model-requested
tool execution remains exclusively inside :class:`AgentRuntime`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, cast

import typer
from rich.console import Console
from rich.table import Table

from photomatagent.errors import ToolExecutionError
from photomatagent.config import LLMConfig
from photomatagent.logging.event_logger import EventLogger, default_sessions_dir
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.scientific.evolution.executor import (
    EpisodeExecutionResult,
    EventSink,
    ScientificEpisodeExecutor,
)
from photomatagent.scientific.evolution.models import (
    EpisodeRecord,
    EpisodeVersion,
    EvolutionTask,
)
from photomatagent.scientific.evolution.service import (
    EvolutionService,
    MutationResult,
)
from photomatagent.scientific.evolution.store import EvolutionStore
from photomatagent.scientific.loop import (
    ScientificJudge,
    ScientificLoopConfig,
    TargetSpec,
)
from photomatagent.workspace import Workspace

ApprovalMode = Literal["ask", "auto", "deny"]

evolve_app = typer.Typer(
    help=(
        "Run persistent expert-feedback evolution tasks. "
        "Planned entry points: feedback and iterate (not available yet)."
    ),
)
console = Console()


@evolve_app.command("start")
def evolve_start(
    goal: str | None = typer.Option(None, "--goal", "-g"),
    target_json: str | None = typer.Option(
        None,
        "--target-json",
        help="Explicit JSON TargetSpec; overrides --demo.",
    ),
    target_file: Path | None = typer.Option(
        None,
        "--target-file",
        help="Workspace-contained file containing a JSON TargetSpec.",
    ),
    demo: bool = typer.Option(
        False,
        "--demo",
        help="Use the built-in 8-14 um LWIR photodetector demo target.",
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="Override .env preference: fake | openai | anthropic",
    ),
    model: str | None = typer.Option(None, "--model"),
    workspace: Path = typer.Option(
        Path.cwd(), "--workspace", exists=True, file_okay=False
    ),
    approval: str = typer.Option("auto", "--approval", help="ask | auto | deny"),
    max_rounds: int = typer.Option(6, "--max-rounds", min=1),
    patience: int = typer.Option(3, "--patience", min=1),
    min_confidence: float = typer.Option(
        0.6, "--min-confidence", min=0.0, max=1.0
    ),
    judge_provider: str | None = typer.Option(
        None,
        "--judge-provider",
        help="Enable the advisory LLM Scientific Judge (fake | openai | anthropic).",
    ),
    judge_model: str | None = typer.Option(None, "--judge-model"),
    judge_min_quality: float = typer.Option(
        0.6, "--judge-min-quality", min=0.0, max=1.0
    ),
    require_judge: bool = typer.Option(
        False,
        "--require-judge",
        help="Block SUCCESS while the judge is unavailable or below quality.",
    ),
    log_events: bool = typer.Option(True, "--log-events/--no-log-events"),
) -> None:
    """Create, reserve, and execute the first persistent evolution episode."""

    if approval not in {"ask", "auto", "deny"}:
        console.print("[red]--approval must be ask | auto | deny[/]")
        raise typer.Exit(code=2)

    try:
        target = _resolve_target(
            workspace=workspace,
            goal=goal,
            target_json=target_json,
            target_file=target_file,
            demo=demo,
        )
        config = _resolve_provider_config(workspace, provider, model)
    except (OSError, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=2) from None

    boundary = Workspace(workspace)
    store = EvolutionStore(boundary)
    service = EvolutionService(store)
    created = service.create_task(goal=target.goal, target=target)
    task = created.entity
    reserved = service.reserve_episode(
        task.evolution_id,
        mode="NORMAL",
        provider=config.provider,
        model=config.model,
    )
    episode = reserved.entity

    try:
        # Local imports avoid making the app/evolve/loop factory graph cyclic.
        from photomatagent.cli.chat import build_runtime
        from photomatagent.cli.loop import _build_judge, _render_event, _render_summary

        runtime, logger = build_runtime(
            provider=config.provider,
            model=config.model,
            workspace_root=boundary.root,
            approval=cast(ApprovalMode, approval),
            max_iterations=10000,
            session_dir=boundary.root / default_sessions_dir(),
            log_events=log_events,
        )
        judge = _build_judge(judge_provider, judge_model, console)
        execution = asyncio.run(
            _execute_initial_episode(
                store=store,
                task=task,
                episode=episode,
                runtime=runtime,
                logger=logger,
                config=ScientificLoopConfig(
                    max_rounds=max_rounds,
                    patience=patience,
                    min_confidence=min_confidence,
                    judge_min_quality=judge_min_quality,
                    require_judge=require_judge,
                ),
                judge=judge,
                prior_mutations=(created, reserved),
                on_event=lambda event: _render_event(console, event),
            )
        )
    except Exception as exc:
        recovery_note = _fail_active_episode(
            service,
            task.evolution_id,
            episode.version,
            exc,
        )
        failed_task = service.get(task.evolution_id)
        console.print(f"[red]evolution start failed: {_bounded_error(exc)}[/]")
        if recovery_note is not None:
            console.print(f"[red]failure reconciliation also failed: {recovery_note}[/]")
        _render_task_details(failed_task)
        raise typer.Exit(code=1) from None

    _render_summary(console, execution.scientific_summary)
    completed_task = service.get(task.evolution_id)
    _render_start_result(completed_task, execution)
    if logger is not None:
        console.print(f"[dim]events logged: {logger.events_path}[/]")


@evolve_app.command("list")
def evolve_list(
    workspace: Path = typer.Option(
        Path.cwd(), "--workspace", exists=True, file_okay=False
    ),
) -> None:
    """List persisted evolution tasks without constructing a model provider."""

    tasks = EvolutionStore(Workspace(workspace)).list_tasks()
    for task in tasks:
        _render_task_details(task)
    if not tasks:
        console.print("[dim]No evolution tasks found.[/]")


@evolve_app.command("status")
def evolve_status(
    evolution_id: str = typer.Argument(..., help="Persistent evolution task ID"),
    workspace: Path = typer.Option(
        Path.cwd(), "--workspace", exists=True, file_okay=False
    ),
) -> None:
    """Show the exact persisted state and next command for one task."""

    task = _load_task(workspace, evolution_id)
    _render_task_details(task)


@evolve_app.command("history")
def evolve_history(
    evolution_id: str = typer.Argument(..., help="Persistent evolution task ID"),
    workspace: Path = typer.Option(
        Path.cwd(), "--workspace", exists=True, file_okay=False
    ),
) -> None:
    """Show persisted episode history without constructing a model provider."""

    boundary = Workspace(workspace)
    store = EvolutionStore(boundary)
    try:
        task = store.load_task(evolution_id)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="evolution_id") from exc

    _render_task_details(task)
    table = Table(
        "Version",
        "Episode status",
        "Parent",
        "Runtime session",
        "Primary result",
        "Error",
    )
    for index in range(1, len(task.episode_ids) + 1):
        version = f"v{index:03d}"
        episode = store.load_episode(task.evolution_id, version)
        table.add_row(
            episode.version,
            episode.status,
            episode.parent_version or "—",
            episode.runtime_session_id or "—",
            episode.artifact.path if episode.artifact is not None else "—",
            episode.error or "—",
        )
    console.print(table)
    for index in range(1, len(task.episode_ids) + 1):
        version = f"v{index:03d}"
        episode = store.load_episode(task.evolution_id, version)
        if episode.artifact is not None:
            console.print(
                f"Primary result [{episode.version}]: {episode.artifact.path}",
                soft_wrap=True,
            )


async def _execute_initial_episode(
    *,
    store: EvolutionStore,
    task: EvolutionTask,
    episode: EpisodeRecord,
    runtime: AgentRuntime,
    logger: EventLogger | None,
    config: ScientificLoopConfig,
    judge: ScientificJudge | None,
    prior_mutations: Iterable[MutationResult[object]],
    on_event: EventSink | None,
) -> EpisodeExecutionResult:
    if logger is not None:
        for mutation in prior_mutations:
            for event in mutation.events:
                await logger.log(event)
    return await ScientificEpisodeExecutor(store, event_logger=logger).execute(
        task=task,
        episode=episode,
        runtime=runtime,
        config=config,
        judge=judge,
        on_event=on_event,
    )


def _resolve_target(
    *,
    workspace: Path,
    goal: str | None,
    target_json: str | None,
    target_file: Path | None,
    demo: bool,
) -> TargetSpec:
    from photomatagent.cli.loop import resolve_loop_target

    if target_file is not None and target_json is not None:
        raise ValueError("pass only one of --target-json or --target-file")
    if target_file is not None:
        boundary = Workspace(workspace)
        try:
            path = boundary.resolve(str(target_file), must_exist=True)
        except (OSError, ValueError, ToolExecutionError) as exc:
            raise ValueError(f"invalid --target-file: {exc}") from exc
        if not path.is_file():
            raise ValueError("invalid --target-file: path is not a regular file")
        try:
            target_json = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"invalid --target-file: {exc}") from exc
    target = resolve_loop_target(goal=goal, target_json=target_json, demo=demo)
    if not target.constraints:
        raise ValueError(
            "target must include at least one machine-verifiable constraint"
        )
    return target


def _resolve_provider_config(
    workspace: Path,
    provider: str | None,
    model: str | None,
) -> LLMConfig:
    from photomatagent.cli.loop import _resolve_config
    from photomatagent.config import DotEnvConfig

    config, created = _resolve_config(DotEnvConfig(workspace), provider, model)
    if created:
        console.print(f"[dim]created configuration file: {DotEnvConfig(workspace).path}[/]")
    return config


def _fail_active_episode(
    service: EvolutionService,
    evolution_id: str,
    version: EpisodeVersion,
    error: Exception,
) -> str | None:
    try:
        task = service.get(evolution_id)
        episode = service.store.load_episode(evolution_id, version)
        if task.status == "RUNNING" and episode.status in {"RESERVED", "RUNNING"}:
            service.fail_episode(evolution_id, version, _bounded_error(error))
    except Exception as recovery_error:
        return _bounded_error(recovery_error)
    return None


def _bounded_error(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:1000]


def _load_task(workspace: Path, evolution_id: str) -> EvolutionTask:
    try:
        return EvolutionStore(Workspace(workspace)).load_task(evolution_id)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="evolution_id") from exc


def _render_start_result(
    task: EvolutionTask,
    execution: EpisodeExecutionResult,
) -> None:
    table = Table("Field", "Value")
    table.add_row("Evolution ID", task.evolution_id)
    table.add_row("Status", task.status)
    table.add_row("Episode", execution.episode.version)
    table.add_row("Runtime session", execution.runtime_session_id)
    table.add_row("Primary result", execution.artifact.path)
    console.print(table)
    console.print(f"Primary result: {execution.artifact.path}", soft_wrap=True)
    _print_next_command(task)


def _render_task_details(task: EvolutionTask) -> None:
    table = Table("Field", "Value")
    table.add_row("Evolution ID", task.evolution_id)
    table.add_row("Status", task.status)
    table.add_row("Current version", task.current_version or "—")
    table.add_row("Last successful version", task.last_completed_version or "—")
    table.add_row("Feedback records", str(len(task.feedback_ids)))
    table.add_row("Revision plans", str(len(task.revision_ids)))
    table.add_row(
        "Feedback / revisions",
        f"{len(task.feedback_ids)} / {len(task.revision_ids)}",
    )
    table.add_row("Next command", _next_command(task))
    console.print(table)
    _print_next_command(task)


def _print_next_command(task: EvolutionTask) -> None:
    command = _next_command(task)
    console.print(f"Next command: [bold]{command}[/]", soft_wrap=True)


def _next_action(task: EvolutionTask) -> str:
    return {
        "AWAITING_EXPERT_FEEDBACK": "feedback",
        "FEEDBACK_RECORDED": "compile",
        "REVISION_READY": "iterate",
        "ACCEPTED": "reopen",
        "STOPPED": "reopen",
        "BLOCKED": "reopen",
        "BUDGET_EXHAUSTED": "reopen",
    }.get(task.status, "status")


def _next_command(task: EvolutionTask) -> str:
    prefix = f"photomatagent evolve {_next_action(task)} {task.evolution_id}"
    if task.status in {"AWAITING_EXPERT_FEEDBACK", "FEEDBACK_RECORDED"}:
        version = task.last_completed_version or task.current_version
        if version is not None:
            return f"{prefix} --version {version}"
    return prefix


__all__ = ["evolve_app"]
