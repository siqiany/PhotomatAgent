"""Typer commands for persistent expert-feedback evolution tasks.

The commands in this module are deliberately a thin application layer.  They
compose the existing runtime and scientific-loop factories; model-requested
tool execution remains exclusively inside :class:`AgentRuntime`.
"""

from __future__ import annotations

import asyncio
import shlex
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, cast

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from photomatagent.errors import ToolExecutionError
from photomatagent.config import LLMConfig
from photomatagent.logging.event_logger import EventLogger, default_sessions_dir
from photomatagent.redaction import redact_secrets, redact_text
from photomatagent.runtime.events import RuntimeEvent
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
    EvolutionServiceError,
    EvolutionService,
    MutationResult,
)
from photomatagent.scientific.evolution.store import EvolutionStore
from photomatagent.scientific.loop import (
    ScientificJudge,
    ScientificLoopConfig,
    ScientificLoopSummary,
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
    resume: str | None = typer.Option(
        None,
        "--resume",
        help="Retry a persisted initial task from its stored goal and target.",
    ),
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
    if resume is not None and any(
        value is not None and value is not False
        for value in (goal, target_json, target_file, demo)
    ):
        console.print(
            "[red]--resume is mutually exclusive with --goal, --target-json, "
            "--target-file, and --demo[/]"
        )
        raise typer.Exit(code=2)

    try:
        boundary = Workspace(workspace)
        store = EvolutionStore(boundary)
        service = EvolutionService(store)
        prior_mutations: list[MutationResult[object]] = []
        if resume is None:
            target = _resolve_target(
                workspace=boundary.root,
                goal=goal,
                target_json=target_json,
                target_file=target_file,
                demo=demo,
            )
            config = _resolve_provider_config(boundary.root, provider, model)
            created = service.create_task(goal=target.goal, target=target)
            prior_mutations.append(created)
            task = created.entity
        else:
            reconciled = service.reconcile(resume)
            prior_mutations.append(reconciled)
            task = reconciled.entity
            if not task.target.constraints:
                raise ValueError(
                    "stored target must include at least one machine-verifiable constraint"
                )
            if task.status == "BLOCKED" and task.resume_status != "CREATED":
                raise ValueError(
                    "start --resume only retries an initial episode; use the "
                    "task's displayed next command for revised work"
                )
            if task.status not in {"CREATED", "BLOCKED"}:
                raise ValueError(
                    f"start --resume requires CREATED or an initial BLOCKED task; "
                    f"task is {task.status}"
                )
            config = _resolve_provider_config(boundary.root, provider, model)
            if task.status == "BLOCKED":
                reopened = service.reopen(task.evolution_id)
                prior_mutations.append(reopened)
                task = reopened.entity

        reserved = service.reserve_episode(
            task.evolution_id,
            mode="NORMAL",
            provider=config.provider,
            model=config.model,
        )
        prior_mutations.append(reserved)
        episode = reserved.entity
    except (OSError, ValueError) as exc:
        console.print(f"[red]{redact_text(str(exc))}[/]")
        raise typer.Exit(code=2) from None

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
                prior_mutations=prior_mutations,
                on_event=lambda event: _render_event(
                    console,
                    _redacted_event(event),
                ),
            )
        )
    except KeyboardInterrupt as exc:
        _report_start_failure(
            service=service,
            task=task,
            episode=episode,
            error=exc,
            workspace=boundary.root,
        )
        raise typer.Exit(code=130) from None
    except Exception as exc:
        _report_start_failure(
            service=service,
            task=task,
            episode=episode,
            error=exc,
            workspace=boundary.root,
        )
        raise typer.Exit(code=1) from None

    _render_summary(console, _redacted_summary(execution.scientific_summary))
    completed_task = service.get(task.evolution_id)
    _render_start_result(completed_task, execution, boundary.root)
    if logger is not None:
        console.print(f"[dim]events logged: {logger.events_path}[/]")


@evolve_app.command("list")
def evolve_list(
    workspace: Path = typer.Option(
        Path.cwd(), "--workspace", exists=True, file_okay=False
    ),
) -> None:
    """List persisted evolution tasks without constructing a model provider."""

    boundary = Workspace(workspace)
    tasks = EvolutionStore(boundary).list_tasks()
    for task in tasks:
        _render_task_details(task, boundary.root)
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

    boundary = Workspace(workspace)
    task = _load_task(boundary.root, evolution_id)
    _render_task_details(task, boundary.root)


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

    _render_task_details(task, boundary.root)
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
            redact_text(episode.error) if episode.error is not None else "—",
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
        if episode.error is not None:
            console.print(
                f"Error [{episode.version}]: {redact_text(episode.error)}",
                soft_wrap=True,
            )


@evolve_app.command("cancel")
def evolve_cancel(
    evolution_id: str = typer.Argument(..., help="Persistent evolution task ID"),
    workspace: Path = typer.Option(
        Path.cwd(), "--workspace", exists=True, file_okay=False
    ),
) -> None:
    """Cancel the current active episode or reconcile its terminal state."""

    boundary = Workspace(workspace)
    service = EvolutionService(EvolutionStore(boundary))
    try:
        cancelled = service.cancel(evolution_id)
    except (FileNotFoundError, ValueError, EvolutionServiceError) as exc:
        raise typer.BadParameter(
            redact_text(str(exc)),
            param_hint="evolution_id",
        ) from exc
    _render_task_details(cancelled.entity, boundary.root)


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
    safe_error: str,
) -> str | None:
    try:
        task = service.get(evolution_id)
        episode = service.store.load_episode(evolution_id, version)
        if task.status == "RUNNING" and episode.status in {"RESERVED", "RUNNING"}:
            service.fail_episode(evolution_id, version, safe_error)
    except Exception as recovery_error:
        return _bounded_error(recovery_error)
    return None


def _bounded_error(error: BaseException) -> str:
    return redact_text(f"{type(error).__name__}: {error}")[:1000]


def _redacted_event(event: RuntimeEvent) -> RuntimeEvent:
    payload = redact_secrets(event.model_dump(mode="python"))
    return type(event).model_validate(payload)


def _redacted_summary(summary: ScientificLoopSummary) -> ScientificLoopSummary:
    payload = redact_secrets(summary.model_dump(mode="python"))
    return ScientificLoopSummary.model_validate(payload)


def _report_start_failure(
    *,
    service: EvolutionService,
    task: EvolutionTask,
    episode: EpisodeRecord,
    error: BaseException,
    workspace: Path,
) -> None:
    safe_error = _bounded_error(error)
    recovery_note = _fail_active_episode(
        service,
        task.evolution_id,
        episode.version,
        safe_error,
    )
    failed_task = service.get(task.evolution_id)
    console.print(f"[red]evolution start failed: {safe_error}[/]")
    if recovery_note is not None:
        console.print(f"[red]failure reconciliation also failed: {recovery_note}[/]")
    _render_task_details(failed_task, workspace)


def _load_task(workspace: Path, evolution_id: str) -> EvolutionTask:
    try:
        return EvolutionStore(Workspace(workspace)).load_task(evolution_id)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="evolution_id") from exc


def _render_start_result(
    task: EvolutionTask,
    execution: EpisodeExecutionResult,
    workspace: Path,
) -> None:
    table = Table("Field", "Value")
    table.add_row("Evolution ID", task.evolution_id)
    table.add_row("Status", task.status)
    table.add_row("Episode", execution.episode.version)
    table.add_row("Runtime session", execution.runtime_session_id)
    table.add_row("Primary result", execution.artifact.path)
    console.print(table)
    console.print(f"Primary result: {execution.artifact.path}", soft_wrap=True)
    _print_next_command(task, workspace)


def _render_task_details(task: EvolutionTask, workspace: Path) -> None:
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
    table.add_row("Next command", Text(_next_command(task, workspace)))
    console.print(table)
    _print_next_command(task, workspace)


def _print_next_command(task: EvolutionTask, workspace: Path) -> None:
    command = _next_command(task, workspace)
    line = Text("Next command: ")
    line.append(command, style="bold")
    console.print(line, soft_wrap=True)


def _next_command(task: EvolutionTask, workspace: Path) -> str:
    if task.status == "CREATED" or (
        task.status == "BLOCKED" and task.resume_status == "CREATED"
    ):
        command = f"photomatagent evolve start --resume {task.evolution_id}"
    else:
        action = {
            "RUNNING": "cancel",
            "AWAITING_EXPERT_FEEDBACK": "feedback",
            "FEEDBACK_RECORDED": "compile",
            "REVISION_READY": "iterate",
            "ACCEPTED": "reopen",
            "STOPPED": "reopen",
            "BLOCKED": "reopen",
            "BUDGET_EXHAUSTED": "reopen",
        }[task.status]
        command = f"photomatagent evolve {action} {task.evolution_id}"
    if task.status in {"AWAITING_EXPERT_FEEDBACK", "FEEDBACK_RECORDED"}:
        version = task.last_completed_version or task.current_version
        if version is not None:
            command = f"{command} --version {version}"
    resolved_workspace = Path(workspace).resolve()
    return f"{command} --workspace {shlex.quote(str(resolved_workspace))}"


__all__ = ["evolve_app"]
