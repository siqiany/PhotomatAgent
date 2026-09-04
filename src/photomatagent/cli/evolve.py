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
from typing import Literal, Protocol, cast

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from photomatagent.cli.prompt import make_prompt_session
from photomatagent.config import LLMConfig
from photomatagent.errors import ToolExecutionError
from photomatagent.logging.event_logger import EventLogger, default_sessions_dir
from photomatagent.redaction import redact_secrets, redact_text
from photomatagent.runtime.events import RuntimeEvent
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.scientific.evolution.executor import (
    EpisodeExecutionResult,
    EventSink,
    ScientificEpisodeExecutor,
)
from photomatagent.scientific.evolution.evidence import (
    build_inherited_scientific_state,
)
from photomatagent.scientific.evolution.feedback import FeedbackCompiler
from photomatagent.scientific.evolution.models import (
    EpisodeRecord,
    EpisodeVersion,
    EvolutionTask,
    ExpertFeedbackDraft,
    ExpertFeedbackRecord,
    FeedbackCompilation,
    RevisionPlan,
    RubricFlags,
    RubricScores,
    StrategyVersion,
    new_feedback_id,
)
from photomatagent.scientific.evolution.revision import build_revision_plan
from photomatagent.scientific.evolution.rubric import (
    RUBRIC_ANCHORS,
    RUBRIC_DIMENSIONS,
    RUBRIC_VERSION,
    assess_hard_caps,
)
from photomatagent.scientific.evolution.service import (
    EvolutionServiceError,
    EvolutionService,
    MutationResult,
)
from photomatagent.scientific.evolution.strategy import FixedStrategySelector
from photomatagent.scientific.evolution.store import EvolutionStore
from photomatagent.scientific.loop import (
    ScientificJudge,
    ScientificLoopConfig,
    ScientificLoopSummary,
    TargetSpec,
)
from photomatagent.workspace import Workspace

ApprovalMode = Literal["ask", "auto", "deny"]
_RUBRIC_FIELDS = (
    "scientific_correctness",
    "evidence_sufficiency",
    "novelty",
    "actionability",
    "overall",
)
_FLAG_PROMPTS = (
    ("fabricated_source", "存在伪造来源"),
    ("conclusion_changing_error", "存在会改变结论的科学错误"),
    ("abstract_only_core_evidence", "核心结论只有摘要支持"),
    ("unsupported_novelty", "创新性缺少定义、基线或证据"),
    ("process_parameters_only", "工艺只有路线名称和少数参数"),
)
_MAX_RENDERED_COMPILATION_ITEMS = 20
_MAX_RENDERED_COMPILATION_WARNINGS = 20
_MAX_RENDERED_COMPILATION_TEXT = 500
_MAX_RENDERED_PLAN_ITEMS = 20
_MAX_RENDERED_PLAN_TEXT = 500


class PromptSessionLike(Protocol):
    async def prompt_async(self, message: str) -> str: ...


class FeedbackEntryCancelled(Exception):
    """Internal control signal for a feedback form cancelled without writes."""

evolve_app = typer.Typer(
    help="Run persistent expert-feedback evolution tasks.",
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


@evolve_app.command("feedback")
def evolve_feedback(
    evolution_id: str = typer.Argument(..., help="Persistent evolution task ID"),
    version: str | None = typer.Option(
        None,
        "--version",
        help="Completed episode version; defaults to the latest completed version.",
    ),
    feedback_file: Path | None = typer.Option(
        None,
        "--file",
        help="Workspace-contained strict ExpertFeedbackDraft JSON file.",
    ),
    workspace: Path = typer.Option(
        Path.cwd(), "--workspace", exists=True, file_okay=False
    ),
) -> None:
    """Record one confirmed expert review without constructing a runtime."""

    try:
        boundary = Workspace(workspace)
        service = EvolutionService(EvolutionStore(boundary))
        task = service.get(evolution_id)
        selected_version = version or task.last_completed_version
        if selected_version is None:
            raise ValueError("task has no completed episode available for feedback")
        draft: ExpertFeedbackDraft | None = None
        raw_input: str | None = None
        if feedback_file is not None:
            resolved = boundary.resolve(str(feedback_file), must_exist=True)
            if not resolved.is_file():
                raise ValueError("--file must name a regular file")
            raw_input = resolved.read_text(encoding="utf-8")
            try:
                draft = load_feedback_file(resolved)
            except ValueError as exc:
                raise ValueError(
                    "invalid --file: expected strict ExpertFeedbackDraft JSON"
                ) from exc
        record = asyncio.run(
            run_feedback_flow(
                session=make_prompt_session(),
                console=console,
                service=service,
                evolution_id=evolution_id,
                version=cast(EpisodeVersion, selected_version),
                draft=draft,
                raw_input=raw_input,
            )
        )
    except (OSError, ValueError, ToolExecutionError, EvolutionServiceError) as exc:
        console.print(f"[red]{redact_text(str(exc))}[/]")
        raise typer.Exit(code=2) from None

    if record is None:
        console.print("[dim]Expert feedback cancelled; no data was written.[/]")
        return
    console.print(
        f"[green]Recorded feedback {record.feedback_id} for "
        f"{record.evolution_id} {record.episode_version}.[/]"
    )
    _render_task_details(service.get(evolution_id), boundary.root)


@evolve_app.command("iterate")
def evolve_iterate(
    evolution_id: str = typer.Argument(..., help="Persistent evolution task ID"),
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
    judge_provider: str | None = typer.Option(None, "--judge-provider"),
    judge_model: str | None = typer.Option(None, "--judge-model"),
    judge_min_quality: float = typer.Option(
        0.6, "--judge-min-quality", min=0.0, max=1.0
    ),
    require_judge: bool = typer.Option(False, "--require-judge"),
    log_events: bool = typer.Option(True, "--log-events/--no-log-events"),
) -> None:
    """Run the next episode from an exact confirmed revision checkpoint."""

    if approval not in {"ask", "auto", "deny"}:
        console.print("[red]--approval must be ask | auto | deny[/]")
        raise typer.Exit(code=2)
    try:
        boundary = Workspace(workspace)
        store = EvolutionStore(boundary)
        service = EvolutionService(store)
        context = service.iteration_context(evolution_id)
        config = _resolve_provider_config(boundary.root, provider, model)
        reserved = service.reserve_episode(
            evolution_id,
            mode="CARRY_VERIFIED_EVIDENCE",
            provider=config.provider,
            model=config.model,
        )
        episode = reserved.entity
    except (OSError, ValueError, ToolExecutionError, EvolutionServiceError) as exc:
        console.print(f"[red]{_bounded_error(exc)}[/]")
        raise typer.Exit(code=2) from None

    try:
        from photomatagent.cli.chat import build_runtime
        from photomatagent.cli.loop import _build_judge, _render_event, _render_summary

        inherited, _decisions = build_inherited_scientific_state(
            context.previous_scientific_state,
            source_episode=context.source_episode.version,
            invalidated_evidence_ids=context.revision.invalidated_evidence_ids,
            subject=_target_subject(context.task.target),
        )
        runtime, logger = build_runtime(
            provider=config.provider,
            model=config.model,
            workspace_root=boundary.root,
            approval=cast(ApprovalMode, approval),
            max_iterations=10000,
            session_dir=boundary.root / default_sessions_dir(),
            log_events=log_events,
            scientific_state=inherited,
        )
        judge = _build_judge(judge_provider, judge_model, console)
        execution = asyncio.run(
            _execute_revised_episode(
                store=store,
                task=context.task,
                episode=episode,
                revision=context.revision,
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
                prior_mutations=(reserved,),
                on_event=lambda event: _render_event(
                    console,
                    _redacted_event(event),
                ),
            )
        )
    except KeyboardInterrupt as exc:
        _report_start_failure(
            service=service,
            task=context.task,
            episode=episode,
            error=exc,
            workspace=boundary.root,
            operation="iteration",
        )
        raise typer.Exit(code=130) from None
    except Exception as exc:
        _report_start_failure(
            service=service,
            task=context.task,
            episode=episode,
            error=exc,
            workspace=boundary.root,
            operation="iteration",
        )
        raise typer.Exit(code=1) from None

    _render_summary(console, _redacted_summary(execution.scientific_summary))
    completed_task = service.get(evolution_id)
    _render_start_result(completed_task, execution, boundary.root)
    if logger is not None:
        console.print(f"[dim]events logged: {logger.events_path}[/]")


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


@evolve_app.command("compile")
def evolve_compile(
    evolution_id: str = typer.Argument(..., help="Persistent evolution task ID"),
    version: str | None = typer.Option(
        None,
        "--version",
        help="Feedback episode version; defaults to the latest completed version.",
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="Compiler provider override: fake | openai | anthropic",
    ),
    model: str | None = typer.Option(None, "--model"),
    workspace: Path = typer.Option(
        Path.cwd(), "--workspace", exists=True, file_okay=False
    ),
) -> None:
    """Compile the active immutable review through an isolated tool-free model."""

    try:
        boundary = Workspace(workspace)
        service = EvolutionService(EvolutionStore(boundary))
        config = _resolve_provider_config(boundary.root, provider, model)
        compiler = _build_feedback_compiler(config)
        compilation = asyncio.run(
            run_compilation_flow(
                service=service,
                evolution_id=evolution_id,
                version=cast(EpisodeVersion | None, version),
                compiler=compiler,
            )
        )
    except (OSError, UnicodeError, ValueError, ToolExecutionError, EvolutionServiceError) as exc:
        console.print(f"[red]{_bounded_error(exc)}[/]")
        raise typer.Exit(code=2) from None

    _render_compilation(console, compilation)
    if compilation.status == "UNAVAILABLE":
        _render_task_details(service.get(evolution_id), boundary.root)
        raise typer.Exit(code=1)
    try:
        confirmed = asyncio.run(
            run_revision_confirmation_flow(
                session=make_prompt_session(),
                console=console,
                service=service,
                evolution_id=evolution_id,
                compilation=compilation,
            )
        )
    except (ValueError, EvolutionServiceError) as exc:
        console.print(f"[red]{_bounded_error(exc)}[/]")
        _render_task_details(service.get(evolution_id), boundary.root)
        raise typer.Exit(code=2) from None
    if confirmed is None:
        console.print("[dim]Revision plan rejected; no plan or strategy was written.[/]")
    else:
        console.print(f"[green]Confirmed revision plan {confirmed.revision_id}.[/]")
    _render_task_details(service.get(evolution_id), boundary.root)


async def run_compilation_flow(
    *,
    service: EvolutionService,
    evolution_id: str,
    version: EpisodeVersion | None,
    compiler: FeedbackCompiler,
) -> FeedbackCompilation:
    """Resolve, compile, and persist exactly the active feedback record."""

    task, episode, feedback = service.compilation_context(evolution_id, version)
    existing = service.available_compilation(evolution_id, feedback.feedback_id)
    if existing is not None:
        mutation = service.save_compilation(evolution_id, existing)
        await service.publish(mutation)
        return mutation.entity
    if episode.artifact is None:  # guarded by compilation_context
        raise ValueError("selected episode has no primary result artifact")
    path = service.store.workspace.resolve(episode.artifact.path, must_exist=True)
    with path.open("r", encoding="utf-8") as handle:
        result_text = handle.read(12_001)
    compilation = await compiler.compile(
        task=task,
        episode=episode,
        feedback=feedback,
        result_text=result_text,
    )
    mutation = service.save_compilation(evolution_id, compilation)
    await service.publish(mutation)
    return mutation.entity


async def run_revision_confirmation_flow(
    *,
    session: PromptSessionLike,
    console: Console,
    service: EvolutionService,
    evolution_id: str,
    compilation: FeedbackCompilation,
) -> RevisionPlan | None:
    """Build, preview, and explicitly confirm one deterministic revision plan."""

    task, episode, feedback = service.compilation_context(
        evolution_id,
        compilation.episode_version,
    )
    plan = build_revision_plan(
        feedback=feedback,
        compilation=compilation,
        target=episode.target_snapshot,
        previous_summary=episode.summary,
    )
    strategy = FixedStrategySelector().select(task, plan)
    _render_revision_plan(console, plan, strategy)
    if plan.has_blocking_ambiguity:
        raise EvolutionServiceError(
            "revision plan has blocking CRITICAL ambiguity; add an action or "
            "acceptance test before confirmation"
        )
    try:
        confirmation = await _prompt_value(
            session,
            _expert_prompt(
                evolution_id,
                plan.source_version,
                "confirm revision plan with exact input 'y'",
            ),
        )
    except FeedbackEntryCancelled:
        return None
    if confirmation.strip().lower() != "y":
        return None
    confirmed = plan.model_copy(update={"confirmed": True})
    mutation = service.confirm_revision(
        evolution_id,
        confirmed,
        strategy=strategy,
    )
    await service.publish(mutation)
    return mutation.entity


def _build_feedback_compiler(config: LLMConfig) -> FeedbackCompiler:
    from photomatagent.models.factory import create_provider

    return FeedbackCompiler(create_provider(config.provider, config.model))


def _render_compilation(output: Console, compilation: FeedbackCompilation) -> None:
    table = Table("Field", "Value")
    table.add_row("Compilation", compilation.compilation_id or "—")
    table.add_row("Status", compilation.status)
    table.add_row("Provider", compilation.provider or "—")
    table.add_row("Model", compilation.model or "—")
    table.add_row("Items", str(len(compilation.items)))
    if compilation.error:
        table.add_row("Error", redact_text(compilation.error))
    output.print(table)
    rendered_items = compilation.items[:_MAX_RENDERED_COMPILATION_ITEMS]
    for item in rendered_items:
        output.print(
            Text(
                f"[{item.severity}] {item.status} {item.category}: "
                f"{_bounded_compilation_text(item.problem)}"
            ),
            soft_wrap=True,
        )
    omitted_items = len(compilation.items) - len(rendered_items)
    if omitted_items:
        output.print(Text(f"… {omitted_items} additional items omitted", style="dim"))
    rendered_warnings = compilation.warnings[:_MAX_RENDERED_COMPILATION_WARNINGS]
    for warning in rendered_warnings:
        output.print(
            Text(
                f"Warning: {_bounded_compilation_text(warning)}",
                style="yellow",
            ),
            soft_wrap=True,
        )
    omitted_warnings = len(compilation.warnings) - len(rendered_warnings)
    if omitted_warnings:
        output.print(
            Text(f"… {omitted_warnings} additional warnings omitted", style="dim")
        )


def _render_revision_plan(
    output: Console,
    plan: RevisionPlan,
    strategy: StrategyVersion,
) -> None:
    summary = Table("Revision plan", "Value")
    summary.add_row("Revision ID", plan.revision_id)
    summary.add_row("Source episode", plan.source_version)
    summary.add_row("Feedback ID", plan.feedback_id)
    summary.add_row("Strategy ID", strategy.strategy_id)
    summary.add_row("Strategy", strategy.arm)
    summary.add_row("Strategy reason", _bounded_plan_text(strategy.reason))
    summary.add_row("Strategy SHA-256", strategy.strategy_sha256 or "—")
    summary.add_row("Selector", str(strategy.parameters.get("selector", "—")))
    summary.add_row(
        "Blocking ambiguity",
        "yes" if plan.has_blocking_ambiguity else "no",
    )
    output.print(summary)
    sections = (
        ("Contract changes", plan.contract_changes),
        ("Evidence requirements", plan.evidence_requirements),
        ("Output schema requirements", plan.output_schema_requirements),
        ("Preserve", plan.preserved_facts),
        ("Preserved evidence IDs", plan.preserved_evidence_ids),
        ("Prohibited repeats", plan.prohibited_repeats),
        ("Invalidated conclusions", plan.invalidated_conclusions),
        ("Invalidated evidence IDs", plan.invalidated_evidence_ids),
        ("Machine acceptance", plan.machine_acceptance_tests),
        ("Human acceptance", plan.human_acceptance_tests),
        ("Warnings", plan.warnings),
        ("Unresolved ambiguities", plan.unresolved_ambiguities),
    )
    for heading, values in sections:
        if not values:
            continue
        output.print(Text(f"{heading}:", style="bold"))
        rendered = values[:_MAX_RENDERED_PLAN_ITEMS]
        for value in rendered:
            output.print(Text(f"- {_bounded_plan_text(value)}"), soft_wrap=True)
        omitted = len(values) - len(rendered)
        if omitted:
            output.print(Text(f"… {omitted} additional items omitted", style="dim"))
    output.print("[bold]Only exact input 'y' confirms this revision plan.[/]")


def _bounded_plan_text(value: str) -> str:
    safe = redact_text(value)
    if len(safe) <= _MAX_RENDERED_PLAN_TEXT:
        return safe
    return safe[: _MAX_RENDERED_PLAN_TEXT - 1] + "…"


def _bounded_compilation_text(value: str) -> str:
    safe = redact_text(value)
    if len(safe) <= _MAX_RENDERED_COMPILATION_TEXT:
        return safe
    return safe[: _MAX_RENDERED_COMPILATION_TEXT - 1] + "…"


def load_feedback_file(path: Path) -> ExpertFeedbackDraft:
    """Load a JSON file through the strict draft schema.

    CLI callers must pass a path already resolved by their ``Workspace``
    boundary. Keeping schema parsing separate makes import behavior reusable
    without weakening the command's path containment check.
    """

    return ExpertFeedbackDraft.model_validate_json(path.read_text(encoding="utf-8"))


async def collect_expert_feedback(
    *,
    session: PromptSessionLike,
    console: Console,
    evolution_id: str,
    version: EpisodeVersion,
) -> ExpertFeedbackDraft:
    """Collect one complete ``expert-review-v1`` draft without persisting it."""

    console.print(
        f"[bold cyan]EXPERT FEEDBACK — {RUBRIC_VERSION} — "
        f"{evolution_id} — {version}[/]"
    )
    console.print("[dim]输入 /cancel 可随时取消；评论输入 /submit 后结束。[/]")
    scores: dict[str, int] = {}
    for field in _RUBRIC_FIELDS:
        label = RUBRIC_DIMENSIONS[field]
        console.print(f"[bold]{label}[/]")
        for number, anchor in enumerate(RUBRIC_ANCHORS[field], start=1):
            console.print(f"  {number}: {anchor}")
        scores[field] = await _prompt_score(
            session,
            console,
            _expert_prompt(evolution_id, version, field),
        )

    flags: dict[str, bool] = {}
    for field, label in _FLAG_PROMPTS:
        flags[field] = await _prompt_boolean(
            session,
            console,
            _expert_prompt(evolution_id, version, f"{field} [y/N] ({label})"),
        )
    fatal_issue = await _prompt_boolean(
        session,
        console,
        _expert_prompt(evolution_id, version, "fatal_issue [y/N] (存在致命问题)"),
    )
    comments = await _prompt_multiline(
        session,
        _expert_prompt(evolution_id, version, "comments"),
    )
    return ExpertFeedbackDraft(
        scores=RubricScores.model_validate(scores),
        flags=RubricFlags.model_validate(flags),
        fatal_issue=fatal_issue,
        comments=comments,
    )


async def run_feedback_flow(
    *,
    session: PromptSessionLike,
    console: Console,
    service: EvolutionService,
    evolution_id: str,
    version: EpisodeVersion,
    draft: ExpertFeedbackDraft | None = None,
    raw_input: str | None = None,
) -> ExpertFeedbackRecord | None:
    """Collect/import, summarize, confirm, and atomically attach one review."""

    try:
        task = service.get(evolution_id)
        episode = service.store.load_episode(evolution_id, version)
        if episode.artifact is None:
            raise ValueError("selected episode has no primary result artifact")
        selected = draft or await collect_expert_feedback(
            session=session,
            console=console,
            evolution_id=evolution_id,
            version=version,
        )
        assessment = assess_hard_caps(selected.scores, selected.flags)
        override_reason: str | None = None
        exceeds_cap = any(
            getattr(selected.scores, field) > getattr(assessment.suggested_scores, field)
            for field in _RUBRIC_FIELDS
        )
        if exceeds_cap:
            console.print("[yellow]Deterministic hard-cap suggestion:[/]")
            for reason in assessment.reasons:
                console.print(f"  - {reason}")
            override_reason = await _prompt_nonempty(
                session,
                console,
                _expert_prompt(evolution_id, version, "hard_cap_override_reason"),
            )

        _render_feedback_confirmation(
            console,
            task=task,
            version=version,
            result_sha256=episode.artifact.sha256,
            draft=selected,
        )
        confirmation = await _prompt_value(
            session,
            _expert_prompt(evolution_id, version, "confirm exact input 'y'"),
        )
        if confirmation.strip().lower() != "y":
            return None

        feedback_id = _feedback_id_for_retry(service, evolution_id, version)
        mutation = service.attach_feedback(
            evolution_id,
            version,
            feedback_id=feedback_id,
            draft=selected,
            result_sha256=episode.artifact.sha256,
            raw_input=raw_input or selected.model_dump_json(),
            hard_cap_override_reason=override_reason,
        )
        await service.publish(mutation)
        return mutation.entity
    except FeedbackEntryCancelled:
        return None


def _expert_prompt(evolution_id: str, version: str, field: str) -> str:
    return f"[EXPERT FEEDBACK | {evolution_id} | {version}] {field}> "


async def _prompt_value(session: PromptSessionLike, prompt: str) -> str:
    try:
        value = await session.prompt_async(prompt)
    except (EOFError, KeyboardInterrupt) as exc:
        raise FeedbackEntryCancelled from exc
    if value.strip().lower() == "/cancel":
        raise FeedbackEntryCancelled
    return value


async def _prompt_score(
    session: PromptSessionLike,
    console: Console,
    prompt: str,
) -> int:
    while True:
        value = (await _prompt_value(session, prompt)).strip()
        if value in {"1", "2", "3", "4", "5"}:
            return int(value)
        console.print("[red]请输入 1–5 的整数。[/]")


async def _prompt_boolean(
    session: PromptSessionLike,
    console: Console,
    prompt: str,
) -> bool:
    while True:
        value = (await _prompt_value(session, prompt)).strip().lower()
        if value in {"", "n", "no"}:
            return False
        if value in {"y", "yes"}:
            return True
        console.print("[red]请输入 y 或 n（默认 n）。[/]")


async def _prompt_multiline(session: PromptSessionLike, prompt: str) -> str:
    lines: list[str] = []
    while True:
        value = await _prompt_value(session, prompt)
        if value.strip().lower() == "/submit":
            return "\n".join(lines)
        lines.append(value)


async def _prompt_nonempty(
    session: PromptSessionLike,
    console: Console,
    prompt: str,
) -> str:
    while True:
        value = (await _prompt_value(session, prompt)).strip()
        if value:
            return value
        console.print("[red]覆盖确定性封顶时必须填写专家理由。[/]")


def _render_feedback_confirmation(
    output: Console,
    *,
    task: EvolutionTask,
    version: EpisodeVersion,
    result_sha256: str,
    draft: ExpertFeedbackDraft,
) -> None:
    table = Table("Feedback confirmation", "Value")
    table.add_row("Evolution task", task.evolution_id)
    table.add_row("Episode", version)
    table.add_row("Result SHA-256", result_sha256)
    for field in _RUBRIC_FIELDS:
        table.add_row(RUBRIC_DIMENSIONS[field], str(getattr(draft.scores, field)))
    active_flags = [label for field, label in _FLAG_PROMPTS if getattr(draft.flags, field)]
    table.add_row("Flags", ", ".join(active_flags) if active_flags else "none")
    table.add_row("Fatal issue", "yes" if draft.fatal_issue else "no")
    table.add_row("Comments length", str(len(draft.comments)))
    output.print(table)
    output.print(f"Result SHA-256: {result_sha256}", soft_wrap=True)
    output.print("[bold]Only exact input 'y' records this feedback.[/]")


def _feedback_id_for_retry(
    service: EvolutionService,
    evolution_id: str,
    version: EpisodeVersion,
) -> str:
    records = service.store.list_feedback(evolution_id)
    superseded = {
        record.supersedes_feedback_id
        for record in records
        if record.supersedes_feedback_id is not None
    }
    active = [
        record
        for record in records
        if record.episode_version == version and record.feedback_id not in superseded
    ]
    if len(active) > 1:
        raise ValueError(f"episode {version} has multiple active feedback records")
    return active[0].feedback_id if active else new_feedback_id()


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


async def _execute_revised_episode(
    *,
    store: EvolutionStore,
    task: EvolutionTask,
    episode: EpisodeRecord,
    revision: RevisionPlan,
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
        revision=revision,
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


def _target_subject(target: TargetSpec) -> str | None:
    """Return an explicit task subject without guessing from free-form prose."""

    for values in (target.metadata, target.operating_conditions):
        for key in ("subject", "formula", "material", "candidate_id"):
            value = values.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


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
    operation: str = "start",
) -> None:
    safe_error = _bounded_error(error)
    recovery_note = _fail_active_episode(
        service,
        task.evolution_id,
        episode.version,
        safe_error,
    )
    failed_task = service.get(task.evolution_id)
    console.print(f"[red]evolution {operation} failed: {safe_error}[/]")
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


__all__ = [
    "FeedbackEntryCancelled",
    "collect_expert_feedback",
    "evolve_app",
    "load_feedback_file",
    "run_compilation_flow",
    "run_feedback_flow",
    "run_revision_confirmation_flow",
]
