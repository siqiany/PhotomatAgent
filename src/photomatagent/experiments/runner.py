"""Sequential experiment runner over independent AgentRuntime sessions."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal
import json

from photomatagent.experiments.evaluator import evaluate_expectations
from photomatagent.experiments.models import (
    ConfigurationSnapshot,
    ExperimentConfig,
    ExperimentResult,
    ExperimentSummary,
    ExperimentTaskRun,
    ExperimentTask,
    ScientificLoopVariant,
)
from photomatagent.experiments.storage import new_experiment_id
from photomatagent.logging.event_logger import EventLogger, default_sessions_dir
from photomatagent.models.factory import create_provider
from photomatagent.models.types import AssistantMessage
from photomatagent.observability.analyzer import analyze_trace
from photomatagent.observability.trace import load_trace
from photomatagent.runtime.budget import BudgetState
from photomatagent.runtime.context import (
    SYSTEM_PROMPT,
    ContextBuilder,
    format_skill_index,
)
from photomatagent.runtime.context_engine import ContextEngineConfig
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import AllowAllPolicy, DenyAllPolicy
from photomatagent.runtime.stop_policy import StopPolicy
from photomatagent.scientific.state import ScientificState
from photomatagent.skills.loader import SkillLoader
from photomatagent.tools.factory import create_default_registry
from photomatagent.tools.surface import ToolSurfaceConfig, ToolSurfacePlanner
from photomatagent.workspace import Workspace


async def run_experiment(
    config: ExperimentConfig,
    *,
    provider: str,
    model: str,
    workspace_root: Path | str | None = None,
    sessions_dir: Path | str | None = None,
) -> ExperimentResult:
    """Run tasks in config order; each task receives fresh state and a trace."""
    workspace = Workspace(workspace_root or Path.cwd())
    base_sessions = Path(sessions_dir) if sessions_dir is not None else default_sessions_dir()
    snapshot = configuration_snapshot(
        provider=provider,
        model=model,
        max_iterations=config.variant.max_iterations,
        tool_surface=config.variant.tool_surface,
        tasks=config.tasks,
    )
    runs: list[ExperimentTaskRun] = []
    for task in config.tasks:
        scientific = ScientificState()
        logger = EventLogger(base_sessions)
        surface_config = ToolSurfaceConfig(mode=config.variant.tool_surface)
        registry = create_default_registry(
            scientific, workspace, surface_config=surface_config
        )
        runtime = AgentRuntime(
            model=create_provider(provider, model),
            tools=registry,
            workspace=workspace,
            scientific_state=scientific,
            context_builder=ContextBuilder(),
            tool_surface_planner=ToolSurfacePlanner(registry, surface_config),
            permission_policy=(
                AllowAllPolicy()
                if config.variant.approval == "auto"
                else DenyAllPolicy()
            ),
            stop_policy=StopPolicy(),
            budget=BudgetState(max_iterations=config.variant.max_iterations),
            event_sinks=[logger.log],
            session_id=logger.session_id,
        )
        runtime_error: str | None = None
        if config.loop is not None:
            answer, runtime_error = await _run_loop_task(
                runtime=runtime,
                logger=logger,
                loop=config.loop,
                prompt=task.prompt,
            )
        else:
            try:
                async for _ in runtime.run(task.prompt):
                    pass
            except Exception as exc:
                runtime_error = f"{type(exc).__name__}: {exc}"
            answer = _final_answer(runtime)
        trace = load_trace(logger.session_dir)
        session_summary = analyze_trace(trace)
        evaluation = evaluate_expectations(
            task.expect, answer=answer, summary=session_summary
        )
        runs.append(
            ExperimentTaskRun(
                task_id=task.id,
                session_id=logger.session_id,
                runtime_status=(
                    "COMPLETED" if session_summary.runtime_completed else "FAILED"
                ),
                evaluation=evaluation,
                answer=answer,
                error=runtime_error,
                summary=session_summary,
            )
        )
    experiment_id = new_experiment_id(config.name)
    experiment_summary = summarize_experiment(
        experiment_id, config, snapshot, runs
    )
    return ExperimentResult(
        experiment_id=experiment_id,
        config=config,
        summary=experiment_summary,
        runs=runs,
    )


def configuration_snapshot(
    *,
    provider: str,
    model: str,
    max_iterations: int,
    tool_surface: Literal["progressive", "eager"] = "progressive",
    tasks: list[ExperimentTask] | None = None,
) -> ConfigurationSnapshot:
    serialized_tasks = json.dumps(
        [task.model_dump(mode="json") for task in tasks or []],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    skill_index = format_skill_index(SkillLoader())
    return ConfigurationSnapshot(
        provider=provider,
        model=model,
        system_prompt={
            "identifier": "photomatagent.runtime.context.SYSTEM_PROMPT",
            "sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        },
        stop_policy={
            "identifier": f"{StopPolicy.__module__}.{StopPolicy.__qualname__}",
            "config": {"max_iterations": max_iterations},
        },
        context_builder={
            "identifier": f"{ContextBuilder.__module__}.{ContextBuilder.__qualname__}",
            "config": {"working_ledger": "derived-bounded"},
        },
        context_engine={
            "identifier": "photomatagent.runtime.context_engine.ContextEngine",
            "config": ContextEngineConfig().model_dump(),
        },
        tool_surface={
            "identifier": f"{ToolSurfacePlanner.__module__}.{ToolSurfacePlanner.__qualname__}",
            "config": ToolSurfaceConfig(mode=tool_surface).model_dump(),
        },
        task_set_sha256=hashlib.sha256(serialized_tasks.encode("utf-8")).hexdigest(),
        skill_index_sha256=hashlib.sha256(skill_index.encode("utf-8")).hexdigest(),
    )


def summarize_experiment(
    experiment_id: str,
    config: ExperimentConfig,
    snapshot: ConfigurationSnapshot,
    runs: list[ExperimentTaskRun],
) -> ExperimentSummary:
    count = len(runs)
    passed = sum(run.evaluation.status == "PASS" for run in runs)
    failed = sum(run.evaluation.status == "FAIL" for run in runs)
    unevaluated = sum(run.evaluation.status == "UNEVALUATED" for run in runs)
    evaluated = passed + failed
    summaries = [run.summary for run in runs]
    total_tool_calls = sum(summary.tool_calls for summary in summaries)
    total_tool_failures = sum(summary.tool_failures for summary in summaries)
    repeated = sum(summary.repeated_tool_calls for summary in summaries)
    model_calls = sum(summary.model_calls for summary in summaries)
    estimated_schema_total = sum(
        (summary.model_visible_schema_estimated_tokens_per_call or 0)
        * summary.model_calls
        for summary in summaries
    )
    return ExperimentSummary(
        experiment_id=experiment_id,
        name=config.name,
        configuration=snapshot,
        tasks_total=count,
        tasks_completed=sum(run.runtime_status == "COMPLETED" for run in runs),
        expectations_passed=passed,
        expectations_failed=failed,
        tasks_unevaluated=unevaluated,
        expectation_pass_rate=(passed / evaluated if evaluated else None),
        average_iterations=_average([summary.iterations for summary in summaries]),
        average_model_calls=_average([summary.model_calls for summary in summaries]),
        average_tool_calls=_average([summary.tool_calls for summary in summaries]),
        average_tool_failures=_average(
            [summary.tool_failures for summary in summaries]
        ),
        tool_failure_rate=(
            total_tool_failures / total_tool_calls if total_tool_calls else 0.0
        ),
        repeated_tool_calls=repeated,
        average_repeated_tool_calls=_average(
            [summary.repeated_tool_calls for summary in summaries]
        ),
        input_tokens=_sum_optional([summary.input_tokens for summary in summaries]),
        output_tokens=_sum_optional([summary.output_tokens for summary in summaries]),
        duration_seconds=sum(summary.duration_seconds for summary in summaries),
        model_latency_seconds=sum(
            summary.model_latency_seconds for summary in summaries
        ),
        estimated_tool_schema_tokens_per_call=(
            estimated_schema_total / model_calls if model_calls else None
        ),
        tool_search_calls=sum(summary.tool_search_calls for summary in summaries),
        tool_describe_calls=sum(
            summary.tool_describe_calls for summary in summaries
        ),
        tool_call_bridge_calls=sum(
            summary.tool_call_bridge_calls for summary in summaries
        ),
        peak_working_context_tokens=max(
            (
                summary.peak_working_context_tokens
                for summary in summaries
                if summary.peak_working_context_tokens is not None
            ),
            default=None,
        ),
        pruned_tool_results=sum(summary.pruned_tool_results for summary in summaries),
        compaction_count=sum(summary.compaction_count for summary in summaries),
        compaction_failures=sum(summary.compaction_failures for summary in summaries),
    )


async def _run_loop_task(
    *,
    runtime: AgentRuntime,
    logger: EventLogger,
    loop: ScientificLoopVariant,
    prompt: str,
) -> tuple[str, str | None]:
    """Run one task through the Evidence-Guided Scientific Feedback Loop.

    The loop summary text becomes the task's ``answer`` so the standard
    expectation checks (answer_contains, tools_used, ...) keep working; the
    full event trajectory (inner runtime + outer loop) lands in the same
    JSONL session.
    """
    from photomatagent.scientific.loop.controller import (
        ScientificLoopConfig,
        ScientificLoopController,
    )

    error: str | None = None
    controller = ScientificLoopController(
        target=loop.target,
        runtime=runtime,
        config=ScientificLoopConfig(
            max_rounds=loop.max_rounds,
            max_candidates=loop.max_candidates,
            patience=loop.patience,
            min_confidence=loop.min_confidence,
            judge_min_quality=loop.judge_min_quality,
            require_judge=loop.require_judge,
        ),
        event_sinks=[logger.log],
        session_id=logger.session_id,
    )
    try:
        async for _ in controller.run(goal=prompt):
            pass
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return _loop_summary_text(controller.summary), error


def _loop_summary_text(summary) -> str:
    if summary is None:
        return "Scientific loop completed: INCONCLUSIVE (no summary)"
    lines = [
        f"Scientific loop completed: {summary.status}",
        f"Rounds: {summary.rounds}",
        f"Candidates evaluated: {summary.candidate_count}",
        f"Best candidate: {summary.best_candidate_id or '-'}",
        f"Score: {summary.best_score:.3f}",
    ]
    if summary.unresolved_violations:
        lines.append(
            "Unresolved: "
            + "; ".join(v.short() for v in summary.unresolved_violations)
        )
    if summary.unresolved_evidence_gaps:
        lines.append(
            "Evidence gaps: " + ", ".join(summary.unresolved_evidence_gaps)
        )
    return "\n".join(lines)


def _final_answer(runtime: AgentRuntime) -> str:
    assistants = [
        message
        for message in runtime.conversation_state.messages
        if isinstance(message, AssistantMessage) and message.text
    ]
    return assistants[-1].text if assistants else ""


def _average(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sum_optional(values: list[int | None]) -> int | None:
    known = [value for value in values if value is not None]
    return sum(known) if known else None
