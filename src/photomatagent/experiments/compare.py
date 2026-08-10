"""Metric-only experiment comparison without qualitative ranking."""

from __future__ import annotations

from pydantic import BaseModel

from photomatagent.experiments.models import ExperimentSummary


class ComparisonRow(BaseModel):
    metric: str
    a: float | int | None
    b: float | int | None
    delta: float | int | None


def compare_summaries(
    a: ExperimentSummary, b: ExperimentSummary
) -> list[ComparisonRow]:
    mismatches = _controlled_comparison_mismatches(a, b)
    if mismatches:
        raise ValueError(
            "experiments are not a controlled comparison; mismatched: "
            + ", ".join(mismatches)
        )
    metrics = [
        ("Expectation pass rate", a.expectation_pass_rate, b.expectation_pass_rate),
        ("Avg iterations", a.average_iterations, b.average_iterations),
        ("Avg model calls", a.average_model_calls, b.average_model_calls),
        ("Avg tool calls", a.average_tool_calls, b.average_tool_calls),
        ("Avg tool failures", a.average_tool_failures, b.average_tool_failures),
        ("Tool failure rate", a.tool_failure_rate, b.tool_failure_rate),
        (
            "Avg repeated calls",
            a.average_repeated_tool_calls,
            b.average_repeated_tool_calls,
        ),
        ("Input tokens", a.input_tokens, b.input_tokens),
        ("Output tokens", a.output_tokens, b.output_tokens),
        ("Duration (s)", a.duration_seconds, b.duration_seconds),
        ("Model latency (s)", a.model_latency_seconds, b.model_latency_seconds),
        (
            "Estimated tool-schema tokens / call",
            a.estimated_tool_schema_tokens_per_call,
            b.estimated_tool_schema_tokens_per_call,
        ),
        ("tool_search calls", a.tool_search_calls, b.tool_search_calls),
        ("tool_describe calls", a.tool_describe_calls, b.tool_describe_calls),
        ("tool_call bridge calls", a.tool_call_bridge_calls, b.tool_call_bridge_calls),
        (
            "Peak working context (estimated tokens)",
            a.peak_working_context_tokens,
            b.peak_working_context_tokens,
        ),
        ("Pruned tool results", a.pruned_tool_results, b.pruned_tool_results),
        ("Compaction count", a.compaction_count, b.compaction_count),
        ("Compaction failures", a.compaction_failures, b.compaction_failures),
    ]
    return [
        ComparisonRow(
            metric=name,
            a=value_a,
            b=value_b,
            delta=(value_b - value_a if value_a is not None and value_b is not None else None),
        )
        for name, value_a, value_b in metrics
    ]


def _controlled_comparison_mismatches(
    a: ExperimentSummary, b: ExperimentSummary
) -> list[str]:
    config_a = a.configuration
    config_b = b.configuration
    checks = {
        "provider": (config_a.provider, config_b.provider),
        "model": (config_a.model, config_b.model),
        "system_prompt": (config_a.system_prompt, config_b.system_prompt),
        "stop_policy": (config_a.stop_policy, config_b.stop_policy),
        "context_builder": (config_a.context_builder, config_b.context_builder),
        "task_set": (config_a.task_set_sha256, config_b.task_set_sha256),
        "skill_index": (config_a.skill_index_sha256, config_b.skill_index_sha256),
        "tasks_total": (a.tasks_total, b.tasks_total),
    }
    return [name for name, values in checks.items() if values[0] != values[1]]
