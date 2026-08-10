"""Deterministic, offline metrics and anomaly flags for execution traces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from photomatagent.observability.trace import AgentExecutionTrace
from photomatagent.runtime.events import (
    ContextCompactionCompleted,
    ContextCompactionFailed,
    ContextCompactionStarted,
    ContextPruneCompleted,
    LoopCompleted,
    LoopFailed,
    LoopIterationStarted,
    LoopStarted,
    ModelRequestStarted,
    ModelResponseCompleted,
    ProviderFailed,
    ScientificTraceMeta,
    ToolCallCompleted,
    ToolCompleted,
    ToolFailed,
    ToolPermissionDenied,
    ToolRequested,
    ToolStarted,
)

AnomalyCode = Literal[
    "REPEATED_ACTION",
    "TOOL_FAILURE_LOOP",
    "MAX_ITERATIONS_REACHED",
    "HIGH_TOOL_CHURN",
]


class AnalyzerConfig(BaseModel):
    """Thresholds affect diagnostics only; they never alter Runtime behavior."""

    repeated_action_threshold: int = Field(default=2, ge=2)
    tool_failure_loop_threshold: int = Field(default=2, ge=2)
    high_tool_churn_threshold: int = Field(default=20, ge=1)


class AnomalyFlag(BaseModel):
    code: AnomalyCode
    detail: str
    iteration: int | None = None


class SessionSummary(BaseModel):
    session_id: str
    path: Path
    provider: str = "unknown"
    model: str = "unknown"
    run_ids: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    event_count: int = 0
    iterations: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    unique_tools: int = 0
    tools_used: list[str] = Field(default_factory=list)
    tool_failures: int = 0
    permission_denials: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    duration_seconds: float = 0.0
    model_latency_seconds: float = 0.0
    tool_latency_seconds: float = 0.0
    stop_reason: str | None = None
    repeated_tool_calls: int = 0
    consecutive_repeat_count: int = 0
    tool_failure_rate: float = 0.0
    tools_per_iteration: float | None = None
    model_calls_per_completed_session: float | None = None
    runtime_completed: bool = False
    registered_tools: int = 0
    direct_tools: int = 0
    deferred_tools: int = 0
    hidden_tools: int = 0
    direct_schema_estimated_tokens: int | None = None
    manifest_estimated_tokens_per_call: float | None = None
    deferred_schemas_avoided_estimated_tokens_per_call: float | None = None
    cumulative_deferred_schemas_avoided_estimated_tokens: int = 0
    bridge_schema_estimated_tokens: int | None = None
    model_visible_schema_estimated_tokens_per_call: float | None = None
    estimated_prompt_tokens_per_call: float | None = None
    tool_search_calls: int = 0
    tool_describe_calls: int = 0
    tool_call_bridge_calls: int = 0
    peak_working_context_tokens: int | None = None
    last_working_context_tokens: int | None = None
    last_working_context_chars: int | None = None
    durable_transcript_chars: int = 0
    pruned_tool_results: int = 0
    compaction_count: int = 0
    compaction_failures: int = 0
    last_compaction_tokens_before: int | None = None
    last_compaction_tokens_after: int | None = None
    last_compaction_chars_before: int | None = None
    last_compaction_chars_after: int | None = None
    skills_loaded: list[str] = Field(default_factory=list)
    scientific_tools_used: list[str] = Field(default_factory=list)
    evidence_created: int = 0
    evidence_sources: list[str] = Field(default_factory=list)
    evidence_gaps_identified: list[str] = Field(default_factory=list)
    capability_escalations: list[str] = Field(default_factory=list)
    anomalies: list[AnomalyFlag] = Field(default_factory=list)


@dataclass(frozen=True)
class ToolAction:
    index: int
    iteration: int
    call_id: str
    name: str
    arguments: dict[str, object]
    status: str

    @property
    def normalized_arguments(self) -> str:
        return normalize_arguments(self.arguments)

    @property
    def signature(self) -> str:
        return f"{self.name}:{self.normalized_arguments}"


def normalize_arguments(arguments: dict[str, object]) -> str:
    """Return the deterministic identity representation for tool arguments."""
    return json.dumps(
        arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def extract_tool_actions(trace: AgentExecutionTrace) -> list[ToolAction]:
    terminals: dict[str, str] = {}
    for event in trace.events:
        if isinstance(event, ToolCompleted):
            terminals[event.tool_call_id] = "success"
        elif isinstance(event, ToolFailed):
            terminals[event.tool_call_id] = "failed"
        elif isinstance(event, ToolPermissionDenied):
            terminals[event.tool_call_id] = "permission_denied"

    candidates: dict[str, tuple[int, int, str, dict[str, object], int]] = {}
    priority = {ToolRequested: 3, ToolCallCompleted: 2, ToolStarted: 1}
    for index, event in enumerate(trace.events):
        if not isinstance(event, (ToolRequested, ToolCallCompleted, ToolStarted)):
            continue
        call_id = event.tool_call_id
        event_priority = priority[type(event)]
        arguments = (
            event.arguments
            if isinstance(event, (ToolRequested, ToolCallCompleted))
            else {}
        )
        existing = candidates.get(call_id)
        if existing is None or event_priority > existing[4]:
            candidates[call_id] = (
                index,
                event.iteration,
                event.tool_name,
                arguments,
                event_priority,
            )
    actions = [
        ToolAction(
            index=value[0],
            iteration=value[1],
            call_id=call_id,
            name=value[2],
            arguments=value[3],
            status=terminals.get(call_id, "requested"),
        )
        for call_id, value in candidates.items()
    ]
    return sorted(actions, key=lambda action: action.index)


def analyze_trace(
    trace: AgentExecutionTrace, config: AnalyzerConfig | None = None
) -> SessionSummary:
    config = config or AnalyzerConfig()
    starts = [event for event in trace.events if isinstance(event, LoopStarted)]
    provider = starts[0].provider if starts else "unknown"
    model = starts[0].model if starts else "unknown"
    iterations = sum(isinstance(event, LoopIterationStarted) for event in trace.events)
    model_calls = sum(
        isinstance(event, (ModelRequestStarted, ContextCompactionStarted))
        for event in trace.events
    )
    model_requests = [
        event for event in trace.events if isinstance(event, ModelRequestStarted)
    ]
    surface_requests = [
        event
        for event in model_requests
        if event.registered_tools > 0
        or event.estimated_schema_tokens > 0
        or event.manifest_chars > 0
    ]
    model_completions = [
        event for event in trace.events if isinstance(event, ModelResponseCompleted)
    ]
    actions = extract_tool_actions(trace)
    tool_failures = sum(action.status == "failed" for action in actions)
    # Count terminal denial events directly so legacy traces that did not log
    # ToolRequested are still represented accurately.
    permission_denials = sum(
        isinstance(event, ToolPermissionDenied) for event in trace.events
    )
    tools_used = sorted({action.name for action in actions})
    bridge_call_count = sum(
        isinstance(event, ToolRequested) and event.bridge_tool == "tool_call"
        for event in trace.events
    )
    prune_events = [
        event for event in trace.events if isinstance(event, ContextPruneCompleted)
    ]
    compactions = [
        event for event in trace.events if isinstance(event, ContextCompactionCompleted)
    ]
    compaction_failures = sum(
        isinstance(event, ContextCompactionFailed) for event in trace.events
    )
    working_tokens = [
        event.estimated_current_prompt_tokens
        for event in model_requests
        if event.estimated_current_prompt_tokens is not None
    ]
    last_request = model_requests[-1] if model_requests else None
    last_compaction = compactions[-1] if compactions else None

    signatures = [action.signature for action in actions]
    signature_counts: dict[str, int] = {}
    for signature in signatures:
        signature_counts[signature] = signature_counts.get(signature, 0) + 1
    repeated_tool_calls = sum(count - 1 for count in signature_counts.values())
    consecutive_repeat_count = sum(
        current == previous
        for previous, current in zip(signatures, signatures[1:], strict=False)
    )
    max_repeat_streak = _max_streak(signatures)

    completed = [event for event in trace.events if isinstance(event, LoopCompleted)]
    failed = [event for event in trace.events if isinstance(event, LoopFailed)]
    terminal_events = [
        event
        for event in trace.events
        if isinstance(event, (LoopCompleted, LoopFailed))
    ]
    duration_seconds = sum(event.duration_ms for event in terminal_events) / 1000
    if duration_seconds == 0 and trace.started_at and trace.ended_at:
        duration_seconds = (trace.ended_at - trace.started_at).total_seconds()
    stop_reason: str | None = None
    if terminal_events:
        last_terminal = terminal_events[-1]
        stop_reason = (
            last_terminal.reason
            if isinstance(last_terminal, LoopCompleted)
            else "loop_failed"
        )

    compaction_usage = [event.usage for event in compactions if event.usage]
    usage_known = any(
        (event.usage.get("total_tokens") is not None)
        or bool(event.usage.get("input_tokens"))
        or bool(event.usage.get("output_tokens"))
        for event in model_completions
    ) or bool(compaction_usage)
    input_tokens = (
        sum(int(event.usage.get("input_tokens") or 0) for event in model_completions)
        + sum(int(usage.get("input_tokens") or 0) for usage in compaction_usage)
        if usage_known
        else None
    )
    output_tokens = (
        sum(int(event.usage.get("output_tokens") or 0) for event in model_completions)
        + sum(int(usage.get("output_tokens") or 0) for usage in compaction_usage)
        if usage_known
        else None
    )
    total_tokens = None
    if usage_known:
        total_tokens = sum(_event_total_tokens(event) for event in model_completions)
        total_tokens += sum(
            int(usage.get("total_tokens") or 0)
            or int(usage.get("input_tokens") or 0)
            + int(usage.get("output_tokens") or 0)
            for usage in compaction_usage
        )

    anomalies: list[AnomalyFlag] = []
    if max_repeat_streak >= config.repeated_action_threshold:
        anomalies.append(
            AnomalyFlag(
                code="REPEATED_ACTION",
                detail=(
                    f"identical tool action repeated {max_repeat_streak} times "
                    "consecutively"
                ),
                iteration=_first_repeat_iteration(actions, config.repeated_action_threshold),
            )
        )
    failure_streak = _max_failure_streak(actions)
    if failure_streak >= config.tool_failure_loop_threshold:
        anomalies.append(
            AnomalyFlag(
                code="TOOL_FAILURE_LOOP",
                detail=(
                    f"same tool action failed {failure_streak} times consecutively"
                ),
            )
        )
    if stop_reason == "max_iterations":
        anomalies.append(
            AnomalyFlag(
                code="MAX_ITERATIONS_REACHED",
                detail="runtime stopped at its configured iteration limit",
            )
        )
    if len(actions) >= config.high_tool_churn_threshold:
        anomalies.append(
            AnomalyFlag(
                code="HIGH_TOOL_CHURN",
                detail=(
                    f"session requested {len(actions)} tools; threshold is "
                    f"{config.high_tool_churn_threshold}"
                ),
            )
        )

    run_ids = list(
        dict.fromkeys(event.run_id for event in trace.events if event.run_id)
    )
    meta_events = [
        event for event in trace.events if isinstance(event, ScientificTraceMeta)
    ]
    skills_loaded = list(
        dict.fromkeys(skill for event in meta_events for skill in event.skills_loaded)
    )
    scientific_tools_used = list(
        dict.fromkeys(
            tool for event in meta_events for tool in event.scientific_tools_used
        )
    )
    evidence_sources = list(
        dict.fromkeys(
            source for event in meta_events for source in event.evidence_sources
        )
    )
    evidence_gaps_identified = list(
        dict.fromkeys(
            gap for event in meta_events for gap in event.evidence_gaps_identified
        )
    )
    capability_escalations = list(
        dict.fromkeys(
            escalation
            for event in meta_events
            for escalation in event.capability_escalations
        )
    )
    return SessionSummary(
        session_id=trace.session_id,
        path=trace.session_dir,
        provider=provider,
        model=model,
        run_ids=run_ids,
        started_at=trace.started_at,
        ended_at=trace.ended_at,
        event_count=len(trace.events),
        iterations=iterations,
        model_calls=model_calls,
        tool_calls=len(actions),
        unique_tools=len(tools_used),
        tools_used=tools_used,
        tool_failures=tool_failures,
        permission_denials=permission_denials,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        duration_seconds=max(duration_seconds, 0.0),
        model_latency_seconds=(
            sum(event.duration_ms for event in model_completions)
            + sum(
                event.duration_ms
                for event in trace.events
                if isinstance(event, ProviderFailed)
            )
            + sum(event.duration_ms for event in compactions)
            + sum(
                event.duration_ms
                for event in trace.events
                if isinstance(event, ContextCompactionFailed)
            )
        )
        / 1000,
        tool_latency_seconds=(
            sum(
                event.duration_ms
                for event in trace.events
                if isinstance(event, (ToolCompleted, ToolFailed))
            )
            / 1000
        ),
        stop_reason=stop_reason,
        repeated_tool_calls=repeated_tool_calls,
        consecutive_repeat_count=consecutive_repeat_count,
        tool_failure_rate=(tool_failures / len(actions) if actions else 0.0),
        tools_per_iteration=(len(actions) / iterations if iterations else None),
        model_calls_per_completed_session=(
            model_calls / len(completed) if completed else None
        ),
        runtime_completed=bool(completed) and not failed,
        registered_tools=max(
            (event.registered_tools for event in model_requests), default=0
        ),
        direct_tools=max((event.direct_count for event in model_requests), default=0),
        deferred_tools=max(
            (event.deferred_count for event in model_requests), default=0
        ),
        hidden_tools=max((event.hidden_count for event in model_requests), default=0),
        direct_schema_estimated_tokens=_max_positive_optional(
            [event.estimated_direct_schema_tokens for event in surface_requests]
        ),
        manifest_estimated_tokens_per_call=_average_optional(
            [event.estimated_manifest_tokens for event in surface_requests]
        ),
        deferred_schemas_avoided_estimated_tokens_per_call=_average_optional(
            [event.estimated_avoided_tokens for event in surface_requests]
        ),
        cumulative_deferred_schemas_avoided_estimated_tokens=sum(
            event.estimated_avoided_tokens for event in model_requests
        ),
        bridge_schema_estimated_tokens=_max_positive_optional(
            [event.estimated_bridge_schema_tokens for event in surface_requests]
        ),
        model_visible_schema_estimated_tokens_per_call=_average_optional(
            [event.estimated_schema_tokens for event in surface_requests]
        ),
        estimated_prompt_tokens_per_call=_average_optional(
            [
                event.estimated_current_prompt_tokens
                for event in model_requests
                if event.estimated_current_prompt_tokens is not None
            ]
        ),
        tool_search_calls=sum(action.name == "tool_search" for action in actions),
        tool_describe_calls=sum(action.name == "tool_describe" for action in actions),
        tool_call_bridge_calls=bridge_call_count,
        peak_working_context_tokens=max(working_tokens, default=None),
        last_working_context_tokens=(working_tokens[-1] if working_tokens else None),
        last_working_context_chars=(
            last_request.working_context_chars if last_request else None
        ),
        durable_transcript_chars=(
            trace.events_path.stat().st_size if trace.events_path.exists() else 0
        ),
        pruned_tool_results=sum(event.tool_results_pruned for event in prune_events),
        compaction_count=len(compactions),
        compaction_failures=compaction_failures,
        last_compaction_tokens_before=(
            last_compaction.tokens_before if last_compaction else None
        ),
        last_compaction_tokens_after=(
            last_compaction.tokens_after if last_compaction else None
        ),
        last_compaction_chars_before=(
            last_compaction.chars_before if last_compaction else None
        ),
        last_compaction_chars_after=(
            last_compaction.chars_after if last_compaction else None
        ),
        skills_loaded=skills_loaded,
        scientific_tools_used=scientific_tools_used,
        evidence_created=sum(event.evidence_created for event in meta_events),
        evidence_sources=evidence_sources,
        evidence_gaps_identified=evidence_gaps_identified,
        capability_escalations=capability_escalations,
        anomalies=anomalies,
    )


def _max_streak(values: list[str]) -> int:
    maximum = 0
    current = 0
    previous: str | None = None
    for value in values:
        current = current + 1 if value == previous else 1
        maximum = max(maximum, current)
        previous = value
    return maximum


def _event_total_tokens(event: ModelResponseCompleted) -> int:
    reported = event.usage.get("total_tokens")
    if isinstance(reported, int):
        return reported
    return int(event.usage.get("input_tokens") or 0) + int(
        event.usage.get("output_tokens") or 0
    )


def _max_failure_streak(actions: list[ToolAction]) -> int:
    maximum = 0
    current = 0
    previous_signature: str | None = None
    for action in actions:
        if action.status == "failed" and action.signature == previous_signature:
            current += 1
        elif action.status == "failed":
            current = 1
        else:
            current = 0
        maximum = max(maximum, current)
        previous_signature = action.signature
    return maximum


def _first_repeat_iteration(actions: list[ToolAction], threshold: int) -> int | None:
    streak = 0
    previous: str | None = None
    for action in actions:
        streak = streak + 1 if action.signature == previous else 1
        if streak >= threshold:
            return action.iteration
        previous = action.signature
    return None


def _average_optional(values: list[int]) -> float | None:
    return sum(values) / len(values) if values else None


def _max_positive_optional(values: list[int]) -> int | None:
    maximum = max(values, default=0)
    return maximum if maximum > 0 else None
