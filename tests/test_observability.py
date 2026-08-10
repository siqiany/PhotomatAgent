from __future__ import annotations

import json
from pathlib import Path

import pytest

from photomatagent.logging.event_logger import EventLogger
from photomatagent.observability.analyzer import AnalyzerConfig, analyze_trace
from photomatagent.observability.replay import build_replay
from photomatagent.observability.trace import load_trace
from photomatagent.runtime.events import (
    LoopCompleted,
    LoopIterationStarted,
    LoopStarted,
    ModelRequestStarted,
    ModelResponseCompleted,
    TextDelta,
    ToolCompleted,
    ToolFailed,
    ToolRequested,
)


async def _write(logger: EventLogger, events: list) -> None:
    for event in events:
        event.session_id = logger.session_id
        event.run_id = "run-1"
        await logger.log(event)


@pytest.mark.asyncio
async def test_trace_metrics_normalize_repeated_arguments(tmp_path: Path) -> None:
    logger = EventLogger(tmp_path)
    events = [
        LoopStarted(goal="find foo", provider="fake", model="fake", workspace=str(tmp_path)),
        LoopIterationStarted(iteration=1),
        ModelRequestStarted(iteration=1, message_count=2, provider="fake", model="fake"),
        ModelResponseCompleted(
            iteration=1,
            provider="fake",
            model="fake",
            finish_reason="tool_calls",
            tool_call_count=2,
            duration_ms=5,
        ),
        ToolRequested(
            iteration=1,
            tool_call_id="c1",
            tool_name="grep",
            arguments={"path": "src", "pattern": "foo"},
        ),
        ToolCompleted(
            iteration=1,
            tool_call_id="c1",
            tool_name="grep",
            output="one",
            duration_ms=2,
        ),
        ToolRequested(
            iteration=1,
            tool_call_id="c2",
            tool_name="grep",
            arguments={"pattern": "foo", "path": "src"},
        ),
        ToolFailed(
            iteration=1,
            tool_call_id="c2",
            tool_name="grep",
            error="failed",
            duration_ms=3,
        ),
        LoopIterationStarted(iteration=2),
        ModelRequestStarted(iteration=2, message_count=5, provider="fake", model="fake"),
        ModelResponseCompleted(
            iteration=2,
            provider="fake",
            model="fake",
            finish_reason="tool_calls",
            tool_call_count=1,
            duration_ms=7,
        ),
        ToolRequested(
            iteration=2,
            tool_call_id="c3",
            tool_name="read",
            arguments={"path": "src/a.py"},
        ),
        ToolCompleted(
            iteration=2,
            tool_call_id="c3",
            tool_name="read",
            output="content",
            duration_ms=4,
        ),
        LoopCompleted(iterations=2, reason="max_iterations", duration_ms=30),
    ]
    await _write(logger, events)

    trace = load_trace(logger.session_dir)
    summary = analyze_trace(trace)

    assert len(trace.events) == len(events)
    assert summary.iterations == 2
    assert summary.model_calls == 2
    assert summary.tool_calls == 3
    assert summary.unique_tools == 2
    assert summary.tool_failures == 1
    assert summary.repeated_tool_calls == 1
    assert summary.consecutive_repeat_count == 1
    assert summary.tool_failure_rate == pytest.approx(1 / 3)
    assert summary.input_tokens is None
    assert {flag.code for flag in summary.anomalies} == {
        "REPEATED_ACTION",
        "MAX_ITERATIONS_REACHED",
    }


@pytest.mark.asyncio
async def test_failure_loop_and_high_churn_flags_are_configurable(tmp_path: Path) -> None:
    logger = EventLogger(tmp_path)
    events = [
        LoopStarted(goal="x", provider="fake", model="fake", workspace=str(tmp_path)),
        *[
            event
            for call_id in ("c1", "c2")
            for event in (
                ToolRequested(
                    iteration=1,
                    tool_call_id=call_id,
                    tool_name="grep",
                    arguments={"pattern": "x"},
                ),
                ToolFailed(
                    iteration=1,
                    tool_call_id=call_id,
                    tool_name="grep",
                    error="bad",
                ),
            )
        ],
        LoopCompleted(iterations=1, reason="final_response", duration_ms=1),
    ]
    await _write(logger, events)
    summary = analyze_trace(
        load_trace(logger.session_dir),
        AnalyzerConfig(high_tool_churn_threshold=2),
    )
    assert {flag.code for flag in summary.anomalies} == {
        "REPEATED_ACTION",
        "TOOL_FAILURE_LOOP",
        "HIGH_TOOL_CHURN",
    }


@pytest.mark.asyncio
async def test_replay_builds_deterministic_safe_intermediate_model(tmp_path: Path) -> None:
    logger = EventLogger(tmp_path)
    await _write(
        logger,
        [
            LoopStarted(goal="hello", provider="fake", model="fake", workspace=str(tmp_path)),
            LoopIterationStarted(iteration=1),
            ModelRequestStarted(iteration=1, message_count=2, provider="fake", model="fake"),
            TextDelta(iteration=1, text="visible "),
            TextDelta(iteration=1, text="answer"),
            ModelResponseCompleted(
                iteration=1,
                provider="fake",
                model="fake",
                finish_reason="stop",
                tool_call_count=0,
                duration_ms=1,
            ),
            LoopCompleted(iterations=1, reason="final_response", duration_ms=2),
        ],
    )
    replay = build_replay(load_trace(logger.session_dir))
    assert [item.kind for item in replay.items] == [
        "goal",
        "iteration",
        "model",
        "final_response",
        "stop",
    ]
    assert replay.items[3].content == "visible answer"


@pytest.mark.asyncio
async def test_redaction_covers_secret_fields_values_and_dotenv_text(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("CUSTOM_AUTH_TOKEN", "arbitrary-secret-value")
    logger = EventLogger(tmp_path)
    event = ToolRequested(
        iteration=1,
        tool_call_id="c1",
        tool_name="echo",
        arguments={
            "api_key": "field-secret",
            "text": "CUSTOM_API_KEY=unloaded-value\narbitrary-secret-value",
        },
    )
    await logger.log(event)
    content = logger.events_path.read_text(encoding="utf-8")
    assert "field-secret" not in content
    assert "unloaded-value" not in content
    assert "arbitrary-secret-value" not in content
    assert content.count("[REDACTED") >= 2


def test_legacy_trace_is_upgraded_during_typed_loading(tmp_path: Path) -> None:
    session = tmp_path / "legacy"
    session.mkdir()
    rows = [
        {"kind": "loop_started", "timestamp": "2026-01-01T00:00:00Z", "goal": "x"},
        {"kind": "loop_iteration_started", "timestamp": "2026-01-01T00:00:01Z", "iteration": 1},
        {
            "kind": "tool_requested",
            "timestamp": "2026-01-01T00:00:02Z",
            "tool_name": "echo",
            "arguments": {"text": "x"},
        },
        {
            "kind": "tool_started",
            "timestamp": "2026-01-01T00:00:03Z",
            "tool_name": "echo",
            "tool_call_id": "legacy-call",
        },
        {
            "kind": "tool_completed",
            "timestamp": "2026-01-01T00:00:04Z",
            "tool_name": "echo",
            "tool_call_id": "legacy-call",
            "output": "x",
        },
        {
            "kind": "loop_completed",
            "timestamp": "2026-01-01T00:00:05Z",
            "iterations": 1,
            "reason": "final_response",
        },
    ]
    (session / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    summary = analyze_trace(load_trace(session))
    assert summary.provider == "unknown"
    assert summary.tool_calls == 1
    assert summary.duration_seconds == 5
