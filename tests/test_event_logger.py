from __future__ import annotations

import json

import pytest

from photomatagent.logging.event_logger import EventLogger
from photomatagent.logging.session_stats import read_session_stats
from photomatagent.runtime.events import (
    EvolutionTaskCreated,
    LoopCompleted,
    LoopIterationStarted,
    LoopStarted,
    ModelRequestStarted,
    ModelResponseCompleted,
    ProviderFailed,
    ToolFailed,
    ToolPermissionDenied,
    ToolStarted,
)


@pytest.mark.asyncio
async def test_events_roundtrip_via_jsonl(tmp_path):
    logger = EventLogger(tmp_path)
    events = [
        LoopStarted(goal="hello", provider="fake", model="fake", workspace=str(tmp_path)),
        LoopCompleted(iterations=1, reason="final_response", duration_ms=10),
    ]
    for event in events:
        event.session_id = logger.session_id
        await logger.log(event)
    restored = logger.read_events()
    assert [event.kind for event in restored] == ["loop_started", "loop_completed"]
    assert restored[0].session_id == logger.session_id


@pytest.mark.asyncio
async def test_evolution_event_is_redacted_in_jsonl(tmp_path):
    logger = EventLogger(tmp_path, session_id="evolution")
    await logger.log(EvolutionTaskCreated(evolution_id="evo_test", goal_summary="safe"))

    assert logger.read_events()[0].kind == "evolution_task_created"


@pytest.mark.asyncio
async def test_api_key_is_redacted(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-super-secret-value")
    logger = EventLogger(tmp_path)
    await logger.log(
        ProviderFailed(
            iteration=1,
            provider="openai",
            model="model",
            error="request used sk-test-super-secret-value",
        )
    )
    content = logger.events_path.read_text(encoding="utf-8")
    assert "sk-test-super-secret-value" not in content
    assert "[REDACTED]" in content


@pytest.mark.asyncio
async def test_session_stats_from_fixture_log(tmp_path):
    logger = EventLogger(tmp_path)
    events = [
        LoopStarted(goal="x", provider="openai", model="gpt-test", workspace=str(tmp_path)),
        LoopIterationStarted(iteration=1),
        ModelRequestStarted(iteration=1, message_count=2, provider="openai", model="gpt-test"),
        ModelResponseCompleted(
            iteration=1,
            provider="openai",
            model="gpt-test",
            finish_reason="tool_calls",
            tool_call_count=1,
            usage={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
            duration_ms=20,
        ),
        ToolStarted(iteration=1, tool_name="read", tool_call_id="c1"),
        ToolFailed(iteration=1, tool_name="read", tool_call_id="c1", error="nope"),
        ToolPermissionDenied(iteration=1, tool_name="bash", tool_call_id="c2", reason="deny"),
        LoopCompleted(iterations=1, reason="final_response", duration_ms=250),
    ]
    for event in events:
        event.session_id = logger.session_id
        await logger.log(event)
    stats = read_session_stats(logger.session_dir)
    assert stats.provider == "openai"
    assert stats.iterations == 1
    assert stats.model_calls == 1
    assert stats.tool_calls == 1
    assert stats.tool_failures == 1
    assert stats.permission_denials == 1
    assert stats.input_tokens == 10
    assert stats.output_tokens == 4
    assert stats.duration_seconds == 0.25
