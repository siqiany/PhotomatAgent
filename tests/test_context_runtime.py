from __future__ import annotations

from pathlib import Path

import pytest

from photomatagent.models.fake import FakeModelProvider
from photomatagent.models.types import (
    AssistantMessage,
    ModelUsage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from photomatagent.runtime.context import ContextBuilder
from photomatagent.runtime.context_engine import (
    CompactionState,
    ContextBuildResult,
    ContextEngine,
    ContextEngineConfig,
    ContextLimitExceeded,
    ContextSize,
)
from photomatagent.runtime.events import ContextCompactionCompleted
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import AllowAllPolicy
from photomatagent.runtime.state import ConversationState
from photomatagent.scientific.state import ScientificState
from photomatagent.sessions.store import EngineSnapshot, SessionSnapshot
from photomatagent.skills.loader import SkillLoader
from photomatagent.tools.registry import ToolRegistry
from photomatagent.workspace import Workspace


class FailingSummarizer:
    async def summarize(self, messages, previous):
        raise RuntimeError("summary unavailable")


def _surface():
    from photomatagent.tools.surface import ToolSurfaceStats

    return ToolSurfaceStats(
        registered_tools=0,
        direct_tools=0,
        deferred_tools=0,
        hidden_tools=0,
        direct_schema_chars=0,
        deferred_schema_chars=0,
        bridge_schema_chars=0,
        manifest_chars=0,
        visible_schema_chars=0,
        estimated_direct_schema_tokens=0,
        estimated_deferred_schema_tokens=0,
        estimated_bridge_schema_tokens=0,
        estimated_manifest_tokens=0,
        estimated_visible_schema_tokens=0,
        estimated_avoided_tokens=0,
    )


@pytest.mark.asyncio
async def test_automatic_compaction_failure_at_hard_limit_skips_provider(tmp_path):
    model = FakeModelProvider([])
    engine = ContextEngine(
        config=ContextEngineConfig(
            context_limit_tokens=4_000,
            response_reserve_tokens=64,
            safety_margin_tokens=64,
            compact_trigger_tokens=1_000,
            compact_target_tokens=600,
        ),
        summarizer=FailingSummarizer(),
    )
    runtime = AgentRuntime(
        model=model,
        tools=ToolRegistry(),
        workspace=Workspace(tmp_path),
        context_builder=ContextBuilder(SkillLoader(tmp_path / "empty-skills")),
        context_engine=engine,
        permission_policy=AllowAllPolicy(),
    )
    (tmp_path / "empty-skills").mkdir()
    for index in range(8):
        runtime.conversation_state.add(UserMessage(content=f"old goal {index}"))
        runtime.conversation_state.add(
            AssistantMessage(text=("old work " * 2_000) + str(index))
        )

    events = []
    with pytest.raises(ContextLimitExceeded):
        async for event in runtime.run("new goal"):
            events.append(event)

    assert model.requests == []
    assert any(event.kind == "context_compaction_failed" for event in events)
    assert any(
        event.kind == "context_compaction_failed" for event in engine_result_events(events)
    )


def engine_result_events(events):
    return events


@pytest.mark.asyncio
async def test_restore_without_engine_resets_previous_cursor(tmp_path):
    runtime = AgentRuntime(
        model=FakeModelProvider([]),
        tools=ToolRegistry(),
        workspace=Workspace(tmp_path),
        permission_policy=AllowAllPolicy(),
    )
    runtime.context_engine.restore(
        compaction_state=CompactionState(goal="old").model_dump(mode="json"),
        compacted_message_count=1,
        compaction_count=1,
    )
    snapshot = SessionSnapshot(
        conversation=ConversationState(messages=[UserMessage(content="fresh")]),
        scientific=ScientificState(goal="fresh"),
        engine=None,
    )
    runtime.restore_session(snapshot)
    assert runtime.context_engine.snapshot() == {
        "compaction_state": None,
        "compacted_message_count": 0,
        "compaction_count": 0,
    }


@pytest.mark.asyncio
async def test_restore_rejects_bad_cursor_without_mutating_current_runtime(tmp_path):
    runtime = AgentRuntime(
        model=FakeModelProvider([]),
        tools=ToolRegistry(),
        workspace=Workspace(tmp_path),
        permission_policy=AllowAllPolicy(),
    )
    runtime.conversation_state.add(UserMessage(content="live"))
    live_messages = runtime.conversation_state.messages
    live_engine = runtime.context_engine.snapshot()

    bad = SessionSnapshot(
        conversation=ConversationState(messages=[UserMessage(content="one")]),
        scientific=ScientificState(goal="bad"),
        engine=EngineSnapshot(
            compaction_state=CompactionState(goal="bad"),
            compacted_message_count=2,
            compaction_count=1,
        ),
    )
    with pytest.raises(ValueError):
        runtime.restore_session(bad)

    assert runtime.conversation_state.messages is live_messages
    assert runtime.context_engine.snapshot() == live_engine
    assert runtime.scientific_state.goal == ""


@pytest.mark.asyncio
async def test_restore_rejects_cursor_inside_tool_transaction(tmp_path):
    call = ToolCall(id="c1", name="read", arguments={"path": "x"})
    conversation = ConversationState(
        messages=[
            UserMessage(content="goal"),
            AssistantMessage(tool_calls=[call]),
            ToolResultMessage(tool_call_id="c1", tool_name="read", content="x"),
        ]
    )
    runtime = AgentRuntime(
        model=FakeModelProvider([]),
        tools=ToolRegistry(),
        workspace=Workspace(tmp_path),
        permission_policy=AllowAllPolicy(),
    )
    snapshot = SessionSnapshot(
        conversation=conversation,
        scientific=ScientificState(),
        engine=EngineSnapshot(
            compaction_state=CompactionState(goal="old"),
            compacted_message_count=2,
            compaction_count=1,
        ),
    )
    with pytest.raises(ValueError):
        runtime.restore_session(snapshot)
    assert runtime.conversation_state.messages == []


@pytest.mark.asyncio
async def test_consume_context_result_accounts_calls_without_usage_double_count(tmp_path):
    runtime = AgentRuntime(
        model=FakeModelProvider([]),
        tools=ToolRegistry(),
        workspace=Workspace(tmp_path),
        permission_policy=AllowAllPolicy(),
    )
    usage = ModelUsage(input_tokens=10, output_tokens=5)
    result = ContextBuildResult(
        messages=[],
        size=ContextSize(chars=0, tokens=0, messages=0),
        durable_size=ContextSize(chars=0, tokens=0, messages=0),
        events=[
            ContextCompactionCompleted(
                tokens_before=100,
                tokens_after=50,
                chars_before=400,
                chars_after=200,
                messages_before=4,
                messages_after=2,
                protected_turns=1,
            )
        ],
        compaction_model_calls=3,
        compaction_usages=[usage],
    )
    await runtime._consume_context_result(result)
    assert runtime.budget.model_calls == 3
    assert runtime.budget.total_tokens == usage.resolved_total_tokens
