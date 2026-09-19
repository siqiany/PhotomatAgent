from __future__ import annotations

from pathlib import Path

import pytest

from photomatagent.models.types import (
    AssistantMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from photomatagent.runtime.context import ContextBuilder
from photomatagent.runtime.context_engine import (
    CompactionState,
    ContextEngine,
    ContextEngineConfig,
)
from photomatagent.runtime.context_segments import split_atomic_spans
from photomatagent.runtime.state import ConversationState
from photomatagent.scientific.state import ScientificState
from photomatagent.skills.loader import SkillLoader
from photomatagent.tools.surface import ToolSurfaceStats


def _surface() -> ToolSurfaceStats:
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


def _builder(tmp_path: Path) -> ContextBuilder:
    empty = tmp_path / "empty-skills"
    empty.mkdir(exist_ok=True)
    return ContextBuilder(SkillLoader(empty))


def _transaction(index: int, *, content_size: int = 0) -> list:
    call = ToolCall(id=f"c{index}", name="read", arguments={"path": f"r{index}"})
    return [
        AssistantMessage(tool_calls=[call]),
        ToolResultMessage(
            tool_call_id=call.id,
            tool_name="read",
            content=(f"mock sample {index}; unknown validation; " * 200)
            + ("x" * content_size),
        ),
    ]


class RecordingSummarizer:
    def __init__(self) -> None:
        self.calls = 0
        self.previous_goals: list[str | None] = []
        self.last_model_calls = 0
        self.last_usages = []

    async def summarize(self, messages, previous):
        self.calls += 1
        self.last_model_calls += 1
        self.previous_goals.append(previous.goal if previous else None)
        return CompactionState(
            goal=f"summary-{self.calls}",
            progress=["preserved old work"],
        )


def _long_turn(count: int = 8) -> ConversationState:
    messages = [UserMessage(content="研究 GaAs；只用 mock 验证流程，不提交真实计算")]
    for index in range(count):
        messages.extend(_transaction(index))
    return ConversationState(messages=messages)


def _config() -> ContextEngineConfig:
    return ContextEngineConfig(
        context_limit_tokens=32_000,
        response_reserve_tokens=1_024,
        safety_margin_tokens=1_024,
        compact_trigger_tokens=12_000,
        compact_target_tokens=6_000,
        summary_max_tokens=1_024,
        protect_recent_turns=1,
        recent_transaction_count=2,
    )


async def _build(engine, conversation, builder, *, force=False):
    return await engine.build(
        conversation=conversation,
        scientific=ScientificState(),
        context_builder=builder,
        capability_manifest="",
        surface=_surface(),
        session_id="acceptance",
        force_compaction=force,
    )


@pytest.mark.asyncio
async def test_single_long_turn_compacts_and_next_context_is_paired(tmp_path):
    summarizer = RecordingSummarizer()
    engine = ContextEngine(config=_config(), summarizer=summarizer)
    conversation = _long_turn(8)
    before_durable = conversation.model_dump_json()

    result = await _build(engine, conversation, _builder(tmp_path))

    assert result.compaction_status == "completed"
    assert summarizer.calls == 1
    assert result.compaction_count == 1
    assert result.size.tokens <= _config().compact_target_tokens
    assert conversation.model_dump_json() == before_durable

    user_goals = [
        message.content
        for message in result.messages
        if isinstance(message, UserMessage)
        and message.content.startswith("研究 GaAs")
    ]
    assert user_goals == ["研究 GaAs；只用 mock 验证流程，不提交真实计算"]
    result_ids = [
        message.tool_call_id
        for message in result.messages
        if isinstance(message, ToolResultMessage)
    ]
    # The newest two complete transactions survive; older ones are summarized.
    assert result_ids == ["c6", "c7"]
    conversation_part = [
        message
        for message in result.messages
        if not (
            getattr(message, "kind", None) == "user"
            and "Current scientific state" in message.content
        )
        and not (
            getattr(message, "kind", None) == "system"
            and "Compaction Summary" in message.content
        )
    ]
    # Removing the compacted-goal anchor? The original goal is intentionally
    # re-anchored, so tool grouping is validated on the suffix only.
    suffix = [
        message
        for message in conversation_part
        if isinstance(message, (AssistantMessage, ToolResultMessage))
    ]
    assert [call.id for message in suffix if isinstance(message, AssistantMessage) for call in message.tool_calls] == [
        "c6",
        "c7",
    ]
    split_atomic_spans(suffix)


@pytest.mark.asyncio
async def test_two_automatic_compactions_are_monotonic_and_keep_durable_history(tmp_path):
    summarizer = RecordingSummarizer()
    engine = ContextEngine(config=_config(), summarizer=summarizer)
    conversation = _long_turn(8)
    first = await _build(engine, conversation, _builder(tmp_path))
    cursor_after_first = engine.snapshot()["compacted_message_count"]
    assert first.compaction_status == "completed"

    for index in range(8, 16):
        conversation.messages.extend(_transaction(index))
    before_durable = conversation.model_dump_json()

    second = await _build(engine, conversation, _builder(tmp_path))
    cursor_after_second = engine.snapshot()["compacted_message_count"]

    assert second.compaction_status == "completed"
    assert summarizer.calls == 2
    assert summarizer.previous_goals == [None, "summary-1"]
    assert cursor_after_second > cursor_after_first
    assert engine.snapshot()["compaction_count"] == 2
    assert conversation.model_dump_json() == before_durable
    # The second summary input is the newly eligible prefix, not a replay of
    # already-summarized transactions.
    second_goal = [
        message.content
        for message in second.messages
        if isinstance(message, UserMessage) and message.content.startswith("研究 GaAs")
    ]
    assert second_goal == ["研究 GaAs；只用 mock 验证流程，不提交真实计算"]
    result_ids = [
        message.tool_call_id
        for message in second.messages
        if isinstance(message, ToolResultMessage)
    ]
    assert result_ids == ["c14", "c15"]


@pytest.mark.asyncio
async def test_restore_does_not_replay_already_summarized_tool_results(tmp_path):
    summarizer = RecordingSummarizer()
    engine = ContextEngine(config=_config(), summarizer=summarizer)
    conversation = _long_turn(8)
    await _build(engine, conversation, _builder(tmp_path))
    snapshot = engine.snapshot()

    restored = ContextEngine(config=_config(), summarizer=summarizer)
    restored.restore(**snapshot)
    result = await _build(restored, conversation, _builder(tmp_path))

    result_ids = [
        message.tool_call_id
        for message in result.messages
        if isinstance(message, ToolResultMessage)
    ]
    assert result_ids == ["c6", "c7"]
    assert result.compaction_count == 1


@pytest.mark.asyncio
async def test_unsafe_tool_pairing_fails_without_summarizer_call(tmp_path):
    summarizer = RecordingSummarizer()
    engine = ContextEngine(config=_config(), summarizer=summarizer)
    conversation = ConversationState(
        messages=[
            UserMessage(content="goal"),
            ToolResultMessage(
                tool_call_id="orphan", tool_name="read", content="x" * 10_000
            ),
        ]
    )
    before = engine.snapshot()
    result = await _build(engine, conversation, _builder(tmp_path), force=True)

    assert result.compaction_status == "failed"
    assert result.compaction_reason == "unsafe_history"
    assert summarizer.calls == 0
    assert engine.snapshot() == before


@pytest.mark.asyncio
async def test_multi_tool_assistant_group_is_never_split(tmp_path):
    summarizer = RecordingSummarizer()
    engine = ContextEngine(config=_config(), summarizer=summarizer)
    messages = [UserMessage(content="one goal with parallel tools")]
    for index in range(6):
        messages.extend(_transaction(index))
    parallel_calls = [
        ToolCall(id="p1", name="read", arguments={"path": "p1"}),
        ToolCall(id="p2", name="read", arguments={"path": "p2"}),
    ]
    messages.append(AssistantMessage(tool_calls=parallel_calls))
    messages.extend(
        [
            ToolResultMessage(
                tool_call_id="p1", tool_name="read", content="P1 " * 800
            ),
            ToolResultMessage(
                tool_call_id="p2", tool_name="read", content="P2 " * 800
            ),
        ]
    )
    conversation = ConversationState(messages=messages)

    result = await _build(engine, conversation, _builder(tmp_path))

    assert result.compaction_status == "completed"
    suffix = [
        message
        for message in result.messages
        if isinstance(message, (AssistantMessage, ToolResultMessage))
    ]
    suffix_calls = [
        call.id
        for message in suffix
        if isinstance(message, AssistantMessage)
        for call in message.tool_calls
    ]
    suffix_results = [
        message.tool_call_id
        for message in suffix
        if isinstance(message, ToolResultMessage)
    ]
    assert suffix_calls[-2:] == ["p1", "p2"]
    assert suffix_results[-2:] == ["p1", "p2"]
    split_atomic_spans(suffix)
