from __future__ import annotations

from pathlib import Path

import pytest

from photomatagent.models.types import (
    AssistantMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from photomatagent.runtime import context_engine as context_engine_module
from photomatagent.runtime.context import ContextBuilder
from photomatagent.runtime.context_engine import (
    CompactionState,
    ContextEngine,
    ContextEngineConfig,
    ContextSize,
)
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


class CountingSummarizer:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    async def summarize(self, messages, previous):
        self.calls += 1
        if self.fail:
            raise RuntimeError("summary failed")
        return CompactionState(goal="summary", progress=["preserved"])


async def _build(
    engine: ContextEngine,
    conversation: ConversationState,
    builder: ContextBuilder,
    *,
    force: bool = False,
):
    return await engine.build(
        conversation=conversation,
        scientific=ScientificState(),
        context_builder=builder,
        capability_manifest="",
        surface=_surface(),
        session_id="s",
        force_compaction=force,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "before_tokens, expected_calls",
    [(255_999, 0), (256_000, 1), (256_001, 1)],
)
async def test_absolute_trigger_boundary_calls_summarizer_at_exactly_threshold(
    tmp_path, monkeypatch, before_tokens, expected_calls
):
    summarizer = CountingSummarizer()
    engine = ContextEngine(
        config=ContextEngineConfig(
            context_limit_tokens=512_000,
            protect_recent_turns=1,
            compact_trigger_tokens=256_000,
            compact_target_tokens=128_000,
        ),
        summarizer=summarizer,
    )
    conversation = ConversationState(
        messages=[
            UserMessage(content="old goal"),
            AssistantMessage(text="old work"),
            UserMessage(content="current goal"),
        ]
    )

    def fake_measure(messages, surface):
        has_summary = any(
            getattr(message, "kind", None) == "system"
            and "Compaction Summary" in message.content
            for message in messages
        )
        tokens = 100_000 if has_summary else before_tokens
        return ContextSize(chars=tokens * 4, tokens=tokens, messages=len(messages))

    monkeypatch.setattr(context_engine_module, "_measure", fake_measure)
    result = await _build(engine, conversation, _builder(tmp_path), force=False)
    assert summarizer.calls == expected_calls
    if expected_calls:
        assert result.compaction_status == "completed"
        assert result.size.tokens == 100_000
    else:
        assert result.compaction_status == "not_requested"


@pytest.mark.asyncio
async def test_trigger_latched_before_prune_still_compacts(tmp_path, monkeypatch):
    summarizer = CountingSummarizer()
    engine = ContextEngine(
        config=ContextEngineConfig(
            context_limit_tokens=512_000,
            protect_recent_turns=1,
            compact_trigger_tokens=256_000,
            compact_target_tokens=128_000,
        ),
        summarizer=summarizer,
    )
    call = ToolCall(id="old", name="read", arguments={"path": "old.txt"})
    conversation = ConversationState(
        messages=[
            UserMessage(content="old goal"),
            AssistantMessage(tool_calls=[call]),
            ToolResultMessage(
                tool_call_id="old", tool_name="read", content="x" * 100_000
            ),
            UserMessage(content="current goal"),
        ]
    )

    def fake_measure(messages, surface):
        has_placeholder = any(
            getattr(message, "kind", None) == "tool_result"
            and message.content.startswith("[Previous tool output omitted")
            for message in messages
        )
        has_summary = any(
            getattr(message, "kind", None) == "system"
            and "Compaction Summary" in message.content
            for message in messages
        )
        if has_summary:
            tokens = 1_000
        elif has_placeholder:
            tokens = 1_000
        else:
            tokens = 256_000
        return ContextSize(chars=tokens * 4, tokens=tokens, messages=len(messages))

    monkeypatch.setattr(context_engine_module, "_measure", fake_measure)
    result = await _build(engine, conversation, _builder(tmp_path))
    assert summarizer.calls == 1
    assert result.compaction_status == "completed"
    assert any(event.kind == "context_prune_completed" for event in result.events)
    assert any(event.kind == "context_compaction_completed" for event in result.events)


@pytest.mark.asyncio
async def test_failed_auto_compaction_uses_retry_cooldown_until_hard_limit(
    tmp_path, monkeypatch
):
    summarizer = CountingSummarizer(fail=True)
    engine = ContextEngine(
        config=ContextEngineConfig(
            context_limit_tokens=512_000,
            protect_recent_turns=1,
            compact_trigger_tokens=256_000,
            compact_target_tokens=128_000,
        ),
        summarizer=summarizer,
    )
    conversation = ConversationState(
        messages=[
            UserMessage(content="old goal"),
            AssistantMessage(text="old work"),
            UserMessage(content="current goal"),
        ]
    )

    def fake_measure(messages, surface):
        return ContextSize(chars=256_000 * 4, tokens=256_000, messages=len(messages))

    monkeypatch.setattr(context_engine_module, "_measure", fake_measure)
    first = await _build(engine, conversation, _builder(tmp_path))
    assert summarizer.calls == 1
    assert first.compaction_status == "failed"
    second = await _build(engine, conversation, _builder(tmp_path))
    assert summarizer.calls == 1
    assert second.compaction_status == "skipped"
    assert second.compaction_reason == "retry_cooldown"
    third = await _build(engine, conversation, _builder(tmp_path), force=True)
    assert summarizer.calls == 2
    assert third.compaction_status == "failed"


@pytest.mark.asyncio
async def test_invalid_summary_keeps_engine_cursor_and_durable_history(tmp_path):
    class InvalidSummary:
        async def summarize(self, messages, previous):
            raise ValueError("invalid JSON")

    engine = ContextEngine(
        config=ContextEngineConfig(
            context_limit_tokens=6_000,
            response_reserve_tokens=1_024,
            safety_margin_tokens=64,
            compact_trigger_tokens=None,
            compact_trigger_ratio=0.90,
            prune_trigger_ratio=0.50,
            target_ratio=0.40,
            protect_recent_turns=1,
        ),
        summarizer=InvalidSummary(),
    )
    conversation = ConversationState(
        messages=[
            UserMessage(content="old goal"),
            AssistantMessage(text="old work " * 400),
            UserMessage(content="current goal"),
        ]
    )
    before_engine = engine.snapshot()
    before_history = conversation.model_dump_json()
    result = await _build(engine, conversation, _builder(tmp_path), force=True)

    assert result.compaction_status == "failed"
    assert engine.snapshot() == before_engine
    assert conversation.model_dump_json() == before_history


@pytest.mark.asyncio
async def test_no_reduction_is_skipped_without_committing(tmp_path, monkeypatch):
    summarizer = CountingSummarizer()
    engine = ContextEngine(
        config=ContextEngineConfig(
            context_limit_tokens=512_000,
            protect_recent_turns=1,
            compact_trigger_tokens=256_000,
            compact_target_tokens=128_000,
        ),
        summarizer=summarizer,
    )
    conversation = ConversationState(
        messages=[
            UserMessage(content="old goal"),
            AssistantMessage(text="old work"),
            UserMessage(content="current goal"),
        ]
    )

    def fake_measure(messages, surface):
        expanded = any(
            getattr(message, "kind", None) == "system"
            and "Compaction Summary" in message.content
            for message in messages
        )
        tokens = 300_000 if expanded else 256_000
        return ContextSize(chars=tokens * 4, tokens=tokens, messages=len(messages))

    monkeypatch.setattr(context_engine_module, "_measure", fake_measure)
    before_engine = engine.snapshot()
    result = await _build(engine, conversation, _builder(tmp_path))
    assert summarizer.calls == 1
    assert result.compaction_status == "skipped"
    assert result.compaction_reason == "no_reduction"
    assert engine.snapshot() == before_engine


@pytest.mark.asyncio
async def test_target_unreachable_keeps_cursor_and_uses_safe_copy(tmp_path, monkeypatch):
    summarizer = CountingSummarizer()
    engine = ContextEngine(
        config=ContextEngineConfig(
            context_limit_tokens=512_000,
            protect_recent_turns=1,
            compact_trigger_tokens=256_000,
            compact_target_tokens=128_000,
        ),
        summarizer=summarizer,
    )
    conversation = ConversationState(
        messages=[
            UserMessage(content="old goal"),
            AssistantMessage(text="old work"),
            UserMessage(content="current goal"),
        ]
    )

    def fake_measure(messages, surface):
        expanded = any(
            getattr(message, "kind", None) == "system"
            and "Compaction Summary" in message.content
            for message in messages
        )
        tokens = 200_000 if expanded else 256_000
        return ContextSize(chars=tokens * 4, tokens=tokens, messages=len(messages))

    monkeypatch.setattr(context_engine_module, "_measure", fake_measure)
    before_engine = engine.snapshot()
    result = await _build(engine, conversation, _builder(tmp_path))
    assert summarizer.calls == 1
    assert result.compaction_status == "failed"
    assert result.compaction_reason == "target_unreachable"
    assert result.request_allowed is True
    assert engine.snapshot() == before_engine


@pytest.mark.asyncio
async def test_target_unreachable_above_hard_limit_denies_request(tmp_path, monkeypatch):
    summarizer = CountingSummarizer()
    engine = ContextEngine(
        config=ContextEngineConfig(
            context_limit_tokens=512_000,
            protect_recent_turns=1,
            compact_trigger_tokens=256_000,
            compact_target_tokens=128_000,
        ),
        summarizer=summarizer,
    )
    conversation = ConversationState(
        messages=[
            UserMessage(content="old goal"),
            AssistantMessage(text="old work"),
            UserMessage(content="current goal"),
        ]
    )

    def fake_measure(messages, surface):
        expanded = any(
            getattr(message, "kind", None) == "system"
            and "Compaction Summary" in message.content
            for message in messages
        )
        tokens = 490_000 if expanded else 500_000
        return ContextSize(chars=tokens * 4, tokens=tokens, messages=len(messages))

    monkeypatch.setattr(context_engine_module, "_measure", fake_measure)
    result = await _build(engine, conversation, _builder(tmp_path))
    assert summarizer.calls == 1
    assert result.compaction_status == "failed"
    assert result.compaction_reason == "target_unreachable"
    assert result.request_allowed is False
    assert result.limit_error is not None


@pytest.mark.asyncio
async def test_failed_summary_preserves_partial_call_and_usage_accounting(tmp_path):
    from photomatagent.models.types import ModelUsage
    from photomatagent.runtime.context_engine import SummaryError

    usage = ModelUsage(input_tokens=30, output_tokens=10)

    class PartialFailure:
        def __init__(self):
            self.last_model_calls = 2
            self.last_usages = [usage]

        async def summarize(self, messages, previous):
            raise SummaryError("provider_error", "second chunk failed")

    engine = ContextEngine(
        config=ContextEngineConfig(
            context_limit_tokens=512_000,
            protect_recent_turns=1,
            compact_trigger_tokens=256_000,
            compact_target_tokens=128_000,
        ),
        summarizer=PartialFailure(),
    )
    conversation = ConversationState(
        messages=[
            UserMessage(content="old goal"),
            AssistantMessage(text="old work"),
            UserMessage(content="current goal"),
        ]
    )
    result = await _build(engine, conversation, _builder(tmp_path), force=True)
    assert result.compaction_status == "failed"
    assert result.compaction_model_calls == 2
    assert result.compaction_usages == [usage]


@pytest.mark.asyncio
async def test_cancelled_summary_raises_typed_cancellation_with_partial_result(
    tmp_path,
):
    from photomatagent.models.types import ModelUsage
    from photomatagent.runtime.context_engine import ContextBuildCancelled

    usage = ModelUsage(input_tokens=2, output_tokens=3)

    class CancellingSummarizer:
        def __init__(self):
            self.last_model_calls = 1
            self.last_usages = [usage]

        async def summarize(self, messages, previous):
            raise __import__("asyncio").CancelledError()

    engine = ContextEngine(
        config=ContextEngineConfig(
            context_limit_tokens=512_000,
            protect_recent_turns=1,
            compact_trigger_tokens=256_000,
            compact_target_tokens=128_000,
        ),
        summarizer=CancellingSummarizer(),
    )
    conversation = ConversationState(
        messages=[
            UserMessage(content="old goal"),
            AssistantMessage(text="old work"),
            UserMessage(content="current goal"),
        ]
    )
    with pytest.raises(ContextBuildCancelled) as exc_info:
        await _build(engine, conversation, _builder(tmp_path), force=True)
    result = exc_info.value.result
    assert result.compaction_status == "failed"
    assert result.compaction_reason == "cancelled"
    assert result.request_allowed is True
    assert result.compaction_model_calls == 1
    assert result.compaction_usages == [usage]
