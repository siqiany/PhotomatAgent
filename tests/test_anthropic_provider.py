from __future__ import annotations

import pytest

from photomatagent.errors import ProviderError
from photomatagent.models.anthropic import AnthropicStreamMapper, anthropic_input
from photomatagent.models.types import (
    AssistantMessage,
    ModelCompleted,
    ModelTextDelta,
    ModelToolCallArgumentsDelta,
    ModelToolCallCompleted,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


def _feed(mapper, events):
    return [mapped for event in events for mapped in mapper.feed(event)]


def test_anthropic_streaming_text_fragmented_input_and_usage():
    mapper = AnthropicStreamMapper("claude-test")
    mapped = _feed(
        mapper,
        [
            {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 12}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Checking "}},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "read", "input": {}}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"pa"}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "th\":\"x.py\"}"}},
            {"type": "content_block_stop", "index": 1},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 8}},
            {"type": "message_stop"},
        ],
    )
    assert "".join(event.text for event in mapped if isinstance(event, ModelTextDelta)) == "Checking "
    assert "".join(event.delta for event in mapped if isinstance(event, ModelToolCallArgumentsDelta)) == '{"path":"x.py"}'
    call = next(event.tool_call for event in mapped if isinstance(event, ModelToolCallCompleted))
    assert call.id == "toolu_1"
    assert call.arguments == {"path": "x.py"}
    completed = next(event for event in mapped if isinstance(event, ModelCompleted))
    assert completed.response.finish_reason == "tool_calls"
    assert completed.response.usage.input_tokens == 12
    assert completed.response.usage.output_tokens == 8


def test_anthropic_multiple_tool_calls():
    mapper = AnthropicStreamMapper("claude-test")
    mapped = _feed(
        mapper,
        [
            {"type": "message_start", "message": {"id": "m", "usage": {}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "t1", "name": "glob", "input": {}}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"pattern\":\"*.py\"}"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "t2", "name": "grep", "input": {}}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"pattern\":\"x\"}"}},
            {"type": "content_block_stop", "index": 1},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {}},
            {"type": "message_stop"},
        ],
    )
    calls = [event.tool_call for event in mapped if isinstance(event, ModelToolCallCompleted)]
    assert [call.id for call in calls] == ["t1", "t2"]


def test_anthropic_tool_result_round_trip_and_grouping():
    system, messages = anthropic_input(
        [
            SystemMessage(content="system"),
            UserMessage(content="question"),
            AssistantMessage(
                tool_calls=[
                    ToolCall(id="t1", name="glob", arguments={"pattern": "*.py"}),
                    ToolCall(id="t2", name="grep", arguments={"pattern": "x"}),
                ]
            ),
            ToolResultMessage(tool_call_id="t1", tool_name="glob", content="a.py"),
            ToolResultMessage(tool_call_id="t2", tool_name="grep", content="a.py:1:x", is_error=True),
        ]
    )
    assert system == "system"
    assert messages[-1]["role"] == "user"
    assert [block["tool_use_id"] for block in messages[-1]["content"]] == ["t1", "t2"]
    assert messages[-1]["content"][1]["is_error"] is True


def test_anthropic_mapper_provider_error():
    with pytest.raises(ProviderError):
        AnthropicStreamMapper("claude-test").feed(
            {"type": "error", "error": {"message": "overloaded"}}
        )
