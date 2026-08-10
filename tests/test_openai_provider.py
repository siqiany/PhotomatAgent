from __future__ import annotations

import re

import openai
import pytest

from photomatagent.errors import ProviderError
from photomatagent.models.openai import (
    OpenAIProvider,
    OpenAIStreamMapper,
    OpenAIToolNameCodec,
    openai_input,
    openai_tools,
)
from photomatagent.models.types import (
    AssistantMessage,
    ModelCompleted,
    ModelTextDelta,
    ModelToolCallArgumentsDelta,
    ModelToolCallCompleted,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    ModelRequest,
    ToolDefinition,
)
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.factory import create_default_registry
from photomatagent.workspace import Workspace


def _feed(mapper, events):
    return [mapped for event in events for mapped in mapper.feed(event)]


def test_openai_text_fragmented_tool_arguments_multiple_calls_and_usage():
    mapper = OpenAIStreamMapper("gpt-test")
    events = [
        {"type": "response.created", "response": {"id": "resp_1"}},
        {"type": "response.output_text.delta", "delta": "Looking "},
        {"type": "response.output_text.delta", "delta": "now."},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "function_call", "id": "item_1", "call_id": "call_1", "name": "glob", "arguments": ""},
        },
        {"type": "response.function_call_arguments.delta", "item_id": "item_1", "delta": "{\"pat"},
        {"type": "response.function_call_arguments.delta", "item_id": "item_1", "delta": "tern\":\"*.py\"}"},
        {"type": "response.function_call_arguments.done", "item_id": "item_1", "name": "glob", "arguments": "{\"pattern\":\"*.py\"}"},
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {"type": "function_call", "id": "item_2", "call_id": "call_2", "name": "grep", "arguments": ""},
        },
        {"type": "response.function_call_arguments.delta", "item_id": "item_2", "delta": "{\"pattern\":\"Agent\"}"},
        {"type": "response.function_call_arguments.done", "item_id": "item_2", "name": "grep", "arguments": "{\"pattern\":\"Agent\"}"},
        {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
            },
        },
    ]
    mapped = _feed(mapper, events)
    assert "".join(event.text for event in mapped if isinstance(event, ModelTextDelta)) == "Looking now."
    assert "".join(event.delta for event in mapped if isinstance(event, ModelToolCallArgumentsDelta) and event.tool_call_id == "call_1") == '{"pattern":"*.py"}'
    calls = [event.tool_call for event in mapped if isinstance(event, ModelToolCallCompleted)]
    assert [call.id for call in calls] == ["call_1", "call_2"]
    assert calls[0].arguments == {"pattern": "*.py"}
    completed = next(event for event in mapped if isinstance(event, ModelCompleted))
    assert completed.response.finish_reason == "tool_calls"
    assert completed.response.usage.total_tokens == 18
    assert completed.response.text == "Looking now."


def test_openai_tool_result_round_trip_input_keeps_call_id():
    items = openai_input(
        [
            AssistantMessage(tool_calls=[ToolCall(id="call_xyz", name="read", arguments={"path": "x"})]),
            ToolResultMessage(tool_call_id="call_xyz", tool_name="read", content="contents"),
        ]
    )
    assert items[0]["call_id"] == "call_xyz"
    assert items[1] == {"type": "function_call_output", "call_id": "call_xyz", "output": "contents"}


def test_openai_namespaced_tool_name_is_safe_and_round_trips():
    tools = [
        ToolDefinition(
            name="mock.run_calculation",
            description="run a calculation",
            input_schema={"type": "object", "properties": {}},
        ),
        ToolDefinition(
            name="echo",
            description="echo text",
            input_schema={"type": "object", "properties": {}},
        ),
    ]
    codec = OpenAIToolNameCodec(tools)
    definitions = openai_tools(tools, codec)
    encoded = definitions[0]["name"]

    assert encoded != "mock.run_calculation"
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", encoded)
    assert definitions[1]["name"] == "echo"

    mapper = OpenAIStreamMapper("gpt-test", codec)
    mapped = _feed(
        mapper,
        [
            {"type": "response.created", "response": {"id": "resp"}},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "item",
                    "call_id": "call",
                    "name": encoded,
                },
            },
            {
                "type": "response.function_call_arguments.done",
                "item_id": "item",
                "name": encoded,
                "arguments": "{}",
            },
        ],
    )
    completed = next(
        event for event in mapped if isinstance(event, ModelToolCallCompleted)
    )
    assert completed.tool_call.name == "mock.run_calculation"

    replay = openai_input(
        [
            AssistantMessage(
                tool_calls=[
                    ToolCall(
                        id="call", name="mock.run_calculation", arguments={}
                    )
                ]
            )
        ],
        codec,
    )
    assert replay[0]["name"] == encoded


def test_openai_mapper_provider_error():
    with pytest.raises(ProviderError):
        OpenAIStreamMapper("gpt-test").feed(
            {"type": "response.failed", "error": {"message": "bad request"}}
        )


def test_openai_provider_passes_base_url_to_official_sdk(monkeypatch):
    captured = {}

    class Client:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(openai, "AsyncOpenAI", Client)

    OpenAIProvider(
        "compatible-model",
        api_key="test-key",
        base_url="https://compatible.example/v1",
    )

    assert captured == {
        "api_key": "test-key",
        "base_url": "https://compatible.example/v1",
    }


def test_openai_invalid_arguments_are_not_parsed_before_done():
    mapper = OpenAIStreamMapper("gpt-test")
    mapper.feed({"type": "response.created", "response": {"id": "r"}})
    mapper.feed(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "function_call", "id": "i", "call_id": "c", "name": "read"},
        }
    )
    mapper.feed({"type": "response.function_call_arguments.delta", "item_id": "i", "delta": "{\"path\":"})
    with pytest.raises(ProviderError):
        mapper.feed(
            {"type": "response.function_call_arguments.done", "item_id": "i", "name": "read", "arguments": "{\"path\":"}
        )


@pytest.mark.asyncio
async def test_openai_provider_sends_full_stateless_history_each_round():
    class AsyncEvents:
        def __init__(self, events):
            self.events = events

        def __aiter__(self):
            return self._iterate()

        async def _iterate(self):
            for event in self.events:
                yield event

    class Responses:
        def __init__(self):
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            number = len(self.calls)
            return AsyncEvents(
                [
                    {"type": "response.created", "response": {"id": f"resp_{number}"}},
                    {"type": "response.output_text.delta", "delta": f"answer {number}"},
                    {
                        "type": "response.completed",
                        "response": {
                            "id": f"resp_{number}",
                            "status": "completed",
                            "output": [],
                            "usage": {},
                        },
                    },
                ]
            )

    class Client:
        def __init__(self):
            self.responses = Responses()

    client = Client()
    provider = OpenAIProvider("gpt-test", client=client)
    first = ModelRequest(messages=[UserMessage(content="one")])
    _ = [event async for event in provider.stream(first)]
    second = ModelRequest(
        messages=[
            UserMessage(content="one"),
            AssistantMessage(text="answer 1"),
            UserMessage(content="two"),
        ]
    )
    _ = [event async for event in provider.stream(second)]
    assert "previous_response_id" not in client.responses.calls[1]
    assert client.responses.calls[1]["input"] == [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "answer 1"},
        {"role": "user", "content": "two"},
    ]


@pytest.mark.asyncio
async def test_openai_provider_pairs_tool_calls_with_outputs_statelessly():
    class AsyncEvents:
        def __init__(self, events):
            self.events = events

        def __aiter__(self):
            return self._iterate()

        async def _iterate(self):
            for event in self.events:
                yield event

    class Responses:
        def __init__(self):
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            number = len(self.calls)
            return AsyncEvents(
                [
                    {"type": "response.created", "response": {"id": f"resp_{number}"}},
                    {
                        "type": "response.completed",
                        "response": {
                            "id": f"resp_{number}",
                            "status": "completed",
                            "output": [],
                            "usage": {},
                        },
                    },
                ]
            )

    class Client:
        def __init__(self):
            self.responses = Responses()

    client = Client()
    provider = OpenAIProvider("gpt-test", client=client)
    first = ModelRequest(messages=[UserMessage(content="one")])
    _ = [event async for event in provider.stream(first)]

    second = ModelRequest(
        messages=[
            UserMessage(content="one"),
            AssistantMessage(
                tool_calls=[
                    ToolCall(id="call_1", name="bash", arguments={"command": "ls"}),
                    ToolCall(id="call_2", name="glob", arguments={"pattern": "**/*"}),
                ]
            ),
            ToolResultMessage(
                tool_call_id="call_1",
                tool_name="bash",
                content="stdout:\nREADME.md",
            ),
            ToolResultMessage(
                tool_call_id="call_2",
                tool_name="glob",
                content="glob failed: ValueError",
                is_error=True,
            ),
        ]
    )
    _ = [event async for event in provider.stream(second)]

    sent = client.responses.calls[1]["input"]
    assert "previous_response_id" not in client.responses.calls[1]
    function_calls = [item for item in sent if item.get("type") == "function_call"]
    outputs = [item for item in sent if item.get("type") == "function_call_output"]
    assert {item["call_id"] for item in function_calls} == {"call_1", "call_2"}
    assert {item["call_id"] for item in outputs} == {"call_1", "call_2"}
    assert function_calls[0]["arguments"] == '{"command": "ls"}'


def test_default_registry_tool_names_are_openai_safe(tmp_path):
    registry = create_default_registry(ScientificState(), Workspace(tmp_path))
    codec = OpenAIToolNameCodec(registry.definitions())
    definitions = openai_tools(registry.definitions(), codec)
    assert len(definitions) == len(registry.list_tools()) == 14
    for definition in definitions:
        assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", definition["name"])
    for definition in registry.definitions():
        assert codec.decode(codec.encode(definition.name)) == definition.name
