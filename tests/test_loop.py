from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from photomatagent.errors import ProviderError
from photomatagent.models.fake import FakeModelProvider, FakeResponse, scripted_tool_call
from photomatagent.models.types import (
    AssistantMessage,
    ModelRequest,
    ModelStreamEvent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from photomatagent.workspace import Workspace

from conftest import collect, make_runtime


@pytest.mark.asyncio
async def test_each_run_has_a_distinct_run_id_within_one_session():
    model = FakeModelProvider([FakeResponse(text="one"), FakeResponse(text="two")])
    runtime = make_runtime(model)

    first = await collect(runtime, "first")
    second = await collect(runtime, "second")

    assert {event.run_id for event in first} == {first[0].run_id}
    assert {event.run_id for event in second} == {second[0].run_id}
    assert first[0].run_id is not None
    assert first[0].run_id != second[0].run_id
    assert first[0].session_id == second[0].session_id


@pytest.mark.asyncio
async def test_full_streaming_tool_loop_event_order():
    model = FakeModelProvider(
        [
            scripted_tool_call(
                "tool_call",
                {
                    "name": "mock.run_calculation",
                    "arguments": {
                        "material": "GaAs",
                        "calculation_type": "band_structure",
                    },
                },
                tool_call_id="provider-call-1",
            ),
            FakeResponse(
                text="The result is 0.31 eV.",
                text_deltas=["The result ", "is 0.31 eV."],
            ),
        ]
    )
    runtime = make_runtime(model)
    events = await collect(runtime, "investigate material GaAs")
    assert [event.kind for event in events] == [
        "loop_started",
        "loop_iteration_started",
        "model_request_started",
        "model_stream_started",
        "tool_call_started",
        "tool_call_arguments_delta",
        "tool_call_completed",
        "model_response_completed",
        "tool_requested",
        "tool_started",
        "tool_completed",
        "scientific_state_updated",
        "budget_updated",
        "loop_iteration_started",
        "model_request_started",
        "model_stream_started",
        "text_delta",
        "text_delta",
        "model_response_completed",
        "budget_updated",
        "loop_completed",
    ]
    assert len(runtime.scientific_state.evidence) == 1
    assert runtime.budget.model_calls == 2
    assert runtime.budget.tool_calls == 1


@pytest.mark.asyncio
async def test_tool_call_id_round_trips_into_next_model_request():
    model = FakeModelProvider(
        [
            scripted_tool_call(
                "glob", {"pattern": "pyproject.toml"}, tool_call_id="call_vendor_123"
            ),
            FakeResponse(text="done"),
        ]
    )
    runtime = make_runtime(model)
    await collect(runtime, "echo")
    second_request = model.requests[1]
    assistant = next(message for message in second_request.messages if isinstance(message, AssistantMessage))
    result = next(message for message in second_request.messages if isinstance(message, ToolResultMessage))
    assert assistant.tool_calls[0].id == "call_vendor_123"
    assert result.tool_call_id == "call_vendor_123"
    assert result.tool_name == "glob"


@pytest.mark.asyncio
async def test_glob_then_read_then_final_multi_turn(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "loop.py").write_text("class AgentRuntime: pass\n", encoding="utf-8")
    model = FakeModelProvider(
        [
            scripted_tool_call("glob", {"pattern": "src/**/*.py"}, tool_call_id="glob-1"),
            scripted_tool_call("read", {"path": "src/loop.py"}, tool_call_id="read-1"),
            FakeResponse(text="Agent Loop is in src/loop.py"),
        ]
    )
    runtime = make_runtime(model, workspace=Workspace(tmp_path))
    events = await collect(runtime, "find the loop")
    assert [event.tool_name for event in events if event.kind == "tool_completed"] == ["glob", "read"]
    assert len(model.requests) == 3
    assert model.requests[2].messages[-1].tool_call_id == "read-1"


@pytest.mark.asyncio
async def test_multiple_tool_calls_execute_sequentially(tmp_path):
    (tmp_path / "a.py").write_text("needle\n", encoding="utf-8")
    calls = [
        ToolCall(id="glob-call", name="glob", arguments={"pattern": "*.py"}),
        ToolCall(id="grep-call", name="grep", arguments={"pattern": "needle"}),
    ]
    model = FakeModelProvider([FakeResponse(tool_calls=calls), FakeResponse(text="done")])
    runtime = make_runtime(model, workspace=Workspace(tmp_path))
    events = await collect(runtime, "inspect")
    assert [event.tool_call_id for event in events if event.kind == "tool_started"] == [
        "glob-call",
        "grep-call",
    ]
    results = [
        message
        for message in model.requests[1].messages
        if isinstance(message, ToolResultMessage)
    ]
    assert [result.tool_call_id for result in results] == ["glob-call", "grep-call"]


@pytest.mark.asyncio
async def test_conversation_persists_across_user_turns():
    model = FakeModelProvider([FakeResponse(text="first"), FakeResponse(text="second")])
    runtime = make_runtime(model)
    await collect(runtime, "question one")
    await collect(runtime, "question two")
    assert sum(isinstance(message, UserMessage) for message in runtime.conversation_state.messages) == 2
    assert sum(isinstance(message, AssistantMessage) for message in runtime.conversation_state.messages) == 2


@pytest.mark.asyncio
async def test_max_iterations_stops_loop():
    class AlwaysToolModel:
        provider = "fake"
        model = "always-tool"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            provider = FakeModelProvider([scripted_tool_call("echo", {"text": "loop"})])
            async for event in provider.stream(request):
                yield event

    runtime = make_runtime(AlwaysToolModel(), max_iterations=2)
    events = await collect(runtime, "keep going")
    completed = next(event for event in events if event.kind == "loop_completed")
    assert completed.reason == "max_iterations"
    assert completed.iterations == 2


@pytest.mark.asyncio
async def test_next_turn_drops_unfulfilled_tool_calls_after_max_iterations():
    model = FakeModelProvider(
        [
            scripted_tool_call("echo", {"text": "loop"}, tool_call_id="call_pending"),
            FakeResponse(text="second run answer"),
        ]
    )
    runtime = make_runtime(model, max_iterations=1)
    await collect(runtime, "first goal")

    # First model request of the second user turn must not replay the
    # assistant tool call that was never executed (no tool result exists),
    # otherwise providers reject the unmatched function_call.
    await collect(runtime, "second goal")
    second_turn_first_request = model.requests[1]
    assistant_calls = [
        call
        for message in second_turn_first_request.messages
        if isinstance(message, AssistantMessage)
        for call in message.tool_calls
    ]
    assert {call.id for call in assistant_calls} == set()
    pending_results = [
        message
        for message in second_turn_first_request.messages
        if isinstance(message, ToolResultMessage)
    ]
    assert pending_results == []


@pytest.mark.asyncio
async def test_provider_failure_is_emitted_before_loop_failure():
    class BrokenModel:
        provider = "broken"
        model = "broken-model"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            if False:
                yield
            raise ProviderError("broken", "boom")

    runtime = make_runtime(BrokenModel())
    seen = []
    with pytest.raises(ProviderError):
        async for event in runtime.run("x"):
            seen.append(event)
    kinds = [event.kind for event in seen]
    assert kinds[-2:] == ["provider_failed", "loop_failed"]
