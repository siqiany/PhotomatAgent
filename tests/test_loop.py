from __future__ import annotations

import pytest

from photomatagent.models.fake import FakeModelProvider, FakeResponse, scripted_tool_call
from photomatagent.models.types import ModelResponse, ToolCall
from photomatagent.runtime.loop import AgentRuntime

from conftest import collect, make_runtime


@pytest.mark.asyncio
async def test_full_loop_event_order():
    """Model -> ToolRequested -> ToolStarted -> ToolCompleted -> Model -> LoopCompleted."""
    model = FakeModelProvider(
        [
            scripted_tool_call(
                "mock.run_calculation",
                {"material": "GaAs", "calculation_type": "band_structure"},
            ),
            FakeResponse(text="The mock calculation suggests a band gap of 0.31 eV."),
        ]
    )
    runtime = make_runtime(model)
    events = await collect(runtime, "investigate material GaAs")
    kinds = [e.kind for e in events]
    assert kinds == [
        "loop_started",
        "loop_iteration_started",
        "model_request_started",
        "model_response_completed",
        "tool_requested",
        "tool_started",
        "tool_completed",
        "scientific_state_updated",
        "budget_updated",
        "loop_iteration_started",
        "model_request_started",
        "text_delta",
        "model_response_completed",
        "budget_updated",
        "loop_completed",
    ]

    # Scientific state gained evidence + a calculation record.
    assert len(runtime.scientific_state.evidence) == 1
    assert len(runtime.scientific_state.calculations) == 1
    assert runtime.scientific_state.goal == "investigate material GaAs"
    assert runtime.budget.model_calls == 2
    assert runtime.budget.tool_calls == 1


@pytest.mark.asyncio
async def test_auto_fake_model_completes_tool_loop():
    """Auto-mode fake: request mock calc, then summarize the tool result."""
    model = FakeModelProvider(auto=True)
    runtime = make_runtime(model)
    events = await collect(runtime, "investigate material InAs")
    kinds = [e.kind for e in events]
    assert "tool_requested" in kinds
    assert "tool_completed" in kinds
    assert kinds[-1] == "loop_completed"
    calc = runtime.scientific_state.calculations[0]
    assert calc.input_reference["material"] == "InAs"


@pytest.mark.asyncio
async def test_conversation_persists_across_runs():
    """One runtime, multiple goals: history accumulates, loop still completes."""
    model = FakeModelProvider(
        [FakeResponse(text="first"), FakeResponse(text="second")]
    )
    runtime = make_runtime(model)
    await collect(runtime, "question one")
    await collect(runtime, "question two")

    # Two user messages + two assistant messages.
    assert sum(1 for e in runtime._conversation.messages if e.role == "user") == 2
    assert sum(1 for e in runtime._conversation.messages if e.role == "assistant") == 2


@pytest.mark.asyncio
async def test_max_iterations_stops_loop():
    """A model that always calls tools must be cut off by max_iterations."""

    class AlwaysToolModel:
        name = "always_tool"

        async def complete(self, messages, tools):
            return ModelResponse(
                tool_calls=[ToolCall(name="echo", arguments={"text": "loop"})],
                finish_reason="tool_calls",
            )

    runtime = make_runtime(AlwaysToolModel(), max_iterations=2)
    events = await collect(runtime, "keep going")
    completed = [e for e in events if e.kind == "loop_completed"][0]
    assert completed.reason == "max_iterations"
    assert completed.iterations == 2

@pytest.mark.asyncio
async def test_loop_failed_emitted_before_raise():
    class BrokenModel:
        name = "broken"

        async def complete(self, messages, tools):
            raise RuntimeError("boom")

    runtime = make_runtime(BrokenModel())
    seen = []
    with pytest.raises(RuntimeError):
        async for event in runtime.run("x"):
            seen.append(event)
    assert any(e.kind == "loop_failed" for e in seen)
