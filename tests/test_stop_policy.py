from __future__ import annotations

from photomatagent.models.types import ModelResponse, ToolCall
from photomatagent.runtime.budget import BudgetState
from photomatagent.runtime.stop_policy import StopPolicy


def test_final_response_stops():
    policy = StopPolicy()
    decision = policy.should_stop(
        iteration=1,
        response=ModelResponse(text="done", finish_reason="stop"),
        budget=BudgetState(max_iterations=10),
    )
    assert decision.should_stop and decision.reason == "final_response"


def test_tool_calls_continue():
    policy = StopPolicy()
    decision = policy.should_stop(
        iteration=1,
        response=ModelResponse(
            finish_reason="tool_calls",
            tool_calls=[ToolCall(name="echo", arguments={"text": "x"})],
        ),
        budget=BudgetState(max_iterations=10),
    )
    assert not decision.should_stop


def test_max_iterations_stops_even_with_tool_calls():
    policy = StopPolicy()
    decision = policy.should_stop(
        iteration=10,
        response=ModelResponse(
            finish_reason="tool_calls",
            tool_calls=[ToolCall(name="echo", arguments={"text": "x"})],
        ),
        budget=BudgetState(max_iterations=10),
    )
    assert decision.should_stop and decision.reason == "max_iterations"


def test_fatal_error_stops():
    policy = StopPolicy()
    decision = policy.should_stop(
        iteration=1,
        response=ModelResponse(),
        budget=BudgetState(),
        fatal_error="kaboom",
    )
    assert decision.should_stop and decision.reason.startswith("fatal_error")
