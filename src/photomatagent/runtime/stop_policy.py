"""StopPolicy: decides when the loop ends.

Kept outside the loop body so future scientific stopping criteria
(confidence thresholds, contradictions, information gain) can be added
without touching the loop implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

from photomatagent.models.types import ModelResponse
from photomatagent.runtime.budget import BudgetState


@dataclass(frozen=True)
class StopDecision:
    should_stop: bool
    reason: str = ""


class StopPolicy:
    def should_stop(
        self,
        *,
        iteration: int,
        response: ModelResponse,
        budget: BudgetState,
        fatal_error: str | None = None,
    ) -> StopDecision:
        if fatal_error is not None:
            return StopDecision(True, f"fatal_error: {fatal_error}")
        if iteration >= budget.max_iterations:
            return StopDecision(True, "max_iterations")
        if (budget.max_total_tokens is not None
                and budget.total_tokens >= budget.max_total_tokens):
            return StopDecision(True, "max_total_tokens")

        finish_reason = response.finish_reason
        if finish_reason == "max_tokens":
            # The provider cut the response off.  Even when tool-call
            # fragments are present, they are not a trustworthy request and
            # must not take the normal tool execution path.
            return StopDecision(True, "output_truncated")
        if finish_reason == "error":
            return StopDecision(True, "error")
        if finish_reason == "cancelled":
            return StopDecision(True, "cancelled")
        if finish_reason == "unknown":
            return StopDecision(True, "unknown")

        if finish_reason == "tool_calls":
            if response.tool_calls:
                return StopDecision(False)
            # A response that claims tool calls but carries none cannot advance
            # the loop and is not a successful final answer.
            return StopDecision(True, "empty_tool_calls")

        if finish_reason == "stop":
            if response.tool_calls:
                # Tool calls are only legal with finish_reason=tool_calls;
                # never execute an inconsistent fragment.
                return StopDecision(True, "unexpected_tool_calls")
            if response.text.strip():
                return StopDecision(True, "final_response")
            return StopDecision(True, "empty_response")

        # Defensive fallback for future literals.  Do not silently treat an
        # unrecognised provider termination as a successful final answer.
        return StopDecision(True, "unknown")
