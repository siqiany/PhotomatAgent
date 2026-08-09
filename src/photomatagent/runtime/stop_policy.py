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
        if not response.tool_calls:
            return StopDecision(True, "final_response")
        return StopDecision(False)
