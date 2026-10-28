"""BudgetState: lightweight accounting for the loop."""

from __future__ import annotations

from pydantic import BaseModel, Field

from photomatagent.models.types import ModelUsage


class BudgetState(BaseModel):
    """Tracks model calls, tool calls, iterations and optional cost/tokens.

    ``compute_cost`` is reserved for future scientific compute accounting
    (VASP / HPC hours); this phase only books the counters.
    """

    model_calls: int = 0
    tool_calls: int = 0
    iterations: int = 0
    # Effectively unbounded agent loop; CLI flags can still cap it explicitly.
    max_iterations: int = 10000
    max_total_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    compute_cost: float = 0.0

    def record_model_call(self, usage: ModelUsage | None = None, *, fallback_tokens: int = 0) -> None:
        self.model_calls += 1
        if usage is not None:
            self.input_tokens += usage.input_tokens
            self.output_tokens += usage.output_tokens
            reported_tokens = usage.resolved_total_tokens
            self.total_tokens += (
                max(reported_tokens, fallback_tokens)
                if self.max_total_tokens is not None and reported_tokens == 0
                else reported_tokens
            )
        elif self.max_total_tokens is not None:
            self.total_tokens += fallback_tokens

    def record_tool_call(self) -> None:
        self.tool_calls += 1

    def record_iteration(self) -> None:
        self.iterations += 1

    def iteration_limit_reached(self) -> bool:
        return self.iterations >= self.max_iterations

    def snapshot(self) -> dict[str, int | float]:
        return self.model_dump(exclude={"max_iterations", "max_total_tokens"})
