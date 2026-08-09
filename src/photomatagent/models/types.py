"""Provider-agnostic types for model input/output."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A tool invocation requested by the model."""

    id: str = Field(default_factory=lambda: f"call_{id(object())}")
    name: str
    arguments: dict[str, object] = Field(default_factory=dict)


class ModelUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ModelResponse(BaseModel):
    """Unified model output. No SDK-specific objects allowed here."""

    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str = "stop"
    usage: ModelUsage = Field(default_factory=ModelUsage)
