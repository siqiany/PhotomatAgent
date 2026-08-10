"""Canonical, provider-neutral model protocol types.

Only provider adapters translate these models to/from vendor SDK objects.
Runtime, state, tools, logging, and CLI depend exclusively on this module.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, Field


FinishReason = Literal["stop", "tool_calls", "max_tokens", "cancelled", "error", "unknown"]


def _empty_schema() -> dict[str, object]:
    return {"type": "object", "properties": {}}


class ToolDefinition(BaseModel):
    name: str
    description: str = ""
    input_schema: dict[str, object] = Field(default_factory=_empty_schema)
    namespace: str = "core"


class ToolCall(BaseModel):
    """A complete tool invocation requested by a model.

    Provider-issued ids are preserved verbatim. The default exists only for
    deterministic/offline callers that do not supply one.
    """

    id: str = Field(default_factory=lambda: f"call_{uuid4().hex[:16]}")
    name: str
    arguments: dict[str, object] = Field(default_factory=dict)


class SystemMessage(BaseModel):
    kind: Literal["system"] = "system"
    content: str


class UserMessage(BaseModel):
    kind: Literal["user"] = "user"
    content: str


class AssistantMessage(BaseModel):
    kind: Literal["assistant"] = "assistant"
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ToolResultMessage(BaseModel):
    kind: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    protected: bool = False


ModelMessage = Annotated[
    Union[SystemMessage, UserMessage, AssistantMessage, ToolResultMessage],
    Field(discriminator="kind"),
]


class ModelUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int | None = None

    @property
    def resolved_total_tokens(self) -> int:
        return self.total_tokens if self.total_tokens is not None else self.input_tokens + self.output_tokens


class ModelRequest(BaseModel):
    messages: list[ModelMessage]
    tools: list[ToolDefinition] = Field(default_factory=list)


class ModelResponse(BaseModel):
    """Final aggregate emitted by every provider stream."""

    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: FinishReason = "unknown"
    usage: ModelUsage = Field(default_factory=ModelUsage)
    response_id: str | None = None


class ModelStreamStarted(BaseModel):
    kind: Literal["stream_started"] = "stream_started"
    provider: str
    model: str
    response_id: str | None = None


class ModelTextDelta(BaseModel):
    kind: Literal["text_delta"] = "text_delta"
    text: str


class ModelToolCallStarted(BaseModel):
    kind: Literal["tool_call_started"] = "tool_call_started"
    tool_call_id: str
    tool_name: str
    index: int


class ModelToolCallArgumentsDelta(BaseModel):
    kind: Literal["tool_call_arguments_delta"] = "tool_call_arguments_delta"
    tool_call_id: str
    delta: str
    index: int


class ModelToolCallCompleted(BaseModel):
    kind: Literal["tool_call_completed"] = "tool_call_completed"
    tool_call: ToolCall
    index: int


class ModelUsageUpdated(BaseModel):
    kind: Literal["usage_updated"] = "usage_updated"
    usage: ModelUsage


class ModelCompleted(BaseModel):
    kind: Literal["model_completed"] = "model_completed"
    response: ModelResponse


ModelStreamEvent = Annotated[
    Union[
        ModelStreamStarted,
        ModelTextDelta,
        ModelToolCallStarted,
        ModelToolCallArgumentsDelta,
        ModelToolCallCompleted,
        ModelUsageUpdated,
        ModelCompleted,
    ],
    Field(discriminator="kind"),
]
