"""Canonical, provider-neutral model protocol types.

Only provider adapters translate these models to/from vendor SDK objects.
Runtime, state, tools, logging, and CLI depend exclusively on this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, Field, SerializerFunctionWrapHandler, model_serializer


FinishReason = Literal["stop", "tool_calls", "max_tokens", "invalid_tool_markup", "cancelled", "error", "unknown"]


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


class ImageReference(BaseModel):
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: Literal["image/png", "image/jpeg", "image/webp"]
    byte_size: int = Field(gt=0, le=4 * 1024 * 1024)


class ProviderContinuation(BaseModel):
    protocol: Literal["deepseek_responses"] = "deepseek_responses"
    model: str
    endpoint_key: str = ""
    reasoning: tuple[str, ...] = Field(default=(), max_length=16, repr=False)


class UserMessage(BaseModel):
    kind: Literal["user"] = "user"
    content: str
    images: tuple[ImageReference, ...] = Field(default=(), max_length=4)
    context_only: bool = False

    @model_serializer(mode="wrap")
    def serialize_message(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        payload: dict[str, object] = handler(self)
        if not self.images:
            payload.pop("images", None)
        if not self.context_only:
            payload.pop("context_only", None)
        return payload


class AssistantMessage(BaseModel):
    kind: Literal["assistant"] = "assistant"
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    continuation: ProviderContinuation | None = Field(default=None, repr=False)

    @model_serializer(mode="wrap")
    def serialize_message(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        payload: dict[str, object] = handler(self)
        if self.continuation is None:
            payload.pop("continuation", None)
        return payload


class ToolResultMessage(BaseModel):
    kind: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    protected: bool = False

    images: tuple[ImageReference, ...] = Field(default=(), max_length=4)

    @model_serializer(mode="wrap")
    def serialize_message(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        payload: dict[str, object] = handler(self)
        if not self.images:
            payload.pop("images", None)
        return payload


ModelMessage = Annotated[
    Union[SystemMessage, UserMessage, AssistantMessage, ToolResultMessage],
    Field(discriminator="kind"),
]


def active_image_entries(messages: Sequence[ModelMessage]) -> dict[int, tuple[ImageReference, ...]]:
    """Select at most four distinct images for the active user turn."""
    last_user = max((i for i, message in enumerate(messages)
                     if isinstance(message, UserMessage) and not message.context_only), default=-1)
    order = ([last_user] if last_user >= 0 else []) + list(range(len(messages) - 1, last_user, -1))
    selected: dict[int, tuple[ImageReference, ...]] = {}
    seen: set[str] = set()
    for index in order:
        message = messages[index]
        if not isinstance(message, (UserMessage, ToolResultMessage)):
            continue
        if isinstance(message, UserMessage) and index != last_user:
            continue
        images = []
        for image in message.images:
            if image.sha256 not in seen and len(seen) < 4:
                seen.add(image.sha256)
                images.append(image)
        if images:
            selected[index] = tuple(images)
    return selected


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
    max_output_tokens: int | None = Field(default=None, ge=1)
    image_assets: dict[str, str] = Field(default_factory=dict, exclude=True, repr=False)


class ModelResponse(BaseModel):
    """Final aggregate emitted by every provider stream."""

    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: FinishReason = "unknown"
    usage: ModelUsage = Field(default_factory=ModelUsage)
    response_id: str | None = None
    continuation: ProviderContinuation | None = Field(default=None, repr=False)


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
