"""Serializable event protocol emitted by the PhotomatAgent runtime."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RuntimeEvent(BaseModel):
    kind: str
    timestamp: datetime = Field(default_factory=_now)
    session_id: str | None = None


class LoopStarted(RuntimeEvent):
    kind: Literal["loop_started"] = "loop_started"
    goal: str
    provider: str
    model: str
    workspace: str


class LoopIterationStarted(RuntimeEvent):
    kind: Literal["loop_iteration_started"] = "loop_iteration_started"
    iteration: int


class ModelRequestStarted(RuntimeEvent):
    kind: Literal["model_request_started"] = "model_request_started"
    iteration: int
    message_count: int
    provider: str
    model: str


class ModelStreamStarted(RuntimeEvent):
    kind: Literal["model_stream_started"] = "model_stream_started"
    iteration: int
    provider: str
    model: str
    response_id: str | None = None


class TextDelta(RuntimeEvent):
    kind: Literal["text_delta"] = "text_delta"
    iteration: int
    text: str


class ToolCallStarted(RuntimeEvent):
    kind: Literal["tool_call_started"] = "tool_call_started"
    iteration: int
    tool_call_id: str
    tool_name: str
    index: int


class ToolCallArgumentsDelta(RuntimeEvent):
    kind: Literal["tool_call_arguments_delta"] = "tool_call_arguments_delta"
    iteration: int
    tool_call_id: str
    delta: str
    index: int


class ToolCallCompleted(RuntimeEvent):
    kind: Literal["tool_call_completed"] = "tool_call_completed"
    iteration: int
    tool_call_id: str
    tool_name: str
    arguments: dict[str, object]
    index: int


class ModelResponseCompleted(RuntimeEvent):
    kind: Literal["model_response_completed"] = "model_response_completed"
    iteration: int
    provider: str
    model: str
    response_id: str | None = None
    finish_reason: str
    tool_call_count: int
    usage: dict[str, int | None] = Field(default_factory=dict)
    duration_ms: float


class ProviderFailed(RuntimeEvent):
    kind: Literal["provider_failed"] = "provider_failed"
    iteration: int
    provider: str
    model: str
    error: str


class ToolRequested(RuntimeEvent):
    kind: Literal["tool_requested"] = "tool_requested"
    iteration: int
    tool_call_id: str
    tool_name: str
    arguments: dict[str, object] = Field(default_factory=dict)


class ToolApprovalRequired(RuntimeEvent):
    kind: Literal["tool_approval_required"] = "tool_approval_required"
    iteration: int
    tool_call_id: str
    tool_name: str
    arguments: dict[str, object] = Field(default_factory=dict)
    reason: str = "permission policy requires approval"


class ToolPermissionDenied(RuntimeEvent):
    kind: Literal["tool_permission_denied"] = "tool_permission_denied"
    iteration: int
    tool_call_id: str
    tool_name: str
    reason: str


class ToolStarted(RuntimeEvent):
    kind: Literal["tool_started"] = "tool_started"
    iteration: int
    tool_name: str
    tool_call_id: str


class ToolCompleted(RuntimeEvent):
    kind: Literal["tool_completed"] = "tool_completed"
    iteration: int
    tool_name: str
    tool_call_id: str
    output: str
    duration_ms: float


class ToolFailed(RuntimeEvent):
    kind: Literal["tool_failed"] = "tool_failed"
    iteration: int
    tool_name: str
    tool_call_id: str
    error: str
    duration_ms: float = 0.0


class ScientificStateUpdated(RuntimeEvent):
    kind: Literal["scientific_state_updated"] = "scientific_state_updated"
    summary: str


class BudgetUpdated(RuntimeEvent):
    kind: Literal["budget_updated"] = "budget_updated"
    model_calls: int
    tool_calls: int
    iteration: int
    input_tokens: int = 0
    output_tokens: int = 0


class LoopCompleted(RuntimeEvent):
    kind: Literal["loop_completed"] = "loop_completed"
    iterations: int
    reason: str
    duration_ms: float


class LoopFailed(RuntimeEvent):
    kind: Literal["loop_failed"] = "loop_failed"
    error: str
    duration_ms: float


AnyRuntimeEvent = Annotated[
    Union[
        LoopStarted,
        LoopIterationStarted,
        ModelRequestStarted,
        ModelStreamStarted,
        TextDelta,
        ToolCallStarted,
        ToolCallArgumentsDelta,
        ToolCallCompleted,
        ModelResponseCompleted,
        ProviderFailed,
        ToolRequested,
        ToolApprovalRequired,
        ToolPermissionDenied,
        ToolStarted,
        ToolCompleted,
        ToolFailed,
        ScientificStateUpdated,
        BudgetUpdated,
        LoopCompleted,
        LoopFailed,
    ],
    Field(discriminator="kind"),
]

_EVENT_ADAPTER: TypeAdapter[AnyRuntimeEvent] = TypeAdapter(AnyRuntimeEvent)


def parse_event(payload: dict[str, object]) -> RuntimeEvent:
    return _EVENT_ADAPTER.validate_python(payload)
