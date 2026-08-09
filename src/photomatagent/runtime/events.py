"""RuntimeEvent hierarchy.

Every event is a Pydantic model, so each one can be serialized to JSON
(``model_dump_json``) for logging / replay, and can be routed through a
discriminated union for type-safe consumption.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RuntimeEvent(BaseModel):
    """Base class for all runtime events."""

    kind: str
    timestamp: datetime = Field(default_factory=_now)


class LoopStarted(RuntimeEvent):
    kind: Literal["loop_started"] = "loop_started"
    goal: str


class LoopIterationStarted(RuntimeEvent):
    kind: Literal["loop_iteration_started"] = "loop_iteration_started"
    iteration: int


class ModelRequestStarted(RuntimeEvent):
    kind: Literal["model_request_started"] = "model_request_started"
    iteration: int
    message_count: int


class TextDelta(RuntimeEvent):
    kind: Literal["text_delta"] = "text_delta"
    text: str


class ModelResponseCompleted(RuntimeEvent):
    kind: Literal["model_response_completed"] = "model_response_completed"
    iteration: int
    finish_reason: str
    tool_call_count: int
    usage: dict[str, int] = Field(default_factory=dict)


class ToolRequested(RuntimeEvent):
    kind: Literal["tool_requested"] = "tool_requested"
    tool_name: str
    arguments: dict[str, object] = Field(default_factory=dict)


class ToolApprovalRequired(RuntimeEvent):
    kind: Literal["tool_approval_required"] = "tool_approval_required"
    tool_name: str
    arguments: dict[str, object] = Field(default_factory=dict)
    reason: str = "permission policy requires approval"


class ToolStarted(RuntimeEvent):
    kind: Literal["tool_started"] = "tool_started"
    tool_name: str
    tool_call_id: str


class ToolCompleted(RuntimeEvent):
    kind: Literal["tool_completed"] = "tool_completed"
    tool_name: str
    tool_call_id: str
    output: str


class ToolFailed(RuntimeEvent):
    kind: Literal["tool_failed"] = "tool_failed"
    tool_name: str
    tool_call_id: str
    error: str


class ScientificStateUpdated(RuntimeEvent):
    kind: Literal["scientific_state_updated"] = "scientific_state_updated"
    summary: str


class BudgetUpdated(RuntimeEvent):
    kind: Literal["budget_updated"] = "budget_updated"
    model_calls: int
    tool_calls: int
    iteration: int


class LoopCompleted(RuntimeEvent):
    kind: Literal["loop_completed"] = "loop_completed"
    iterations: int
    reason: str


class LoopFailed(RuntimeEvent):
    kind: Literal["loop_failed"] = "loop_failed"
    error: str


#: Discriminated union over every event type, for type-safe consumption
#: and uniform JSON serialization.
AnyRuntimeEvent = Annotated[
    Union[
        LoopStarted,
        LoopIterationStarted,
        ModelRequestStarted,
        TextDelta,
        ModelResponseCompleted,
        ToolRequested,
        ToolApprovalRequired,
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


def parse_event(payload: dict[str, object]) -> RuntimeEvent:
    """Parse a serialized event payload back into a typed RuntimeEvent."""
    return TypeAdapter(AnyRuntimeEvent).validate_python(payload)
