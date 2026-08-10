"""Safe, offline replay model built solely from recorded RuntimeEvents."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from photomatagent.observability.trace import AgentExecutionTrace
from photomatagent.runtime.events import (
    LoopCompleted,
    LoopFailed,
    LoopIterationStarted,
    LoopStarted,
    ModelRequestStarted,
    ModelResponseCompleted,
    ProviderFailed,
    TextDelta,
    ToolCompleted,
    ToolFailed,
    ToolPermissionDenied,
    ToolRequested,
)

ReplayKind = Literal[
    "goal",
    "iteration",
    "model",
    "model_text",
    "tool",
    "tool_result",
    "final_response",
    "stop",
    "error",
]


class ReplayItem(BaseModel):
    kind: ReplayKind
    iteration: int | None = None
    label: str
    content: str = ""
    metadata: dict[str, object] = Field(default_factory=dict)


class TraceReplay(BaseModel):
    session_id: str
    items: list[ReplayItem]


def build_replay(trace: AgentExecutionTrace) -> TraceReplay:
    """Build deterministic replay items; never executes tools or calls a model."""
    items: list[ReplayItem] = []
    text_by_iteration: dict[int, list[str]] = {}
    current_iteration: int | None = None
    for event in trace.events:
        if isinstance(event, LoopStarted):
            items.append(
                ReplayItem(
                    kind="goal",
                    label="USER GOAL",
                    content=event.goal,
                    metadata={
                        "provider": event.provider,
                        "model": event.model,
                        "run_id": event.run_id or "",
                    },
                )
            )
        elif isinstance(event, LoopIterationStarted):
            current_iteration = event.iteration
            items.append(
                ReplayItem(
                    kind="iteration",
                    iteration=event.iteration,
                    label=f"Iteration {event.iteration}",
                )
            )
        elif isinstance(event, ModelRequestStarted):
            items.append(
                ReplayItem(
                    kind="model",
                    iteration=event.iteration,
                    label="MODEL",
                    metadata={
                        "provider": event.provider,
                        "model": event.model,
                        "message_count": event.message_count,
                    },
                )
            )
        elif isinstance(event, TextDelta):
            text_by_iteration.setdefault(event.iteration, []).append(event.text)
        elif isinstance(event, ModelResponseCompleted):
            text = "".join(text_by_iteration.pop(event.iteration, []))
            if text:
                final = event.tool_call_count == 0
                items.append(
                    ReplayItem(
                        kind="final_response" if final else "model_text",
                        iteration=event.iteration,
                        label="FINAL RESPONSE" if final else "MODEL TEXT",
                        content=text,
                        metadata={
                            "finish_reason": event.finish_reason,
                            "duration_ms": event.duration_ms,
                            "usage": event.usage,
                        },
                    )
                )
        elif isinstance(event, ToolRequested):
            items.append(
                ReplayItem(
                    kind="tool",
                    iteration=event.iteration,
                    label=event.tool_name,
                    metadata={
                        "tool_call_id": event.tool_call_id,
                        "arguments": event.arguments,
                    },
                )
            )
        elif isinstance(event, ToolCompleted):
            items.append(
                ReplayItem(
                    kind="tool_result",
                    iteration=event.iteration,
                    label=f"{event.tool_name} result",
                    content=event.output,
                    metadata={
                        "tool_call_id": event.tool_call_id,
                        "status": event.tool_status,
                        "duration_ms": event.duration_ms,
                    },
                )
            )
        elif isinstance(event, ToolFailed):
            items.append(
                ReplayItem(
                    kind="error",
                    iteration=event.iteration,
                    label=f"{event.tool_name} failed",
                    content=event.error,
                    metadata={
                        "tool_call_id": event.tool_call_id,
                        "status": event.tool_status,
                        "duration_ms": event.duration_ms,
                        "error_type": event.error_type or "",
                    },
                )
            )
        elif isinstance(event, ToolPermissionDenied):
            items.append(
                ReplayItem(
                    kind="error",
                    iteration=event.iteration,
                    label=f"{event.tool_name} permission denied",
                    content=event.reason,
                    metadata={"tool_call_id": event.tool_call_id},
                )
            )
        elif isinstance(event, ProviderFailed):
            items.append(
                ReplayItem(
                    kind="error",
                    iteration=event.iteration,
                    label="PROVIDER FAILED",
                    content=event.error,
                )
            )
        elif isinstance(event, LoopCompleted):
            items.append(
                ReplayItem(
                    kind="stop",
                    iteration=current_iteration,
                    label="LOOP COMPLETED",
                    content=event.reason,
                    metadata={
                        "iterations": event.iterations,
                        "duration_ms": event.duration_ms,
                    },
                )
            )
        elif isinstance(event, LoopFailed):
            items.append(
                ReplayItem(
                    kind="error",
                    iteration=current_iteration,
                    label="LOOP FAILED",
                    content=event.error,
                    metadata={
                        "duration_ms": event.duration_ms,
                        "error_type": event.error_type or "",
                    },
                )
            )
    return TraceReplay(session_id=trace.session_id, items=items)
