"""Approximate prompt component accounting for future compaction policies."""

from __future__ import annotations

import json

from pydantic import BaseModel

from photomatagent.models.types import ModelMessage, ToolResultMessage
from photomatagent.tools.surface import ToolSurfaceStats, estimate_tokens


class ContextBudget(BaseModel):
    model_context_limit: int | None = None
    estimated_current_prompt_tokens: int
    estimated_tool_schema_tokens: int
    estimated_manifest_tokens: int
    estimated_message_history_tokens: int
    estimated_tool_result_tokens: int


def account_context(
    messages: list[ModelMessage],
    surface: ToolSurfaceStats,
    *,
    model_context_limit: int | None = None,
) -> ContextBudget:
    message_chars = sum(
        len(json.dumps(message.model_dump(), ensure_ascii=False, separators=(",", ":")))
        for message in messages
    )
    tool_result_chars = sum(
        len(message.content)
        for message in messages
        if isinstance(message, ToolResultMessage)
    )
    history_chars = max(0, message_chars - surface.manifest_chars - tool_result_chars)
    schema_tokens = surface.estimated_visible_schema_tokens
    manifest_tokens = surface.estimated_manifest_tokens
    history_tokens = estimate_tokens(history_chars)
    result_tokens = estimate_tokens(tool_result_chars)
    return ContextBudget(
        model_context_limit=model_context_limit,
        estimated_current_prompt_tokens=(
            schema_tokens + manifest_tokens + history_tokens + result_tokens
        ),
        estimated_tool_schema_tokens=schema_tokens,
        estimated_manifest_tokens=manifest_tokens,
        estimated_message_history_tokens=history_tokens,
        estimated_tool_result_tokens=result_tokens,
    )
