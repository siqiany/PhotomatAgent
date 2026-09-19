"""Approximate prompt component accounting and unified compaction thresholds."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import BaseModel

from photomatagent.models.types import ModelMessage, ToolResultMessage
from photomatagent.tools.surface import ToolSurfaceStats, estimate_tokens

if TYPE_CHECKING:
    from photomatagent.runtime.context_engine import ContextEngineConfig


class PromptMeasure(BaseModel):
    """One consistent character/token/message count for the rendered prompt."""

    chars: int
    tokens: int
    messages: int


class ContextThresholds(BaseModel):
    hard_input_limit: int
    compact_trigger: int
    prune_trigger: int
    target: int


class ContextBudget(BaseModel):
    model_context_limit: int | None = None
    estimated_current_prompt_tokens: int
    estimated_tool_schema_tokens: int
    estimated_manifest_tokens: int
    estimated_message_history_tokens: int
    estimated_tool_result_tokens: int


def measure_prompt(
    messages: list[ModelMessage], surface: ToolSurfaceStats
) -> PromptMeasure:
    """Measure the rendered prompt with the project's conservative estimate.

    The manifest is already part of the static system message, so it must not be
    added a second time.  The first version still uses ``ceil(chars / 4)`` as an
    estimate rather than a provider tokenizer.
    """
    chars = sum(
        len(
            json.dumps(
                message.model_dump(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        for message in messages
    )
    chars += surface.visible_schema_chars
    return PromptMeasure(
        chars=chars,
        tokens=estimate_tokens(chars),
        messages=len(messages),
    )


def resolve_thresholds(config: "ContextEngineConfig") -> ContextThresholds:
    """Resolve one consistent C/H/A/T policy for pruning and compaction."""
    hard = (
        config.context_limit_tokens
        - config.response_reserve_tokens
        - config.safety_margin_tokens
    )
    if config.compact_trigger_tokens is None:
        trigger = min(
            int(config.context_limit_tokens * config.compact_trigger_ratio),
            int(hard * 0.9),
        )
        prune = min(
            int(config.context_limit_tokens * config.prune_trigger_ratio),
            trigger,
        )
        target = min(
            int(config.context_limit_tokens * config.target_ratio),
            int(trigger * 0.6),
        )
    else:
        trigger = min(config.compact_trigger_tokens, int(hard * 0.9))
        prune = trigger
        target = min(config.compact_target_tokens, int(trigger * 0.6))
    return ContextThresholds(
        hard_input_limit=hard,
        compact_trigger=max(1, trigger),
        prune_trigger=max(1, prune),
        target=max(1, target),
    )


def account_context(
    messages: list[ModelMessage],
    surface: ToolSurfaceStats,
    *,
    model_context_limit: int | None = None,
) -> ContextBudget:
    message_chars = sum(
        len(
            json.dumps(
                message.model_dump(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
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
    total = measure_prompt(messages, surface)
    return ContextBudget(
        model_context_limit=model_context_limit,
        estimated_current_prompt_tokens=total.tokens,
        estimated_tool_schema_tokens=schema_tokens,
        estimated_manifest_tokens=manifest_tokens,
        estimated_message_history_tokens=history_tokens,
        estimated_tool_result_tokens=result_tokens,
    )
