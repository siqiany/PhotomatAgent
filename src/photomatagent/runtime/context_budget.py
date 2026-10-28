"""Approximate prompt component accounting and unified compaction thresholds."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any
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


def prompt_token_upper_bound(messages: Sequence[ModelMessage], tools: Sequence[Any]) -> int:
    """Return a conservative bound for model input tokens without a provider tokenizer.

    UTF-8 bytes bound byte-fallback tokenizers, while explicit framing headroom
    covers message and tool separators added by provider adapters. This is
    intentionally more conservative than the UI's chars/4 estimate.
    """
    serialized_messages = [message.model_dump(mode="json") for message in messages]
    serialized_tools = [
        item.model_dump(mode="json") if hasattr(item, "model_dump") else item
        for item in tools
    ]
    payload = json.dumps(
        {"messages": serialized_messages, "tools": serialized_tools},
        ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return len(payload) + 256 + 16 * (len(messages) + len(tools))


class PromptTokenEstimator:
    """Reuse measured input cost; charge changed content at its byte bound.

    This is an admission estimate, not a tokenizer or a guaranteed spend cap.
    Measurements stay local to this provider runtime. Unknown usage keeps the
    original byte bound. No durable message is shortened or rewritten here.
    """

    def __init__(self) -> None:
        self._parts: list[bytes] = []
        self._input_tokens = 0

    @property
    def calibrated(self) -> bool:
        return self._input_tokens > 0

    @staticmethod
    def _serialize(messages: Sequence[ModelMessage], tools: Sequence[Any]) -> list[bytes]:
        return [json.dumps(
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item,
            ensure_ascii=False, separators=(",", ":"), sort_keys=True,
        ).encode("utf-8") for item in [*messages, *tools]]

    def observe(self, messages: Sequence[ModelMessage], tools: Sequence[Any], *, input_tokens: int) -> None:
        if input_tokens > 0:
            self._parts = self._serialize(messages, tools)
            self._input_tokens = input_tokens

    def estimate(self, messages: Sequence[ModelMessage], tools: Sequence[Any]) -> int:
        bound = prompt_token_upper_bound(messages, tools)
        if self._input_tokens == 0:
            return bound
        # Stable message bodies dominate long research prompts. Retain the full
        # previous measured input even when content was removed, and account for
        # every added byte (including changes in tool schemas).
        previous = self._parts.copy()
        added = 0
        for part in self._serialize(messages, tools):
            if part in previous:
                previous.remove(part)
                continue
            # Scientific state system messages change each iteration. Count the
            # replacement bytes between their common prefix/suffix rather than
            # charging the unchanged evidence body as new content every time.
            best: tuple[int, bytes] | None = None
            for old in previous:
                prefix = 0
                limit = min(len(old), len(part))
                while prefix < limit and old[prefix] == part[prefix]:
                    prefix += 1
                suffix = 0
                while suffix < limit - prefix and old[-suffix - 1] == part[-suffix - 1]:
                    suffix += 1
                matched = prefix + suffix
                if matched > 64 and (best is None or matched > best[0]):
                    best = (matched, old)
            if best is None:
                added += len(part)
            else:
                added += len(part) - best[0] + 32  # retokenization/framing allowance
                previous.remove(best[1])
        calibrated = math.ceil(self._input_tokens * 1.25) + added + 512
        return min(bound, calibrated)


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
