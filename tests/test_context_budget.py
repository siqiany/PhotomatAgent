from __future__ import annotations

import pytest

from photomatagent.models.types import SystemMessage, ToolResultMessage, UserMessage
from photomatagent.runtime.context_budget import (
    ContextThresholds,
    PromptMeasure,
    account_context,
    measure_prompt,
    resolve_thresholds,
)
from photomatagent.runtime.context_engine import ContextEngineConfig
from photomatagent.tools.surface import ToolSurfaceStats


def _surface(*, visible_schema_chars: int = 0, manifest_chars: int = 0) -> ToolSurfaceStats:
    return ToolSurfaceStats(
        registered_tools=0,
        direct_tools=0,
        deferred_tools=0,
        hidden_tools=0,
        direct_schema_chars=0,
        deferred_schema_chars=0,
        bridge_schema_chars=0,
        manifest_chars=manifest_chars,
        visible_schema_chars=visible_schema_chars,
        estimated_direct_schema_tokens=0,
        estimated_deferred_schema_tokens=0,
        estimated_bridge_schema_tokens=0,
        estimated_manifest_tokens=0,
        estimated_visible_schema_tokens=0,
        estimated_avoided_tokens=0,
    )


@pytest.mark.parametrize(
    "window, hard, trigger, target",
    [
        (512_000, 495_616, 256_000, 128_000),
        (256_000, 239_616, 215_654, 128_000),
        (128_000, 111_616, 100_454, 60_272),
    ],
)
def test_effective_thresholds(window: int, hard: int, trigger: int, target: int) -> None:
    config = ContextEngineConfig(context_limit_tokens=window)
    actual = resolve_thresholds(config)
    assert actual == ContextThresholds(
        hard_input_limit=hard,
        compact_trigger=trigger,
        prune_trigger=trigger,
        target=target,
    )


def test_ratio_mode_thresholds_are_available_for_compatibility() -> None:
    config = ContextEngineConfig(
        context_limit_tokens=4_000,
        response_reserve_tokens=64,
        safety_margin_tokens=64,
        compact_trigger_tokens=None,
        prune_trigger_ratio=0.50,
        compact_trigger_ratio=0.90,
        target_ratio=0.40,
    )
    actual = resolve_thresholds(config)
    assert actual.prune_trigger == 2_000
    assert actual.compact_trigger == 3_484
    assert actual.target == 1_600


def test_context_config_rejects_unsafe_reserve_and_absolute_target() -> None:
    with pytest.raises(ValueError):
        ContextEngineConfig(context_limit_tokens=1_024)
    with pytest.raises(ValueError):
        ContextEngineConfig(
            context_limit_tokens=32_000,
            response_reserve_tokens=1_024,
            safety_margin_tokens=1_024,
            compact_trigger_tokens=6_000,
            compact_target_tokens=7_000,
        )


def test_measure_prompt_uses_single_ceil_for_total() -> None:
    messages = [SystemMessage(content="s"), UserMessage(content="u")]
    surface = _surface(visible_schema_chars=5)
    measured = measure_prompt(messages, surface)
    assert isinstance(measured, PromptMeasure)
    assert measured.chars == sum(
        len(message.model_dump_json()) for message in messages
    ) + 5
    assert measured.tokens == (measured.chars + 3) // 4
    assert measured.messages == 2


def test_account_context_uses_same_total_as_measure_prompt() -> None:
    messages = [
        UserMessage(content="hello"),
        ToolResultMessage(tool_call_id="c", tool_name="read", content="world"),
    ]
    surface = _surface(visible_schema_chars=11, manifest_chars=7)
    measured = measure_prompt(messages, surface)
    budget = account_context(messages, surface, model_context_limit=128_000)
    assert budget.estimated_current_prompt_tokens == measured.tokens
    assert budget.model_context_limit == 128_000
