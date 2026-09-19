"""Resolve public context-window policy values without touching secrets."""

from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic import ValidationError

from photomatagent.runtime.context_engine import ContextEngineConfig


ENV_CONTEXT_LIMIT = "PHOTOMATAGENT_CONTEXT_LIMIT_TOKENS"
ENV_COMPACT_TRIGGER = "PHOTOMATAGENT_COMPACT_TRIGGER_TOKENS"
ENV_COMPACT_TARGET = "PHOTOMATAGENT_COMPACT_TARGET_TOKENS"
ENV_RESPONSE_RESERVE = "PHOTOMATAGENT_RESPONSE_RESERVE_TOKENS"


def _read_positive_int(
    raw: str | None, name: str
) -> int | None:
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def resolve_context_config(
    *,
    context_limit_tokens: int | None = None,
    compact_trigger_tokens: int | None = None,
    environ: Mapping[str, str] | None = None,
) -> ContextEngineConfig:
    """Resolve explicit CLI values, then process env, then schema defaults.

    The resolver reads only the four public numeric context variables.  It never
    reads or prints API keys, model names, base URLs, or the full environment.
    """

    env = os.environ if environ is None else environ
    limit = (
        context_limit_tokens
        if context_limit_tokens is not None
        else _read_positive_int(env.get(ENV_CONTEXT_LIMIT), ENV_CONTEXT_LIMIT)
    )
    trigger = (
        compact_trigger_tokens
        if compact_trigger_tokens is not None
        else _read_positive_int(env.get(ENV_COMPACT_TRIGGER), ENV_COMPACT_TRIGGER)
    )
    target = _read_positive_int(env.get(ENV_COMPACT_TARGET), ENV_COMPACT_TARGET)
    reserve = _read_positive_int(env.get(ENV_RESPONSE_RESERVE), ENV_RESPONSE_RESERVE)

    data: dict[str, object] = {}
    if limit is not None:
        data["context_limit_tokens"] = limit
    if trigger is not None:
        data["compact_trigger_tokens"] = trigger
    if target is not None:
        data["compact_target_tokens"] = target
    if reserve is not None:
        data["response_reserve_tokens"] = reserve

    try:
        return ContextEngineConfig.model_validate(data)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        raise ValueError(f"invalid context policy configuration: {details}") from exc


__all__ = ["resolve_context_config"]
