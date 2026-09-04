"""Safe helpers for constructing bounded evolution event payloads."""

from __future__ import annotations

from photomatagent.runtime.events import EVOLUTION_SUMMARY_MAX_CHARS


def bounded_summary(value: str) -> str:
    """Return an event-safe summary capped at the protocol's character limit."""

    return value[:EVOLUTION_SUMMARY_MAX_CHARS]
