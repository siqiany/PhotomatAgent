"""Small helpers shared by provider adapters."""

from __future__ import annotations

from typing import Any


def as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="python")
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"cannot convert {type(value).__name__} to mapping")


def safe_provider_message(exc: Exception) -> str:
    """Keep errors useful without echoing credentials or request headers."""
    text = str(exc)
    if len(text) > 1000:
        text = text[:1000] + "... [truncated]"
    return text.replace("Authorization", "[redacted-header]").replace("X-Api-Key", "[redacted-header]")
