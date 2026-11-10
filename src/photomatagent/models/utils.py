"""Small helpers shared by provider adapters."""

from __future__ import annotations

from typing import Any

from photomatagent.redaction import redact_text


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
    text = redact_text(str(exc))
    if len(text) > 1000:
        text = text[:1000] + "... [truncated]"
    return text.replace("Authorization", "[redacted-header]").replace("X-Api-Key", "[redacted-header]")


def provider_failure_code(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    if status == 429:
        return "RATE_LIMITED"
    if status in (401, 403):
        return "AUTHENTICATION_OR_ACCESS_DENIED"
    if status in (400, 422):
        return "REQUEST_INVALID"
    if type(status) is int and status >= 500:
        return "UPSTREAM_UNAVAILABLE"
    if isinstance(exc, TimeoutError) or "Timeout" in type(exc).__name__:
        return "NETWORK_TIMEOUT"
    if isinstance(exc, ConnectionError) or "Connection" in type(exc).__name__:
        return "NETWORK_UNAVAILABLE"
    return "PROVIDER_ERROR"
