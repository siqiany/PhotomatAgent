"""Projection of typed runtime events into durable workbench events."""

from __future__ import annotations

from typing import Any

from photomatagent.runtime.events import RuntimeEvent


def runtime_event_payload(event: RuntimeEvent) -> tuple[str, dict[str, object]]:
    """Return a JSON-safe, UI-neutral projection without changing the source event."""
    payload = event.model_dump(mode="json")
    payload.pop("kind", None)
    return event.kind, payload


def exception_payload(error: BaseException) -> dict[str, object]:
    """Keep the durable failure description bounded and free of tracebacks."""
    return {"error_type": type(error).__name__, "message": str(error)[:2000]}


def cost_payload(runtime: Any) -> dict[str, object]:
    """Snapshot only provider/runtime accounting that the runtime actually knows."""
    budget = getattr(runtime, "budget", None)
    snapshot = getattr(budget, "snapshot", None)
    if callable(snapshot):
        value = snapshot()
        if isinstance(value, dict):
            return dict(value)
    return {}
