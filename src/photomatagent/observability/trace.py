"""Typed loading and discovery for JSONL Agent Execution Traces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from photomatagent.logging.event_logger import default_sessions_dir
from photomatagent.runtime.events import RuntimeEvent, ToolRequested, parse_event


class TraceError(ValueError):
    """Raised when a trace cannot be located or parsed."""


@dataclass(frozen=True)
class AgentExecutionTrace:
    """One session's ordered, typed runtime events and source location."""

    session_id: str
    session_dir: Path
    events: tuple[RuntimeEvent, ...]

    @property
    def events_path(self) -> Path:
        return self.session_dir / "events.jsonl"

    @property
    def started_at(self) -> datetime | None:
        return self.events[0].timestamp if self.events else None

    @property
    def ended_at(self) -> datetime | None:
        return self.events[-1].timestamp if self.events else None


def list_session_paths(sessions_dir: Path | str | None = None) -> list[Path]:
    base = Path(sessions_dir) if sessions_dir is not None else default_sessions_dir()
    if not base.is_dir():
        return []
    return sorted(
        (
            path
            for path in base.iterdir()
            if path.is_dir()
            and (path / "events.jsonl").is_file()
            and (path / "events.jsonl").stat().st_size > 0
        ),
        key=lambda path: path.name,
        reverse=True,
    )


def resolve_session_path(
    target: str | Path, sessions_dir: Path | str | None = None
) -> Path:
    if str(target) == "latest":
        sessions = list_session_paths(sessions_dir)
        if not sessions:
            raise TraceError("no session traces found")
        return sessions[0]
    candidate = Path(target)
    if candidate.is_dir():
        return candidate
    base = Path(sessions_dir) if sessions_dir is not None else default_sessions_dir()
    candidate = base / str(target)
    if candidate.is_dir():
        return candidate
    raise TraceError(f"session not found: {target}")


def load_trace(
    target: str | Path, sessions_dir: Path | str | None = None
) -> AgentExecutionTrace:
    session_dir = resolve_session_path(target, sessions_dir)
    events_path = session_dir / "events.jsonl"
    if not events_path.is_file():
        raise TraceError(f"events.jsonl not found in {session_dir}")
    events: list[RuntimeEvent] = []
    current_iteration = 0
    pending_legacy_requests: list[ToolRequested] = []
    with events_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise TypeError("event must be a JSON object")
                kind = raw.get("kind")
                if kind == "loop_iteration_started":
                    current_iteration = int(raw.get("iteration") or current_iteration)
                if kind in {
                    "text_delta",
                    "tool_requested",
                    "tool_started",
                    "tool_completed",
                    "tool_failed",
                    "tool_permission_denied",
                }:
                    raw.setdefault("iteration", current_iteration)
                if kind == "tool_requested" and not raw.get("tool_call_id"):
                    raw["tool_call_id"] = f"legacy_pending_{line_number}"
                event = parse_event(raw)
                if isinstance(event, ToolRequested) and str(raw["tool_call_id"]).startswith(
                    "legacy_pending_"
                ):
                    pending_legacy_requests.append(event)
                if kind == "tool_started" and raw.get("tool_call_id"):
                    pending = next(
                        (
                            request
                            for request in pending_legacy_requests
                            if request.tool_name == raw.get("tool_name")
                        ),
                        None,
                    )
                    if pending is not None:
                        pending.tool_call_id = str(raw["tool_call_id"])
                        pending_legacy_requests.remove(pending)
                events.append(event)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise TraceError(
                    f"invalid trace event at {events_path}:{line_number}: {exc}"
                ) from exc
    session_id = next(
        (event.session_id for event in events if event.session_id), session_dir.name
    )
    for event in events:
        if event.session_id is None:
            event.session_id = session_id
    return AgentExecutionTrace(
        session_id=session_id,
        session_dir=session_dir,
        events=tuple(events),
    )
