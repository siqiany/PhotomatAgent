"""Lightweight JSONL-derived session statistics (not a replay framework)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from photomatagent.logging.event_logger import default_sessions_dir


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    path: Path
    provider: str = "unknown"
    model: str = "unknown"
    iterations: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    permission_denials: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_seconds: float = 0.0


def list_sessions(sessions_dir: Path | str | None = None) -> list[Path]:
    base = Path(sessions_dir) if sessions_dir is not None else default_sessions_dir()
    if not base.is_dir():
        return []
    return sorted(
        (
            path
            for path in base.iterdir()
            if (path / "events.jsonl").is_file()
            and (path / "events.jsonl").stat().st_size > 0
        ),
        key=lambda path: path.name,
        reverse=True,
    )


def latest_session(sessions_dir: Path | str | None = None) -> Path | None:
    sessions = list_sessions(sessions_dir)
    return sessions[0] if sessions else None


def read_session_stats(session: Path) -> SessionSummary:
    event_path = session / "events.jsonl"
    events: list[dict[str, Any]] = []
    with event_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    iterations = max(
        (int(event.get("iteration", 0)) for event in events if event.get("kind") == "loop_iteration_started"),
        default=0,
    )
    model_events = [event for event in events if event.get("kind") == "model_response_completed"]
    input_tokens = sum(int((event.get("usage") or {}).get("input_tokens") or 0) for event in model_events)
    output_tokens = sum(int((event.get("usage") or {}).get("output_tokens") or 0) for event in model_events)
    started = next((event for event in events if event.get("kind") == "loop_started"), {})
    completed = next((event for event in reversed(events) if event.get("kind") in {"loop_completed", "loop_failed"}), {})
    duration_ms = completed.get("duration_ms")
    if duration_ms is None and len(events) >= 2:
        try:
            first = datetime.fromisoformat(str(events[0]["timestamp"]).replace("Z", "+00:00"))
            last = datetime.fromisoformat(str(events[-1]["timestamp"]).replace("Z", "+00:00"))
            duration_ms = (last - first).total_seconds() * 1000
        except (KeyError, ValueError):
            duration_ms = 0.0
    return SessionSummary(
        session_id=str(started.get("session_id") or session.name),
        path=session,
        provider=str(started.get("provider") or "unknown"),
        model=str(started.get("model") or "unknown"),
        iterations=iterations,
        model_calls=len([event for event in events if event.get("kind") == "model_request_started"]),
        tool_calls=len([event for event in events if event.get("kind") == "tool_started"]),
        tool_failures=len([event for event in events if event.get("kind") == "tool_failed"]),
        permission_denials=len(
            [event for event in events if event.get("kind") == "tool_permission_denied"]
        ),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration_seconds=float(duration_ms or 0.0) / 1000,
    )
