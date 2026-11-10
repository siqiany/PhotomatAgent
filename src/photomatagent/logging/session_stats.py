"""Compatibility facade over the Loop Observatory trace analyzer."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import json

from photomatagent.observability.analyzer import SessionSummary, analyze_trace
from photomatagent.observability.trace import list_session_paths, load_trace


@dataclass(frozen=True)
class SessionOverview:
    provider: str = "unknown"
    model: str = "unknown"
    stop_reason: str | None = None


def read_session_overview(session: Path) -> SessionOverview:
    """Read list metadata without materializing every event in a conversation."""
    from photomatagent.sessions.reader import database_session
    from photomatagent.runtime.events import parse_event, LoopStarted, LoopCompleted

    first = None
    last = None
    with database_session(session) as repository:
        if repository is not None:
            with repository.db.lock:
                start = repository.conn.execute("SELECT payload_json FROM runtime_events WHERE session_id=? AND kind='loop_started' ORDER BY session_seq LIMIT 1", (session.name,)).fetchone()
                stop = repository.conn.execute("SELECT payload_json FROM runtime_events WHERE session_id=? AND kind='loop_completed' ORDER BY session_seq DESC LIMIT 1", (session.name,)).fetchone()
            first = json.loads(start[0]) if start else None
            last = json.loads(stop[0]) if stop else None
        else:
            events = session / "events.jsonl"
            if not events.is_file():
                if not (session / "session_state.json").is_file():
                    raise ValueError("session has neither a trace nor a snapshot")
            else:
                with events.open(encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        if not isinstance(item, dict):
                            raise ValueError("trace event must be an object")
                        if item.get("kind") == "loop_started" and first is None:
                            first = item
                        if item.get("kind") == "loop_completed":
                            last = item
    start_event = parse_event(first) if first else None
    stop_event = parse_event(last) if last else None
    return SessionOverview(
        provider=start_event.provider if isinstance(start_event, LoopStarted) else "unknown",
        model=start_event.model if isinstance(start_event, LoopStarted) else "unknown",
        stop_reason=stop_event.reason if isinstance(stop_event, LoopCompleted) else None,
    )


def list_sessions(sessions_dir: Path | str | None = None) -> list[Path]:
    return list_session_paths(sessions_dir)


def latest_session(sessions_dir: Path | str | None = None) -> Path | None:
    sessions = list_sessions(sessions_dir)
    return sessions[0] if sessions else None


def read_session_stats(session: Path | str) -> SessionSummary:
    return analyze_trace(load_trace(session))


__all__ = [
    "SessionSummary",
    "latest_session",
    "list_sessions",
    "read_session_stats",
    "read_session_overview",
]
