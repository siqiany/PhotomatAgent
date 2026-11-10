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
    paths = {path for path in base.iterdir() if path.is_dir() and not path.is_symlink() and ((path / "session_state.json").is_file() or ((path / "events.jsonl").is_file() and (path / "events.jsonl").stat().st_size > 0))} if base.is_dir() else set()
    activity = {path:max((file.stat().st_mtime for file in (path/"session_state.json",path/"events.jsonl") if file.is_file()),default=0) for path in paths}
    database_path = base.parent / "workbench.sqlite3"
    if database_path.is_file():
        import sqlite3
        with sqlite3.connect(f"file:{database_path}?mode=ro",uri=True) as conn:
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name='sessions'").fetchone():
                for row in conn.execute("SELECT session_id,created_at,(SELECT payload_json FROM runtime_events WHERE runtime_events.session_id=sessions.session_id ORDER BY session_seq DESC LIMIT 1) FROM sessions WHERE backend='sqlite' AND (latest_checkpoint_id IS NOT NULL OR EXISTS(SELECT 1 FROM runtime_events WHERE runtime_events.session_id=sessions.session_id))"):
                    if Path(row[0]).name == row[0] and row[0] not in {".", ".."}:
                        path = base / row[0]
                        paths.add(path)
                        try:
                            # Import time is not conversation activity. Preserve
                            # original event time even when task IDs are random.
                            timestamp = json.loads(row[2])["timestamp"] if row[2] else None
                            activity[path] = datetime.fromisoformat(timestamp).timestamp() if timestamp else activity.get(path,datetime.fromisoformat(row[1]).timestamp())
                        except (ValueError,TypeError,KeyError):
                            activity.setdefault(path,0)
    return sorted(paths,key=lambda path:(activity.get(path,0),path.name),reverse=True)


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
    if candidate.parent.name == "sessions":
        from photomatagent.sessions.reader import database_session
        with database_session(candidate) as repository:
            if repository is not None:
                return candidate
    base = Path(sessions_dir) if sessions_dir is not None else default_sessions_dir()
    candidate = base / str(target)
    if candidate.is_dir():
        return candidate
    if candidate in list_session_paths(base):
        return candidate
    raise TraceError(f"session not found: {target}")


def load_trace(
    target: str | Path, sessions_dir: Path | str | None = None
) -> AgentExecutionTrace:
    session_dir = resolve_session_path(target, sessions_dir)
    from photomatagent.sessions.reader import database_session
    with database_session(session_dir) as repository:
        if repository is not None:
            with repository.db.lock:
                rows = repository.conn.execute("SELECT payload_json FROM runtime_events WHERE session_id=? ORDER BY session_seq", (session_dir.name,)).fetchall()
            return AgentExecutionTrace(session_id=session_dir.name,session_dir=session_dir,events=tuple(parse_event(json.loads(row[0])) for row in rows))
    events_path = session_dir / "events.jsonl"
    if not events_path.is_file():
        if (session_dir / "session_state.json").is_file():
            return AgentExecutionTrace(session_id=session_dir.name, session_dir=session_dir, events=())
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
