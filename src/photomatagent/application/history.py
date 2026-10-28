"""Read-only import of existing runtime-session history into the workbench index."""

from __future__ import annotations

import hashlib
from pathlib import Path

from photomatagent.application.models import TaskCreate
from photomatagent.application.store import WorkbenchStore
from photomatagent.observability.trace import load_trace
from photomatagent.runtime.events import LoopStarted
from photomatagent.sessions.store import snapshot_path


class HistoricalWorkbenchImporter:
    def __init__(self, store: WorkbenchStore) -> None:
        self.store = store

    def import_session(self, session_dir: Path | str) -> str:
        """Index a session without modifying its source files or scientific state."""
        events_path = Path(session_dir) / "events.jsonl"
        source = events_path.read_bytes()
        snapshot_file = snapshot_path(events_path.parent)
        snapshot_digest = (
            hashlib.sha256(snapshot_file.read_bytes()).hexdigest()
            if snapshot_file.is_file()
            else None
        )
        trace = load_trace(events_path.parent, sessions_dir=events_path.parent.parent)
        goals = [
            event.goal
            for event in trace.events
            if isinstance(event, LoopStarted) and event.goal.strip()
        ]
        if not goals:
            raise ValueError("historical session has no recorded goal")
        original_goal = goals[-1].strip()
        first_line = next(line.strip() for line in original_goal.splitlines() if line.strip())
        title = first_line if len(first_line) <= 120 else first_line[:119].rstrip() + "…"
        suffix = "…（完整原文保留在只读历史来源中）"
        goal = (
            original_goal
            if len(original_goal) <= 20_000
            else original_goal[: 20_000 - len(suffix)].rstrip() + suffix
        )
        return self.store.create_historical_task(
            request=TaskCreate(title=title, goal=goal),
            source_type="runtime_session",
            source_id=trace.session_id,
            source_digest=hashlib.sha256(source).hexdigest(),
            snapshot_digest=snapshot_digest,
        )


__all__ = ["HistoricalWorkbenchImporter"]
