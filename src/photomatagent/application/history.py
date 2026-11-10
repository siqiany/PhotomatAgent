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
        from photomatagent.sessions.reader import database_session
        with database_session(Path(session_dir)) as repository:
            if repository is not None:
                if repository.db.path.resolve() != self.store.database_path.resolve():
                    raise ValueError("database history must belong to the same workspace")
                checkpoints = repository.checkpoints(Path(session_dir).name, 1)
                if not checkpoints:
                    raise ValueError("database session has no safe checkpoint")
                checkpoint = checkpoints[0]
                snapshot = repository.load(Path(session_dir).name, checkpoint["checkpoint_id"])
                goal = snapshot.scientific.goal.strip()
                if not goal:
                    from photomatagent.models.types import UserMessage
                    goal = next((item.content for item in snapshot.conversation.messages if isinstance(item, UserMessage) and item.content.strip()), "")
                if not goal:
                    raise ValueError("historical session has no recorded goal")
                title = goal.splitlines()[0][:120]
                return self.store.create_historical_task(request=TaskCreate(title=title,goal=goal[:20_000]),source_type="sqlite_session",source_id=checkpoint["checkpoint_id"],source_digest=checkpoint["sha256"],snapshot_digest=hashlib.sha256(snapshot.model_dump_json().encode()).hexdigest())
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
        if snapshot_file.is_file():
            from photomatagent.sessions.store import parse_session_snapshot
            snapshot = parse_session_snapshot(snapshot_file.read_bytes())
            original_goal = snapshot.scientific.goal.strip() or original_goal
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
