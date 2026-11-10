"""Switch a standalone chat's transcript, event sink and write ownership together."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import uuid4

from photomatagent.logging.event_logger import EventLogger
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.sessions.store import SessionSnapshot


def managed_task(directory: Path) -> str | None:
    database = directory.parent.parent / "workbench.sqlite3"
    if not database.is_file():
        return None
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as conn:
        row = conn.execute("SELECT task_id FROM tasks WHERE session_id=?", (directory.name,)).fetchone()
    return str(row[0]) if row else None


def attach_session(runtime: AgentRuntime, logger: EventLogger, directory: Path,
                   snapshot: SessionSnapshot | None = None) -> bool:
    """Prepare the new writer before releasing the old one; return whether forked.

    Imported read-only source files are never continued in place. Their state is
    copied into a new standalone session with a parent link instead.
    """
    from photomatagent.persistence.database import OwnershipLock
    from photomatagent.sessions.reader import database_session
    from photomatagent.sessions.sqlite_store import SessionRepository, RuntimePersistence
    from photomatagent.sessions.store import load_session_snapshot

    if snapshot is not None:
        runtime._validate_engine_snapshot(snapshot.conversation.messages, snapshot.engine)
    owner = None
    repository = None
    lock = None
    forked = False
    original = directory
    try:
        if directory.parent.resolve() != runtime.workspace.root / ".photomatagent/sessions":
            with database_session(directory) as external:
                if external is not None:
                    raise ValueError("CROSS_WORKSPACE_SESSION: open the session's workspace before continuing its database history")
        if directory.parent.resolve() == runtime.workspace.root / ".photomatagent/sessions":
            from photomatagent.application.store import WorkbenchStore
            owner = WorkbenchStore(runtime.workspace.root / ".photomatagent/workbench.sqlite3")
            if managed_task(directory):
                raise ValueError("MANAGED_SESSION: use /research --task to continue this task")
            linked = owner._connection.execute("SELECT 1 FROM links WHERE source_type='runtime_session' AND source_id=?", (directory.name,)).fetchone()
            if linked:
                snapshot = snapshot or load_session_snapshot(directory)
                directory = directory.parent / ("resume_" + uuid4().hex)
                forked = True
            repository = SessionRepository(owner.database)
        if logger.session_lock is not None and directory.resolve() == logger.session_dir.resolve():
            if snapshot is not None:
                runtime.restore_session(snapshot)
            if owner is not None:
                owner.close()
            return False
        lock = OwnershipLock(directory / ".session.lock")
        lock.acquire()
        if repository is not None:
            if repository.unresolved(directory.name):
                raise ValueError("RECONCILIATION_REQUIRED: session has unfinished tool operations")
            with database_session(directory) as prior:
                database_backed = prior is not None
            legacy = directory / "session_state.json"
            if legacy.is_file() and not database_backed:
                snapshot = snapshot or load_session_snapshot(directory)
                from photomatagent.observability.trace import load_trace
                events = load_trace(directory).events if (directory / "events.jsonl").is_file() else ()
                with repository.db.transaction():
                    repository.ensure(directory.name)
                    for event in events:
                        repository.event(directory.name, event)
                    repository.checkpoint(directory.name, snapshot, boundary="legacy_resume")
            else:
                repository.ensure(directory.name)
            if forked:
                with repository.db.transaction():
                    repository.ensure(original.name, backend="legacy_files")
                    repository.conn.execute("UPDATE sessions SET parent_session_id=? WHERE session_id=?", (original.name, directory.name))
                    assert snapshot is not None
                    repository.checkpoint(directory.name, snapshot, boundary="read_only_source_resume")
        if snapshot is not None:
            runtime.restore_session(snapshot)
        logger.close()
        logger.session_id = directory.name
        logger.session_dir = directory
        logger.events_path = directory / "events.jsonl"
        logger.repository, logger.storage_owner, logger.session_lock = repository, owner, lock
        runtime._session_id = directory.name
        runtime.persistence = RuntimePersistence(repository, directory.name, "standalone_" + uuid4().hex) if repository else None
        return forked
    except BaseException:
        if lock is not None:
            lock.release()
        if owner is not None:
            owner.close()
        raise
