"""Resolve authoritative backend without silently falling back to stale exports."""
from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from photomatagent.persistence.database import Database
from photomatagent.sessions.sqlite_store import SessionRepository


@contextmanager
def database_session(directory: Path) -> Iterator[SessionRepository | None]:
    path = directory.parent.parent / "workbench.sqlite3"
    if not path.is_file():
        yield None
        return
    with sqlite3.connect(f"file:{path}?mode=ro",uri=True) as conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='sessions'").fetchone():
            yield None
            return
        row = conn.execute("SELECT backend FROM sessions WHERE session_id=?",(directory.name,)).fetchone()
    if not row or row[0]!="sqlite":
        yield None
        return
    database = Database(path)
    try:
        yield SessionRepository(database)
    finally:
        database.connection.close()
