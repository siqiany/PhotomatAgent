"""SQLite configuration, reentrant transactions and process ownership."""
from __future__ import annotations

import fcntl
import os
import sqlite3
import threading
import tempfile
from contextlib import closing
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Any


class Database:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.connection = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA busy_timeout=5000")
        self._serial = 0

    def backup_once(self, destination: Path, version: int) -> None:
        """Publish only a verified backup; an interrupted temporary file is never reused."""
        if destination.exists():
            with closing(sqlite3.connect(f"file:{destination}?mode=ro", uri=True)) as existing:
                if existing.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise RuntimeError("MIGRATION_BACKUP_INVALID")
                saved = existing.execute("SELECT version FROM schema_version WHERE id=1").fetchone()
                if saved is None or saved[0] != version:
                    raise RuntimeError("MIGRATION_BACKUP_VERSION_CONFLICT")
            return
        descriptor, name = tempfile.mkstemp(prefix=".migration-backup-", dir=destination.parent)
        os.close(descriptor)
        temporary = Path(name)
        try:
            with self.lock, closing(sqlite3.connect(temporary)) as target:
                self.connection.backup(target)
                if target.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise RuntimeError("MIGRATION_BACKUP_INVALID")
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self.lock:
            nested = self.connection.in_transaction
            self._serial += 1
            name = f"transaction_{self._serial}"
            self.connection.execute(f"SAVEPOINT {name}" if nested else "BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                if nested:
                    self.connection.execute(f"ROLLBACK TO {name}")
                    self.connection.execute(f"RELEASE {name}")
                else:
                    self.connection.rollback()
                raise
            else:
                self.connection.execute(f"RELEASE {name}" if nested else "COMMIT")


class OwnershipLock:
    """OS-released nonblocking lock; never infer process liveness from a PID."""
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any = None

    def acquire(self) -> None:
        if self.handle is not None:
            raise RuntimeError("execution owner already acquired")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        handle = os.fdopen(descriptor, "a+")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError("workspace execution is owned by another process") from exc
        self.handle = handle

    def release(self) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None

    def __enter__(self) -> OwnershipLock:
        self.acquire()
        return self

    def __exit__(self, *args: object) -> None:
        self.release()
