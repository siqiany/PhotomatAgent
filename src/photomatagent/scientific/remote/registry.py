"""Persistent, idempotent job registry (SQLite) for remote VASP/SCNet work.

The registry is the local source of truth for every remote job lifecycle:
``request_id`` is the primary key and the idempotency anchor. A second
submission attempt with the same ``request_id`` can never create a second
remote job; ambiguous sbatch-client timeouts are recovered through
reconciliation (see :mod:`photomatagent.scientific.remote.lifecycle`).

Security contract: the registry stores no secrets. The remote request marker
payload contains only identifiers (request id, job name, content hash); it
never contains SSH keys or POTCAR content.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from photomatagent.scientific.remote.models import HPCJobState, ResourceRequest


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_KEEP_ERROR = object()  # sentinel: "do not touch the stored last_error"


class JobLifecycleState(str, Enum):
    """Unified, persisted job state: preparation, scheduler and validation.

    This superset of ``HPCJobState`` adds the lifecycle-only states that a
    Slurm status cannot express: prepared-but-not-gated, preflight passed,
    client-side submitting ambiguity, collected/validated results and the
    reconciliation-required trap state.
    """

    PREPARED = "PREPARED"
    PREFLIGHT_PASSED = "PREFLIGHT_PASSED"
    SUBMITTING = "SUBMITTING"
    SUBMITTED = "SUBMITTED"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    COLLECTED = "COLLECTED"
    VALIDATED = "VALIDATED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    CANCELLED = "CANCELLED"
    UNKNOWN_RECONCILIATION_REQUIRED = "UNKNOWN_RECONCILIATION_REQUIRED"

    @property
    def terminal(self) -> bool:
        """States after which the registry will not auto-resubmit."""
        return self in {
            JobLifecycleState.COMPLETED,
            JobLifecycleState.COLLECTED,
            JobLifecycleState.VALIDATED,
            JobLifecycleState.FAILED,
            JobLifecycleState.TIMEOUT,
            JobLifecycleState.OUT_OF_MEMORY,
            JobLifecycleState.CANCELLED,
            JobLifecycleState.UNKNOWN_RECONCILIATION_REQUIRED,
        }

    @property
    def requires_attention(self) -> bool:
        """States that need a human (or a deliberate new request) to proceed."""
        return self in {
            JobLifecycleState.FAILED,
            JobLifecycleState.TIMEOUT,
            JobLifecycleState.OUT_OF_MEMORY,
            JobLifecycleState.CANCELLED,
            JobLifecycleState.UNKNOWN_RECONCILIATION_REQUIRED,
        }


def lifecycle_from_hpc(state: HPCJobState | None) -> JobLifecycleState | None:
    """Map a scheduler state onto the lifecycle vocabulary."""
    if state is None:
        return None
    mapping = {
        HPCJobState.SUBMITTED: JobLifecycleState.SUBMITTED,
        HPCJobState.PENDING: JobLifecycleState.PENDING,
        HPCJobState.RUNNING: JobLifecycleState.RUNNING,
        HPCJobState.COMPLETED: JobLifecycleState.COMPLETED,
        HPCJobState.FAILED: JobLifecycleState.FAILED,
        HPCJobState.CANCELLED: JobLifecycleState.CANCELLED,
        HPCJobState.TIMEOUT: JobLifecycleState.TIMEOUT,
        HPCJobState.OUT_OF_MEMORY: JobLifecycleState.OUT_OF_MEMORY,
        HPCJobState.NODE_FAIL: JobLifecycleState.FAILED,
    }
    return mapping.get(state)  # HPCJobState.UNKNOWN -> None (never misjudged)


# -- canonical hashing ------------------------------------------------------


def canonical_input_hash(
    local_input_dir: str | Path,
    *,
    include_names: tuple[str, ...] = ("POSCAR", "INCAR", "KPOINTS"),
) -> str:
    """Deterministic SHA-256 over the sorted staged input file contents.

    The hash covers file names AND bytes so a regenerated POSCAR/INCAR makes
    the request materially different. Missing files hash as their name with
    an empty body marker, keeping the manifest hash stable for validation.
    """
    root = Path(local_input_dir).expanduser().resolve()
    digest = hashlib.sha256()
    for name in sorted(include_names):
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        path = root / name
        if path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(b"<missing>")
        digest.update(b"\x00")
    return digest.hexdigest()


def derive_request_id(
    application: str, workflow_stage: str, input_hash: str
) -> str:
    """Derive a stable, globally unique request id from the task manifest."""
    source = f"{application}|{workflow_stage}|{input_hash}".encode("utf-8")
    return hashlib.sha256(source).hexdigest()[:40]


_MARKER_FILENAME = "photomatagent.request.json"


def request_marker_payload(
    *,
    request_id: str,
    job_name: str,
    canonical_hash: str,
    remote_directory: str,
    attempt: int,
) -> dict[str, Any]:
    """Safe marker content written into the unique remote directory.

    Contains identifiers only. It deliberately excludes SSH keys, private
    paths and POTCAR contents; ``assert_marker_safe`` enforces this.
    """
    return {
        "kind": "photomatagent-request-marker",
        "version": 1,
        "request_id": request_id,
        "job_name": job_name,
        "canonical_input_hash": canonical_hash,
        "remote_directory": remote_directory,
        "attempt": attempt,
        "created_at": _now_iso(),
    }


def assert_marker_safe(payload: dict[str, Any]) -> None:
    """Reject any marker that would leak secrets or pseudopotential content."""
    text = json.dumps(payload, ensure_ascii=False)
    lowered = text.lower()
    for token in ("private_key", "id_rsa", "id_ed25519", ".ssh", "potcar"):
        if token in lowered:
            raise ValueError(f"request marker must not contain {token!r}")


# -- JobRecord / JobRegistry ------------------------------------------------


class JobRegistry:
    """SQLite-backed job registry with row-level invariants.

    Thread-safe within one process (a single connection guarded by a lock).
    Every state transition goes through :meth:`update`, which refuses to
    regress the recorded ``job_id`` and always bumps ``updated_at``.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(self.db_path), check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._create_schema()

    def _create_schema(self) -> None:
        with self._lock:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS job_registry (
                    request_id          TEXT PRIMARY KEY,
                    canonical_input_hash TEXT NOT NULL,
                    job_name            TEXT NOT NULL,
                    job_id              TEXT,
                    local_input_dir     TEXT,
                    remote_directory    TEXT,
                    workflow_stage      TEXT NOT NULL,
                    resource_json       TEXT NOT NULL,
                    scheduler_state     TEXT,
                    scientific_validation_state TEXT NOT NULL DEFAULT 'not_checked',
                    state               TEXT NOT NULL,
                    submitted_at        TEXT,
                    updated_at          TEXT NOT NULL,
                    completed_at        TEXT,
                    retry_count         INTEGER NOT NULL DEFAULT 0,
                    last_error          TEXT,
                    provenance_json     TEXT,
                    parent_request_id   TEXT
                )
                """
            )
            # Unique remote directories are an invariant: two jobs must never
            # write into the same remote directory.
            self._connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "idx_job_registry_remote_dir ON job_registry(remote_directory)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_job_registry_state "
                "ON job_registry(state)"
            )
            self._connection.commit()

    # -- CRUD ---------------------------------------------------------------

    def put(self, record: "JobRecord") -> None:
        """Insert or replace one record (caller constructs the full state).

        Upsert keyed on ``request_id`` only (an explicit
        ``ON CONFLICT(request_id) DO UPDATE``). ``INSERT OR REPLACE`` would
        silently DELETE another request's row when a different unique index
        (e.g. ``remote_directory``) collides -- that must raise instead.
        """
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO job_registry (
                    request_id, canonical_input_hash, job_name, job_id,
                    local_input_dir, remote_directory, workflow_stage,
                    resource_json, scheduler_state, scientific_validation_state,
                    state, submitted_at, updated_at, completed_at,
                    retry_count, last_error, provenance_json, parent_request_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(request_id) DO UPDATE SET
                    canonical_input_hash = excluded.canonical_input_hash,
                    job_name = excluded.job_name,
                    job_id = excluded.job_id,
                    local_input_dir = excluded.local_input_dir,
                    remote_directory = excluded.remote_directory,
                    workflow_stage = excluded.workflow_stage,
                    resource_json = excluded.resource_json,
                    scheduler_state = excluded.scheduler_state,
                    scientific_validation_state =
                        excluded.scientific_validation_state,
                    state = excluded.state,
                    submitted_at = excluded.submitted_at,
                    updated_at = excluded.updated_at,
                    completed_at = excluded.completed_at,
                    retry_count = excluded.retry_count,
                    last_error = excluded.last_error,
                    provenance_json = excluded.provenance_json,
                    parent_request_id = excluded.parent_request_id
                """,
                (
                    record.request_id,
                    record.canonical_input_hash,
                    record.job_name,
                    record.job_id,
                    record.local_input_dir,
                    record.remote_directory,
                    record.workflow_stage,
                    record.resource.model_dump_json(),
                    (
                        record.scheduler_state.value
                        if record.scheduler_state
                        else None
                    ),
                    record.scientific_validation_state,
                    record.state.value,
                    record.submitted_at,
                    record.updated_at,
                    record.completed_at,
                    record.retry_count,
                    record.last_error,
                    json.dumps(record.provenance, ensure_ascii=False),
                    record.parent_request_id,
                ),
            )
            self._connection.commit()

    def get(self, request_id: str) -> "JobRecord | None":
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM job_registry WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        return _row_to_record(row) if row is not None else None

    def list(self) -> list["JobRecord"]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM job_registry ORDER BY updated_at DESC"
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    def find_by_remote_directory(
        self, remote_directory: str, *, exclude_request_id: str | None = None
    ) -> "JobRecord | None":
        """Any other record claiming the same remote directory."""
        with self._lock:
            if exclude_request_id is None:
                row = self._connection.execute(
                    "SELECT * FROM job_registry WHERE remote_directory = ?",
                    (remote_directory,),
                ).fetchone()
            else:
                row = self._connection.execute(
                    "SELECT * FROM job_registry WHERE remote_directory = ? "
                    "AND request_id != ?",
                    (remote_directory, exclude_request_id),
                ).fetchone()
        return _row_to_record(row) if row is not None else None

    def update(
        self,
        request_id: str,
        *,
        state: JobLifecycleState | None = None,
        scheduler_state: HPCJobState | None = None,
        scientific_validation_state: str | None = None,
        job_id: str | None = None,
        remote_directory: str | None = None,
        submitted_at: str | None = None,
        completed_at: str | None = None,
        last_error: str | None = None,
        retry_count: int | None = None,
    ) -> "JobRecord | None":
        """Apply a guarded state transition; returns the updated record.

        ``last_error`` is written verbatim (``None`` clears it) so a success
        never keeps a stale error from an earlier failed attempt.
        """
        with self._lock:
            current = self.get(request_id)
            if current is None:
                return None
            if job_id is not None:
                # job_id must never regress: a submitted job stays identified.
                row = self._connection.execute(
                    "SELECT job_id FROM job_registry WHERE request_id = ?",
                    (request_id,),
                ).fetchone()
                if row["job_id"] and row["job_id"] != job_id:
                    raise ValueError(
                        f"refusing to change job_id for {request_id} "
                        f"from {row['job_id']} to {job_id}"
                    )
            next_state = state.value if state is not None else None
            next_sched = (
                scheduler_state.value if scheduler_state is not None else None
            )
            error_value = (
                current.last_error
                if last_error is _KEEP_ERROR
                else last_error
            )
            now = _now_iso()
            self._connection.execute(
                """
                UPDATE job_registry SET
                    state = COALESCE(?, state),
                    scheduler_state = COALESCE(?, scheduler_state),
                    scientific_validation_state =
                        COALESCE(?, scientific_validation_state),
                    job_id = COALESCE(?, job_id),
                    remote_directory = COALESCE(?, remote_directory),
                    submitted_at = COALESCE(?, submitted_at),
                    completed_at = COALESCE(?, completed_at),
                    last_error = ?,
                    retry_count = COALESCE(?, retry_count),
                    updated_at = ?
                WHERE request_id = ?
                """,
                (
                    next_state,
                    next_sched,
                    scientific_validation_state,
                    job_id,
                    remote_directory,
                    submitted_at,
                    completed_at,
                    error_value,
                    retry_count,
                    now,
                    request_id,
                ),
            )
            self._connection.commit()
            return self.get(request_id)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "JobRegistry":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


class JobRecord:
    """One persisted job row; constructed by the lifecycle, owned by registry."""

    def __init__(
        self,
        *,
        request_id: str,
        canonical_input_hash: str,
        job_name: str,
        workflow_stage: str,
        resource: ResourceRequest,
        state: JobLifecycleState,
        local_input_dir: str | Path | None = None,
        remote_directory: str | None = None,
        job_id: str | None = None,
        scheduler_state: HPCJobState | None = None,
        scientific_validation_state: str = "not_checked",
        submitted_at: str | None = None,
        updated_at: str | None = None,
        completed_at: str | None = None,
        retry_count: int = 0,
        last_error: str | None = None,
        provenance: dict[str, Any] | None = None,
        parent_request_id: str | None = None,
    ) -> None:
        self.request_id = request_id
        self.canonical_input_hash = canonical_input_hash
        self.job_name = job_name
        self.workflow_stage = workflow_stage
        self.resource = resource
        self.state = state
        self.local_input_dir = (
            str(local_input_dir) if local_input_dir is not None else None
        )
        self.remote_directory = remote_directory
        self.job_id = job_id
        self.scheduler_state = scheduler_state
        self.scientific_validation_state = scientific_validation_state
        self.submitted_at = submitted_at or _now_iso()
        self.updated_at = updated_at or _now_iso()
        self.completed_at = completed_at
        self.retry_count = int(retry_count)
        self.last_error = last_error
        self.provenance = dict(provenance or {})
        self.parent_request_id = parent_request_id

    def public_dict(self) -> dict[str, Any]:
        """Model-context-safe summary (no secrets by construction)."""
        return {
            "request_id": self.request_id,
            "canonical_input_hash": self.canonical_input_hash[:16] + "...",
            "job_name": self.job_name,
            "job_id": self.job_id,
            "local_input_dir": self.local_input_dir,
            "remote_directory": self.remote_directory,
            "workflow_stage": self.workflow_stage,
            "resource": self.resource.model_dump(mode="json"),
            "scheduler_state": (
                self.scheduler_state.value if self.scheduler_state else None
            ),
            "scientific_validation_state": self.scientific_validation_state,
            "state": self.state.value,
            "submitted_at": self.submitted_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "retry_count": self.retry_count,
            "last_error": self.last_error,
            "parent_request_id": self.parent_request_id,
        }


def _row_to_record(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        request_id=row["request_id"],
        canonical_input_hash=row["canonical_input_hash"],
        job_name=row["job_name"],
        workflow_stage=row["workflow_stage"],
        resource=ResourceRequest.model_validate_json(row["resource_json"]),
        state=JobLifecycleState(row["state"]),
        local_input_dir=row["local_input_dir"],
        remote_directory=row["remote_directory"],
        job_id=row["job_id"],
        scheduler_state=(
            HPCJobState(row["scheduler_state"])
            if row["scheduler_state"]
            else None
        ),
        scientific_validation_state=row["scientific_validation_state"],
        submitted_at=row["submitted_at"],
        updated_at=row["updated_at"],
        completed_at=row["completed_at"],
        retry_count=row["retry_count"],
        last_error=row["last_error"],
        provenance=(
            json.loads(row["provenance_json"])
            if row["provenance_json"]
            else {}
        ),
        parent_request_id=row["parent_request_id"],
    )


def default_registry_path() -> Path:
    """Default registry location (env-overridable, never inside a repo)."""
    base = os.environ.get("PHOTOMATAGENT_STATE_DIR", "~/.photomatagent")
    return Path(base).expanduser() / "jobs.sqlite3"
