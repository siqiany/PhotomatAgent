"""SQLite index for workbench tasks without replacing scientific source stores."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from functools import wraps
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

from photomatagent.application.models import (
    ResearchTask,
    RunRecord,
    RunRef,
    RunRequest,
    TaskCreate,
    WorkbenchEvent,
    WorkbenchLink,
    WorkbenchRevisionPlan,
    FeedbackRequest,
    RunRef,
)

_ACTIVE_STATES = ("QUEUED", "RUNNING", "WAITING_INPUT", "CANCEL_REQUESTED")
_TERMINAL_STATES = ("SUCCEEDED", "PARTIAL", "FAILED", "CANCELLED", "INTERRUPTED")
_SCHEMA_VERSION = 8
_StoreMethod = TypeVar("_StoreMethod", bound=Callable[..., Any])


def _serialized(method: _StoreMethod) -> _StoreMethod:
    """Serialize access to the one cross-thread SQLite connection."""

    @wraps(method)
    def wrapped(self: WorkbenchStore, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return cast(_StoreMethod, wrapped)


class WorkbenchStoreError(RuntimeError):
    """Base error for workbench persistence operations."""


class IdempotencyConflict(WorkbenchStoreError):
    """A request key was reused for a different request body."""


class ActiveRunConflict(WorkbenchStoreError):
    """The task already has an active writing run."""


class HistoricalSourceConflict(WorkbenchStoreError):
    """A historical source ID was changed after it had been imported."""


class WorkbenchStore:
    """A small durable index, intentionally separate from session/evolution stores."""

    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.database_path, isolation_level=None, check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA busy_timeout = 5000")
        self._initialize()

    @_serialized
    def close(self) -> None:
        self._connection.close()

    def _initialize(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                version INTEGER NOT NULL
            );
            INSERT OR IGNORE INTO schema_version (id, version) VALUES (1, 8);

            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                goal TEXT NOT NULL,
                attachment_ids_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL REFERENCES tasks(task_id),
                request_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                finished_at TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_run_per_task
                ON runs(task_id)
                WHERE status IN ('QUEUED', 'RUNNING', 'WAITING_INPUT', 'CANCEL_REQUESTED');
            CREATE TABLE IF NOT EXISTS links (
                task_id TEXT NOT NULL REFERENCES tasks(task_id),
                source_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                source_digest TEXT NOT NULL,
                snapshot_digest TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY (source_type, source_id)
            );
            CREATE TABLE IF NOT EXISTS events (
                task_id TEXT NOT NULL REFERENCES tasks(task_id),
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                seq INTEGER NOT NULL,
                type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (task_id, seq)
            );
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL REFERENCES tasks(task_id),
                run_id TEXT REFERENCES runs(run_id),
                path TEXT NOT NULL,
                filename TEXT NOT NULL DEFAULT '',
                sha256 TEXT NOT NULL,
                media_type TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS request_keys (
                request_key TEXT PRIMARY KEY,
                task_id TEXT NOT NULL REFERENCES tasks(task_id),
                operation TEXT NOT NULL,
                body_sha256 TEXT NOT NULL,
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS revision_feedback (
                feedback_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL REFERENCES tasks(task_id),
                source_run_id TEXT NOT NULL REFERENCES runs(run_id),
                source_digest TEXT NOT NULL,
                source_artifacts_json TEXT NOT NULL,
                feedback_text TEXT NOT NULL,
                feedback_sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (task_id, feedback_sha256)
            );
            CREATE TABLE IF NOT EXISTS revision_plans (
                plan_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL REFERENCES tasks(task_id),
                feedback_id TEXT NOT NULL REFERENCES revision_feedback(feedback_id),
                source_run_id TEXT NOT NULL REFERENCES runs(run_id),
                source_digest TEXT NOT NULL,
                attempt INTEGER NOT NULL CHECK (attempt >= 1),
                status TEXT NOT NULL,
                plan_json TEXT NOT NULL,
                error TEXT,
                prepare_key TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (task_id, feedback_id, attempt)
            );
            CREATE TABLE IF NOT EXISTS revision_runs (
                plan_id TEXT NOT NULL REFERENCES revision_plans(plan_id),
                task_id TEXT NOT NULL REFERENCES tasks(task_id),
                run_id TEXT NOT NULL UNIQUE REFERENCES runs(run_id),
                attempt INTEGER NOT NULL CHECK (attempt >= 1),
                request_key TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                PRIMARY KEY (plan_id, attempt)
            );
            CREATE TABLE IF NOT EXISTS revision_compiler_cost (
                plan_id TEXT PRIMARY KEY REFERENCES revision_plans(plan_id),
                cost_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS evaluations (
                evaluation_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL REFERENCES tasks(task_id),
                run_id TEXT NOT NULL UNIQUE REFERENCES runs(run_id),
                manifest_json TEXT NOT NULL,
                status TEXT NOT NULL,
                report_json TEXT,
                error_code TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        version = self._connection.execute(
            "SELECT version FROM schema_version WHERE id = 1"
        ).fetchone()["version"]
        if version < 2:
            self._migrate_request_keys()
        if version < 3:
            self._migrate_task_session_ids()
        if version < 4:
            self._migrate_artifact_filenames()
        if version < 5:
            self._migrate_link_snapshot_digests()
        if version < 6:
            self._migrate_revisions()
        if version < 7:
            self._migrate_evaluations()
        if version < 8:
            self._migrate_revision_run_attempts()

    def _migrate_task_session_ids(self) -> None:
        with self._transaction():
            self._connection.execute("ALTER TABLE tasks ADD COLUMN session_id TEXT")
            self._connection.execute("UPDATE tasks SET session_id = task_id")
            self._connection.execute("CREATE UNIQUE INDEX task_session_id ON tasks(session_id)")
            self._connection.execute("UPDATE schema_version SET version = 3 WHERE id = 1")

    def _migrate_artifact_filenames(self) -> None:
        with self._transaction():
            columns = {
                str(row["name"])
                for row in self._connection.execute("PRAGMA table_info(artifacts)")
            }
            if "filename" not in columns:
                self._connection.execute(
                    "ALTER TABLE artifacts ADD COLUMN filename TEXT NOT NULL DEFAULT ''"
                )
            self._connection.execute("UPDATE schema_version SET version = 4 WHERE id = 1")

    def _migrate_link_snapshot_digests(self) -> None:
        with self._transaction():
            columns = {
                str(row["name"])
                for row in self._connection.execute("PRAGMA table_info(links)")
            }
            if "snapshot_digest" not in columns:
                self._connection.execute("ALTER TABLE links ADD COLUMN snapshot_digest TEXT")
            self._connection.execute("UPDATE schema_version SET version = 5 WHERE id = 1")

    def _migrate_revisions(self) -> None:
        with self._transaction():
            self._connection.execute("UPDATE schema_version SET version = 6 WHERE id = 1")

    def _migrate_evaluations(self) -> None:
        with self._transaction():
            self._connection.execute("UPDATE schema_version SET version = 7 WHERE id = 1")

    def _migrate_revision_run_attempts(self) -> None:
        with self._transaction():
            self._connection.execute("ALTER TABLE revision_runs RENAME TO revision_runs_v7")
            self._connection.execute(
                "CREATE TABLE revision_runs ("
                "plan_id TEXT NOT NULL REFERENCES revision_plans(plan_id), "
                "task_id TEXT NOT NULL REFERENCES tasks(task_id), "
                "run_id TEXT NOT NULL UNIQUE REFERENCES runs(run_id), "
                "attempt INTEGER NOT NULL CHECK (attempt >= 1), "
                "request_key TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL, "
                "PRIMARY KEY (plan_id, attempt))"
            )
            self._connection.execute(
                "INSERT INTO revision_runs (plan_id, task_id, run_id, attempt, request_key, created_at) "
                "SELECT plan_id, task_id, run_id, 1, request_key, created_at FROM revision_runs_v7"
            )
            self._connection.execute("DROP TABLE revision_runs_v7")
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS revision_compiler_cost ("
                "plan_id TEXT PRIMARY KEY REFERENCES revision_plans(plan_id), cost_json TEXT NOT NULL)"
            )
            self._connection.execute("UPDATE schema_version SET version = 8 WHERE id = 1")

    def _migrate_request_keys(self) -> None:
        """Make request keys global without accepting ambiguous old records."""
        with self._transaction():
            legacy_rows = self._connection.execute(
                "SELECT task_id, operation, request_key, body_sha256, run_id, created_at FROM request_keys"
            ).fetchall()
            normalized: dict[str, sqlite3.Row] = {}
            for row in legacy_rows:
                request_key = str(row["request_key"]).strip()
                if not request_key:
                    raise WorkbenchStoreError("cannot migrate blank request key")
                existing = normalized.get(request_key)
                identity = (
                    row["task_id"], row["operation"], row["body_sha256"], row["run_id"]
                )
                if existing is not None:
                    existing_identity = (
                        existing["task_id"],
                        existing["operation"],
                        existing["body_sha256"],
                        existing["run_id"],
                    )
                    if existing_identity != identity:
                        raise WorkbenchStoreError(
                            f"normalized request key collision for {request_key!r}"
                        )
                    continue
                normalized[request_key] = row
            self._connection.execute("ALTER TABLE request_keys RENAME TO request_keys_v1")
            self._connection.execute(
                """
                CREATE TABLE request_keys (
                    request_key TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES tasks(task_id),
                    operation TEXT NOT NULL,
                    body_sha256 TEXT NOT NULL,
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    created_at TEXT NOT NULL
                )
                """
            )
            for request_key, row in normalized.items():
                self._connection.execute(
                    "INSERT INTO request_keys VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        request_key,
                        row["task_id"],
                        row["operation"],
                        row["body_sha256"],
                        row["run_id"],
                        row["created_at"],
                    ),
                )
            self._connection.execute("DROP TABLE request_keys_v1")
            self._connection.execute(
                "UPDATE schema_version SET version = 2 WHERE id = 1"
            )

    @_serialized
    def create_task(self, request: TaskCreate) -> str:
        task_id = f"task_{uuid.uuid4().hex}"
        now = _now()
        self._connection.execute(
            "INSERT INTO tasks (task_id, session_id, title, goal, attachment_ids_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (task_id, task_id, request.title, request.goal, _json(request.attachment_ids), now),
        )
        return task_id

    @_serialized
    def create_historical_task(
        self,
        *,
        request: TaskCreate,
        source_type: str,
        source_id: str,
        source_digest: str,
        snapshot_digest: str | None = None,
    ) -> str:
        """Create or return one task linked to an immutable imported source."""
        with self._transaction():
            existing = self._connection.execute(
                "SELECT task_id, source_digest, snapshot_digest FROM links WHERE source_type = ? AND source_id = ?",
                (source_type, source_id),
            ).fetchone()
            if existing is not None:
                if existing["source_digest"] != source_digest or (
                    existing["snapshot_digest"] is not None
                    and existing["snapshot_digest"] != snapshot_digest
                ):
                    raise HistoricalSourceConflict("historical source digest changed")
                return str(existing["task_id"])
            task_id = f"task_{uuid.uuid4().hex}"
            now = _now()
            self._connection.execute(
                "INSERT INTO tasks (task_id, session_id, title, goal, attachment_ids_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (task_id, task_id, request.title, request.goal, _json(request.attachment_ids), now),
            )
            self._connection.execute(
                "INSERT INTO links (task_id, source_type, source_id, source_digest, snapshot_digest, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (task_id, source_type, source_id, source_digest, snapshot_digest, now),
            )
            return task_id

    @_serialized
    def reserve_run(
        self,
        task_id: str,
        request: RunRequest,
        request_key: str,
        *,
        operation: str = "reserve_run",
        idempotency_binding: object | None = None,
    ) -> RunRef:
        request_key = request_key.strip()
        if not request_key:
            raise ValueError("request_key must not be blank")
        request_digest: object = request.model_dump(mode="json")
        if idempotency_binding is not None:
            request_digest = {"request": request_digest, "binding": idempotency_binding}
        body_sha256 = _digest(request_digest)
        with self._transaction():
            self._require_task(task_id)
            existing = self._connection.execute(
                "SELECT task_id, operation, body_sha256, run_id FROM request_keys WHERE request_key = ?",
                (request_key,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["task_id"] != task_id
                    or existing["operation"] != operation
                    or existing["body_sha256"] != body_sha256
                ):
                    raise IdempotencyConflict("request key was used with a different request")
                return RunRef(task_id=task_id, run_id=str(existing["run_id"]))
            active = self._connection.execute(
                "SELECT run_id FROM runs WHERE task_id = ? AND status IN ('QUEUED', 'RUNNING', 'WAITING_INPUT', 'CANCEL_REQUESTED')",
                (task_id,),
            ).fetchone()
            if active is not None:
                raise ActiveRunConflict(f"task {task_id} already has active run {active['run_id']}")
            run_id = f"run_{uuid.uuid4().hex}"
            now = _now()
            self._connection.execute(
                "INSERT INTO runs (run_id, task_id, request_json, status, created_at) VALUES (?, ?, ?, 'QUEUED', ?)",
                (run_id, task_id, _json(request.model_dump(mode="json")), now),
            )
            self._connection.execute(
                "INSERT INTO request_keys VALUES (?, ?, ?, ?, ?, ?)",
                (request_key, task_id, operation, body_sha256, run_id, now),
            )
            return RunRef(task_id=task_id, run_id=run_id)

    @_serialized
    def reserve_evaluation_run(
        self, *, task_id: str, request: RunRequest, request_key: str,
        evaluation_id: str, manifest: dict[str, object],
    ) -> RunRef:
        """Atomically reserve the host run and its immutable evaluation record."""
        request_key = request_key.strip()
        if not request_key or not evaluation_id:
            raise ValueError("evaluation reservation identity invalid")
        manifest_json = _json(manifest)
        body_sha256 = _digest({"request": request.model_dump(mode="json"), "manifest": manifest})
        with self._transaction():
            self._require_task(task_id)
            existing = self._connection.execute(
                "SELECT task_id, operation, body_sha256, run_id FROM request_keys WHERE request_key = ?",
                (request_key,),
            ).fetchone()
            if existing is not None:
                if (existing["task_id"] != task_id or existing["operation"] != "start_evaluation"
                        or existing["body_sha256"] != body_sha256):
                    raise IdempotencyConflict("request key was used with a different evaluation")
                row = self._connection.execute(
                    "SELECT evaluation_id, manifest_json FROM evaluations WHERE run_id = ?",
                    (existing["run_id"],),
                ).fetchone()
                if row is None or row["evaluation_id"] != evaluation_id or row["manifest_json"] != manifest_json:
                    raise WorkbenchStoreError("evaluation request key is not bound to its manifest")
                return RunRef(task_id=task_id, run_id=str(existing["run_id"]))
            active = self._connection.execute(
                "SELECT run_id FROM runs WHERE task_id = ? AND status IN "
                "('QUEUED', 'RUNNING', 'WAITING_INPUT', 'CANCEL_REQUESTED')",
                (task_id,),
            ).fetchone()
            if active is not None:
                raise ActiveRunConflict(f"task {task_id} already has active run {active['run_id']}")
            now = _now()
            run_id = f"run_{uuid.uuid4().hex}"
            self._connection.execute(
                "INSERT INTO runs (run_id, task_id, request_json, status, created_at) VALUES (?, ?, ?, 'QUEUED', ?)",
                (run_id, task_id, _json(request.model_dump(mode="json")), now),
            )
            self._connection.execute(
                "INSERT INTO request_keys VALUES (?, ?, ?, ?, ?, ?)",
                (request_key, task_id, "start_evaluation", body_sha256, run_id, now),
            )
            self._connection.execute(
                "INSERT INTO evaluations (evaluation_id, task_id, run_id, manifest_json, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 'QUEUED', ?, ?)",
                (evaluation_id, task_id, run_id, manifest_json, now, now),
            )
            return RunRef(task_id=task_id, run_id=run_id)

    @_serialized
    def reserve_revision_plan(
        self,
        task_id: str,
        request: FeedbackRequest,
        request_key: str,
        *,
        source_digest: str,
        source_artifacts: list[dict[str, object]],
    ) -> tuple[str, bool]:
        """Durably bind feedback and a compilation attempt to an immutable run snapshot."""
        request_key = request_key.strip()
        if not request_key:
            raise ValueError("request_key must not be blank")
        body = {
            "feedback": request.model_dump(mode="json"),
            "source_digest": source_digest,
            "source_artifacts": source_artifacts,
        }
        body_sha256 = _digest(body)
        feedback_sha256 = _digest(body)
        with self._transaction():
            self._require_task(task_id)
            source_run = self._connection.execute(
                "SELECT task_id, status FROM runs WHERE run_id = ?", (request.source_run_id,)
            ).fetchone()
            if source_run is None or source_run["task_id"] != task_id:
                raise KeyError("feedback source run does not belong to task")
            if source_run["status"] not in _TERMINAL_STATES:
                raise ValueError("feedback source run is not terminal")

            existing_request = self._connection.execute(
                "SELECT task_id, operation, body_sha256 FROM request_keys WHERE request_key = ?",
                (request_key,),
            ).fetchone()
            if existing_request is not None:
                if (
                    existing_request["task_id"] != task_id
                    or existing_request["operation"] != "prepare_revision"
                    or existing_request["body_sha256"] != body_sha256
                ):
                    raise IdempotencyConflict("request key was used with a different request")
                existing_plan = self._connection.execute(
                    "SELECT plan_id FROM revision_plans WHERE prepare_key = ?", (request_key,)
                ).fetchone()
                if existing_plan is None:
                    raise WorkbenchStoreError("revision request is missing its durable plan")
                return str(existing_plan["plan_id"]), False

            feedback_row = self._connection.execute(
                "SELECT feedback_id, source_run_id, source_digest, source_artifacts_json, feedback_text "
                "FROM revision_feedback WHERE task_id = ? AND feedback_sha256 = ?",
                (task_id, feedback_sha256),
            ).fetchone()
            now = _now()
            if feedback_row is None:
                feedback_id = f"feedback_{uuid.uuid4().hex}"
                self._connection.execute(
                    "INSERT INTO revision_feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        feedback_id,
                        task_id,
                        request.source_run_id,
                        source_digest,
                        _json(source_artifacts),
                        request.text,
                        feedback_sha256,
                        now,
                    ),
                )
            else:
                feedback_id = str(feedback_row["feedback_id"])
                if (
                    feedback_row["source_run_id"] != request.source_run_id
                    or feedback_row["source_digest"] != source_digest
                    or feedback_row["source_artifacts_json"] != _json(source_artifacts)
                    or feedback_row["feedback_text"] != request.text
                ):
                    raise WorkbenchStoreError("feedback digest collides with different content")
            attempt = int(self._connection.execute(
                "SELECT COALESCE(MAX(attempt), 0) AS attempt FROM revision_plans "
                "WHERE task_id = ? AND feedback_id = ?",
                (task_id, feedback_id),
            ).fetchone()["attempt"]) + 1
            plan_id = f"revision_{uuid.uuid4().hex}"
            self._connection.execute(
                "INSERT INTO revision_plans "
                "(plan_id, task_id, feedback_id, source_run_id, source_digest, attempt, status, "
                "plan_json, error, prepare_key, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'PREPARING', '{}', NULL, ?, ?, ?)",
                (plan_id, task_id, feedback_id, request.source_run_id, source_digest, attempt, request_key, now, now),
            )
            self._connection.execute(
                "INSERT INTO request_keys VALUES (?, ?, 'prepare_revision', ?, ?, ?)",
                (request_key, task_id, body_sha256, request.source_run_id, now),
            )
            self._connection.execute(
                "INSERT INTO revision_compiler_cost VALUES (?, ?)",
                (plan_id, _json({"model_calls": 0, "input_tokens": 0, "output_tokens": 0,
                                 "total_tokens": 0, "estimated": False, "settled": True,
                                 "budget": request.budget.model_dump(mode="json")})),
            )
            return plan_id, True

    @_serialized
    def save_revision_compiler_cost(self, plan_id: str, cost: dict[str, object]) -> None:
        cursor = self._connection.execute(
            "UPDATE revision_compiler_cost SET cost_json = ? WHERE plan_id = ?",
            (_json(cost), plan_id),
        )
        if cursor.rowcount != 1:
            raise KeyError(f"unknown revision plan cost {plan_id}")

    @_serialized
    def revision_plan_inputs(self, plan_id: str) -> dict[str, object]:
        row = self._connection.execute(
            "SELECT p.*, f.feedback_text, f.source_artifacts_json "
            "FROM revision_plans p JOIN revision_feedback f ON f.feedback_id = p.feedback_id "
            "WHERE p.plan_id = ?",
            (plan_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown revision plan {plan_id}")
        return {
            "task_id": row["task_id"],
            "feedback_id": row["feedback_id"],
            "source_run_id": row["source_run_id"],
            "source_digest": row["source_digest"],
            "feedback_text": row["feedback_text"],
            "source_artifacts": json.loads(row["source_artifacts_json"]),
        }

    @_serialized
    def save_revision_plan(
        self,
        plan_id: str,
        *,
        status: str,
        plan_json: dict[str, object],
        error: str | None = None,
    ) -> None:
        if status not in {"READY", "UNAVAILABLE"}:
            raise ValueError("revision plan completion status must be READY or UNAVAILABLE")
        with self._transaction():
            row = self._connection.execute(
                "SELECT status, plan_json, error FROM revision_plans WHERE plan_id = ?", (plan_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown revision plan {plan_id}")
            if row["status"] != "PREPARING":
                if row["status"] == status and row["plan_json"] == _json(plan_json) and row["error"] == error:
                    return
                raise WorkbenchStoreError("revision plan already completed with different content")
            self._connection.execute(
                "UPDATE revision_plans SET status = ?, plan_json = ?, error = ?, updated_at = ? WHERE plan_id = ?",
                (status, _json(plan_json), error, _now(), plan_id),
            )

    @_serialized
    def revision_plans_for_task(self, task_id: str) -> list[WorkbenchRevisionPlan]:
        self._require_task(task_id)
        rows = self._connection.execute(
            "SELECT plan_id FROM revision_plans WHERE task_id = ? ORDER BY created_at, plan_id",
            (task_id,),
        ).fetchall()
        return [self.revision_plan(str(row["plan_id"])) for row in rows]

    @_serialized
    def revision_plan(self, plan_id: str) -> WorkbenchRevisionPlan:
        row = self._connection.execute(
            "SELECT p.*, f.feedback_text, f.source_artifacts_json, rr.run_id "
            "FROM revision_plans p JOIN revision_feedback f ON f.feedback_id = p.feedback_id "
            "LEFT JOIN revision_runs rr ON rr.plan_id = p.plan_id AND rr.attempt = "
            "(SELECT MAX(attempt) FROM revision_runs WHERE plan_id = p.plan_id) "
            "WHERE p.plan_id = ?",
            (plan_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown revision plan {plan_id}")
        payload = json.loads(row["plan_json"])
        attempts = self._connection.execute(
            "SELECT rr.attempt, rr.run_id, r.status FROM revision_runs rr "
            "JOIN runs r ON r.run_id = rr.run_id WHERE rr.plan_id = ? ORDER BY rr.attempt",
            (plan_id,),
        ).fetchall()
        cost_row = self._connection.execute(
            "SELECT cost_json FROM revision_compiler_cost WHERE plan_id = ?", (plan_id,)
        ).fetchone()
        return WorkbenchRevisionPlan.model_validate({
            "plan_id": row["plan_id"],
            "task_id": row["task_id"],
            "feedback_id": row["feedback_id"],
            "source_run_id": row["source_run_id"],
            "source_digest": row["source_digest"],
            "source_run_status": self.run(str(row["source_run_id"])).status,
            "source_artifacts": json.loads(row["source_artifacts_json"]),
            "feedback_text": row["feedback_text"],
            "status": row["status"],
            "issues": payload.get("issues", []),
            "warnings": payload.get("warnings", []),
            "compiler_provider": payload.get("compiler_provider"),
            "compiler_model": payload.get("compiler_model"),
            "error": row["error"],
            "instruction": payload.get("instruction"),
            "attempt": row["attempt"],
            "created_at": row["created_at"],
            "run_id": row["run_id"],
            "run_attempts": [dict(item) for item in attempts],
            "compiler_cost": json.loads(cost_row["cost_json"]) if cost_row else {},
        })

    @_serialized
    def reserve_revision_run(
        self,
        plan_id: str,
        request: RunRequest,
        request_key: str,
    ) -> RunRef:
        request_key = request_key.strip()
        if not request_key:
            raise ValueError("request_key must not be blank")
        body_sha256 = _digest({"request": request.model_dump(mode="json"), "revision_plan_id": plan_id})
        with self._transaction():
            plan = self._connection.execute(
                "SELECT task_id, status FROM revision_plans WHERE plan_id = ?", (plan_id,)
            ).fetchone()
            if plan is None:
                raise KeyError(f"unknown revision plan {plan_id}")
            task_id = str(plan["task_id"])
            existing = self._connection.execute(
                "SELECT task_id, operation, body_sha256, run_id FROM request_keys WHERE request_key = ?",
                (request_key,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["task_id"] != task_id
                    or existing["operation"] != "revision_run"
                    or existing["body_sha256"] != body_sha256
                ):
                    raise IdempotencyConflict("request key was used with a different request")
                binding = self._connection.execute(
                    "SELECT plan_id FROM revision_runs WHERE run_id = ?", (existing["run_id"],)
                ).fetchone()
                if binding is None or binding["plan_id"] != plan_id:
                    raise WorkbenchStoreError("revision request is missing its plan binding")
                return RunRef(task_id=task_id, run_id=str(existing["run_id"]))
            if plan["status"] != "READY":
                raise ValueError("only a READY revision plan can be executed")
            prior = self._connection.execute(
                "SELECT rr.attempt, r.status FROM revision_runs rr "
                "JOIN runs r ON r.run_id = rr.run_id WHERE rr.plan_id = ? "
                "ORDER BY rr.attempt DESC LIMIT 1", (plan_id,),
            ).fetchone()
            if prior is not None:
                if prior["status"] in {"SUCCEEDED", "PARTIAL"}:
                    raise ActiveRunConflict("this revision plan already succeeded")
                if prior["status"] not in {"FAILED", "CANCELLED", "INTERRUPTED"}:
                    raise ActiveRunConflict("this revision plan already has a run")
            self._require_task(task_id)
            active = self._connection.execute(
                "SELECT run_id FROM runs WHERE task_id = ? AND status IN ('QUEUED', 'RUNNING', 'WAITING_INPUT', 'CANCEL_REQUESTED')",
                (task_id,),
            ).fetchone()
            if active is not None:
                raise ActiveRunConflict(f"task {task_id} already has active run {active['run_id']}")
            run_id = f"run_{uuid.uuid4().hex}"
            now = _now()
            self._connection.execute(
                "INSERT INTO runs (run_id, task_id, request_json, status, created_at) VALUES (?, ?, ?, 'QUEUED', ?)",
                (run_id, task_id, _json(request.model_dump(mode="json")), now),
            )
            self._connection.execute(
                "INSERT INTO request_keys VALUES (?, ?, 'revision_run', ?, ?, ?)",
                (request_key, task_id, body_sha256, run_id, now),
            )
            self._connection.execute(
                "INSERT INTO revision_runs (plan_id, task_id, run_id, attempt, request_key, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (plan_id, task_id, run_id, int(prior["attempt"]) + 1 if prior else 1, request_key, now),
            )
            return RunRef(task_id=task_id, run_id=run_id)

    @_serialized
    def finish_run(self, run_id: str, status: str) -> None:
        if status not in _TERMINAL_STATES:
            raise ValueError(f"status must be terminal, got {status!r}")
        result = self._connection.execute(
            "UPDATE runs SET status = ?, finished_at = ? WHERE run_id = ?",
            (status, _now(), run_id),
        )
        if result.rowcount != 1:
            raise KeyError(f"unknown run {run_id}")

    @_serialized
    def transition_run(self, run_id: str, status: str) -> None:
        """Persist an active lifecycle state without bypassing the run index."""
        if status not in _ACTIVE_STATES:
            raise ValueError(f"status must be active, got {status!r}")
        result = self._connection.execute(
            "UPDATE runs SET status = ? WHERE run_id = ?", (status, run_id)
        )
        if result.rowcount != 1:
            raise KeyError(f"unknown run {run_id}")

    @_serialized
    def append_event(self, event: WorkbenchEvent) -> None:
        with self._transaction():
            run = self._connection.execute(
                "SELECT task_id FROM runs WHERE run_id = ?", (event.run_id,)
            ).fetchone()
            if run is None or run["task_id"] != event.task_id:
                raise ValueError("event run does not belong to task")
            current = self._connection.execute(
                "SELECT COALESCE(MAX(seq), 0) AS seq FROM events WHERE task_id = ?",
                (event.task_id,),
            ).fetchone()["seq"]
            if event.seq != current + 1:
                raise ValueError("event sequence must be next for task")
            self._connection.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
                (event.task_id, event.run_id, event.seq, event.type, _json(event.payload), event.created_at),
            )

    @_serialized
    def append_event_for_run(
        self, run_id: str, event_type: str, payload: dict[str, object]
    ) -> WorkbenchEvent:
        """Append the next durable event for a run before any notification."""
        with self._transaction():
            row = self._connection.execute(
                "SELECT task_id FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown run {run_id}")
            task_id = str(row["task_id"])
            seq = int(self._connection.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 AS seq FROM events WHERE task_id = ?",
                (task_id,),
            ).fetchone()["seq"])
            event = WorkbenchEvent(
                task_id=task_id,
                run_id=run_id,
                seq=seq,
                type=event_type,
                payload=payload,
                created_at=_now(),
            )
            self._connection.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
                (task_id, run_id, seq, event_type, _json(payload), event.created_at),
            )
            return event

    @_serialized
    def tool_request_payload(self, run_id: str, tool_call_id: str, tool_name: str) -> dict[str, object] | None:
        """Load only the matching request, without materializing event history."""
        row = self._connection.execute(
            "SELECT payload_json FROM events WHERE run_id = ? AND type = 'tool_requested' "
            "AND json_extract(payload_json, '$.tool_call_id') = ? "
            "AND json_extract(payload_json, '$.tool_name') = ? ORDER BY seq DESC LIMIT 1",
            (run_id, tool_call_id, tool_name),
        ).fetchone()
        return json.loads(row["payload_json"]) if row is not None else None

    @_serialized
    def events_after(
        self, task_id: str, seq: int, *, limit: int | None = None
    ) -> list[WorkbenchEvent]:
        if limit is not None and limit < 1:
            raise ValueError("event limit must be positive")
        query = "SELECT * FROM events WHERE task_id = ? AND seq > ? ORDER BY seq"
        parameters: tuple[object, ...] = (task_id, seq)
        if limit is not None:
            query += " LIMIT ?"
            parameters += (limit,)
        rows = self._connection.execute(query, parameters).fetchall()
        return [
            WorkbenchEvent(task_id=row["task_id"], run_id=row["run_id"], seq=row["seq"], type=row["type"], payload=json.loads(row["payload_json"]), created_at=row["created_at"])
            for row in rows
        ]

    @_serialized
    def task(self, task_id: str) -> ResearchTask:
        row = self._connection.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown task {task_id}")
        return _task(row)

    @_serialized
    def tasks(self) -> list[ResearchTask]:
        return [_task(row) for row in self._connection.execute("SELECT * FROM tasks ORDER BY created_at, task_id")]

    @_serialized
    def active_runs(self, task_id: str) -> list[RunRecord]:
        rows = self._connection.execute(
            "SELECT * FROM runs WHERE task_id = ? AND status IN ('QUEUED', 'RUNNING', 'WAITING_INPUT', 'CANCEL_REQUESTED')", (task_id,)
        )
        return [_run(row) for row in rows]

    @_serialized
    def run(self, run_id: str) -> RunRecord:
        row = self._connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown run {run_id}")
        return _run(row)

    @_serialized
    def unfinished_runs(self) -> list[RunRecord]:
        rows = self._connection.execute(
            "SELECT * FROM runs WHERE status IN ('QUEUED', 'RUNNING', 'WAITING_INPUT', 'CANCEL_REQUESTED')"
        )
        return [_run(row) for row in rows]

    @_serialized
    def link_source(self, task_id: str, source_type: str, source_id: str, source_digest: str) -> None:
        """Bind a produced authoritative source to its owning research task."""
        self._require_task(task_id)
        existing = self._connection.execute("SELECT task_id, source_digest FROM links WHERE source_type=? AND source_id=?", (source_type, source_id)).fetchone()
        if existing is not None:
            if existing["task_id"] != task_id or existing["source_digest"] != source_digest:
                raise HistoricalSourceConflict("source ownership or digest changed")
            return
        self._connection.execute("INSERT INTO links (task_id, source_type, source_id, source_digest, created_at) VALUES (?, ?, ?, ?, ?)", (task_id, source_type, source_id, source_digest, _now()))

    @_serialized
    def links_for_task(self, task_id: str) -> list[WorkbenchLink]:
        rows = self._connection.execute("SELECT * FROM links WHERE task_id = ?", (task_id,))
        return [WorkbenchLink(**dict(row)) for row in rows]

    @_serialized
    def runs_for_task(self, task_id: str) -> list[RunRecord]:
        self._require_task(task_id)
        rows = self._connection.execute(
            "SELECT * FROM runs WHERE task_id = ? ORDER BY created_at, rowid", (task_id,)
        )
        return [_run(row) for row in rows]

    @_serialized
    def register_artifact(
        self,
        *,
        artifact_id: str,
        task_id: str,
        path: str,
        filename: str,
        sha256: str,
        media_type: str,
        size_bytes: int,
        run_id: str | None = None,
    ) -> None:
        self._require_task(task_id)
        if run_id is not None and self.run(run_id).task_id != task_id:
            raise ValueError("artifact run does not belong to task")
        self._connection.execute(
            "INSERT INTO artifacts (artifact_id, task_id, run_id, path, filename, sha256, media_type, size_bytes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (artifact_id, task_id, run_id, path, filename, sha256, media_type, size_bytes, _now()),
        )

    @_serialized
    def register_attachment(
        self,
        *,
        artifact_id: str,
        task_id: str,
        path: str,
        filename: str,
        sha256: str,
        media_type: str,
        size_bytes: int,
    ) -> None:
        """Register upload metadata and its task link as one transaction."""
        with self._transaction():
            self._require_task(task_id)
            self._connection.execute(
                "INSERT INTO artifacts (artifact_id, task_id, run_id, path, filename, sha256, media_type, size_bytes, created_at) VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?)",
                (artifact_id, task_id, path, filename, sha256, media_type, size_bytes, _now()),
            )
            row = self._connection.execute(
                "SELECT attachment_ids_json FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            ids = list(json.loads(row["attachment_ids_json"]))
            if artifact_id not in ids:
                ids.append(artifact_id)
                self._connection.execute(
                    "UPDATE tasks SET attachment_ids_json = ? WHERE task_id = ?",
                    (_json(ids), task_id),
                )

    @_serialized
    def artifact(self, artifact_id: str) -> sqlite3.Row:
        row = self._connection.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown artifact {artifact_id}")
        return row

    @_serialized
    def artifacts_for_task(self, task_id: str) -> list[sqlite3.Row]:
        self._require_task(task_id)
        return self._connection.execute(
            "SELECT * FROM artifacts WHERE task_id = ? ORDER BY created_at, artifact_id", (task_id,)
        ).fetchall()

    @_serialized
    def evaluation(self, evaluation_id: str) -> dict[str, object]:
        row = self._connection.execute(
            "SELECT * FROM evaluations WHERE evaluation_id = ?", (evaluation_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown evaluation {evaluation_id}")
        return {
            "evaluation_id": str(row["evaluation_id"]),
            "task_id": str(row["task_id"]),
            "run_id": str(row["run_id"]),
            "manifest": json.loads(row["manifest_json"]),
            "status": str(row["status"]),
            "report": json.loads(row["report_json"]) if row["report_json"] else None,
            "error_code": row["error_code"],
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    @_serialized
    def evaluations_for_task(self, task_id: str) -> list[dict[str, object]]:
        self._require_task(task_id)
        rows = self._connection.execute(
            "SELECT evaluation_id FROM evaluations WHERE task_id = ? ORDER BY created_at DESC, evaluation_id DESC",
            (task_id,),
        ).fetchall()
        return [self.evaluation(str(row["evaluation_id"])) for row in rows]

    @_serialized
    def update_evaluation(
        self, evaluation_id: str, *, status: str,
        report: dict[str, object] | None = None, error_code: str | None = None,
    ) -> None:
        allowed = {"QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED"}
        if status not in allowed:
            raise ValueError("evaluation status invalid")
        with self._transaction():
            existing = self._connection.execute(
                "SELECT report_json FROM evaluations WHERE evaluation_id = ?", (evaluation_id,)
            ).fetchone()
            if existing is None:
                raise KeyError(f"unknown evaluation {evaluation_id}")
            report_json = _json(report) if report is not None else existing["report_json"]
            self._connection.execute(
                "UPDATE evaluations SET status = ?, report_json = ?, error_code = ?, updated_at = ? "
                "WHERE evaluation_id = ?",
                (status, report_json, error_code, _now(), evaluation_id),
            )

    @_serialized
    def add_attachment_id(self, task_id: str, artifact_id: str) -> None:
        with self._transaction():
            row = self._connection.execute(
                "SELECT attachment_ids_json FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown task {task_id}")
            ids = list(json.loads(row["attachment_ids_json"]))
            if artifact_id not in ids:
                ids.append(artifact_id)
                self._connection.execute(
                    "UPDATE tasks SET attachment_ids_json = ? WHERE task_id = ?",
                    (_json(ids), task_id),
                )

    def _require_task(self, task_id: str) -> None:
        if self._connection.execute("SELECT 1 FROM tasks WHERE task_id = ?", (task_id,)).fetchone() is None:
            raise KeyError(f"unknown task {task_id}")

    def _transaction(self):
        return _ImmediateTransaction(self._connection)


class _ImmediateTransaction:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def __enter__(self) -> None:
        self.connection.execute("BEGIN IMMEDIATE")

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.connection.execute("ROLLBACK" if exc_type else "COMMIT")


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _digest(value: object) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _task(row: sqlite3.Row) -> ResearchTask:
    return ResearchTask(task_id=row["task_id"], session_id=row["session_id"], title=row["title"], goal=row["goal"], attachment_ids=json.loads(row["attachment_ids_json"]), created_at=row["created_at"])


def _run(row: sqlite3.Row) -> RunRecord:
    return RunRecord(run_id=row["run_id"], task_id=row["task_id"], request=RunRequest.model_validate_json(row["request_json"]), status=row["status"], created_at=row["created_at"])


__all__ = [
    "ActiveRunConflict", "HistoricalSourceConflict", "IdempotencyConflict", "WorkbenchStore", "WorkbenchStoreError",
]
