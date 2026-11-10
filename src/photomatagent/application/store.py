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
_SCHEMA_VERSION = 9
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
        from photomatagent.persistence.database import Database
        from photomatagent.persistence.migrations import upgrade, VERSION
        self.database = Database(self.database_path)
        self._lock = self.database.lock
        self._connection = self.database.connection
        from contextlib import ExitStack
        from photomatagent.persistence.database import OwnershipLock
        with OwnershipLock(self.database_path.with_suffix(".migration.lock")), ExitStack() as guards:
            existing = self._connection.execute("SELECT name FROM sqlite_master WHERE name='schema_version'").fetchone()
            version = self._connection.execute("SELECT version FROM schema_version WHERE id=1").fetchone() if existing else None
            if version and version[0] > VERSION:
                self._connection.close()
                raise WorkbenchStoreError("DATABASE_TOO_NEW")
            if version is None or version[0] < VERSION:
                guards.enter_context(OwnershipLock(self.database_path.with_suffix(".owner.lock")))
                if version:
                    backup = self.database_path.with_name(self.database_path.name + f".pre-v{VERSION}.bak")
                    self.database.backup_once(backup, int(version[0]))
            self._initialize()
            upgrade(self.database, _now())
        from photomatagent.application.projects import ProjectRepository
        self.projects = ProjectRepository(self.database)
        self.projects.initialize_search()
        self._connection.execute("""CREATE TRIGGER IF NOT EXISTS run_management_gate BEFORE INSERT ON runs
            WHEN EXISTS(SELECT 1 FROM tasks JOIN projects USING(project_id) WHERE tasks.task_id=NEW.task_id AND (tasks.management_status!='OPEN' OR tasks.archived_at IS NOT NULL OR projects.archived_at IS NOT NULL))
            BEGIN SELECT RAISE(ABORT,'TASK_NOT_OPEN'); END""")
        self._connection.execute("""CREATE TRIGGER IF NOT EXISTS run_summary AFTER INSERT ON runs BEGIN
            INSERT OR IGNORE INTO run_usage(run_id) VALUES (NEW.run_id);
            UPDATE runs SET project_id_at_start=(SELECT project_id FROM tasks WHERE task_id=NEW.task_id),base_checkpoint_id=(SELECT latest_checkpoint_id FROM sessions WHERE task_id=NEW.task_id) WHERE run_id=NEW.run_id;
            UPDATE task_summaries SET latest_run_id=NEW.run_id,last_activity_at=NEW.created_at WHERE task_id=NEW.task_id;
            END""")
        self._connection.execute("""CREATE TRIGGER IF NOT EXISTS event_summary AFTER INSERT ON events BEGIN
            UPDATE task_summaries SET last_activity_at=NEW.created_at,last_event_seq=NEW.seq WHERE task_id=NEW.task_id;
            END""")

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
        with self._transaction():
            project = self.projects.project(request.project_id)
            if project["archived_at"]:
                from photomatagent.application.errors import ServiceError
                raise ServiceError("PROJECT_ARCHIVED", "课题已归档。", status_code=409)
            now = _now()
            self._connection.execute(
                "INSERT INTO tasks (task_id,session_id,title,goal,attachment_ids_json,created_at,project_id,updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (task_id,task_id,request.title,request.goal,_json(request.attachment_ids),now,request.project_id,now),
            )
            self._connection.execute("INSERT INTO task_summaries(task_id,last_activity_at) VALUES (?,?)", (task_id,now))
            self.projects._set_tags("task", task_id, request.tags)
            self.projects.index_task(task_id)
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
            self._connection.execute("UPDATE tasks SET updated_at=created_at WHERE task_id=?", (task_id,))
            self._connection.execute("INSERT INTO task_summaries(task_id,last_activity_at) VALUES (?,?)", (task_id,now))
            self.projects.index_task(task_id)
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
            self.projects.require_runnable(task_id)
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
            self.projects.require_runnable(task_id)
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
            pending_ids = self.result_invalidations(task_id).get(request.source_run_id, ())
            carried = [
                {"plan_id": row["plan_id"], "feedback_text": row["feedback_text"]}
                for row in self._connection.execute(
                    "SELECT p.plan_id, f.feedback_text FROM revision_plans p "
                    "JOIN revision_feedback f USING(feedback_id) WHERE p.task_id=? "
                    "ORDER BY p.created_at, p.plan_id", (task_id,),
                )
                if row["plan_id"] in pending_ids
            ]
            self._connection.execute(
                "INSERT INTO revision_plans "
                "(plan_id, task_id, feedback_id, source_run_id, source_digest, attempt, status, "
                "plan_json, error, prepare_key, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'PREPARING', ?, NULL, ?, ?, ?)",
                (plan_id, task_id, feedback_id, request.source_run_id, source_digest, attempt,
                 _json({"carried_feedback": carried}) if carried else "{}", request_key, now, now),
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
    def revision_carried_feedback(self, plan_id: str) -> list[dict[str, str]]:
        row = self._connection.execute("SELECT plan_json FROM revision_plans WHERE plan_id=?", (plan_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown revision plan {plan_id}")
        return cast(list[dict[str, str]], json.loads(row["plan_json"]).get("carried_feedback", []))

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
            carried = json.loads(row["plan_json"]).get("carried_feedback")
            if carried:
                plan_json = {**plan_json, "carried_feedback": carried}
            elif "carried_feedback" in plan_json:
                raise ValueError("carried feedback must be reserved with the revision source")
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
            "carried_feedback_plan_ids": [item["plan_id"] for item in payload.get("carried_feedback", [])],
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
            self.projects.require_runnable(task_id)
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
    def complete_run(self, run_id: str, status: str, event_type: str, payload: dict[str, object]) -> None:
        with self._transaction():
            self.finish_run(run_id, status)
            self.append_event_for_run(run_id, event_type, payload)

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

    def _event_payload(self, raw: str) -> dict[str, object]:
        payload = json.loads(raw)
        reference = payload.pop("_runtime_ref", None)
        if reference:
            source = self._connection.execute("SELECT payload_json FROM runtime_events WHERE event_id=?", (reference,)).fetchone()
            if source is None:
                raise WorkbenchStoreError("RUNTIME_EVENT_SOURCE_MISSING")
            full = json.loads(source[0])
            full.pop("kind", None)
            return full | payload
        return payload

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
            stored_payload = dict(payload)
            event_id = payload.get("trace_event_id")
            if event_type in {"text_delta", "tool_completed"} and event_id and self._connection.execute("SELECT 1 FROM runtime_events WHERE event_id=?", (event_id,)).fetchone():
                stored_payload.pop("text", None)
                stored_payload.pop("output", None)
                stored_payload["_runtime_ref"] = event_id
            if event_type == "model_request_started":
                self._connection.execute("INSERT OR IGNORE INTO run_usage(run_id) VALUES (?)", (run_id,))
                self._connection.execute("UPDATE run_usage SET unsettled_requests=unsettled_requests+1 WHERE run_id=?", (run_id,))
            elif event_type == "model_response_completed":
                usage = payload.get("usage")
                usage = usage if isinstance(usage, dict) else {}
                tokens = int(usage.get("total_tokens") or (int(usage.get("input_tokens") or 0)+int(usage.get("output_tokens") or 0)))
                self._connection.execute("INSERT OR IGNORE INTO run_usage(run_id) VALUES (?)", (run_id,))
                self._connection.execute("UPDATE run_usage SET model_calls=model_calls+1,known_tokens=known_tokens+?,unsettled_requests=MAX(unsettled_requests-1,0) WHERE run_id=?", (tokens,run_id))
            self._connection.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
                (task_id, run_id, seq, event_type, _json(stored_payload), event.created_at),
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
            WorkbenchEvent(task_id=row["task_id"], run_id=row["run_id"], seq=row["seq"], type=row["type"], payload=self._event_payload(row["payload_json"]), created_at=row["created_at"])
            for row in rows
        ]

    @_serialized
    def task(self, task_id: str) -> ResearchTask:
        row = self._connection.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown task {task_id}")
        item = _task(row)
        item.tags = self.projects._tags("task", task_id)
        return item

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
        record = _run(row)
        record.result_invalidation_ids = self.result_invalidations(record.task_id).get(run_id, [])
        return record

    @_serialized
    def result_invalidations(self, task_id: str) -> dict[str, list[str]]:
        """Project historical results from committed feedback and explicit run dependencies."""
        related_tasks = """
            WITH RECURSIVE related(task_id) AS (
                SELECT ? UNION
                SELECT origin.task_id FROM related
                JOIN artifacts linked ON linked.task_id=related.task_id
                JOIN artifacts origin ON origin.artifact_id=linked.origin_artifact_id
            )
        """
        feedback = self._connection.execute(
            related_tasks + "SELECT p.plan_id, p.task_id, p.feedback_id, p.source_run_id, p.plan_json, "
            "p.updated_at, f.feedback_text "
            "FROM revision_plans p JOIN revision_feedback f USING(feedback_id) "
            "WHERE p.task_id IN (SELECT task_id FROM related) AND p.status='READY' "
            "ORDER BY p.created_at, p.plan_id", (task_id,),
        ).fetchall()
        corrections = [
            row for row in feedback
            if any(issue.get("status") == "CORRECTION" and issue.get("category") in {
                "TASK_DEFINITION", "SCIENTIFIC_CORRECTNESS", "EVIDENCE_SUFFICIENCY", "SAFETY",
            } and (span := " ".join(issue.get("original_feedback_span", "").split()).casefold())
                and span in " ".join(row["feedback_text"].split()).casefold()
                for issue in json.loads(row["plan_json"]).get("issues", ()))
        ]
        if not corrections:
            return {}
        dependencies = self._connection.execute(
            related_tasks + "SELECT r.run_id, r.task_id, r.resumes_run_id, c.run_id AS checkpoint_run_id, "
            "p.source_run_id, p.feedback_id, p.plan_json, r.status, r.created_at, "
            "json_extract(r.request_json, '$.host_operation') AS host_operation FROM runs r "
            "LEFT JOIN checkpoints c ON c.checkpoint_id=r.base_checkpoint_id "
            "LEFT JOIN revision_runs rr ON rr.run_id=r.run_id "
            "LEFT JOIN revision_plans p ON p.plan_id=rr.plan_id "
            "WHERE r.task_id IN (SELECT task_id FROM related)", (task_id,),
        )
        descendants: dict[str, set[str]] = {}
        applied_feedback: dict[str, str] = {}
        carried_by_run: dict[str, set[str]] = {}
        run_tasks: dict[str, str] = {}
        completed_rechecks: dict[str, str] = {}
        for row in dependencies:
            child = row["run_id"]
            run_tasks[child] = row["task_id"]
            if row["status"] == "SUCCEEDED" and row["host_operation"] is None:
                completed_rechecks[child] = row["created_at"]
            if row["feedback_id"] and row["status"] == "SUCCEEDED":
                applied_feedback[child] = row["feedback_id"]
            if row["plan_json"] and row["status"] == "SUCCEEDED":
                carried_by_run[child] = {item["plan_id"]
                                         for item in json.loads(row["plan_json"]).get("carried_feedback", [])}
            for parent in (row["resumes_run_id"], row["checkpoint_run_id"], row["source_run_id"]):
                if parent and parent != child:
                    descendants.setdefault(parent, set()).add(child)
        replacements = {
            row["previous_id"]: (row["replacement_id"], row["created_at"])
            for row in self._connection.execute(
                related_tasks + "SELECT json_extract(payload_json, '$.previous_artifact_id') AS previous_id, "
                "json_extract(payload_json, '$.replacement_artifact_id') AS replacement_id, "
                "created_at FROM management_events WHERE kind='task_input_replaced' "
                "AND entity_id IN (SELECT task_id FROM related)", (task_id,),
            )
        }
        checked_inputs = {
            row["run_id"]: set(json.loads(row["payload_json"])["artifact_ids"])
            for row in self._connection.execute(
                related_tasks + "SELECT run_id,payload_json FROM events WHERE type='replacement_inputs_checked' "
                "AND task_id IN (SELECT task_id FROM related) ORDER BY seq", (task_id,),
            )
        } if replacements else {}

        def used_replacement(run_id: str, artifact_id: str) -> bool:
            reserved = completed_rechecks.get(run_id)
            if reserved is None or artifact_id not in replacements:
                return False
            current = artifact_id
            while current in replacements and replacements[current][1] <= reserved:
                current = replacements[current][0]
            return current != artifact_id and current in checked_inputs.get(run_id, ())

        links = self._connection.execute(
            related_tasks + """,
            origins(link_id, origin_id) AS (
                SELECT a.artifact_id, a.origin_artifact_id FROM artifacts a
                JOIN related USING(task_id) WHERE a.origin_artifact_id IS NOT NULL
                UNION
                SELECT origins.link_id, a.origin_artifact_id FROM origins
                JOIN artifacts a ON a.artifact_id=origins.origin_id WHERE a.origin_artifact_id IS NOT NULL
            )
            SELECT source.run_id AS parent, r.run_id AS child, r.created_at, linked.artifact_id
            FROM origins JOIN artifacts source ON source.artifact_id=origins.origin_id
            JOIN artifacts linked ON linked.artifact_id=origins.link_id
            JOIN runs r ON r.task_id=linked.task_id AND r.created_at>=linked.created_at
            WHERE source.run_id IS NOT NULL AND source.run_id<>r.run_id
            """, (task_id,),
        )
        linked_descendants: dict[str, list[tuple[str, str]]] = {}
        for row in links:
            retired_at = replacements.get(row["artifact_id"])
            if retired_at is None or row["created_at"] < retired_at[1]:
                linked_descendants.setdefault(row["parent"], []).append((row["child"], row["artifact_id"]))
        historical: dict[str, list[str]] = {}
        for row in corrections:
            pending: list[tuple[str, str | None]] = [(row["source_run_id"], None)]
            visited: set[tuple[str, str | None]] = set()
            affected: set[str] = set()
            while pending:
                run_id, attachment = pending.pop()
                if ((run_id, attachment) in visited or applied_feedback.get(run_id) == row["feedback_id"]
                        or row["plan_id"] in carried_by_run.get(run_id, ())):
                    continue
                visited.add((run_id, attachment))
                if attachment is not None and used_replacement(run_id, attachment):
                    continue
                if run_tasks.get(run_id) == task_id and run_id not in affected:
                    historical.setdefault(run_id, []).append(row["plan_id"])
                    affected.add(run_id)
                pending.extend((child, attachment) for child in descendants.get(run_id, ()))
                pending.extend((child, linked if run_tasks.get(child) != run_tasks.get(run_id) else attachment)
                               for child, linked in linked_descendants.get(run_id, ()))
        return historical

    @_serialized
    def replacement_input_ids(self, task_id: str) -> set[str]:
        return {
            row[0] for row in self._connection.execute(
                "SELECT json_extract(payload_json, '$.replacement_artifact_id') FROM management_events "
                "WHERE kind='task_input_replaced' AND entity_id=?", (task_id,),
            )
        }

    @_serialized
    def bind_replacement_input(self, task_id: str, previous_id: str, replacement_id: str, source_id: str) -> None:
        """Retire an active link while retaining both artifact rows and their provenance."""
        with self._transaction():
            task = self.task(task_id)
            if previous_id not in task.attachment_ids or replacement_id not in task.attachment_ids:
                raise ValueError("REPLACEMENT_INPUT_BINDING_CHANGED")
            ids = [replacement_id if item == previous_id else item
                   for item in task.attachment_ids if item != replacement_id]
            now = _now()
            self._connection.execute(
                "UPDATE tasks SET attachment_ids_json=?, revision=revision+1, updated_at=? WHERE task_id=?",
                (_json(ids), now, task_id),
            )
            self._connection.execute(
                "INSERT INTO management_events(entity_id,revision,kind,payload_json,created_at) VALUES (?,?,?,?,?)",
                (task_id, task.revision + 1, "task_input_replaced", _json({
                    "previous_artifact_id": previous_id, "replacement_artifact_id": replacement_id,
                    "source_artifact_id": source_id,
                }), now),
            )

    @_serialized
    def artifact_result_invalidations(self, artifact_id: str) -> list[str]:
        ancestors = self._connection.execute(
            """
            WITH RECURSIVE lineage(artifact_id, task_id, run_id, origin_artifact_id) AS (
                SELECT artifact_id, task_id, run_id, origin_artifact_id FROM artifacts WHERE artifact_id=?
                UNION
                SELECT a.artifact_id, a.task_id, a.run_id, a.origin_artifact_id
                FROM artifacts a JOIN lineage l ON a.artifact_id=l.origin_artifact_id
            ) SELECT DISTINCT task_id, run_id FROM lineage WHERE run_id IS NOT NULL
            """, (artifact_id,),
        ).fetchall()
        return sorted({plan_id for row in ancestors
                       for plan_id in self.result_invalidations(row["task_id"]).get(row["run_id"], ())})

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
        historical = self.result_invalidations(task_id)
        return [
            _run(row).model_copy(update={"result_invalidation_ids": historical.get(row["run_id"], [])})
            for row in rows
        ]

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

        project_id = self.task(task_id).project_id
        self.projects.index_document("artifact:" + artifact_id, "artifact", filename, path, project_id, task_id)

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

            self.projects.index_document("artifact:" + artifact_id, "artifact", filename, path, self.task(task_id).project_id, task_id)

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
        return self.database.transaction()


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
    return ResearchTask(task_id=row["task_id"], session_id=row["session_id"], title=row["title"], goal=row["goal"], attachment_ids=json.loads(row["attachment_ids_json"]), created_at=row["created_at"], project_id=row["project_id"], management_status=row["management_status"], archived_at=row["archived_at"], updated_at=row["updated_at"], revision=row["revision"])


def _run(row: sqlite3.Row) -> RunRecord:
    return RunRecord(run_id=row["run_id"], task_id=row["task_id"], request=RunRequest.model_validate_json(row["request_json"]), status=row["status"], created_at=row["created_at"], base_checkpoint_id=row["base_checkpoint_id"], final_checkpoint_id=row["final_checkpoint_id"], recovery_status=row["recovery_status"], resumes_run_id=row["resumes_run_id"])


__all__ = [
    "ActiveRunConflict", "HistoricalSourceConflict", "IdempotencyConflict", "WorkbenchStore", "WorkbenchStoreError",
]
