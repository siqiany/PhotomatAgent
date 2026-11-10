"""Transactional project management and indexed workspace queries."""
from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from photomatagent.application.errors import ServiceError
from photomatagent.persistence.database import Database
from photomatagent.persistence.migrations import UNCLASSIFIED


def now() -> str:
    return datetime.now(UTC).isoformat()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class ProjectCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=500)
    description: str = Field(default="", max_length=20000)
    tags: list[str] = Field(default_factory=list, max_length=30)

    @field_validator("name")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("name cannot be blank")
        return value.strip()


class ProjectPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=1)
    name: str | None = Field(default=None, min_length=1, max_length=500)
    description: str | None = Field(default=None, max_length=20000)
    tags: list[str] | None = Field(default=None, max_length=30)


class TaskPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=1)
    title: str | None = Field(default=None, min_length=1, max_length=500)
    management_status: Literal["OPEN", "PAUSED", "DONE"] | None = None
    tags: list[str] | None = Field(default=None, max_length=30)


class RevisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=1)


class MoveRequest(RevisionRequest):
    project_id: str
    target_revision: int = Field(ge=1)


class ProjectRepository:
    def __init__(self, database: Database) -> None:
        self.db = database
        self.conn = database.connection

    def initialize_search(self) -> None:
        """Optional built-in extension, never load external shared libraries."""
        with self.db.transaction():
            existed = self.conn.execute("SELECT 1 FROM sqlite_master WHERE name='search_fts'").fetchone()
            try:
                self.conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS search_fts USING fts5(title,text,content='search_documents',content_rowid='rowid',tokenize='trigram')")
                self.conn.execute("""CREATE TRIGGER IF NOT EXISTS search_insert AFTER INSERT ON search_documents BEGIN
                    INSERT INTO search_fts(rowid,title,text) VALUES (NEW.rowid,NEW.title,NEW.text); END""")
                self.conn.execute("""CREATE TRIGGER IF NOT EXISTS search_delete AFTER DELETE ON search_documents BEGIN
                    INSERT INTO search_fts(search_fts,rowid,title,text) VALUES ('delete',OLD.rowid,OLD.title,OLD.text); END""")
                self.conn.execute("""CREATE TRIGGER IF NOT EXISTS search_update AFTER UPDATE ON search_documents BEGIN
                    INSERT INTO search_fts(search_fts,rowid,title,text) VALUES ('delete',OLD.rowid,OLD.title,OLD.text);
                    INSERT INTO search_fts(rowid,title,text) VALUES (NEW.rowid,NEW.title,NEW.text); END""")
                if not existed:
                    self.conn.execute("INSERT INTO search_fts(search_fts) VALUES ('rebuild')")
            except sqlite3.OperationalError:
                # FTS5/trigram is optional; all authoritative data remains usable.
                for trigger in ("search_insert", "search_delete", "search_update"):
                    self.conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")

    def _row(self, table: str, key: str, value: str) -> dict[str, Any]:
        row = self.conn.execute(f"SELECT * FROM {table} WHERE {key}=?", (value,)).fetchone()
        if row is None:
            raise ServiceError("NOT_FOUND", "记录不存在。", status_code=404)
        return dict(row)

    def _tags(self, kind: str, entity_id: str) -> list[str]:
        return [row[0] for row in self.conn.execute(f"SELECT name FROM tags JOIN {kind}_tags USING(tag_id) WHERE {kind}_id=? ORDER BY name", (entity_id,))]

    def _set_tags(self, kind: str, entity_id: str, tags: list[str]) -> None:
        normalized = sorted({tag.strip().casefold() for tag in tags})
        if any(not tag or len(tag) > 100 for tag in normalized) or len(normalized) > 30:
            raise ServiceError("INVALID_TAG", "标签不能为空，每个不超过 100 字。", status_code=422)
        self.conn.execute(f"DELETE FROM {kind}_tags WHERE {kind}_id=?", (entity_id,))
        for tag in normalized:
            self.conn.execute("INSERT OR IGNORE INTO tags(name) VALUES (?)", (tag,))
            self.conn.execute(f"INSERT INTO {kind}_tags VALUES (?,(SELECT tag_id FROM tags WHERE name=?))", (entity_id, tag))

    def _audit(self, entity_id: str, revision: int, kind: str, payload: Any) -> None:
        from photomatagent.redaction import redact_secrets
        self.conn.execute("INSERT INTO management_events(entity_id,revision,kind,payload_json,created_at) VALUES (?,?,?,?,?)", (entity_id, revision, kind, canonical(redact_secrets(payload)), now()))

    def project(self, project_id: str) -> dict[str, Any]:
        with self.db.lock:
            item = self._row("projects", "project_id", project_id)
            item["tags"] = self._tags("project", project_id)
            counts = self.conn.execute("SELECT COUNT(*) AS tasks, SUM(CASE WHEN archived_at IS NULL THEN 1 ELSE 0 END) AS visible_tasks FROM tasks WHERE project_id=?", (project_id,)).fetchone()
            item.update(dict(counts))
            item["usage"] = dict(self.conn.execute("SELECT COALESCE(SUM(known_tokens),0) AS known_tokens,COALESCE(SUM(model_calls),0) AS model_calls,COALESCE(SUM(unsettled_requests),0) AS unsettled_requests FROM run_usage JOIN runs USING(run_id) JOIN tasks USING(task_id) WHERE tasks.project_id=?", (project_id,)).fetchone())
            return item

    def list(self, archived: bool = False) -> list[dict[str, Any]]:
        with self.db.lock:
            return [self.project(row[0]) for row in self.conn.execute("SELECT project_id FROM projects WHERE (? OR archived_at IS NULL) ORDER BY created_at,project_id", (archived,)).fetchall()]

    def create(self, request: ProjectCreate) -> dict[str, Any]:
        identifier, timestamp = "project_" + uuid.uuid4().hex, now()
        with self.db.transaction():
            self.conn.execute("INSERT INTO projects(project_id,name,description,created_at,updated_at) VALUES (?,?,?,?,?)", (identifier, request.name, request.description, timestamp, timestamp))
            self._set_tags("project", identifier, request.tags)
            self._audit(identifier, 1, "project_created", request.model_dump())
            self._index_project(identifier)
        return self.project(identifier)

    def _expect(self, row: dict[str, Any], revision: int) -> None:
        if row["revision"] != revision:
            raise ServiceError("REVISION_CONFLICT", "记录已更新，请刷新后重试。", status_code=409)

    def _idle(self, task_id: str | None = None, project_id: str | None = None) -> None:
        row = self.conn.execute("SELECT runs.run_id FROM runs JOIN tasks USING(task_id) WHERE runs.status IN ('QUEUED','RUNNING','WAITING_INPUT','CANCEL_REQUESTED') AND (? IS NULL OR tasks.task_id=?) AND (? IS NULL OR tasks.project_id=?) LIMIT 1", (task_id, task_id, project_id, project_id)).fetchone()
        if row:
            raise ServiceError("ACTIVE_RUN_EXISTS", "请先等待或中止活跃运行。", status_code=409)

    def patch_project(self, project_id: str, patch: ProjectPatch) -> dict[str, Any]:
        with self.db.transaction():
            row = self._row("projects", "project_id", project_id)
            self._expect(row, patch.expected_revision)
            if patch.name is not None and not patch.name.strip():
                raise ServiceError("INVALID_NAME", "名称不能为空。", status_code=422)
            self.conn.execute("UPDATE projects SET name=?,description=?,revision=revision+1,updated_at=? WHERE project_id=?", (patch.name.strip() if patch.name is not None else row["name"], patch.description if patch.description is not None else row["description"], now(), project_id))
            if patch.tags is not None:
                self._set_tags("project", project_id, patch.tags)
            self._audit(project_id, row["revision"] + 1, "project_updated", patch.model_dump())
            self._index_project(project_id)
        return self.project(project_id)

    def archive_project(self, project_id: str, revision: int, archived: bool) -> dict[str, Any]:
        with self.db.transaction():
            row = self._row("projects", "project_id", project_id)
            self._expect(row, revision)
            if project_id == UNCLASSIFIED:
                raise ServiceError("DEFAULT_PROJECT", "未分类课题不能归档。", status_code=409)
            self._idle(project_id=project_id)
            self.conn.execute("UPDATE projects SET archived_at=?,revision=revision+1,updated_at=? WHERE project_id=?", (now() if archived else None, now(), project_id))
            self._audit(project_id, revision + 1, "project_archived" if archived else "project_unarchived", {})
        return self.project(project_id)

    def edit_task(self, task_id: str, patch: TaskPatch | None = None, *, revision: int | None = None, project_id: str | None = None, target_revision: int | None = None, archived: bool | None = None) -> dict[str, Any]:
        with self.db.transaction():
            row = self._row("tasks", "task_id", task_id)
            self._expect(row, patch.expected_revision if patch else int(revision or 0))
            self._idle(task_id=task_id)
            if project_id is not None:
                target = self._row("projects", "project_id", project_id)
                self._expect(target, int(target_revision or 0))
                if target["archived_at"]:
                    raise ServiceError("PROJECT_ARCHIVED", "目标课题已归档。", status_code=409)
                row["project_id"] = project_id
            if patch:
                if patch.title is not None:
                    if not patch.title.strip():
                        raise ServiceError("INVALID_NAME", "标题不能为空。", status_code=422)
                    row["title"] = patch.title.strip()
                if patch.management_status:
                    row["management_status"] = patch.management_status
                if patch.tags is not None:
                    self._set_tags("task", task_id, patch.tags)
            if archived is not None:
                row["archived_at"] = now() if archived else None
            self.conn.execute("UPDATE tasks SET title=?,project_id=?,management_status=?,archived_at=?,updated_at=?,revision=revision+1 WHERE task_id=?", (row["title"], row["project_id"], row["management_status"], row["archived_at"], now(), task_id))
            self.conn.execute("UPDATE search_documents SET project_id=? WHERE task_id=?", (row["project_id"], task_id))
            self._audit(task_id, row["revision"] + 1, "task_updated", {"project_id": row["project_id"], "archived_at": row["archived_at"], "patch": patch.model_dump() if patch else None})
            self.index_task(task_id)
            result = self._row("tasks", "task_id", task_id)
            result["tags"] = self._tags("task", task_id)
            return result

    def require_runnable(self, task_id: str) -> None:
        task = self._row("tasks", "task_id", task_id)
        project = self._row("projects", "project_id", task["project_id"])
        if task["archived_at"] or project["archived_at"] or task["management_status"] != "OPEN":
            raise ServiceError("TASK_NOT_OPEN", "请先解除归档并将任务设为 OPEN。", status_code=409)

    def index_document(self, identifier: str, kind: str, title: str, text: str, project_id: str | None = None, task_id: str | None = None) -> None:
        from photomatagent.redaction import redact_text
        text, title = redact_text(text), redact_text(title)
        digest = hashlib.sha256(text.encode()).hexdigest()
        self.conn.execute("INSERT INTO search_documents VALUES (?,?,?,?,?,?,?) ON CONFLICT(document_id) DO UPDATE SET project_id=excluded.project_id,task_id=excluded.task_id,title=excluded.title,text=excluded.text,source_digest=excluded.source_digest", (identifier, project_id, task_id, kind, title, text, digest))

    def index_task(self, task_id: str) -> None:
        row = self._row("tasks", "task_id", task_id)
        self.index_document(task_id, "task", row["title"], row["goal"] + " " + " ".join(self._tags("task", task_id)), row["project_id"], task_id)

    def _index_project(self, project_id: str) -> None:
        row = self._row("projects", "project_id", project_id)
        self.index_document(project_id, "project", row["name"], row["description"] + " " + " ".join(self._tags("project", project_id)), project_id)

    def task_page(self, *, project_id: str | None = None, management_status: str | None = None, archived: bool = False, tag: str | None = None, q: str = "", cursor: str | None = None, limit: int = 50) -> dict[str, Any]:
        limit = min(max(limit, 1), 200)
        binding = hashlib.sha256(canonical([project_id, management_status, archived, tag, q]).encode()).hexdigest()
        conditions = ["(? IS NULL OR t.project_id=?)", "(? IS NULL OR t.management_status=?)", "(? OR t.archived_at IS NULL)", "(? IS NULL OR EXISTS(SELECT 1 FROM task_tags tt JOIN tags g USING(tag_id) WHERE tt.task_id=t.task_id AND g.name=?))", "(t.title LIKE ? ESCAPE '!' OR t.goal LIKE ? ESCAPE '!')"]
        pattern = "%" + q.replace("!", "!!").replace("%", "!%").replace("_", "!_") + "%"
        params: list[Any] = [project_id, project_id, management_status, management_status, archived, tag, tag.casefold() if tag else None, pattern, pattern]
        if cursor:
            try:
                decoded = json.loads(base64.urlsafe_b64decode(cursor.encode()))
                if decoded[0] != binding:
                    raise ValueError("scope changed")
                conditions.append("(s.last_activity_at,t.task_id)<(?,?)")
                params += [decoded[1], decoded[2]]
            except (ValueError, IndexError, TypeError, KeyError) as exc:
                raise ServiceError("INVALID_CURSOR", "分页游标无效或筛选条件已变化。", status_code=422) from exc
        with self.db.lock:
            rows = self.conn.execute("SELECT t.*,s.last_activity_at,r.run_id AS latest_id,r.request_json AS latest_request,r.status AS latest_status,r.created_at AS latest_created FROM tasks t JOIN task_summaries s USING(task_id) LEFT JOIN runs r ON r.run_id=s.latest_run_id WHERE " + " AND ".join(conditions) + " ORDER BY s.last_activity_at DESC,t.task_id DESC LIMIT ?", (*params, limit + 1)).fetchall()
            items = []
            identifiers = [row["task_id"] for row in rows[:limit]]
            tag_map: dict[str, list[str]] = {}
            if identifiers:
                placeholders = ",".join("?" for _ in identifiers)
                for tag_row in self.conn.execute(f"SELECT task_id,name FROM task_tags JOIN tags USING(tag_id) WHERE task_id IN ({placeholders}) ORDER BY name", identifiers):
                    tag_map.setdefault(tag_row[0], []).append(tag_row[1])
            for row in rows[:limit]:
                item = dict(row)
                item["attachment_ids"] = json.loads(item.pop("attachment_ids_json"))
                item["tags"] = tag_map.get(item["task_id"], [])
                item["latest_run"] = {"run_id": row["latest_id"], "task_id": row["task_id"], "request": json.loads(row["latest_request"]), "status": row["latest_status"], "created_at": row["latest_created"]} if row["latest_id"] else None
                for key in [key for key in item if key.startswith("latest_")]:
                    if key != "latest_run":
                        item.pop(key)
                items.append(item)
            next_cursor = None
            if len(rows) > limit:
                last = rows[limit - 1]
                next_cursor = base64.urlsafe_b64encode(canonical([binding, last["last_activity_at"], last["task_id"]]).encode()).decode()
            return {"items": items, "next_cursor": next_cursor}

    def search(self, q: str, *, project_id: str | None = None, limit: int = 30) -> dict[str, Any]:
        if not q.strip() or len(q) > 500:
            raise ServiceError("INVALID_QUERY", "检索词不能为空或超过 500 字。", status_code=422)
        pattern = "%" + q.replace("!", "!!").replace("%", "!%").replace("_", "!_") + "%"
        with self.db.lock:
            fts = len(q) >= 3 and bool(self.conn.execute("SELECT 1 FROM sqlite_master WHERE name='search_fts'").fetchone())
            restriction = " AND rowid IN (SELECT rowid FROM search_fts WHERE search_fts MATCH ?)" if fts else ""
            params: list[Any] = [project_id, project_id, pattern, pattern]
            if fts:
                params.append('"' + q.replace('"', '""') + '"')
            rows = self.conn.execute("SELECT document_id,project_id,task_id,kind,title,substr(text,1,1600) AS excerpt,source_digest FROM search_documents WHERE (? IS NULL OR project_id=?) AND (title LIKE ? ESCAPE '!' OR text LIKE ? ESCAPE '!')" + restriction + " ORDER BY document_id LIMIT ?", (*params, min(max(limit, 1), 100))).fetchall()
            return {"items": [dict(row) for row in rows], "mode": "trigram" if fts else "literal", "diagnostic": "Literal substring search; no semantic or chemical-equivalence inference."}
