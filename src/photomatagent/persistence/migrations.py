"""Additive information-management migration, preserving all legacy rows."""
from __future__ import annotations
from photomatagent.persistence.database import Database

VERSION = 9
UNCLASSIFIED = "project_unclassified"

SCHEMA = [
    "CREATE TABLE projects (project_id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, updated_at TEXT NOT NULL, archived_at TEXT, revision INTEGER NOT NULL DEFAULT 1)",
    "CREATE TABLE tags (tag_id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE)",
    "CREATE TABLE task_tags (task_id TEXT NOT NULL REFERENCES tasks(task_id), tag_id INTEGER NOT NULL REFERENCES tags(tag_id), PRIMARY KEY(task_id,tag_id))",
    "CREATE TABLE project_tags (project_id TEXT NOT NULL REFERENCES projects(project_id), tag_id INTEGER NOT NULL REFERENCES tags(tag_id), PRIMARY KEY(project_id,tag_id))",
    "CREATE TABLE management_events (event_id INTEGER PRIMARY KEY, entity_id TEXT NOT NULL, revision INTEGER NOT NULL, kind TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL)",
    "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, task_id TEXT UNIQUE REFERENCES tasks(task_id), backend TEXT NOT NULL CHECK(backend IN ('sqlite','legacy_files')), parent_session_id TEXT, latest_checkpoint_id TEXT, created_at TEXT NOT NULL)",
    "CREATE TABLE messages (session_id TEXT NOT NULL REFERENCES sessions(session_id), message_seq INTEGER NOT NULL, run_id TEXT, payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL, PRIMARY KEY(session_id,message_seq))",
    "CREATE TABLE runtime_events (session_id TEXT NOT NULL REFERENCES sessions(session_id), session_seq INTEGER NOT NULL, event_id TEXT NOT NULL UNIQUE, run_id TEXT, kind TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(session_id,session_seq))",
    "CREATE TABLE state_payloads (payload_sha256 TEXT PRIMARY KEY, payload_json TEXT NOT NULL)",
    "CREATE TABLE checkpoints (checkpoint_id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(session_id), run_id TEXT, parent_checkpoint_id TEXT REFERENCES checkpoints(checkpoint_id), message_end_seq INTEGER NOT NULL, runtime_event_end_seq INTEGER NOT NULL, boundary_kind TEXT NOT NULL, manifest_json TEXT NOT NULL, sha256 TEXT NOT NULL, created_at TEXT NOT NULL)",
    "CREATE TABLE checkpoint_components (checkpoint_id TEXT NOT NULL REFERENCES checkpoints(checkpoint_id), component_kind TEXT NOT NULL, payload_sha256 TEXT NOT NULL REFERENCES state_payloads(payload_sha256), PRIMARY KEY(checkpoint_id,component_kind))",
    "CREATE TABLE tool_operations (operation_id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(session_id), run_id TEXT, tool_call_id TEXT NOT NULL, tool_name TEXT NOT NULL, arguments_sha256 TEXT NOT NULL, status TEXT NOT NULL, result_json TEXT, created_at TEXT NOT NULL)",
    "CREATE TABLE approval_records (request_id TEXT NOT NULL, run_id TEXT NOT NULL, status TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(request_id,run_id))",
    "CREATE TABLE run_job_links (run_id TEXT NOT NULL REFERENCES runs(run_id), registry_key TEXT NOT NULL, request_id TEXT NOT NULL, job_id TEXT, state TEXT, observed_at TEXT NOT NULL, PRIMARY KEY(run_id,registry_key,request_id))",
    "CREATE TABLE run_usage (run_id TEXT PRIMARY KEY REFERENCES runs(run_id), model_calls INTEGER NOT NULL DEFAULT 0, known_tokens INTEGER NOT NULL DEFAULT 0, unsettled_requests INTEGER NOT NULL DEFAULT 0)",
    "CREATE TABLE task_summaries (task_id TEXT PRIMARY KEY REFERENCES tasks(task_id), latest_run_id TEXT, last_activity_at TEXT NOT NULL, last_event_seq INTEGER NOT NULL DEFAULT 0)",
    "CREATE TABLE source_records (source_key TEXT PRIMARY KEY, kind TEXT NOT NULL, source_id TEXT NOT NULL, path TEXT, digest TEXT NOT NULL, payload_json TEXT NOT NULL, observed_at TEXT NOT NULL, freshness TEXT NOT NULL DEFAULT 'CURRENT')",
    "CREATE TABLE source_task_links (source_key TEXT NOT NULL REFERENCES source_records(source_key), task_id TEXT NOT NULL REFERENCES tasks(task_id), PRIMARY KEY(source_key,task_id))",
    "CREATE TABLE search_documents (document_id TEXT PRIMARY KEY, project_id TEXT, task_id TEXT, kind TEXT NOT NULL, title TEXT NOT NULL, text TEXT NOT NULL, source_digest TEXT NOT NULL)",
    "CREATE TABLE source_cursors (source_key TEXT PRIMARY KEY REFERENCES source_records(source_key), imported_bytes INTEGER NOT NULL, prefix_sha256 TEXT NOT NULL)",
    "CREATE TABLE import_sources (source_id TEXT PRIMARY KEY, digest TEXT NOT NULL, status TEXT NOT NULL, diagnostic TEXT, updated_at TEXT NOT NULL)",
    "CREATE TABLE maintenance_operations (operation_id TEXT PRIMARY KEY, request_key TEXT UNIQUE, kind TEXT NOT NULL, status TEXT NOT NULL, result_json TEXT, created_at TEXT NOT NULL)",
    "CREATE TABLE migration_history (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)",
    "ALTER TABLE tasks ADD COLUMN project_id TEXT NOT NULL DEFAULT 'project_unclassified'",
    "CREATE TRIGGER task_project_insert BEFORE INSERT ON tasks WHEN NOT EXISTS(SELECT 1 FROM projects WHERE project_id=NEW.project_id) BEGIN SELECT RAISE(ABORT,'PROJECT_NOT_FOUND'); END",
    "CREATE TRIGGER task_project_update BEFORE UPDATE OF project_id ON tasks WHEN NOT EXISTS(SELECT 1 FROM projects WHERE project_id=NEW.project_id) BEGIN SELECT RAISE(ABORT,'PROJECT_NOT_FOUND'); END",
    "ALTER TABLE tasks ADD COLUMN management_status TEXT NOT NULL DEFAULT 'OPEN' CHECK(management_status IN ('OPEN','PAUSED','DONE'))",
    "ALTER TABLE tasks ADD COLUMN archived_at TEXT",
    "ALTER TABLE tasks ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE tasks ADD COLUMN revision INTEGER NOT NULL DEFAULT 1",
    "ALTER TABLE runs ADD COLUMN project_id_at_start TEXT",
    "ALTER TABLE runs ADD COLUMN base_checkpoint_id TEXT",
    "ALTER TABLE runs ADD COLUMN final_checkpoint_id TEXT",
    "ALTER TABLE runs ADD COLUMN recovery_status TEXT",
    "ALTER TABLE runs ADD COLUMN resumes_run_id TEXT",
    "ALTER TABLE artifacts ADD COLUMN origin_artifact_id TEXT REFERENCES artifacts(artifact_id)",
    "ALTER TABLE artifacts ADD COLUMN logical_name TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE artifacts ADD COLUMN version INTEGER NOT NULL DEFAULT 1",
    "ALTER TABLE artifacts ADD COLUMN integrity_status TEXT NOT NULL DEFAULT 'UNVERIFIED'",
    "CREATE TRIGGER artifact_version AFTER INSERT ON artifacts BEGIN UPDATE artifacts SET logical_name=NEW.filename,version=(SELECT COALESCE(MAX(version),0)+1 FROM artifacts WHERE task_id=NEW.task_id AND filename=NEW.filename AND artifact_id!=NEW.artifact_id) WHERE artifact_id=NEW.artifact_id; END",
    "CREATE INDEX task_project ON tasks(project_id,archived_at,management_status)",
    "CREATE INDEX run_task_time ON runs(task_id,created_at,run_id)",
    "CREATE INDEX event_run ON events(run_id,seq)",
    "CREATE INDEX event_kind ON events(task_id,type,seq)",
    "CREATE INDEX artifact_task_time ON artifacts(task_id,created_at,artifact_id)",
    "CREATE INDEX task_activity ON task_summaries(last_activity_at DESC,task_id DESC)",
    "CREATE INDEX checkpoint_session ON checkpoints(session_id,created_at)",
    "CREATE INDEX operation_session ON tool_operations(session_id,status)",
    "CREATE INDEX search_scope ON search_documents(project_id,kind)",
]


def upgrade(database: Database, now: str) -> None:
    conn = database.connection
    version = int(conn.execute("SELECT version FROM schema_version WHERE id=1").fetchone()[0])
    if version > VERSION:
        raise RuntimeError("DATABASE_TOO_NEW: upgrade the application before opening this database")
    if version == VERSION:
        return
    with database.transaction():
        # Older migration adapters can repair their own tables and advance the
        # legacy scalar version independently. The committed migration ledger
        # still proves that the additive schema was applied atomically.
        ledger = conn.execute("SELECT 1 FROM sqlite_master WHERE name='migration_history'").fetchone()
        if ledger and conn.execute("SELECT 1 FROM migration_history WHERE version=?", (VERSION,)).fetchone():
            import re
            for statement in SCHEMA:
                table = re.match(r"CREATE TABLE (\w+)", statement)
                if table and not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table[1],)).fetchone():
                    raise RuntimeError("MIGRATION_HISTORY_CONFLICT")
                column = re.match(r"ALTER TABLE (\w+) ADD COLUMN (\w+)", statement)
                if column and column[2] not in {row[1] for row in conn.execute(f"PRAGMA table_info({column[1]})")}:
                    raise RuntimeError("MIGRATION_HISTORY_CONFLICT")
            conn.execute("UPDATE schema_version SET version=? WHERE id=1", (VERSION,))
            return
        for statement in SCHEMA[:1]:
            conn.execute(statement)
        conn.execute("INSERT INTO projects(project_id,name,created_at,updated_at) VALUES (?,?,?,?)", (UNCLASSIFIED, "未分类", now, now))
        for statement in SCHEMA[1:]:
            conn.execute(statement)
        conn.execute("UPDATE tasks SET updated_at=created_at")
        conn.execute("UPDATE runs SET project_id_at_start=(SELECT project_id FROM tasks WHERE tasks.task_id=runs.task_id)")
        conn.execute("INSERT INTO task_summaries(task_id,last_activity_at,latest_run_id,last_event_seq) SELECT task_id,updated_at,(SELECT run_id FROM runs WHERE runs.task_id=tasks.task_id ORDER BY created_at DESC,run_id DESC LIMIT 1),(SELECT COALESCE(MAX(seq),0) FROM events WHERE events.task_id=tasks.task_id) FROM tasks")
        conn.execute("UPDATE artifacts SET logical_name=filename")
        conn.execute("INSERT INTO run_usage(run_id,model_calls,known_tokens,unsettled_requests) SELECT runs.run_id,SUM(CASE WHEN events.type='model_response_completed' THEN 1 ELSE 0 END),SUM(CASE WHEN events.type='model_response_completed' THEN COALESCE(json_extract(payload_json,'$.usage.total_tokens'),0) ELSE 0 END),MAX(SUM(CASE WHEN events.type='model_request_started' THEN 1 ELSE 0 END)-SUM(CASE WHEN events.type='model_response_completed' THEN 1 ELSE 0 END),0) FROM runs LEFT JOIN events USING(run_id) GROUP BY runs.run_id")
        conn.execute("INSERT INTO migration_history VALUES (?,?)", (VERSION, now))
        conn.execute("UPDATE schema_version SET version=? WHERE id=1", (VERSION,))
