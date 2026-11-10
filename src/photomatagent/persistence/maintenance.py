"""Explicit, local maintenance of authoritative data and rebuildable indexes."""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

from photomatagent.application.projects import canonical, now
from photomatagent.application.errors import ServiceError
from photomatagent.application.store import WorkbenchStore
from photomatagent.persistence.database import OwnershipLock
from photomatagent.redaction import redact_secrets
from photomatagent.sessions.sqlite_store import SessionRepository
from photomatagent.sessions.store import parse_session_snapshot
from photomatagent.workspace import Workspace
from photomatagent.errors import ToolError


class StorageService:
    def __init__(self, store: WorkbenchStore, root: Path) -> None:
        self.store = store
        self.db = store.database
        self.conn = self.db.connection
        self.workspace = Workspace(root)
        self.sessions = SessionRepository(self.db)

    def status(self, deep: bool = False) -> dict[str, Any]:
        with self.db.lock:
            tables = ("projects","tasks","runs","sessions","messages","checkpoints","runtime_events","source_records","tool_operations")
            return {"schema_version":self.conn.execute("SELECT version FROM schema_version").fetchone()[0], "counts":{table:self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in tables}, "integrity":[row[0] for row in self.conn.execute("PRAGMA integrity_check" if deep else "PRAGMA quick_check")], "foreign_key_errors":[list(row) for row in self.conn.execute("PRAGMA foreign_key_check")], "unresolved_operations":[dict(row) for row in self.conn.execute("SELECT operation_id,session_id,tool_name,status FROM tool_operations WHERE status!='CHECKPOINTED' LIMIT 200")], "stale_sources":self.conn.execute("SELECT COUNT(*) FROM source_records WHERE freshness!='CURRENT'").fetchone()[0], "search_mode":"literal"}

    def diagnostics(self) -> dict[str, Any]:
        report = self.status()
        with self.db.lock:
            report["import_diagnostics"] = [dict(row) for row in self.conn.execute("SELECT source_id,status,diagnostic FROM import_sources WHERE status IN ('CONFLICT','RECONCILIATION_REQUIRED') ORDER BY source_id")]
        issues = []
        with self.db.lock:
            sessions = self.conn.execute("SELECT session_id FROM sessions WHERE latest_checkpoint_id IS NOT NULL").fetchall()
            artifacts = self.conn.execute("SELECT artifact_id,path,sha256 FROM artifacts").fetchall()
        for row in sessions:
            try:
                self.sessions.load(row[0])
            except (ValueError, OSError, ToolError) as exc:
                issues.append({"session_id":row[0],"code":type(exc).__name__})
        for row in artifacts:
            try:
                path = self.workspace.resolve(row["path"],must_exist=False)
                if path.is_symlink() or not path.is_file() or _sha(path) != row["sha256"]:
                    issues.append({"artifact_id":row["artifact_id"],"code":"ARTIFACT_INTEGRITY_UNKNOWN"})
            except (ValueError, OSError, ToolError):
                issues.append({"artifact_id":row["artifact_id"],"code":"ARTIFACT_UNAVAILABLE"})
        report["issues"] = issues
        return report

    def import_sessions(self) -> dict[str, Any]:
        root = self.workspace.root / ".photomatagent/sessions"
        results = []
        for directory in sorted(root.iterdir()) if root.is_dir() else []:
            if directory.is_symlink() or not directory.is_dir():
                continue
            source = directory / "session_state.json"
            events_file = directory / "events.jsonl"
            has_snapshot = source.is_file() and not source.is_symlink()
            has_events = events_file.is_file() and not events_file.is_symlink() and events_file.stat().st_size > 0
            if not has_snapshot and not has_events:
                continue
            key = "session:" + directory.name
            checksum = ""
            try:
                if has_snapshot and source.stat().st_size > 50 * 1024 * 1024:
                    raise ValueError("SOURCE_TOO_LARGE")
                raw = source.read_bytes() if has_snapshot else b""
                event_digest = _sha(events_file) if has_events else ""
                checksum = hashlib.sha256(raw + b"\0" + event_digest.encode()).hexdigest()
                with self.db.lock:
                    prior = self.conn.execute("SELECT digest,status FROM import_sources WHERE source_id=?", (key,)).fetchone()
                    session = self.conn.execute("SELECT backend FROM sessions WHERE session_id=?", (directory.name,)).fetchone()
                if prior and prior[0] == checksum and prior[1] in {"IMPORTED", "TRACE_ONLY", "READ_ONLY", "RECONCILIATION_REQUIRED"}:
                    results.append({"session_id":directory.name,"status":"UNCHANGED"})
                    continue
                if prior and prior[0] != checksum:
                    raise ValueError("SOURCE_CHANGED")
                if session and session[0] == "sqlite":
                    results.append({"session_id":directory.name,"status":"DATABASE_AUTHORITY"})
                    continue
                snapshot = parse_session_snapshot(raw) if has_snapshot else None
                diagnostic = None
                if snapshot is not None:
                    from photomatagent.runtime.context_segments import split_atomic_spans
                    try:
                        split_atomic_spans(snapshot.conversation.messages)
                    except ValueError:
                        # Keep the raw snapshot and import its trace, but never
                        # fabricate results or resume a partially executed group.
                        snapshot = None
                        diagnostic = "INCOMPLETE_TOOL_GROUP: original snapshot retained; reconcile tool effects before continuing"
                if has_snapshot and source.read_bytes() != raw:
                    raise ValueError("SOURCE_CHANGED_DURING_IMPORT")
                # Historical source links remain read-only. Ordinary task sessions may migrate.
                with self.db.lock:
                    linked = self.conn.execute("SELECT 1 FROM links WHERE source_type='runtime_session' AND source_id=?", (directory.name,)).fetchone()
                    task = self.conn.execute("SELECT task_id FROM tasks WHERE session_id=?", (directory.name,)).fetchone()
                if linked:
                    self.sessions.ensure(directory.name,backend="legacy_files")
                    state = "READ_ONLY"
                else:
                    from photomatagent.observability.trace import load_trace
                    events = load_trace(directory).events if has_events else ()
                    with self.db.transaction():
                        self.sessions.ensure(directory.name,task[0] if task else None)
                        for event in events:
                            self.sessions.event(directory.name,event)
                        if snapshot is not None:
                            self.sessions.checkpoint(directory.name,snapshot,boundary="legacy_import")
                        if has_events and _sha(events_file) != event_digest:
                            raise ValueError("EVENT_SOURCE_CHANGED_DURING_IMPORT")
                    state = "RECONCILIATION_REQUIRED" if diagnostic else ("IMPORTED" if snapshot is not None else "TRACE_ONLY")
                with self.db.transaction():
                    self.conn.execute("INSERT INTO import_sources VALUES (?,?,?,?,?) ON CONFLICT(source_id) DO UPDATE SET status=excluded.status,diagnostic=excluded.diagnostic,updated_at=excluded.updated_at", (key,checksum,state,diagnostic,now()))
                results.append({"session_id":directory.name,"status":state,**({"diagnostic":diagnostic} if diagnostic else {})})
            except (ValueError,OSError,ToolError) as exc:
                with self.db.transaction():
                    self.conn.execute("INSERT INTO import_sources VALUES (?,?,?,?,?) ON CONFLICT(source_id) DO UPDATE SET status=excluded.status,diagnostic=excluded.diagnostic,updated_at=excluded.updated_at", (key,checksum,"CONFLICT",str(exc)[:200],now()))
                results.append({"session_id":directory.name,"status":"CONFLICT","diagnostic":str(exc)[:200]})
        return {"items":results}

    def reindex(self) -> dict[str, Any]:
        with self.db.transaction():
            self.conn.execute("DELETE FROM search_documents")
            for row in self.conn.execute("SELECT project_id FROM projects").fetchall():
                self.store.projects._index_project(row[0])
            for row in self.conn.execute("SELECT task_id FROM tasks").fetchall():
                self.store.projects.index_task(row[0])
            for row in self.conn.execute("SELECT messages.*,tasks.project_id,sessions.task_id FROM messages JOIN sessions USING(session_id) JOIN tasks USING(task_id)").fetchall():
                payload = json.loads(row["payload_json"])
                text = payload.get("content") or payload.get("text")
                if isinstance(text,str):
                    self.store.projects.index_document(f"message:{row['session_id']}:{row['message_seq']}","message",f"Message {row['message_seq']}",text,row["project_id"],row["task_id"])
            for row in self.conn.execute("SELECT artifacts.*,tasks.project_id FROM artifacts JOIN tasks USING(task_id)").fetchall():
                self.store.projects.index_document("artifact:"+row["artifact_id"],"artifact",row["filename"],row["path"],row["project_id"],row["task_id"])
            for session in self.conn.execute("SELECT sessions.session_id,sessions.task_id,tasks.project_id FROM sessions JOIN tasks USING(task_id) WHERE latest_checkpoint_id IS NOT NULL").fetchall():
                state = self.sessions.load(session["session_id"])
                for evidence in state.scientific.evidence:
                    text = canonical(evidence.model_dump(mode="json"))
                    source_id = hashlib.sha256(text.encode()).hexdigest()
                    self.store.projects.index_document(f"evidence:{session['session_id']}:{source_id}","evidence","Historical evidence; requires validation",text,session["project_id"],session["task_id"])
        return self.refresh()

    def index_run(self, run_id: str) -> None:
        run = self.store.run(run_id)
        task = self.store.task(run.task_id)
        with self.db.transaction():
            for row in self.conn.execute("SELECT evaluation_id,manifest_json,report_json,status FROM evaluations WHERE run_id=?", (run_id,)).fetchall():
                self.store.projects.index_document("evaluation:"+row["evaluation_id"],"evaluation",row["status"],canonical(dict(row)),task.project_id,task.task_id)
        for relative in (".photomatagent/state/jobs.sqlite3", ".photomatagent/vasp/jobs.sqlite3"):
            path = self.workspace.resolve(relative,must_exist=False)
            if not path.is_file() or path.is_symlink():
                continue
            with sqlite3.connect(f"file:{path}?mode=ro",uri=True) as registry:
                if not registry.execute("SELECT 1 FROM sqlite_master WHERE name='job_registry'").fetchone():
                    continue
                rows = registry.execute("SELECT request_id,job_id,state FROM job_registry WHERE json_extract(provenance_json,'$.run_id')=? OR json_extract(provenance_json,'$.task_id')=?", (run_id,task.task_id)).fetchall()
            with self.db.transaction():
                for request_id,job_id,state in rows:
                    prior = self.conn.execute("SELECT job_id FROM run_job_links WHERE run_id=? AND registry_key=? AND request_id=?",(run_id,relative,request_id)).fetchone()
                    if prior and prior[0] and job_id!=prior[0]:
                        raise ValueError("JOB_ID_CHANGED: authoritative reconciliation required")
                    self.conn.execute("INSERT INTO run_job_links VALUES (?,?,?,?,?,?) ON CONFLICT(run_id,registry_key,request_id) DO UPDATE SET job_id=excluded.job_id,state=excluded.state,observed_at=excluded.observed_at",(run_id,relative,request_id,job_id,state,now()))

    def refresh(self, prefixes: tuple[str, ...] | None = None) -> dict[str, Any]:
        indexed, conflicts = 0, []
        metadata = self.workspace.root / ".photomatagent"
        # Only explicit authoritative record directories; never scan user_input/config/secrets.
        for name in prefixes or ("local-skills","procedural-skills","executions","evolutions","qd","capability-executions","capability-evolution","capability-benchmarks"):
            root = metadata / name
            for path in sorted([*root.rglob("*.json"),*root.rglob("*.jsonl")]) if root.is_dir() and not root.is_symlink() else []:
                try:
                    relative = str(path.relative_to(self.workspace.root))
                    self.workspace.resolve(relative)
                    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != self.workspace.root) or path.stat().st_size > 50*1024*1024:
                        continue
                    raw = path.read_bytes()
                    checksum = hashlib.sha256(raw).hexdigest()
                    consumed = len(raw)
                    if path.suffix == ".jsonl":
                        with self.db.lock:
                            cursor = self.conn.execute("SELECT imported_bytes,prefix_sha256 FROM source_cursors WHERE source_key=?",(relative,)).fetchone()
                        if cursor and (len(raw)<cursor[0] or hashlib.sha256(raw[:cursor[0]]).hexdigest()!=cursor[1]):
                            with self.db.transaction():
                                self.conn.execute("UPDATE source_records SET freshness='CONFLICT' WHERE source_key=?",(relative,))
                            raise ValueError("SOURCE_PREFIX_CHANGED")
                        consumed = raw.rfind(b"\n")+1
                        payload = redact_secrets([json.loads(line) for line in raw[:consumed].splitlines() if line.strip()])
                    else:
                        payload = redact_secrets(json.loads(raw))
                    if path.read_bytes() != raw:
                        raise ValueError("SOURCE_CHANGED_DURING_INDEX")
                    with self.db.transaction():
                        self.conn.execute("INSERT INTO source_records VALUES (?,?,?,?,?,?,?,'CURRENT') ON CONFLICT(source_key) DO UPDATE SET digest=excluded.digest,payload_json=excluded.payload_json,observed_at=excluded.observed_at,freshness='CURRENT'", (relative,name,path.stem,relative,checksum,canonical(payload),now()))
                        if path.suffix == ".jsonl":
                            self.conn.execute("INSERT INTO source_cursors VALUES (?,?,?) ON CONFLICT(source_key) DO UPDATE SET imported_bytes=excluded.imported_bytes,prefix_sha256=excluded.prefix_sha256",(relative,consumed,hashlib.sha256(raw[:consumed]).hexdigest()))
                            if consumed!=len(raw):
                                self.conn.execute("UPDATE source_records SET freshness='PARTIAL_TAIL' WHERE source_key=?",(relative,))
                        task_id = payload.get("task_id") if isinstance(payload,dict) else None
                        if task_id is None and isinstance(payload,dict) and isinstance(payload.get("source"),dict):
                            task_id = payload["source"].get("task_id")
                        task = self.conn.execute("SELECT project_id FROM tasks WHERE task_id=?",(task_id,)).fetchone() if isinstance(task_id,str) else None
                        self.store.projects.index_document("source:"+relative,name,path.name,canonical(payload)[:100000],task[0] if task else None,task_id if task else None)
                        if task:
                            self.conn.execute("INSERT OR IGNORE INTO source_task_links VALUES (?,?)",(relative,task_id))
                    indexed += 1
                except (ValueError,OSError,ToolError):
                    relative = str(path.relative_to(self.workspace.root))
                    with self.db.transaction():
                        self.conn.execute("UPDATE source_records SET freshness='UNREADABLE' WHERE source_key=? AND freshness!='CONFLICT'", (relative,))
                    conflicts.append(relative)
        with self.db.transaction():
            for row in self.conn.execute("SELECT source_key,path,digest FROM source_records").fetchall():
                path = self.workspace.resolve(row["path"],must_exist=False)
                if not path.is_file() or path.is_symlink():
                    self.conn.execute("UPDATE source_records SET freshness='MISSING' WHERE source_key=?",(row["source_key"],))
        return {"indexed_sources":indexed,"conflicts":conflicts,"observed_at":now()}

    def backup(self, full: bool = False) -> dict[str, Any]:
        owner = OwnershipLock(self.store.database_path.with_suffix(".owner.lock")) if full else None
        if owner:
            owner.acquire()
        from contextlib import ExitStack
        guards = ExitStack()
        if full:
            try:
                for prefix in ("sessions","evolutions","qd","local-skills","procedural-skills","executions"):
                    directory = self.workspace.root/".photomatagent"/prefix
                    for lock_path in sorted(directory.rglob("*.lock")) if directory.is_dir() and not directory.is_symlink() else []:
                        if lock_path.is_symlink():
                            raise ValueError("UNSAFE_LOCK_PATH")
                        guards.enter_context(OwnershipLock(lock_path))
            except BaseException:
                guards.close()
                if owner:
                    owner.release()
                raise
        destination = self.workspace.root / ".photomatagent/backups" / ("backup_"+uuid4().hex)
        try:
            destination.mkdir(parents=True)
            with self.db.lock, sqlite3.connect(destination/"workbench.sqlite3") as target:
                self.conn.backup(target)
            files: dict[str,str] = {"workbench.sqlite3":_sha(destination/"workbench.sqlite3")}
            missing = []
            if full:
                # Caller must stop independent specialized writers as well.
                paths: set[Path] = set()
                for prefix in ("sessions","evolutions","qd","local-skills","procedural-skills","executions","capability-executions","capability-evolution","capability-benchmarks"):
                    directory = self.workspace.root/".photomatagent"/prefix
                    if directory.is_dir() and not directory.is_symlink():
                        paths.update(path for path in directory.rglob("*") if path.is_file())
                with self.db.lock:
                    for row in self.conn.execute("SELECT path FROM artifacts"):
                        paths.add(self.workspace.resolve(row[0],must_exist=False))
                for path in sorted(paths):
                    relative = str(path.relative_to(self.workspace.root))
                    self.workspace.resolve(relative)
                    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent!=self.workspace.root):
                        missing.append(relative)
                        continue
                    if not path.is_file():
                        missing.append(relative)
                        continue
                    checksum = _sha(path)
                    target_file = destination/"files"/relative
                    target_file.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(path,target_file)
                    if _sha(target_file)!=checksum or _sha(path)!=checksum:
                        raise ValueError("SOURCE_CHANGED_DURING_BACKUP")
                    files["files/"+relative]=checksum
                for relative in (".photomatagent/state/jobs.sqlite3",".photomatagent/vasp/jobs.sqlite3"):
                    source = self.workspace.resolve(relative,must_exist=False)
                    if source.is_file() and not source.is_symlink():
                        target_file = destination/"files"/relative
                        target_file.parent.mkdir(parents=True,exist_ok=True)
                        with sqlite3.connect(f"file:{source}?mode=ro",uri=True) as origin, sqlite3.connect(target_file) as target_db:
                            origin.backup(target_db)
                        files["files/"+relative]=_sha(target_file)
            if full:
                for relative,checksum in files.items():
                    if relative.startswith("files/") and not relative.endswith(".sqlite3"):
                        original = self.workspace.resolve(relative[6:])
                        if _sha(original)!=checksum:
                            raise ValueError("SOURCE_CHANGED_DURING_BACKUP")
            report = {"kind":"workspace" if full else "database_snapshot","complete_local":bool(full and not missing),"missing_local":missing,"remote_artifacts":"Not included; reconcile remote jobs after restore.","created_at":now(),"files":files}
            (destination/"manifest.json").write_text(canonical(report))
            return {"path":str(destination.relative_to(self.workspace.root)),**report}
        finally:
            guards.close()
            if owner:
                owner.release()

    def export(self) -> dict[str, Any]:
        destination = self.workspace.root/"user_output"/"information-export"/("export_"+uuid4().hex)
        with self.db.lock:
            sessions = self.conn.execute("SELECT session_id FROM sessions WHERE latest_checkpoint_id IS NOT NULL").fetchall()
        for row in sessions:
            self.sessions.export(row[0],destination/"sessions"/row[0])
        destination.mkdir(parents=True,exist_ok=True)
        with self.db.lock:
            content = {name:[dict(row) for row in self.conn.execute(f"SELECT * FROM {name}")] for name in ("projects","tasks","runs","management_events","links","artifacts")}
        (destination/"catalog.json").write_text(canonical(redact_secrets(content)))
        return {"path":str(destination.relative_to(self.workspace.root)),"sessions":len(sessions)}


def _sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024*1024):
            value.update(chunk)
    return value.hexdigest()


def restore_preview(root: Path, backup: Path, destination: Path) -> dict[str, Any]:
    """Restore only to a new directory, never overwrite a live workspace."""
    workspace = Workspace(root)
    backup = workspace.resolve(str(backup))
    destination = workspace.resolve(str(destination),must_exist=False)
    if destination.exists():
        raise ValueError("RESTORE_DESTINATION_EXISTS")
    manifest = json.loads((backup/"manifest.json").read_text())
    if not isinstance(manifest.get("files"),dict) or "workbench.sqlite3" not in manifest["files"]:
        raise ValueError("INVALID_BACKUP_MANIFEST")
    from photomatagent.persistence.migrations import VERSION
    with sqlite3.connect(f"file:{backup / 'workbench.sqlite3'}?mode=ro",uri=True) as source:
        version = source.execute("SELECT version FROM schema_version WHERE id=1").fetchone()
        if version is None or version[0] > VERSION:
            raise ValueError("DATABASE_TOO_NEW")
    with OwnershipLock(root/".photomatagent/workbench.owner.lock"):
        for relative,checksum in manifest["files"].items():
            path = Workspace(backup).resolve(relative)
            if path.is_symlink() or _sha(path)!=checksum:
                raise ValueError("BACKUP_INTEGRITY_FAILED")
        destination.mkdir(parents=True)
        for relative in manifest["files"]:
            if relative=="workbench.sqlite3":
                target = destination/".photomatagent/workbench.sqlite3"
            elif relative.startswith("files/"):
                target = Workspace(destination).resolve(relative[6:],must_exist=False)
            else:
                raise ValueError("INVALID_BACKUP_ENTRY")
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(backup/relative,target)
        with sqlite3.connect(destination/".photomatagent/workbench.sqlite3") as conn:
            if conn.execute("PRAGMA integrity_check").fetchone()[0]!="ok" or conn.execute("PRAGMA foreign_key_check").fetchall():
                raise ValueError("RESTORE_DATABASE_INVALID")
        return {"path":str(destination.relative_to(root)),"activated":False,"reconciliation_required":True}
