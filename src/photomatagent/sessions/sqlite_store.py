"""Append-only conversation checkpoints and a durable tool-intent journal."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from photomatagent.application.projects import canonical, now, ProjectRepository
from photomatagent.persistence.database import Database
from photomatagent.redaction import redact_secrets
from photomatagent.sessions.store import SessionSnapshot, parse_session_snapshot


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class SessionRepository:
    def __init__(self, database: Database) -> None:
        self.db = database
        self.conn = database.connection

    def ensure(self, session_id: str, task_id: str | None = None, backend: str = "sqlite") -> None:
        if Path(session_id).name != session_id or session_id in {".", ".."}:
            raise ValueError("INVALID_SESSION_ID")
        with self.db.transaction():
            self.conn.execute("INSERT OR IGNORE INTO sessions(session_id,task_id,backend,created_at) VALUES (?,?,?,?)", (session_id, task_id, backend, now()))
            row = self.conn.execute("SELECT task_id FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            if task_id is not None and row[0] not in (None, task_id):
                raise ValueError("SESSION_TASK_CONFLICT")
            if task_id is not None:
                self.conn.execute("UPDATE sessions SET task_id=? WHERE session_id=?", (task_id, session_id))

    def checkpoint(self, session_id: str, snapshot: SessionSnapshot, *, run_id: str | None = None, boundary: str = "safe", components: dict[str, Any] | None = None) -> str:
        from photomatagent.runtime.context_segments import split_atomic_spans
        split_atomic_spans(snapshot.conversation.messages)
        from photomatagent.runtime.loop import AgentRuntime
        AgentRuntime._validate_engine_snapshot(snapshot.conversation.messages, snapshot.engine)
        payload = redact_secrets(json.loads(snapshot.model_dump_json()))
        messages = payload.pop("conversation")["messages"]
        scientific_payload = payload.pop("scientific")
        engine_payload = payload.pop("engine", None)
        parts = {"snapshot": payload, "scientific": scientific_payload, "engine": engine_payload, **(components or {})}
        serialized = {key: canonical(redact_secrets(value)) for key, value in parts.items()}
        identifier = "checkpoint_" + uuid4().hex
        with self.db.transaction():
            self.ensure(session_id)
            session = self.conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            existing = self.conn.execute("SELECT message_seq,payload_json FROM messages WHERE session_id=? ORDER BY message_seq", (session_id,)).fetchall()
            if len(existing) > len(messages) or any(row["message_seq"] != index + 1 or row["payload_json"] != canonical(messages[index]) for index, row in enumerate(existing)):
                raise ValueError("DURABLE_HISTORY_CONFLICT: persisted conversation is append-only")
            for index in range(len(existing), len(messages)):
                text = canonical(messages[index])
                self.conn.execute("INSERT INTO messages VALUES (?,?,?,?,?)", (session_id,index + 1,run_id,text,digest(text)))
            event_end = self.conn.execute("SELECT COALESCE(MAX(session_seq),0) FROM runtime_events WHERE session_id=?", (session_id,)).fetchone()[0]
            manifest = canonical({"version": 1, "session_id": session_id, "message_end_seq": len(messages), "messages_sha256": digest(canonical(messages)), "event_end_seq": event_end, "components": {key:digest(value) for key,value in serialized.items()}})
            self.conn.execute("INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?,?,?,?)", (identifier,session_id,run_id,session["latest_checkpoint_id"],len(messages),event_end,boundary,manifest,digest(manifest),now()))
            for key, value in serialized.items():
                checksum = digest(value)
                self.conn.execute("INSERT OR IGNORE INTO state_payloads VALUES (?,?)", (checksum,value))
                if self.conn.execute("SELECT payload_json FROM state_payloads WHERE payload_sha256=?", (checksum,)).fetchone()[0] != value:
                    raise ValueError("STATE_DIGEST_CONFLICT")
                self.conn.execute("INSERT INTO checkpoint_components VALUES (?,?,?)", (identifier,key,checksum))
            self.conn.execute("UPDATE sessions SET latest_checkpoint_id=?,backend='sqlite' WHERE session_id=?", (identifier,session_id))
            # Result receipts only become replay-safe once their state is checkpointed.
            self.conn.execute("UPDATE tool_operations SET status='CHECKPOINTED' WHERE session_id=? AND status='RESULT_COMMITTED'", (session_id,))
            if run_id:
                self.conn.execute("UPDATE runs SET final_checkpoint_id=? WHERE run_id=?", (identifier,run_id))
            if session["task_id"]:
                project = self.conn.execute("SELECT project_id FROM tasks WHERE task_id=?", (session["task_id"],)).fetchone()[0]
                indexes = ProjectRepository(self.db)
                for evidence in scientific_payload.get("evidence", []):
                    source_text = canonical(evidence)
                    source_id = digest(source_text)
                    indexes.index_document(f"evidence:{session_id}:{source_id}", "evidence", "Historical evidence; requires validation", source_text, project, session["task_id"])
                for index in range(len(existing),len(messages)):
                    item = messages[index]
                    text = item.get("content") or item.get("text")
                    if isinstance(text, str):
                        indexes.index_document(f"message:{session_id}:{index+1}", "message", f"Message {index+1}", text, project, session["task_id"])
        return identifier

    def load(self, session_id: str, checkpoint_id: str | None = None) -> SessionSnapshot:
        with self.db.lock:
            session = self.conn.execute("SELECT latest_checkpoint_id FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            if session is None or not (checkpoint_id or session[0]):
                raise FileNotFoundError("session has no committed checkpoint")
            row = self.conn.execute("SELECT * FROM checkpoints WHERE checkpoint_id=? AND session_id=?", (checkpoint_id or session[0],session_id)).fetchone()
            if row is None or digest(row["manifest_json"]) != row["sha256"]:
                raise ValueError("CHECKPOINT_CORRUPT")
            manifest = json.loads(row["manifest_json"])
            parts = self.components(row["checkpoint_id"])
            if {key: digest(canonical(value)) for key,value in parts.items()} != manifest["components"]:
                raise ValueError("CHECKPOINT_COMPONENT_CORRUPT")
            messages = []
            for index,item in enumerate(self.conn.execute("SELECT * FROM messages WHERE session_id=? AND message_seq<=? ORDER BY message_seq", (session_id,row["message_end_seq"]))):
                if item["message_seq"] != index + 1 or digest(item["payload_json"]) != item["payload_sha256"]:
                    raise ValueError("MESSAGE_CORRUPT")
                messages.append(json.loads(item["payload_json"]))
            if len(messages) != row["message_end_seq"] or digest(canonical(messages)) != manifest["messages_sha256"]:
                raise ValueError("CHECKPOINT_HISTORY_CORRUPT")
            payload = parts["snapshot"]
            if "scientific" in parts:
                payload["scientific"] = parts["scientific"]
                payload["engine"] = parts["engine"]
            payload["conversation"] = {"messages":messages}
            return parse_session_snapshot(canonical(payload))

    def components(self, checkpoint_id: str) -> dict[str, Any]:
        with self.db.lock:
            result = {}
            for row in self.conn.execute("SELECT component_kind,payload_sha256,payload_json FROM checkpoint_components JOIN state_payloads USING(payload_sha256) WHERE checkpoint_id=?", (checkpoint_id,)):
                if digest(row["payload_json"]) != row["payload_sha256"]:
                    raise ValueError("STATE_CORRUPT")
                result[row["component_kind"]] = json.loads(row["payload_json"])
            return result

    def checkpoints(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self.db.lock:
            return [dict(row) for row in self.conn.execute("SELECT checkpoint_id,session_id,run_id,parent_checkpoint_id,message_end_seq,runtime_event_end_seq,boundary_kind,sha256,created_at FROM checkpoints WHERE session_id=? ORDER BY created_at DESC LIMIT ?", (session_id,min(max(limit,1),200)))]

    def event(self, session_id: str, event: Any, run_id: str | None = None) -> None:
        payload = redact_secrets(event.model_dump(mode="json"))
        event_id = event.trace_event_id
        payload["trace_event_id"] = event_id
        with self.db.transaction():
            self.ensure(session_id)
            self.conn.execute("INSERT OR IGNORE INTO runtime_events SELECT ?,COALESCE(MAX(session_seq),0)+1,?,?,?,?,? FROM runtime_events WHERE session_id=?", (session_id,event_id,run_id,payload["kind"],canonical(payload),now(),session_id))

    def unresolved(self, session_id: str) -> list[dict[str, Any]]:
        with self.db.lock:
            return [dict(row) for row in self.conn.execute("SELECT operation_id,tool_name,status,run_id FROM tool_operations WHERE session_id=? AND status!='CHECKPOINTED'", (session_id,))]

    def intent(self, session_id: str, run_id: str | None, call_id: str, name: str, arguments: Any) -> str:
        operation_id = digest(canonical([session_id,run_id,call_id]))
        with self.db.transaction():
            self.ensure(session_id)
            self.conn.execute("INSERT INTO tool_operations VALUES (?,?,?,?,?,?, 'INTENT',NULL,?)", (operation_id,session_id,run_id,call_id,name,digest(canonical(redact_secrets(arguments))),now()))
        return operation_id

    def result(self, operation_id: str, result: Any) -> None:
        with self.db.transaction():
            self.conn.execute("UPDATE tool_operations SET status='RESULT_COMMITTED',result_json=? WHERE operation_id=? AND status='INTENT'", (canonical(redact_secrets(result.model_dump(mode="json"))),operation_id))

    def export(self, session_id: str, directory: Path) -> None:
        """Explicit compatibility export. Never used as a recovery fallback."""
        import os
        import tempfile
        with self.db.lock:
            checkpoint_id = self.conn.execute("SELECT latest_checkpoint_id FROM sessions WHERE session_id=?", (session_id,)).fetchone()[0]
            snapshot = self.load(session_id, checkpoint_id)
            event_end = self.conn.execute("SELECT runtime_event_end_seq FROM checkpoints WHERE checkpoint_id=?", (checkpoint_id,)).fetchone()[0]
            events = [row[0] for row in self.conn.execute("SELECT payload_json FROM runtime_events WHERE session_id=? AND session_seq<=? ORDER BY session_seq", (session_id,event_end))]
        directory.mkdir(parents=True,exist_ok=True)
        for name, content in {"session_state.json":snapshot.model_dump_json(), "events.jsonl":"".join(item+"\n" for item in events), "export_manifest.json":canonical({"checkpoint_id":checkpoint_id,"exported_at":now(),"authority":"sqlite"})}.items():
            with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",dir=directory,delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary,directory/name)


class RuntimePersistence:
    """Bounded awaited writes; no background writes can outlive their runtime."""
    def __init__(self, repository: SessionRepository, session_id: str, run_id: str) -> None:
        self.repository = repository
        self.session_id = session_id
        self.run_id = run_id
        self._gate: asyncio.Semaphore | None = None
        self.components: dict[str, Any] = {}
        self.restored_components: dict[str, Any] = {}

    async def write(self, method: Any, *args: Any, **kwargs: Any) -> Any:
        if self._gate is None:
            self._gate = asyncio.Semaphore(1)
        async with self._gate:
            task = asyncio.create_task(asyncio.to_thread(method,*args,**kwargs))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                await task
                raise

    async def checkpoint(self, runtime: Any, boundary: str) -> None:
        snapshot = SessionSnapshot(conversation=runtime.conversation_state, scientific=runtime.scientific_state, engine=runtime.context_engine.snapshot())
        await self.write(self.repository.checkpoint,self.session_id,snapshot,run_id=self.run_id,boundary=boundary,components={**self.components,"runtime_budget":runtime.budget.model_dump()})

    async def intent(self, call_id: str, name: str, arguments: Any) -> str:
        return str(await self.write(self.repository.intent,self.session_id,self.run_id,call_id,name,arguments))

    async def result(self, operation_id: str, result: Any) -> None:
        await self.write(self.repository.result,operation_id,result)
