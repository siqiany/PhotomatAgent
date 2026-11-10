"""JSONL RuntimeEvent logger with a replaceable redaction hook."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from photomatagent.runtime.events import (
    EVOLUTION_SUMMARY_MAX_CHARS,
    EvolutionRuntimeEvent,
    RuntimeEvent,
    parse_event,
)
from photomatagent.redaction import redact_secrets

Redactor = Callable[[dict[str, Any]], dict[str, Any]]


def _bound_evolution_summaries(payload: dict[str, Any]) -> dict[str, Any]:
    bounded = dict(payload)
    for key, value in bounded.items():
        if isinstance(value, str) and (key == "summary" or key.endswith("_summary")):
            bounded[key] = value[:EVOLUTION_SUMMARY_MAX_CHARS]
    return bounded


def _session_id() -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{now}_{uuid4().hex[:6]}"


def default_sessions_dir() -> Path:
    return Path(".photomatagent") / "sessions"


class EventLogger:
    def __init__(
        self,
        sessions_dir: Path | str | None = None,
        *,
        redactor: Redactor | None = None,
        session_id: str | None = None,
    ) -> None:
        self.repository: Any = None
        self.storage_owner: Any = None
        self.session_lock: Any = None
        self.session_id = session_id or _session_id()
        base = Path(sessions_dir) if sessions_dir is not None else default_sessions_dir()
        self.session_dir = base / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.session_dir / "events.jsonl"
        self.events_path.touch(exist_ok=True)
        self._redactor = redactor or redact_secrets

    async def log(self, event: RuntimeEvent) -> None:
        if self.repository is not None:
            import asyncio
            await asyncio.to_thread(self.repository.event,self.session_id,event)
            return
        from photomatagent.sessions.reader import database_session
        with database_session(self.session_dir) as repository:
            if repository is not None:
                repository.event(self.session_id, event)
                return
        payload = self._redactor(event.model_dump(mode="json"))
        payload["trace_event_id"] = event.trace_event_id
        if isinstance(event, EvolutionRuntimeEvent):
            payload = _bound_evolution_summaries(payload)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def read_events(self) -> list[RuntimeEvent]:
        from photomatagent.sessions.reader import database_session
        with database_session(self.session_dir) as repository:
            if repository is not None:
                with repository.db.lock:
                    return [parse_event(json.loads(row[0])) for row in repository.conn.execute("SELECT payload_json FROM runtime_events WHERE session_id=? ORDER BY session_seq", (self.session_id,))]
        events: list[RuntimeEvent] = []
        if not self.events_path.exists():
            return events
        with self.events_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    events.append(parse_event(json.loads(line)))
        return events

    def close(self) -> None:
        if self.session_lock is not None:
            self.session_lock.release()
            self.session_lock = None
        if self.storage_owner is not None:
            self.storage_owner.close()
            self.storage_owner = None
            self.repository = None
