"""EventLogger: append RuntimeEvents to a session's events.jsonl."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from photomatagent.runtime.events import RuntimeEvent, parse_event


def _session_id() -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{now}_{uuid4().hex[:6]}"


def default_sessions_dir() -> Path:
    return Path(".photomatagent") / "sessions"


class EventLogger:
    """Appends every RuntimeEvent to ``<sessions_dir>/<session-id>/events.jsonl``."""

    def __init__(self, sessions_dir: Path | str | None = None) -> None:
        self.session_id = _session_id()
        base = Path(sessions_dir) if sessions_dir is not None else default_sessions_dir()
        self.session_dir = base / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.session_dir / "events.jsonl"
        self.events_path.touch(exist_ok=True)

    async def log(self, event: RuntimeEvent) -> None:
        """Async sink signature; the write itself is a fast append."""
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")

    def read_events(self) -> list[RuntimeEvent]:
        """Load logged events back as typed RuntimeEvents (for replay/debug)."""
        if not self.events_path.exists():
            return []
        events: list[RuntimeEvent] = []
        with self.events_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    events.append(parse_event(json.loads(line)))
        return events
