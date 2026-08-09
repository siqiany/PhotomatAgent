"""JSONL RuntimeEvent logger with a replaceable redaction hook."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from photomatagent.runtime.events import RuntimeEvent, parse_event

Redactor = Callable[[dict[str, Any]], dict[str, Any]]
_SECRET_KEY = re.compile(
    r"(?i)(api[_-]?key|authorization|password|secret|access[_-]?token|auth[_-]?token)"
)
_SECRET_VALUE = re.compile(r"\b(?:sk-ant-|sk-)[A-Za-z0-9_-]{8,}\b")


def _session_id() -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{now}_{uuid4().hex[:6]}"


def default_sessions_dir() -> Path:
    return Path(".photomatagent") / "sessions"


def redact_secrets(payload: dict[str, Any]) -> dict[str, Any]:
    """Conservative default; callers can inject a domain-specific redactor."""
    known_values = [
        value
        for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
        if (value := os.getenv(name))
    ]

    def visit(value: Any, key: str | None = None) -> Any:
        if key and _SECRET_KEY.search(key):
            return "[REDACTED]"
        if isinstance(value, dict):
            return {item_key: visit(item, item_key) for item_key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, str):
            redacted = _SECRET_VALUE.sub("[REDACTED]", value)
            for secret in known_values:
                redacted = redacted.replace(secret, "[REDACTED]")
            return redacted
        return value

    return visit(deepcopy(payload))


class EventLogger:
    def __init__(
        self,
        sessions_dir: Path | str | None = None,
        *,
        redactor: Redactor | None = None,
    ) -> None:
        self.session_id = _session_id()
        base = Path(sessions_dir) if sessions_dir is not None else default_sessions_dir()
        self.session_dir = base / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.session_dir / "events.jsonl"
        self.events_path.touch(exist_ok=True)
        self._redactor = redactor or redact_secrets

    async def log(self, event: RuntimeEvent) -> None:
        payload = self._redactor(event.model_dump(mode="json"))
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def read_events(self) -> list[RuntimeEvent]:
        events: list[RuntimeEvent] = []
        if not self.events_path.exists():
            return events
        with self.events_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    events.append(parse_event(json.loads(line)))
        return events
