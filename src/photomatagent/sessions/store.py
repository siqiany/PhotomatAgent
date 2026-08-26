"""Persistent session snapshots so a finished session can be resumed.

A snapshot is the exact runtime state that makes a session resumable:
the durable conversation, the scientific state, and the ContextEngine
compaction cursor. It is stored next to ``events.jsonl`` inside the session
directory as ``session_state.json`` so a later process can reload the session
and continue asking questions on top of it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from photomatagent.redaction import redact_secrets
from photomatagent.runtime.context_engine import CompactionState
from photomatagent.runtime.state import ConversationState
from photomatagent.scientific.state import ScientificState

SESSION_STATE_FILENAME = "session_state.json"
SESSION_STATE_SCHEMA_VERSION = 1


class EngineSnapshot(BaseModel):
    """ContextEngine cursor so a resumed session does not lose compaction."""

    compaction_state: CompactionState | None = None
    compacted_message_count: int = 0
    compaction_count: int = 0


class SessionSnapshot(BaseModel):
    schema_version: int = SESSION_STATE_SCHEMA_VERSION
    conversation: ConversationState
    scientific: ScientificState
    engine: EngineSnapshot | None = None


def snapshot_path(session_dir: Path | str) -> Path:
    return Path(session_dir) / SESSION_STATE_FILENAME


def save_session_snapshot(
    session_dir: Path | str,
    *,
    conversation: ConversationState,
    scientific: ScientificState,
    engine: dict[str, Any] | None = None,
) -> Path:
    """Write the current runtime state to the session directory."""
    snapshot = SessionSnapshot(
        conversation=conversation,
        scientific=scientific,
        engine=EngineSnapshot.model_validate(engine) if engine else None,
    )
    payload = redact_secrets(json.loads(snapshot.model_dump_json()))
    path = snapshot_path(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_session_snapshot(session_dir: Path | str) -> SessionSnapshot:
    """Load a previously saved session snapshot from a session directory."""
    path = snapshot_path(session_dir)
    if not path.is_file():
        raise FileNotFoundError(f"session snapshot not found: {path}")
    return SessionSnapshot.model_validate_json(path.read_text(encoding="utf-8"))


def session_is_resumable(session_dir: Path | str) -> bool:
    return snapshot_path(session_dir).is_file()


__all__ = [
    "EngineSnapshot",
    "SESSION_STATE_FILENAME",
    "SessionSnapshot",
    "load_session_snapshot",
    "save_session_snapshot",
    "session_is_resumable",
    "snapshot_path",
]
