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
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from photomatagent.redaction import redact_secrets
from photomatagent.runtime.context_engine import CompactionState
from photomatagent.runtime.state import ConversationState
from photomatagent.scientific.state import ScientificState

SESSION_STATE_FILENAME = "session_state.json"
SESSION_STATE_SCHEMA_VERSION = 2


class EngineSnapshot(BaseModel):
    """ContextEngine cursor so a resumed session does not lose compaction."""

    compaction_state: CompactionState | None = None
    compacted_message_count: int = 0
    compaction_count: int = 0


class SessionMigrationDiagnostic(BaseModel):
    code: Literal["EVIDENCE_AUTHORITY_DOWNGRADED"]
    from_schema_version: int
    to_schema_version: int = SESSION_STATE_SCHEMA_VERSION
    detail: str


class SessionSnapshot(BaseModel):
    schema_version: Literal[2] = 2
    conversation: ConversationState
    scientific: ScientificState
    engine: EngineSnapshot | None = None
    migration_diagnostics: list[SessionMigrationDiagnostic] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def report_nonpersistent_authority(self) -> SessionSnapshot:
        if self.scientific.evidence_attestations and not any(
            item.code == "EVIDENCE_AUTHORITY_DOWNGRADED"
            for item in self.migration_diagnostics
        ):
            self.migration_diagnostics.append(
                SessionMigrationDiagnostic(
                    code="EVIDENCE_AUTHORITY_DOWNGRADED",
                    from_schema_version=self.schema_version,
                    detail=(
                        "Evidence authority is runtime-only; restored attestation "
                        "metadata is background until a trusted tool observes it again."
                    ),
                )
            )
        return self


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
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("schema_version", 1)
    if version == 1:
        payload["schema_version"] = SESSION_STATE_SCHEMA_VERSION
        scientific = payload.get("scientific", {})
        if isinstance(scientific, dict) and scientific.get("evidence_attestations"):
            payload.setdefault("migration_diagnostics", []).append(
                {
                    "code": "EVIDENCE_AUTHORITY_DOWNGRADED",
                    "from_schema_version": 1,
                    "to_schema_version": SESSION_STATE_SCHEMA_VERSION,
                    "detail": (
                        "Schema-v1 attestation metadata cannot restore runtime "
                        "authority and was loaded as background evidence."
                    ),
                }
            )
    elif version != SESSION_STATE_SCHEMA_VERSION:
        raise ValueError(f"unsupported session snapshot schema_version={version!r}")
    return SessionSnapshot.model_validate(payload)


def session_is_resumable(session_dir: Path | str) -> bool:
    return snapshot_path(session_dir).is_file()


__all__ = [
    "EngineSnapshot",
    "SESSION_STATE_FILENAME",
    "SESSION_STATE_SCHEMA_VERSION",
    "SessionMigrationDiagnostic",
    "SessionSnapshot",
    "load_session_snapshot",
    "save_session_snapshot",
    "session_is_resumable",
    "snapshot_path",
]
