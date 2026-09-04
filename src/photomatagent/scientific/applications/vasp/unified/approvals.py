"""Hash-bound pending decisions and SQLite approval receipts.

Approvals are user-controlled, non-model actions. This module is an internal
API used by the unified service and the CLI approval command; it is never
exposed as a Tool.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from photomatagent.scientific.applications.vasp.unified.fingerprints import (
    canonical_json,
)
from photomatagent.scientific.applications.vasp.unified.models import (
    ScientificChange,
    UnifiedVaspManifest,
    WorkflowEvent,
)


class ApprovalKind(str, Enum):
    RESOURCE = "resource"
    SCIENTIFIC = "scientific"


class DecisionConflictError(RuntimeError):
    """A decision identifier was reused for different immutable authority."""


class PendingDecision(BaseModel):
    decision_id: str
    workflow_id: str
    kind: ApprovalKind
    decision_hash: str
    scientific_fingerprint: str
    execution_fingerprint: str | None = None
    summary: str
    changes: list[ScientificChange] = Field(default_factory=list)
    manifest_schema_version: str = "2.0"
    manifest_revision: int = 0
    decision_epoch: int = 0
    stage: str | None = None
    resource_proposal: dict[str, Any] | None = None
    recovery_attempt_inputs: dict[str, Any] | None = None


class ApprovalReceipt(BaseModel):
    receipt_id: str
    decision_id: str
    decision_hash: str
    workflow_id: str
    kind: ApprovalKind
    scientific_fingerprint: str
    execution_fingerprint: str | None = None
    manifest_schema_version: str = "2.0"
    manifest_revision: int = 0
    decision_epoch: int = 0
    approved_at: datetime
    approved_by: str


def _now() -> datetime:
    return datetime.now(timezone.utc)


def decision_payload(
    *,
    workflow_id: str,
    kind: ApprovalKind,
    manifest_schema_version: str,
    manifest_revision: int,
    decision_epoch: int,
    scientific_fingerprint: str,
    execution_fingerprint: str | None,
    summary: str,
    changes: list[ScientificChange],
    stage: str | None = None,
    resource_proposal: dict[str, Any] | None = None,
    recovery_attempt_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Immutable, canonical application-approval payload.

    The decision identifier is deliberately absent: it is a locator, not
    authority.  Every field that could make a receipt stale is represented
    here, then hashed by :func:`decision_hash`.
    """
    return {
        "workflow_id": workflow_id,
        "kind": kind.value,
        "manifest_schema_version": manifest_schema_version,
        "manifest_revision": manifest_revision,
        "decision_epoch": decision_epoch,
        "scientific_fingerprint": scientific_fingerprint,
        "execution_fingerprint": execution_fingerprint,
        "summary": summary,
        "changes": [change.model_dump(mode="json") for change in changes],
        "resource_proposal": resource_proposal,
        "stage": stage,
        "recovery_attempt_inputs": recovery_attempt_inputs,
    }


def decision_hash(payload: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def pending_decision(
    *,
    manifest: UnifiedVaspManifest,
    kind: ApprovalKind,
    summary: str,
    changes: list[ScientificChange] | None = None,
    stage: str | None = None,
    resource_proposal: dict[str, Any] | None = None,
    recovery_attempt_inputs: dict[str, Any] | None = None,
    decision_id: str | None = None,
) -> PendingDecision:
    """Build a decision bound to one exact persisted manifest revision."""
    target = next((item for item in manifest.stages if item.name == stage), None)
    if kind is ApprovalKind.SCIENTIFIC and target is None:
        raise ValueError("stage execution_fingerprint is required for scientific execution approval")
    effective_execution = (
        target.execution_fingerprint if target is not None else manifest.execution_fingerprint
    )
    decision_epoch = target.decision_epoch if target is not None else manifest.decision_epoch
    execution_scoped = (
        kind in {ApprovalKind.RESOURCE, ApprovalKind.SCIENTIFIC}
        or resource_proposal is not None
        or recovery_attempt_inputs is not None
    )
    if execution_scoped and not effective_execution:
        raise ValueError("stage execution_fingerprint is required for resource or recovery decisions")
    normalized_changes = [ScientificChange.model_validate(change) for change in (changes or [])]
    payload = decision_payload(
        workflow_id=manifest.workflow_id,
        kind=kind,
        manifest_schema_version=manifest.schema_version,
        manifest_revision=manifest.revision,
        decision_epoch=decision_epoch,
        scientific_fingerprint=manifest.scientific_fingerprint,
        execution_fingerprint=effective_execution,
        summary=summary,
        changes=normalized_changes,
        stage=stage,
        resource_proposal=resource_proposal,
        recovery_attempt_inputs=recovery_attempt_inputs,
    )
    digest = decision_hash(payload)
    return PendingDecision(
        decision_id=decision_id or f"{kind.value[:3]}_{digest[:16]}",
        workflow_id=manifest.workflow_id,
        kind=kind,
        decision_hash=digest,
        scientific_fingerprint=manifest.scientific_fingerprint,
        execution_fingerprint=effective_execution,
        summary=summary,
        changes=normalized_changes,
        manifest_schema_version=manifest.schema_version,
        manifest_revision=manifest.revision,
        decision_epoch=decision_epoch,
        stage=stage,
        resource_proposal=resource_proposal,
        recovery_attempt_inputs=recovery_attempt_inputs,
    )


def _pending_hash_matches(pending: PendingDecision) -> bool:
    if pending.kind in {ApprovalKind.RESOURCE, ApprovalKind.SCIENTIFIC} and (
        not pending.stage or not pending.execution_fingerprint
    ):
        return False
    payload = decision_payload(
        workflow_id=pending.workflow_id,
        kind=pending.kind,
        manifest_schema_version=pending.manifest_schema_version,
        manifest_revision=pending.manifest_revision,
        decision_epoch=pending.decision_epoch,
        scientific_fingerprint=pending.scientific_fingerprint,
        execution_fingerprint=pending.execution_fingerprint,
        summary=pending.summary,
        changes=pending.changes,
        stage=pending.stage,
        resource_proposal=pending.resource_proposal,
        recovery_attempt_inputs=pending.recovery_attempt_inputs,
    )
    return pending.decision_hash == decision_hash(payload)


class ApprovalReceiptStore:
    """SQLite-backed pending decision and approval receipt store."""

    def __init__(self, workspace_root: Path | str) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.db_path = (
            self.workspace_root
            / ".photomatagent"
            / "vasp"
            / "approvals.sqlite3"
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS pending_decisions (
                    decision_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    decision_hash TEXT NOT NULL,
                    scientific_fingerprint TEXT NOT NULL,
                    execution_fingerprint TEXT,
                    summary TEXT NOT NULL,
                    changes_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS approval_receipts (
                    receipt_id TEXT PRIMARY KEY,
                    decision_id TEXT UNIQUE NOT NULL,
                    decision_hash TEXT NOT NULL,
                    workflow_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    scientific_fingerprint TEXT NOT NULL,
                    execution_fingerprint TEXT,
                    approved_at TEXT NOT NULL,
                    approved_by TEXT NOT NULL
                );
                """
            )
            self._add_column(connection, "pending_decisions", "manifest_schema_version", "TEXT")
            self._add_column(connection, "pending_decisions", "manifest_revision", "INTEGER")
            self._add_column(connection, "pending_decisions", "decision_epoch", "INTEGER")
            self._add_column(connection, "pending_decisions", "stage", "TEXT")
            self._add_column(connection, "pending_decisions", "resource_proposal_json", "TEXT")
            self._add_column(connection, "pending_decisions", "recovery_attempt_inputs_json", "TEXT")
            self._add_column(connection, "approval_receipts", "manifest_schema_version", "TEXT")
            self._add_column(connection, "approval_receipts", "manifest_revision", "INTEGER")
            self._add_column(connection, "approval_receipts", "decision_epoch", "INTEGER")

    @staticmethod
    def _add_column(connection: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    # -- pending decisions ---------------------------------------------------

    def record_pending(self, decision: PendingDecision) -> None:
        if not _pending_hash_matches(decision):
            raise ValueError("pending decision hash does not match canonical payload")
        existing = self.load_pending(decision.decision_id)
        if existing is not None:
            if existing == decision:
                return
            raise DecisionConflictError(
                f"pending decision id collision: {decision.decision_id}"
            )
        data = decision.model_dump(mode="json")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO pending_decisions (
                    decision_id, workflow_id, kind, decision_hash,
                    scientific_fingerprint, execution_fingerprint,
                    summary, changes_json, manifest_schema_version,
                    manifest_revision, decision_epoch, stage, resource_proposal_json,
                    recovery_attempt_inputs_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.decision_id,
                    decision.workflow_id,
                    decision.kind.value,
                    decision.decision_hash,
                    decision.scientific_fingerprint,
                    decision.execution_fingerprint,
                    decision.summary,
                    json.dumps(data.get("changes", []), ensure_ascii=False, sort_keys=True),
                    decision.manifest_schema_version,
                    decision.manifest_revision,
                    decision.decision_epoch,
                    decision.stage,
                    json.dumps(decision.resource_proposal, ensure_ascii=False, sort_keys=True),
                    json.dumps(decision.recovery_attempt_inputs, ensure_ascii=False, sort_keys=True),
                ),
            )

    def load_pending(self, decision_id: str) -> PendingDecision | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM pending_decisions WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            pending = self._pending_from_row(row)
        except (TypeError, ValueError, KeyError):
            return None
        return pending if _pending_hash_matches(pending) else None

    @staticmethod
    def _pending_from_row(row: sqlite3.Row) -> PendingDecision:
        changes = json.loads(row["changes_json"] or "[]")
        return PendingDecision(
            decision_id=row["decision_id"],
            workflow_id=row["workflow_id"],
            kind=ApprovalKind(row["kind"]),
            decision_hash=row["decision_hash"],
            scientific_fingerprint=row["scientific_fingerprint"],
            execution_fingerprint=row["execution_fingerprint"],
            summary=row["summary"],
            changes=[ScientificChange.model_validate(item) for item in changes],
            manifest_schema_version=row["manifest_schema_version"],
            manifest_revision=row["manifest_revision"],
            decision_epoch=row["decision_epoch"],
            stage=row["stage"],
            resource_proposal=json.loads(row["resource_proposal_json"] or "null"),
            recovery_attempt_inputs=json.loads(row["recovery_attempt_inputs_json"] or "null"),
        )

    # -- approvals -----------------------------------------------------------

    def approve(self, decision_id: str, approved_by: str) -> ApprovalReceipt:
        """Record explicit user approval for one pending decision.

        This is an internal API called by the user-only CLI command, never by
        a model-visible Tool.
        """
        pending = self.load_pending(decision_id)
        if pending is None:
            raise KeyError(f"unknown or invalid pending decision: {decision_id}")
        existing = self._receipt_for_decision(decision_id)
        if existing is not None:
            return existing
        receipt = ApprovalReceipt(
            receipt_id=f"rec_{uuid.uuid4().hex[:16]}",
            decision_id=pending.decision_id,
            decision_hash=pending.decision_hash,
            workflow_id=pending.workflow_id,
            kind=pending.kind,
            scientific_fingerprint=pending.scientific_fingerprint,
            execution_fingerprint=pending.execution_fingerprint,
            manifest_schema_version=pending.manifest_schema_version,
            manifest_revision=pending.manifest_revision,
            decision_epoch=pending.decision_epoch,
            approved_at=_now(),
            approved_by=approved_by,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO approval_receipts (
                    receipt_id, decision_id, decision_hash, workflow_id,
                    kind, scientific_fingerprint, execution_fingerprint,
                    approved_at, approved_by, manifest_schema_version,
                    manifest_revision, decision_epoch
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt.receipt_id,
                    receipt.decision_id,
                    receipt.decision_hash,
                    receipt.workflow_id,
                    receipt.kind.value,
                    receipt.scientific_fingerprint,
                    receipt.execution_fingerprint,
                    receipt.approved_at.isoformat(),
                    receipt.approved_by,
                    receipt.manifest_schema_version,
                    receipt.manifest_revision,
                    receipt.decision_epoch,
                ),
            )
        return receipt

    def valid_receipt(
        self,
        pending: PendingDecision,
        manifest: UnifiedVaspManifest,
    ) -> ApprovalReceipt | None:
        """Return the receipt only if every bound value still matches."""
        receipt = self._receipt_for_decision(pending.decision_id)
        if receipt is None:
            return None
        if receipt.workflow_id != pending.workflow_id:
            return None
        if receipt.workflow_id != manifest.workflow_id:
            return None
        if receipt.kind != pending.kind:
            return None
        if receipt.decision_hash != pending.decision_hash:
            return None
        if not _pending_hash_matches(pending):
            return None
        stored = self.load_pending(pending.decision_id)
        if stored is None or stored != pending:
            return None
        if receipt.manifest_schema_version != pending.manifest_schema_version:
            return None
        if receipt.decision_epoch != pending.decision_epoch:
            return None
        if manifest.schema_version != pending.manifest_schema_version:
            return None
        if receipt.scientific_fingerprint != pending.scientific_fingerprint:
            return None
        if receipt.scientific_fingerprint != manifest.scientific_fingerprint:
            return None
        if receipt.execution_fingerprint != pending.execution_fingerprint:
            return None
        if pending.kind in {ApprovalKind.RESOURCE, ApprovalKind.SCIENTIFIC} and (
            not pending.stage or not pending.execution_fingerprint
        ):
            return None
        target = next((item for item in manifest.stages if item.name == pending.stage), None)
        current_execution = target.execution_fingerprint if target is not None else None
        if current_execution != pending.execution_fingerprint:
            return None
        current_epoch = target.decision_epoch if target is not None else manifest.decision_epoch
        if current_epoch != pending.decision_epoch:
            return None
        return receipt

    def _receipt_for_decision(self, decision_id: str) -> ApprovalReceipt | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM approval_receipts WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()
        if row is None:
            return None
        return ApprovalReceipt(
            receipt_id=row["receipt_id"],
            decision_id=row["decision_id"],
            decision_hash=row["decision_hash"],
            workflow_id=row["workflow_id"],
            kind=ApprovalKind(row["kind"]),
            scientific_fingerprint=row["scientific_fingerprint"],
            execution_fingerprint=row["execution_fingerprint"],
            manifest_schema_version=row["manifest_schema_version"],
            manifest_revision=row["manifest_revision"],
            decision_epoch=row["decision_epoch"],
            approved_at=datetime.fromisoformat(row["approved_at"]),
            approved_by=row["approved_by"],
        )
