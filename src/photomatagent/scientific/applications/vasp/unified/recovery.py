"""Automatic versus confirmation-required unified VASP recovery policy."""

from __future__ import annotations

from enum import Enum
import hashlib
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from photomatagent.scientific.applications.vasp.unified.approvals import (
    ApprovalReceiptStore,
    PendingDecision,
)
from photomatagent.scientific.applications.vasp.unified.executors import (
    RestartStructuralValidation,
)
from photomatagent.scientific.applications.vasp.unified.models import (
    ScientificChange,
    UnifiedVaspManifest,
)
from photomatagent.scientific.remote.models import ResourceRequest


class RecoveryAction(str, Enum):
    AUTO_RESUME = "AUTO_RESUME"
    RECONCILE = "RECONCILE"
    NEEDS_RESOURCE_CONFIRMATION = "NEEDS_RESOURCE_CONFIRMATION"
    NEEDS_SCIENTIFIC_CONFIRMATION = "NEEDS_SCIENTIFIC_CONFIRMATION"
    STOP = "STOP"


class RecoveryOutcome(BaseModel):
    action: RecoveryAction
    reasons: list[str]
    scientific_changes: list[ScientificChange] = Field(default_factory=list)
    resource_recommendation: ResourceRequest | None = None
    artifact_hash: str | None = None


_RESTART_PROOF_CAPABILITY = object()


class ValidatedRestartProof:
    """Opaque proof produced only after service-side artifact verification."""

    __slots__ = ("artifact_hash", "structural_validation")

    def __init__(
        self,
        artifact_hash: str,
        structural_validation: RestartStructuralValidation,
        *,
        _capability: object,
    ) -> None:
        if _capability is not _RESTART_PROOF_CAPABILITY:
            raise TypeError("restart validation proofs are service-derived")
        self.artifact_hash = artifact_hash
        self.structural_validation = structural_validation


def _verified_restart_proof(
    artifact: Path,
    claimed_sha256: str | None,
    structural_validation: RestartStructuralValidation | None,
) -> ValidatedRestartProof | None:
    """Create a proof only after hashing the regular artifact bytes itself."""
    if (
        not claimed_sha256
        or re.fullmatch(r"[0-9a-fA-F]{64}", claimed_sha256) is None
        or not isinstance(structural_validation, RestartStructuralValidation)
    ):
        return None
    try:
        computed = hashlib.sha256(artifact.read_bytes()).hexdigest()
    except OSError:
        return None
    if computed != claimed_sha256.lower():
        return None
    return ValidatedRestartProof(
        computed,
        structural_validation,
        _capability=_RESTART_PROOF_CAPABILITY,
    )


def classify_recovery(
    *,
    status_failed: bool = False,
    ambiguous_submission: bool = False,
    scientific_changes: list[ScientificChange] | None = None,
    resource_escalation: bool = False,
    resource_recommendation: ResourceRequest | None = None,
    contcar_restart: bool = False,
    scientific_intent_unchanged: bool = True,
    artifact_hash: str | None = None,
    structural_validation: dict[str, Any] | None = None,
    restart_proof: ValidatedRestartProof | None = None,
    approval_store: ApprovalReceiptStore | None = None,
    manifest: UnifiedVaspManifest | None = None,
    pending_decision: PendingDecision | None = None,
) -> RecoveryOutcome:
    """Classify a recovery action without performing any submission.

    A failed status query never submits. Ambiguous submissions reconcile
    first. Scientific/resource changes require the exact matching receipt.
    """
    if status_failed:
        return RecoveryOutcome(
            action=RecoveryAction.STOP,
            reasons=["status query failed; no submission or resume is safe"],
        )
    if ambiguous_submission:
        return RecoveryOutcome(
            action=RecoveryAction.RECONCILE,
            reasons=["ambiguous submission state requires reconciliation first"],
        )
    changes = scientific_changes or []
    receipt_valid = (
        approval_store is not None
        and manifest is not None
        and pending_decision is not None
        and approval_store.valid_receipt(pending_decision, manifest) is not None
    )
    if changes:
        if receipt_valid and pending_decision is not None and pending_decision.kind.value == "scientific":
            return RecoveryOutcome(
                action=RecoveryAction.AUTO_RESUME,
                reasons=["matching scientific approval receipt permits recovery"],
                scientific_changes=changes,
                artifact_hash=artifact_hash,
            )
        return RecoveryOutcome(
            action=RecoveryAction.NEEDS_SCIENTIFIC_CONFIRMATION,
            reasons=["scientific changes require a pending decision"],
            scientific_changes=changes,
        )
    if resource_escalation:
        if receipt_valid and pending_decision is not None and pending_decision.kind.value == "resource":
            return RecoveryOutcome(
                action=RecoveryAction.AUTO_RESUME,
                reasons=["matching resource approval receipt permits escalation"],
                resource_recommendation=resource_recommendation,
            )
        return RecoveryOutcome(
            action=RecoveryAction.NEEDS_RESOURCE_CONFIRMATION,
            reasons=["resource escalation requires confirmation"],
            resource_recommendation=resource_recommendation,
        )
    if (
        contcar_restart
        and scientific_intent_unchanged
        and isinstance(restart_proof, ValidatedRestartProof)
    ):
        return RecoveryOutcome(
            action=RecoveryAction.AUTO_RESUME,
            reasons=["validated CONTCAR restart with identical scientific intent"],
            artifact_hash=restart_proof.artifact_hash,
        )
    return RecoveryOutcome(action=RecoveryAction.STOP, reasons=["no automatic recovery"])
