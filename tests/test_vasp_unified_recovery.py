"""Task 14: unified VASP recovery decisions."""

from __future__ import annotations

from photomatagent.scientific.applications.vasp.unified.approvals import (
    ApprovalKind,
    ApprovalReceiptStore,
    pending_decision,
)
from photomatagent.scientific.applications.vasp.unified.fingerprints import scientific_fingerprint
from photomatagent.scientific.applications.vasp.unified.models import (
    PeriodicScientificSpec,
    ScientificChange,
    UnifiedStage,
    UnifiedVaspManifest,
    VaspWorkflowKind,
)
from photomatagent.scientific.applications.vasp.unified.recovery import (
    RecoveryAction,
    _verified_restart_proof,
    classify_recovery,
)
from photomatagent.scientific.applications.vasp.unified.executors import (
    RecoveryResult,
    RestartStructuralValidation,
)
from pydantic import ValidationError
import pytest
from photomatagent.scientific.remote.models import ResourceRequest


def test_ssh_status_failure_never_submits():
    outcome = classify_recovery(status_failed=True)
    assert outcome.action is RecoveryAction.STOP
    assert any("status query failed" in reason for reason in outcome.reasons)


def test_ambiguous_submission_always_reconciles_first():
    outcome = classify_recovery(ambiguous_submission=True)
    assert outcome.action is RecoveryAction.RECONCILE


def test_arbitrary_restart_validation_dict_never_authorizes_automatic_resume():
    outcome = classify_recovery(
        contcar_restart=True,
        scientific_intent_unchanged=True,
        artifact_hash="a" * 64,
        structural_validation={"atom_count": 2, "validated_by": "executor"},
    )
    assert outcome.action is RecoveryAction.STOP


def test_recovery_result_rejects_untyped_structural_validation_payload():
    with pytest.raises(ValidationError):
        RecoveryResult(
            ok=True,
            action="AUTO_RESUME",
            contcar_restart=True,
            restart_artifact_path="CONTCAR",
            restart_artifact_sha256="a" * 64,
            restart_structural_validation={"arbitrary": "nonempty"},
        )


def test_service_verified_restart_proof_authorizes_automatic_resume(tmp_path):
    artifact = tmp_path / "CONTCAR"
    artifact.write_bytes(b"validated restart")
    import hashlib
    proof = _verified_restart_proof(
        artifact,
        hashlib.sha256(artifact.read_bytes()).hexdigest(),
        RestartStructuralValidation(
            atom_count=2,
            validator="deterministic executor",
        ),
    )
    assert proof is not None
    outcome = classify_recovery(
        contcar_restart=True,
        scientific_intent_unchanged=True,
        restart_proof=proof,
    )
    assert outcome.action is RecoveryAction.AUTO_RESUME
    assert outcome.artifact_hash == proof.artifact_hash


def test_scientific_changes_need_pending_decision():
    outcome = classify_recovery(
        scientific_changes=[
            ScientificChange(parameter="ENCUT", old_value=400, new_value=520, reason="convergence")
        ]
    )
    assert outcome.action is RecoveryAction.NEEDS_SCIENTIFIC_CONFIRMATION


def test_oom_time_limit_escalation_produces_resource_decision():
    outcome = classify_recovery(
        resource_escalation=True,
        resource_recommendation=ResourceRequest(
            nodes=2, tasks_per_node=32, walltime_minutes=480
        ),
    )
    assert outcome.action is RecoveryAction.NEEDS_RESOURCE_CONFIRMATION
    assert outcome.resource_recommendation is not None


def _manifest(workflow_id: str) -> UnifiedVaspManifest:
    spec = PeriodicScientificSpec(structure_path="structure.cif", profile="standard_semiconductor")
    return UnifiedVaspManifest(
        workflow_id=workflow_id,
        workflow_kind=VaspWorkflowKind.PERIODIC,
        scientific_spec=spec,
        scientific_fingerprint=scientific_fingerprint(spec),
        execution_fingerprint="e" * 64,
        stages=[UnifiedStage(name="relax", execution_fingerprint="e" * 64)],
    )


def test_matching_store_validated_receipt_permits_exact_proposed_recovery(tmp_path):
    manifest = _manifest("wf")
    store = ApprovalReceiptStore(tmp_path)
    pending = pending_decision(
        manifest=manifest,
        kind=ApprovalKind.SCIENTIFIC,
        summary="Change ENCUT",
        changes=[ScientificChange(parameter="ENCUT", old_value=400, new_value=520, reason="convergence")],
        stage="relax",
    )
    store.record_pending(pending)
    store.approve(pending.decision_id, approved_by="user")
    outcome = classify_recovery(
        scientific_changes=[
            ScientificChange(parameter="ENCUT", old_value=400, new_value=520, reason="convergence")
        ],
        approval_store=store,
        manifest=manifest,
        pending_decision=pending,
    )
    assert outcome.action is RecoveryAction.AUTO_RESUME


def test_same_kind_receipt_from_another_workflow_cannot_resume(tmp_path):
    store = ApprovalReceiptStore(tmp_path)
    foreign = _manifest("foreign")
    pending = pending_decision(
        manifest=foreign,
        kind=ApprovalKind.RESOURCE,
        summary="Resource confirmation for relax",
        stage="relax",
        resource_proposal={"nodes": 2},
    )
    store.record_pending(pending)
    store.approve(pending.decision_id, approved_by="user")

    outcome = classify_recovery(
        resource_escalation=True,
        resource_recommendation=ResourceRequest(nodes=2, tasks_per_node=32, walltime_minutes=240),
        approval_store=store,
        manifest=_manifest("current"),
        pending_decision=pending,
    )
    assert outcome.action is RecoveryAction.NEEDS_RESOURCE_CONFIRMATION


def test_contcar_without_valid_sha_and_structural_validation_never_auto_resumes():
    outcome = classify_recovery(
        contcar_restart=True,
        scientific_intent_unchanged=True,
        artifact_hash="not-a-sha",
        structural_validation=None,
    )
    assert outcome.action is RecoveryAction.STOP
