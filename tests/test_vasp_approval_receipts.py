"""Task 3: hash-bound VASP approval receipts in SQLite."""

from __future__ import annotations

import pytest

from photomatagent.scientific.applications.vasp.unified.approvals import (
    ApprovalKind,
    ApprovalReceiptStore,
    DecisionConflictError,
    PendingDecision,
    pending_decision,
)
from photomatagent.scientific.applications.vasp.unified.fingerprints import (
    scientific_fingerprint,
)
from photomatagent.scientific.applications.vasp.unified.models import (
    PeriodicScientificSpec,
    ScientificChange,
    UnifiedVaspManifest,
    UnifiedStage,
    VaspWorkflowKind,
)
from photomatagent.workspace import Workspace


def make_manifest(
    *,
    workflow_id: str = "wf-1",
    sci_fp: str | None = None,
    execution_fingerprint: str | None = "exec-1",
) -> UnifiedVaspManifest:
    spec = PeriodicScientificSpec(
        structure_path="structure.cif",
        profile="standard_semiconductor",
        scientific_overrides={"encut_ev": 520},
    )
    resolved_fp = sci_fp or scientific_fingerprint(spec)
    return UnifiedVaspManifest(
        workflow_id=workflow_id,
        workflow_kind=VaspWorkflowKind.PERIODIC,
        scientific_spec=spec,
        scientific_fingerprint=resolved_fp,
        execution_fingerprint=execution_fingerprint,
        stages=[UnifiedStage(name="relax", execution_fingerprint=execution_fingerprint)],
    )


def make_pending(
    *,
    decision_id: str = "dec-1",
    workflow_id: str = "wf-1",
    kind: ApprovalKind = ApprovalKind.SCIENTIFIC,
    decision_hash: str = "hash-1",
    scientific_fingerprint: str = "sci-1",
    execution_fingerprint: str | None = "exec-1",
) -> PendingDecision:
    manifest = make_manifest(
        workflow_id=workflow_id,
        sci_fp=scientific_fingerprint,
        execution_fingerprint=execution_fingerprint,
    )
    return pending_decision(
        manifest=manifest,
        kind=kind,
        summary="Change ENCUT from 500 to 520",
        changes=[ScientificChange(parameter="encut_ev", old_value=500, new_value=520, reason="convergence")],
        stage="relax",
        decision_id=decision_id,
    )


def test_forged_receipt_id_is_rejected(tmp_path):
    store = ApprovalReceiptStore(tmp_path)
    try:
        store.approve("no-such-decision", approved_by="user")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown decision id must raise KeyError")


def test_episode_scoped_store_does_not_carry_prior_receipt(tmp_path):
    manifest = make_manifest()
    pending = make_pending(
        scientific_fingerprint=manifest.scientific_fingerprint,
        execution_fingerprint=manifest.execution_fingerprint,
    )
    prior = ApprovalReceiptStore(tmp_path / "episode-v001")
    prior.record_pending(pending)
    assert prior.approve(pending.decision_id, approved_by="user") is not None

    fresh = ApprovalReceiptStore(tmp_path / "episode-v002")
    assert fresh.load_pending(pending.decision_id) is None
    assert fresh.valid_receipt(pending, manifest) is None
    fresh.record_pending(pending)
    assert fresh.approve(pending.decision_id, approved_by="user") is not None


def test_receipt_from_another_workflow_is_rejected(tmp_path):
    store = ApprovalReceiptStore(tmp_path)
    manifest = make_manifest()
    pending = make_pending(
        scientific_fingerprint=manifest.scientific_fingerprint,
        execution_fingerprint=manifest.execution_fingerprint,
    )
    store.record_pending(pending)
    store.approve(pending.decision_id, approved_by="user")

    other_workflow = pending.model_copy(update={"workflow_id": "wf-other"})
    assert store.valid_receipt(other_workflow, manifest) is None


def test_any_bound_fingerprint_or_decision_hash_change_invalidates_receipt(tmp_path):
    store = ApprovalReceiptStore(tmp_path)
    manifest = make_manifest()
    pending = make_pending(
        scientific_fingerprint=manifest.scientific_fingerprint,
        execution_fingerprint=manifest.execution_fingerprint,
    )
    store.record_pending(pending)
    store.approve(pending.decision_id, approved_by="user")

    changed_hash = pending.model_copy(update={"decision_hash": "other-hash"})
    assert store.valid_receipt(changed_hash, manifest) is None

    changed_sci = pending.model_copy(
        update={"scientific_fingerprint": "other-scientific"}
    )
    assert store.valid_receipt(changed_sci, manifest) is None

    changed_exec = pending.model_copy(
        update={"execution_fingerprint": "other-exec"}
    )
    assert store.valid_receipt(changed_exec, manifest) is None

    changed_manifest = manifest.model_copy(
        update={"scientific_fingerprint": "other-scientific"}
    )
    assert store.valid_receipt(pending, changed_manifest) is None


def test_approval_is_idempotent_for_one_decision_id(tmp_path):
    store = ApprovalReceiptStore(tmp_path)
    manifest = make_manifest()
    pending = make_pending(
        scientific_fingerprint=manifest.scientific_fingerprint,
        execution_fingerprint=manifest.execution_fingerprint,
    )
    store.record_pending(pending)
    first = store.approve(pending.decision_id, approved_by="user")
    second = store.approve(pending.decision_id, approved_by="user")

    assert first.receipt_id == second.receipt_id
    assert store.valid_receipt(pending, manifest) is not None


def test_runtime_allow_all_state_does_not_create_application_receipt(tmp_path):
    store = ApprovalReceiptStore(tmp_path)
    manifest = make_manifest()
    pending = make_pending(
        scientific_fingerprint=manifest.scientific_fingerprint,
        execution_fingerprint=manifest.execution_fingerprint,
    )

    store.record_pending(pending)

    # A pending decision alone is not approval; no allow-all shortcut exists.
    assert store.valid_receipt(pending, manifest) is None
    with store._connect() as connection:
        count = connection.execute(
            "SELECT COUNT(*) AS n FROM approval_receipts"
        ).fetchone()["n"]
    assert count == 0


def test_tampered_pending_payload_with_retained_hash_is_rejected(tmp_path):
    store = ApprovalReceiptStore(tmp_path)
    manifest = make_manifest()
    pending = make_pending(
        scientific_fingerprint=manifest.scientific_fingerprint,
        execution_fingerprint=manifest.execution_fingerprint,
    )
    store.record_pending(pending)
    with store._connect() as connection:
        connection.execute(
            "UPDATE pending_decisions SET summary = ? WHERE decision_id = ?",
            ("tampered", pending.decision_id),
        )

    assert store.load_pending(pending.decision_id) is None
    with pytest.raises(KeyError, match="invalid pending decision"):
        store.approve(pending.decision_id, approved_by="user")


def test_resource_receipt_requires_exact_execution_fingerprint_but_not_audit_revision(tmp_path):
    store = ApprovalReceiptStore(tmp_path)
    manifest = make_manifest(execution_fingerprint="execution-a")
    pending = pending_decision(
        manifest=manifest,
        kind=ApprovalKind.RESOURCE,
        summary="Resource confirmation for relax",
        resource_proposal={"nodes": 2},
        stage="relax",
    )
    store.record_pending(pending)
    store.approve(pending.decision_id, approved_by="user")

    assert store.valid_receipt(pending, manifest) is not None
    changed_execution = manifest.model_copy(deep=True)
    changed_execution.stages[0].execution_fingerprint = "execution-b"
    assert store.valid_receipt(pending, changed_execution) is None
    # Status/report persistence increments the audit revision but does not
    # alter the stage's semantic decision epoch or execution identity.
    assert store.valid_receipt(
        pending, manifest.model_copy(update={"revision": manifest.revision + 1})
    ) is not None


def test_resource_receipt_requires_exact_stage_decision_epoch(tmp_path):
    store = ApprovalReceiptStore(tmp_path)
    manifest = make_manifest(execution_fingerprint="execution-a")
    manifest.stages[0].execution_fingerprint = "execution-a"
    pending = pending_decision(
        manifest=manifest,
        kind=ApprovalKind.RESOURCE,
        summary="Resource confirmation for relax",
        resource_proposal={"nodes": 2},
        stage="relax",
    )
    store.record_pending(pending)
    store.approve(pending.decision_id, approved_by="user")

    assert store.valid_receipt(pending, manifest) is not None
    changed = manifest.model_copy(deep=True)
    changed.stages[0].decision_epoch += 1
    assert store.valid_receipt(pending, changed) is None


def test_later_stage_execution_identity_does_not_orphan_earlier_stage_receipt(tmp_path):
    store = ApprovalReceiptStore(tmp_path)
    manifest = make_manifest(execution_fingerprint="execution-relax")
    manifest.stages = [
        UnifiedStage(name="relax", execution_fingerprint="execution-relax"),
        UnifiedStage(name="static", depends_on=["relax"]),
    ]
    pending = pending_decision(
        manifest=manifest,
        kind=ApprovalKind.RESOURCE,
        summary="Resource confirmation for relax",
        resource_proposal={"nodes": 2},
        stage="relax",
    )
    store.record_pending(pending)
    store.approve(pending.decision_id, approved_by="user")
    later = manifest.model_copy(deep=True)
    later.stages[1].execution_fingerprint = "execution-static"
    later.stages[1].decision_epoch += 1
    later.execution_fingerprint = "execution-static"

    assert store.valid_receipt(pending, later) is not None


def test_resource_decision_with_null_execution_fingerprint_is_not_approvable(tmp_path):
    store = ApprovalReceiptStore(tmp_path)
    manifest = make_manifest(execution_fingerprint=None)
    with pytest.raises(ValueError, match="execution_fingerprint"):
        pending_decision(
            manifest=manifest,
            kind=ApprovalKind.RESOURCE,
            summary="Resource confirmation for relax",
            resource_proposal={"nodes": 2},
            stage="relax",
        )


def test_scientific_execution_approval_requires_an_exact_stage_identity(tmp_path):
    manifest = make_manifest(execution_fingerprint="execution-a")
    with pytest.raises(ValueError, match="stage execution_fingerprint"):
        pending_decision(
            manifest=manifest,
            kind=ApprovalKind.SCIENTIFIC,
            summary="Scientific recovery confirmation",
        )


def test_pending_decision_id_collision_with_different_payload_is_rejected(tmp_path):
    store = ApprovalReceiptStore(tmp_path)
    manifest = make_manifest(execution_fingerprint="execution-a")
    first = pending_decision(
        manifest=manifest, kind=ApprovalKind.RESOURCE,
        summary="Resource confirmation for relax", stage="relax",
        resource_proposal={"nodes": 2}, decision_id="collision",
    )
    second = pending_decision(
        manifest=manifest, kind=ApprovalKind.RESOURCE,
        summary="Resource confirmation for relax", stage="relax",
        resource_proposal={"nodes": 3}, decision_id="collision",
    )
    store.record_pending(first)
    with pytest.raises(DecisionConflictError):
        store.record_pending(second)
    assert store.load_pending("collision") == first
