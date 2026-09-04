"""Task 10: deterministic router and unified service state machine."""

from __future__ import annotations

from pathlib import Path

import pytest

from photomatagent.scientific.applications.vasp.unified.approvals import (
    ApprovalKind,
    ApprovalReceiptStore,
    pending_decision,
)
from photomatagent.scientific.applications.vasp.unified.executors import (
    CollectionResult,
    OperationResult,
    PreflightResult,
    RecoveryResult,
    ReportResult,
    StatusResult,
    SubmissionResult,
)
from photomatagent.scientific.applications.vasp.unified.models import (
    MolecularScientificSpec,
    PeriodicScientificSpec,
    ReportKind,
    ReportRequest,
    UnifiedStage,
    UnifiedVaspRequest,
    VaspWorkflowKind,
    WorkflowState,
)
from photomatagent.scientific.applications.vasp.molecular.models import MoleculeSpec, StageName, StageSpec, WorkflowSpec
from photomatagent.scientific.applications.vasp.unified.repository import (
    ManifestRepository,
)
from photomatagent.scientific.applications.vasp.unified.resources import (
    AutomaticBudget,
    ResourceAuthorizationService,
)
from photomatagent.scientific.applications.vasp.unified.router import (
    UnifiedVaspRouter,
)
from photomatagent.scientific.applications.vasp.unified.service import (
    UnifiedVaspService,
)
from photomatagent.scientific.remote.models import ResourcePolicy, ResourceRequest
from photomatagent.workspace import Workspace


class FakeVaspExecutor:
    prepared = False
    preflighted = False
    submitted = False

    async def prepare(self, manifest):
        self.prepared = True
        return OperationResult(ok=True, data={"prepared": True})

    async def preflight(self, manifest):
        self.preflighted = True
        return PreflightResult(ok=True, passed=True, data={"preflight": True})

    async def submit(self, manifest, stage, resource):
        self.submitted = True
        return SubmissionResult(
            ok=True, request_id="req-1", job_id="1001", submitted=True, data={"stage": stage.name}
        )

    async def status(self, manifest):
        return StatusResult(ok=True, stage_states={"relax": "RUNNING"})

    async def reconcile(self, manifest):
        return RecoveryResult(ok=True, action="AUTO_RESUME")

    async def collect(self, manifest):
        return CollectionResult(ok=True, validated=True, data={"ok": True})

    async def report(self, manifest, request):
        return ReportResult(ok=True, report_kind=request.kind)


class FailedPrepareExecutor(FakeVaspExecutor):
    async def prepare(self, manifest):
        return OperationResult(ok=False)


class FailedPreflightExecutor(FakeVaspExecutor):
    async def preflight(self, manifest):
        return PreflightResult(ok=False, passed=False)


class StageAwareExecutor(FakeVaspExecutor):
    def __init__(self):
        self.calls = []

    async def submit(self, manifest, stage, resource):
        self.calls.append(stage.name)
        return SubmissionResult(ok=True, request_id=f"req-{stage.name}", submitted=True)

    async def collect(self, manifest):
        return CollectionResult(ok=True, validated=True, stage_states={"relax": "VALIDATED"})


class RecoveryActionExecutor(FakeVaspExecutor):
    def __init__(self, action, pending=None, ok=True):
        self.action = action
        self.pending = pending
        self.ok = ok
        self.submit_calls = 0

    async def reconcile(self, manifest):
        return RecoveryResult(ok=self.ok, action=self.action, pending_decision=self.pending)

    async def submit(self, manifest, stage, resource):
        self.submit_calls += 1
        return await super().submit(manifest, stage, resource)


class ContcarRecoveryExecutor(FakeVaspExecutor):
    def __init__(self, artifact_path: str, artifact_hash: str = "b" * 64):
        self.artifact_path = artifact_path
        self.artifact_hash = artifact_hash

    async def reconcile(self, manifest):
        return RecoveryResult(
            ok=True,
            action="AUTO_RESUME",
            contcar_restart=True,
            scientific_fingerprint=manifest.scientific_fingerprint,
            restart_artifact_path=self.artifact_path,
            restart_artifact_sha256=self.artifact_hash,
            restart_structural_validation={"atom_count": 2, "validator": "fake-executor"},
        )


def make_service(tmp_path, executor=None):
    workspace = Workspace(tmp_path)
    repo = ManifestRepository(workspace)
    approvals = ApprovalReceiptStore(tmp_path)
    router = UnifiedVaspRouter(periodic=executor or FakeVaspExecutor())
    resource_service = ResourceAuthorizationService(
        approvals,
        policy=ResourcePolicy(
            allow_hpc_submit=True,
            max_nodes=4,
            max_tasks_per_node=64,
            max_walltime_minutes=600,
            allowed_partitions=["kshcnormal"],
        ),
        automatic_budget=AutomaticBudget(
            max_nodes=4,
            max_tasks_per_node=64,
            max_walltime_minutes=600,
        ),
    )
    service = UnifiedVaspService(repo, approvals, router, resources=resource_service)
    return service, workspace


def periodic_request(workspace):
    (workspace.root / "structure.cif").write_text("CIF", encoding="utf-8")
    return UnifiedVaspRequest(
        workflow_kind=VaspWorkflowKind.PERIODIC,
        scientific_spec=PeriodicScientificSpec(
            structure_path="structure.cif",
            profile="standard_semiconductor",
            scientific_overrides={},
        ),
    )


def test_plan_sets_stages_and_saves(tmp_path):
    service, workspace = make_service(tmp_path)
    manifest = service.plan(periodic_request(workspace))
    assert manifest.workflow_id.startswith("vasp_")
    assert [stage.name for stage in manifest.stages] == [
        "relax", "static", "band", "dos"
    ]
    assert manifest.revision == 1


@pytest.mark.asyncio
async def test_legal_prepare_preflight_and_illegal_second_prepare(tmp_path):
    service, workspace = make_service(tmp_path)
    manifest = service.plan(periodic_request(workspace))
    prepared = await service.prepare(manifest.workflow_id)
    assert prepared.state is WorkflowState.PREPARED
    assert prepared.data == {"prepared": True}
    preflighted = await service.preflight(manifest.workflow_id)
    assert preflighted.state is WorkflowState.PREFLIGHTED
    with pytest.raises(ValueError, match="illegal VASP state transition"):
        await service.prepare(manifest.workflow_id)


@pytest.mark.asyncio
async def test_prepare_executor_ok_false_is_reported_as_failed(tmp_path):
    service, workspace = make_service(tmp_path, FailedPrepareExecutor())
    manifest = service.plan(periodic_request(workspace))

    result = await service.prepare(manifest.workflow_id)

    assert not result.ok
    assert result.state is WorkflowState.FAILED


@pytest.mark.asyncio
async def test_preflight_executor_ok_false_is_reported_as_failed(tmp_path):
    service, workspace = make_service(tmp_path, FailedPreflightExecutor())
    manifest = service.plan(periodic_request(workspace))
    await service.prepare(manifest.workflow_id)

    result = await service.preflight(manifest.workflow_id)

    assert not result.ok
    assert result.state is WorkflowState.FAILED


@pytest.mark.asyncio
async def test_submit_before_preflight_is_blocked(tmp_path):
    service, workspace = make_service(tmp_path)
    manifest = service.plan(periodic_request(workspace))
    with pytest.raises(ValueError, match="submit is only allowed"):
        await service.submit(manifest.workflow_id)


@pytest.mark.asyncio
async def test_submit_after_preflight_succeeds(tmp_path):
    executor = FakeVaspExecutor()
    service, workspace = make_service(tmp_path, executor)
    service.router.register(VaspWorkflowKind.MOLECULAR, executor)
    manifest = service.plan(periodic_request(workspace))
    await service.prepare(manifest.workflow_id)
    await service.preflight(manifest.workflow_id)
    result = await service.submit(manifest.workflow_id)
    assert result.ok
    assert result.state is WorkflowState.SUBMITTED


def test_stale_revision_write_is_rejected(tmp_path):
    service, workspace = make_service(tmp_path)
    manifest = service.plan(periodic_request(workspace))
    with pytest.raises(Exception, match="stale"):
        service.repository.save(manifest, expected_revision=0)


@pytest.mark.asyncio
async def test_optional_executor_unavailability_returns_diagnostic(tmp_path):
    service, workspace = make_service(tmp_path, executor=None)
    # Router without periodic executor should still return a typed result.
    manifest = service.plan(periodic_request(workspace))
    # Replace router with an empty one to simulate optional unavailability.
    service.router = UnifiedVaspRouter()
    result = await service.prepare(manifest.workflow_id)
    assert not result.ok
    assert "no executor" in " ".join(result.errors).lower()


@pytest.mark.asyncio
async def test_child_service_advances_dependency_satisfied_stage(tmp_path):
    executor = StageAwareExecutor()
    service, workspace = make_service(tmp_path, executor)
    service.router.register(VaspWorkflowKind.MOLECULAR, executor)
    structure = workspace.root / "molecule.xyz"
    structure.write_text("1\nX\nX 0 0 0\n", encoding="utf-8")
    workflow = WorkflowSpec(
        molecule=MoleculeSpec(name="X", structure_path="molecule.xyz", structure_kind="xyz", total_charge=0),
        stages=[StageSpec(name=StageName.RELAX), StageSpec(name=StageName.STATIC, depends_on=StageName.RELAX)],
        scientific_method="PBE-D3(BJ)",
    )
    request = UnifiedVaspRequest(
        workflow_kind=VaspWorkflowKind.MOLECULAR,
        scientific_spec=MolecularScientificSpec(workflow=workflow),
    )
    manifest = service.plan(request)
    await service.prepare(manifest.workflow_id)
    await service.preflight(manifest.workflow_id)
    await service.submit(manifest.workflow_id, stage="relax")
    await service.collect(manifest.workflow_id)
    await service.preflight(manifest.workflow_id)
    await service.submit(manifest.workflow_id, stage="static")

    assert executor.calls == ["relax", "static"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("RECONCILE", WorkflowState.RECONCILING),
        ("NEEDS_RESOURCE_CONFIRMATION", WorkflowState.AWAITING_RESOURCE_CONFIRMATION),
        ("NEEDS_SCIENTIFIC_CONFIRMATION", WorkflowState.AWAITING_SCIENTIFIC_CONFIRMATION),
        ("STOP", WorkflowState.FAILED),
    ],
)
async def test_resume_persists_recovery_actions_without_promoting_running(tmp_path, action, expected):
    executor = RecoveryActionExecutor(action)
    service, workspace = make_service(tmp_path, executor)
    manifest = service.plan(periodic_request(workspace))

    result = await service.resume(manifest.workflow_id)

    assert result.state is expected
    assert service.load_manifest(manifest.workflow_id).state is expected
    assert executor.submit_calls == 0
    if action.startswith("NEEDS_"):
        assert result.pending_decision is not None


@pytest.mark.asyncio
async def test_scientific_recovery_confirmation_binds_target_execution_identity(tmp_path):
    executor = RecoveryActionExecutor("NEEDS_SCIENTIFIC_CONFIRMATION")
    service, workspace = make_service(tmp_path, executor)
    manifest = service.plan(periodic_request(workspace))

    result = await service.resume(manifest.workflow_id)

    assert result.pending_decision is not None
    assert result.pending_decision.stage == "relax"
    assert result.pending_decision.execution_fingerprint is not None


@pytest.mark.asyncio
async def test_submit_persists_execution_identity_and_rejects_stale_resource_receipt(tmp_path):
    executor = FakeVaspExecutor()
    service, workspace = make_service(tmp_path, executor)
    service.resources.automatic_budget = AutomaticBudget(max_nodes=1, max_tasks_per_node=32, max_walltime_minutes=120)
    manifest = service.plan(periodic_request(workspace))
    await service.prepare(manifest.workflow_id)
    await service.preflight(manifest.workflow_id)

    current = service.load_manifest(manifest.workflow_id)
    current.stages[0].resource_recommendation = ResourceRequest(
        partition="kshcnormal", nodes=2, tasks_per_node=32, walltime_minutes=240
    )
    service.repository.save(current, expected_revision=current.revision)
    first = await service.submit(manifest.workflow_id)
    assert first.state is WorkflowState.AWAITING_RESOURCE_CONFIRMATION
    assert first.pending_decision is not None
    persisted = service.load_manifest(manifest.workflow_id)
    assert persisted.execution_fingerprint is not None
    assert persisted.stages[0].execution_fingerprint == persisted.execution_fingerprint
    assert executor.submitted is False


@pytest.mark.asyncio
async def test_resource_change_bumps_only_target_stage_decision_epoch(tmp_path):
    executor = FakeVaspExecutor()
    service, workspace = make_service(tmp_path, executor)
    service.resources.automatic_budget = AutomaticBudget(max_nodes=1, max_tasks_per_node=32, max_walltime_minutes=120)
    manifest = service.plan(periodic_request(workspace))
    await service.prepare(manifest.workflow_id)
    await service.preflight(manifest.workflow_id)
    initial = service.load_manifest(manifest.workflow_id)
    initial.stages[0].resource_recommendation = ResourceRequest(
        partition="kshcnormal", nodes=2, tasks_per_node=32, walltime_minutes=240
    )
    service.repository.save(initial, expected_revision=initial.revision)
    first = await service.submit(manifest.workflow_id)
    first_manifest = service.load_manifest(manifest.workflow_id)
    assert first.pending_decision is not None
    first_epoch = first_manifest.stages[0].decision_epoch
    first_scientific = first_manifest.scientific_fingerprint

    service.approvals.approve(first.pending_decision.decision_id, approved_by="user")
    changed = service.load_manifest(manifest.workflow_id)
    changed.stages[0].resource_recommendation = ResourceRequest(
        partition="kshcnormal", nodes=3, tasks_per_node=32, walltime_minutes=240
    )
    service.repository.save(changed, expected_revision=changed.revision)
    await service.submit(manifest.workflow_id)
    second_manifest = service.load_manifest(manifest.workflow_id)

    assert second_manifest.scientific_fingerprint == first_scientific
    assert second_manifest.stages[0].decision_epoch == first_epoch + 1

    service.approvals.approve(first.pending_decision.decision_id, approved_by="user")
    changed = service.load_manifest(manifest.workflow_id)
    changed.stages[0].resource_recommendation = ResourceRequest(
        partition="kshcnormal", nodes=3, tasks_per_node=32, walltime_minutes=240
    )
    service.repository.save(changed, expected_revision=changed.revision)
    stale = await service.submit(manifest.workflow_id)

    assert stale.state is WorkflowState.AWAITING_RESOURCE_CONFIRMATION
    assert stale.pending_decision is not None
    assert stale.pending_decision.decision_id != first.pending_decision.decision_id
    assert executor.submitted is False


@pytest.mark.asyncio
async def test_validated_contcar_resume_records_artifact_hash_in_execution_identity(tmp_path):
    artifact = tmp_path / "restart.contcar"
    artifact.write_bytes(b"validated CONTCAR")
    import hashlib
    service, workspace = make_service(
        tmp_path,
        ContcarRecoveryExecutor(artifact.name, hashlib.sha256(artifact.read_bytes()).hexdigest()),
    )
    manifest = service.plan(periodic_request(workspace))

    resumed = await service.resume(manifest.workflow_id)
    persisted = service.load_manifest(manifest.workflow_id)

    assert resumed.state is WorkflowState.RUNNING
    assert persisted.execution_fingerprint is not None
    assert persisted.stages[0].attempt_inputs["restart_artifact_sha256"] == hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()


@pytest.mark.asyncio
async def test_contcar_resume_rejects_executor_hash_not_matching_workspace_artifact(tmp_path):
    artifact = tmp_path / "restart.contcar"
    artifact.write_bytes(b"actual bytes")
    service, workspace = make_service(
        tmp_path, ContcarRecoveryExecutor(artifact.name, "b" * 64)
    )
    manifest = service.plan(periodic_request(workspace))

    result = await service.resume(manifest.workflow_id)

    assert result.state is WorkflowState.FAILED


@pytest.mark.asyncio
async def test_contcar_resume_rejects_traversal_or_symlink_restart_artifact(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-restart.contcar"
    outside.write_bytes(b"outside")
    linked = tmp_path / "linked.contcar"
    linked.symlink_to(outside)
    import hashlib
    digest = hashlib.sha256(outside.read_bytes()).hexdigest()
    try:
        for reference in ("../" + outside.name, linked.name):
            service, workspace = make_service(
                tmp_path, ContcarRecoveryExecutor(reference, digest)
            )
            manifest = service.plan(periodic_request(workspace))
            assert (await service.resume(manifest.workflow_id)).state is WorkflowState.FAILED
    finally:
        linked.unlink(missing_ok=True)
        outside.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_unsubstantiated_executor_auto_resume_stays_reconciling(tmp_path):
    service, workspace = make_service(tmp_path, FakeVaspExecutor())
    manifest = service.plan(periodic_request(workspace))

    result = await service.resume(manifest.workflow_id)

    assert result.state is WorkflowState.RECONCILING
