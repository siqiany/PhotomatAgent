"""Task 5: unify VASP resource authorization around existing ResourcePolicy."""

from __future__ import annotations

from photomatagent.scientific.applications.vasp.molecular.models import (
    MoleculeSpec,
    ResourcePlan,
    ResourceProfile,
    StageName,
    StageSpec,
    WorkflowSpec,
)
from photomatagent.scientific.applications.vasp.unified.approvals import (
    ApprovalReceiptStore,
)
from photomatagent.scientific.applications.vasp.unified.fingerprints import (
    execution_fingerprint,
    scientific_fingerprint,
)
from photomatagent.scientific.applications.vasp.unified.models import (
    MolecularScientificSpec,
    PeriodicScientificSpec,
    UnifiedStage,
    UnifiedVaspManifest,
    VaspWorkflowKind,
)
from photomatagent.scientific.applications.vasp.unified.resources import (
    AutomaticBudget,
    ResourceAuthorizationService,
    ResourceDecisionState,
    VaspResourcePlanner,
)
from photomatagent.scientific.remote.models import ResourcePolicy, ResourceRequest
from photomatagent.workspace import Workspace


def periodic_manifest(*, workflow_id: str = "wf-1") -> UnifiedVaspManifest:
    spec = PeriodicScientificSpec(
        structure_path="structure.cif",
        profile="standard_semiconductor",
        scientific_overrides={},
    )
    return UnifiedVaspManifest(
        workflow_id=workflow_id,
        workflow_kind=VaspWorkflowKind.PERIODIC,
        scientific_spec=spec,
        scientific_fingerprint=scientific_fingerprint(spec),
        stages=[UnifiedStage(
            name="relax",
            execution_fingerprint=execution_fingerprint(
                scientific_fingerprint(spec), stage="relax"
            ),
        )],
        execution_fingerprint=execution_fingerprint(scientific_fingerprint(spec), stage="relax"),
    )


def molecular_manifest(*, calibrated: bool = False) -> UnifiedVaspManifest:
    spec = MolecularScientificSpec(
        workflow=WorkflowSpec(
            molecule=MoleculeSpec(
                name="Li",
                total_charge=0,
                spin_multiplicity=1,
                structure_path="li.xyz",
                structure_kind="xyz",
            ),
            stages=[StageSpec(name=StageName.RELAX, depends_on=None)],
            scientific_method="PBE-D3(BJ)",
            resource_plan=ResourcePlan(
                profile=ResourceProfile.PRODUCTION,
                resource_calibrated=calibrated,
                tasks_per_node=8,
                walltime_minutes=20,
            ),
            resource_ceiling={
                "partition": "kshcnormal",
                "nodes": 1,
                "tasks_per_node": 8,
                "walltime_minutes": 20,
            },
        )
    )
    return UnifiedVaspManifest(
        workflow_id="wf-mol",
        workflow_kind=VaspWorkflowKind.MOLECULAR,
        scientific_spec=spec,
        scientific_fingerprint=scientific_fingerprint(spec),
        stages=[UnifiedStage(
            name="relax",
            execution_fingerprint=execution_fingerprint(
                scientific_fingerprint(spec), stage="relax"
            ),
        )],
        execution_fingerprint=execution_fingerprint(scientific_fingerprint(spec), stage="relax"),
    )


def test_in_budget_recommendation_is_allowed(tmp_path):
    store = ApprovalReceiptStore(tmp_path)
    service = ResourceAuthorizationService(
        approval_store=store,
        policy=ResourcePolicy(
            allow_hpc_submit=True,
            max_nodes=4,
            max_tasks_per_node=64,
            max_walltime_minutes=600,
            allowed_partitions=["kshcnormal"],
        ),
    )
    manifest = periodic_manifest()
    decision = service.decide(
        manifest,
        manifest.stages[0],
        ResourceRequest(
            partition="kshcnormal",
            nodes=1,
            tasks_per_node=16,
            walltime_minutes=60,
        ),
    )
    assert decision.state is ResourceDecisionState.ALLOWED
    assert decision.effective is not None


def test_above_auto_but_under_hard_cap_returns_pending(tmp_path):
    store = ApprovalReceiptStore(tmp_path)
    service = ResourceAuthorizationService(
        approval_store=store,
        # Pin the DEFAULT automatic budget: env overrides
        # (PHOTOMATAGENT_VASP_AUTO_*) must not change this test's outcome.
        automatic_budget=AutomaticBudget(),
        policy=ResourcePolicy(
            allow_hpc_submit=True,
            max_nodes=8,
            max_tasks_per_node=64,
            max_walltime_minutes=600,
            allowed_partitions=["kshcnormal"],
        ),
    )
    manifest = periodic_manifest()
    decision = service.decide(
        manifest,
        manifest.stages[0],
        ResourceRequest(
            partition="kshcnormal",
            nodes=4,
            tasks_per_node=32,
            walltime_minutes=240,
        ),
    )
    assert decision.state is ResourceDecisionState.NEEDS_CONFIRMATION
    assert decision.effective is None
    assert decision.pending_decision is not None
    assert decision.pending_decision.execution_fingerprint is not None


def test_above_hard_cap_is_denied_even_with_receipt(tmp_path):
    store = ApprovalReceiptStore(tmp_path)
    service = ResourceAuthorizationService(
        approval_store=store,
        policy=ResourcePolicy(
            allow_hpc_submit=True,
            max_nodes=2,
            max_tasks_per_node=64,
            max_walltime_minutes=600,
            allowed_partitions=["kshcnormal"],
        ),
    )
    manifest = periodic_manifest()
    request = ResourceRequest(
        partition="kshcnormal",
        nodes=4,
        tasks_per_node=32,
        walltime_minutes=240,
    )
    first = service.decide(manifest, manifest.stages[0], request)
    assert first.state is ResourceDecisionState.DENIED
    assert first.pending_decision is not None
    store.approve(first.pending_decision.decision_id, approved_by="local:user")

    again = service.decide(manifest, manifest.stages[0], request)
    assert again.state is ResourceDecisionState.DENIED


def test_disabled_allow_hpc_submit_is_denied(tmp_path):
    service = ResourceAuthorizationService(
        policy=ResourcePolicy(allow_hpc_submit=False),
    )
    manifest = periodic_manifest()
    decision = service.decide(
        manifest,
        manifest.stages[0],
        ResourceRequest(nodes=1, tasks_per_node=8, walltime_minutes=60),
    )
    assert decision.state is ResourceDecisionState.DENIED
    assert any("disabled" in reason for reason in decision.reasons)


def test_partition_and_calibration_constraints_remain_effective(tmp_path):
    store = ApprovalReceiptStore(tmp_path)
    service = ResourceAuthorizationService(
        approval_store=store,
        policy=ResourcePolicy(
            allow_hpc_submit=True,
            allowed_partitions=["kshcnormal"],
        ),
    )
    manifest = periodic_manifest()
    bad_partition = service.decide(
        manifest,
        manifest.stages[0],
        ResourceRequest(
            partition="wrong_partition",
            nodes=1,
            tasks_per_node=8,
            walltime_minutes=60,
        ),
    )
    assert bad_partition.state is ResourceDecisionState.DENIED

    mol = molecular_manifest(calibrated=False)
    uncalibrated = service.decide(
        mol,
        mol.stages[0],
        ResourceRequest(
            partition="kshcnormal",
            nodes=1,
            tasks_per_node=8,
            walltime_minutes=20,
        ),
    )
    assert uncalibrated.state is ResourceDecisionState.DENIED


def test_decision_and_reason_list_contain_no_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("SCNET_USERNAME", "super-secret-user")
    monkeypatch.setenv("SCNET_HOST", "secret-host")
    service = ResourceAuthorizationService(
        policy=ResourcePolicy(allow_hpc_submit=True, max_nodes=1),
    )
    manifest = periodic_manifest()
    decision = service.decide(
        manifest,
        manifest.stages[0],
        ResourceRequest(nodes=4, tasks_per_node=32, walltime_minutes=240),
    )
    serialized = decision.model_dump_json()
    assert "super-secret-user" not in serialized
    assert "secret-host" not in serialized


def test_planner_recommends_from_stage_or_profile(tmp_path):
    planner = VaspResourcePlanner()
    manifest = periodic_manifest()
    recommended = planner.recommend(manifest, manifest.stages[0])
    assert recommended.partition == "kshcnormal"
    assert recommended.nodes >= 1

    explicit = ResourceRequest(
        partition="kshcnormal", nodes=2, tasks_per_node=16, walltime_minutes=120
    )
    stage = UnifiedStage(name="relax", resource_recommendation=explicit)
    assert planner.recommend(manifest, stage) == explicit
