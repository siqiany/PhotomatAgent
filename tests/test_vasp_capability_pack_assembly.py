"""Production wiring for the one unified VASP capability pack."""

from __future__ import annotations

import pytest

from photomatagent.scientific.applications.vasp.application import VaspApplication
from photomatagent.scientific.applications.vasp.tools import VaspCapabilityPack
from photomatagent.scientific.applications.vasp.unified.molecular import (
    MolecularVaspExecutorAdapter,
)
from photomatagent.scientific.applications.vasp.unified.periodic import (
    PeriodicVaspExecutor,
)
from photomatagent.scientific.applications.vasp.unified.study import (
    VaspStudyExecutorAdapter,
)
from photomatagent.scientific.applications.vasp.unified.models import (
    MolecularScientificSpec,
    PeriodicScientificSpec,
    UnifiedVaspRequest,
    VaspWorkflowKind,
)
from photomatagent.scientific.remote.fake import FakeSCNetBackend
from photomatagent.scientific.remote.models import (
    ResourcePolicy,
    ResourceRequest,
)
from photomatagent.scientific.applications.vasp.molecular.models import (
    MoleculeSpec,
    StageName,
    StageSpec,
    WorkflowSpec,
)
from photomatagent.tools.exposure import ToolExposure


def test_capability_pack_assembles_one_shared_service_for_all_workflow_kinds(
    tmp_path,
) -> None:
    """Registered production tools share periodic, molecular, and study routing."""
    application = VaspApplication(FakeSCNetBackend(), workspace=tmp_path)
    pack = VaspCapabilityPack(application=application, workspace=tmp_path)

    first_tools = pack.tools()
    second_tools = pack.tools()
    first_service = first_tools[0].service
    second_service = second_tools[0].service

    assert first_service is second_service
    assert isinstance(
        first_service.router.executor_for(VaspWorkflowKind.PERIODIC),
        PeriodicVaspExecutor,
    )
    molecular = first_service.router.executor_for(VaspWorkflowKind.MOLECULAR)
    assert isinstance(molecular, MolecularVaspExecutorAdapter)
    assert molecular.workspace is first_service.repository.workspace
    study = first_service.router.executor_for(VaspWorkflowKind.STUDY)
    assert isinstance(study, VaspStudyExecutorAdapter)
    assert study.child_service is first_service
    assert first_service.repository is study.child_service.repository
    assert first_service.approvals is study.child_service.approvals
    assert first_service.resources.approval_store is first_service.approvals
    assert {tool.name for tool in first_tools} == {
        "vasp.capabilities",
        "vasp.plan",
        "vasp.prepare",
        "vasp.preflight",
        "vasp.submit",
        "vasp.status",
        "vasp.wait",
        "vasp.resume",
        "vasp.collect",
        "vasp.report",
    }
    assert all(tool.exposure is ToolExposure.DEFERRED for tool in first_tools)


def test_capability_pack_has_no_legacy_alternative_pack_constructors(tmp_path) -> None:
    pack = VaspCapabilityPack(application=None, workspace=tmp_path)

    assert not hasattr(pack, "_molecular_tools")
    assert not hasattr(pack, "_study_tools")


def test_unconfigured_pack_keeps_molecular_and_study_executors_service_backed(
    tmp_path,
) -> None:
    """Missing SCNet configuration leaves lifecycle diagnostics, not routing gaps."""
    pack = VaspCapabilityPack(application=None, workspace=tmp_path)
    service = pack.tools()[0].service

    molecular = service.router.executor_for(VaspWorkflowKind.MOLECULAR)
    study = service.router.executor_for(VaspWorkflowKind.STUDY)

    assert isinstance(molecular, MolecularVaspExecutorAdapter)
    assert isinstance(study, VaspStudyExecutorAdapter)
    assert study.child_service is service


@pytest.mark.asyncio
async def test_pack_rehomes_periodic_application_to_its_authoritative_workspace(
    tmp_path,
) -> None:
    """Periodic manifests and generated stage roots share the pack workspace."""
    application_root = tmp_path / "application-root"
    pack_root = tmp_path / "pack-root"
    application_root.mkdir()
    pack_root.mkdir()
    (pack_root / "structure.POSCAR").write_text(
        "C\n1.0\n1 0 0\n0 1 0\n0 0 1\nC\n1\nDirect\n0 0 0\n",
        encoding="utf-8",
    )
    application = VaspApplication(
        FakeSCNetBackend(), workspace=application_root
    )
    pack = VaspCapabilityPack(application=application, workspace=pack_root)
    service = pack.tools()[0].service
    periodic = service.router.executor_for(VaspWorkflowKind.PERIODIC)
    manifest = service.plan(
        UnifiedVaspRequest(
            workflow_kind=VaspWorkflowKind.PERIODIC,
            scientific_spec=PeriodicScientificSpec(
                structure_path="structure.POSCAR",
                profile="standard_semiconductor",
            ),
        )
    )

    result = await service.prepare(manifest.workflow_id)

    assert isinstance(periodic, PeriodicVaspExecutor)
    assert periodic.application.workspace == pack_root.resolve()
    assert result.ok
    assert (pack_root / ".photomatagent" / "vasp" / "workflows" / manifest.workflow_id / "inputs").is_dir()
    assert not (application_root / ".photomatagent" / "vasp" / "workflows").exists()


def test_pack_uses_application_hard_resource_policy_over_environment_defaults(
    monkeypatch, tmp_path
) -> None:
    """The application policy denies submission even when env defaults allow it."""
    monkeypatch.setenv("PHOTOMATAGENT_ALLOW_HPC_SUBMIT", "1")
    policy = ResourcePolicy(
        allow_hpc_submit=False,
        max_nodes=1,
        max_tasks_per_node=4,
        max_walltime_minutes=30,
        allowed_partitions=["locked"],
    )
    application = VaspApplication(
        FakeSCNetBackend(policy=policy), workspace=tmp_path, policy=policy
    )
    service = VaspCapabilityPack(application=application, workspace=tmp_path).tools()[0].service
    (tmp_path / "structure.POSCAR").write_text(
        "C\n1.0\n1 0 0\n0 1 0\n0 0 1\nC\n1\nDirect\n0 0 0\n",
        encoding="utf-8",
    )
    manifest = service.plan(
        UnifiedVaspRequest(
            workflow_kind=VaspWorkflowKind.PERIODIC,
            scientific_spec=PeriodicScientificSpec(
                structure_path="structure.POSCAR",
                profile="standard_semiconductor",
            ),
        )
    )
    request = ResourceRequest(
        partition="open", nodes=2, tasks_per_node=8, walltime_minutes=31
    )
    manifest = service._persist_execution_identity(
        manifest, manifest.stages[0], request
    )
    decision = service.resources.decide(manifest, manifest.stages[0], request)

    assert service.resources.policy is policy
    assert decision.effective is None
    assert any("disabled" in reason for reason in decision.reasons)
    assert any("nodes=2" in reason for reason in decision.reasons)
    assert any("tasks_per_node=8" in reason for reason in decision.reasons)
    assert any("walltime 31" in reason for reason in decision.reasons)
    assert any("allowed partitions" in reason for reason in decision.reasons)


def test_pack_shares_explicit_application_remote_root_with_molecular_and_study(
    monkeypatch, tmp_path
) -> None:
    """Application configuration supplies the remote root when env is absent."""
    monkeypatch.delenv("SCNET_REMOTE_ROOT", raising=False)
    application = VaspApplication(
        FakeSCNetBackend(), workspace=tmp_path, remote_root="~/explicit-root"
    )
    service = VaspCapabilityPack(application=application, workspace=tmp_path).tools()[0].service
    molecular = service.router.executor_for(VaspWorkflowKind.MOLECULAR)
    study = service.router.executor_for(VaspWorkflowKind.STUDY)

    assert isinstance(molecular, MolecularVaspExecutorAdapter)
    assert isinstance(study, VaspStudyExecutorAdapter)
    assert molecular.runtime.remote_root == "~/explicit-root"
    assert study.runtime is molecular.runtime
    assert study.runtime.remote_root == "~/explicit-root"


def test_remote_root_environment_override_precedes_application_configuration(
    monkeypatch, tmp_path
) -> None:
    """Existing deployment environment override remains the highest priority."""
    monkeypatch.setenv("SCNET_REMOTE_ROOT", "~/environment-root")
    application = VaspApplication(
        FakeSCNetBackend(), workspace=tmp_path, remote_root="~/explicit-root"
    )
    service = VaspCapabilityPack(application=application, workspace=tmp_path).tools()[0].service
    molecular = service.router.executor_for(VaspWorkflowKind.MOLECULAR)

    assert isinstance(molecular, MolecularVaspExecutorAdapter)
    assert molecular.runtime.remote_root == "~/environment-root"


@pytest.mark.asyncio
async def test_unconfigured_pack_returns_typed_molecular_lifecycle_diagnostics(
    tmp_path,
) -> None:
    """Unconfigured molecular workflows route through service, not a missing executor."""
    (tmp_path / "molecule.xyz").write_text(
        "1\nX\nH 0 0 0\n", encoding="utf-8"
    )
    workflow = WorkflowSpec(
        molecule=MoleculeSpec(
            name="hydrogen",
            structure_path="molecule.xyz",
            structure_kind="xyz",
            total_charge=0,
        ),
        stages=[StageSpec(name=StageName.RELAX)],
        scientific_method="PBE-D3(BJ)",
    )
    service = VaspCapabilityPack(application=None, workspace=tmp_path).tools()[0].service
    manifest = service.plan(
        UnifiedVaspRequest(
            workflow_kind=VaspWorkflowKind.MOLECULAR,
            scientific_spec=MolecularScientificSpec(workflow=workflow),
        )
    )

    result = await service.prepare(manifest.workflow_id)

    assert not result.ok
    assert result.errors
    assert "no executor" not in " ".join(result.errors).lower()
    assert result.state.value == "FAILED"
