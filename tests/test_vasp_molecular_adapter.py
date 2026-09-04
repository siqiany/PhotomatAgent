from __future__ import annotations

from pathlib import Path

import pytest

from photomatagent.scientific.applications.vasp.molecular.models import (
    MoleculeSpec,
    StageName,
    StageSpec,
    WorkflowSpec,
)
from photomatagent.scientific.applications.vasp.molecular.runtime import (
    MolecularVaspRuntime,
)
from photomatagent.scientific.applications.vasp.unified.executors import (
    ReportResult,
)
from photomatagent.scientific.applications.vasp.unified.fingerprints import (
    scientific_fingerprint,
)
from photomatagent.scientific.applications.vasp.unified.models import (
    MolecularScientificSpec,
    ReportKind,
    ReportRequest,
    UnifiedStage,
    UnifiedVaspManifest,
    VaspWorkflowKind,
)
from photomatagent.scientific.applications.vasp.unified.molecular import (
    MolecularVaspExecutorAdapter,
)
from photomatagent.scientific.remote.models import ResourceRequest


def make_manifest(tmp_path: Path, workflow_id: str = "vasp_0123456789abcdef"):
    structure = tmp_path / "molecule.xyz"
    structure.write_text("1\nX\nX 0 0 0\n", encoding="utf-8")
    workflow = WorkflowSpec(
        molecule=MoleculeSpec(
            name="X",
            structure_path=Path("molecule.xyz"),
            structure_kind="xyz",
            total_charge=0,
        ),
        stages=[StageSpec(name=StageName.RELAX)],
        scientific_method="PBE-D3(BJ)",
    )
    spec = MolecularScientificSpec(workflow=workflow)
    return UnifiedVaspManifest(
        workflow_id=workflow_id,
        workflow_kind=VaspWorkflowKind.MOLECULAR,
        scientific_spec=spec,
        scientific_fingerprint=scientific_fingerprint(spec),
        stages=[UnifiedStage(name="relax")],
    )


class FakeFacade:
    async def binding_energy(self, inputs):
        self.inputs = inputs
        return {"ok": True, "results": {"delta_e_ev": -1.0}}

    async def submit(self, stage, workflow, *, resource=None, request_id=None):
        self.resource = resource
        self.request_id = request_id
        return {"ok": True, "summary": {"request_id": request_id, "job_id": "job-1"}}


class TypeErrorFacade(FakeFacade):
    async def submit(self, stage, workflow, *, resource=None, request_id=None):
        if request_id is not None:
            raise TypeError("request_id internal API failure")
        return await super().submit(stage, workflow, resource=resource, request_id=request_id)


@pytest.mark.asyncio
async def test_workflow_directory_is_deterministic_per_manifest(tmp_path):
    runtime = MolecularVaspRuntime(
        backend=None,
        application=None,
        configured=False,
        workflow_dir=tmp_path / "molecules",
        log_dir=tmp_path / "logs",
        registry_path=tmp_path / "jobs.sqlite3",
    )
    adapter = MolecularVaspExecutorAdapter(runtime)
    first = adapter.workflow_dir_for(make_manifest(tmp_path))
    same = adapter.workflow_dir_for(make_manifest(tmp_path))
    other = adapter.workflow_dir_for(make_manifest(tmp_path, "vasp_fedcba9876543210"))

    assert first == same
    assert first != other
    assert first.is_relative_to(tmp_path / "molecules")


@pytest.mark.asyncio
async def test_binding_report_rejects_missing_related_workflows(tmp_path):
    runtime = MolecularVaspRuntime(
        backend=None,
        application=None,
        configured=False,
        workflow_dir=tmp_path / "molecules",
        log_dir=tmp_path / "logs",
        registry_path=tmp_path / "jobs.sqlite3",
    )
    adapter = MolecularVaspExecutorAdapter(runtime)
    result = await adapter.report(
        make_manifest(tmp_path), ReportRequest(kind=ReportKind.BINDING_ENERGY)
    )

    assert not result.ok
    assert result.errors
    assert result.evidence_gaps


@pytest.mark.asyncio
async def test_binding_report_builds_typed_references_from_related_workflows(tmp_path):
    runtime = MolecularVaspRuntime(
        backend=None,
        application=None,
        configured=False,
        workflow_dir=tmp_path / "molecules",
        log_dir=tmp_path / "logs",
        registry_path=tmp_path / "jobs.sqlite3",
    )
    adapter = MolecularVaspExecutorAdapter(runtime)
    complex_manifest = make_manifest(tmp_path)
    fragment_id = "vasp_fedcba9876543210"
    fragment_dir = adapter.workflow_dir_for(complex_manifest).parent / fragment_id
    fragment_dir.mkdir(parents=True)
    (fragment_dir / "workflow.json").write_text(
        complex_manifest.scientific_spec.workflow.model_copy(
            update={
                "molecule": complex_manifest.scientific_spec.workflow.molecule.model_copy(
                    update={"name": "fragment"}
                )
            }
        ).model_dump_json(),
        encoding="utf-8",
    )
    fake = FakeFacade()
    adapter._facade = lambda manifest: fake  # type: ignore[method-assign]

    result = await adapter.report(
        complex_manifest,
        ReportRequest(
            kind=ReportKind.BINDING_ENERGY,
            related_workflow_ids=[fragment_id],
        ),
    )

    assert isinstance(result, ReportResult)
    assert result.ok
    assert fake.inputs.references[0].name == "fragment"


@pytest.mark.asyncio
async def test_submit_forwards_authorized_resource_to_facade(tmp_path):
    runtime = MolecularVaspRuntime(
        backend=None,
        application=None,
        configured=False,
        workflow_dir=tmp_path / "molecules",
        log_dir=tmp_path / "logs",
        registry_path=tmp_path / "jobs.sqlite3",
    )
    adapter = MolecularVaspExecutorAdapter(runtime)
    fake = FakeFacade()
    adapter._facade = lambda manifest: fake  # type: ignore[method-assign]
    resource = ResourceRequest(
        partition="kshcnormal", nodes=2, tasks_per_node=16, walltime_minutes=77
    )

    result = await adapter.submit(
        make_manifest(tmp_path), UnifiedStage(name="relax"), resource
    )

    assert result.ok
    assert fake.resource == resource


@pytest.mark.asyncio
async def test_molecular_resource_change_produces_a_distinct_authorized_request_id(tmp_path):
    runtime = MolecularVaspRuntime(
        backend=None, application=None, configured=False,
        workflow_dir=tmp_path / "molecules", log_dir=tmp_path / "logs",
        registry_path=tmp_path / "jobs.sqlite3",
    )
    adapter = MolecularVaspExecutorAdapter(runtime)
    fake = FakeFacade()
    adapter._facade = lambda manifest: fake  # type: ignore[method-assign]
    manifest = make_manifest(tmp_path)
    first = await adapter.submit(
        manifest,
        UnifiedStage(name="relax", execution_fingerprint="a" * 64),
        ResourceRequest(nodes=1, tasks_per_node=8, walltime_minutes=60),
    )
    second = await adapter.submit(
        manifest,
        UnifiedStage(name="relax", execution_fingerprint="b" * 64),
        ResourceRequest(nodes=2, tasks_per_node=8, walltime_minutes=60),
    )

    assert first.request_id != second.request_id


@pytest.mark.asyncio
async def test_molecular_adapter_does_not_swallow_a_real_facade_type_error(tmp_path):
    runtime = MolecularVaspRuntime(
        backend=None, application=None, configured=False,
        workflow_dir=tmp_path / "molecules", log_dir=tmp_path / "logs",
        registry_path=tmp_path / "jobs.sqlite3",
    )
    adapter = MolecularVaspExecutorAdapter(runtime)
    adapter._facade = lambda manifest: TypeErrorFacade()  # type: ignore[method-assign]

    with pytest.raises(TypeError, match="internal API failure"):
        await adapter.submit(
            make_manifest(tmp_path), UnifiedStage(name="relax", execution_fingerprint="a" * 64),
            ResourceRequest(nodes=1, tasks_per_node=8, walltime_minutes=60),
        )


def test_adapter_rejects_nonrelative_and_symlinked_manifest_structure_paths(tmp_path):
    runtime = MolecularVaspRuntime(
        backend=None, application=None, configured=False,
        workflow_dir=tmp_path / "molecules", log_dir=tmp_path / "logs",
        registry_path=tmp_path / "jobs.sqlite3",
    )
    adapter = MolecularVaspExecutorAdapter(runtime)
    manifest = make_manifest(tmp_path)
    external = tmp_path.parent / f"{tmp_path.name}-external.xyz"
    external.write_text("1\nX\nX 0 0 0\n", encoding="utf-8")
    escape_link = tmp_path / "escape.xyz"
    escape_link.symlink_to(external)
    try:
        for path in (
            tmp_path / "molecule.xyz",
            external,
            Path("..") / "molecule.xyz",
            escape_link,
        ):
            workflow = manifest.scientific_spec.workflow.model_copy(update={
                "molecule": manifest.scientific_spec.workflow.molecule.model_copy(
                    update={"structure_path": path}
                )
            })
            tampered = manifest.model_copy(update={
                "scientific_spec": MolecularScientificSpec(workflow=workflow)
            })
            with pytest.raises(ValueError, match="workspace-relative"):
                adapter._workflow(tampered)
    finally:
        escape_link.unlink(missing_ok=True)
        external.unlink(missing_ok=True)
