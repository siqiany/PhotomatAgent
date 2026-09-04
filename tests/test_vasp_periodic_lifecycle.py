"""Task 7: periodic VASP lifecycle through SubmitOnceSession."""

from __future__ import annotations

from pathlib import Path

import pytest

from photomatagent.scientific.applications.vasp.application import VaspApplication
from photomatagent.scientific.applications.vasp.unified.fingerprints import (
    scientific_fingerprint,
)
from photomatagent.scientific.applications.vasp.unified.models import (
    PeriodicScientificSpec,
    UnifiedStage,
    UnifiedVaspManifest,
    VaspWorkflowKind,
)
from photomatagent.scientific.applications.vasp.unified.periodic import (
    PeriodicVaspExecutor,
)
from photomatagent.scientific.remote.fake import FakeSCNetBackend
from photomatagent.scientific.remote.lifecycle import SubmitOnceSession, SubmissionGate
from photomatagent.scientific.remote.models import (
    HPCJobState,
    ResourcePolicy,
    ResourceRequest,
)
from photomatagent.scientific.remote.registry import JobRegistry
from photomatagent.workspace import Workspace


CIF = """# InAs zincblende test fixture
data_InAs
_symmetry_space_group_name_H-M   'F -4 3 m'
_cell_length_a   6.0583
_cell_length_b   6.0583
_cell_length_c   6.0583
_cell_angle_alpha   90
_cell_angle_beta    90
_cell_angle_gamma   90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
In  In  0.0 0.0 0.0
As  As  0.25 0.25 0.25
"""


def make_manifest(tmp_path: Path, *, profile: str = "standard_semiconductor") -> UnifiedVaspManifest:
    structure = tmp_path / "inAs.cif"
    structure.write_text(CIF, encoding="utf-8")
    spec = PeriodicScientificSpec(
        structure_path="inAs.cif",
        profile=profile,
        scientific_overrides={"encut_ev": 520},
    )
    return UnifiedVaspManifest(
        workflow_id="vasp_0123456789abcdef",
        workflow_kind=VaspWorkflowKind.PERIODIC,
        scientific_spec=spec,
        scientific_fingerprint=scientific_fingerprint(spec),
        stages=[UnifiedStage(name="relax")],
    )


def make_executor(
    tmp_path: Path,
    *,
    backend: FakeSCNetBackend | None = None,
) -> tuple[PeriodicVaspExecutor, FakeSCNetBackend, SubmitOnceSession]:
    backend = backend or FakeSCNetBackend()
    # A fake local PSP library so the executor's POTCAR readiness check
    # passes (real periodic submits refuse to run without a POTCAR source);
    # probe layout element In + target elements In/As are all present.
    psp = tmp_path / "psp"
    for element in ("In", "As"):
        (psp / element).mkdir(parents=True, exist_ok=True)
        (psp / element / "POTCAR").write_text("FAKE POTCAR\n", encoding="utf-8")
    app = VaspApplication(
        backend,
        workspace=tmp_path,
        jobs_local_dir=tmp_path / "vasp_inputs",
        psp_dir=psp,
    )
    registry = JobRegistry(tmp_path / "jobs.sqlite3")
    session = SubmitOnceSession(
        registry,
        backend,
        marker_temp_dir=tmp_path / "markers",
    )
    return PeriodicVaspExecutor(app, session), backend, session


@pytest.mark.asyncio
async def test_two_submits_produce_one_backend_submission_and_immutable_job_id(tmp_path):
    executor, backend, _ = make_executor(tmp_path)
    manifest = make_manifest(tmp_path)
    resource = ResourceRequest(nodes=1, tasks_per_node=8, walltime_minutes=60)

    first = await executor.submit(manifest, manifest.stages[0], resource)
    second = await executor.submit(manifest, manifest.stages[0], resource)

    assert first.ok
    assert first.submitted
    assert first.job_id is not None
    assert second.duplicate
    assert second.job_id == first.job_id
    assert len(backend.submitted_scripts) == 1


@pytest.mark.asyncio
async def test_failed_preflight_causes_zero_upload_and_zero_sbatch_calls(tmp_path):
    executor, backend, _ = make_executor(tmp_path)
    manifest = make_manifest(tmp_path)
    manifest.scientific_spec = manifest.scientific_spec.model_copy(
        update={"structure_path": str(tmp_path / "missing.cif")}
    )
    resource = ResourceRequest(nodes=1, tasks_per_node=8, walltime_minutes=60)

    result = await executor.submit(manifest, manifest.stages[0], resource)

    assert not result.ok
    assert not result.submitted
    assert backend.uploaded == []
    assert backend.submitted_scripts == {}


@pytest.mark.asyncio
async def test_client_timeout_enters_reconciliation(tmp_path):
    backend = FakeSCNetBackend(submit_succeeds_but_times_out=True)
    executor, backend, session = make_executor(tmp_path, backend=backend)
    manifest = make_manifest(tmp_path)
    resource = ResourceRequest(nodes=1, tasks_per_node=8, walltime_minutes=60)

    first = await executor.submit(manifest, manifest.stages[0], resource)

    assert first.needs_reconciliation
    assert first.job_id is None

    recovery = await executor.reconcile(manifest)
    assert recovery.ok
    assert recovery.action == "AUTO_RESUME"


@pytest.mark.asyncio
async def test_multiple_reconciliation_candidates_block(tmp_path):
    backend = FakeSCNetBackend(submit_succeeds_but_times_out=True)
    executor, backend, session = make_executor(tmp_path, backend=backend)
    manifest = make_manifest(tmp_path)
    resource = ResourceRequest(nodes=1, tasks_per_node=8, walltime_minutes=60)

    first = await executor.submit(manifest, manifest.stages[0], resource)
    assert first.needs_reconciliation
    record = session.registry.get(first.request_id)
    assert record is not None and record.job_name is not None
    # The timeout already left one remote job; add a second candidate.
    backend.submitted_job_names["999999"] = record.job_name

    recovery = await executor.reconcile(manifest)
    assert not recovery.ok
    assert recovery.action == "RECONCILE"
    assert any("multiple" in error.lower() for error in recovery.errors)


@pytest.mark.asyncio
async def test_new_attempt_uses_distinct_remote_directory(tmp_path):
    backend = FakeSCNetBackend()
    executor, backend, session = make_executor(tmp_path, backend=backend)
    manifest = make_manifest(tmp_path)
    resource = ResourceRequest(nodes=1, tasks_per_node=8, walltime_minutes=60)

    first = await executor.submit(manifest, manifest.stages[0], resource)
    record_one = session.registry.get(first.request_id)
    assert record_one is not None and record_one.remote_directory is not None

    # A deliberately forced new attempt must not reuse the first directory.
    second = await session.submit_once(
        application="vasp",
        workflow_stage="relax",
        job_name="wf-periodic-relax",
        local_input_dir=record_one.local_input_dir,
        gate=SubmissionGate(passed=True),
        resource=resource,
        executable="vasp_std",
        script_name="vasp.slurm",
        request_id=first.request_id,
        force_new_attempt=True,
    )
    record_two = session.registry.get(second.request_id)
    assert record_two is not None
    assert record_two.remote_directory != record_one.remote_directory
