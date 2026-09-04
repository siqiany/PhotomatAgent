"""> confirmed-failure reset and fresh-attempt resubmission for unified VASP.

Covers the recovery path behind ``vasp.resume``: a workflow whose previously
submitted job is scheduler-confirmed FAILED can be reset back to PREFLIGHTED
and re-submitted; the re-submission creates a NEW attempt request id (parent
pointer kept) instead of returning the terminal duplicate. All of it is
offline (FakeSCNetBackend) and never touches SSH/Slurm.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from photomatagent.scientific.applications.vasp.application import VaspApplication
from photomatagent.scientific.applications.vasp.unified.approvals import (
    ApprovalReceiptStore,
)
from photomatagent.scientific.applications.vasp.unified.models import (
    PeriodicScientificSpec,
    UnifiedVaspRequest,
    VaspWorkflowKind,
    WorkflowState,
)
from photomatagent.scientific.applications.vasp.unified.periodic import (
    PeriodicVaspExecutor,
)
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
from photomatagent.scientific.remote.fake import FakeSCNetBackend
from photomatagent.scientific.remote.lifecycle import SubmitOnceSession
from photomatagent.scientific.remote.models import (
    HPCJobState,
    ResourcePolicy,
)
from photomatagent.scientific.remote.registry import JobLifecycleState, JobRegistry
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


def make_service(
    tmp_path: Path, *, scripted_states: list[HPCJobState] | None = None
) -> tuple[UnifiedVaspService, FakeSCNetBackend, SubmitOnceSession, dict]:
    """Full offline stack: unified service + periodic executor + fake SCNet.

    A fake local PSP library (In/As) is provided so POTCAR readiness passes.
    """
    (tmp_path / "structure.cif").write_text(CIF, encoding="utf-8")
    psp = tmp_path / "psp"
    for element in ("In", "As"):
        (psp / element).mkdir(parents=True, exist_ok=True)
        (psp / element / "POTCAR").write_text("FAKE POTCAR\n", encoding="utf-8")
    backend = FakeSCNetBackend(scripted_states=scripted_states or [])
    app = VaspApplication(
        backend,
        workspace=tmp_path,
        jobs_local_dir=tmp_path / "vasp_inputs",
        psp_dir=psp,
    )
    workspace = Workspace(tmp_path)
    registry = JobRegistry(tmp_path / "jobs.sqlite3")
    session = SubmitOnceSession(
        registry,
        backend,
        marker_temp_dir=tmp_path / "markers",
    )
    executor = PeriodicVaspExecutor(app, session)
    repo = ManifestRepository(workspace)
    approvals = ApprovalReceiptStore(tmp_path)
    resources = ResourceAuthorizationService(
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
    service = UnifiedVaspService(
        repo,
        approvals,
        UnifiedVaspRouter(periodic=executor),
        resources=resources,
    )
    return service, backend, session, {"workspace": workspace, "registry": registry}


def planned_workflow_id(service: UnifiedVaspService) -> str:
    request = UnifiedVaspRequest(
        workflow_kind=VaspWorkflowKind.PERIODIC,
        scientific_spec=PeriodicScientificSpec(
            structure_path="structure.cif",
            profile="standard_semiconductor",
            scientific_overrides={},
        ),
    )
    return service.plan(request).workflow_id


async def run_to_submitted(service: UnifiedVaspService, workflow_id: str) -> None:
    await service.prepare(workflow_id)
    await service.preflight(workflow_id)


async def submit_first_attempt(
    service: UnifiedVaspService, workflow_id: str
) -> dict:
    return (await service.submit(workflow_id)).model_dump(mode="json")


@pytest.mark.asyncio
async def test_confirmed_failure_resets_to_preflighted_and_creates_new_attempt(
    tmp_path,
):
    service, backend, session, extra = make_service(
        tmp_path, scripted_states=[HPCJobState.FAILED]
    )
    workflow_id = planned_workflow_id(service)
    await run_to_submitted(service, workflow_id)

    first = await service.submit(workflow_id)
    assert first.ok
    assert first.state is WorkflowState.SUBMITTED
    original_request_id = first.data["record"]["request_id"]
    original_job_id = first.data["record"]["job_id"]
    assert original_job_id is not None
    registry = extra["registry"]
    assert (
        registry.get(original_request_id).state is JobLifecycleState.SUBMITTED
    )

    # Scheduler-confirmed failure: the fake job now reports FAILED.
    status = await service.status(workflow_id)
    assert status.state is WorkflowState.FAILED

    # resume confirms the terminal failure and resets to PREFLIGHTED.
    resumed = await service.resume(workflow_id)
    assert resumed.ok
    assert resumed.state is WorkflowState.PREFLIGHTED
    assert resumed.data.get("reset") is True

    # Re-submit: a fresh attempt, never the terminal duplicate.
    second = await service.submit(workflow_id)
    assert second.ok
    assert second.state is WorkflowState.SUBMITTED
    new_request_id = second.data["record"]["request_id"]
    new_job_id = second.data["record"]["job_id"]
    assert new_request_id != original_request_id
    assert new_job_id != original_job_id
    new_record = registry.get(new_request_id)
    assert new_record is not None
    assert new_record.parent_request_id == original_request_id
    # Immutable job id: the original request still pins its first job.
    assert registry.get(original_request_id).job_id == original_job_id
    # Two unique remote directories, never reused.
    assert (
        registry.get(original_request_id).remote_directory
        != registry.get(new_request_id).remote_directory
    )

    # Subsequent lifecycle operations must follow the new attempt rather than
    # rediscovering the terminal parent request.
    assert service.load_manifest(workflow_id).stages[0].request_id == new_request_id
    follow_up_status = await service.status(workflow_id)
    assert follow_up_status.state is WorkflowState.FAILED
    follow_up_collect = await service.collect(workflow_id)
    assert follow_up_collect.state is WorkflowState.VALIDATION_FAILED


@pytest.mark.asyncio
async def test_resume_does_not_reset_while_job_is_active(tmp_path):
    service, backend, session, _ = make_service(
        tmp_path, scripted_states=[HPCJobState.RUNNING]
    )
    workflow_id = planned_workflow_id(service)
    await run_to_submitted(service, workflow_id)
    first = await service.submit(workflow_id)
    assert first.ok

    status = await service.status(workflow_id)
    assert status.state is WorkflowState.RUNNING

    resumed = await service.resume(workflow_id)
    assert resumed.ok
    assert resumed.state is WorkflowState.RUNNING
    assert resumed.data.get("reset") is not True


@pytest.mark.asyncio
async def test_resubmit_without_resume_returns_terminal_duplicate(tmp_path):
    service, backend, session, _ = make_service(
        tmp_path, scripted_states=[HPCJobState.FAILED]
    )
    workflow_id = planned_workflow_id(service)
    await run_to_submitted(service, workflow_id)
    first = await service.submit(workflow_id)
    assert first.ok
    await service.status(workflow_id)  # moves workflow to FAILED

    # submit is only valid from PREFLIGHTED/AWAITING_*: no blind resubmission.
    with pytest.raises(ValueError, match="submit is only allowed"):
        await service.submit(workflow_id)
