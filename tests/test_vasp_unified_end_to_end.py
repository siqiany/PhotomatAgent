"""Task 16: unified periodic/molecular/study flow through fake backends.

This is an offline end-to-end verification using fake executors; it never
connects to SSH/SCNet or submits a real Slurm job.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from photomatagent.scientific.applications.vasp.unified.approvals import (
    ApprovalReceiptStore,
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
    PeriodicScientificSpec,
    ReportKind,
    ReportRequest,
    UnifiedVaspRequest,
    VaspWorkflowKind,
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
from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.remote.models import ResourcePolicy, ResourceRequest
from photomatagent.workspace import Workspace


class FakeE2EExecutor:
    def __init__(self) -> None:
        self.submit_calls = 0

    async def prepare(self, manifest):
        return OperationResult(ok=True, data={"prepared": True})

    async def preflight(self, manifest):
        return PreflightResult(ok=True, passed=True)

    async def submit(self, manifest, stage, resource):
        self.submit_calls += 1
        return SubmissionResult(
            ok=True,
            request_id=f"req-{manifest.workflow_id}-{stage.name}",
            job_id="1001",
            submitted=True,
        )

    async def status(self, manifest):
        return StatusResult(ok=True, stage_states={s.name: "RUNNING" for s in manifest.stages})

    async def reconcile(self, manifest):
        return RecoveryResult(ok=True, action="AUTO_RESUME")

    async def collect(self, manifest):
        return CollectionResult(
            ok=True,
            validated=True,
            evidence=[
                ScientificEvidence(
                    subject=manifest.workflow_id,
                    property="total_energy",
                    value=-1.0,
                    unit="eV",
                    source="fake",
                    method="dft",
                    fidelity="dft",
                )
            ],
        )

    async def report(self, manifest, request):
        return ReportResult(ok=True, report_kind=request.kind)


def make_service(tmp_path, executor):
    workspace = Workspace(tmp_path)
    repo = ManifestRepository(workspace)
    approvals = ApprovalReceiptStore(tmp_path)
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
            max_nodes=4, max_tasks_per_node=64, max_walltime_minutes=600
        ),
    )
    router = UnifiedVaspRouter(periodic=executor)
    service = UnifiedVaspService(repo, approvals, router, resources=resource_service)
    return service, workspace


def request_for(workspace):
    (workspace.root / "structure.cif").write_text("CIF", encoding="utf-8")
    return UnifiedVaspRequest(
        workflow_kind=VaspWorkflowKind.PERIODIC,
        scientific_spec=PeriodicScientificSpec(
            structure_path="structure.cif",
            profile="standard_semiconductor",
            scientific_overrides={},
        ),
    )


@pytest.mark.asyncio
async def test_periodic_end_to_end_fake_backend(tmp_path):
    executor = FakeE2EExecutor()
    service, workspace = make_service(tmp_path, executor)
    manifest = service.plan(request_for(workspace))
    await service.prepare(manifest.workflow_id)
    await service.preflight(manifest.workflow_id)
    first = await service.submit(manifest.workflow_id)
    second = await service.submit(manifest.workflow_id)

    assert first.ok
    assert second.ok
    assert executor.submit_calls == 1  # second service submit is duplicate
    status = await service.status(manifest.workflow_id)
    collected = await service.collect(manifest.workflow_id)
    report = await service.report(
        manifest.workflow_id, ReportRequest(kind=ReportKind.SUMMARY)
    )
    assert status.ok
    assert collected.ok
    assert len(collected.evidence) == 1
    assert report.ok
