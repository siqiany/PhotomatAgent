"""Task 6: internal VASP executor contract and typed operation results."""

from __future__ import annotations

import inspect

import pytest

from photomatagent.scientific.applications.vasp.unified.executors import (
    CollectionResult,
    OperationResult,
    PreflightResult,
    RecoveryResult,
    ReportResult,
    ServiceResult,
    StatusResult,
    SubmissionResult,
    VaspWorkflowExecutor,
)
from photomatagent.scientific.applications.vasp.unified.models import (
    ReportKind,
    ReportRequest,
    UnifiedStage,
    UnifiedVaspManifest,
    WorkflowState,
)
from photomatagent.scientific.remote.models import ResourceRequest


class FakePeriodicExecutor:
    async def prepare(self, manifest: UnifiedVaspManifest) -> OperationResult:
        return OperationResult(ok=True)

    async def preflight(self, manifest: UnifiedVaspManifest) -> PreflightResult:
        return PreflightResult(ok=True, passed=True)

    async def submit(
        self,
        manifest: UnifiedVaspManifest,
        stage: UnifiedStage,
        resource: ResourceRequest,
    ) -> SubmissionResult:
        return SubmissionResult(ok=True, request_id="req", submitted=True, job_id="job")

    async def status(self, manifest: UnifiedVaspManifest) -> StatusResult:
        return StatusResult(ok=True, stage_states={"relax": "RUNNING"})

    async def reconcile(self, manifest: UnifiedVaspManifest) -> RecoveryResult:
        return RecoveryResult(ok=True, action="RECONCILE")

    async def collect(self, manifest: UnifiedVaspManifest) -> CollectionResult:
        return CollectionResult(ok=True, validated=True)

    async def report(
        self, manifest: UnifiedVaspManifest, request: ReportRequest
    ) -> ReportResult:
        return ReportResult(ok=True, report_kind=request.kind)


class FakeMolecularExecutor(FakePeriodicExecutor):
    pass


class FakeStudyExecutor(FakePeriodicExecutor):
    pass


def test_three_fake_executors_satisfy_protocol():
    for executor in (
        FakePeriodicExecutor(),
        FakeMolecularExecutor(),
        FakeStudyExecutor(),
    ):
        assert isinstance(executor, VaspWorkflowExecutor)


def test_executor_operations_are_all_awaitable():
    for method_name in (
        "prepare",
        "preflight",
        "submit",
        "status",
        "reconcile",
        "collect",
        "report",
    ):
        assert inspect.iscoroutinefunction(
            getattr(VaspWorkflowExecutor, method_name)
        ), method_name
        assert inspect.iscoroutinefunction(
            getattr(FakePeriodicExecutor, method_name)
        ), method_name


@pytest.mark.asyncio
async def test_async_executor_can_traverse_service_prepare_and_preflight(tmp_path):
    from photomatagent.scientific.applications.vasp.unified.approvals import (
        ApprovalReceiptStore,
    )
    from photomatagent.scientific.applications.vasp.unified.repository import (
        ManifestRepository,
    )
    from photomatagent.scientific.applications.vasp.unified.router import (
        UnifiedVaspRouter,
    )
    from photomatagent.scientific.applications.vasp.unified.service import (
        UnifiedVaspService,
    )
    from photomatagent.scientific.applications.vasp.unified.models import (
        PeriodicScientificSpec,
        UnifiedVaspRequest,
        VaspWorkflowKind,
    )
    from photomatagent.workspace import Workspace

    workspace = Workspace(tmp_path)
    (workspace.root / "structure.cif").write_text("CIF", encoding="utf-8")
    executor = FakePeriodicExecutor()
    service = UnifiedVaspService(
        ManifestRepository(workspace),
        ApprovalReceiptStore(tmp_path),
        UnifiedVaspRouter(periodic=executor),
    )
    manifest = service.plan(
        UnifiedVaspRequest(
            workflow_kind=VaspWorkflowKind.PERIODIC,
            scientific_spec=PeriodicScientificSpec(
                structure_path="structure.cif",
                profile="standard_semiconductor",
            ),
        )
    )

    prepared = await service.prepare(manifest.workflow_id)
    preflighted = await service.preflight(manifest.workflow_id)

    assert prepared.ok
    assert prepared.state is WorkflowState.PREPARED
    assert preflighted.ok
    assert preflighted.state is WorkflowState.PREFLIGHTED


def test_executor_interface_accepts_only_typed_manifest_args():
    expected = {
        "prepare": ["manifest"],
        "preflight": ["manifest"],
        "submit": ["manifest", "stage", "resource"],
        "status": ["manifest"],
        "reconcile": ["manifest"],
        "collect": ["manifest"],
        "report": ["manifest", "request"],
    }
    for method_name, parameter_names in expected.items():
        method = getattr(VaspWorkflowExecutor, method_name)
        signature = inspect.signature(method)
        actual = [
            name
            for name in signature.parameters
            if name != "self" and signature.parameters[name].kind
            not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ]
        assert actual == parameter_names, method_name
        for forbidden in (
            "approval_id",
            "approval_ids",
            "tool",
            "tool_instance",
            "fingerprint",
            "scientific_fingerprint",
            "execution_fingerprint",
            "workflow_dir",
            "raw_slurm",
        ):
            assert forbidden not in actual, method_name


@pytest.mark.parametrize(
    "result_type",
    [
        OperationResult,
        PreflightResult,
        SubmissionResult,
        StatusResult,
        RecoveryResult,
        CollectionResult,
        ReportResult,
        ServiceResult,
    ],
)
def test_typed_results_carry_bounded_structured_fields(result_type):
    # All result models have BaseModel semantics; they must not alias a ToolResult.
    assert hasattr(result_type, "model_dump")
    assert "ok" in result_type.model_fields
    assert "data" in result_type.model_fields


def test_service_result_carries_workflow_state_and_evidence():
    result = ServiceResult(
        ok=True,
        workflow_id="wf-1",
        state=WorkflowState.VALIDATED,
        data={"provenance": {"ok": True}},
    )
    assert result.state is WorkflowState.VALIDATED
    assert result.evidence == []
    assert result.pending_decision is None


def test_report_result_requires_report_kind():
    result = ReportResult(ok=True, report_kind=ReportKind.SUMMARY)
    assert result.report_kind is ReportKind.SUMMARY
