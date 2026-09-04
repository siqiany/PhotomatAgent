"""Task 11: deferred unified VASP tools and evidence mapping."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from photomatagent.scientific.applications.vasp.unified.executors import (
    ServiceResult,
)
from photomatagent.scientific.applications.vasp.unified.models import (
    ReportKind,
    ReportRequest,
    UnifiedStage,
    UnifiedVaspManifest,
    VaspWorkflowKind,
    WorkflowState,
)
from photomatagent.scientific.applications.vasp.unified.tool_pack import (
    VaspCollectTool,
    VaspPlanTool,
    VaspSubmitTool,
    VaspUnifiedCapabilityPack,
)
from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.tools.exposure import ToolExposure


class FakeService:
    def plan(self, request):
        return UnifiedVaspManifest(
            workflow_id="wf-1",
            workflow_kind=request.workflow_kind,
            scientific_spec=request.scientific_spec,
            scientific_fingerprint="fp",
            stages=[UnifiedStage(name="relax")],
        )

    async def collect(self, workflow_id):
        return ServiceResult(
            ok=True,
            workflow_id=workflow_id,
            state=WorkflowState.VALIDATED,
            evidence=[
                ScientificEvidence(
                    subject="wf-1",
                    property="total_energy",
                    value=-10.0,
                    unit="eV",
                    source="fake",
                    method="dft",
                    fidelity="dft",
                    provenance={"workflow_id": workflow_id},
                )
            ],
        )

    async def prepare(self, workflow_id):
        return ServiceResult(
            ok=True,
            workflow_id=workflow_id,
            state=WorkflowState.PREPARED,
            data={"prepared": True},
        )

    async def preflight(self, workflow_id):
        return ServiceResult(
            ok=True,
            workflow_id=workflow_id,
            state=WorkflowState.PREFLIGHTED,
            data={"passed": True},
        )

    async def submit(self, workflow_id, stage=None):
        return ServiceResult(
            ok=True,
            workflow_id=workflow_id,
            state=WorkflowState.SUBMITTED,
            data={"stage": stage},
        )


def assert_no_forbidden_submit_keys(schema):
    allowed = {"workflow_id", "stage", "type", "properties", "required"}
    for name in ("workflow_dir", "job_name", "input_dir", "profile", "nodes",
                 "tasks_per_node", "walltime_minutes", "approval_ids", "fingerprint"):
        assert name not in schema["properties"]
        assert name not in schema


def test_submit_schema_excludes_forbidden_fields():
    tool = VaspSubmitTool(FakeService())
    assert tool.name == "vasp.submit"
    assert tool.exposure is ToolExposure.DEFERRED
    assert_no_forbidden_submit_keys(tool.input_schema)


def test_nested_pydantic_validation_catches_malformed_scientific_spec():
    from photomatagent.scientific.applications.vasp.unified.tool_pack import (
        _PlanArguments,
    )

    with pytest.raises(ValidationError):
        # Missing discriminated kind inside scientific_spec.
        _PlanArguments.model_validate(
            {
                "workflow_kind": "periodic",
                "scientific_spec": {"profile": "standard_semiconductor"},
            }
        )


@pytest.mark.asyncio
async def test_prepare_and_preflight_tools_await_service_operations():
    from photomatagent.scientific.applications.vasp.unified.tool_pack import (
        VaspPreflightTool,
        VaspPrepareTool,
    )

    service = FakeService()
    prepared = await VaspPrepareTool(service).execute({"workflow_id": "wf-1"})
    preflighted = await VaspPreflightTool(service).execute(
        {"workflow_id": "wf-1"}
    )

    assert not prepared.is_error
    assert prepared.data["state"] == WorkflowState.PREPARED.value
    assert not preflighted.is_error
    assert preflighted.data["state"] == WorkflowState.PREFLIGHTED.value


@pytest.mark.asyncio
async def test_valid_collection_maps_evidence_to_tool_result():
    tool = VaspCollectTool(FakeService())
    result = await tool.execute({"workflow_id": "wf-1"})
    assert not result.is_error
    assert len(result.evidence) == 1
    assert any(isinstance(item, ScientificEvidence) for item in result.state_updates)


@pytest.mark.asyncio
async def test_invalid_collection_returns_evidence_gaps_and_no_evidence():
    class BadCollectService(FakeService):
        async def collect(self, workflow_id):
            return ServiceResult(
                ok=False,
                workflow_id=workflow_id,
                state=WorkflowState.VALIDATION_FAILED,
                errors=["validation failed"],
                evidence_gaps=["validation failed"],
            )

    tool = VaspCollectTool(BadCollectService())
    result = await tool.execute({"workflow_id": "wf-1"})
    assert result.is_error
    assert result.evidence == []
    assert result.data["evidence_gaps"] == ["validation failed"]


def test_capability_pack_returns_exactly_the_documented_surface():
    pack = VaspUnifiedCapabilityPack(FakeService())
    names = [tool.name for tool in pack.tools()]
    assert len(names) == 10
    assert set(names) == {
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
