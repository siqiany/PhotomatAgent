"""The model-visible deferred VASP tools backed by UnifiedVaspService."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from photomatagent.scientific.applications.vasp.unified.models import (
    ReportRequest,
    ScientificSpec,
    UnifiedVaspRequest,
    VaspWorkflowKind,
)
from photomatagent.scientific.applications.vasp.unified.service import (
    UnifiedVaspService,
)
from photomatagent.scientific.capabilities.base import CapabilityPack
from photomatagent.scientific.capabilities.contracts import (
    ScientificEvidence,
    ScientificToolResult,
)
from photomatagent.tools.base import Tool
from photomatagent.tools.exposure import ToolExposure


class _PlanArguments(BaseModel):
    workflow_kind: VaspWorkflowKind
    scientific_spec: ScientificSpec


class _SubmitArguments(BaseModel):
    workflow_id: str
    stage: str | None = None


class _ReportArguments(BaseModel):
    workflow_id: str
    report_request: ReportRequest


class _CapabilitiesArguments(BaseModel):
    workflow_kind: VaspWorkflowKind | None = None


class _UnifiedVaspTool(Tool):
    namespace = "vasp"
    source = "photomatagent unified VASP service"
    exposure = ToolExposure.DEFERRED
    tags = ("vasp", "dft", "scnet", "hpc")
    cost_class = "EXPENSIVE"

    def __init__(self, service: UnifiedVaspService) -> None:
        self.service = service


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _error(exc: Exception) -> ScientificToolResult:
    return ScientificToolResult(
        output=_json(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        ),
        is_error=True,
        data={"error_type": type(exc).__name__, "message": str(exc)},
    )


class VaspCapabilitiesTool(_UnifiedVaspTool):
    name = "vasp.capabilities"
    description = (
        "List the unified VASP capability surface for DFT band structure, "
        "DOS, relaxation, optics, SOC, molecules and studies: supported "
        "workflow kinds, periodic profiles, backend/configuration state, and "
        "the vasp.* tools. Read-only; never submits."
    )
    short_description = "Unified VASP capabilities and configuration."
    cost_class = "CHEAP"
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "workflow_kind": {
                "type": "string",
                "enum": ["periodic", "molecular", "study"],
            }
        },
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        try:
            parsed = _CapabilitiesArguments.model_validate(arguments)
        except ValidationError as exc:
            return _error(exc)
        from photomatagent.scientific.applications.vasp.profiles import profiles

        payload = {
            "ok": True,
            "workflow_kinds": [kind.value for kind in VaspWorkflowKind],
            "tools": [
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
            ],
            "periodic_profiles": [
                {
                    "name": profile.name,
                    "stages": profile.stages,
                    "soc": profile.soc,
                }
                for profile in profiles()
            ],
            "filter": parsed.workflow_kind.value if parsed.workflow_kind else None,
            "note": (
                "all VASP tools are DEFERRED; use tool_search/tool_describe "
                "then tool_call"
            ),
        }
        return ScientificToolResult(output=_json(payload), data=payload)


class VaspPlanTool(_UnifiedVaspTool):
    name = "vasp.plan"
    description = (
        "Create a unified VASP workflow manifest for periodic band "
        "structure/DOS/relaxation/optics/SOC, isolated molecules, or studies "
        "from a typed scientific spec. Never accepts raw paths, fingerprints, "
        "or approval IDs."
    )
    short_description = "Create a unified VASP workflow manifest."
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "workflow_kind": {
                "type": "string",
                "enum": ["periodic", "molecular", "study"],
            },
            "scientific_spec": {"type": "object"},
        },
        "required": ["workflow_kind", "scientific_spec"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        try:
            parsed = _PlanArguments.model_validate(arguments)
        except ValidationError as exc:
            return _error(exc)
        request = UnifiedVaspRequest(
            workflow_kind=parsed.workflow_kind,
            scientific_spec=parsed.scientific_spec,
        )
        try:
            manifest = self.service.plan(request)
        except Exception as exc:
            return _error(exc)
        payload = manifest.model_dump(mode="json")
        payload["ok"] = True
        return ScientificToolResult(output=_json(payload), data=payload)


class VaspPrepareTool(_UnifiedVaspTool):
    name = "vasp.prepare"
    description = "Prepare a planned unified VASP workflow without submitting."
    short_description = "Generate VASP inputs for a planned workflow."
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {"workflow_id": {"type": "string"}},
        "required": ["workflow_id"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        workflow_id = str(arguments.get("workflow_id", ""))
        if not workflow_id:
            return _error(ValueError("workflow_id is required"))
        try:
            result = await self.service.prepare(workflow_id)
        except Exception as exc:
            return _error(exc)
        return ScientificToolResult(
            output=_json(result.model_dump(mode="json")),
            data=result.model_dump(mode="json"),
            is_error=not result.ok,
        )


class VaspPreflightTool(_UnifiedVaspTool):
    name = "vasp.preflight"
    description = "Run deterministic scientific preflight for a VASP workflow."
    short_description = "Run deterministic VASP preflight."
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {"workflow_id": {"type": "string"}},
        "required": ["workflow_id"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        workflow_id = str(arguments.get("workflow_id", ""))
        if not workflow_id:
            return _error(ValueError("workflow_id is required"))
        try:
            result = await self.service.preflight(workflow_id)
        except Exception as exc:
            return _error(exc)
        return ScientificToolResult(
            output=_json(result.model_dump(mode="json")),
            data=result.model_dump(mode="json"),
            is_error=not result.ok,
        )


class VaspSubmitTool(_UnifiedVaspTool):
    name = "vasp.submit"
    description = (
        "Submit one prepared/preflighted unified VASP workflow stage through "
        "the idempotent SubmitOnceSession lifecycle. Accepts workflow_id and "
        "optional stage only; never accepts raw Slurm controls, paths, "
        "fingerprints, or approval IDs."
    )
    short_description = "Submit a unified VASP stage (submit-once)."
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "workflow_id": {"type": "string"},
            "stage": {"type": "string"},
        },
        "required": ["workflow_id"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        try:
            parsed = _SubmitArguments.model_validate(arguments)
        except ValidationError as exc:
            return _error(exc)
        try:
            result = await self.service.submit(
                parsed.workflow_id, parsed.stage
            )
        except Exception as exc:
            return _error(exc)
        payload = result.model_dump(mode="json")
        if result.ok and result.state.value in {
            "SUBMITTED",
            "RUNNING",
            "RECONCILING",
        }:
            payload["next_step"] = (
                "job is running: call vasp.wait (internal timer, ONE model "
                "round-trip) instead of polling vasp.status repeatedly; do "
                "not run unrelated tools while the job runs"
            )
        return ScientificToolResult(
            output=_json(payload),
            data=payload,
            is_error=not result.ok,
        )


class VaspStatusTool(_UnifiedVaspTool):
    name = "vasp.status"
    description = "Query unified VASP workflow/scheduler status."
    short_description = "Query VASP workflow status."
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {"workflow_id": {"type": "string"}},
        "required": ["workflow_id"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        workflow_id = str(arguments.get("workflow_id", ""))
        if not workflow_id:
            return _error(ValueError("workflow_id is required"))
        try:
            result = await self.service.status(workflow_id)
        except Exception as exc:
            return _error(exc)
        return ScientificToolResult(
            output=_json(result.model_dump(mode="json")),
            data=result.model_dump(mode="json"),
            is_error=not result.ok,
        )


class VaspResumeTool(_UnifiedVaspTool):
    name = "vasp.resume"
    description = (
        "Reconcile/resume a unified VASP workflow after ambiguity, or reset "
        "a scheduler-confirmed failed workflow back to re-submittable "
        "PREFLIGHTED state (a later vasp.submit creates a fresh attempt)."
    )
    short_description = "Reconcile, resume, or reset a failed VASP workflow."
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {"workflow_id": {"type": "string"}},
        "required": ["workflow_id"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        workflow_id = str(arguments.get("workflow_id", ""))
        if not workflow_id:
            return _error(ValueError("workflow_id is required"))
        try:
            result = await self.service.resume(workflow_id)
        except Exception as exc:
            return _error(exc)
        return ScientificToolResult(
            output=_json(result.model_dump(mode="json")),
            data=result.model_dump(mode="json"),
            is_error=not result.ok,
        )


class VaspWaitTool(_UnifiedVaspTool):
    name = "vasp.wait"
    description = (
        "Wait for a submitted VASP workflow to leave the SUBMITTED/RUNNING "
        "state before asking again. Polls with an internal timer (no model "
        "round-trip per poll) and returns the first non-running state or a "
        "timeout note. Saves tokens compared to repeatedly calling "
        "vasp.status yourself: submit -> vasp.wait -> react to the result. "
        "While waiting, do NOT fill the loop with unrelated tools -- simply "
        "wait."
    )
    short_description = "Wait (with internal timer) until a VASP workflow settles."
    cost_class = "CHEAP"
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "workflow_id": {"type": "string"},
            "timeout_seconds": {
                "type": "integer",
                "minimum": 5,
                "maximum": 1800,
                "description": "Max seconds to wait before returning (default 300).",
            },
            "poll_interval_seconds": {
                "type": "integer",
                "minimum": 5,
                "maximum": 300,
                "description": "Scheduler poll interval (default 30).",
            },
        },
        "required": ["workflow_id"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        workflow_id = str(arguments.get("workflow_id", ""))
        if not workflow_id:
            return _error(ValueError("workflow_id is required"))
        timeout_seconds = int(arguments.get("timeout_seconds", 300))
        poll_interval = int(arguments.get("poll_interval_seconds", 30))
        terminal_states = {
            "SCHEDULER_COMPLETED",
            "VALIDATED",
            "VALIDATION_FAILED",
            "FAILED",
        }
        import time

        deadline = time.monotonic() + timeout_seconds
        waited = 0.0
        last: dict[str, Any] = {}
        try:
            while time.monotonic() < deadline:
                result = await self.service.status(workflow_id)
                last = result.model_dump(mode="json")
                if result.state.value in terminal_states:
                    payload = dict(last)
                    payload["wait"] = {
                        "settled": True,
                        "waited_seconds": round(waited, 1),
                        "state": result.state.value,
                        "note": (
                            "workflow settled; react to this state "
                            "(collect/resume/report as appropriate)"
                        ),
                    }
                    return ScientificToolResult(output=_json(payload), data=payload)
                await asyncio.sleep(poll_interval)
                waited += poll_interval
        except Exception as exc:
            return _error(exc)
        payload = dict(last)
        payload["wait"] = {
            "settled": False,
            "waited_seconds": round(waited, 1),
            "timeout_seconds": timeout_seconds,
            "note": (
                "still SUBMITTED/RUNNING after timeout; call vasp.status "
                "again later or vasp.wait with a longer timeout"
            ),
        }
        return ScientificToolResult(output=_json(payload), data=payload)


class VaspCollectTool(_UnifiedVaspTool):
    name = "vasp.collect"
    description = (
        "Collect and validate VASP outputs. Scientific evidence is produced "
        "only after deterministic validation succeeds; otherwise evidence_gaps "
        "are returned and no evidence is emitted."
    )
    short_description = "Collect, validate, and map VASP evidence."
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {"workflow_id": {"type": "string"}},
        "required": ["workflow_id"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        workflow_id = str(arguments.get("workflow_id", ""))
        if not workflow_id:
            return _error(ValueError("workflow_id is required"))
        try:
            result = await self.service.collect(workflow_id)
        except Exception as exc:
            return _error(exc)
        payload = result.model_dump(mode="json")
        if not result.ok:
            payload["evidence_gaps"] = result.evidence_gaps or payload.get("errors", [])
        return ScientificToolResult(
            output=_json(payload),
            data=payload,
            is_error=not result.ok,
            evidence=result.evidence,
        )


class VaspReportTool(_UnifiedVaspTool):
    name = "vasp.report"
    description = (
        "Generate a typed VASP report: summary, orbitals, ESP, "
        "binding_energy, or study."
    )
    short_description = "Generate a unified VASP report."
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "workflow_id": {"type": "string"},
            "report_request": {"type": "object"},
        },
        "required": ["workflow_id", "report_request"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        try:
            parsed = _ReportArguments.model_validate(arguments)
        except ValidationError as exc:
            return _error(exc)
        try:
            result = await self.service.report(
                parsed.workflow_id, parsed.report_request
            )
        except Exception as exc:
            return _error(exc)
        return ScientificToolResult(
            output=_json(result.model_dump(mode="json")),
            data=result.model_dump(mode="json"),
            is_error=not result.ok,
        )


class VaspUnifiedCapabilityPack(CapabilityPack):
    name = "vasp"
    description = "Unified VASP DFT workflows: plan, prepare, submit, collect, report."
    execution_mode = "mcp/scnet"
    backend_name = "SCNet (SSH + Slurm)"

    def __init__(self, service: UnifiedVaspService) -> None:
        self.service = service

    def probe(self):
        from photomatagent.scientific.capabilities.base import (
            CapabilityStatus,
            ProbeResult,
        )

        return ProbeResult(
            status=CapabilityStatus.AVAILABLE,
            detail="unified VASP service assembled",
        )

    def tools(self) -> list[Tool]:
        return [
            VaspCapabilitiesTool(self.service),
            VaspPlanTool(self.service),
            VaspPrepareTool(self.service),
            VaspPreflightTool(self.service),
            VaspSubmitTool(self.service),
            VaspStatusTool(self.service),
            VaspWaitTool(self.service),
            VaspResumeTool(self.service),
            VaspCollectTool(self.service),
            VaspReportTool(self.service),
        ]
