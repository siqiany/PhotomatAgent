"""Deferred ``vasp.*`` tools exposed to the agent (Sprint 3 section 22).

All tools are DEFERRED and go through ToolRegistry -> tool_search ->
tool_call -> ToolExecutor. The agent never receives generic remote shell
access; it only sees these narrow VASP tools.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from photomatagent.scientific.applications.vasp.application import (
    VaspApplication,
    default_vasp_application,
)
from photomatagent.scientific.applications.vasp.profiles import profiles
from photomatagent.scientific.capabilities.base import (
    CapabilityPack,
    CapabilityStatus,
    ProbeResult,
)
from photomatagent.scientific.capabilities.contracts import (
    ScientificEvidence,
    ScientificToolResult,
)
from photomatagent.scientific.remote.models import (
    HPCJobState,
    RemoteJobRef,
    ResourceRequest,
)
from photomatagent.tools.base import Tool
from photomatagent.tools.exposure import ToolExposure


def _evidence_from_collect(
    report: dict[str, Any], *, tool: str
) -> list[ScientificEvidence]:
    evidence: list[ScientificEvidence] = []
    parsed = report.get("parsed", {})
    if "final_energy_eV" in parsed:
        evidence.append(
            ScientificEvidence(
                subject=f"vasp_job_{report.get('job_id', '?')}",
                property="total_energy",
                value=parsed["final_energy_eV"],
                unit="eV",
                source="SCNet VASP calculation",
                source_type="dft_calculation",
                method="VASP (profile=" + report.get("profile", "?") + ")",
                fidelity="dft",
                summary=(
                    f"VASP total energy {parsed['final_energy_eV']:.4f} eV "
                    f"(job {report.get('job_id', '?')})"
                ),
                limitations="; ".join(report.get("validation_problems", [])),
                provenance={
                    "tool": tool,
                    "job_id": report.get("job_id"),
                    "profile": report.get("profile"),
                    "scheduler_state": report.get("scheduler_state"),
                },
            )
        )
    return evidence


class VaspCapabilitiesTool(Tool):
    name = "vasp.capabilities"
    description = (
        "List VASP capabilities: available profiles (standard_semiconductor, "
        "narrow_gap_soc, optics, namd_preparation), SOC support, backend "
        "connection state, POTCAR policy, and resource limits. Read-only; "
        "never submits. Use for DFT band structure, DOS, relaxation, "
        "optical spectrum, and SOC calculations on SCNet."
    )
    short_description = "VASP profiles, backend state, and limits."
    exposure = ToolExposure.DEFERRED
    namespace = "vasp"
    source = "application metadata"
    tags = ("vasp", "dft", "capabilities", "scnet")
    cost_class = "CHEAP"
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
    }

    def __init__(self, application: VaspApplication | None = None) -> None:
        self.application = application

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        application = self.application or default_vasp_application()
        if application is None:
            return ScientificToolResult(
                output=(
                    "VASP is UNCONFIGURED: no SCNet backend configured "
                    "(set SUPERCOMPUTING_HOST/USERNAME or SCNET_HOST/USERNAME)"
                ),
                is_error=True,
                data={
                    "error_type": "missing_prerequisites",
                    "missing": ["SCNET_HOST", "SCNET_USERNAME"],
                },
            )
        payload = await application.probe_environment_async()
        payload["profiles"] = [
            {
                "name": profile.name,
                "description": profile.description,
                "soc": profile.soc,
                "stages": profile.stages,
                "needs_configuration": profile.needs_configuration,
                "limitations": profile.limitations,
            }
            for profile in profiles()
        ]
        payload["cost_class"] = "EXPENSIVE"
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False, indent=2),
            data=payload,
        )


class VaspPrepareTool(Tool):
    name = "vasp.prepare"
    description = (
        "Prepare VASP inputs for a profile (standard_semiconductor | "
        "narrow_gap_soc | optics | namd_preparation) from a structure file "
        "(CIF/POSCAR). Generates the full stage workflow locally "
        "(POSCAR/INCAR/KPOINTS + POTCAR.policy) and writes workflow.json. "
        "NEVER submits. POTCAR is resolved at submit time from "
        "PMG_VASP_PSP_DIR or a remote pseudopotential location. Covers DFT "
        "band structure, DOS, geometry relaxation, and optics stages."
    )
    short_description = "Generate VASP inputs for a profile (no submit)."
    exposure = ToolExposure.DEFERRED
    namespace = "vasp"
    source = "photomatagent VASP input generator"
    tags = ("vasp", "dft", "input generation", "incar")
    cost_class = "CHEAP"
    input_schema = {
        "type": "object",
        "properties": {
            "structure_path": {"type": "string"},
            "profile": {
                "type": "string",
                "enum": [
                    "standard_semiconductor",
                    "narrow_gap_soc",
                    "optics",
                    "namd_preparation",
                ],
            },
            "output_dir": {"type": "string"},
            "encut_ev": {"type": "number", "minimum": 200, "maximum": 1000},
            "kpoint_density": {"type": "number", "minimum": 100, "maximum": 20000},
            "kpoint_grid": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1},
                "minItems": 3,
                "maxItems": 3,
            },
        },
        "required": ["structure_path", "profile"],
    }

    def __init__(self, application: VaspApplication | None = None) -> None:
        self.application = application

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        application = self.application or default_vasp_application()
        if application is None:
            return _unconfigured()
        overrides = {
            key: arguments[key]
            for key in ("encut_ev", "kpoint_density", "kpoint_grid")
            if arguments.get(key) is not None
        }
        try:
            manifest = application.prepare_inputs(
                structure_path=str(arguments["structure_path"]),
                profile_name=str(arguments["profile"]),
                output_dir=(
                    str(arguments["output_dir"])
                    if arguments.get("output_dir")
                    else None
                ),
                spec_overrides=overrides,
            )
        except Exception as exc:
            return ScientificToolResult(
                output=f"vasp.prepare failed: {type(exc).__name__}: {exc}",
                is_error=True,
                data={"error_type": type(exc).__name__, "message": str(exc)},
            )
        manifest["submitted"] = False
        manifest["cost_class"] = "EXPENSIVE"
        return ScientificToolResult(
            output=json.dumps(manifest, ensure_ascii=False, indent=2),
            data=manifest,
        )


class VaspSubmitTool(Tool):
    name = "vasp.submit"
    description = (
        "Submit one prepared VASP stage (directory containing "
        "POSCAR/INCAR/KPOINTS) to SCNet via Slurm. Requires "
        "PHOTOMATAGENT_ALLOW_HPC_SUBMIT=1 and passes the deterministic "
        "resource policy. Returns a detached RemoteJobRef; poll with "
        "vasp.status."
    )
    short_description = "Submit a VASP stage to SCNet (detached job)."
    exposure = ToolExposure.DEFERRED
    namespace = "vasp"
    source = "SCNet VASP application"
    tags = ("vasp", "dft", "scnet", "slurm", "hpc")
    cost_class = "EXPENSIVE"
    input_schema = {
        "type": "object",
        "properties": {
            "job_name": {"type": "string"},
            "input_dir": {"type": "string"},
            "profile": {"type": "string"},
            "partition": {"type": "string"},
            "nodes": {"type": "integer", "minimum": 1},
            "tasks_per_node": {"type": "integer", "minimum": 1},
            "walltime_minutes": {"type": "integer", "minimum": 1},
        },
        "required": ["job_name", "input_dir", "profile"],
    }

    def __init__(self, application: VaspApplication | None = None) -> None:
        self.application = application

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        application = self.application or default_vasp_application()
        if application is None:
            return _unconfigured()
        resource = ResourceRequest(
            partition=str(arguments.get("partition", "kshcnormal")),
            nodes=int(arguments.get("nodes", 1)),
            tasks_per_node=int(arguments.get("tasks_per_node", 32)),
            walltime_minutes=int(arguments.get("walltime_minutes", 240)),
        )
        try:
            ref = await application.submit_stage(
                job_name=str(arguments["job_name"]),
                input_dir=str(arguments["input_dir"]),
                profile_name=str(arguments["profile"]),
                resource=resource,
            )
        except Exception as exc:
            return ScientificToolResult(
                output=f"vasp.submit refused: {type(exc).__name__}: {exc}",
                is_error=True,
                data={
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "hint": (
                        "HPC submission requires PHOTOMATAGENT_ALLOW_HPC_SUBMIT=1 "
                        "and a passing resource policy"
                    ),
                },
            )
        payload = ref.model_dump()
        payload["cost_class"] = "EXPENSIVE"
        payload["note"] = (
            "detached job: poll with vasp.status; Slurm COMPLETED is not "
            "scientific validity -- collect with vasp.collect"
        )
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False, indent=2),
            data=payload,
        )


class VaspStatusTool(Tool):
    name = "vasp.status"
    description = (
        "Query the Slurm state of a VASP job id (PENDING/RUNNING/COMPLETED/"
        "FAILED/...). Scheduler state only: COMPLETED does not imply a "
        "scientifically valid calculation; use vasp.collect to validate "
        "vasprun.xml."
    )
    short_description = "Poll Slurm state of a VASP job."
    exposure = ToolExposure.DEFERRED
    namespace = "vasp"
    source = "SCNet scheduler"
    tags = ("vasp", "slurm", "status", "hpc")
    cost_class = "CHEAP"
    input_schema = {
        "type": "object",
        "properties": {"job_id": {"type": "string"}},
        "required": ["job_id"],
    }

    def __init__(self, application: VaspApplication | None = None) -> None:
        self.application = application

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        application = self.application or default_vasp_application()
        if application is None:
            return _unconfigured()
        try:
            state = await application.status(str(arguments["job_id"]))
        except Exception as exc:
            return ScientificToolResult(
                output=f"vasp.status failed: {type(exc).__name__}: {exc}",
                is_error=True,
                data={"error_type": type(exc).__name__},
            )
        payload = {
            "job_id": arguments["job_id"],
            "state": state.value,
            "terminal": state.terminal,
            "note": (
                "scheduler state only; scientific validity requires "
                "vasp.collect validation"
            ),
        }
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False),
            data=payload,
        )


class VaspCollectTool(Tool):
    name = "vasp.collect"
    description = (
        "Download a finished VASP job's results (OUTCAR/CONTCAR/CHGCAR/"
        "vasprun.xml/OSZICAR), validate the vasprun.xml contract "
        "(well-formed, SCF convergence markers), and parse bounded values "
        "(total energy, dielectric summary) into ScientificEvidence. "
        "Returns validation problems explicitly; nothing is claimed valid "
        "without an empty problems list."
    )
    short_description = "Download, validate, and parse a VASP job result."
    exposure = ToolExposure.DEFERRED
    namespace = "vasp"
    source = "SCNet VASP application"
    tags = ("vasp", "dft", "validation", "results")
    cost_class = "CHEAP"
    input_schema = {
        "type": "object",
        "properties": {
            "job_id": {"type": "string"},
            "remote_directory": {"type": "string"},
            "local_dir": {"type": "string"},
            "profile": {"type": "string"},
        },
        "required": ["job_id", "remote_directory", "profile"],
    }

    def __init__(self, application: VaspApplication | None = None) -> None:
        self.application = application

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        application = self.application or default_vasp_application()
        if application is None:
            return _unconfigured()
        job_ref = RemoteJobRef(
            backend="scnet",
            application="vasp",
            job_id=str(arguments["job_id"]),
            remote_directory=str(arguments["remote_directory"]),
        )
        local_dir = str(arguments.get("local_dir") or "output/vasp_results")
        try:
            report = await application.collect(
                job_ref=job_ref,
                local_dir=local_dir,
                profile_name=str(arguments["profile"]),
            )
        except Exception as exc:
            return ScientificToolResult(
                output=f"vasp.collect failed: {type(exc).__name__}: {exc}",
                is_error=True,
                data={"error_type": type(exc).__name__, "message": str(exc)},
            )
        evidence = _evidence_from_collect(report, tool=self.name)
        return ScientificToolResult(
            output=json.dumps(report, ensure_ascii=False, indent=2),
            data=report,
            evidence=evidence,
            artifacts=[str(path) for path in report.get("artifacts", [])],
        )


class VaspInspectResultTool(Tool):
    name = "vasp.inspect_result"
    description = (
        "Validate and parse a local VASP result directory (vasprun.xml + "
        "OUTCAR) without any remote contact: convergence markers, total "
        "energy, dielectric summary. Use on collected artifacts."
    )
    short_description = "Parse a local VASP result directory."
    exposure = ToolExposure.DEFERRED
    namespace = "vasp"
    source = "photomatagent vasprun parser"
    tags = ("vasp", "dft", "vasprun", "parse")
    cost_class = "CHEAP"
    input_schema = {
        "type": "object",
        "properties": {
            "result_dir": {"type": "string"},
            "profile": {"type": "string"},
        },
        "required": ["result_dir", "profile"],
    }

    def __init__(self, application: VaspApplication | None = None) -> None:
        self.application = application

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        application = self.application or default_vasp_application()
        if application is None:
            return _unconfigured()
        directory = str(arguments["result_dir"])
        profile = str(arguments["profile"])
        problems = application.validate_output(directory, profile_name=profile)
        parsed = application.parse_result(directory)
        payload = {
            "result_dir": directory,
            "profile": profile,
            "validation_problems": problems,
            "scientifically_valid": not problems,
            "parsed": parsed,
            "note": (
                "local inspection only; no scheduler state involved"
            ),
        }
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False, indent=2),
            data=payload,
            evidence=_evidence_from_collect(
                {
                    "job_id": "local",
                    "profile": profile,
                    "scheduler_state": "LOCAL",
                    "validation_problems": problems,
                    "parsed": parsed,
                },
                tool=self.name,
            ),
        )


class VaspRunWorkflowTool(Tool):
    name = "vasp.run_workflow"
    description = (
        "Bounded convenience API: submit every prepared stage sequentially, "
        "wait for each Slurm job (timeout-bounded), collect and validate "
        "results. Only for small, fully-authorized smoke runs; production "
        "use should prepare, submit, poll, and collect detached jobs. "
        "Requires PHOTOMATAGENT_ALLOW_HPC_SUBMIT=1."
    )
    short_description = "Run a full prepared VASP workflow (bounded)."
    exposure = ToolExposure.DEFERRED
    namespace = "vasp"
    source = "SCNet VASP application"
    tags = ("vasp", "dft", "workflow", "hpc")
    cost_class = "VERY_EXPENSIVE"
    input_schema = {
        "type": "object",
        "properties": {
            "workflow_dir": {"type": "string"},
            "profile": {"type": "string"},
            "poll_interval_seconds": {"type": "number", "minimum": 1},
            "timeout_seconds": {"type": "number", "minimum": 60},
        },
        "required": ["workflow_dir", "profile"],
    }

    def __init__(self, application: VaspApplication | None = None) -> None:
        self.application = application

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        application = self.application or default_vasp_application()
        if application is None:
            return _unconfigured()
        try:
            report = await application.submit_workflow(
                workflow_dir=str(arguments["workflow_dir"]),
                profile_name=str(arguments["profile"]),
            )
        except Exception as exc:
            return ScientificToolResult(
                output=f"vasp.run_workflow failed: {type(exc).__name__}: {exc}",
                is_error=True,
                data={"error_type": type(exc).__name__, "message": str(exc)},
            )
        report["cost_class"] = "VERY_EXPENSIVE"
        return ScientificToolResult(
            output=json.dumps(report, ensure_ascii=False, indent=2),
            data=report,
        )


def _unconfigured() -> ScientificToolResult:
    return ScientificToolResult(
        output=(
            "VASP is UNCONFIGURED: no SCNet backend configured "
            "(set SUPERCOMPUTING_HOST/USERNAME or SCNET_HOST/USERNAME)"
        ),
        is_error=True,
        data={
            "error_type": "missing_prerequisites",
            "missing": ["SCNET_HOST", "SCNET_USERNAME"],
            "hint": "configure the backend, then run vasp.capabilities",
        },
    )


class VaspCapabilityPack(CapabilityPack):
    name = "vasp"
    description = "VASP DFT on SCNet (profiles, submit, collect, validate)."
    execution_mode = "mcp/scnet"
    backend_name = "SCNet (SSH + Slurm)"

    def __init__(
        self, application: VaspApplication | None = None, workspace: Any = None
    ) -> None:
        self.application = application or default_vasp_application()

    def probe(self) -> ProbeResult:
        if self.application is None:
            return ProbeResult(
                status=CapabilityStatus.UNCONFIGURED,
                detail="no SCNet backend configured (SCNET_HOST/USERNAME)",
            )
        report = self.application.probe_environment()
        connected = report.get("connection", {}).get("connected") == "true"
        if connected:
            return ProbeResult(
                status=CapabilityStatus.AVAILABLE,
                detail="SCNet reachable; profiles ready",
                version="vasp-6.x",
            )
        return ProbeResult(
            status=CapabilityStatus.MISSING_DEPENDENCY,
            detail=f"backend configured but unreachable: {report}",
        )

    def tools(self) -> list[Tool]:
        return [
            VaspCapabilitiesTool(self.application),
            VaspPrepareTool(self.application),
            VaspSubmitTool(self.application),
            VaspStatusTool(self.application),
            VaspCollectTool(self.application),
            VaspInspectResultTool(self.application),
            VaspRunWorkflowTool(self.application),
        ]


def vasp_pack(workspace: Any = None) -> VaspCapabilityPack:
    return VaspCapabilityPack(workspace=workspace)
