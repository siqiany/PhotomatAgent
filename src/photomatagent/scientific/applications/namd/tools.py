"""Deferred ``namd.*`` tools (Sprint 3 section 35)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from photomatagent.scientific.applications.namd.application import (
    NamdApplication,
)
from photomatagent.scientific.capabilities.base import (
    CapabilityPack,
    CapabilityStatus,
    ProbeResult,
)
from photomatagent.scientific.capabilities.contracts import ScientificToolResult
from photomatagent.scientific.remote.models import ResourceRequest
from photomatagent.tools.base import Tool
from photomatagent.tools.exposure import ToolExposure


def _unconfigured() -> ScientificToolResult:
    return ScientificToolResult(
        output=(
            "Hefei-NAMD is UNCONFIGURED: no SCNet backend and/or module "
            "name configured; run namd.capabilities"
        ),
        is_error=True,
        data={
            "error_type": "missing_prerequisites",
            "missing": ["SCNET_HOST", "SCNET_USERNAME", "namd module"],
        },
    )


class NamdCapabilitiesTool(Tool):
    name = "namd.capabilities"
    description = (
        "Probe the SCNet Hefei-NAMD environment: module availability, "
        "supported workflow (VASP AIMD trajectory + per-snapshot WAVECARs), "
        "required VASP artifacts, and evidence scope. Never produces "
        "carrier-dynamics numbers without real NAMD output."
    )
    short_description = "Hefei-NAMD environment probe and requirements."
    exposure = ToolExposure.DEFERRED
    namespace = "namd"
    source = "SCNet environment probe"
    tags = ("namd", "hefei-namd", "carrier dynamics", "scnet")
    cost_class = "CHEAP"
    input_schema: dict[str, Any] = {"type": "object", "properties": {}}

    def __init__(self, application: NamdApplication | None = None) -> None:
        self.application = application

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        application = self.application or NamdApplication()
        payload = await application.probe_environment_async()
        payload["cost_class"] = "VERY_EXPENSIVE"
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False, indent=2),
            data=payload,
        )


class NamdValidateInputsTool(Tool):
    name = "namd.validate_inputs"
    description = (
        "Validate a VASP AIMD trajectory tree for Hefei-NAMD: reference "
        "POSCAR + XDATCAR + OUTCAR, per-snapshot directories with "
        "POSCAR/WAVECAR/OUTCAR, and identical WAVECAR sizes across all "
        "snapshots."
    )
    short_description = "Validate VASP trajectory inputs for Hefei-NAMD."
    exposure = ToolExposure.DEFERRED
    namespace = "namd"
    source = "photomatagent NAMD input contract"
    tags = ("namd", "validation", "trajectory", "wavecar")
    cost_class = "CHEAP"
    input_schema = {
        "type": "object",
        "properties": {"trajectory_dir": {"type": "string"}},
        "required": ["trajectory_dir"],
    }

    def __init__(self, application: NamdApplication | None = None) -> None:
        self.application = application

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        application = self.application or NamdApplication()
        problems = application.validate_inputs(str(arguments["trajectory_dir"]))
        payload = {
            "trajectory_dir": arguments["trajectory_dir"],
            "problems": problems,
            "valid": not problems,
        }
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False, indent=2),
            is_error=bool(problems),
            data=payload,
        )


class NamdPrepareTool(Tool):
    name = "namd.prepare"
    description = (
        "Prepare the Hefei-NAMD job tree from a validated VASP AIMD "
        "trajectory: manifest with reference POSCAR, XDATCAR, OUTCAR, and "
        "per-snapshot WAVECAR references. Runtime inputs (inp/INICON) are "
        "NOT fabricated: they are generated only after the SCNet module has "
        "been confirmed."
    )
    short_description = "Prepare the Hefei-NAMD input tree (no submit)."
    exposure = ToolExposure.DEFERRED
    namespace = "namd"
    source = "photomatagent NAMD preparation"
    tags = ("namd", "prepare", "trajectory", "input")
    cost_class = "CHEAP"
    input_schema = {
        "type": "object",
        "properties": {
            "trajectory_dir": {"type": "string"},
            "output_dir": {"type": "string"},
        },
        "required": ["trajectory_dir", "output_dir"],
    }

    def __init__(self, application: NamdApplication | None = None) -> None:
        self.application = application

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        application = self.application or NamdApplication()
        try:
            manifest = application.prepare(
                trajectory_dir=str(arguments["trajectory_dir"]),
                output_dir=str(arguments["output_dir"]),
            )
        except Exception as exc:
            return ScientificToolResult(
                output=f"namd.prepare failed: {type(exc).__name__}: {exc}",
                is_error=True,
                data={"error_type": type(exc).__name__, "message": str(exc)},
            )
        manifest["submitted"] = False
        return ScientificToolResult(
            output=json.dumps(manifest, ensure_ascii=False, indent=2),
            data=manifest,
        )


class NamdSubmitTool(Tool):
    name = "namd.submit"
    description = (
        "Submit a prepared Hefei-NAMD job to SCNet (requires the confirmed "
        "module name, PHOTOMATAGENT_ALLOW_HPC_SUBMIT=1, and a passing "
        "resource policy). Returns a detached job ref."
    )
    short_description = "Submit a Hefei-NAMD job (detached)."
    exposure = ToolExposure.DEFERRED
    namespace = "namd"
    source = "SCNet Hefei-NAMD application"
    tags = ("namd", "hpc", "scnet", "slurm")
    cost_class = "VERY_EXPENSIVE"
    input_schema = {
        "type": "object",
        "properties": {
            "job_name": {"type": "string"},
            "prepared_dir": {"type": "string"},
        },
        "required": ["job_name", "prepared_dir"],
    }

    def __init__(self, application: NamdApplication | None = None) -> None:
        self.application = application

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        application = self.application or NamdApplication()
        try:
            ref = await application.submit(
                job_name=str(arguments["job_name"]),
                prepared_dir=str(arguments["prepared_dir"]),
            )
        except Exception as exc:
            return ScientificToolResult(
                output=f"namd.submit refused: {type(exc).__name__}: {exc}",
                is_error=True,
                data={"error_type": type(exc).__name__, "message": str(exc)},
            )
        payload = ref.model_dump()
        payload["note"] = "detached job; poll with namd.status"
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False, indent=2),
            data=payload,
        )


class NamdStatusTool(Tool):
    name = "namd.status"
    description = "Query the Slurm state of a Hefei-NAMD job id."
    short_description = "Poll Slurm state of a Hefei-NAMD job."
    exposure = ToolExposure.DEFERRED
    namespace = "namd"
    source = "SCNet scheduler"
    tags = ("namd", "slurm", "status")
    cost_class = "CHEAP"
    input_schema = {
        "type": "object",
        "properties": {"job_id": {"type": "string"}},
        "required": ["job_id"],
    }

    def __init__(self, application: NamdApplication | None = None) -> None:
        self.application = application

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        application = self.application or NamdApplication()
        try:
            state = await application.status(str(arguments["job_id"]))
        except Exception as exc:
            return ScientificToolResult(
                output=f"namd.status failed: {type(exc).__name__}: {exc}",
                is_error=True,
            )
        payload = {
            "job_id": arguments["job_id"],
            "state": state.value,
            "terminal": state.terminal,
            "note": "scheduler state only; scientific evidence requires output files",
        }
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False),
            data=payload,
        )


class NamdCollectTool(Tool):
    name = "namd.collect"
    description = (
        "Download Hefei-NAMD output files. Evidence (population dynamics, "
        "lifetimes) is produced ONLY from files that actually exist; with "
        "no output files, no carrier-dynamics evidence is returned."
    )
    short_description = "Collect Hefei-NAMD outputs."
    exposure = ToolExposure.DEFERRED
    namespace = "namd"
    source = "SCNet Hefei-NAMD application"
    tags = ("namd", "collect", "results")
    cost_class = "CHEAP"
    input_schema = {
        "type": "object",
        "properties": {
            "job_id": {"type": "string"},
            "remote_directory": {"type": "string"},
            "local_dir": {"type": "string"},
        },
        "required": ["job_id", "remote_directory"],
    }

    def __init__(self, application: NamdApplication | None = None) -> None:
        self.application = application

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        from photomatagent.scientific.remote.models import RemoteJobRef

        application = self.application or NamdApplication()
        ref = RemoteJobRef(
            backend="scnet",
            application="hefei-namd",
            job_id=str(arguments["job_id"]),
            remote_directory=str(arguments["remote_directory"]),
        )
        try:
            report = await application.collect(
                job_ref=ref,
                local_dir=str(arguments.get("local_dir") or "output/namd"),
            )
        except Exception as exc:
            return ScientificToolResult(
                output=f"namd.collect failed: {type(exc).__name__}: {exc}",
                is_error=True,
            )
        return ScientificToolResult(
            output=json.dumps(report, ensure_ascii=False, indent=2),
            data=report,
        )


class NamdInspectResultTool(Tool):
    name = "namd.inspect_result"
    description = (
        "Inspect a local Hefei-NAMD result directory: list available output "
        "files. Does NOT interpret numbers that are not present."
    )
    short_description = "List local Hefei-NAMD result files."
    exposure = ToolExposure.DEFERRED
    namespace = "namd"
    source = "photomatagent NAMD result inspection"
    tags = ("namd", "inspect", "results")
    cost_class = "CHEAP"
    input_schema = {
        "type": "object",
        "properties": {"result_dir": {"type": "string"}},
        "required": ["result_dir"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        root = Path(str(arguments["result_dir"])).expanduser().resolve()
        files = (
            [
                {"name": path.name, "size_bytes": path.stat().st_size}
                for path in sorted(root.rglob("*"))
                if path.is_file()
            ]
            if root.is_dir()
            else []
        )
        payload = {
            "result_dir": str(root),
            "files": files[:100],
            "file_count": len(files),
            "interpretation": (
                "no population-dynamics/lifetime numbers are derived here; "
                "use the NAMD analysis workflow on real outputs"
            ),
        }
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False, indent=2),
            data=payload,
        )


class NamdCapabilityPack(CapabilityPack):
    name = "namd"
    description = "Hefei-NAMD carrier dynamics on SCNet (probe + prepare)."
    execution_mode = "mcp/scnet"
    backend_name = "SCNet (SSH + Slurm)"

    def __init__(self, application: NamdApplication | None = None) -> None:
        self.application = application or NamdApplication()

    def probe(self) -> ProbeResult:
        report = self.application.probe_environment()
        if report.get("status") == "AVAILABLE":
            return ProbeResult(
                status=CapabilityStatus.AVAILABLE,
                detail=report.get("detail", "module found"),
                version="hefei-namd",
            )
        return ProbeResult(
            status=CapabilityStatus.UNCONFIGURED,
            detail=report.get("detail", "module not confirmed"),
        )

    def tools(self) -> list[Tool]:
        return [
            NamdCapabilitiesTool(self.application),
            NamdValidateInputsTool(self.application),
            NamdPrepareTool(self.application),
            NamdSubmitTool(self.application),
            NamdStatusTool(self.application),
            NamdCollectTool(self.application),
            NamdInspectResultTool(),
        ]


def namd_pack(workspace: Any = None) -> NamdCapabilityPack:
    return NamdCapabilityPack()
