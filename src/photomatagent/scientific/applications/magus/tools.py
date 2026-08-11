"""Deferred ``magus.*`` tools (Sprint 4 lifecycle surface).

Native pack and SCNet MCP server expose the same application layer, so a
tool behaves identically whether reached via ToolRegistry or MCP.

Error contract (Sprint 4 section 71): UNCONFIGURED / MISSING_DEPENDENCY /
MISSING_PREREQUISITE / MISSING_PSEUDOPOTENTIALS / SUBMISSION_BLOCKED /
EXECUTION_FAILED. There is never an LLM fallback guess.
"""

from __future__ import annotations

import json
from typing import Any

from photomatagent.scientific.applications.magus.application import (
    MagusApplication,
    MagusDependencyError,
    MagusExecutionError,
    MagusPrerequisiteError,
    MagusPseudopotentialMissingError,
    MagusSubmissionBlockedError,
    MagusUnconfiguredError,
    default_magus_application,
)
from photomatagent.scientific.applications.magus.models import (
    MagusGenerateRequest,
    MagusSearchRequest,
    SUPPORTED_STRUCTURE_TYPES,
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


def _error(error_type: str, message: str) -> ScientificToolResult:
    return ScientificToolResult(
        output=message,
        is_error=True,
        data={"error_type": error_type, "message": message},
    )


def _result(payload: dict[str, Any]) -> ScientificToolResult:
    return ScientificToolResult(
        output=json.dumps(payload, ensure_ascii=False, indent=2),
        data=payload,
    )


class MagusCapabilitiesTool(Tool):
    name = "magus.capabilities"
    description = (
        "Probe the remote SCNet MAGUS installation: root, actual executable, "
        "version, CLI commands, calculators (vasp/emt/lj/...), supported "
        "structure types, JOB_SYSTEM and VASP/pseudopotential readiness. "
        "Read-only; never submits. MAGUS searches atomic structures; it does "
        "NOT directly predict responsivity/EQE/detectivity/dark current -- "
        "its output is candidate generation/search evidence."
    )
    short_description = "Remote MAGUS capabilities (read-only probe)."
    exposure = ToolExposure.DEFERRED
    namespace = "magus"
    source = "SCNet remote probe"
    tags = ("magus", "structure search", "capabilities", "scnet")
    cost_class = "CHEAP"
    input_schema: dict[str, Any] = {"type": "object", "properties": {}}

    def __init__(self, application: MagusApplication | None = None) -> None:
        self.application = application

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        application = self.application or default_magus_application()
        if application is None:
            return _error(
                "UNCONFIGURED",
                "MAGUS is UNCONFIGURED: set SCNET_HOST / SCNET_USERNAME / "
                "SCNET_PRIVATE_KEY_PATH (and SCNET_MAGUS_ROOT)",
            )
        payload = await application.probe_environment_async()
        return _result(payload)


class MagusProbeTool(MagusCapabilitiesTool):
    """Deprecated alias of magus.capabilities (kept for regression)."""

    name = "magus.probe"
    description = (
        "Deprecated alias of magus.capabilities: same read-only remote probe."
    )


def _prepare_generate_tool() -> type[Any]:
    class _MagusPrepareGenerate(Tool):
        name = "magus.prepare_generate"
        description = (
            "Prepare a MAGUS atomic-structure candidate generation job "
            "(magus generate, no expensive property evaluation) for a "
            "composition. Writes input.yaml + magus.slurm + "
            "photomat_manifest.json into job_dir; NEVER submits. Use for "
            "atomic structure candidate generation without property "
            "evaluation; generated structures are NOT energy validated."
        )
        short_description = "Prepare a MAGUS structure-generation job."
        exposure = ToolExposure.DEFERRED
        namespace = "magus"
        source = "MAGUS deterministic renderer"
        tags = ("magus", "structure search", "generate")
        cost_class = "EXPENSIVE"
        input_schema = {
            "type": "object",
            "properties": {
                "composition": {"type": "string"},
                "structure_type": {
                    "type": "string",
                    "enum": ["bulk", "cluster", "surface"],
                },
                "number": {"type": "integer", "minimum": 1, "maximum": 100},
                "min_atoms": {"type": "integer", "minimum": 1},
                "max_atoms": {"type": "integer", "minimum": 1},
                "job_dir": {"type": "string"},
            },
            "required": ["composition", "job_dir"],
        }

        def __init__(self, application: MagusApplication | None = None) -> None:
            self.application = application

        async def execute(
            self, arguments: dict[str, Any]
        ) -> ScientificToolResult:
            application = self.application or default_magus_application()
            if application is None:
                return _error(
                    "UNCONFIGURED",
                    "MAGUS is UNCONFIGURED: no SCNet backend configured",
                )
            try:
                request = MagusGenerateRequest.from_composition(
                    str(arguments["composition"]),
                    structure_type=arguments.get("structure_type", "bulk"),
                    number=int(arguments.get("number", 5)),
                    min_atoms=(
                        int(arguments["min_atoms"])
                        if arguments.get("min_atoms") is not None
                        else None
                    ),
                    max_atoms=(
                        int(arguments["max_atoms"])
                        if arguments.get("max_atoms") is not None
                        else None
                    ),
                )
                manifest = application.prepare_generate(
                    request, str(arguments["job_dir"])
                )
            except Exception as exc:
                return _error(
                    "INVALID_REQUEST",
                    f"magus.prepare_generate failed: {type(exc).__name__}: {exc}",
                )
            return _result(manifest)

    return _MagusPrepareGenerate


def _prepare_search_tool() -> type[Any]:
    class _MagusPrepareSearch(Tool):
        name = "magus.prepare_search"
        description = (
            "Prepare a constrained MAGUS structure-search job (magus search; "
            "an explicit calculator is required, e.g. vasp/emt/lj). Writes "
            "input.yaml + inputFold/VASP/INCAR (VASP) + magus.slurm + "
            "photomat_manifest.json; NEVER submits. Runs serially inside a "
            "Slurm allocation (nested queue submission is not supported). "
            "VERY_EXPENSIVE when calculator=vasp."
        )
        short_description = "Prepare a MAGUS constrained structure search."
        exposure = ToolExposure.DEFERRED
        namespace = "magus"
        source = "MAGUS deterministic renderer"
        tags = ("magus", "structure search", "search")
        cost_class = "VERY_EXPENSIVE"
        input_schema = {
            "type": "object",
            "properties": {
                "composition": {"type": "string"},
                "structure_type": {
                    "type": "string",
                    "enum": ["bulk", "cluster", "surface"],
                },
                "calculator": {"type": "string"},
                "init_size": {"type": "integer", "minimum": 1, "maximum": 100},
                "population_size": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                },
                "generations": {"type": "integer", "minimum": 1, "maximum": 50},
                "save_good": {"type": "integer", "minimum": 1, "maximum": 100},
                "pressure_gpa": {"type": "number", "minimum": 0},
                "min_atoms": {"type": "integer", "minimum": 1},
                "max_atoms": {"type": "integer", "minimum": 1},
                "job_dir": {"type": "string"},
                "slab": {"type": "object"},
            },
            "required": ["composition", "job_dir"],
        }

        def __init__(self, application: MagusApplication | None = None) -> None:
            self.application = application

        async def execute(
            self, arguments: dict[str, Any]
        ) -> ScientificToolResult:
            application = self.application or default_magus_application()
            if application is None:
                return _error(
                    "UNCONFIGURED",
                    "MAGUS is UNCONFIGURED: no SCNet backend configured",
                )
            try:
                request = MagusSearchRequest.from_composition(
                    str(arguments["composition"]),
                    structure_type=arguments.get("structure_type", "bulk"),
                    calculator=arguments.get("calculator", "vasp"),
                    init_size=int(arguments.get("init_size", 4)),
                    population_size=int(arguments.get("population_size", 4)),
                    generations=int(arguments.get("generations", 1)),
                    save_good=int(arguments.get("save_good", 2)),
                    pressure_gpa=float(arguments.get("pressure_gpa", 0.0)),
                    min_atoms=(
                        int(arguments["min_atoms"])
                        if arguments.get("min_atoms") is not None
                        else None
                    ),
                    max_atoms=(
                        int(arguments["max_atoms"])
                        if arguments.get("max_atoms") is not None
                        else None
                    ),
                )
                manifest = application.prepare_search(
                    request, str(arguments["job_dir"])
                )
            except Exception as exc:
                return _error(
                    "INVALID_REQUEST",
                    f"magus.prepare_search failed: {type(exc).__name__}: {exc}",
                )
            return _result(manifest)

    return _MagusPrepareSearch


class _LifecycleTool(Tool):
    """Shared plumbing for submit/status/collect/inspect tools."""

    exposure = ToolExposure.DEFERRED
    namespace = "magus"
    source = "SCNet MAGUS application"

    def __init__(self, application: MagusApplication | None = None) -> None:
        self.application = application

    def _application(self) -> MagusApplication | None:
        return self.application or default_magus_application()

    def _error_for(self, exc: Exception) -> ScientificToolResult:
        if isinstance(exc, MagusUnconfiguredError):
            return _error("UNCONFIGURED", str(exc))
        if isinstance(exc, MagusDependencyError):
            return _error("MISSING_DEPENDENCY", str(exc))
        if isinstance(exc, MagusPseudopotentialMissingError):
            return _error("MISSING_PSEUDOPOTENTIALS", str(exc))
        if isinstance(exc, MagusPrerequisiteError):
            return _error("MISSING_PREREQUISITE", str(exc))
        if isinstance(exc, MagusSubmissionBlockedError):
            return _error("SUBMISSION_BLOCKED", str(exc))
        return _error("EXECUTION_FAILED", f"{type(exc).__name__}: {exc}")


class MagusSubmitTool(_LifecycleTool):
    name = "magus.submit"
    description = (
        "Submit a prepared MAGUS job tree (from magus.prepare_generate / "
        "magus.prepare_search) to SCNet via Slurm. Requires "
        "PHOTOMATAGENT_ALLOW_HPC_SUBMIT=1 and a passing resource policy; "
        "VASP searches additionally require the POTCAR setups present "
        "remotely. Returns a detached job ref; poll with magus.status."
    )
    short_description = "Submit a prepared MAGUS job to SCNet."
    tags = ("magus", "submit", "scnet", "slurm")
    cost_class = "VERY_EXPENSIVE"
    input_schema = {
        "type": "object",
        "properties": {
            "job_name": {"type": "string"},
            "prepared_dir": {"type": "string"},
            "partition": {"type": "string"},
            "nodes": {"type": "integer", "minimum": 1},
            "tasks_per_node": {"type": "integer", "minimum": 1},
            "walltime_minutes": {"type": "integer", "minimum": 1},
        },
        "required": ["job_name", "prepared_dir"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        application = self._application()
        if application is None:
            return _error("UNCONFIGURED", "MAGUS is UNCONFIGURED")
        try:
            resource = ResourceRequest(
                partition=str(arguments.get("partition") or ""),
                nodes=int(arguments.get("nodes", 1)),
                tasks_per_node=int(arguments.get("tasks_per_node", 8)),
                walltime_minutes=int(arguments.get("walltime_minutes", 120)),
            )
            ref = await application.submit(
                job_name=str(arguments["job_name"]),
                prepared_dir=str(arguments["prepared_dir"]),
                resource=resource,
            )
        except Exception as exc:
            return self._error_for(exc)
        payload = ref.model_dump()
        payload["note"] = (
            "detached job; poll with magus.status; Slurm COMPLETED is "
            "scheduler state, not scientific validity"
        )
        return _result(payload)


class MagusStatusTool(_LifecycleTool):
    name = "magus.status"
    description = (
        "Query the Slurm state of a MAGUS job id (PENDING/RUNNING/COMPLETED/"
        "FAILED/...). Scheduler state only; COMPLETED does not imply "
        "scientific validity."
    )
    short_description = "MAGUS job scheduler state."
    tags = ("magus", "status", "scnet")
    cost_class = "CHEAP"
    input_schema = {
        "type": "object",
        "properties": {"job_id": {"type": "string"}},
        "required": ["job_id"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        application = self._application()
        if application is None:
            return _error("UNCONFIGURED", "MAGUS is UNCONFIGURED")
        try:
            state = await application.status(str(arguments["job_id"]))
        except Exception as exc:
            return self._error_for(exc)
        return _result(
            {
                "job_id": arguments["job_id"],
                "state": state.value,
                "terminal": state.terminal,
                "note": "scheduler state only; use magus.collect for artifacts",
            }
        )


class MagusCollectTool(_LifecycleTool):
    name = "magus.collect"
    description = (
        "Download a finished MAGUS job's bounded artifacts (input.yaml, "
        "summary, traj files, logs) and produce a structured report with "
        "candidate count. Candidates remain UNVALIDATED_GENERATED_STRUCTURE "
        "unless energies were actually computed by the internal calculator."
    )
    short_description = "Collect bounded MAGUS job artifacts."
    tags = ("magus", "collect", "scnet")
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

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        application = self._application()
        if application is None:
            return _error("UNCONFIGURED", "MAGUS is UNCONFIGURED")
        from photomatagent.scientific.remote.models import RemoteJobRef

        job_ref = RemoteJobRef(
            backend="scnet",
            application="magus",
            job_id=str(arguments["job_id"]),
            remote_directory=str(arguments["remote_directory"]),
        )
        try:
            report = await application.collect(
                job_ref=job_ref,
                local_dir=arguments.get("local_dir") or "output/magus_results",
            )
        except Exception as exc:
            return self._error_for(exc)
        return _result(report)


class MagusInspectResultsTool(_LifecycleTool):
    name = "magus.inspect_results"
    description = (
        "Parse a local MAGUS result directory (collected by magus.collect): "
        "bounded summary text, candidate artifact list and frame counts. "
        "Never fabricates energies or candidate counts."
    )
    short_description = "Inspect collected MAGUS results."
    tags = ("magus", "inspect", "results")
    cost_class = "CHEAP"
    input_schema = {
        "type": "object",
        "properties": {
            "result_dir": {"type": "string"},
            "operation": {"type": "string", "enum": ["generate", "search"]},
        },
        "required": ["result_dir"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        application = self._application()
        if application is None:
            application = MagusApplication()
        payload = application.inspect_results(
            str(arguments["result_dir"]),
            operation=str(arguments.get("operation", "generate")),
        )
        return _result(payload)


def _search_alias_tool(search_type: str) -> type[Any]:
    """Deprecated prepare-only alias for one geometry type."""

    class _MagusSearchAlias(_LifecycleTool):
        name = f"magus.search_{search_type}"
        description = (
            f"DEPRECATED prepare-only alias of magus.prepare_search for "
            f"{search_type} searches; use magus.prepare_search instead. "
            "MAGUS candidates are UNVALIDATED_GENERATED_STRUCTURE."
        )
        short_description = f"Deprecated: prepare a MAGUS {search_type} search."
        tags = ("magus", "structure search", search_type, "deprecated")
        cost_class = "VERY_EXPENSIVE"
        input_schema = {
            "type": "object",
            "properties": {
                "composition": {"type": "string"},
                "target_dir": {"type": "string"},
                "output_dir": {"type": "string"},
                "generations": {"type": "integer", "minimum": 1},
                "population_size": {"type": "integer", "minimum": 1},
            },
            "required": ["composition", "target_dir", "output_dir"],
        }

        async def execute(
            self, arguments: dict[str, Any]
        ) -> ScientificToolResult:
            application = self._application()
            if application is None:
                return _error("UNCONFIGURED", "MAGUS is UNCONFIGURED")
            try:
                manifest = application.prepare(
                    search_type=search_type,
                    composition=str(arguments["composition"]),
                    target_dir=str(arguments["target_dir"]),
                    output_dir=str(arguments["output_dir"]),
                    generations=int(arguments.get("generations", 4)),
                    population_size=int(arguments.get("population_size", 4)),
                )
            except Exception as exc:
                return _error(
                    "EXECUTION_FAILED",
                    f"magus.search_{search_type} failed: {type(exc).__name__}: {exc}",
                )
            return _result(manifest)

    return _MagusSearchAlias


class MagusCapabilityPack(CapabilityPack):
    name = "magus"
    description = (
        "MAGUS structure search over SCNet (candidate generation/search "
        "evidence, never detector performance evidence)."
    )
    execution_mode = "mcp/scnet"
    backend_name = "SCNet (SSH + Slurm)"

    def __init__(self, application: MagusApplication | None = None) -> None:
        self.application = application

    def probe(self) -> ProbeResult:
        application = self.application or default_magus_application()
        if application is None:
            return ProbeResult(
                status=CapabilityStatus.UNCONFIGURED,
                detail="no SCNet backend configured",
            )
        report = application.probe_environment()
        status = report.get("status")
        if status == "AVAILABLE":
            return ProbeResult(
                status=CapabilityStatus.AVAILABLE,
                detail=(
                    f"remote MAGUS {report.get('version', '?')} at "
                    f"{report.get('executable', '?')}"
                ),
                version=report.get("version", ""),
            )
        if status in {"MISSING_DEPENDENCY", "MISSING_PREREQUISITE"}:
            return ProbeResult(
                status=CapabilityStatus.MISSING_DEPENDENCY,
                detail=str(report.get("detail", status)),
            )
        return ProbeResult(
            status=CapabilityStatus.UNCONFIGURED,
            detail=str(report.get("detail", "MAGUS not configured")),
        )

    def tools(self) -> list[Tool]:
        application = self.application or default_magus_application()
        prepare_generate = _prepare_generate_tool()
        prepare_search = _prepare_search_tool()
        tools: list[Tool] = [
            MagusCapabilitiesTool(application),
            MagusProbeTool(application),
            prepare_generate(application),
            prepare_search(application),
            MagusSubmitTool(application),
            MagusStatusTool(application),
            MagusCollectTool(application),
            MagusInspectResultsTool(application),
        ]
        for search_type in SUPPORTED_STRUCTURE_TYPES:
            tools.append(_search_alias_tool(search_type)(application))
        return tools


def magus_pack(workspace: Any = None) -> MagusCapabilityPack:
    del workspace
    return MagusCapabilityPack()
