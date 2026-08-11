"""Deferred ``magus.*`` tools (Sprint 3 section 40-41)."""

from __future__ import annotations

import json
from typing import Any

from photomatagent.scientific.applications.magus.application import (
    MagusApplication,
)
from photomatagent.scientific.capabilities.base import (
    CapabilityPack,
    CapabilityStatus,
    ProbeResult,
)
from photomatagent.scientific.capabilities.contracts import ScientificToolResult
from photomatagent.tools.base import Tool
from photomatagent.tools.exposure import ToolExposure


def _search_tool(
    search_type: str, application: MagusApplication
) -> type[Tool]:
    """Build a search tool class for one geometry type."""

    class _MagusSearchTool(Tool):
        name = f"magus.search_{search_type}"
        description = (
            f"MAGUS {search_type} structure search: prepare a search job "
            "manifest for the given composition (no execution). MAGUS "
            "candidates are UNVALIDATED_GENERATED_STRUCTURE and require "
            "CHGNet/DFT validation before any stability claim."
        )
        short_description = f"Prepare a MAGUS {search_type} search."
        exposure = ToolExposure.DEFERRED
        namespace = "magus"
        source = "MAGUS structure search"
        tags = ("magus", "structure search", search_type)
        cost_class = "EXPENSIVE"
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

        def __init__(self, application: MagusApplication | None = None) -> None:
            self.application = application or MagusApplication()

        async def execute(
            self, arguments: dict[str, Any]
        ) -> ScientificToolResult:
            try:
                manifest = self.application.prepare(
                    search_type=search_type,
                    composition=str(arguments["composition"]),
                    target_dir=str(arguments["target_dir"]),
                    output_dir=str(arguments["output_dir"]),
                    generations=int(arguments.get("generations", 30)),
                    population_size=int(arguments.get("population_size", 20)),
                )
            except Exception as exc:
                return ScientificToolResult(
                    output=(
                        f"magus.search_{search_type} failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    is_error=True,
                    data={"error_type": type(exc).__name__, "message": str(exc)},
                )
            return ScientificToolResult(
                output=json.dumps(manifest, ensure_ascii=False, indent=2),
                data=manifest,
            )

    return _MagusSearchTool


class MagusCapabilitiesTool(Tool):
    name = "magus.capabilities"
    description = (
        "Probe MAGUS availability (local binary or SCNet module) and list "
        "the search types supported by this installation. When MAGUS is not "
        "installed, reports UNCONFIGURED with the installation requirement "
        "and PhotoMatAgent keeps working."
    )
    short_description = "MAGUS availability and supported search types."
    exposure = ToolExposure.DEFERRED
    namespace = "magus"
    source = "environment probe"
    tags = ("magus", "structure search", "capabilities")
    cost_class = "CHEAP"
    input_schema: dict[str, Any] = {"type": "object", "properties": {}}

    def __init__(self, application: MagusApplication | None = None) -> None:
        self.application = application

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        application = self.application or MagusApplication()
        payload = application.probe_environment()
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False, indent=2),
            data=payload,
        )


class MagusProbeTool(Tool):
    name = "magus.probe"
    description = (
        "Same as magus.capabilities but only returns the probe result "
        "(status + executable detection). Read-only."
    )
    short_description = "MAGUS probe (status only)."
    exposure = ToolExposure.DEFERRED
    namespace = "magus"
    source = "environment probe"
    tags = ("magus", "probe")
    cost_class = "CHEAP"
    input_schema: dict[str, Any] = {"type": "object", "properties": {}}

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        payload = MagusApplication().probe_environment()
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False, indent=2),
            data=payload,
        )


class MagusCapabilityPack(CapabilityPack):
    name = "magus"
    description = "MAGUS structure search (optional, probe-gated)."
    execution_mode = "mcp/scnet"
    backend_name = "SCNet (SSH + Slurm)"

    def __init__(self, application: MagusApplication | None = None) -> None:
        self.application = application or MagusApplication()

    def probe(self) -> ProbeResult:
        report = self.application.probe_environment()
        if report.get("status") == "AVAILABLE":
            return ProbeResult(
                status=CapabilityStatus.AVAILABLE,
                detail=f"executable {self.application.executable} found",
            )
        return ProbeResult(
            status=CapabilityStatus.UNCONFIGURED,
            detail=(
                "MAGUS not installed; installation requirement recorded, "
                "agent keeps working"
            ),
        )

    def tools(self) -> list[Tool]:
        tools: list[Tool] = [
            MagusCapabilitiesTool(self.application),
            MagusProbeTool(),
        ]
        for search_type in self.application.search_types:
            tool_class = _search_tool(search_type, self.application)
            tools.append(tool_class(self.application))  # type: ignore[call-arg]
        return tools


def magus_pack(workspace: Any = None) -> MagusCapabilityPack:
    return MagusCapabilityPack()
