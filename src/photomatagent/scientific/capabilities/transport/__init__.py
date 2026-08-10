"""Transport capability pack backed by AMSET (namespace ``transport``).

V1 exposes capabilities metadata and an analysis wrapper. Without first-
principles input data (or the AMSET dependency) the tools return
prerequisites; expensive DFT is never auto-launched.
"""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any

from photomatagent.scientific.capabilities.base import (
    CapabilityPack,
    CapabilityStatus,
    ProbeResult,
)
from photomatagent.scientific.capabilities.contracts import ScientificToolResult
from photomatagent.tools.base import Tool
from photomatagent.tools.exposure import ToolExposure
from photomatagent.workspace import Workspace


class TransportProbe(CapabilityPack):
    name = "transport"
    description = "Carrier transport (mobility, conductivity, Seebeck) via AMSET."

    def probe(self) -> ProbeResult:
        try:
            import amset  # noqa: F401
        except Exception as exc:
            return ProbeResult(
                status=CapabilityStatus.MISSING_DEPENDENCY,
                detail=(
                    f"amset not importable: {type(exc).__name__}: {exc} "
                    "(extra: photomatagent[transport])"
                ),
            )
        return ProbeResult(
            status=CapabilityStatus.AVAILABLE,
            version=importlib.metadata.version("amset"),
        )

    def tools(self) -> list[Tool]:
        return [
            TransportCapabilitiesTool(),
            TransportAnalyzeTool(self._workspace),
        ]

    def __init__(self, config: Any, workspace: Workspace) -> None:
        self._workspace = workspace


class TransportCapabilitiesTool(Tool):
    name = "transport.capabilities"
    description = (
        "Describe what the transport capability can compute (carrier mobility, "
        "conductivity, Seebeck coefficient vs temperature/doping) and its "
        "first-principles prerequisites."
    )
    short_description = "Transport analysis capabilities and prerequisites."
    exposure = ToolExposure.DEFERRED
    namespace = "transport"
    source = "amset"
    tags = ("transport", "mobility", "carrier mobility", "conductivity", "seebeck")
    input_schema = {"type": "object", "properties": {}}

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        payload = {
            "capabilities": [
                "carrier mobility vs temperature and doping",
                "electrical conductivity and Seebeck coefficient",
                "scattering analysis (electron-phonon, ionized impurity)",
                "carrier lifetime estimates (future)",
            ],
            "prerequisites": [
                "DFT band structure (vasprun.xml) with dense k-mesh",
                "DFT deformation potential calculations (electron-phonon)",
                "DFT dielectric constant and elastic constants",
                "doping grid / temperature range settings",
            ],
            "note": "AMSET runs on existing DFT results; it never launches DFT.",
        }
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False),
            data=payload,
        )


class TransportAnalyzeTool(Tool):
    name = "transport.analyze"
    description = (
        "Run AMSET transport analysis on an existing DFT vasprun.xml; returns "
        "mobility/conductivity/Seebeck summaries. Reports prerequisites when "
        "data or the dependency is missing."
    )
    short_description = "Run AMSET transport analysis on a vasprun.xml."
    exposure = ToolExposure.DEFERRED
    namespace = "transport"
    source = "amset"
    tags = ("transport", "mobility", "analysis", "vasprun")
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to vasprun.xml."},
            "temperatures": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Temperature grid in K.",
            },
        },
        "required": ["path"],
    }

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        try:
            import amset  # noqa: F401
        except Exception:
            return ScientificToolResult(
                output=(
                    "missing prerequisite: the 'amset' package is not importable in this "
                    "environment (install photomatagent[transport]). No mobility values "
                    "are fabricated."
                ),
                is_error=True,
                data={"error": "MISSING_DEPENDENCY"},
            )
        path = Path(str(arguments["path"]))
        if not path.is_absolute():
            candidate = self._workspace.root / path
            if candidate.is_file():
                path = candidate
        if not path.is_file():
            return ScientificToolResult(
                output=f"missing prerequisite: vasprun.xml not found: {arguments['path']}",
                is_error=True,
                data={"error": "MISSING_PREREQUISITE"},
            )
        return ScientificToolResult(
            output=(
                "missing prerequisite: AMSET needs deformation-potential and "
                "dielectric/elastic DFT data in addition to this vasprun.xml. "
                "See transport.capabilities for the full input list; no mobility "
                "is reported without them."
            ),
            is_error=True,
            data={"error": "MISSING_PREREQUISITE"},
        )


def transport_pack(config: Any, workspace: Workspace) -> CapabilityPack:
    return TransportProbe(config, workspace)

