"""Transport capability pack backed by AMSET (namespace ``transport``).

Exposes capabilities metadata and a prerequisite-only analysis wrapper.
The AMSET solver integration is not implemented; importing AMSET does not
make this wrapper executable. Expensive DFT is never auto-launched.
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
    description = "AMSET transport prerequisites; full analysis is not implemented."

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
            detail="AMSET dependency importable; transport.analyze is prerequisite-only.",
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
        "Describe the current prerequisite-only AMSET wrapper, upstream transport "
        "quantities and first-principles inputs. The full solver is not implemented."
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
                "describe AMSET inputs and prerequisites",
                "check AMSET dependency and vasprun.xml presence",
            ],
            "implementation_status": {"transport.analyze": "PREREQUISITE_ONLY"},
            "upstream_capabilities": [
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
            "note": (
                "The current wrapper does not run AMSET or produce transport values. "
                "Installing AMSET or providing a vasprun.xml does not complete the "
                "missing solver integration. No DFT is launched."
            ),
        }
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False),
            data=payload,
        )


class TransportAnalyzeTool(Tool):
    name = "transport.analyze"
    description = (
        "Check AMSET dependency and vasprun.xml prerequisites only. The full solver "
        "is not implemented; this tool does not return mobility, conductivity or "
        "Seebeck values. Use transport.capabilities for implementation limits."
    )
    short_description = "Check AMSET prerequisites; transport solver not implemented."
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
                "transport.analyze is prerequisite-only: full AMSET solver integration "
                "is not implemented. AMSET also needs deformation-potential and "
                "dielectric/elastic inputs, which this schema cannot consume. Use a "
                "supported separately executable method or report the implementation "
                "gap; adding dependencies or retrying this wrapper cannot produce mobility."
            ),
            is_error=True,
            data={"error": "NOT_IMPLEMENTED", "implementation_status": "PREREQUISITE_ONLY"},
        )


def transport_pack(config: Any, workspace: Workspace) -> CapabilityPack:
    return TransportProbe(config, workspace)

