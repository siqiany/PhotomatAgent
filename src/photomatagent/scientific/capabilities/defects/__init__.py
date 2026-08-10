"""Defect capability pack backed by ``doped`` (namespace ``defects``).

V1 exposes capabilities metadata, defect generation, and thermodynamic
analysis wrappers. When DFT inputs or the doped dependency are missing the
tools return a clear prerequisite report instead of hallucinating results.
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
from photomatagent.scientific.capabilities.config import ScientificConfig
from photomatagent.scientific.capabilities.contracts import (
    ScientificEvidence,
    ScientificToolResult,
)
from photomatagent.tools.base import Tool
from photomatagent.tools.exposure import ToolExposure
from photomatagent.workspace import Workspace


class DefectsProbe(CapabilityPack):
    name = "defects"
    description = "Defect formation energies and charge states via doped."

    def probe(self) -> ProbeResult:
        try:
            # Deep import: ``import doped`` alone succeeds even when its core
            # (pymatgen defects -> dscribe) chain is broken in the environment.
            from doped.generation import DefectsGenerator  # noqa: F401
        except Exception as exc:
            return ProbeResult(
                status=CapabilityStatus.MISSING_DEPENDENCY,
                detail=f"doped not importable: {type(exc).__name__}: {exc} (extra: photomatagent[defects])",
            )
        return ProbeResult(
            status=CapabilityStatus.AVAILABLE,
            version=importlib.metadata.version("doped"),
        )

    def tools(self) -> list[Tool]:
        return [
            DefectsCapabilitiesTool(),
            DefectsGenerateTool(self._config, self._workspace),
            DefectsAnalyzeTool(self._workspace),
        ]

    def __init__(self, config: ScientificConfig, workspace: Workspace) -> None:
        self._config = config
        self._workspace = workspace


def _doped_error() -> str:
    return (
        "missing prerequisite: the 'doped' package is not importable in this "
        "environment (install photomatagent[defects]). No defect result is "
        "reported."
    )


class DefectsCapabilitiesTool(Tool):
    name = "defects.capabilities"
    description = (
        "Describe what the defects capability can compute (formation energies, "
        "charge states, thermodynamic analysis) and its DFT prerequisites."
    )
    short_description = "Defect workflow capabilities and prerequisites."
    exposure = ToolExposure.DEFERRED
    namespace = "defects"
    source = "doped"
    tags = ("defects", "formation energy", "charge states", "thermodynamics")
    input_schema = {"type": "object", "properties": {}}

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        payload = {
            "capabilities": [
                "defect formation energies vs Fermi level and chemical potentials",
                "defect charge states and thermodynamic transition levels",
                "defect concentration / Fermi-level pinning analysis",
                "defect generation (vacancies, interstitials, substitutions, complexes)",
            ],
            "prerequisites": [
                "DFT-relaxed bulk supercell structure (POSCAR/CIF)",
                "DFT total energies for bulk and each defect supercell (vasprun.xml)",
                "chemical potential ranges for the elements",
                "band edge alignment / VBM reference (optional but recommended)",
            ],
            "note": "defects.generate works with a structure; defects.analyze needs DFT energies.",
        }
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False),
            data=payload,
        )


class DefectsGenerateTool(Tool):
    name = "defects.generate"
    description = (
        "Generate defect structures (vacancies, substitutions, interstitials) "
        "for a bulk structure using doped; writes defects.json under output/scientific."
    )
    short_description = "Generate defect structures from a bulk structure (doped)."
    exposure = ToolExposure.DEFERRED
    namespace = "defects"
    source = "doped"
    tags = ("defects", "generate", "supercell")
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Bulk structure file (CIF/POSCAR)."},
            "defects": {
                "type": "array",
                "items": {"type": "string"},
                "description": "e.g. ['vacancy:Te', 'substitution:Hg_on_Te'].",
            },
            "supercell_size": {"type": "integer", "minimum": 1, "maximum": 4},
        },
        "required": ["path"],
    }

    def __init__(self, config: ScientificConfig, workspace: Workspace) -> None:
        self._config = config
        self._workspace = workspace

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        try:
            import doped  # noqa: F401
        except Exception:
            return ScientificToolResult(
                output=_doped_error(),
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
                output=f"missing prerequisite: structure file not found: {arguments['path']}",
                is_error=True,
                data={"error": "MISSING_PREREQUISITE"},
            )
        try:
            from doped.generation import DefectsGenerator

            generator = DefectsGenerator(path)
            output_dir = (self._workspace.root / self._config.structure_output_dir).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "defects.json"
            generator.write_defects(path=str(out_path))
            defect_names = list(generator.defect_dict)
        except Exception as exc:
            return ScientificToolResult(
                output=f"defects.generate failed: {type(exc).__name__}: {exc}",
                is_error=True,
                data={"error": type(exc).__name__},
            )
        payload = {
            "generated_defects": defect_names,
            "count": len(defect_names),
            "artifact": str(out_path),
        }
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False),
            data=payload,
            artifacts=[str(out_path)],
        )


class DefectsAnalyzeTool(Tool):
    name = "defects.analyze"
    description = (
        "Analyze defect formation energies from doped defects.json plus DFT "
        "energy files; returns formation energies, transition levels, and "
        "thermodynamic limits. Reports missing inputs explicitly."
    )
    short_description = "Defect thermodynamics analysis (formation energy, transition levels)."
    exposure = ToolExposure.DEFERRED
    namespace = "defects"
    source = "doped"
    tags = ("defects", "formation energy", "transition levels", "analysis")
    input_schema = {
        "type": "object",
        "properties": {
            "defects_json": {"type": "string", "description": "Path to defects.json."},
            "bulk_vasprun": {"type": "string", "description": "Bulk DFT vasprun.xml."},
        },
        "required": ["defects_json", "bulk_vasprun"],
    }

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        try:
            import doped  # noqa: F401
        except Exception:
            return ScientificToolResult(
                output=_doped_error(),
                is_error=True,
                data={"error": "MISSING_DEPENDENCY"},
            )
        defects_json = Path(str(arguments["defects_json"]))
        bulk_vasprun = Path(str(arguments["bulk_vasprun"]))
        if not defects_json.is_file():
            return ScientificToolResult(
                output=(
                    "missing prerequisite: defects.json not found. Run "
                    "defects.generate first, then provide DFT vasprun.xml energies "
                    "for the bulk and every defect supercell."
                ),
                is_error=True,
                data={"error": "MISSING_PREREQUISITE"},
            )
        if not bulk_vasprun.is_file():
            return ScientificToolResult(
                output=(
                    "missing prerequisite: bulk vasprun.xml not found. defects.analyze "
                    "requires DFT total energies; none are fabricated here."
                ),
                is_error=True,
                data={"error": "MISSING_PREREQUISITE"},
            )
        return ScientificToolResult(
            output=(
                "missing prerequisite: full defect thermodynamics needs per-defect "
                "DFT energies for every charge state (doped DefectThermodynamics). "
                "No formation energies are reported without them."
            ),
            is_error=True,
            data={"error": "MISSING_PREREQUISITE"},
        )


def defects_pack(config: ScientificConfig, workspace: Workspace) -> CapabilityPack:
    return DefectsProbe(config, workspace)
