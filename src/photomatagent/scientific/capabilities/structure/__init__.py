"""pymatgen structure capability pack (namespace ``structure``).

Wraps mature pymatgen functionality; no crystallography is reimplemented.
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


class StructureProbe(CapabilityPack):
    name = "structure"
    description = "Crystal structure analysis via pymatgen."

    def probe(self) -> ProbeResult:
        try:
            import pymatgen.core  # noqa: F401
        except ImportError:
            return ProbeResult(
                status=CapabilityStatus.MISSING_DEPENDENCY,
                detail="pymatgen is not installed (base dependency)",
            )
        return ProbeResult(
            status=CapabilityStatus.AVAILABLE,
            version=importlib.metadata.version("pymatgen"),
        )

    def tools(self) -> list[Tool]:
        return [
            StructureSummaryTool(self._workspace),
            StructureSymmetryTool(self._workspace),
            StructureDensityTool(self._workspace),
            StructureNeighborsTool(self._workspace),
            StructureConvertTool(self._config, self._workspace),
        ]

    def __init__(self, config: ScientificConfig, workspace: Workspace) -> None:
        self._config = config
        self._workspace = workspace


def _load_structure(path_value: str, workspace: Workspace) -> Any:
    from pymatgen.core import Structure

    path = Path(path_value).expanduser()
    if not path.is_absolute():
        candidate = workspace.root / path
        if candidate.is_file():
            path = candidate
    if not path.is_file():
        raise ValueError(f"structure file not found: {path_value}")
    return Structure.from_file(str(path)), path


def _evidence_for_structure(
    structure: Any, *, property_name: str, value: Any, unit: str, method: str, note: str
) -> ScientificEvidence:
    return ScientificEvidence(
        subject=structure.composition.reduced_formula,
        property=property_name,
        value=value,
        unit=unit,
        source="pymatgen",
        source_type="calculation",
        method=method,
        summary=f"{structure.composition.reduced_formula}: {property_name} = {value} {unit}",
        limitations=note,
        provenance={"formula": structure.composition.reduced_formula, "n_sites": len(structure)},
    )


class StructureSummaryTool(Tool):
    name = "structure.summary"
    description = (
        "Summarize a crystal structure file (CIF, POSCAR, xsf, or pymatgen JSON): "
        "formula, sites, lattice, density, and space group."
    )
    short_description = "Summarize a structure file (formula, lattice, density, symmetry)."
    exposure = ToolExposure.DEFERRED
    namespace = "structure"
    source = "pymatgen"
    tags = ("structure", "crystallography", "pymatgen")
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Structure file path (CIF/POSCAR/xsf/JSON)."}
        },
        "required": ["path"],
    }

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        try:
            structure, path = _load_structure(str(arguments["path"]), self._workspace)
        except Exception as exc:
            return ScientificToolResult(
                output=f"structure.summary failed: {type(exc).__name__}: {exc}",
                is_error=True,
                data={"error": type(exc).__name__},
            )
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        try:
            sg = SpacegroupAnalyzer(structure).get_space_group_symbol()
        except Exception:
            sg = "unknown"
        card = {
            "file": path.name,
            "formula": structure.composition.reduced_formula,
            "n_sites": len(structure),
            "volume_ang3": round(float(structure.volume), 4),
            "density_g_cm3": round(float(structure.density), 4),
            "space_group": sg,
            "lattice": [
                [round(float(v), 4) for v in row] for row in structure.lattice.matrix
            ],
        }
        evidence = [
            _evidence_for_structure(
                structure,
                property_name="density",
                value=card["density_g_cm3"],
                unit="g/cm3",
                method="pymatgen structure density",
                note="Depends on the input structure; DFT-relaxed inputs are more reliable",
            )
        ]
        return ScientificToolResult(
            output=json.dumps(card, ensure_ascii=False),
            data=card,
            evidence=evidence,
        )


class StructureSymmetryTool(Tool):
    name = "structure.symmetry"
    description = "Determine space group, point group, and symmetry operations of a structure."
    short_description = "Space group / symmetry analysis of a structure file."
    exposure = ToolExposure.DEFERRED
    namespace = "structure"
    source = "pymatgen"
    tags = ("structure", "symmetry", "space group")
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        try:
            structure, path = _load_structure(str(arguments["path"]), self._workspace)
        except Exception as exc:
            return ScientificToolResult(
                output=f"structure.symmetry failed: {type(exc).__name__}: {exc}",
                is_error=True,
                data={"error": type(exc).__name__},
            )
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        try:
            analyzer = SpacegroupAnalyzer(structure)
            card = {
                "file": path.name,
                "formula": structure.composition.reduced_formula,
                "space_group_symbol": analyzer.get_space_group_symbol(),
                "space_group_number": analyzer.get_space_group_number(),
                "point_group": analyzer.get_point_group_symbol(),
                "crystal_system": analyzer.get_crystal_system(),
            }
        except Exception as exc:
            return ScientificToolResult(
                output=f"symmetry analysis failed: {type(exc).__name__}: {exc}",
                is_error=True,
                data={"error": type(exc).__name__},
            )
        evidence = [
            _evidence_for_structure(
                structure,
                property_name="space_group",
                value=card["space_group_symbol"],
                unit="",
                method="pymatgen SpacegroupAnalyzer (spglib)",
                note="Symmetry depends on tolerances and structure quality",
            )
        ]
        return ScientificToolResult(
            output=json.dumps(card, ensure_ascii=False),
            data=card,
            evidence=evidence,
        )


class StructureDensityTool(Tool):
    name = "structure.density"
    description = "Compute the mass density of a structure file."
    short_description = "Mass density (g/cm3) of a structure file."
    exposure = ToolExposure.DEFERRED
    namespace = "structure"
    source = "pymatgen"
    tags = ("structure", "density")
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        try:
            structure, path = _load_structure(str(arguments["path"]), self._workspace)
        except Exception as exc:
            return ScientificToolResult(
                output=f"structure.density failed: {type(exc).__name__}: {exc}",
                is_error=True,
                data={"error": type(exc).__name__},
            )
        density = round(float(structure.density), 4)
        card = {
            "file": path.name,
            "formula": structure.composition.reduced_formula,
            "density_g_cm3": density,
        }
        evidence = [
            _evidence_for_structure(
                structure,
                property_name="density",
                value=density,
                unit="g/cm3",
                method="pymatgen structure density",
                note="Input-structure dependent",
            )
        ]
        return ScientificToolResult(
            output=json.dumps(card, ensure_ascii=False),
            data=card,
            evidence=evidence,
        )


class StructureNeighborsTool(Tool):
    name = "structure.neighbors"
    description = (
        "List coordination environment / neighbors of one site (or all sites of an "
        "element) within a radius."
    )
    short_description = "Neighbor environments of a site or element in a structure."
    exposure = ToolExposure.DEFERRED
    namespace = "structure"
    source = "pymatgen"
    tags = ("structure", "neighbors", "coordination")
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "site_index": {"type": "integer", "minimum": 0},
            "element": {"type": "string", "description": "Element whose sites to analyze."},
            "radius_ang": {"type": "number", "minimum": 0.5, "maximum": 8.0},
            "max_neighbors": {"type": "integer", "minimum": 1, "maximum": 40},
        },
        "required": ["path"],
    }

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        try:
            structure, path = _load_structure(str(arguments["path"]), self._workspace)
        except Exception as exc:
            return ScientificToolResult(
                output=f"structure.neighbors failed: {type(exc).__name__}: {exc}",
                is_error=True,
                data={"error": type(exc).__name__},
            )
        radius = float(arguments.get("radius_ang", 3.5))
        max_neighbors = int(arguments.get("max_neighbors", 20))
        element = arguments.get("element")
        site_index = arguments.get("site_index")
        if element is not None and site_index is not None:
            return ScientificToolResult(
                output="provide either site_index or element, not both",
                is_error=True,
                data={"error": "ambiguous_target"},
            )
        if element is not None:
            targets = [
                (index, site)
                for index, site in enumerate(structure)
                if site.specie.symbol == element
            ][:1]
        elif site_index is not None:
            index = int(site_index)
            targets = [(index, structure[index])]
        else:
            targets = [(0, structure[0])]
        results = []
        for index, site in targets:
            neighbors = structure.get_neighbors(site, r=radius)
            neighbors = sorted(neighbors, key=lambda n: n.nn_distance)[:max_neighbors]
            results.append(
                {
                    "site_index": index,
                    "element": site.specie.symbol,
                    "coordination_number": len(neighbors),
                    "neighbors": [
                        {
                            "element": neighbor.specie.symbol,
                            "distance_ang": round(float(neighbor.nn_distance), 4),
                        }
                        for neighbor in neighbors
                    ],
                }
            )
        payload = {
            "file": path.name,
            "formula": structure.composition.reduced_formula,
            "radius_ang": radius,
            "environments": results,
        }
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False),
            data=payload,
        )


class StructureConvertTool(Tool):
    name = "structure.convert"
    description = (
        "Convert a structure file to another format (cif, poscar, json, xsf) and "
        "write it under the workspace output directory."
    )
    short_description = "Convert a structure file format (cif/poscar/json/xsf)."
    exposure = ToolExposure.DEFERRED
    namespace = "structure"
    source = "pymatgen"
    tags = ("structure", "convert", "format")
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "output_format": {
                "type": "string",
                "enum": ["cif", "poscar", "json", "xsf"],
            },
        },
        "required": ["path", "output_format"],
    }

    def __init__(self, config: ScientificConfig, workspace: Workspace) -> None:
        self._config = config
        self._workspace = workspace

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        try:
            structure, path = _load_structure(str(arguments["path"]), self._workspace)
        except Exception as exc:
            return ScientificToolResult(
                output=f"structure.convert failed: {type(exc).__name__}: {exc}",
                is_error=True,
                data={"error": type(exc).__name__},
            )
        fmt = str(arguments["output_format"])
        output_dir = (self._workspace.root / self._config.structure_output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        extension = {"cif": "cif", "poscar": "vasp", "json": "json", "xsf": "xsf"}[fmt]
        stem = path.stem.replace(" ", "_")
        out_path = output_dir / f"{stem}.{extension}"
        try:
            structure.to(filename=str(out_path), fmt=fmt)
        except Exception as exc:
            return ScientificToolResult(
                output=f"structure.convert failed: {type(exc).__name__}: {exc}",
                is_error=True,
                data={"error": type(exc).__name__},
            )
        payload = {
            "source": str(path),
            "output_format": fmt,
            "artifact": str(out_path),
            "bytes": out_path.stat().st_size,
        }
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False),
            data=payload,
            artifacts=[str(out_path)],
        )


def structure_pack(config: ScientificConfig, workspace: Workspace) -> CapabilityPack:
    return StructureProbe(config, workspace)

