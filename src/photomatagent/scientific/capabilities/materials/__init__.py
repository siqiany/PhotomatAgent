"""Materials Project capability pack backed by the official ``mp-api``.

All tools are DEFERRED with ``namespace = materials``. The API key is read
from the secret configuration at call time and never appears in context,
traces, or tool payloads.
"""

from __future__ import annotations

import importlib.metadata
import json
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


def _mp_version() -> str:
    try:
        return importlib.metadata.version("mp-api")
    except Exception:
        return ""


class _MaterialsProbe(CapabilityPack):
    name = "materials"
    description = "Materials Project database access (mp-api)."

    def __init__(self, config: ScientificConfig) -> None:
        self._config = config

    def probe(self) -> ProbeResult:
        try:
            import mp_api  # noqa: F401
        except ImportError:
            return ProbeResult(
                status=CapabilityStatus.MISSING_DEPENDENCY,
                detail="mp-api is not installed (extra: photomatagent[materials])",
            )
        if not self._config.materials_api_key():
            return ProbeResult(
                status=CapabilityStatus.UNCONFIGURED,
                detail=(
                    f"mp-api installed but no API key; set {self._config.materials_api_key_env}"
                    " in the workspace .env"
                ),
                version=_mp_version(),
            )
        return ProbeResult(
            status=CapabilityStatus.AVAILABLE,
            detail="mp-api installed and key configured",
            version=_mp_version(),
        )

    def tools(self) -> list[Tool]:
        return [
            MaterialsSearchTool(self._config),
            MaterialsGetSummaryTool(self._config),
            MaterialsGetStructureTool(self._config),
        ]


def _open_rester(config: ScientificConfig) -> Any:
    from mp_api.client import MPRester

    key = config.materials_api_key()
    if not key:
        raise _Unconfigured("Materials Project API key is not configured")
    return MPRester(api_key=key, mute_progress_bars=True)


class _Unconfigured(RuntimeError):
    pass


def _materials_error(exc: Exception) -> str:
    if isinstance(exc, _Unconfigured):
        return str(exc)
    return f"materials API call failed: {type(exc).__name__}: {exc}"


class MaterialsSearchTool(Tool):
    """Search Materials Project by formula, elements, and band gap range."""

    name = "materials.search"
    description = (
        "Search Materials Project for compounds by formula or elements with an optional "
        "band gap range; returns a limited list of material ids, formulas, and gaps."
    )
    short_description = "Search Materials Project compounds (formula, elements, band gap)."
    exposure = ToolExposure.DEFERRED
    namespace = "materials"
    source = "mp-api"
    tags = ("database", "materials project", "band gap", "screening")
    input_schema = {
        "type": "object",
        "properties": {
            "formula": {"type": "string", "description": "Chemical formula, e.g. HgTe."},
            "elements": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Element symbols, e.g. ['Hg', 'Te'].",
            },
            "band_gap_min": {"type": "number", "minimum": 0},
            "band_gap_max": {"type": "number", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
        },
    }

    def __init__(self, config: ScientificConfig) -> None:
        self._config = config

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        limit = min(int(arguments.get("limit", 10)), self._config.materials_max_results)
        gap_range = None
        if arguments.get("band_gap_min") is not None or arguments.get("band_gap_max") is not None:
            gap_range = (
                float(arguments.get("band_gap_min", 0) or 0),
                float(arguments.get("band_gap_max", 100) or 100),
            )
        try:
            rester = _open_rester(self._config)
        except Exception as exc:
            return ScientificToolResult(
                output=_materials_error(exc),
                is_error=True,
                data={"error": type(exc).__name__},
            )
        try:
            docs = list(
                rester.summary.search(
                    formula=arguments.get("formula"),
                    elements=arguments.get("elements"),
                    band_gap=gap_range,
                    fields=[
                        "material_id",
                        "formula_pretty",
                        "band_gap",
                        "symmetry",
                        "density",
                        "energy_above_hull",
                    ],
                    num_chunks=1,
                    chunk_size=limit,
                )
            )[:limit]
        except Exception as exc:
            return ScientificToolResult(
                output=_materials_error(exc),
                is_error=True,
                data={"error": type(exc).__name__},
            )
        cards = []
        evidence = []
        for doc in docs:
            symmetry = getattr(doc, "symmetry", None) or {}
            card = {
                "material_id": str(getattr(doc, "material_id", "")),
                "formula": str(getattr(doc, "formula_pretty", "")),
                "band_gap": _opt(getattr(doc, "band_gap", None)),
                "space_group": str(symmetry.get("symbol", "") if isinstance(symmetry, dict) else ""),
                "density": _opt(getattr(doc, "density", None)),
                "energy_above_hull": _opt(getattr(doc, "energy_above_hull", None)),
            }
            cards.append(card)
            gap = getattr(doc, "band_gap", None)
            if gap is not None:
                evidence.append(
                    ScientificEvidence(
                        subject=card["formula"],
                        property="band_gap",
                        value=float(gap),
                        unit="eV",
                        source="Materials Project",
                        source_type="database",
                        method="mp-api summary search (DFT-GGA database value)",
                        summary=f"Database band gap for {card['formula']} is {gap:.3f} eV",
                        limitations="DFT-derived; not a validated experimental gap",
                        provenance={"material_id": card["material_id"], "tool": self.name},
                    )
                )
        payload = {"count": len(cards), "results": cards, "note": "database match, not validated detector"}
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False),
            data={"results": cards},
            evidence=evidence,
        )


class MaterialsGetSummaryTool(Tool):
    name = "materials.get_summary"
    description = "Fetch the Materials Project summary document for one material id."
    short_description = "Fetch MP summary (gap, stability, symmetry) for a material id."
    exposure = ToolExposure.DEFERRED
    namespace = "materials"
    source = "mp-api"
    tags = ("database", "materials project", "summary")
    input_schema = {
        "type": "object",
        "properties": {
            "material_id": {"type": "string", "description": "MP id such as mp-1990."}
        },
        "required": ["material_id"],
    }

    def __init__(self, config: ScientificConfig) -> None:
        self._config = config

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        material_id = str(arguments["material_id"])
        try:
            rester = _open_rester(self._config)
            doc = rester.summary.get_data_by_id(material_id)
        except Exception as exc:
            return ScientificToolResult(
                output=_materials_error(exc),
                is_error=True,
                data={"error": type(exc).__name__},
            )
        if doc is None:
            return ScientificToolResult(
                output=f"no Materials Project summary for {material_id}",
                is_error=True,
                data={"error": "not_found"},
            )
        symmetry = getattr(doc, "symmetry", None) or {}
        card = {
            "material_id": material_id,
            "formula": str(getattr(doc, "formula_pretty", "")),
            "band_gap": _opt(getattr(doc, "band_gap", None)),
            "is_metal": bool(getattr(doc, "is_metal", False)),
            "density": _opt(getattr(doc, "density", None)),
            "space_group": str(symmetry.get("symbol", "") if isinstance(symmetry, dict) else ""),
            "energy_above_hull": _opt(getattr(doc, "energy_above_hull", None)),
            "nsites": _opt(getattr(doc, "nsites", None)),
        }
        evidence = []
        gap = getattr(doc, "band_gap", None)
        if gap is not None:
            evidence.append(
                ScientificEvidence(
                    subject=card["formula"],
                    property="band_gap",
                    value=float(gap),
                    unit="eV",
                    source="Materials Project",
                    source_type="database",
                    method="mp-api summary document",
                    summary=f"Database band gap for {card['formula']} is {gap:.3f} eV",
                    limitations="DFT-derived; not a validated experimental gap",
                    provenance={"material_id": material_id, "tool": self.name},
                )
            )
        return ScientificToolResult(
            output=json.dumps(card, ensure_ascii=False),
            data=card,
            evidence=evidence,
        )


class MaterialsGetStructureTool(Tool):
    name = "materials.get_structure"
    description = (
        "Fetch a Materials Project structure and return a compact summary plus CIF text."
    )
    short_description = "Fetch MP structure (CIF + lattice summary) for a material id."
    exposure = ToolExposure.DEFERRED
    namespace = "materials"
    source = "mp-api"
    tags = ("database", "materials project", "structure")
    input_schema = {
        "type": "object",
        "properties": {
            "material_id": {"type": "string"},
            "max_cif_chars": {"type": "integer", "minimum": 500, "maximum": 8000},
        },
        "required": ["material_id"],
    }

    def __init__(self, config: ScientificConfig) -> None:
        self._config = config

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        material_id = str(arguments["material_id"])
        max_cif = int(arguments.get("max_cif_chars", 4000))
        try:
            rester = _open_rester(self._config)
            structure = rester.get_structure_by_material_id(material_id)
        except Exception as exc:
            return ScientificToolResult(
                output=_materials_error(exc),
                is_error=True,
                data={"error": type(exc).__name__},
            )
        if structure is None:
            return ScientificToolResult(
                output=f"no Materials Project structure for {material_id}",
                is_error=True,
                data={"error": "not_found"},
            )
        cif = structure.to(fmt="cif")
        if len(cif) > max_cif:
            cif = cif[:max_cif] + "\n...[truncated]"
        summary = {
            "material_id": material_id,
            "formula": structure.composition.reduced_formula,
            "n_sites": len(structure),
            "volume": round(float(structure.volume), 4),
            "density": round(float(structure.density), 4),
            "lattice": [
                [round(float(v), 4) for v in row] for row in structure.lattice.matrix
            ],
        }
        evidence = [
            ScientificEvidence(
                subject=summary["formula"],
                property="density",
                value=summary["density"],
                unit="g/cm3",
                source="Materials Project",
                source_type="database",
                method="mp-api structure document",
                summary=f"Database density for {summary['formula']} is {summary['density']:.3f} g/cm3",
                limitations="DFT-relaxed structure; experimental density may differ",
                provenance={"material_id": material_id, "tool": self.name},
            )
        ]
        payload = {**summary, "cif": cif}
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False),
            data=payload,
            evidence=evidence,
        )


def _opt(value: Any) -> Any:
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return value


def materials_pack(config: ScientificConfig) -> CapabilityPack:
    return _MaterialsProbe(config)

