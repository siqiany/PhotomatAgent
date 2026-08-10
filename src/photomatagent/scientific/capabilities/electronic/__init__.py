"""Electronic-structure capability pack (Sumo + effmass + pymatgen).

Namespace ``electronic``, DEFERRED. Tools that need a VASP ``vasprun.xml``
parse it with pymatgen/Sumo; effmass is only used for post-processing an
existing band structure (never to run DFT).
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


class ElectronicProbe(CapabilityPack):
    name = "electronic"
    description = "Band structure / DOS / effective mass analysis (Sumo, effmass)."

    def probe(self) -> ProbeResult:
        statuses = []
        details = []
        try:
            import sumo  # noqa: F401

            statuses.append("sumo")
            details.append(f"sumo={importlib.metadata.version('sumo')}")
        except ImportError:
            details.append("sumo=MISSING")
        try:
            import effmass  # noqa: F401

            statuses.append("effmass")
            details.append(f"effmass={importlib.metadata.version('effmass')}")
        except ImportError:
            details.append("effmass=MISSING")
        if not statuses:
            return ProbeResult(
                status=CapabilityStatus.MISSING_DEPENDENCY,
                detail="; ".join(details) + " (extra: photomatagent[electronic])",
            )
        return ProbeResult(
            status=CapabilityStatus.AVAILABLE,
            detail="; ".join(details),
        )

    def tools(self) -> list[Tool]:
        tools: list[Tool] = [
            ElectronicBandSummaryTool(self._workspace),
            ElectronicDosSummaryTool(self._workspace),
        ]
        try:
            import sumo  # noqa: F401

            tools.extend(
                [
                    ElectronicPlotBandTool(self._config, self._workspace),
                    ElectronicPlotDosTool(self._config, self._workspace),
                ]
            )
        except ImportError:
            pass
        try:
            import effmass  # noqa: F401

            tools.append(ElectronicEffectiveMassTool(self._workspace))
        except ImportError:
            pass
        return tools

    def __init__(self, config: ScientificConfig, workspace: Workspace) -> None:
        self._config = config
        self._workspace = workspace


def _resolve(path_value: str, workspace: Workspace) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        candidate = workspace.root / path
        if candidate.is_file():
            path = candidate
    if not path.is_file():
        raise ValueError(f"file not found: {path_value}")
    return path


def _load_vasprun(path_value: str, workspace: Workspace) -> tuple[Any, Path]:
    from pymatgen.io.vasp.outputs import Vasprun

    path = _resolve(path_value, workspace)
    return Vasprun(str(path), parse_potcar_file=False), path


def _gap_evidence(
    formula: str, gap: float | None, direct: bool | None, source: str, method: str
) -> list[ScientificEvidence]:
    if gap is None:
        return []
    return [
        ScientificEvidence(
            subject=formula,
            property="band_gap",
            value=round(float(gap), 4),
            unit="eV",
            source=source,
            source_type="calculation",
            method=method,
            summary=f"Computed band gap for {formula} is {gap:.3f} eV"
            + (f" ({'direct' if direct else 'indirect'})" if direct is not None else ""),
            limitations="DFT band gap; expect systematic underestimation",
            provenance={"tool": method, "source_file": source},
        )
    ]


class ElectronicBandSummaryTool(Tool):
    name = "electronic.band_summary"
    description = (
        "Summarize a VASP band structure (vasprun.xml): band gap, CBM/VBM "
        "energies, direct/indirect character, and metallicity."
    )
    short_description = "Band gap and CBM/VBM summary from vasprun.xml."
    exposure = ToolExposure.DEFERRED
    namespace = "electronic"
    source = "pymatgen"
    tags = ("electronic", "band structure", "band gap", "vasprun")
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to vasprun.xml."}
        },
        "required": ["path"],
    }

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        try:
            vasprun, path = _load_vasprun(str(arguments["path"]), self._workspace)
            structure = vasprun.final_structure
            formula = structure.composition.reduced_formula
        except Exception as exc:
            return ScientificToolResult(
                output=f"electronic.band_summary failed: {type(exc).__name__}: {exc}",
                is_error=True,
                data={"error": type(exc).__name__},
            )
        try:
            properties = vasprun.eigenvalue_band_properties
            gap, cbm, vbm, is_direct = (
                float(properties[0]),
                float(properties[1]),
                float(properties[2]),
                bool(properties[3]),
            )
            is_metal = gap <= 1e-4
        except Exception as exc:
            return ScientificToolResult(
                output=f"band structure parsing failed: {type(exc).__name__}: {exc}",
                is_error=True,
                data={"error": type(exc).__name__},
            )
        card = {
            "file": path.name,
            "formula": formula,
            "band_gap_eV": round(gap, 4),
            "cbm_eV": round(cbm, 4),
            "vbm_eV": round(vbm, 4),
            "direct_gap": is_direct,
            "is_metal": is_metal,
        }
        evidence = _gap_evidence(
            formula, None if is_metal else gap, is_direct, str(path), "pymatgen Vasprun"
        )
        return ScientificToolResult(
            output=json.dumps(card, ensure_ascii=False),
            data=card,
            evidence=evidence,
        )


class ElectronicDosSummaryTool(Tool):
    name = "electronic.dos_summary"
    description = (
        "Summarize the DOS from vasprun.xml: Fermi level, gap, and integrated "
        "DOS near the band edges."
    )
    short_description = "DOS summary (Fermi level, gap, edge DOS) from vasprun.xml."
    exposure = ToolExposure.DEFERRED
    namespace = "electronic"
    source = "pymatgen"
    tags = ("electronic", "dos", "fermi level", "vasprun")
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "energy_window_eV": {"type": "number", "minimum": 0.1, "maximum": 5.0},
        },
        "required": ["path"],
    }

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        try:
            vasprun, path = _load_vasprun(str(arguments["path"]), self._workspace)
            structure = vasprun.final_structure
            formula = structure.composition.reduced_formula
            dos = vasprun.complete_dos
        except Exception as exc:
            return ScientificToolResult(
                output=f"electronic.dos_summary failed: {type(exc).__name__}: {exc}",
                is_error=True,
                data={"error": type(exc).__name__},
            )
        try:
            fermi = float(dos.efermi)
            window = float(arguments.get("energy_window_eV", 1.0))
            energy = dos.energies
            total = dos.get_densities()
            idx = list(range(len(energy)))
            near = [
                (float(energy[i]), float(total[i]))
                for i in idx
                if abs(float(energy[i]) - fermi) <= window
            ]
            valence_peak = max(near, key=lambda item: item[1]) if near else (0.0, 0.0)
            spin = vasprun.eigenvalue_band_properties
            gap = float(spin[0]) if spin else None
        except Exception as exc:
            return ScientificToolResult(
                output=f"DOS parsing failed: {type(exc).__name__}: {exc}",
                is_error=True,
                data={"error": type(exc).__name__},
            )
        card = {
            "file": path.name,
            "formula": formula,
            "fermi_level_eV": round(fermi, 4),
            "band_gap_eV": round(gap, 4) if gap is not None and gap > 1e-4 else None,
            "dos_max_energy_eV": round(valence_peak[0], 4),
            "dos_max_density": round(valence_peak[1], 4),
            "window_eV": window,
        }
        evidence = _gap_evidence(
            formula, None if gap is None or gap <= 1e-4 else gap, None, str(path), "pymatgen CompleteDos"
        )
        return ScientificToolResult(
            output=json.dumps(card, ensure_ascii=False),
            data=card,
            evidence=evidence,
        )


class ElectronicPlotBandTool(Tool):
    name = "electronic.plot_band"
    description = (
        "Render a band structure plot from vasprun.xml using Sumo and save a "
        "PNG artifact under output/scientific."
    )
    short_description = "Plot band structure (Sumo) from vasprun.xml."
    exposure = ToolExposure.DEFERRED
    namespace = "electronic"
    source = "sumo"
    tags = ("electronic", "plot", "band structure", "figure")
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    def __init__(self, config: ScientificConfig, workspace: Workspace) -> None:
        self._config = config
        self._workspace = workspace

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        try:
            vasprun, path = _load_vasprun(str(arguments["path"]), self._workspace)
            structure = vasprun.final_structure
            formula = structure.composition.reduced_formula
        except Exception as exc:
            return ScientificToolResult(
                output=f"electronic.plot_band failed: {type(exc).__name__}: {exc}",
                is_error=True,
                data={"error": type(exc).__name__},
            )
        try:
            from sumo.plotting.bs_plotter import BSPlotter

            bs = vasprun.get_band_structure()
            plotter = BSPlotter(bs)
            output_dir = (self._workspace.root / self._config.structure_output_dir).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / f"{path.stem}_band.png"
            plotter.save_plot(str(out_path))
        except Exception as exc:
            return ScientificToolResult(
                output=f"band plot failed: {type(exc).__name__}: {exc}",
                is_error=True,
                data={"error": type(exc).__name__},
            )
        payload = {
            "formula": formula,
            "artifact": str(out_path),
            "bytes": out_path.stat().st_size,
        }
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False),
            data=payload,
            artifacts=[str(out_path)],
        )


class ElectronicPlotDosTool(Tool):
    name = "electronic.plot_dos"
    description = (
        "Render a DOS plot from vasprun.xml using Sumo and save a PNG artifact "
        "under output/scientific."
    )
    short_description = "Plot DOS (Sumo) from vasprun.xml."
    exposure = ToolExposure.DEFERRED
    namespace = "electronic"
    source = "sumo"
    tags = ("electronic", "plot", "dos", "figure")
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    def __init__(self, config: ScientificConfig, workspace: Workspace) -> None:
        self._config = config
        self._workspace = workspace

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        try:
            vasprun, path = _load_vasprun(str(arguments["path"]), self._workspace)
            structure = vasprun.final_structure
            formula = structure.composition.reduced_formula
        except Exception as exc:
            return ScientificToolResult(
                output=f"electronic.plot_dos failed: {type(exc).__name__}: {exc}",
                is_error=True,
                data={"error": type(exc).__name__},
            )
        try:
            from sumo.plotting.dos_plotter import SDOSPlotter

            dos = vasprun.complete_dos
            plotter = SDOSPlotter(dos)
            figure = plotter.get_plot()
            output_dir = (self._workspace.root / self._config.structure_output_dir).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / f"{path.stem}_dos.png"
            figure.savefig(out_path, dpi=150, bbox_inches="tight")
        except Exception as exc:
            return ScientificToolResult(
                output=f"DOS plot failed: {type(exc).__name__}: {exc}",
                is_error=True,
                data={"error": type(exc).__name__},
            )
        payload = {
            "formula": formula,
            "artifact": str(out_path),
            "bytes": out_path.stat().st_size,
        }
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False),
            data=payload,
            artifacts=[str(out_path)],
        )


class ElectronicEffectiveMassTool(Tool):
    name = "electronic.effective_mass"
    description = (
        "Compute band-edge effective masses from an existing VASP band structure "
        "(vasprun.xml) using effmass. Never runs DFT."
    )
    short_description = "Effective mass (effmass) from an existing band structure."
    exposure = ToolExposure.DEFERRED
    namespace = "electronic"
    source = "effmass"
    tags = ("electronic", "effective mass", "carrier", "transport")
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to vasprun.xml with band structure."},
            "carrier": {"type": "string", "enum": ["electron", "hole"], "description": "Carrier type."},
            "method": {
                "type": "string",
                "enum": ["five_point_leastsq", "kane"],
                "description": "Mass fitting method.",
            },
        },
        "required": ["path", "carrier"],
    }

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        try:
            path = _resolve(str(arguments["path"]), self._workspace)
            from effmass import extrema, inputs

            data = inputs.BSVasprun(str(path))
            cb_index, vb_index = extrema.calc_CBM_VBM_from_Fermi(data)
            settings = inputs.Settings(
                conduction_band=True,
                valence_band=True,
                frontier_bands_only=False,
            )
            segments = extrema.generate_segments(settings, data)
            carrier = str(arguments["carrier"])
            method = str(arguments.get("method", "five_point_leastsq"))
        except Exception as exc:
            return ScientificToolResult(
                output=f"electronic.effective_mass failed: {type(exc).__name__}: {exc}",
                is_error=True,
                data={"error": type(exc).__name__},
            )
        masses = []
        for segment in segments:
            if carrier == "electron" and segment.band_type != "conduction_band":
                continue
            if carrier == "hole" and segment.band_type != "valence_band":
                continue
            try:
                if method == "kane":
                    mass = segment.kane_mass_band_edge()
                else:
                    mass = segment.five_point_leastsq_effmass()
                masses.append(
                    {
                        "band": int(segment.band),
                        "kpoint_indices": [int(k) for k in segment.kpoint_indices],
                        f"{method}_effective_mass_m0": round(float(mass), 4),
                    }
                )
            except Exception:
                continue
        if not masses:
            return ScientificToolResult(
                output=(
                    f"no {carrier} band-edge segment could be fitted in {path.name}; "
                    "check that the vasprun.xml contains a band structure along k-paths"
                ),
                is_error=True,
                data={"error": "no_segments"},
            )
        payload = {
            "file": path.name,
            "carrier": carrier,
            "method": method,
            "cbm_index": int(cb_index),
            "vbm_index": int(vb_index),
            "masses": masses,
        }
        evidence = [
            ScientificEvidence(
                subject=path.stem,
                property=f"effective_mass_{carrier}",
                value=masses[0][f"{method}_effective_mass_m0"],
                unit="m0",
                source=str(path),
                source_type="calculation",
                method=f"effmass {method}",
                summary=f"{carrier} effective mass ≈ {masses[0][f'{method}_effective_mass_m0']:.3f} m0",
                limitations="Depends on band fitting window and k-point sampling",
                provenance={"band": masses[0]["band"], "tool": self.name},
            )
        ]
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False),
            data=payload,
            evidence=evidence,
        )


def electronic_pack(config: ScientificConfig, workspace: Workspace) -> CapabilityPack:
    return ElectronicProbe(config, workspace)
