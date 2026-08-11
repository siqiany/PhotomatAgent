"""Meep 1D thin-film R/T/A (donor migration, Sprint 3 section 52-54).

Migrated from the donor ``MeepAdapter`` / ``optical_data`` with a renamed
scope: the tool is ``optics.meep_thinfilm`` (1D thin film, two-run flux
normalization) -- NOT a 3D device simulation, NOT electrical transport, and
NOT an EQE simulation. Every conversion (vasprun dielectric -> n/k -> Meep)
is deterministic code with tests; the tool returns reflectance /
transmittance / absorptance plus the energy-conservation residual.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from photomatagent.scientific.capabilities.contracts import (
    ScientificEvidence,
    ScientificToolResult,
)
from photomatagent.tools.base import Tool
from photomatagent.tools.exposure import ToolExposure

HC_EV_UM = 1.239841984


def dielectric_to_nk(epsilon_real: float, epsilon_imag: float) -> tuple[float, float]:
    """Convert complex dielectric to n and k via (n+ik)^2 = epsilon."""
    magnitude = math.hypot(epsilon_real, epsilon_imag)
    refractive_index = math.sqrt(max(0.0, (magnitude + epsilon_real) / 2.0))
    extinction = math.sqrt(max(0.0, (magnitude - epsilon_real) / 2.0))
    return refractive_index, extinction


def interpolate_dielectric(
    energies_ev: list[float],
    real_tensors: list[list[float]],
    imag_tensors: list[list[float]],
    wavelength_um: float,
    source: str = "dielectric spectrum",
) -> dict[str, Any]:
    """Average the xx/yy/zz diagonal and interpolate to the target wavelength."""
    import numpy as np

    if wavelength_um <= 0:
        raise ValueError("target wavelength must be positive")
    energies = np.asarray(energies_ev, dtype=float)
    real = np.asarray(real_tensors, dtype=float)
    imag = np.asarray(imag_tensors, dtype=float)
    if energies.ndim != 1 or len(energies) < 2:
        raise ValueError("dielectric spectrum needs at least two energy points")
    if real.ndim != 2 or imag.ndim != 2 or real.shape[1] < 3 or imag.shape[1] < 3:
        raise ValueError("dielectric tensors must contain xx, yy and zz components")
    if real.shape[0] != len(energies) or imag.shape[0] != len(energies):
        raise ValueError("dielectric tensor length does not match the energy grid")
    order = np.argsort(energies)
    energies = energies[order]
    target_energy = HC_EV_UM / wavelength_um
    if target_energy < energies[0] or target_energy > energies[-1]:
        raise ValueError(
            f"target energy {target_energy:.4f} eV lies outside the spectrum "
            f"[{energies[0]:.4f}, {energies[-1]:.4f}] eV"
        )
    epsilon_real = float(np.interp(target_energy, energies, real[order, :3].mean(axis=1)))
    epsilon_imag = float(np.interp(target_energy, energies, imag[order, :3].mean(axis=1)))
    refractive_index, extinction = dielectric_to_nk(epsilon_real, epsilon_imag)
    return {
        "wavelength_um": wavelength_um,
        "energy_ev": target_energy,
        "epsilon_real": epsilon_real,
        "epsilon_imag": epsilon_imag,
        "refractive_index": refractive_index,
        "extinction_coefficient": extinction,
        "source": source,
    }


def optical_point_from_vasprun(
    vasprun_path: str | Path, wavelength_um: float
) -> dict[str, Any]:
    """Extract isotropic n/k at a wavelength from a VASP optics vasprun.xml."""
    path = Path(vasprun_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"VASP optical result not found: {path}")
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    dielectric = root.find("calculation/dielectricfunction")
    if dielectric is None:
        raise ValueError(
            f"no dielectricfunction in {path}; run the VASP optics profile"
        )
    real = dielectric.find("varray[@name='real']")
    imag = dielectric.find("varray[@name='imag']")
    if real is None or imag is None:
        raise ValueError(f"dielectric varrays missing in {path}")
    energies: list[float] = []
    real_rows: list[list[float]] = []
    imag_rows: list[list[float]] = []
    for row in real.findall("v"):
        values = [float(value) for value in (row.text or "").split()]
        if values:
            real_rows.append(values)
    for row in imag.findall("v"):
        values = [float(value) for value in (row.text or "").split()]
        if values:
            imag_rows.append(values)
    if not real_rows or len(real_rows) != len(imag_rows):
        raise ValueError(f"dielectric spectrum malformed in {path}")
    # Energy grid is the row index in VASP's dielectric output; the grid
    # spacing is stored in the parameters, but VASP writes one row per
    # energy point in increasing order. Reconstruct a linear grid.
    parameters = root.find("parameters")
    emax = None
    npoints = None
    if parameters is not None:
        emax_text = parameters.findtext("i[@name='EMAX']")
        nedos_text = parameters.findtext("i[@name='NEDOS']")
        emax = float(emax_text) if emax_text and emax_text.strip() else None
        npoints = int(nedos_text) if nedos_text and nedos_text.strip() else None
    if emax and npoints and npoints == len(real_rows):
        energies = [emax * index / (npoints - 1) for index in range(npoints)]
    else:
        energies = list(range(len(real_rows)))
    return interpolate_dielectric(
        energies,
        real_rows,
        imag_rows,
        wavelength_um,
        source=f"VASP:{path}",
    )


def normalize_fluxes(
    incident_flux: float, reflected_flux: float, transmitted_flux: float
) -> tuple[float, float, float, float]:
    """R/T/A from normalized fluxes plus the untruncated conservation residual."""
    if abs(incident_flux) < 1e-15:
        raise ValueError("incident flux is zero; cannot normalize Meep result")
    raw_r = -reflected_flux / incident_flux
    raw_t = transmitted_flux / incident_flux
    raw_a = 1.0 - raw_r - raw_t
    reflectance = min(1.0, max(0.0, raw_r))
    transmittance = min(1.0, max(0.0, raw_t))
    absorptance = min(1.0, max(0.0, raw_a))
    residual = 1.0 - (reflectance + transmittance + absorptance)
    return reflectance, transmittance, absorptance, residual


def _validate_nk_inputs(
    *,
    wavelength_um: float,
    thickness_um: float,
    refractive_index: float,
    extinction_coefficient: float,
) -> None:
    if min(wavelength_um, thickness_um, refractive_index) <= 0:
        raise ValueError(
            "wavelength, thickness and refractive index must be positive"
        )
    if extinction_coefficient < 0:
        raise ValueError("extinction coefficient must be >= 0")


def run_meep_thinfilm(
    *,
    wavelength_um: float,
    thickness_um: float,
    refractive_index: float,
    extinction_coefficient: float = 0.0,
    resolution: int = 20,
) -> dict[str, Any]:
    """Run the 1D two-run Meep simulation; requires the ``meep`` package."""
    _validate_nk_inputs(
        wavelength_um=wavelength_um,
        thickness_um=thickness_um,
        refractive_index=refractive_index,
        extinction_coefficient=extinction_coefficient,
    )
    try:
        import meep as mp
    except ImportError as exc:
        raise RuntimeError(
            "MISSING_DEPENDENCY: the 'meep' package is not importable; "
            "install meep in an isolated environment to run the thin-film "
            "simulation"
        ) from exc
    frequency = 1.0 / wavelength_um
    pml = 0.5
    air = max(1.0, wavelength_um)
    cell_z = thickness_um + 2 * air + 2 * pml
    cell = mp.Vector3(0, 0, cell_z)
    source_z = -thickness_um / 2 - 0.7 * air
    reflection_z = -thickness_um / 2 - 0.3 * air
    transmission_z = thickness_um / 2 + 0.3 * air
    source = mp.Source(
        mp.GaussianSource(frequency, fwidth=0.2 * frequency),
        component=mp.Ex,
        center=mp.Vector3(0, 0, source_z),
    )
    boundary_layers = [mp.PML(pml)]
    reflection_region = mp.FluxRegion(center=mp.Vector3(0, 0, reflection_z))
    transmission_region = mp.FluxRegion(center=mp.Vector3(0, 0, transmission_z))
    stop = mp.stop_when_fields_decayed(
        50, mp.Ex, mp.Vector3(0, 0, transmission_z), 1e-8
    )

    reference = mp.Simulation(
        cell_size=cell,
        boundary_layers=boundary_layers,
        sources=[source],
        resolution=resolution,
        dimensions=1,
    )
    reference_reflection = reference.add_flux(frequency, 0, 1, reflection_region)
    reference.run(until_after_sources=stop)
    incident_flux = float(mp.get_fluxes(reference_reflection)[0])
    incident_data = reference.get_flux_data(reference_reflection)
    reference.reset_meep()

    epsilon_real = refractive_index**2 - extinction_coefficient**2
    epsilon_imag = 2 * refractive_index * extinction_coefficient
    if epsilon_real <= 0:
        raise ValueError(
            "the current Meep conductivity approximation requires "
            "positive epsilon_real"
        )
    conductivity = (
        2 * math.pi * frequency * epsilon_imag / epsilon_real
        if extinction_coefficient > 0
        else 0.0
    )
    material = mp.Medium(epsilon=epsilon_real, D_conductivity=conductivity)
    geometry = [
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, thickness_um), material=material)
    ]
    simulation = mp.Simulation(
        cell_size=cell,
        boundary_layers=boundary_layers,
        geometry=geometry,
        sources=[source],
        resolution=resolution,
        dimensions=1,
    )
    reflected_monitor = simulation.add_flux(frequency, 0, 1, reflection_region)
    transmitted_monitor = simulation.add_flux(frequency, 0, 1, transmission_region)
    simulation.load_minus_flux_data(reflected_monitor, incident_data)
    simulation.run(until_after_sources=stop)
    reflected_flux = float(mp.get_fluxes(reflected_monitor)[0])
    transmitted_flux = float(mp.get_fluxes(transmitted_monitor)[0])
    reflectance, transmittance, absorptance, residual = normalize_fluxes(
        incident_flux, reflected_flux, transmitted_flux
    )
    simulation.reset_meep()
    return {
        "wavelength_um": wavelength_um,
        "thickness_um": thickness_um,
        "refractive_index": refractive_index,
        "extinction_coefficient": extinction_coefficient,
        "reflectance": reflectance,
        "transmittance": transmittance,
        "absorptance": absorptance,
        "energy_conservation_residual": residual,
        "method": "1D two-run flux normalization (Meep)",
        "warnings": (
            [
                "lossless optical model (k=0): absorptance is numerical "
                "residual, not material absorption"
            ]
            if extinction_coefficient == 0
            else []
        ),
    }


class MeepThinFilmTool(Tool):
    name = "optics.meep_thinfilm"
    description = (
        "1D thin-film reflectance/transmittance/absorptance via Meep "
        "(two-run flux normalization). Inputs: n/k constants OR a VASP "
        "optics vasprun.xml (dielectric -> n/k conversion is deterministic "
        "code, never LLM-provided). Returns energy-conservation residual. "
        "SCOPE: NOT a 3D device simulation, NOT electrical transport, NOT "
        "an EQE simulation. Requires the meep package (isolated env)."
    )
    short_description = "Meep 1D thin-film R/T/A (two-run flux normalization)."
    exposure = ToolExposure.DEFERRED
    namespace = "optics"
    source = "Meep (1D thin film)"
    tags = ("optics", "meep", "thin film", "rta", "reflectance")
    cost_class = "MODERATE"
    input_schema = {
        "type": "object",
        "properties": {
            "wavelength_um": {"type": "number", "minimum": 0},
            "thickness_um": {"type": "number", "minimum": 0},
            "refractive_index": {"type": "number", "minimum": 0},
            "extinction_coefficient": {"type": "number", "minimum": 0},
            "vasprun_xml": {"type": "string"},
            "resolution": {"type": "integer", "minimum": 4, "maximum": 200},
            "output_dir": {"type": "string"},
        },
        "required": ["wavelength_um", "thickness_um"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        wavelength = float(arguments["wavelength_um"])
        thickness = float(arguments["thickness_um"])
        vasprun = arguments.get("vasprun_xml")
        has_constants = (
            arguments.get("refractive_index") is not None
        )
        if not vasprun and not has_constants:
            return ScientificToolResult(
                output=(
                    "missing prerequisite: provide either refractive_index/"
                    "extinction_coefficient or vasprun_xml (VASP optics "
                    "result)"
                ),
                is_error=True,
                data={
                    "error_type": "missing_prerequisites",
                    "missing": ["refractive_index | vasprun_xml"],
                },
            )
        optical_source = ""
        try:
            if vasprun:
                point = optical_point_from_vasprun(vasprun, wavelength)
                refractive_index = point["refractive_index"]
                extinction = point["extinction_coefficient"]
                optical_source = point["source"]
            else:
                refractive_index = float(arguments["refractive_index"])
                extinction = float(
                    arguments.get("extinction_coefficient", 0.0)
                )
                optical_source = "user-provided constants"
            result = run_meep_thinfilm(
                wavelength_um=wavelength,
                thickness_um=thickness,
                refractive_index=refractive_index,
                extinction_coefficient=extinction,
                resolution=int(arguments.get("resolution", 20)),
            )
        except Exception as exc:
            return ScientificToolResult(
                output=f"optics.meep_thinfilm failed: {type(exc).__name__}: {exc}",
                is_error=True,
                data={"error_type": type(exc).__name__, "message": str(exc)},
            )
        result["optical_constants_source"] = optical_source
        artifact = None
        output_dir = arguments.get("output_dir")
        if output_dir:
            import os

            target = Path(output_dir) / f"rta_{wavelength:g}um.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            artifact = str(target)
        evidence = [
            ScientificEvidence(
                subject=f"thin_film_{thickness:g}um",
                property="absorptance",
                value=result["absorptance"],
                unit="fraction",
                source="Meep 1D thin-film simulation",
                source_type="electromagnetic_simulation",
                method="1D two-run flux normalization",
                fidelity="electromagnetic",
                summary=(
                    f"R={result['reflectance']:.4f} T={result['transmittance']:.4f} "
                    f"A={result['absorptance']:.4f} at {wavelength:.3f} um "
                    f"({optical_source})"
                ),
                limitations=(
                    "1D thin film only; not a device or EQE simulation; "
                    "conservation residual "
                    f"{result['energy_conservation_residual']:.2e}"
                ),
                provenance={
                    "tool": self.name,
                    "wavelength_um": wavelength,
                    "thickness_um": thickness,
                    "optical_constants_source": optical_source,
                },
            )
        ]
        return ScientificToolResult(
            output=json.dumps(result, ensure_ascii=False, indent=2),
            data=result,
            evidence=evidence,
            artifacts=[artifact] if artifact else [],
        )
