"""Composition x size screening for a target transition window (L1).

Combines the generic bowing model (``alloy.bandgap_bowing``) with the Brus
confinement model (``brus.transition_energy``). Endpoint masses are
interpolated linearly with composition. If any required parameter is
missing, the tool returns a typed ``missing_prerequisites`` failure -- it
never invents default values.
"""

from __future__ import annotations

from typing import Any

from photomatagent.scientific.capabilities.quantum_dot.alloy import bandgap_bowing
from photomatagent.scientific.capabilities.quantum_dot.brus import (
    HC_EV_UM,
    transition_energy,
)
from photomatagent.scientific.errors import MissingScientificPrerequisite


def screen_size_composition(
    *,
    target_wavelength_min_um: float,
    target_wavelength_max_um: float,
    composition_min: float,
    composition_max: float,
    composition_points: int,
    radius_min_nm: float,
    radius_max_nm: float,
    radius_points: int,
    band_gap_a_eV: float,
    band_gap_b_eV: float,
    bowing_parameter_eV: float,
    electron_mass_a_m0: float,
    hole_mass_a_m0: float,
    electron_mass_b_m0: float,
    hole_mass_b_m0: float,
    relative_dielectric_constant: float,
    include_coulomb_term: bool = True,
) -> dict[str, Any]:
    """Grid-search composition/size pairs whose L1 transition falls in window."""
    required: list[str] = []
    named = {
        "target_wavelength_min_um": target_wavelength_min_um,
        "target_wavelength_max_um": target_wavelength_max_um,
        "composition_min": composition_min,
        "composition_max": composition_max,
        "radius_min_nm": radius_min_nm,
        "radius_max_nm": radius_max_nm,
        "band_gap_a_eV": band_gap_a_eV,
        "band_gap_b_eV": band_gap_b_eV,
        "bowing_parameter_eV": bowing_parameter_eV,
        "electron_mass_a_m0": electron_mass_a_m0,
        "hole_mass_a_m0": hole_mass_a_m0,
        "electron_mass_b_m0": electron_mass_b_m0,
        "hole_mass_b_m0": hole_mass_b_m0,
        "relative_dielectric_constant": relative_dielectric_constant,
    }
    for name, value in named.items():
        if value is None:
            required.append(name)
    if required:
        raise MissingScientificPrerequisite(
            "screen_size_composition requires complete material parameters",
            missing=required,
        )
    if target_wavelength_min_um <= 0 or target_wavelength_max_um <= 0:
        raise ValueError("target wavelength bounds must be positive")
    if target_wavelength_min_um > target_wavelength_max_um:
        raise ValueError("target_wavelength_min_um must be <= target_wavelength_max_um")
    if not 0.0 <= composition_min <= composition_max <= 1.0:
        raise ValueError("composition bounds must satisfy 0 <= min <= max <= 1")
    if radius_min_nm <= 0 or radius_max_nm <= radius_min_nm:
        raise ValueError("invalid radius bounds")

    x_grid = [
        composition_min
        + (composition_max - composition_min) * i / max(composition_points - 1, 1)
        for i in range(composition_points)
    ]
    r_grid = [
        radius_min_nm + (radius_max_nm - radius_min_nm) * i / max(radius_points - 1, 1)
        for i in range(radius_points)
    ]
    energy_min = HC_EV_UM / target_wavelength_max_um
    energy_max = HC_EV_UM / target_wavelength_min_um
    candidates: list[dict[str, Any]] = []
    center = HC_EV_UM / (0.5 * (target_wavelength_min_um + target_wavelength_max_um))
    for x in x_grid:
        bowing = bandgap_bowing(
            x=x,
            band_gap_a_eV=band_gap_a_eV,
            band_gap_b_eV=band_gap_b_eV,
            bowing_parameter_eV=bowing_parameter_eV,
        )
        gap = bowing["band_gap_eV"]
        me = (1.0 - x) * electron_mass_a_m0 + x * electron_mass_b_m0
        mh = (1.0 - x) * hole_mass_a_m0 + x * hole_mass_b_m0
        for radius in r_grid:
            result = transition_energy(
                radius_nm=radius,
                bulk_band_gap_eV=gap,
                electron_effective_mass_m0=me,
                hole_effective_mass_m0=mh,
                relative_dielectric_constant=relative_dielectric_constant,
                include_coulomb_term=include_coulomb_term,
            )
            energy = result["transition_energy_eV"]
            if energy_min <= energy <= energy_max:
                candidates.append(
                    {
                        "composition_x": round(x, 4),
                        "radius_nm": round(radius, 3),
                        "transition_energy_eV": round(energy, 5),
                        "transition_wavelength_um": round(
                            result["transition_wavelength_um"], 5
                        ),
                        "band_gap_eV": round(gap, 5),
                    }
                )
    candidates.sort(
        key=lambda row: abs(row["transition_energy_eV"] - center)
    )
    selected = candidates[:20]
    return {
        "target_window": {
            "wavelength_min_um": target_wavelength_min_um,
            "wavelength_max_um": target_wavelength_max_um,
            "energy_min_eV": energy_min,
            "energy_max_eV": energy_max,
        },
        "grid": {
            "composition_points": composition_points,
            "radius_points": radius_points,
            "total_evaluated": composition_points * radius_points,
        },
        "match_count": len(candidates),
        "match_fraction": (
            len(candidates) / max(composition_points * radius_points, 1)
        ),
        "selected_candidates": selected,
        "fidelity_level": "L1",
        "fidelity": "analytical",
        "method": (
            "quadratic bowing for Eg(x) + Brus effective-mass confinement; "
            "linear endpoint-mass interpolation"
        ),
        "assumptions": [
            "spherical QD, infinite barrier EMA",
            "linear interpolation of effective masses with composition",
            "single scalar dielectric constant",
            "no band-offset/interface or multiband effects",
        ],
        "warnings": (
            [
                "narrow-gap endpoint: EMA/Brus fidelity is low; verify any "
                "candidate with higher-fidelity electronic structure"
            ]
            if min(band_gap_a_eV, band_gap_b_eV) <= 0.15
            else []
        ),
        "note": (
            "candidates are L1 analytical estimates only; composition and size "
            "must be validated against higher-fidelity solvers before design"
        ),
    }
