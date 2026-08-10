"""Deterministic Brus / effective-mass quantum-dot solvers.

Model and constants
-------------------
Transition energy of a strongly confined spherical nanocrystal
(Brus 1984; Coulomb prefactor 1.786 per Kayanuma 1988):

    E(R) = E_g + (h^2 / (8 m0 R^2)) * (1/m_e* + 1/m_h*) - 1.786 e^2 / (4 pi eps0 eps_r R)

with R the radius in meters, masses in units of m0, E in eV. The Coulomb
term is optional. Exciton Bohr radius (diagnostic):

    a_B* = 4 pi eps0 eps_r hbar^2 / (e^2 mu),   mu = m0 / (1/m_e* + 1/m_h*)

Validity (all returned with every result)
-----------------------------------------
* spherical approximation
* single-band effective-mass approximation, parabolic bands
* idealized infinite confining barrier
* surface chemistry, ligands, and finite barrier effects ignored
* multiband / spin-orbit coupling effects ignored

For narrow-gap / inverted systems (e.g. HgTe, PbTe with strong SOC) the EMA
is suspect; tools emit strong warnings and label results ``fidelity=L1``.
"""

from __future__ import annotations

import math
from typing import Any

from photomatagent.scientific.errors import MissingScientificPrerequisite

HBAR_EV_S = 6.582119569e-16  # hbar in eV*s
E_C = 1.602176634e-19  # elementary charge in C
EPS0 = 8.8541878128e-12  # vacuum permittivity F/m
M0_KG = 9.1093837015e-31  # electron rest mass in kg
HC_EV_UM = 1.239841984  # h*c in eV*um
KAYANUMA_C = 1.786  # Coulomb correction prefactor (Kayanuma 1988)
PI_SQ = math.pi**2

MIN_RADIUS_NM = 0.3  # below this the atomistic limit is hit; EMA invalid
MAX_RADIUS_NM = 200.0


def _confinement_prefactor_eV_m2(mass_sum_inv: float) -> float:
    """A in E(R) = E_g + A/R^2 - B/R with R in meters, E in eV."""
    # hbar^2/(2 m0) in J*m^2, converted to eV*m^2, times pi^2 and (1/me+1/mh).
    return HBAR_EV_S**2 * E_C * PI_SQ / (2.0 * M0_KG) * mass_sum_inv


def _coulomb_prefactor_eV_m(relative_dielectric: float) -> float:
    """B in E(R) = E_g + A/R^2 - B/R with R in meters, E in eV."""
    return KAYANUMA_C * E_C / (4.0 * math.pi * EPS0 * relative_dielectric)


def transition_energy(
    *,
    radius_nm: float,
    bulk_band_gap_eV: float,
    electron_effective_mass_m0: float,
    hole_effective_mass_m0: float,
    relative_dielectric_constant: float | None = None,
    include_coulomb_term: bool = True,
) -> dict[str, Any]:
    """Compute the Brus transition energy; returns a structured result dict."""
    _validate_inputs(
        radius_nm=radius_nm,
        electron_effective_mass_m0=electron_effective_mass_m0,
        hole_effective_mass_m0=hole_effective_mass_m0,
    )
    radius_m = radius_nm * 1e-9
    mass_sum_inv = 1.0 / electron_effective_mass_m0 + 1.0 / hole_effective_mass_m0
    confinement_shift = _confinement_prefactor_eV_m2(mass_sum_inv) / radius_m**2
    coulomb = 0.0
    if include_coulomb_term:
        if relative_dielectric_constant is None:
            raise MissingScientificPrerequisite(
                "Coulomb term requested but relative_dielectric_constant is missing",
                missing=["relative_dielectric_constant"],
            )
        coulomb = _coulomb_prefactor_eV_m(relative_dielectric_constant) / radius_m
    transition = bulk_band_gap_eV + confinement_shift - coulomb
    wavelength_um = _wavelength_from_energy(transition)
    warnings = _validity_warnings(
        bulk_band_gap_eV=bulk_band_gap_eV,
        electron_effective_mass_m0=electron_effective_mass_m0,
        radius_nm=radius_nm,
    )
    return {
        "transition_energy_eV": transition,
        "transition_wavelength_um": wavelength_um,
        "confinement_shift_eV": confinement_shift,
        "coulomb_correction_eV": coulomb,
        "coulomb_correction_sign_note": (
            "positive value: the Coulomb term lowers the transition energy "
            "by this amount"
        ),
        "input_parameters": {
            "radius_nm": radius_nm,
            "bulk_band_gap_eV": bulk_band_gap_eV,
            "electron_effective_mass_m0": electron_effective_mass_m0,
            "hole_effective_mass_m0": hole_effective_mass_m0,
            "relative_dielectric_constant": relative_dielectric_constant,
            "include_coulomb_term": include_coulomb_term,
        },
        "fidelity_level": "L1",
        "fidelity": "analytical",
        "method": "Brus effective-mass model (Brus 1984; Kayanuma 1988)",
        "assumptions": [
            "spherical nanocrystal",
            "single-band effective-mass approximation, parabolic bands",
            "infinite confining barrier",
            "surface chemistry / ligands ignored",
            "multiband and spin-orbit effects ignored",
        ],
        "warnings": warnings,
    }


def exciton_bohr_radius_nm(
    *,
    electron_effective_mass_m0: float,
    hole_effective_mass_m0: float,
    relative_dielectric_constant: float,
) -> float:
    """Exciton Bohr radius a_B* in nm (effective-mass definition)."""
    mass_sum_inv = 1.0 / electron_effective_mass_m0 + 1.0 / hole_effective_mass_m0
    radius_m = (
        4.0
        * math.pi
        * EPS0
        * relative_dielectric_constant
        * HBAR_EV_S**2
        * mass_sum_inv
        / M0_KG
    )
    return radius_m * 1e9


def excitonic_regime(
    *,
    radius_nm: float,
    electron_effective_mass_m0: float,
    hole_effective_mass_m0: float,
    relative_dielectric_constant: float,
) -> dict[str, Any]:
    """Diagnose strong/weak confinement from R/a_B*."""
    a_star = exciton_bohr_radius_nm(
        electron_effective_mass_m0=electron_effective_mass_m0,
        hole_effective_mass_m0=hole_effective_mass_m0,
        relative_dielectric_constant=relative_dielectric_constant,
    )
    ratio = radius_nm / a_star
    if ratio < 0.5:
        regime = "strong"
    elif ratio < 2.0:
        regime = "intermediate"
    else:
        regime = "weak"
    return {
        "exciton_bohr_radius_nm": a_star,
        "radius_over_bohr_radius": ratio,
        "confinement_regime": regime,
        "diagnostic": (
            "strong confinement assumed by the Brus model is "
            f"{'supported' if regime == 'strong' else 'questionable'} "
            f"(R/a_B* = {ratio:.2f})"
        ),
    }


def solve_size_for_transition(
    *,
    target_energy_eV: float | None = None,
    target_wavelength_um: float | None = None,
    bulk_band_gap_eV: float,
    electron_effective_mass_m0: float,
    hole_effective_mass_m0: float,
    relative_dielectric_constant: float | None = None,
    include_coulomb_term: bool = True,
    radius_min_nm: float = MIN_RADIUS_NM,
) -> dict[str, Any]:
    """Invert the Brus model for the radius at a target transition energy.

    With the Coulomb term E(R) has a minimum at R* = 2A/B below the bulk gap;
    the solver returns the unique root on the strong-confinement branch
    (0, R*]. No LLM guessing: bisection on the monotonic branch.
    """
    if (target_energy_eV is None) == (target_wavelength_um is None):
        raise ValueError("provide exactly one of target_energy_eV / target_wavelength_um")
    energy = (
        target_energy_eV
        if target_energy_eV is not None
        else _energy_from_wavelength(target_wavelength_um)  # type: ignore[arg-type]
    )
    mass_sum_inv = 1.0 / electron_effective_mass_m0 + 1.0 / hole_effective_mass_m0
    a = _confinement_prefactor_eV_m2(mass_sum_inv)
    b = 0.0
    if include_coulomb_term:
        if relative_dielectric_constant is None:
            raise MissingScientificPrerequisite(
                "Coulomb term requested but relative_dielectric_constant is missing",
                missing=["relative_dielectric_constant"],
            )
        b = _coulomb_prefactor_eV_m(relative_dielectric_constant)
    if energy < bulk_band_gap_eV:
        return {
            "outcome": "NO_PHYSICAL_SOLUTION",
            "reason": (
                f"target energy {energy:.4f} eV is below the bulk band gap "
                f"{bulk_band_gap_eV:.4f} eV; Brus confinement only blueshifts "
                "above the bulk gap within its validity range"
            ),
            "input_parameters": _input_params(
                bulk_band_gap_eV=bulk_band_gap_eV,
                electron_effective_mass_m0=electron_effective_mass_m0,
                hole_effective_mass_m0=hole_effective_mass_m0,
                relative_dielectric_constant=relative_dielectric_constant,
                include_coulomb_term=include_coulomb_term,
                target_energy_eV=energy,
            ),
            "fidelity_level": "L1",
            "fidelity": "analytical",
            "warnings": _validity_warnings(
                bulk_band_gap_eV=bulk_band_gap_eV,
                electron_effective_mass_m0=electron_effective_mass_m0,
            ),
        }
    if b == 0.0 and energy <= bulk_band_gap_eV:
        return {
            "outcome": "NO_PHYSICAL_SOLUTION",
            "reason": (
                "target energy equals the bulk band gap: without the Coulomb "
                "term this requires an infinite radius"
            ),
            "input_parameters": _input_params(
                bulk_band_gap_eV=bulk_band_gap_eV,
                electron_effective_mass_m0=electron_effective_mass_m0,
                hole_effective_mass_m0=hole_effective_mass_m0,
                relative_dielectric_constant=relative_dielectric_constant,
                include_coulomb_term=include_coulomb_term,
                target_energy_eV=energy,
            ),
            "fidelity_level": "L1",
            "fidelity": "analytical",
            "warnings": _validity_warnings(
                bulk_band_gap_eV=bulk_band_gap_eV,
                electron_effective_mass_m0=electron_effective_mass_m0,
            ),
        }
    if b > 0:
        r_star = 2.0 * a / b
        r_hi = min(r_star * 0.999, math.sqrt(a / max(energy - bulk_band_gap_eV, 1e-12)) * 1.5)
    else:
        r_hi = math.sqrt(a / max(energy - bulk_band_gap_eV, 1e-12)) * 1.5
    r_lo = radius_min_nm * 1e-9
    r_hi = max(r_hi, r_lo * 2)
    r_hi = min(r_hi, MAX_RADIUS_NM * 1e-9)

    def e_of(r: float) -> float:
        return bulk_band_gap_eV + a / r**2 - b / r

    if e_of(r_lo) <= energy:
        return {
            "outcome": "NO_PHYSICAL_SOLUTION",
            "reason": (
                f"no root on the strong-confinement branch above "
                f"{radius_min_nm} nm (E({radius_min_nm} nm) = {e_of(r_lo):.4f} eV "
                f"<= target {energy:.4f} eV)"
            ),
            "input_parameters": _input_params(
                bulk_band_gap_eV=bulk_band_gap_eV,
                electron_effective_mass_m0=electron_effective_mass_m0,
                hole_effective_mass_m0=hole_effective_mass_m0,
                relative_dielectric_constant=relative_dielectric_constant,
                include_coulomb_term=include_coulomb_term,
                target_energy_eV=energy,
            ),
            "fidelity_level": "L1",
            "fidelity": "analytical",
            "warnings": _validity_warnings(
                bulk_band_gap_eV=bulk_band_gap_eV,
                electron_effective_mass_m0=electron_effective_mass_m0,
            ),
        }
    # Bisection on (r_lo, r_hi), strictly decreasing branch.
    for _ in range(300):
        r_mid = 0.5 * (r_lo + r_hi)
        if e_of(r_mid) > energy:
            r_lo = r_mid
        else:
            r_hi = r_mid
    radius = 0.5 * (r_lo + r_hi)
    forward = transition_energy(
        radius_nm=radius * 1e9,
        bulk_band_gap_eV=bulk_band_gap_eV,
        electron_effective_mass_m0=electron_effective_mass_m0,
        hole_effective_mass_m0=hole_effective_mass_m0,
        relative_dielectric_constant=relative_dielectric_constant,
        include_coulomb_term=include_coulomb_term,
    )
    return {
        "outcome": "SOLVED",
        "candidate_radius_nm": radius * 1e9,
        "candidate_diameter_nm": radius * 2e9,
        "predicted_transition_energy_eV": forward["transition_energy_eV"],
        "predicted_transition_wavelength_um": forward["transition_wavelength_um"],
        "residual_eV": abs(forward["transition_energy_eV"] - energy),
        "target_energy_eV": energy,
        "input_parameters": _input_params(
            bulk_band_gap_eV=bulk_band_gap_eV,
            electron_effective_mass_m0=electron_effective_mass_m0,
            hole_effective_mass_m0=hole_effective_mass_m0,
            relative_dielectric_constant=relative_dielectric_constant,
            include_coulomb_term=include_coulomb_term,
            target_energy_eV=energy,
        ),
        "fidelity_level": "L1",
        "fidelity": "analytical",
        "method": "bisection on monotonic strong-confinement branch of Brus model",
        "assumptions": [
            "spherical nanocrystal",
            "single-band effective-mass approximation, parabolic bands",
            "infinite confining barrier",
            "surface chemistry / ligands ignored",
            "multiband and spin-orbit effects ignored",
        ],
        "warnings": forward["warnings"],
    }


def size_sweep(
    *,
    min_size_nm: float,
    max_size_nm: float,
    points: int,
    bulk_band_gap_eV: float,
    electron_effective_mass_m0: float,
    hole_effective_mass_m0: float,
    relative_dielectric_constant: float | None = None,
    include_coulomb_term: bool = True,
) -> dict[str, Any]:
    """Bounded size sweep; large tables are summarized and persisted to disk."""
    if points < 2:
        raise ValueError("points must be >= 2")
    radii = [
        min_size_nm + (max_size_nm - min_size_nm) * i / (points - 1)
        for i in range(points)
    ]
    rows: list[dict[str, float]] = []
    for radius in radii:
        result = transition_energy(
            radius_nm=radius,
            bulk_band_gap_eV=bulk_band_gap_eV,
            electron_effective_mass_m0=electron_effective_mass_m0,
            hole_effective_mass_m0=hole_effective_mass_m0,
            relative_dielectric_constant=relative_dielectric_constant,
            include_coulomb_term=include_coulomb_term,
        )
        rows.append(
            {
                "size_nm": round(radius, 4),
                "transition_energy_eV": round(result["transition_energy_eV"], 5),
                "transition_wavelength_um": round(
                    result["transition_wavelength_um"], 5
                ),
            }
        )
    max_rows = 100
    selected = rows if len(rows) <= max_rows else rows[:: max(1, len(rows) // max_rows)]
    wavelengths = [row["transition_wavelength_um"] for row in rows]
    energies = [row["transition_energy_eV"] for row in rows]
    return {
        "count": len(rows),
        "selected_rows": selected,
        "summary": {
            "wavelength_min_um": min(wavelengths),
            "wavelength_max_um": max(wavelengths),
            "energy_min_eV": min(energies),
            "energy_max_eV": max(energies),
        },
        "note": (
            f"showing {len(selected)} of {len(rows)} rows; full table available "
            "on request with points <= 100"
        ),
        "fidelity_level": "L1",
        "fidelity": "analytical",
        "warnings": _validity_warnings(
            bulk_band_gap_eV=bulk_band_gap_eV,
            electron_effective_mass_m0=electron_effective_mass_m0,
        ),
    }


def _energy_from_wavelength(wavelength_um: float) -> float:
    if wavelength_um <= 0:
        raise ValueError("wavelength must be positive")
    return HC_EV_UM / wavelength_um


def _wavelength_from_energy(energy_eV: float) -> float:
    if energy_eV <= 0:
        raise ValueError("transition energy must be positive")
    return HC_EV_UM / energy_eV


def _input_params(**kwargs: Any) -> dict[str, Any]:
    return dict(kwargs)


def _validate_inputs(
    *,
    radius_nm: float | None = None,
    electron_effective_mass_m0: float | None = None,
    hole_effective_mass_m0: float | None = None,
) -> None:
    if radius_nm is not None and radius_nm <= 0:
        raise ValueError(f"radius_nm must be positive, got {radius_nm}")
    if radius_nm is not None and radius_nm < MIN_RADIUS_NM:
        raise ValueError(
            f"radius_nm {radius_nm} is below the EMA validity floor "
            f"({MIN_RADIUS_NM} nm)"
        )
    if electron_effective_mass_m0 is not None and electron_effective_mass_m0 <= 0:
        raise ValueError(
            "electron_effective_mass_m0 must be positive, got "
            f"{electron_effective_mass_m0}"
        )
    if hole_effective_mass_m0 is not None and hole_effective_mass_m0 <= 0:
        raise ValueError(
            f"hole_effective_mass_m0 must be positive, got {hole_effective_mass_m0}"
        )


def _validity_warnings(
    *,
    bulk_band_gap_eV: float,
    electron_effective_mass_m0: float,
    radius_nm: float | None = None,
) -> list[str]:
    warnings: list[str] = []
    if bulk_band_gap_eV <= 0.15:
        warnings.append(
            "narrow-gap/inverted-band system (bulk gap <= 0.15 eV): single-band "
            "effective-mass approximation is suspect; strong spin-orbit and "
            "multiband effects are ignored by Brus"
        )
    if bulk_band_gap_eV < 0:
        warnings.append(
            "bulk gap is negative (semimetal): the Brus result is not a "
            "physical confinement blueshift in the usual sense; treat as "
            "illustrative only and use higher-fidelity electronic structure"
        )
    if electron_effective_mass_m0 < 0.02:
        warnings.append(
            "very light electron mass: parabolic-band EMA is least reliable "
            "in this regime"
        )
    if radius_nm is not None and radius_nm < MIN_RADIUS_NM:
        warnings.append(
            f"radius {radius_nm:.2f} nm is below the EMA validity floor "
            f"({MIN_RADIUS_NM} nm); atomistic effects dominate"
        )
    if radius_nm is not None and radius_nm > 50.0:
        warnings.append(
            "radius > 50 nm: confinement shift is negligible; bulk behavior "
            "dominates and measurement/interface effects matter more"
        )
    return warnings
