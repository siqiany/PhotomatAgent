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

Dielectric parameter typing (Sprint 3)
--------------------------------------
The Brus Coulomb term and the exciton Bohr radius screen the electron-hole
Coulomb attraction with ``epsilon_r``. That interaction is a low-frequency
process: lattice (ionic) polarization participates, so the *static*
dielectric constant is the physically appropriate input. These solvers
therefore declare ``relative_dielectric_kind`` of ``static`` (recommended)
or ``unknown`` as acceptable; ``optical`` / ``high_frequency`` values are
rejected with a typed ``INCOMPATIBLE_SCIENTIFIC_PARAMETER`` diagnostic
instead of being silently used. A bare float without a kind is treated as
``unknown`` and flagged in the result (never silently claimed to be static).
"""

from __future__ import annotations

import math
from typing import Any

from photomatagent.scientific.errors import MissingScientificPrerequisite
from photomatagent.scientific.errors import UnsupportedScientificRegime

HBAR_EV_S = 6.582119569e-16  # hbar in eV*s
E_C = 1.602176634e-19  # elementary charge in C
EPS0 = 8.8541878128e-12  # vacuum permittivity F/m
M0_KG = 9.1093837015e-31  # electron rest mass in kg
HC_EV_UM = 1.239841984  # h*c in eV*um
KAYANUMA_C = 1.786  # Coulomb correction prefactor (Kayanuma 1988)
PI_SQ = math.pi**2

MIN_RADIUS_NM = 0.3  # below this the atomistic limit is hit; EMA invalid
MAX_RADIUS_NM = 200.0

# Physical regime accepted by the Brus/exciton screening term.
ACCEPTED_DIELECTRIC_KINDS = ("static", "unknown")
REJECTED_DIELECTRIC_KINDS = ("optical", "high_frequency")


def normalize_dielectric_kind(kind: str | None) -> str:
    """Normalize a user-supplied dielectric kind to the model vocabulary."""
    if kind is None:
        return "unknown"
    normalized = str(kind).strip().lower().replace("-", "_")
    if normalized in {"optic"}:
        return "optical"
    if normalized in {"hf", "high_frequency_optical"}:
        return "high_frequency"
    if normalized in {"static", "optical", "high_frequency", "unknown"}:
        return normalized
    return "unknown"


def validate_dielectric_kind(kind: str | None) -> str:
    """Return the normalized kind or raise a typed incompatibility error.

    The Brus/exciton Coulomb screening term is a low-frequency (static)
    screening process; optical / high-frequency dielectric constants omit
    the lattice polarization that actually screens the electron-hole
    attraction and must not be silently substituted.
    """
    normalized = normalize_dielectric_kind(kind)
    if normalized in REJECTED_DIELECTRIC_KINDS:
        raise UnsupportedScientificRegime(
            "INCOMPATIBLE_SCIENTIFIC_PARAMETER: "
            f"relative_dielectric_kind={normalized!r} is not accepted by the "
            "Brus/exciton Coulomb term, which requires static screening of "
            "the electron-hole attraction (lattice polarization included). "
            "Accepted kinds: static (recommended) or unknown. Provide the "
            "static dielectric constant, or label the value as "
            "kind=unknown if its regime is not established."
        )
    return normalized


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
    relative_dielectric_kind: str | None = None,
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
    dielectric_kind = "unknown"
    if include_coulomb_term:
        if relative_dielectric_constant is None:
            raise MissingScientificPrerequisite(
                "Coulomb term requested but relative_dielectric_constant is missing",
                missing=["relative_dielectric_constant"],
            )
        dielectric_kind = validate_dielectric_kind(relative_dielectric_kind)
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
            "relative_dielectric_kind": dielectric_kind,
            "include_coulomb_term": include_coulomb_term,
        },
        "dielectric_kind_note": (
            f"relative_dielectric_kind = {dielectric_kind!r}: the Brus "
            "Coulomb term assumes static (low-frequency) screening; an "
            "'unknown' kind means the screening regime was not declared and "
            "the value was used without a regime claim"
            if include_coulomb_term
            else "Coulomb term disabled; dielectric kind not used"
        ),
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
    relative_dielectric_kind: str | None = None,
) -> float:
    """Exciton Bohr radius a_B* in nm (effective-mass definition)."""
    validate_dielectric_kind(relative_dielectric_kind)
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
    relative_dielectric_kind: str | None = None,
) -> dict[str, Any]:
    """Diagnose strong/weak confinement from R/a_B*."""
    dielectric_kind = validate_dielectric_kind(relative_dielectric_kind)
    a_star = exciton_bohr_radius_nm(
        electron_effective_mass_m0=electron_effective_mass_m0,
        hole_effective_mass_m0=hole_effective_mass_m0,
        relative_dielectric_constant=relative_dielectric_constant,
        relative_dielectric_kind=dielectric_kind,
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
        "relative_dielectric_kind": dielectric_kind,
        "dielectric_kind_note": (
            f"exciton screening uses relative_dielectric_kind = "
            f"{dielectric_kind!r}; static is the physically appropriate "
            "regime for the exciton binding"
        ),
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
    relative_dielectric_kind: str | None = None,
    include_coulomb_term: bool = True,
    radius_min_nm: float = MIN_RADIUS_NM,
) -> dict[str, Any]:
    """Invert the Brus model for the radius at a target transition energy.

    Sprint 3 branch analysis (E(R) = Eg + A/R^2 - B/R with A > 0, B >= 0):

    * B = 0: E(R) = Eg + A/R^2 is strictly decreasing; targets at or below
      the bulk gap have NO_MATHEMATICAL_SOLUTION (infinite radius), targets
      above the gap have exactly one root.
    * B > 0: E(R) falls to a minimum E_min = Eg - B^2/(4A) at R* = 2A/B and
      rises back to Eg as R -> inf. Therefore:
        - target < E_min                 -> NO_MATHEMATICAL_SOLUTION
        - target == E_min                -> degenerate root at R* (SOLVED,
                                             branch flagged degenerate)
        - E_min < target < Eg            -> TWO mathematical roots (one on
                                             each side of R*): AMBIGUOUS_BRANCH
                                             if both are inside the model's
                                             validity, SOLVED on the valid
                                             branch otherwise, or
                                             OUTSIDE_MODEL_VALIDITY when a
                                             root exists but the EMA / strong
                                             confinement assumptions do not
                                             support it
        - target >= Eg                   -> exactly one root on (0, R*]

    Mathematical roots are always reported (``mathematical_roots``); the
    ``outcome`` distinguishes SOLVED / NO_MATHEMATICAL_SOLUTION /
    OUTSIDE_MODEL_VALIDITY / AMBIGUOUS_BRANCH. No LLM guessing: roots come
    from bisection on the monotonic segments of E(R).
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
        dielectric_kind = validate_dielectric_kind(relative_dielectric_kind)
        b = _coulomb_prefactor_eV_m(relative_dielectric_constant)
    else:
        dielectric_kind = "unknown"

    def e_of(r: float) -> float:
        return bulk_band_gap_eV + a / r**2 - b / r

    def bisect(lo: float, hi: float, target: float) -> float:
        """Bisection on a monotonic segment containing exactly one root.

        Works for both decreasing (strong-confinement) and increasing
        (weak-confinement) segments: the bracket is shrunk toward whichever
        side shares the sign of (E - target) with the opposite endpoint.
        """
        f_lo = e_of(lo) - target
        f_hi = e_of(hi) - target
        if f_lo * f_hi > 0:
            raise ValueError("bisection bracket does not enclose a root")
        for _ in range(400):
            mid = 0.5 * (lo + hi)
            f_mid = e_of(mid) - target
            if f_mid * f_lo > 0:
                lo, f_lo = mid, f_mid
            else:
                hi, f_hi = mid, f_mid
        return 0.5 * (lo + hi)

    def params(**extra: Any) -> dict[str, Any]:
        return _input_params(
            bulk_band_gap_eV=bulk_band_gap_eV,
            electron_effective_mass_m0=electron_effective_mass_m0,
            hole_effective_mass_m0=hole_effective_mass_m0,
            relative_dielectric_constant=relative_dielectric_constant,
            relative_dielectric_kind=dielectric_kind,
            include_coulomb_term=include_coulomb_term,
            target_energy_eV=energy,
            **extra,
        )

    a_star_nm: float | None = None
    if b > 0:
        assert relative_dielectric_constant is not None  # guaranteed above
        a_star_nm = exciton_bohr_radius_nm(
            electron_effective_mass_m0=electron_effective_mass_m0,
            hole_effective_mass_m0=hole_effective_mass_m0,
            relative_dielectric_constant=relative_dielectric_constant,
            relative_dielectric_kind=dielectric_kind,
        )

    def branch_valid(radius_m: float, *, label: str) -> tuple[bool, list[str]]:
        """Check whether a mathematical root is inside model validity."""
        reasons: list[str] = []
        radius_nm = radius_m * 1e9
        if radius_nm < radius_min_nm:
            reasons.append(
                f"{label} root {radius_nm:.2f} nm is below the EMA atomistic "
                f"limit {radius_min_nm} nm"
            )
        if radius_nm > MAX_RADIUS_NM:
            reasons.append(
                f"{label} root {radius_nm:.1f} nm exceeds the model ceiling "
                f"{MAX_RADIUS_NM} nm"
            )
        if a_star_nm is not None:
            ratio = radius_nm / a_star_nm
            if ratio > 2.0:
                reasons.append(
                    f"{label} root has R/a_B* = {ratio:.2f} > 2: the "
                    "strong/intermediate confinement regime assumed by the "
                    "Brus EMA does not hold"
                )
        return (not reasons, reasons)

    def solved_payload(
        radius_m: float,
        *,
        branch_label: str,
        extra: dict[str, Any],
        rejected: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        forward = transition_energy(
            radius_nm=radius_m * 1e9,
            bulk_band_gap_eV=bulk_band_gap_eV,
            electron_effective_mass_m0=electron_effective_mass_m0,
            hole_effective_mass_m0=hole_effective_mass_m0,
            relative_dielectric_constant=relative_dielectric_constant,
            relative_dielectric_kind=dielectric_kind,
            include_coulomb_term=include_coulomb_term,
        )
        return {
            "outcome": "SOLVED",
            "branch": branch_label,
            "candidate_radius_nm": radius_m * 1e9,
            "candidate_diameter_nm": radius_m * 2e9,
            "predicted_transition_energy_eV": forward["transition_energy_eV"],
            "predicted_transition_wavelength_um": forward[
                "transition_wavelength_um"
            ],
            "residual_eV": abs(forward["transition_energy_eV"] - energy),
            "target_energy_eV": energy,
            "input_parameters": params(),
            "fidelity_level": "L1",
            "fidelity": "analytical",
            "method": "bisection on monotonic segments of the Brus model",
            "assumptions": [
                "spherical nanocrystal",
                "single-band effective-mass approximation, parabolic bands",
                "infinite confining barrier",
                "surface chemistry / ligands ignored",
                "multiband and spin-orbit effects ignored",
            ],
            "warnings": forward["warnings"],
            **extra,
            **({"rejected_mathematical_roots": rejected} if rejected else {}),
        }

    # ------------------------------------------------------------------
    # Branch analysis
    # ------------------------------------------------------------------
    if b == 0.0:
        # E(R) = Eg + A/R^2, strictly decreasing.
        if energy <= bulk_band_gap_eV:
            return {
                "outcome": "NO_MATHEMATICAL_SOLUTION",
                "reason": (
                    f"target energy {energy:.4f} eV is at or below the bulk "
                    f"band gap {bulk_band_gap_eV:.4f} eV; without the Coulomb "
                    "term E(R) = Eg + A/R^2 >= Eg for all finite R, so no "
                    "finite radius can reach the target"
                ),
                "input_parameters": params(),
                "fidelity_level": "L1",
                "fidelity": "analytical",
                "warnings": _validity_warnings(
                    bulk_band_gap_eV=bulk_band_gap_eV,
                    electron_effective_mass_m0=electron_effective_mass_m0,
                ),
            }
        r_lo = radius_min_nm * 1e-9
        r_hi = min(
            math.sqrt(a / (energy - bulk_band_gap_eV)) * 1.5,
            MAX_RADIUS_NM * 1e-9,
        )
        r_hi = max(r_hi, r_lo * 2)
        radius = bisect(r_lo, r_hi, energy)
        valid, reasons = branch_valid(radius, label="unique")
        if not valid:
            return {
                "outcome": "OUTSIDE_MODEL_VALIDITY",
                "reason": (
                    "a mathematical root exists but is outside the model "
                    "validity: " + "; ".join(reasons)
                ),
                "mathematical_roots": [
                    {
                        "radius_nm": round(radius * 1e9, 4),
                        "energy_eV": round(e_of(radius), 6),
                        "valid": False,
                        "reasons": reasons,
                    }
                ],
                "input_parameters": params(),
                "fidelity_level": "L1",
                "fidelity": "analytical",
                "warnings": _validity_warnings(
                    bulk_band_gap_eV=bulk_band_gap_eV,
                    electron_effective_mass_m0=electron_effective_mass_m0,
                ),
            }
        return solved_payload(
            radius, branch_label="unique", extra={}
        )

    # b > 0: E(R) has a minimum at R* = 2A/B.
    r_star = 2.0 * a / b
    e_min = bulk_band_gap_eV - b**2 / (4.0 * a)
    if energy < e_min - 1e-12:
        return {
            "outcome": "NO_MATHEMATICAL_SOLUTION",
            "reason": (
                f"target energy {energy:.6f} eV is below the Coulomb "
                f"minimum E_min = {e_min:.6f} eV of the Brus model; E(R) "
                "cannot be lowered that far by any radius"
            ),
            "input_parameters": params(),
            "fidelity_level": "L1",
            "fidelity": "analytical",
            "warnings": _validity_warnings(
                bulk_band_gap_eV=bulk_band_gap_eV,
                electron_effective_mass_m0=electron_effective_mass_m0,
            ),
        }
    if abs(energy - e_min) <= 1e-12:
        valid, reasons = branch_valid(r_star, label="degenerate")
        if not valid:
            return {
                "outcome": "OUTSIDE_MODEL_VALIDITY",
                "reason": (
                    "the degenerate root at the Coulomb minimum R* exists "
                    "but is outside model validity: " + "; ".join(reasons)
                ),
                "mathematical_roots": [
                    {
                        "radius_nm": round(r_star * 1e9, 4),
                        "energy_eV": round(e_min, 6),
                        "valid": False,
                        "reasons": reasons,
                    }
                ],
                "input_parameters": params(),
                "fidelity_level": "L1",
                "fidelity": "analytical",
                "warnings": _validity_warnings(
                    bulk_band_gap_eV=bulk_band_gap_eV,
                    electron_effective_mass_m0=electron_effective_mass_m0,
                ),
            }
        return solved_payload(
            r_star,
            branch_label="coulomb_minimum_degenerate",
            extra={
                "degenerate_branch_note": (
                    "target equals E_min; the two mathematical branches "
                    "coalesce at R* = 2A/B"
                )
            },
        )

    if energy < bulk_band_gap_eV:
        # Two mathematical roots: r1 on (r_min, r_star), r2 on (r_star, inf).
        lo = radius_min_nm * 1e-9
        r1 = bisect(lo, r_star, energy)
        # Grow the upper bracket until E(r2_hi) exceeds the target so the
        # true mathematical root is found even beyond MAX_RADIUS_NM; the
        # validity check then rejects it with an explicit reason instead of
        # reporting a clamped pseudo-root.
        r2_hi = max(r_star * 2, MAX_RADIUS_NM * 1e-9)
        max_scale = 1e-3  # 1e6 nm: far beyond any physical nanocrystal
        while e_of(r2_hi) <= energy and r2_hi < max_scale:
            r2_hi *= 2
        if e_of(r2_hi) <= energy:
            r2 = None
        else:
            r2 = bisect(r_star, r2_hi, energy)
        valid1, reasons1 = branch_valid(r1, label="strong-confinement")
        if r2 is None:
            valid2 = False
            reasons2 = [
                "the weak-confinement root lies beyond any physical radius "
                "scale (> 1e6 nm) for this target"
            ]
        else:
            valid2, reasons2 = branch_valid(r2, label="weak-confinement")
        roots: list[dict[str, Any]] = [
            {
                "radius_nm": round(r1 * 1e9, 4),
                "energy_eV": round(e_of(r1), 6),
                "branch": "strong-confinement (R < R*)",
                "valid": valid1,
                "reasons": reasons1,
            },
            {
                "radius_nm": (
                    round(r2 * 1e9, 4) if r2 is not None else None
                ),
                "energy_eV": (
                    round(e_of(r2), 6) if r2 is not None else None
                ),
                "branch": "weak-confinement (R > R*)",
                "valid": valid2,
                "reasons": reasons2,
            },
        ]
        if valid1 and valid2:
            return {
                "outcome": "AMBIGUOUS_BRANCH",
                "reason": (
                    "two distinct mathematical roots exist below the bulk "
                    "gap and both lie inside the model validity range; "
                    "choose a branch explicitly or add higher-fidelity "
                    "evidence to disambiguate"
                ),
                "mathematical_roots": roots,
                "input_parameters": params(),
                "fidelity_level": "L1",
                "fidelity": "analytical",
                "warnings": _validity_warnings(
                    bulk_band_gap_eV=bulk_band_gap_eV,
                    electron_effective_mass_m0=electron_effective_mass_m0,
                ),
            }
        if valid1:
            return solved_payload(
                r1,
                branch_label="strong-confinement",
                extra={
                    "ambiguous_branch_note": (
                        "a second mathematical root exists below the bulk "
                        "gap but is outside model validity"
                    )
                },
                rejected=[roots[1]],
            )
        if valid2:
            assert r2 is not None
            return solved_payload(
                r2,
                branch_label="weak-confinement",
                extra={
                    "ambiguous_branch_note": (
                        "the strong-confinement root is outside model "
                        "validity; returning the weak-confinement root"
                    )
                },
                rejected=[roots[0]],
            )
        return {
            "outcome": "OUTSIDE_MODEL_VALIDITY",
            "reason": (
                "mathematical roots exist below the bulk gap but neither is "
                "supported by the model assumptions (EMA range / "
                "strong-confinement regime)"
            ),
            "mathematical_roots": roots,
            "input_parameters": params(),
            "fidelity_level": "L1",
            "fidelity": "analytical",
            "warnings": _validity_warnings(
                bulk_band_gap_eV=bulk_band_gap_eV,
                electron_effective_mass_m0=electron_effective_mass_m0,
            ),
        }

    # energy >= Eg: exactly one root on (0, R*].
    lo = radius_min_nm * 1e-9
    hi = min(r_star * 0.999, MAX_RADIUS_NM * 1e-9)
    hi = max(hi, lo * 2)
    radius = bisect(lo, hi, energy)
    valid, reasons = branch_valid(radius, label="unique")
    if not valid:
        return {
            "outcome": "OUTSIDE_MODEL_VALIDITY",
            "reason": (
                "a mathematical root exists but is outside the model "
                "validity: " + "; ".join(reasons)
            ),
            "mathematical_roots": [
                {
                    "radius_nm": round(radius * 1e9, 4),
                    "energy_eV": round(e_of(radius), 6),
                    "valid": False,
                    "reasons": reasons,
                }
            ],
            "input_parameters": params(),
            "fidelity_level": "L1",
            "fidelity": "analytical",
            "warnings": _validity_warnings(
                bulk_band_gap_eV=bulk_band_gap_eV,
                electron_effective_mass_m0=electron_effective_mass_m0,
            ),
        }
    return solved_payload(radius, branch_label="unique", extra={})


def size_sweep(
    *,
    min_size_nm: float,
    max_size_nm: float,
    points: int,
    bulk_band_gap_eV: float,
    electron_effective_mass_m0: float,
    hole_effective_mass_m0: float,
    relative_dielectric_constant: float | None = None,
    relative_dielectric_kind: str | None = None,
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
            relative_dielectric_kind=relative_dielectric_kind,
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
