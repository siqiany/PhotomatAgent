"""Infrared photodetector constraint compiler (namespace ``ir``).

``ir.compile_constraints`` turns a spectral requirement into deterministic
physics constraints: photon energies, cutoff energy requirement, thermal
limits, ideal responsivity, blackbody photon flux, BLIP detectivity, and the
evidence required to validate a candidate. No LLM arithmetic is involved.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from photomatagent.scientific.capabilities.base import (
    CapabilityPack,
    CapabilityStatus,
    ProbeResult,
)
from photomatagent.scientific.capabilities.contracts import (
    ScientificEvidence,
    ScientificToolResult,
)
from photomatagent.tools.base import Tool
from photomatagent.tools.exposure import ToolExposure

HC_EV_UM = 1.239841984  # h*c in eV*um
KB_EV_K = 8.617333262e-5  # Boltzmann constant in eV/K
HC_J_M = HC_EV_UM * 1.602176634e-19 * 1e-6  # h*c in J*m
SPEED_OF_LIGHT_M_S = 2.99792458e8
KB_J_K = 1.380649e-23
PLANCK_J_S = 6.62607015e-34
ELEMENTARY_CHARGE_C = 1.602176634e-19
HC_EXACT_J_M = PLANCK_J_S * SPEED_OF_LIGHT_M_S

DETECTOR_TYPE_EVIDENCE = {
    "photoconductor": [
        "photoconductive gain and carrier lifetime",
        "dark conductivity and its temperature dependence",
    ],
    "photodiode": [
        "junction R0A product and dark current",
        "quantum efficiency at the target wavelengths",
    ],
    "phototransistor": [
        "internal gain and response time",
        "base/collector dark current",
    ],
    "bolometer": [
        "temperature coefficient of resistance (TCR)",
        "thermal conductance and heat capacity of the pixel",
    ],
    "pyroelectric": [
        "pyroelectric coefficient and thermal time constant",
    ],
    "qwp": [
        "quantum well design and polarization selection rules",
        "dark current vs bias and temperature",
    ],
}


class IRProbe(CapabilityPack):
    name = "ir"
    description = "Infrared photodetector constraint compilation."

    def probe(self) -> ProbeResult:
        return ProbeResult(
            status=CapabilityStatus.AVAILABLE,
            detail="pure Python + numpy; always available",
            version="1.0",
        )

    def tools(self) -> list[Tool]:
        return [CompileConstraintsTool()]


def blackbody_photon_flux(
    lambda_min_um: float,
    lambda_max_um: float,
    temperature_k: float,
) -> float:
    """Blackbody photon flux (photons/s/m2) in [lambda_min, lambda_max]."""
    grid = np.linspace(lambda_min_um, lambda_max_um, 4096)
    x = HC_EV_UM / (grid * KB_EV_K * temperature_k)
    with np.errstate(over="ignore", divide="ignore"):
        # lambda is in um; convert to meters inside the prefactor (1e-24).
        # lambda is in um: convert to meters inside the prefactor (1e-18 =
        # 1e-24 from lambda**4 and 1e6 from d(lambda_m) = 1e-6 * d(lambda_um)).
        spectral = 2.0 * math.pi * SPEED_OF_LIGHT_M_S * 1e18 / (
            grid**4 * (np.exp(x) - 1.0)
        )
    spectral = np.nan_to_num(spectral, nan=0.0, posinf=0.0)
    return float(np.trapezoid(spectral, grid))


def _positive_finite(name: str, value: float | None) -> None:
    if value is not None and (not math.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be finite and positive")


def _precise_constraints(
    *, band_max_um: float, temperature_k: float, detector_type: str,
    target_wavelength_um: float | None, target_eqe_fraction: float | None,
    active_width_um: float | None, active_height_um: float | None,
    target_detectivity: float | None, target_responsivity: float | None,
    target_bandwidth_hz: float | None,
) -> dict[str, dict[str, Any]]:
    """Necessary idealized budgets; absent prerequisites remain UNKNOWN."""
    result: dict[str, dict[str, Any]] = {}

    def add(key: str, value: float | None, unit: str, formula: str,
            assumptions: str, missing: str = "") -> None:
        if value is not None and not math.isfinite(value):
            raise ValueError(f"{key} is outside the finite numerical range")
        result[key] = {
            "value": value,
            "unit": unit,
            "formula": formula,
            "assumptions": assumptions,
            "status": "NECESSARY_CONSTRAINT" if value is not None else "UNKNOWN",
            "missing_inputs": missing if value is None else "",
        }

    interband = detector_type not in {"bolometer", "pyroelectric"}
    add("cutoff_energy_upper_bound_eV",
        HC_EXACT_J_M / (ELEMENTARY_CHARGE_C * band_max_um * 1e-6) if interband else None,
        "eV", "Eg <= hc/(q lambda_max)",
        "Interband photon absorption is assumed; this band-gap limit does not apply to thermal detection.",
        "interband photon-detection mechanism" if not interband else "")
    eqe_responsivity = (
        target_eqe_fraction * ELEMENTARY_CHARGE_C * target_wavelength_um * 1e-6 / HC_EXACT_J_M
        if interband and target_eqe_fraction is not None and target_wavelength_um is not None else None
    )
    responsivity = eqe_responsivity if eqe_responsivity is not None else target_responsivity
    add("responsivity_target_A_per_W", responsivity, "A/W",
        "R = eta q lambda/(hc), G = 1" if eqe_responsivity is not None else "R >= supplied target_responsivity",
        "Unity photoconductive gain for EQE-derived value; all targets are inputs, not measured values.",
        "photon-current mechanism with target_eqe_fraction and target_wavelength_um, or target_responsivity")
    result["responsivity_target_A_per_W"]["basis"] = (
        "eqe_unity_gain" if eqe_responsivity is not None else
        "explicit_target" if target_responsivity is not None else "UNKNOWN"
    )
    result["responsivity_target_A_per_W"]["independent_target_A_per_W"] = target_responsivity
    if eqe_responsivity is not None and target_responsivity is not None:
        result["responsivity_target_A_per_W"]["independent_target_note"] = (
            "The explicit responsivity target is an independent threshold; it does not replace the EQE-derived unity-gain value."
        )
    area = (
        active_width_um * active_height_um * 1e-8
        if active_width_um is not None and active_height_um is not None else None
    )
    add("active_area_cm2", area, "cm2", "A = width_um height_um x 1e-8",
        "Rectangular electrically active area.", "active_width_um and active_height_um")
    noise_responsivity = target_responsivity if target_responsivity is not None else eqe_responsivity
    noise = (
        noise_responsivity * math.sqrt(area) / target_detectivity
        if noise_responsivity is not None and area is not None and target_detectivity is not None
        else None
    )
    add("total_noise_upper_bound_A_per_sqrt_Hz", noise, "A/sqrt(Hz)",
        "i_n <= R sqrt(A)/D*",
        "D* in cm sqrt(Hz)/W; total input-referred current noise. Explicit R target takes precedence for this budget when supplied.",
        "responsivity, active area and target_detectivity")
    result["total_noise_upper_bound_A_per_sqrt_Hz"]["basis"] = (
        "explicit_target" if target_responsivity is not None else
        "eqe_unity_gain" if eqe_responsivity is not None else "UNKNOWN"
    )
    tau = 1 / (2 * math.pi * target_bandwidth_hz) if target_bandwidth_hz is not None else None
    add("time_constant_upper_bound_s", tau, "s", "tau <= 1/(2 pi f_3dB)",
        "Single-pole first-order response; target bandwidth is a 3 dB target.",
        "target_bandwidth_hz")
    add("rise_time_10_90_upper_bound_s", math.log(9) * tau if tau is not None else None,
        "s", "t_10-90 <= ln(9) tau", "Single-pole first-order response.",
        "target_bandwidth_hz")
    noise_power = noise * noise if noise is not None else 0.0
    resistance = 4 * KB_J_K * temperature_k / noise_power if noise_power > 0 else None
    add("johnson_R0_lower_bound_ohm", resistance, "ohm",
        "R0 >= 4 k_B T / i_n^2",
        "Entire allowed noise budget assigned to Johnson current noise; zero-bias resistance.",
        "positive total-noise budget (requires nonzero target EQE)")
    add("johnson_R0A_lower_bound_ohm_cm2", resistance * area if resistance is not None and area is not None else None,
        "ohm cm2", "R0A >= R0 A",
        "Entire allowed noise budget assigned to Johnson noise; same active area.",
        "positive Johnson R0 bound and active area")
    return result


_BOUND_DIRECTIONS = {
    "cutoff_energy_upper_bound_eV": "upper",
    "responsivity_target_A_per_W": "lower",
    "active_area_cm2": "equal",
    "total_noise_upper_bound_A_per_sqrt_Hz": "upper",
    "time_constant_upper_bound_s": "upper",
    "rise_time_10_90_upper_bound_s": "upper",
    "johnson_R0_lower_bound_ohm": "lower",
    "johnson_R0A_lower_bound_ohm_cm2": "lower",
}


def check_ir_constraint_claims(
    constraints: dict[str, Any], claims: dict[str, Any], *, rel_tol: float = 1e-8,
) -> dict[str, dict[str, Any]]:
    """Check arithmetic compatibility only; claims are not validated observations."""
    if not math.isfinite(rel_tol) or rel_tol < 0:
        raise ValueError("rel_tol must be finite and nonnegative")
    bounds = constraints.get("necessary_constraints", {})
    checks: dict[str, dict[str, Any]] = {}
    for name, direction in _BOUND_DIRECTIONS.items():
        bound = bounds.get(name)
        claim = claims.get(name)
        status = "UNKNOWN"
        reason = "missing bound or claimed metric"
        if bound is not None and bound["status"] == "NECESSARY_CONSTRAINT" and isinstance(claim, dict):
            value = claim.get("value")
            unit = claim.get("unit")
            if unit != bound["unit"]:
                reason = "claimed unit does not match bound unit"
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                reason = "claimed value must be a finite number"
            elif value < 0:
                reason = "claimed physical metric must be nonnegative"
            else:
                limit = bound["value"]
                tolerance = rel_tol * max(abs(limit), abs(value))
                compatible = (
                    value <= limit + tolerance if direction == "upper" else
                    value >= limit - tolerance if direction == "lower" else
                    abs(value - limit) <= tolerance
                )
                status = "ARITHMETICALLY_COMPATIBLE" if compatible else "VIOLATES_BOUND"
                reason = "numeric comparison only; source and device identity are unverified"
        checks[name] = {
            "status": status, "reason": reason, "direction": direction,
            "bound_value": bound["value"] if bound is not None else None,
            "unit": bound["unit"] if bound is not None else None,
            "claimed_value": claim.get("value") if isinstance(claim, dict) else None,
            "rel_tol": rel_tol,
        }
    return checks


def compile_ir_constraints(
    *,
    spectral_min_um: float,
    spectral_max_um: float,
    temperature_k: float | None = None,
    detector_type: str | None = None,
    target_detectivity: float | None = None,
    target_responsivity: float | None = None,
    target_dark_current: float | None = None,
    target_netd: float | None = None,
    target_wavelength_um: float | None = None,
    target_eqe_fraction: float | None = None,
    active_width_um: float | None = None,
    active_height_um: float | None = None,
    target_bandwidth_hz: float | None = None,
) -> dict[str, Any]:
    """Deterministically compile IR detector constraints from a spectral band."""
    for name, value in (
        ("spectral_min_um", spectral_min_um), ("spectral_max_um", spectral_max_um),
        ("temperature_k", temperature_k), ("target_detectivity", target_detectivity),
        ("target_responsivity", target_responsivity),
        ("target_netd", target_netd), ("target_wavelength_um", target_wavelength_um),
        ("active_width_um", active_width_um), ("active_height_um", active_height_um),
        ("target_bandwidth_hz", target_bandwidth_hz),
    ):
        _positive_finite(name, value)
    if target_dark_current is not None and (
        not math.isfinite(target_dark_current) or target_dark_current < 0
    ):
        raise ValueError("target_dark_current must be finite and nonnegative")
    if spectral_min_um > spectral_max_um:
        raise ValueError("spectral_min_um must be <= spectral_max_um")
    temperature = float(temperature_k if temperature_k is not None else 300.0)
    if target_wavelength_um is not None and not spectral_min_um <= target_wavelength_um <= spectral_max_um:
        raise ValueError("target_wavelength_um must be within the spectral band")
    if target_eqe_fraction is not None and (not math.isfinite(target_eqe_fraction) or not 0 <= target_eqe_fraction <= 1):
        raise ValueError("target_eqe_fraction must be finite and between 0 and 1")
    if (active_width_um is None) != (active_height_um is None):
        raise ValueError("active_width_um and active_height_um must be supplied together")
    detector = (detector_type or "unspecified").strip().casefold()
    if detector not in DETECTOR_TYPE_EVIDENCE and detector != "unspecified":
        raise ValueError(
            f"unsupported detector_type {detector!r}; choose from "
            f"{sorted(DETECTOR_TYPE_EVIDENCE)}"
        )

    hc = HC_EV_UM
    e_max = hc / spectral_min_um
    e_min = hc / spectral_max_um
    cutoff_energy = hc / spectral_max_um
    kt = KB_EV_K * temperature
    midpoint_um = math.sqrt(spectral_min_um * spectral_max_um)
    ideal_responsivity = {
        "at_peak_wavelength_A_per_W": round(midpoint_um / hc, 4),
        "at_shortest_A_per_W": round(spectral_min_um / hc, 4),
        "at_longest_A_per_W": round(spectral_max_um / hc, 4),
    }
    photon_flux = blackbody_photon_flux(spectral_min_um, spectral_max_um, temperature)
    thermal_ratio = math.exp(-cutoff_energy / kt)
    thermal_gap_guideline = 4.0 * kt
    blip_detectivity = None
    if photon_flux > 0:
        blip_detectivity = round(
            # D*_BLIP = (lambda/hc) * sqrt(eta / (2 * Phi_b)) in m*sqrt(Hz)/W,
            # converted to cm*sqrt(Hz)/W; eta = 1 assumed.
            100.0
            * (midpoint_um * 1e-6 / HC_J_M)
            * math.sqrt(1.0 / (2.0 * photon_flux)),
            4,
        )

    required_evidence = [
        "band gap (experimental and computed; must satisfy Eg <= cutoff energy)",
        "temperature dependence of the gap and carrier population",
        "optical response: absorption coefficient across the spectral band",
        "carrier transport: mobility and lifetime governing collection",
        "defect/trap behavior limiting dark current and lifetime",
        "device dark current and its temperature dependence",
    ]
    if detector != "unspecified":
        required_evidence.extend(DETECTOR_TYPE_EVIDENCE[detector])
    if target_detectivity is not None:
        required_evidence.append("noise characterization needed to reach target detectivity")
    if target_netd is not None:
        required_evidence.append(
            "thermal sensitivity (dR/dT or dV/dT) and noise budget for target NETD"
        )
    if target_dark_current is not None:
        required_evidence.append("dark-current budget decomposition by generation mechanism")

    constraints = {
        "spectral_band_um": [round(spectral_min_um, 6), round(spectral_max_um, 6)],
        "photon_energy_range_eV": [round(e_min, 6), round(e_max, 6)],
        "cutoff_energy_requirement_eV": round(cutoff_energy, 6),
        "band_gap_upper_bound_eV": round(cutoff_energy, 6),
        "thermal": {
            "temperature_K": round(temperature, 2),
            "kBT_eV": round(kt, 6),
            "thermal_suppression_ratio_at_cutoff": round(thermal_ratio, 6),
            "thermal_gap_guideline_eV": round(thermal_gap_guideline, 6),
            "cutoff_exceeds_thermal_guideline": cutoff_energy >= thermal_gap_guideline,
        },
        "ideal_responsivity_A_per_W": ideal_responsivity,
        "blackbody_photon_flux_photons_s_m2": round(photon_flux, 4),
        "blip_detectivity_cm_Hz_W": blip_detectivity,
        "background_reference_conditions": {
            "temperature_K": temperature,
            "temperature_binding": "shared temperature_k input; not independently measured scene temperature",
            "angular_model": "hemispherical photon exitance (cosine-weighted angular integral pi); not a configured detector field of view",
            "spectral_band_um": [spectral_min_um, spectral_max_um],
            "reference_wavelength_um": midpoint_um,
            "quantum_efficiency_fraction": 1.0,
            "validation_status": "UNKNOWN",
        },
        "targets": {
            "detectivity_cm_Hz_W": (
                float(target_detectivity) if target_detectivity is not None else None
            ),
            "responsivity_A_per_W": (
                float(target_responsivity) if target_responsivity is not None else None
            ),
            "dark_current_A": (
                float(target_dark_current) if target_dark_current is not None else None
            ),
            "netd_K": float(target_netd) if target_netd is not None else None,
        },
        "required_evidence": required_evidence,
        "assumptions": [
            "blackbody photon flux uses T = 300 K when temperature_k is omitted",
            "ideal responsivity assumes unity quantum efficiency",
            "BLIP detectivity assumes background-limited ideal detector",
            "database matches are not validated detectors",
        ],
    }
    if any(value is not None for value in (
        target_wavelength_um, target_eqe_fraction, active_width_um, active_height_um,
        target_bandwidth_hz, target_responsivity,
    )):
        constraints["necessary_constraints"] = _precise_constraints(
            band_max_um=spectral_max_um, temperature_k=temperature,
            detector_type=detector,
            target_wavelength_um=target_wavelength_um,
            target_eqe_fraction=target_eqe_fraction,
            active_width_um=active_width_um, active_height_um=active_height_um,
            target_detectivity=target_detectivity,
            target_responsivity=target_responsivity,
            target_bandwidth_hz=target_bandwidth_hz,
        )
        constraints["empirical_performance_status"] = "UNKNOWN"
    if detector in {"bolometer", "pyroelectric"}:
        assumptions = constraints["assumptions"]
        if isinstance(assumptions, list):
            assumptions.append(
                "The legacy band-gap cutoff is an interband photon-energy reference, not a requirement for this thermal detector."
            )
    return constraints


class CompileConstraintsTool(Tool):
    name = "ir.compile_constraints"
    description = (
        "Compile deterministic infrared photodetector constraints from a spectral "
        "band: photon energies, cutoff (band gap) requirement, thermal limits, "
        "ideal responsivity, blackbody photon flux, BLIP detectivity, and the "
        "evidence required to validate a candidate material."
    )
    short_description = "IR constraint compiler: spectral band to physics constraints."
    exposure = ToolExposure.DEFERRED
    namespace = "ir"
    source = "photomatagent"
    tags = ("infrared", "constraints", "band gap", "detectivity", "lwir", "mwir")
    input_schema = {
        "type": "object",
        "properties": {
            "spectral_min_um": {
                "type": "number",
                "exclusiveMinimum": 0,
                "description": "Shortest wavelength in the band (um).",
            },
            "spectral_max_um": {
                "type": "number",
                "exclusiveMinimum": 0,
                "description": "Longest wavelength in the band (um).",
            },
            "temperature_K": {"type": "number", "exclusiveMinimum": 0},
            "detector_type": {
                "type": "string",
                "enum": [
                    "unspecified",
                    "photoconductor",
                    "photodiode",
                    "phototransistor",
                    "bolometer",
                    "pyroelectric",
                    "qwp",
                ],
            },
            "target_detectivity": {"type": "number", "description": "Target D* in cm.Hz^0.5/W."},
            "target_responsivity": {"type": "number", "description": "Target R in A/W."},
            "target_dark_current": {"type": "number", "description": "Target dark current in A."},
            "target_NETD": {"type": "number", "description": "Target NETD in K."},
            "target_wavelength_um": {
                "type": "number", "exclusiveMinimum": 0,
                "description": "Wavelength for ideal unity-gain responsivity budget, within the band (um).",
            },
            "target_eqe_fraction": {
                "type": "number", "minimum": 0, "maximum": 1,
                "description": "Assumed external quantum efficiency as a fraction, not evidence of measured EQE.",
            },
            "active_width_um": {"type": "number", "exclusiveMinimum": 0},
            "active_height_um": {"type": "number", "exclusiveMinimum": 0},
            "target_bandwidth_hz": {
                "type": "number", "exclusiveMinimum": 0,
                "description": "Target 3 dB bandwidth in Hz for a single-pole response.",
            },
            "claimed_metrics": {
                "type": "object",
                "description": "Optional unverified numeric claims keyed by necessary constraint name; each has value and exact unit. Checks arithmetic only.",
                "additionalProperties": {
                    "type": "object",
                    "properties": {"value": {"type": "number"}, "unit": {"type": "string"}},
                    "required": ["value", "unit"],
                },
            },
        },
        "required": ["spectral_min_um", "spectral_max_um"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        try:
            constraints = compile_ir_constraints(
                spectral_min_um=float(arguments["spectral_min_um"]),
                spectral_max_um=float(arguments["spectral_max_um"]),
                temperature_k=(
                    float(arguments["temperature_K"])
                    if arguments.get("temperature_K") is not None
                    else None
                ),
                detector_type=arguments.get("detector_type"),
                target_detectivity=(
                    float(arguments["target_detectivity"])
                    if arguments.get("target_detectivity") is not None
                    else None
                ),
                target_responsivity=(
                    float(arguments["target_responsivity"])
                    if arguments.get("target_responsivity") is not None
                    else None
                ),
                target_dark_current=(
                    float(arguments["target_dark_current"])
                    if arguments.get("target_dark_current") is not None
                    else None
                ),
                target_netd=(
                    float(arguments["target_NETD"])
                    if arguments.get("target_NETD") is not None
                    else None
                ),
                target_wavelength_um=(
                    float(arguments["target_wavelength_um"])
                    if arguments.get("target_wavelength_um") is not None else None
                ),
                target_eqe_fraction=(
                    float(arguments["target_eqe_fraction"])
                    if arguments.get("target_eqe_fraction") is not None else None
                ),
                active_width_um=(
                    float(arguments["active_width_um"])
                    if arguments.get("active_width_um") is not None else None
                ),
                active_height_um=(
                    float(arguments["active_height_um"])
                    if arguments.get("active_height_um") is not None else None
                ),
                target_bandwidth_hz=(
                    float(arguments["target_bandwidth_hz"])
                    if arguments.get("target_bandwidth_hz") is not None else None
                ),
            )
        except (ValueError, TypeError, KeyError) as exc:
            return ScientificToolResult(
                output=f"ir.compile_constraints failed: {exc}",
                is_error=True,
                data={"error": "invalid_input"},
            )
        if arguments.get("claimed_metrics") is not None:
            if "necessary_constraints" not in constraints:
                constraints["necessary_constraints"] = _precise_constraints(
                    band_max_um=float(arguments["spectral_max_um"]),
                    temperature_k=float(arguments.get("temperature_K") or 300.0),
                    detector_type=(arguments.get("detector_type") or "unspecified").strip().casefold(),
                    target_wavelength_um=None, target_eqe_fraction=None,
                    active_width_um=None, active_height_um=None,
                    target_detectivity=None, target_responsivity=None,
                    target_bandwidth_hz=None,
                )
                constraints["empirical_performance_status"] = "UNKNOWN"
            claims = arguments["claimed_metrics"]
            if not isinstance(claims, dict):
                return ScientificToolResult(
                    output="ir.compile_constraints failed: claimed_metrics must be an object",
                    is_error=True, data={"error": "invalid_input"},
                )
            constraints["claim_checks"] = check_ir_constraint_claims(constraints, claims)
        band = constraints["spectral_band_um"]
        evidence = [
            ScientificEvidence(
                subject=f"IR band {band[0]}-{band[1]} um",
                property="cutoff_energy_requirement",
                value=constraints["cutoff_energy_requirement_eV"],
                unit="eV",
                source="photomatagent ir.compile_constraints",
                source_type="derived",
                method="E = hc/lambda",
                summary=(
                    f"Photon energy at {band[1]} um is "
                    f"{constraints['cutoff_energy_requirement_eV']:.4f} eV; no band-gap requirement "
                    "applies to this thermal detector"
                    if arguments.get("detector_type") in {"bolometer", "pyroelectric"} else
                    f"Candidate band gap must be <= "
                    f"{constraints['cutoff_energy_requirement_eV']:.4f} eV to reach "
                    f"{band[1]} um"
                ),
                limitations=(
                    "Photon-energy reference only; an interband band-gap bound does not apply to thermal detection"
                    if arguments.get("detector_type") in {"bolometer", "pyroelectric"} else
                    "Idealized interband photon-absorption assumption; real detectors need absorption above threshold"
                ),
                provenance={"tool": self.name},
            ),
            ScientificEvidence(
                subject=f"IR band {band[0]}-{band[1]} um",
                property="blackbody_photon_flux",
                value=constraints["blackbody_photon_flux_photons_s_m2"],
                unit="photons/s/m2",
                source="photomatagent ir.compile_constraints",
                source_type="derived",
                method="Planck integral at specified temperature",
                summary=(
                    f"Blackbody photon flux in band is "
                    f"{constraints['blackbody_photon_flux_photons_s_m2']:.3e}"
                ),
                limitations="Assumes 300 K background unless temperature_K given",
                provenance={"tool": self.name},
            ),
        ]
        for property_name, bound in constraints.get("necessary_constraints", {}).items():
            if bound["status"] != "NECESSARY_CONSTRAINT":
                continue
            evidence.append(ScientificEvidence(
                subject=f"IR target for band {band[0]}-{band[1]} um",
                property=property_name,
                value=bound["value"],
                unit=bound["unit"],
                source="photomatagent ir.compile_constraints",
                source_type="derived",
                method=bound["formula"],
                summary=f"Idealized necessary constraint: {property_name} = {bound['value']}",
                limitations=bound["assumptions"] + " Empirical device performance is UNKNOWN.",
                provenance={"tool": self.name, "status": bound["status"],
                            "input_targets": {key: arguments.get(key) for key in (
                                "target_wavelength_um", "target_eqe_fraction", "active_width_um",
                                "active_height_um", "target_detectivity", "target_responsivity",
                                "target_bandwidth_hz",
                                "temperature_K",
                            )}},
            ))
        return ScientificToolResult(
            output=json.dumps(constraints, ensure_ascii=False),
            data={**constraints, "evidence_gaps": constraints["required_evidence"]},
            evidence=evidence,
        )


def ir_pack() -> CapabilityPack:
    return IRProbe()
