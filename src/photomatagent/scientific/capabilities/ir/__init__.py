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
) -> dict[str, Any]:
    """Deterministically compile IR detector constraints from a spectral band."""
    if spectral_min_um <= 0 or spectral_max_um <= 0:
        raise ValueError("spectral bounds must be positive")
    if spectral_min_um > spectral_max_um:
        raise ValueError("spectral_min_um must be <= spectral_max_um")
    temperature = float(temperature_k if temperature_k else 300.0)
    if temperature <= 0:
        raise ValueError("temperature_k must be positive")
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
            )
        except ValueError as exc:
            return ScientificToolResult(
                output=f"ir.compile_constraints failed: {exc}",
                is_error=True,
                data={"error": "invalid_input"},
            )
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
                    f"Candidate band gap must be <= "
                    f"{constraints['cutoff_energy_requirement_eV']:.4f} eV to reach "
                    f"{band[1]} um"
                ),
                limitations="Idealized; real detectors need absorption above threshold",
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
        return ScientificToolResult(
            output=json.dumps(constraints, ensure_ascii=False),
            data={**constraints, "evidence_gaps": constraints["required_evidence"]},
            evidence=evidence,
        )


def ir_pack() -> CapabilityPack:
    return IRProbe()
