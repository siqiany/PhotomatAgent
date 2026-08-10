"""Deterministic photodetector device-physics tools (namespace ``photodetector``).

Responsivity - EQE - gain conversions and target consistency checks. These
tools verify physical consistency only; they never claim that a device is
achievable.
"""

from __future__ import annotations

import json
from typing import Any

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

HC_EV_UM = 1.239841984  # h*c in eV*um; R = EQE * gain * lambda / 1.23984 A/W


class PhotodetectorProbe(CapabilityPack):
    name = "photodetector"
    description = "Responsivity/EQE/gain conversions and target consistency."

    def probe(self) -> ProbeResult:
        return ProbeResult(
            status=CapabilityStatus.AVAILABLE,
            detail="pure Python",
            version="1.0",
        )

    def tools(self) -> list[Tool]:
        return [
            ResponsivityFromEQETool(),
            EQEFromResponsivityTool(),
            CheckTargetsTool(),
        ]


def _split_eqe(arguments: dict[str, Any]) -> tuple[float, str] | str:
    """Read EQE from fraction OR percent (exactly one); returns error string."""
    fraction = arguments.get("eqe_fraction")
    percent = arguments.get("eqe_percent")
    if (fraction is None) == (percent is None):
        return "provide exactly one of eqe_fraction (0..1) or eqe_percent (0..100)"
    if fraction is not None:
        value = float(fraction)
        if not 0.0 <= value <= 1.0:
            return f"eqe_fraction must be in [0, 1], got {value}"
        return value, "fraction"
    value = float(percent)  # type: ignore[arg-type]
    if not 0.0 <= value <= 100.0:
        return f"eqe_percent must be in [0, 100], got {value}"
    return value / 100.0, "percent"


def _positive_float(arguments: dict[str, Any], key: str) -> float | None:
    value = arguments.get(key)
    if value is None:
        return None
    number = float(value)
    if number <= 0:
        raise ValueError(f"{key} must be positive")
    return number


def _responsivity_from_eqe(
    wavelength_um: float, eqe_fraction: float, gain: float
) -> float:
    return eqe_fraction * gain * wavelength_um / HC_EV_UM


def _eqe_from_responsivity(
    wavelength_um: float, responsivity_a_w: float, gain: float
) -> float:
    return responsivity_a_w * HC_EV_UM / (gain * wavelength_um)


class ResponsivityFromEQETool(Tool):
    name = "photodetector.responsivity_from_eqe"
    description = (
        "Convert quantum efficiency to responsivity: R = EQE * gain * "
        "lambda / 1.23984 (A/W, lambda in um). Provide EQE as exactly one of "
        "eqe_fraction (0..1) or eqe_percent (0..100) -- they are never "
        "confused. Outputs responsivity, the ideal unity-gain responsivity, "
        "and consistency notes."
    )
    short_description = "Responsivity from EQE (A/W), with gain."
    exposure = ToolExposure.DEFERRED
    namespace = "photodetector"
    source = "native device-physics model"
    tags = ("photodetector", "responsivity", "eqe", "quantum efficiency")
    input_schema = {
        "type": "object",
        "properties": {
            "wavelength_um": {"type": "number", "minimum": 0},
            "eqe_fraction": {"type": "number", "minimum": 0, "maximum": 1},
            "eqe_percent": {"type": "number", "minimum": 0, "maximum": 100},
            "photoconductive_gain": {"type": "number", "minimum": 0},
        },
        "required": ["wavelength_um"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        wavelength = _positive_float(arguments, "wavelength_um")
        if wavelength is None:
            return _invalid("wavelength_um is required")
        eqe = _split_eqe(arguments)
        if isinstance(eqe, str):
            return _invalid(eqe)
        gain = arguments.get("photoconductive_gain", 1.0)
        gain = 1.0 if gain is None else float(gain)
        if gain <= 0:
            return _invalid("photoconductive_gain must be positive")
        responsivity = _responsivity_from_eqe(wavelength, eqe[0], gain)
        ideal = wavelength / HC_EV_UM
        payload = {
            "wavelength_um": wavelength,
            "eqe_fraction": round(eqe[0], 6),
            "eqe_percent": round(eqe[0] * 100.0, 4),
            "eqe_input_units": eqe[1],
            "photoconductive_gain": gain,
            "responsivity_a_w": round(responsivity, 5),
            "ideal_unity_gain_responsivity_a_w": round(ideal, 5),
            "consistency": (
                f"unity-gain photodiode needs EQE "
                f"{responsivity / max(ideal, 1e-12) * 100:.1f}% to reach "
                f"{responsivity:.4f} A/W at {wavelength:.3f} um"
            ),
            "assumptions": [
                "R = EQE * gain * lambda / 1.23984 (lambda in um)",
                "no wavelength-dependent losses included",
            ],
        }
        evidence = [
            ScientificEvidence(
                subject="photodetector",
                property="responsivity",
                value=responsivity,
                unit="A/W",
                source="photomatagent native device-physics model",
                source_type="analytical_model",
                method="R = EQE * gain * lambda / 1.23984",
                fidelity="analytical",
                summary=f"responsivity {responsivity:.4f} A/W at {wavelength:.3f} um",
                limitations="physical conversion only; no device realizability claim",
                provenance={"tool": self.name, "wavelength_um": wavelength},
            )
        ]
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False, indent=2),
            data=payload,
            evidence=evidence,
        )


class EQEFromResponsivityTool(Tool):
    name = "photodetector.eqe_from_responsivity"
    description = (
        "Convert responsivity to quantum efficiency: EQE = R * 1.23984 / "
        "(gain * lambda). Outputs EQE as both fraction and percent."
    )
    short_description = "EQE from responsivity (fraction and percent)."
    exposure = ToolExposure.DEFERRED
    namespace = "photodetector"
    source = "native device-physics model"
    tags = ("photodetector", "responsivity", "eqe", "quantum efficiency")
    input_schema = {
        "type": "object",
        "properties": {
            "wavelength_um": {"type": "number", "minimum": 0},
            "responsivity_a_w": {"type": "number", "minimum": 0},
            "photoconductive_gain": {"type": "number", "minimum": 0},
        },
        "required": ["wavelength_um", "responsivity_a_w"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        wavelength = _positive_float(arguments, "wavelength_um")
        responsivity = _positive_float(arguments, "responsivity_a_w")
        if wavelength is None or responsivity is None:
            return _invalid("wavelength_um and responsivity_a_w are required")
        gain = arguments.get("photoconductive_gain", 1.0)
        gain = 1.0 if gain is None else float(gain)
        if gain <= 0:
            return _invalid("photoconductive_gain must be positive")
        eqe = _eqe_from_responsivity(wavelength, responsivity, gain)
        if eqe > 1.0:
            note = (
                f"EQE {eqe * 100:.1f}% exceeds 100%: requires photoconductive "
                f"gain > {gain:.3f} to be physically consistent"
            )
        else:
            note = "physically consistent at unity gain"
        payload = {
            "wavelength_um": wavelength,
            "responsivity_a_w": responsivity,
            "photoconductive_gain": gain,
            "eqe_fraction": round(eqe, 6),
            "eqe_percent": round(eqe * 100.0, 4),
            "note": note,
        }
        evidence = [
            ScientificEvidence(
                subject="photodetector",
                property="quantum_efficiency",
                value=eqe,
                unit="fraction",
                source="photomatagent native device-physics model",
                source_type="analytical_model",
                method="EQE = R * 1.23984 / (gain * lambda)",
                fidelity="analytical",
                summary=f"EQE {eqe * 100:.1f}% from R={responsivity:.4f} A/W",
                limitations="physical conversion only",
                provenance={"tool": self.name, "wavelength_um": wavelength},
            )
        ]
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False, indent=2),
            data=payload,
            evidence=evidence,
        )


class CheckTargetsTool(Tool):
    name = "photodetector.check_targets"
    description = (
        "Check whether a target responsivity and target EQE are mutually "
        "consistent across a spectral band using R = EQE * gain * lambda / "
        "1.23984. Reports required EQE at each band edge for the R target, "
        "required R for the EQE target, the required gain when inconsistent, "
        "and the wavelength dependence. This checks physical consistency "
        "only -- it never claims a device is realizable."
    )
    short_description = "Responsivity/EQE target consistency check across a band."
    exposure = ToolExposure.DEFERRED
    namespace = "photodetector"
    source = "native device-physics model"
    tags = ("photodetector", "targets", "responsivity", "eqe", "feasibility")
    input_schema = {
        "type": "object",
        "properties": {
            "spectral_min_um": {"type": "number", "minimum": 0},
            "spectral_max_um": {"type": "number", "minimum": 0},
            "target_responsivity_a_w": {"type": "number", "minimum": 0},
            "eqe_fraction": {"type": "number", "minimum": 0, "maximum": 1},
            "eqe_percent": {"type": "number", "minimum": 0, "maximum": 100},
            "photoconductive_gain": {"type": "number", "minimum": 0},
        },
        "required": [
            "spectral_min_um",
            "spectral_max_um",
            "target_responsivity_a_w",
        ],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        band_min = _positive_float(arguments, "spectral_min_um")
        band_max = _positive_float(arguments, "spectral_max_um")
        target_r = _positive_float(arguments, "target_responsivity_a_w")
        if band_min is None or band_max is None or target_r is None:
            return _invalid("spectral bounds and target_responsivity_a_w are required")
        if band_min > band_max:
            return _invalid("spectral_min_um must be <= spectral_max_um")
        eqe = _split_eqe(arguments)
        if isinstance(eqe, str):
            return _invalid(eqe)
        gain = arguments.get("photoconductive_gain", 1.0)
        gain = 1.0 if gain is None else float(gain)
        if gain <= 0:
            return _invalid("photoconductive_gain must be positive")

        edges = [band_min, band_max]
        required_eqe_for_r = {
            round(lam, 4): round(
                target_r * HC_EV_UM / (gain * lam), 5
            )
            for lam in edges
        }
        required_r_for_eqe = {
            round(lam, 4): round(
                _responsivity_from_eqe(lam, eqe[0], gain), 5
            )
            for lam in edges
        }
        max_required_eqe = max(required_eqe_for_r.values())
        max_required_r = max(required_r_for_eqe.values())
        consistent_at_gain = max_required_eqe <= 1.0 and max_required_r >= target_r
        if not consistent_at_gain:
            needed_gain = max(
                target_r * HC_EV_UM / (eqe[0] * band_min),
                target_r * HC_EV_UM / (eqe[0] * band_max),
            )
        else:
            needed_gain = gain
        payload = {
            "spectral_band_um": [band_min, band_max],
            "target_responsivity_a_w": target_r,
            "target_eqe_fraction": round(eqe[0], 6),
            "target_eqe_percent": round(eqe[0] * 100.0, 4),
            "photoconductive_gain": gain,
            "required_eqe_for_r_target": required_eqe_for_r,
            "required_responsivity_for_eqe_target": required_r_for_eqe,
            "mutually_consistent": consistent_at_gain,
            "required_gain_if_not_consistent": round(needed_gain, 4),
            "wavelength_dependence": {
                "responsivity_scales_as": "R ~ lambda (linear) at fixed EQE",
                "worst_case_wavelength_um": (
                    band_max if required_eqe_for_r[band_max] > required_eqe_for_r[band_min] else band_min
                ),
            },
            "statement": (
                "physical consistency check only: achievable device "
                "performance additionally requires absorption, collection, "
                "dark current, and noise evidence"
            ),
        }
        evidence = [
            ScientificEvidence(
                subject="photodetector_targets",
                property="required_eqe_for_responsivity",
                value=max_required_eqe,
                unit="fraction",
                source="photomatagent native device-physics model",
                source_type="analytical_model",
                method="R = EQE * gain * lambda / 1.23984",
                fidelity="analytical",
                summary=(
                    f"required EQE {max_required_eqe * 100:.1f}% for "
                    f"R={target_r:.3f} A/W across {band_min}-{band_max} um"
                ),
                limitations="consistency only; not a realizability claim",
                provenance={"tool": self.name},
            )
        ]
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False, indent=2),
            data=payload,
            evidence=evidence,
        )


def _invalid(message: str) -> ScientificToolResult:
    return ScientificToolResult(
        output=message,
        is_error=True,
        data={"error_type": "invalid_input", "message": message},
    )


def photodetector_pack() -> CapabilityPack:
    return PhotodetectorProbe()
