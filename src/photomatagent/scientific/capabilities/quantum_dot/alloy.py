"""Deterministic alloy band-gap bowing.

Generic quadratic form (standard semiconductor alloy model, e.g. Vurgaftman
& Meyer 2001 for III-V; Hansen et al. 1982 for HgCdTe is cubic and must be
provided as parameters, not hardcoded here):

    Eg(x) = (1 - x) Eg_A + x Eg_B - b x (1 - x)

The tool is deliberately generic: all coefficients are inputs with their own
provenance. Optional Varshni temperature shift per endpoint:

    Eg(T) = Eg(0) - alpha T^2 / (T + beta)
"""

from __future__ import annotations

from typing import Any


def bandgap_bowing(
    *,
    x: float,
    band_gap_a_eV: float,
    band_gap_b_eV: float,
    bowing_parameter_eV: float,
    temperature_k: float | None = None,
    varshni_alpha_a: float | None = None,
    varshni_beta_a: float | None = None,
    varshni_alpha_b: float | None = None,
    varshni_beta_b: float | None = None,
) -> dict[str, Any]:
    """Bowed alloy band gap at composition ``x`` (fraction of material B)."""
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"composition x must be in [0, 1], got {x}")
    gap_a, gap_b = band_gap_a_eV, band_gap_b_eV
    if temperature_k is not None:
        if None in (varshni_alpha_a, varshni_beta_a, varshni_alpha_b, varshni_beta_b):
            missing = [
                name
                for name, value in {
                    "varshni_alpha_a": varshni_alpha_a,
                    "varshni_beta_a": varshni_beta_a,
                    "varshni_alpha_b": varshni_alpha_b,
                    "varshni_beta_b": varshni_beta_b,
                }.items()
                if value is None
            ]
            from photomatagent.scientific.errors import MissingScientificPrerequisite

            raise MissingScientificPrerequisite(
                "temperature shift requested but Varshni parameters are incomplete",
                missing=missing,
            )
        gap_a = _varshni(gap_a, temperature_k, varshni_alpha_a, varshni_beta_a)  # type: ignore[arg-type]
        gap_b = _varshni(gap_b, temperature_k, varshni_alpha_b, varshni_beta_b)  # type: ignore[arg-type]
    band_gap = (1.0 - x) * gap_a + x * gap_b - bowing_parameter_eV * x * (1.0 - x)
    return {
        "composition_x": x,
        "band_gap_a_eV": band_gap_a_eV,
        "band_gap_b_eV": band_gap_b_eV,
        "bowing_parameter_eV": bowing_parameter_eV,
        "temperature_k": temperature_k,
        "band_gap_eV": band_gap,
        "wavelength_um": 1.239841984 / band_gap if band_gap > 0 else None,
        "fidelity": "empirical",
        "method": "quadratic bowing Eg(x) = (1-x)EgA + x EgB - b x(1-x)"
        + (" with Varshni temperature shift" if temperature_k is not None else ""),
        "assumptions": [
            "quadratic composition dependence",
            "single bowing parameter independent of composition",
            "parameters must carry their own source/temperature/validity "
            "range (not hardcoded here)",
        ],
        "warnings": (
            ["bowing fit valid only near the composition range of the source "
             "parameters; do not extrapolate to endpoints without evidence"]
            if 0.0 < x < 1.0
            else []
        ),
    }


def _varshni(
    gap_0: float, temperature_k: float, alpha: float, beta: float
) -> float:
    if temperature_k < 0:
        raise ValueError("temperature must be non-negative")
    return gap_0 - alpha * temperature_k**2 / (temperature_k + beta)
