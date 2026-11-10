"""Shared, explicit method-domain exclusions for discovery and execution."""
from __future__ import annotations

import json
from typing import Any, Literal, TYPE_CHECKING, get_args

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from photomatagent.scientific.capabilities.contracts import ScientificToolResult

Regime = Literal[
    "core_shell_dot", "strong_band_nonparabolicity", "quantitative_narrow_gap_design",
    "multiphase_alloy", "laterally_patterned_film", "zero_dimensional_dot",
    "device_performance_prediction", "synthesis_feasibility_prediction", "thermodynamic_hull_prediction",
]

_DOMAINS: dict[str, dict[str, str]] = {
    "effective-mass-confinement": {
        "core_shell_dot": "A homogeneous sphere cannot describe a core-shell interface.",
        "strong_band_nonparabolicity": "A parabolic effective-mass approximation is outside its domain.",
        "quantitative_narrow_gap_design": "The L1 approximation requires independent higher-fidelity checks for narrow-gap design.",
    },
    "band-bowing": {
        "multiphase_alloy": "A single-phase interpolation cannot describe phase coexistence.",
        "core_shell_dot": "A uniform alloy interpolation cannot represent a radial core-shell composition.",
    },
    "transfer-matrix": {"laterally_patterned_film": "The planar stack assumes lateral homogeneity."},
    "fdtd-thin-film": {"laterally_patterned_film": "This one-dimensional adapter only models laterally homogeneous layers."},
    "kp-heterostructure": {"zero_dimensional_dot": "The configured adapter does not solve zero-dimensional confinement."},
    "structure-relaxation": {
        "thermodynamic_hull_prediction": "A relaxation or energy estimate alone cannot establish the convex hull.",
        "synthesis_feasibility_prediction": "A potential-energy calculation does not validate a synthesis pathway.",
        "device_performance_prediction": "Structural relaxation does not predict device performance.",
    },
    "structure-generation": {
        "thermodynamic_hull_prediction": "Generated candidates require independent hull evaluation.",
        "synthesis_feasibility_prediction": "Candidate generation cannot validate experimental synthesis.",
        "device_performance_prediction": "Generated structures require separate device evaluation.",
    },
}


class MethodApplicability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    method_id: str
    status: Literal["UNSUPPORTED", "UNKNOWN"]
    incompatible_regimes: tuple[str, ...]
    reasons: tuple[str, ...]


def domain_exclusions(method_id: str) -> dict[str, str]:
    return dict(_DOMAINS.get(method_id, {}))


def assess_method(method_id: str, regimes: tuple[Regime, ...]) -> MethodApplicability:
    exclusions = domain_exclusions(method_id)
    matched = tuple(sorted(exclusions.keys() & set(regimes)))
    return MethodApplicability(method_id=method_id, status="UNSUPPORTED" if matched else "UNKNOWN",
                               incompatible_regimes=matched, reasons=tuple(exclusions[item] for item in matched))


def with_domain_input(schema: dict[str, Any]) -> dict[str, Any]:
    return {**schema, "properties": {**schema["properties"], "regimes": {
        "type": "array", "maxItems": len(get_args(Regime)), "uniqueItems": True,
        "items": {"type": "string", "enum": list(get_args(Regime))},
        "description": "Declare known physical conditions or intended scientific use for applicability checking.",
    }}}


def reject_outside_domain(method_id: str, arguments: dict[str, Any]) -> ScientificToolResult | None:
    from photomatagent.scientific.capabilities.contracts import ScientificToolResult

    result = assess_method(method_id, tuple(arguments.get("regimes", ())))
    if result.status == "UNKNOWN":
        return None
    data = {"error_type": "INCOMPATIBLE_SCIENTIFIC_PARAMETER",
            "applicability": result.model_dump(mode="json"),
            "evidence_gaps": ["Choose a method that covers the declared conditions and independently check its inputs."]}
    return ScientificToolResult(output=json.dumps(data), data=data, is_error=True)
