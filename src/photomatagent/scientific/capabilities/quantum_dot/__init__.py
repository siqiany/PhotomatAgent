"""Quantum-dot computational tools (namespace ``qd``) + alloy tools.

Deterministic L1 (Brus/effective-mass) and L0 (bowing) solvers only. Every
result carries ``fidelity``, ``assumptions``, ``warnings``, and Scientific
Evidence; missing parameters produce typed ``missing_prerequisites``
failures. No design skill lives here -- these are tools for a later skill to
orchestrate.
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
from photomatagent.scientific.capabilities.quantum_dot.alloy import bandgap_bowing
from photomatagent.scientific.capabilities.quantum_dot.brus import (
    excitonic_regime,
    size_sweep,
    solve_size_for_transition,
    transition_energy,
)
from photomatagent.scientific.capabilities.quantum_dot.models import (
    MaterialParameterRegistry,
    ScientificParameter,
    default_registry,
)
from photomatagent.scientific.capabilities.quantum_dot.screening import (
    screen_size_composition,
)
from photomatagent.scientific.errors import (
    MissingScientificPrerequisite,
    UnsupportedScientificRegime,
    prerequisite_failure,
)
from photomatagent.tools.base import Tool
from photomatagent.tools.exposure import ToolExposure
from photomatagent.workspace import Workspace


class QuantumDotProbe(CapabilityPack):
    name = "quantum_dot"
    description = "Deterministic quantum-dot confinement solvers (Brus/EMA, L1)."

    def probe(self) -> ProbeResult:
        return ProbeResult(
            status=CapabilityStatus.AVAILABLE,
            detail="pure Python; deterministic analytical solvers",
            version="1.0",
        )

    def tools(self) -> list[Tool]:
        return [
            BrusTransitionEnergyTool(),
            ExcitonicRegimeTool(),
            SolveSizeForTransitionTool(),
            SizeSweepTool(),
            ScreenSizeCompositionTool(),
            ParameterLookupTool(),
        ]


class AlloyProbe(CapabilityPack):
    name = "alloy"
    description = "Deterministic alloy band-gap bowing."

    def probe(self) -> ProbeResult:
        return ProbeResult(
            status=CapabilityStatus.AVAILABLE,
            detail="pure Python",
            version="1.0",
        )

    def tools(self) -> list[Tool]:
        return [BandgapBowingTool()]


def _evidence(
    *,
    subject: str,
    property_name: str,
    value: float,
    unit: str,
    method: str,
    limitations: str,
    provenance: dict[str, Any],
    fidelity: str = "analytical",
    summary: str = "",
) -> ScientificEvidence:
    return ScientificEvidence(
        subject=subject,
        property=property_name,
        value=value,
        unit=unit,
        source="photomatagent native solver",
        source_type="analytical_model",
        method=method,
        fidelity=fidelity,  # type: ignore[arg-type]
        summary=summary,
        limitations=limitations,
        provenance=provenance,
    )


def _params_or_failure(
    arguments: dict[str, Any],
    *,
    tool: str,
    required: list[tuple[str, str]],
) -> dict[str, Any] | None:
    """Return a prerequisite failure dict if any required parameter is absent."""
    missing = [name for name, _ in required if arguments.get(name) is None]
    if missing:
        return prerequisite_failure(
            f"{tool} cannot compute: required parameters missing",
            missing=missing,
            tool=tool,
        )
    return None


class BrusTransitionEnergyTool(Tool):
    name = "qd.brus_transition_energy"
    description = (
        "Compute the L1 Brus effective-mass transition energy of a spherical "
        "nanocrystal: E = Eg + h^2/(8 m0 R^2)(1/me + 1/mh) - 1.786 e^2/(4 pi eps0 "
        "epsr R) (Brus 1984 / Kayanuma 1988). Inputs: radius_nm, "
        "bulk_band_gap_eV, electron_effective_mass_m0, hole_effective_mass_m0, "
        "optional relative_dielectric_constant and include_coulomb_term. Low "
        "fidelity (L1): spherical/EMA/infinite-barrier assumptions; do NOT use "
        "for design values on narrow-gap or inverted-band materials (HgTe etc.) "
        "without higher-fidelity electronic structure."
    )
    short_description = "Brus effective-mass QD transition energy (L1)."
    exposure = ToolExposure.DEFERRED
    namespace = "qd"
    source = "native analytical solver"
    tags = ("quantum dot", "confinement", "brus", "effective mass", "band gap")
    input_schema = {
        "type": "object",
        "properties": {
            "radius_nm": {"type": "number", "minimum": 0.1},
            "bulk_band_gap_eV": {"type": "number"},
            "electron_effective_mass_m0": {"type": "number", "minimum": 0.001},
            "hole_effective_mass_m0": {"type": "number", "minimum": 0.001},
            "relative_dielectric_constant": {"type": "number", "minimum": 1.0},
            "include_coulomb_term": {"type": "boolean"},
        },
        "required": [
            "radius_nm",
            "bulk_band_gap_eV",
            "electron_effective_mass_m0",
            "hole_effective_mass_m0",
        ],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        failure = _params_or_failure(
            arguments,
            tool=self.name,
            required=[
                ("radius_nm", "radius of the nanocrystal in nm"),
                ("bulk_band_gap_eV", "bulk band gap in eV"),
                ("electron_effective_mass_m0", "electron effective mass in m0"),
                ("hole_effective_mass_m0", "hole effective mass in m0"),
            ],
        )
        if failure:
            return ScientificToolResult(
                output=json.dumps(failure, ensure_ascii=False),
                is_error=True,
                data=failure,
            )
        try:
            result = transition_energy(
                radius_nm=float(arguments["radius_nm"]),
                bulk_band_gap_eV=float(arguments["bulk_band_gap_eV"]),
                electron_effective_mass_m0=float(
                    arguments["electron_effective_mass_m0"]
                ),
                hole_effective_mass_m0=float(arguments["hole_effective_mass_m0"]),
                relative_dielectric_constant=(
                    float(arguments["relative_dielectric_constant"])
                    if arguments.get("relative_dielectric_constant") is not None
                    else None
                ),
                include_coulomb_term=bool(
                    arguments.get("include_coulomb_term", True)
                ),
            )
        except (MissingScientificPrerequisite, ValueError) as exc:
            return ScientificToolResult(
                output=str(exc),
                is_error=True,
                data={"error_type": "invalid_input", "message": str(exc)},
            )
        evidence = [
            _evidence(
                subject="quantum_dot",
                property_name="transition_energy",
                value=result["transition_energy_eV"],
                unit="eV",
                method="Brus effective-mass model (L1)",
                limitations="; ".join(result["assumptions"]),
                provenance={
                    "tool": self.name,
                    "radius_nm": arguments["radius_nm"],
                    "bulk_band_gap_eV": arguments["bulk_band_gap_eV"],
                },
                summary=(
                    f"Brus L1 transition energy "
                    f"{result['transition_energy_eV']:.4f} eV "
                    f"({result['transition_wavelength_um']:.3f} um)"
                ),
            )
        ]
        payload = {key: value for key, value in result.items() if key != "assumptions"}
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False, indent=2),
            data=payload,
            evidence=evidence,
        )


class ExcitonicRegimeTool(Tool):
    name = "qd.excitonic_regime"
    description = (
        "Estimate the exciton Bohr radius a_B* = 4 pi eps0 epsr hbar^2/(e^2 mu) "
        "and report R/a_B* with a strong/intermediate/weak confinement "
        "diagnostic. Requires electron/hole effective masses and relative "
        "dielectric constant. Use to judge whether the strong-confinement "
        "assumption of qd.brus_transition_energy holds."
    )
    short_description = "Exciton Bohr radius and confinement regime diagnostic."
    exposure = ToolExposure.DEFERRED
    namespace = "qd"
    source = "native analytical solver"
    tags = ("quantum dot", "exciton", "bohr radius", "confinement")
    input_schema = {
        "type": "object",
        "properties": {
            "radius_nm": {"type": "number", "minimum": 0.1},
            "electron_effective_mass_m0": {"type": "number", "minimum": 0.001},
            "hole_effective_mass_m0": {"type": "number", "minimum": 0.001},
            "relative_dielectric_constant": {"type": "number", "minimum": 1.0},
        },
        "required": [
            "radius_nm",
            "electron_effective_mass_m0",
            "hole_effective_mass_m0",
            "relative_dielectric_constant",
        ],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        try:
            result = excitonic_regime(
                radius_nm=float(arguments["radius_nm"]),
                electron_effective_mass_m0=float(
                    arguments["electron_effective_mass_m0"]
                ),
                hole_effective_mass_m0=float(arguments["hole_effective_mass_m0"]),
                relative_dielectric_constant=float(
                    arguments["relative_dielectric_constant"]
                ),
            )
        except (MissingScientificPrerequisite, ValueError) as exc:
            return ScientificToolResult(
                output=str(exc), is_error=True, data={"error_type": "invalid_input"}
            )
        return ScientificToolResult(
            output=json.dumps(result, ensure_ascii=False),
            data=result,
        )


class SolveSizeForTransitionTool(Tool):
    name = "qd.solve_size_for_transition"
    description = (
        "Invert the Brus model: find the nanocrystal radius whose L1 transition "
        "energy equals target_energy_eV OR target_wavelength_um (exactly one). "
        "Uses numerical bisection on the monotonic strong-confinement branch; "
        "returns NO_PHYSICAL_SOLUTION for targets at/below the bulk gap. "
        "Requires bulk_band_gap_eV, electron/hole effective masses, and (if "
        "include_coulomb_term) relative_dielectric_constant."
    )
    short_description = "Solve QD size for a target transition energy/wavelength."
    exposure = ToolExposure.DEFERRED
    namespace = "qd"
    source = "native analytical solver"
    tags = ("quantum dot", "inverse design", "size", "brus")
    input_schema = {
        "type": "object",
        "properties": {
            "target_energy_eV": {"type": "number", "minimum": 0},
            "target_wavelength_um": {"type": "number", "minimum": 0},
            "bulk_band_gap_eV": {"type": "number"},
            "electron_effective_mass_m0": {"type": "number", "minimum": 0.001},
            "hole_effective_mass_m0": {"type": "number", "minimum": 0.001},
            "relative_dielectric_constant": {"type": "number", "minimum": 1.0},
            "include_coulomb_term": {"type": "boolean"},
        },
        "required": [
            "bulk_band_gap_eV",
            "electron_effective_mass_m0",
            "hole_effective_mass_m0",
        ],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        try:
            result = solve_size_for_transition(
                target_energy_eV=(
                    float(arguments["target_energy_eV"])
                    if arguments.get("target_energy_eV") is not None
                    else None
                ),
                target_wavelength_um=(
                    float(arguments["target_wavelength_um"])
                    if arguments.get("target_wavelength_um") is not None
                    else None
                ),
                bulk_band_gap_eV=float(arguments["bulk_band_gap_eV"]),
                electron_effective_mass_m0=float(
                    arguments["electron_effective_mass_m0"]
                ),
                hole_effective_mass_m0=float(arguments["hole_effective_mass_m0"]),
                relative_dielectric_constant=(
                    float(arguments["relative_dielectric_constant"])
                    if arguments.get("relative_dielectric_constant") is not None
                    else None
                ),
                include_coulomb_term=bool(
                    arguments.get("include_coulomb_term", True)
                ),
            )
        except (MissingScientificPrerequisite, ValueError) as exc:
            return ScientificToolResult(
                output=str(exc), is_error=True, data={"error_type": "invalid_input"}
            )
        outcome = result.get("outcome")
        evidence = []
        if outcome == "SOLVED":
            evidence.append(
                _evidence(
                    subject="quantum_dot",
                    property_name="candidate_radius",
                    value=result["candidate_radius_nm"],
                    unit="nm",
                    method="Brus model inverse solve (bisection, L1)",
                    limitations="L1 effective-mass estimate; not a design value",
                    provenance={"tool": self.name, "outcome": outcome},
                    summary=(
                        f"L1 candidate radius {result['candidate_radius_nm']:.3f} nm "
                        f"for transition at {result['predicted_transition_wavelength_um']:.3f} um"
                    ),
                )
            )
        is_error = outcome != "SOLVED"
        return ScientificToolResult(
            output=json.dumps(result, ensure_ascii=False, indent=2),
            is_error=is_error,
            data=result,
            evidence=evidence,
        )


class SizeSweepTool(Tool):
    name = "qd.size_sweep"
    description = (
        "Deterministic Brus size sweep: returns a bounded table of "
        "size -> transition energy/wavelength plus summary stats. Large "
        "sweeps are downsampled (max 100 rows). Same parameters and L1 "
        "caveats as qd.brus_transition_energy."
    )
    short_description = "Brus size sweep (size -> energy/wavelength table)."
    exposure = ToolExposure.DEFERRED
    namespace = "qd"
    source = "native analytical solver"
    tags = ("quantum dot", "sweep", "size", "confinement")
    input_schema = {
        "type": "object",
        "properties": {
            "min_size_nm": {"type": "number", "minimum": 0.1},
            "max_size_nm": {"type": "number", "minimum": 0.1},
            "points": {"type": "integer", "minimum": 2, "maximum": 1000},
            "bulk_band_gap_eV": {"type": "number"},
            "electron_effective_mass_m0": {"type": "number", "minimum": 0.001},
            "hole_effective_mass_m0": {"type": "number", "minimum": 0.001},
            "relative_dielectric_constant": {"type": "number", "minimum": 1.0},
            "include_coulomb_term": {"type": "boolean"},
        },
        "required": [
            "min_size_nm",
            "max_size_nm",
            "points",
            "bulk_band_gap_eV",
            "electron_effective_mass_m0",
            "hole_effective_mass_m0",
        ],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        try:
            result = size_sweep(
                min_size_nm=float(arguments["min_size_nm"]),
                max_size_nm=float(arguments["max_size_nm"]),
                points=int(arguments["points"]),
                bulk_band_gap_eV=float(arguments["bulk_band_gap_eV"]),
                electron_effective_mass_m0=float(
                    arguments["electron_effective_mass_m0"]
                ),
                hole_effective_mass_m0=float(arguments["hole_effective_mass_m0"]),
                relative_dielectric_constant=(
                    float(arguments["relative_dielectric_constant"])
                    if arguments.get("relative_dielectric_constant") is not None
                    else None
                ),
                include_coulomb_term=bool(
                    arguments.get("include_coulomb_term", True)
                ),
            )
        except (MissingScientificPrerequisite, ValueError) as exc:
            return ScientificToolResult(
                output=str(exc), is_error=True, data={"error_type": "invalid_input"}
            )
        return ScientificToolResult(
            output=json.dumps(result, ensure_ascii=False, indent=2),
            data=result,
        )


class ScreenSizeCompositionTool(Tool):
    name = "qd.screen_size_composition"
    description = (
        "Grid-search composition x and radius pairs whose L1 transition lands "
        "in [target_wavelength_min_um, target_wavelength_max_um]. Uses generic "
        "quadratic bowing for Eg(x) and Brus confinement with linearly "
        "interpolated masses. ALL material parameters are required (no "
        "defaults; missing inputs return missing_prerequisites). Candidates "
        "are L1 estimates only."
    )
    short_description = "Composition x size screening for a target window (L1)."
    exposure = ToolExposure.DEFERRED
    namespace = "qd"
    source = "native analytical solver"
    tags = ("quantum dot", "screening", "composition", "size", "bowing")
    input_schema = {
        "type": "object",
        "properties": {
            "target_wavelength_min_um": {"type": "number", "minimum": 0.01},
            "target_wavelength_max_um": {"type": "number", "minimum": 0.01},
            "composition_min": {"type": "number", "minimum": 0, "maximum": 1},
            "composition_max": {"type": "number", "minimum": 0, "maximum": 1},
            "composition_points": {"type": "integer", "minimum": 2, "maximum": 200},
            "radius_min_nm": {"type": "number", "minimum": 0.1},
            "radius_max_nm": {"type": "number", "minimum": 0.1},
            "radius_points": {"type": "integer", "minimum": 2, "maximum": 200},
            "band_gap_a_eV": {"type": "number"},
            "band_gap_b_eV": {"type": "number"},
            "bowing_parameter_eV": {"type": "number"},
            "electron_mass_a_m0": {"type": "number", "minimum": 0.001},
            "hole_mass_a_m0": {"type": "number", "minimum": 0.001},
            "electron_mass_b_m0": {"type": "number", "minimum": 0.001},
            "hole_mass_b_m0": {"type": "number", "minimum": 0.001},
            "relative_dielectric_constant": {"type": "number", "minimum": 1.0},
            "include_coulomb_term": {"type": "boolean"},
        },
        "required": [
            "target_wavelength_min_um",
            "target_wavelength_max_um",
            "composition_min",
            "composition_max",
            "composition_points",
            "radius_min_nm",
            "radius_max_nm",
            "radius_points",
            "band_gap_a_eV",
            "band_gap_b_eV",
            "bowing_parameter_eV",
            "electron_mass_a_m0",
            "hole_mass_a_m0",
            "electron_mass_b_m0",
            "hole_mass_b_m0",
            "relative_dielectric_constant",
        ],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        try:
            result = screen_size_composition(
                target_wavelength_min_um=float(
                    arguments["target_wavelength_min_um"]
                ),
                target_wavelength_max_um=float(
                    arguments["target_wavelength_max_um"]
                ),
                composition_min=float(arguments["composition_min"]),
                composition_max=float(arguments["composition_max"]),
                composition_points=int(arguments["composition_points"]),
                radius_min_nm=float(arguments["radius_min_nm"]),
                radius_max_nm=float(arguments["radius_max_nm"]),
                radius_points=int(arguments["radius_points"]),
                band_gap_a_eV=float(arguments["band_gap_a_eV"]),
                band_gap_b_eV=float(arguments["band_gap_b_eV"]),
                bowing_parameter_eV=float(arguments["bowing_parameter_eV"]),
                electron_mass_a_m0=float(arguments["electron_mass_a_m0"]),
                hole_mass_a_m0=float(arguments["hole_mass_a_m0"]),
                electron_mass_b_m0=float(arguments["electron_mass_b_m0"]),
                hole_mass_b_m0=float(arguments["hole_mass_b_m0"]),
                relative_dielectric_constant=float(
                    arguments["relative_dielectric_constant"]
                ),
                include_coulomb_term=bool(
                    arguments.get("include_coulomb_term", True)
                ),
            )
        except (MissingScientificPrerequisite, ValueError) as exc:
            if isinstance(exc, MissingScientificPrerequisite):
                failure = prerequisite_failure(
                    str(exc), missing=exc.missing, tool=self.name
                )
            else:
                failure = {"error_type": "invalid_input", "message": str(exc)}
            return ScientificToolResult(
                output=json.dumps(failure, ensure_ascii=False),
                is_error=True,
                data=failure,
            )
        evidence = []
        for candidate in result["selected_candidates"][:5]:
            evidence.append(
                _evidence(
                    subject="quantum_dot_screen",
                    property_name="transition_wavelength",
                    value=candidate["transition_wavelength_um"],
                    unit="um",
                    method="bowing + Brus grid screening (L1)",
                    limitations="L1 analytical estimate; verify with higher fidelity",
                    provenance={
                        "tool": self.name,
                        "composition_x": candidate["composition_x"],
                        "radius_nm": candidate["radius_nm"],
                    },
                    summary=(
                        f"candidate x={candidate['composition_x']:.3f}, "
                        f"R={candidate['radius_nm']:.2f} nm -> "
                        f"{candidate['transition_wavelength_um']:.3f} um"
                    ),
                )
            )
        return ScientificToolResult(
            output=json.dumps(result, ensure_ascii=False, indent=2),
            data=result,
            evidence=evidence,
        )


class BandgapBowingTool(Tool):
    name = "alloy.bandgap_bowing"
    description = (
        "Deterministic alloy band gap Eg(x) = (1-x)EgA + x EgB - b x(1-x). "
        "All coefficients are explicit inputs (endpoint gaps, bowing "
        "parameter, optional Varshni temperature shift). The tool does not "
        "hardcode material constants; provenance must be supplied by the "
        "caller or via qd.parameter_lookup."
    )
    short_description = "Alloy band-gap bowing Eg(x)."
    exposure = ToolExposure.DEFERRED
    namespace = "alloy"
    source = "native analytical solver"
    tags = ("alloy", "bowing", "band gap", "composition")
    input_schema = {
        "type": "object",
        "properties": {
            "x": {"type": "number", "minimum": 0, "maximum": 1},
            "band_gap_a_eV": {"type": "number"},
            "band_gap_b_eV": {"type": "number"},
            "bowing_parameter_eV": {"type": "number"},
            "temperature_k": {"type": "number", "minimum": 0},
            "varshni_alpha_a": {"type": "number"},
            "varshni_beta_a": {"type": "number"},
            "varshni_alpha_b": {"type": "number"},
            "varshni_beta_b": {"type": "number"},
        },
        "required": [
            "x",
            "band_gap_a_eV",
            "band_gap_b_eV",
            "bowing_parameter_eV",
        ],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        try:
            result = bandgap_bowing(
                x=float(arguments["x"]),
                band_gap_a_eV=float(arguments["band_gap_a_eV"]),
                band_gap_b_eV=float(arguments["band_gap_b_eV"]),
                bowing_parameter_eV=float(arguments["bowing_parameter_eV"]),
                temperature_k=(
                    float(arguments["temperature_k"])
                    if arguments.get("temperature_k") is not None
                    else None
                ),
                varshni_alpha_a=(
                    float(arguments["varshni_alpha_a"])
                    if arguments.get("varshni_alpha_a") is not None
                    else None
                ),
                varshni_beta_a=(
                    float(arguments["varshni_beta_a"])
                    if arguments.get("varshni_beta_a") is not None
                    else None
                ),
                varshni_alpha_b=(
                    float(arguments["varshni_alpha_b"])
                    if arguments.get("varshni_alpha_b") is not None
                    else None
                ),
                varshni_beta_b=(
                    float(arguments["varshni_beta_b"])
                    if arguments.get("varshni_beta_b") is not None
                    else None
                ),
            )
        except (MissingScientificPrerequisite, ValueError) as exc:
            if isinstance(exc, MissingScientificPrerequisite):
                failure = prerequisite_failure(
                    str(exc), missing=exc.missing, tool=self.name
                )
            else:
                failure = {"error_type": "invalid_input", "message": str(exc)}
            return ScientificToolResult(
                output=json.dumps(failure, ensure_ascii=False),
                is_error=True,
                data=failure,
            )
        evidence = [
            _evidence(
                subject="alloy",
                property_name="band_gap",
                value=result["band_gap_eV"],
                unit="eV",
                method="quadratic bowing model",
                limitations="empirical; validity limited to sourced x range",
                provenance={
                    "tool": self.name,
                    "x": arguments["x"],
                    "band_gap_a_eV": arguments["band_gap_a_eV"],
                    "band_gap_b_eV": arguments["band_gap_b_eV"],
                    "bowing_parameter_eV": arguments["bowing_parameter_eV"],
                },
                fidelity="empirical",
                summary=f"bowed band gap at x={result['composition_x']:.3f}: {result['band_gap_eV']:.4f} eV",
            )
        ]
        return ScientificToolResult(
            output=json.dumps(result, ensure_ascii=False, indent=2),
            data=result,
            evidence=evidence,
        )


class ParameterLookupTool(Tool):
    name = "qd.parameter_lookup"
    description = (
        "Look up curated, sourced material parameters (band gap, effective "
        "masses, dielectric constant) for a small set of example/test "
        "materials (InAs, PbTe, HgTe, GaAs). Every entry carries source, "
        "method, temperature, uncertainty, and a confidence flag. Entries "
        "flagged 'example' must NOT be used as design-grade values without "
        "verification. Unknown materials return an empty result, never a guess."
    )
    short_description = "Curated sourced material parameters (example data)."
    exposure = ToolExposure.DEFERRED
    namespace = "qd"
    source = "local parameter registry (sourced examples)"
    tags = ("parameters", "effective mass", "band gap", "registry", "provenance")
    input_schema = {
        "type": "object",
        "properties": {
            "material": {"type": "string"},
            "property": {
                "type": "string",
                "enum": [
                    "band_gap",
                    "electron_effective_mass",
                    "hole_effective_mass",
                    "relative_dielectric_constant",
                ],
            },
        },
        "required": ["material"],
    }

    def __init__(self) -> None:
        self._registry: MaterialParameterRegistry | None = None

    def _registry_for(self) -> MaterialParameterRegistry:
        if self._registry is None:
            self._registry = default_registry()
        return self._registry

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        material = str(arguments["material"])
        property_name = (
            str(arguments["property"]) if arguments.get("property") else None
        )
        registry = self._registry_for()
        if property_name:
            parameter = registry.get(material, property_name)
            if parameter is None:
                return ScientificToolResult(
                    output=json.dumps(
                        {
                            "material": material,
                            "property": property_name,
                            "found": False,
                            "available_materials": registry.materials(),
                            "note": "no entry; do not guess parameters",
                        },
                        ensure_ascii=False,
                    ),
                    data={"found": False},
                )
            payload = parameter.to_evidence_dict()
            evidence = [
                ScientificEvidence(
                    subject=material,
                    property=property_name,
                    value=parameter.value,
                    unit=parameter.unit,
                    source=parameter.source,
                    source_type="user_parameter",
                    method=parameter.method or "curated reference value",
                    fidelity="empirical",
                    summary=f"{material} {property_name} = {parameter.value} {parameter.unit}",
                    limitations=parameter.validity or "verify before design use",
                    provenance={
                        "tool": self.name,
                        "confidence": parameter.confidence,
                        "temperature_k": parameter.temperature_k,
                    },
                )
            ]
            return ScientificToolResult(
                output=json.dumps(payload, ensure_ascii=False, indent=2),
                data=payload,
                evidence=evidence,
            )
        parameters = registry.all(material)
        if not parameters:
            return ScientificToolResult(
                output=json.dumps(
                    {
                        "material": material,
                        "found": False,
                        "available_materials": registry.materials(),
                    },
                    ensure_ascii=False,
                ),
                data={"found": False},
            )
        payload_list = [p.to_evidence_dict() for p in parameters]
        return ScientificToolResult(
            output=json.dumps(payload_list, ensure_ascii=False, indent=2),
            data={"material": material, "parameters": payload_list},
        )


def quantum_dot_pack() -> CapabilityPack:
    return QuantumDotProbe()


def alloy_pack() -> CapabilityPack:
    return AlloyProbe()
