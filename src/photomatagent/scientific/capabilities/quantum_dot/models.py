"""Quantum-dot parameter provenance models and a tiny sourced registry.

Policy (Sprint 2, section 27): parameters may only come from user input,
database tools, verified local reference tables, or external solver output.
The registry below deliberately holds only a handful of clearly sourced
example/test entries -- never a large unverified database. Entries flagged
``example`` must not be used as design-grade values without verification.

Sprint 3 typing: ``ScientificParameter.kind`` carries the physical nature of
a parameter. For dielectric constants the kind is one of ``static``,
``optical``, ``high_frequency``, or ``unknown``. Solvers that require a
specific screening regime must declare it and refuse to compute silently
when the supplied kind is incompatible (typed
``INCOMPATIBLE_SCIENTIFIC_PARAMETER`` diagnostic).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DielectricKind(str, Enum):
    """Physical regime of a relative dielectric constant."""

    STATIC = "static"
    OPTICAL = "optical"
    HIGH_FREQUENCY = "high_frequency"
    UNKNOWN = "unknown"


class ScientificParameter(BaseModel):
    """One material parameter with full provenance."""

    name: str
    value: float
    unit: str
    source: str
    method: str = ""
    temperature_k: float | None = None
    uncertainty: float | None = None
    kind: str = ""
    frequency_regime: str = ""
    reference: str = ""
    notes: str = ""
    validity: str = ""
    confidence: str = "example"  # example | established

    def to_evidence_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
            "method": self.method,
            "temperature_k": self.temperature_k,
            "uncertainty": self.uncertainty,
            "kind": self.kind,
            "frequency_regime": self.frequency_regime,
            "reference": self.reference,
            "notes": self.notes,
            "validity": self.validity,
            "confidence": self.confidence,
        }

    @property
    def dielectric_kind(self) -> DielectricKind:
        """Normalized dielectric kind; ``UNKNOWN`` for non-dielectric uses."""
        normalized = self.kind.strip().lower().replace("-", "_")
        if normalized == "static":
            return DielectricKind.STATIC
        if normalized in {"optical", "optic"}:
            return DielectricKind.OPTICAL
        if normalized in {"high_frequency", "high_frequency_optical", "infrared"}:
            return DielectricKind.HIGH_FREQUENCY
        return DielectricKind.UNKNOWN


class MaterialParameterRegistry:
    """Lookup for curated, sourced parameters (example data + test fixtures).

    ``get`` returns ``None`` for unknown properties so tools can emit a typed
    ``missing_prerequisites`` instead of guessing.
    """

    def __init__(self, entries: dict[str, list[ScientificParameter]] | None = None) -> None:
        self._entries: dict[str, list[ScientificParameter]] = dict(entries or {})

    def register(self, material: str, parameters: list[ScientificParameter]) -> None:
        self._entries[material.casefold()] = list(parameters)

    def get(self, material: str, property_name: str) -> ScientificParameter | None:
        for parameter in self._entries.get(material.casefold(), []):
            if parameter.name == property_name:
                return parameter
        return None

    def all(self, material: str) -> list[ScientificParameter]:
        return list(self._entries.get(material.casefold(), []))

    def materials(self) -> list[str]:
        return sorted(self._entries)


# ---------------------------------------------------------------------------
# Curated example entries. All sources are named; none of these should be
# consumed as design-grade without checking the cited literature.
# ---------------------------------------------------------------------------


def default_registry() -> MaterialParameterRegistry:
    registry = MaterialParameterRegistry()
    registry.register(
        "InAs",
        [
            ScientificParameter(
                name="band_gap",
                value=0.354,
                unit="eV",
                temperature_k=300.0,
                source="Vurgaftman & Meyer, J. Appl. Phys. 89, 5815 (2001), Table I",
                method="compilation of experimental and k.p data",
                uncertainty=0.01,
                validity="300 K, bulk zincblende InAs",
                confidence="established",
            ),
            ScientificParameter(
                name="electron_effective_mass",
                value=0.026,
                unit="m0",
                temperature_k=300.0,
                source="Vurgaftman & Meyer, J. Appl. Phys. 89, 5815 (2001), Table I",
                method="k.p (Gamma valley)",
                uncertainty=0.002,
                validity="Gamma valley, parabolic approximation",
                confidence="established",
            ),
            ScientificParameter(
                name="hole_effective_mass",
                value=0.41,
                unit="m0",
                temperature_k=300.0,
                source="Vurgaftman & Meyer, J. Appl. Phys. 89, 5815 (2001), Table I",
                method="heavy-hole band, k.p",
                uncertainty=0.03,
                validity="heavy-hole mass; light-hole differs",
                confidence="established",
            ),
            ScientificParameter(
                name="relative_dielectric_constant",
                value=15.15,
                unit="",
                temperature_k=300.0,
                source="Adachi, Optical Constants of Crystalline and Amorphous "
                "Semiconductors (Springer, 2004)",
                method="optical measurement compilation",
                uncertainty=0.2,
                confidence="established",
            ),
        ],
    )
    registry.register(
        "PbTe",
        [
            ScientificParameter(
                name="band_gap",
                value=0.31,
                unit="eV",
                temperature_k=300.0,
                source="Ravich et al., Semiconducting Lead Chalcogenides (1970); "
                "reviewed in Rogalski, Infrared Detectors (2011)",
                method="optical absorption compilation",
                uncertainty=0.02,
                validity="300 K, L-point direct gap",
                confidence="established",
            ),
            ScientificParameter(
                name="electron_effective_mass",
                value=0.20,
                unit="m0",
                temperature_k=300.0,
                source="average of L-valley masses, Rogalski, Infrared Detectors "
                "(2011); masses are anisotropic (4 equivalent L valleys)",
                method="magneto-optical / Shubnikov-de Haas compilation",
                uncertainty=0.04,
                notes="scalar EMA is a strong simplification for PbTe",
                validity="isotropic scalar approximation only",
                confidence="example",
            ),
            ScientificParameter(
                name="hole_effective_mass",
                value=0.24,
                unit="m0",
                temperature_k=300.0,
                source="average of L-valley masses, Rogalski, Infrared Detectors "
                "(2011); anisotropic",
                method="compilation",
                uncertainty=0.05,
                notes="scalar EMA is a strong simplification for PbTe",
                validity="isotropic scalar approximation only",
                confidence="example",
            ),
            ScientificParameter(
                name="relative_dielectric_constant",
                value=414.0,
                unit="",
                temperature_k=300.0,
                source="Ravich et al. (1970); very large static dielectric "
                "constant of PbTe",
                method="capacitance compilation",
                uncertainty=30.0,
                validity="static (low-frequency) value",
                confidence="example",
            ),
        ],
    )
    registry.register(
        "HgTe",
        [
            ScientificParameter(
                name="band_gap",
                value=-0.302,
                unit="eV",
                temperature_k=300.0,
                source="Hansen, Schmit & Casselman, J. Appl. Phys. 53, 7099 "
                "(1982) Hg1-xCdxTe gap relation at x=0",
                method="empirical relation (inverted semimetal gap)",
                uncertainty=0.02,
                notes="negative gap: HgTe is an inverted narrow-gap semimetal; "
                "EMA/Brus low fidelity for this system",
                validity="x=0 endpoint of HgCdTe system",
                confidence="example",
            ),
            ScientificParameter(
                name="relative_dielectric_constant",
                value=21.0,
                unit="",
                temperature_k=300.0,
                source="Orlowski et al. / HgCdTe literature compilation",
                method="compilation",
                uncertainty=2.0,
                confidence="example",
            ),
        ],
    )
    registry.register(
        "GaAs",
        [
            ScientificParameter(
                name="band_gap",
                value=1.424,
                unit="eV",
                temperature_k=300.0,
                source="Vurgaftman & Meyer, J. Appl. Phys. 89, 5815 (2001)",
                method="compilation",
                confidence="established",
            ),
            ScientificParameter(
                name="electron_effective_mass",
                value=0.067,
                unit="m0",
                temperature_k=300.0,
                source="Vurgaftman & Meyer, J. Appl. Phys. 89, 5815 (2001)",
                method="k.p (Gamma valley)",
                confidence="established",
            ),
            ScientificParameter(
                name="hole_effective_mass",
                value=0.45,
                unit="m0",
                temperature_k=300.0,
                source="Vurgaftman & Meyer, J. Appl. Phys. 89, 5815 (2001)",
                method="heavy-hole band, k.p",
                confidence="established",
            ),
            ScientificParameter(
                name="relative_dielectric_constant",
                value=12.9,
                unit="",
                temperature_k=300.0,
                source="Adachi (2004)",
                method="compilation",
                confidence="established",
            ),
        ],
    )
    return registry
