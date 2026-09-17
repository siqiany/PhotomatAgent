"""Evidence applicability and deliberately bounded property-unit conversions."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.discovery.composition import normalize_composition
from photomatagent.scientific.loop.candidate import CandidateState

EvidenceScope = Literal["composition", "structure", "device"]


class EvidenceRequirements(BaseModel):
    """Host-defined scope required before an observation can check a property."""

    model_config = ConfigDict(frozen=True)

    scope: EvidenceScope = "composition"
    allowed_fidelities: tuple[str, ...] | None = None
    allowed_source_types: tuple[str, ...] | None = None
    allowed_sources: tuple[str, ...] | None = None
    conditions: dict[str, object] = Field(default_factory=dict)
    require_unit: bool = True


def evidence_applicable(
    evidence: ScientificEvidence,
    candidate: CandidateState,
    requirements: EvidenceRequirements,
    *,
    allow_synthetic_evidence: bool = False,
) -> tuple[bool, str]:
    """Check authority and scope before an evidence value is inspected.

    The ordering is intentional. A proposal cannot become authoritative by
    supplying a plausible identity, unit, fidelity, or numeric value.
    """

    if evidence.assessment_role != "observation":
        return False, f"assessment role {evidence.assessment_role!r} is background only"

    source_type = evidence.source_type.casefold()
    fidelity = evidence.fidelity.casefold()
    source = evidence.source.strip().casefold()
    tool_name = str(evidence.provenance.get("tool", "")).strip().casefold()
    if source_type == "generative_model":
        return False, "generative_model evidence is a proposal"
    if fidelity == "ml_generated":
        return False, "ml_generated evidence is generated, not observed"
    if source == "candidate_declared" or source.startswith("candidate_declared:"):
        return False, "candidate-declared evidence is a proposal"
    if source == "generation" or source.startswith("generation."):
        return False, f"generation source {evidence.source!r} is a proposal"
    if tool_name.startswith(("generation.", "vae.")):
        return False, f"generation source {tool_name!r} is a proposal"

    synthetic_reason = _synthetic_source_reason(evidence)
    if synthetic_reason and (
        not allow_synthetic_evidence or _is_mock_source(evidence)
    ):
        return False, synthetic_reason

    if requirements.allowed_source_types is not None:
        allowed_types = {
            item.strip().casefold() for item in requirements.allowed_source_types
        }
        if source_type not in allowed_types:
            return False, f"evidence source_type {evidence.source_type!r} is not allowed"
    if requirements.allowed_sources is not None:
        allowed_sources = {item.strip().casefold() for item in requirements.allowed_sources}
        if evidence.source.strip().casefold() not in allowed_sources:
            return False, f"evidence source {evidence.source!r} is not allowed"
    if requirements.allowed_fidelities is not None:
        allowed = {item.strip().casefold() for item in requirements.allowed_fidelities}
        if fidelity not in allowed:
            return False, f"evidence fidelity {evidence.fidelity!r} is not allowed"

    if evidence.candidate_id and evidence.candidate_id != candidate.candidate_id:
        return False, "candidate_id does not match the evaluated candidate"

    subject_result = _subject_matches_candidate(evidence.subject, candidate)
    if subject_result is False:
        return False, "evidence subject composition does not match candidate composition"
    if subject_result is None and requirements.scope != "composition":
        return False, "legacy evidence subject cannot establish structure or device identity"

    if requirements.scope in {"structure", "device"}:
        candidate_structure = _candidate_structure_hash(candidate)
        if not candidate_structure:
            return False, "candidate has no structure hash required by evidence scope"
        if not evidence.structure_hash:
            return False, "evidence has no structure hash required by evidence scope"
        if evidence.structure_hash != candidate_structure:
            return False, "evidence structure hash does not match candidate structure"

    for name, expected in requirements.conditions.items():
        if name not in evidence.conditions:
            return False, f"evidence condition {name!r} is missing"
        if not _condition_equal(evidence.conditions[name], expected):
            return False, f"evidence condition {name!r} does not match requirement"

    if requirements.require_unit and not evidence.unit.strip():
        return False, "evidence unit is required"
    return True, "applicable"


def convert_property_value(value: float, from_unit: str, to_unit: str) -> float:
    """Convert the deliberately small set of property units in the P0 design."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("property value must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("property value must be finite")

    conversions: dict[str, tuple[str, float]] = {
        "eV": ("energy", 1.0),
        "meV": ("energy", 1e-3),
        "um": ("length", 1.0),
        "nm": ("length", 1e-3),
        "eV/atom": ("energy_per_atom", 1.0),
        "meV/atom": ("energy_per_atom", 1e-3),
    }
    source = conversions.get(from_unit.strip())
    target = conversions.get(to_unit.strip())
    if source is None or target is None:
        raise ValueError(f"unsupported property-unit conversion: {from_unit!r} -> {to_unit!r}")
    if source[0] != target[0]:
        raise ValueError(f"incompatible property units: {from_unit!r} and {to_unit!r}")
    return numeric * source[1] / target[1]


def _synthetic_source_reason(evidence: ScientificEvidence) -> str:
    source = evidence.source.strip().casefold()
    tool_name = str(evidence.provenance.get("tool", "")).strip().casefold()
    if source == "synthetic" or source.startswith("synthetic:"):
        return "synthetic evidence is excluded from production evaluation"
    if source == "mock" or source.startswith("mock:") or tool_name.startswith("mock."):
        return "mock evidence is excluded from production evaluation"
    if source.startswith("test-only") or tool_name.startswith("test."):
        return "synthetic test evidence is excluded from production evaluation"
    if evidence.provenance.get("synthetic") is True:
        return "synthetic evidence is excluded from production evaluation"
    return ""


def _is_mock_source(evidence: ScientificEvidence) -> bool:
    source = evidence.source.strip().casefold()
    tool_name = str(evidence.provenance.get("tool", "")).strip().casefold()
    return source == "mock" or source.startswith("mock:") or tool_name.startswith("mock.")


def _subject_matches_candidate(subject: str, candidate: CandidateState) -> bool | None:
    subject = subject.strip()
    formula = candidate.formula.strip()
    if not subject or not formula:
        return None
    try:
        return normalize_composition(subject) == normalize_composition(formula)
    except (ValueError, RuntimeError):
        return None


def _candidate_structure_hash(candidate: CandidateState) -> str:
    representation = candidate.representation
    for key in ("structure_hash", "cif_hash", "structure_identifier", "structure_id"):
        value = representation.get(key)
        if value:
            return str(value)
    return ""


def _condition_equal(actual: object, expected: object) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12)
    return actual == expected
