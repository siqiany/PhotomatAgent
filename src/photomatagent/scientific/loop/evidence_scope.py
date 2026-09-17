"""Evidence applicability and deliberately bounded property-unit conversions."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.discovery.composition import normalize_composition
from photomatagent.scientific.loop.candidate import CandidateState

EvidenceScope = Literal["composition", "structure", "device"]
EvidenceAuthority = Literal["observation", "synthetic", "background"]


class EvidenceRequirements(BaseModel):
    """Host-defined scope required before an observation can check a property."""

    model_config = ConfigDict(frozen=True)

    scope: EvidenceScope = "composition"
    allowed_fidelities: tuple[str, ...] | None = None
    allowed_source_types: tuple[str, ...] | None = None
    allowed_sources: tuple[str, ...] | None = None
    conditions: dict[str, object] = Field(default_factory=dict)
    required_conditions: tuple[str, ...] = ()
    require_unit: bool = True


def evidence_applicable(
    evidence: ScientificEvidence,
    candidate: CandidateState,
    requirements: EvidenceRequirements,
    *,
    authority: EvidenceAuthority = "background",
    allow_synthetic_evidence: bool = False,
) -> tuple[bool, str]:
    """Check authority and scope before an evidence value is inspected.

    The ordering is intentional. A proposal cannot become authoritative by
    supplying a plausible identity, unit, fidelity, or numeric value.
    """

    if authority == "synthetic" and allow_synthetic_evidence:
        pass
    elif authority != "observation":
        return False, "EVIDENCE_UNATTESTED"
    if evidence.assessment_role != "observation":
        return False, "ROLE_BACKGROUND"

    source_type = evidence.source_type.casefold()
    fidelity = evidence.fidelity.casefold()
    source = evidence.source.strip().casefold()
    tool_name = str(evidence.provenance.get("tool", "")).strip().casefold()
    if source_type == "generative_model":
        return False, "SOURCE_GENERATED"
    if fidelity == "ml_generated":
        return False, "SOURCE_GENERATED"
    if source == "candidate_declared" or source.startswith("candidate_declared:"):
        return False, "SOURCE_CANDIDATE_DECLARED"
    if source == "generation" or source.startswith("generation."):
        return False, "SOURCE_GENERATED"
    if tool_name.startswith(("generation.", "vae.")):
        return False, "SOURCE_GENERATED"

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
            return False, "SOURCE_TYPE_NOT_ALLOWED"
    if requirements.allowed_sources is not None:
        allowed_sources = {item.strip().casefold() for item in requirements.allowed_sources}
        if evidence.source.strip().casefold() not in allowed_sources:
            return False, "SOURCE_NOT_ALLOWED"
    if requirements.allowed_fidelities is not None:
        allowed = {item.strip().casefold() for item in requirements.allowed_fidelities}
        if fidelity not in allowed:
            return False, "FIDELITY_NOT_ALLOWED"

    if evidence.candidate_id and evidence.candidate_id != candidate.candidate_id:
        return False, "CANDIDATE_ID_MISMATCH"

    subject_result = _subject_matches_candidate(evidence.subject, candidate)
    if subject_result is False:
        return False, "COMPOSITION_MISMATCH"
    if subject_result is None:
        return False, "SUBJECT_UNRESOLVED"

    if requirements.scope in {"structure", "device"}:
        candidate_structure = _candidate_structure_hash(candidate)
        if not candidate_structure:
            return False, "CANDIDATE_STRUCTURE_MISSING"
        if not evidence.structure_hash:
            return False, "EVIDENCE_STRUCTURE_MISSING"
        if evidence.structure_hash != candidate_structure:
            return False, "STRUCTURE_MISMATCH"

    for name in requirements.required_conditions:
        if name not in evidence.conditions:
            return False, f"CONDITION_MISSING:{name[:64]}"

    for name, expected in requirements.conditions.items():
        if name not in evidence.conditions:
            return False, f"CONDITION_MISSING:{name[:64]}"
        if not _condition_equal(evidence.conditions[name], expected):
            return False, f"CONDITION_MISMATCH:{name[:64]}"

    if requirements.require_unit and not evidence.unit.strip():
        return False, "UNIT_MISSING"
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
        return "SOURCE_SYNTHETIC"
    if source == "mock" or source.startswith("mock:") or tool_name.startswith("mock."):
        return "SOURCE_MOCK"
    if source.startswith("test-only") or tool_name.startswith("test."):
        return "SOURCE_SYNTHETIC"
    if evidence.provenance.get("synthetic") is True:
        return "SOURCE_SYNTHETIC"
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
