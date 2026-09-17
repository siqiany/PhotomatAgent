"""ScientificEvaluator: the independent Checker.

First principle: anything that can be decided by data and rules is decided
here deterministically -- numeric threshold comparison, evidence presence,
fidelity ranking. The LLM (Maker) never grades its own candidate, and the
Maker's "final answer" never produces a scientific PASS by itself.

Property -> evidence mapping reads:
  1. ``ScientificEvidence`` in ScientificState whose property matches the
     constraint (with a documented alias table);
  2. eligible legacy structured JSON payloads stored in ``Evidence.content``.

Candidate declarations and generated/mock evidence are retained as background,
but cannot satisfy constraints.

Only scientific judgement that cannot be reduced to rules is left for a
future optional LLM critic -- this P0 stays fully deterministic.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.evidence import Evidence
from photomatagent.scientific.loop.candidate import (
    CandidateState,
    extract_json_payload,
)
from photomatagent.scientific.loop.evidence_scope import (
    EvidenceRequirements,
    convert_property_value,
    evidence_applicable,
)
from photomatagent.scientific.loop.scoring import compute_score
from photomatagent.scientific.loop.target import (
    ConstraintCheck,
    ConstraintOutcome,
    ConstraintSpec,
    ConstraintViolation,
    TargetSpec,
    evaluate_constraint,
)
from photomatagent.scientific.state import ScientificState

PropertyResult = Literal["PASS", "FAIL", "UNKNOWN"]
Verdict = Literal["PASS", "FAIL", "REVISE", "INCONCLUSIVE"]

# Documented, extensible evidence-fidelity ladder. Decision aid only: not
# every scientific question obeys the same ordering.
FIDELITY_RANK: dict[str, int] = {
    "ml_generated": 0,
    "analytical": 1,
    "empirical": 1,
    "continuum": 2,
    "kp": 2,
    "tight_binding": 2,
    "electromagnetic": 2,
    "ml_potential": 3,
    "namd": 3,
    "dft": 4,
    "experimental": 5,
}

# Default confidence implied by evidence fidelity (ScientificEvidence has no
# confidence field of its own; Evidence carries its own).
FIDELITY_CONFIDENCE: dict[str, float] = {
    "ml_generated": 0.25,
    "analytical": 0.55,
    "empirical": 0.55,
    "continuum": 0.60,
    "kp": 0.65,
    "tight_binding": 0.65,
    "electromagnetic": 0.70,
    "ml_potential": 0.70,
    "namd": 0.75,
    "dft": 0.85,
    "experimental": 0.92,
}

# Property aliases: raw tool property names -> canonical constraint property.
DEFAULT_PROPERTY_ALIASES: dict[str, set[str]] = {
    "band_gap": {"band_gap", "gap", "gap_selected_eV", "band_gap_eV", "bulk_band_gap_eV"},
    "responsivity": {"responsivity", "responsivity_a_w"},
    "quantum_efficiency": {"quantum_efficiency", "eqe", "eqe_fraction", "eqe_percent"},
    "formation_energy": {"formation_energy", "formation_energy_eV_per_atom"},
    "energy_above_hull": {"energy_above_hull", "energy_above_hull_eV_per_atom"},
    "density": {"density", "density_g_cm3"},
    "cutoff_wavelength": {"cutoff_wavelength", "cutoff_wavelength_um"},
    "detectivity": {"detectivity", "detectivity_jones"},
    "dark_current": {
        "dark_current",
        "dark_current_a",
        "dark_current_density_a_cm2",
    },
    "operating_temperature": {"operating_temperature", "temperature_k"},
    "effective_mass": {"effective_mass", "avg_electron_mass_m0", "avg_hole_mass_m0"},
}

_UNIT_SUFFIX = {
    "_eV": "eV",
    "_um": "um",
    "_a_w": "A/W",
    "_g_cm3": "g/cm3",
    "_k": "K",
    "_jones": "cm Hz^1/2/W",
}


def fidelity_rank(fidelity: str | None) -> int:
    """Rank evidence fidelity; unknown fidelities rank below everything."""
    if fidelity is None:
        return -1
    return FIDELITY_RANK.get(str(fidelity).strip().lower(), -1)


def evidence_confidence(evidence: Evidence | ScientificEvidence, fidelity: str | None = None) -> float:
    if isinstance(evidence, Evidence):
        return float(evidence.confidence)
    return FIDELITY_CONFIDENCE.get(str(fidelity or evidence.fidelity).lower(), 0.5)


@dataclass(frozen=True)
class _ResolvedEvidence:
    value: Any
    unit: str
    fidelity: str | None
    confidence: float
    evidence_id: str
    rank: int
    source: str
    subject: str | None = None
    method: str = ""
    conditions: dict[str, Any] = field(default_factory=dict)
    structure_hash: str = ""


@dataclass(frozen=True)
class EvidenceEvaluationPolicy:
    """Host-owned evaluator switches used by isolated deterministic tests."""

    allow_synthetic_evidence: bool = False


class PropertyEvaluation(BaseModel):
    """One constraint property evaluated against evidence."""

    property: str
    observed_value: Any = None
    unit: str = ""
    result: PropertyResult = "UNKNOWN"
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""


class EvaluationReport(BaseModel):
    """Full evaluation of one candidate against the target."""

    candidate_id: str = ""
    constraint_results: list[PropertyEvaluation] = Field(default_factory=list)
    violations: list[ConstraintViolation] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    critical_evidence_gaps: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    hard_constraints_passed: bool = False
    score: float = 0.0
    confidence: float = 0.0
    verdict: Verdict = "INCONCLUSIVE"
    rationale: str = ""

    def violation_for(self, property_name: str) -> ConstraintViolation | None:
        for violation in self.violations:
            if violation.property == property_name:
                return violation
        return None


class ScientificEvaluator:
    """Deterministic Checker: TargetSpec + candidate + ScientificState -> report."""

    def __init__(
        self,
        target: TargetSpec,
        *,
        property_aliases: dict[str, set[str]] | None = None,
        policy: EvidenceEvaluationPolicy | None = None,
    ) -> None:
        self.target = target
        self.policy = policy or EvidenceEvaluationPolicy()
        self.aliases = {
            **DEFAULT_PROPERTY_ALIASES,
            **(property_aliases or {}),
        }

    def evaluate(
        self,
        candidate: CandidateState | None,
        scientific: ScientificState,
    ) -> EvaluationReport:
        if candidate is None:
            gaps = sorted({c.property for c in self.target.constraints})
            return EvaluationReport(
                candidate_id="",
                evidence_gaps=gaps,
                critical_evidence_gaps=gaps,
                verdict="INCONCLUSIVE",
                rationale="no candidate could be constructed from structured scientific state",
            )
        outcomes: list[ConstraintOutcome] = []
        contradictions: list[str] = []
        used_confidences: list[float] = []
        for constraint in self.target.constraints:
            outcome = self._evaluate_constraint(constraint, candidate, scientific)
            outcomes.append(outcome)
            if outcome.confidence > 0.0:
                used_confidences.append(outcome.confidence)
        contradictions = self._detect_contradictions(candidate, scientific)

        violations = [
            ConstraintViolation.from_constraint(
                self.target.constraint(outcome.property) or ConstraintSpec(
                    property=outcome.property,
                    operator=outcome.operator,
                    value=outcome.target_value,
                    unit=outcome.unit,
                    severity=outcome.severity,
                ),
                observed_value=outcome.observed_value,
                evidence_ids=outcome.evidence_ids,
            )
            for outcome in outcomes
            if outcome.result == "FAIL"
        ]
        gaps = [o.property for o in outcomes if o.result == "UNKNOWN"]
        critical_gaps = [o.property for o in outcomes if o.result == "UNKNOWN" and o.severity == "HARD"]
        hard_constraints_passed = all(
            o.result == "PASS" for o in outcomes if o.severity == "HARD"
        )
        confidence = (
            round(sum(used_confidences) / len(used_confidences), 6)
            if used_confidences
            else 0.0
        )
        score = compute_score(
            target=self.target,
            outcomes=outcomes,
            overall_confidence=confidence,
        )
        verdict = _verdict(outcomes)
        rationale = _rationale(verdict, outcomes, gaps, contradictions)
        constraint_results = [
            PropertyEvaluation(
                property=o.property,
                observed_value=o.observed_value,
                unit=o.unit,
                result=o.result,
                evidence_ids=o.evidence_ids,
                confidence=o.confidence,
                reason=o.reason,
            )
            for o in outcomes
        ]
        return EvaluationReport(
            candidate_id=candidate.candidate_id,
            constraint_results=constraint_results,
            violations=violations,
            evidence_gaps=gaps,
            critical_evidence_gaps=critical_gaps,
            contradictions=contradictions,
            hard_constraints_passed=hard_constraints_passed,
            score=score,
            confidence=confidence,
            verdict=verdict,
            rationale=rationale,
        )

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _aliases_for(self, property_name: str) -> set[str]:
        known = self.aliases.get(property_name, set())
        return {property_name, *known}

    def _evaluate_constraint(
        self,
        constraint: ConstraintSpec,
        candidate: CandidateState,
        scientific: ScientificState,
    ) -> ConstraintOutcome:
        if not _constraint_target_is_finite(constraint):
            return ConstraintOutcome(
                property=constraint.property,
                operator=constraint.operator,
                target_value=constraint.value,
                unit=constraint.unit,
                severity=constraint.severity,
                result="UNKNOWN",
                reason="constraint target must contain only finite numeric limits",
            )
        requirements, requirements_error = self._requirements_for(constraint.property)
        if requirements_error:
            return ConstraintOutcome(
                property=constraint.property,
                operator=constraint.operator,
                target_value=constraint.value,
                unit=constraint.unit,
                severity=constraint.severity,
                result="UNKNOWN",
                reason=requirements_error,
            )
        resolved, exclusions = self._resolve_evidence(
            constraint, candidate, scientific, requirements
        )
        if resolved is None:
            reason = "no applicable evidence for property"
            if exclusions:
                reason += ": " + "; ".join(dict.fromkeys(exclusions))
            return ConstraintOutcome(
                property=constraint.property,
                operator=constraint.operator,
                target_value=constraint.value,
                unit=constraint.unit,
                severity=constraint.severity,
                result="UNKNOWN",
                reason=reason,
            )
        check: ConstraintCheck = evaluate_constraint(constraint, resolved.value)
        if check.passed is None:
            return ConstraintOutcome(
                property=constraint.property,
                operator=constraint.operator,
                observed_value=resolved.value,
                target_value=constraint.value,
                unit=resolved.unit or constraint.unit,
                severity=constraint.severity,
                result="UNKNOWN",
                evidence_found=True,
                evidence_ids=[resolved.evidence_id],
                fidelity=resolved.fidelity,
                confidence=resolved.confidence,
                reason=f"unusable evidence: {check.detail}",
            )
        result: PropertyResult = "PASS" if check.passed else "FAIL"
        return ConstraintOutcome(
            property=constraint.property,
            operator=constraint.operator,
            observed_value=resolved.value,
            target_value=constraint.value,
            unit=resolved.unit or constraint.unit,
            severity=constraint.severity,
            result=result,
            evidence_found=True,
            evidence_ids=[resolved.evidence_id],
            fidelity=resolved.fidelity,
            confidence=resolved.confidence,
            soft_score=check.soft_score,
            reason=check.detail,
        )

    def _resolve_evidence(
        self,
        constraint: ConstraintSpec,
        candidate: CandidateState,
        scientific: ScientificState,
        requirements: EvidenceRequirements,
    ) -> tuple[_ResolvedEvidence | None, list[str]]:
        candidates: list[_ResolvedEvidence] = []
        exclusions: list[str] = []
        for evidence in scientific.evidence:
            resolved, exclusion = self._evidence_for_property(
                evidence, constraint, candidate, requirements
            )
            if resolved is not None:
                candidates.append(resolved)
            elif exclusion:
                exclusions.append(exclusion)
        declarations = candidate.representation.get("properties")
        if isinstance(declarations, dict) and constraint.property in declarations:
            exclusions.append("candidate-declared properties are proposals, not observations")
        if not candidates:
            return None, exclusions
        candidates.sort(key=lambda item: (item.rank, _source_priority(item.source)), reverse=True)
        return candidates[0], exclusions

    def _evidence_for_property(
        self,
        evidence: Evidence | ScientificEvidence,
        constraint: ConstraintSpec,
        candidate: CandidateState,
        requirements: EvidenceRequirements,
    ) -> tuple[_ResolvedEvidence | None, str]:
        aliases = self._aliases_for(constraint.property)
        value: Any = None
        unit = ""
        fidelity: str | None = None
        subject: str | None = None
        scoped_evidence: ScientificEvidence

        if isinstance(evidence, ScientificEvidence):
            if str(evidence.property) not in aliases or evidence.value is None:
                return None, ""
            value = evidence.value
            unit = evidence.unit
            fidelity = evidence.fidelity
            subject = evidence.subject
            scoped_evidence = evidence
        else:
            payload = _parse_json_payload(evidence.content)
            if payload is None or not isinstance(payload, dict):
                return None, ""
            matched_key = next(
                (key for key in payload if str(key) in aliases),
                None,
            )
            if matched_key is None or payload[matched_key] is None:
                return None, ""
            value = payload[matched_key]
            unit = _infer_unit(str(matched_key))
            fidelity = "empirical"
            subject = str(payload.get("material") or payload.get("formula") or "")
            source_type = _legacy_source_type(evidence.type)
            scoped_evidence = ScientificEvidence(
                id=evidence.id,
                subject=subject,
                property=str(matched_key),
                value=value,
                unit=unit,
                source=evidence.source,
                source_type=source_type,
                method=str(evidence.provenance.get("method", evidence.type)),
                fidelity="empirical",
                provenance=evidence.provenance,
            )

        applicable, reason = evidence_applicable(
            scoped_evidence,
            candidate,
            requirements,
            allow_synthetic_evidence=self.policy.allow_synthetic_evidence,
        )
        if not applicable:
            return None, reason
        try:
            normalized_value, normalized_unit = _normalize_property_value(
                value, unit, constraint.unit
            )
        except ValueError as exc:
            return None, f"unusable evidence unit or value: {exc}"
        return (
            _ResolvedEvidence(
                value=normalized_value,
                unit=normalized_unit,
                fidelity=fidelity,
                confidence=evidence_confidence(evidence, fidelity),
                evidence_id=evidence.id,
                rank=(
                    fidelity_rank(fidelity)
                    if fidelity is not None
                    else fidelity_rank(evidence.fidelity)
                ),
                source=_evidence_scope(evidence),
                subject=subject,
                method=scoped_evidence.method,
                conditions=dict(scoped_evidence.conditions),
                structure_hash=scoped_evidence.structure_hash,
            ),
            "",
        )

    def _requirements_for(
        self, property_name: str
    ) -> tuple[EvidenceRequirements, str]:
        raw_all = self.target.metadata.get("evidence_requirements", {})
        if raw_all is None:
            raw_all = {}
        if not isinstance(raw_all, dict):
            return EvidenceRequirements(), "invalid target evidence_requirements metadata"
        raw = raw_all.get(property_name, {})
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            return EvidenceRequirements(), (
                f"invalid evidence requirements for property {property_name!r}"
            )
        try:
            return EvidenceRequirements.model_validate(raw), ""
        except ValidationError as exc:
            return EvidenceRequirements(), (
                f"invalid evidence requirements for property {property_name!r}: {exc.errors()[0]['msg']}"
            )

    def _detect_contradictions(
        self, candidate: CandidateState, scientific: ScientificState
    ) -> list[str]:
        """Same-property evidence that disagrees beyond a small tolerance."""
        by_property: dict[str, list[_ResolvedEvidence]] = {}
        for constraint in self.target.constraints:
            requirements, error = self._requirements_for(constraint.property)
            if error:
                continue
            for evidence in scientific.evidence:
                resolved, _ = self._evidence_for_property(
                    evidence, constraint, candidate, requirements
                )
                if resolved is not None:
                    by_property.setdefault(constraint.property, []).append(resolved)
        contradictions: list[str] = []
        for property_name, items in by_property.items():
            by_scope: dict[tuple[str, str, str], list[_ResolvedEvidence]] = {}
            for item in items:
                if not item.method.strip():
                    continue
                scope_key = (
                    item.method.strip().casefold(),
                    item.structure_hash,
                    json.dumps(
                        _normalized_condition_value(item.conditions),
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    ),
                )
                by_scope.setdefault(scope_key, []).append(item)
            for comparable in by_scope.values():
                seen: list[tuple[float, str]] = []
                for item in comparable:
                    if isinstance(item.value, bool) or not isinstance(item.value, (int, float)):
                        continue
                    value = float(item.value)
                    if not math.isfinite(value):
                        continue
                    for prior_value, prior_source in seen:
                        scale = max(abs(value), abs(prior_value), 1e-12)
                        if abs(value - prior_value) / scale > 0.05:
                            contradictions.append(
                                f"{property_name}: {prior_source}={prior_value} vs "
                                f"{item.source}={value}"
                            )
                    seen.append((value, item.source))
        return contradictions


def _verdict(outcomes: list[ConstraintOutcome]) -> Verdict:
    hard_fail = any(o.result == "FAIL" and o.severity == "HARD" for o in outcomes)
    if hard_fail:
        return "FAIL"
    soft_fail = any(o.result == "FAIL" and o.severity == "SOFT" for o in outcomes)
    if soft_fail:
        return "REVISE"
    hard_unknown = any(o.result == "UNKNOWN" and o.severity == "HARD" for o in outcomes)
    if hard_unknown:
        return "INCONCLUSIVE"
    return "PASS"


def _constraint_target_is_finite(constraint: ConstraintSpec) -> bool:
    values = constraint.value if constraint.operator == "between" else [constraint.value]
    if isinstance(values, (str, bytes, bool)):
        return True
    try:
        items = list(values)
    except TypeError:
        items = [values]
    return all(
        not isinstance(item, (int, float)) or isinstance(item, bool) or math.isfinite(float(item))
        for item in items
    )


def _normalized_condition_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalized_condition_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normalized_condition_value(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return value


def _rationale(
    verdict: Verdict,
    outcomes: list[ConstraintOutcome],
    gaps: list[str],
    contradictions: list[str],
) -> str:
    passed = [o.property for o in outcomes if o.result == "PASS"]
    failed = [o.property for o in outcomes if o.result == "FAIL"]
    parts = [f"verdict={verdict}"]
    if passed:
        parts.append("passed: " + ", ".join(passed))
    if failed:
        parts.append("failed: " + ", ".join(failed))
    if gaps:
        parts.append("missing evidence: " + ", ".join(gaps))
    if contradictions:
        parts.append("contradictions: " + "; ".join(contradictions))
    return "; ".join(parts)


def _parse_json_payload(content: str) -> Any:
    """Deterministic JSON extraction shared with candidate extraction."""
    return extract_json_payload(content)


def _infer_unit(key: str) -> str:
    lowered = key.lower()
    for suffix, unit in _UNIT_SUFFIX.items():
        if lowered.endswith(suffix):
            return unit
    return ""


def _legacy_source_type(
    evidence_type: str,
) -> Literal["database", "literature", "experimental", "calculation"]:
    normalized = evidence_type.strip().casefold()
    if normalized == "database":
        return "database"
    if normalized == "literature":
        return "literature"
    if normalized in {"experiment", "experimental"}:
        return "experimental"
    return "calculation"


def _normalize_property_value(
    value: Any, evidence_unit: str, target_unit: str
) -> tuple[float, str]:
    """Return a finite value expressed in the constraint's supported unit."""

    if evidence_unit.strip() and evidence_unit.strip() == target_unit.strip():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("property value must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("property value must be finite")
        return numeric, target_unit.strip()
    normalized = convert_property_value(value, evidence_unit, target_unit)
    return normalized, target_unit.strip()


def _evidence_scope(evidence: Evidence | ScientificEvidence) -> str:
    if isinstance(evidence, Evidence):
        return f"evidence:{evidence.source}"
    return f"evidence:{evidence.source}:{evidence.fidelity}"


def _source_priority(source: str) -> int:
    """Tie-break: real scientific evidence beats candidate-declared predictions."""
    if source == "candidate_declared":
        return 0
    if source.startswith("evidence:"):
        return 2
    return 1
