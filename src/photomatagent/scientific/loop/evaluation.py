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

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, PrivateAttr, ValidationError

from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.evidence import Evidence
from photomatagent.scientific.evidence_refs import opaque_evidence_ref
from photomatagent.scientific.loop.candidate import (
    CandidateState,
    extract_json_payload,
)
from photomatagent.scientific.loop.evidence_scope import (
    EvidenceAuthority,
    EvidenceRequirements,
    convert_property_value,
    evidence_applicable,
)
from photomatagent.scientific.loop.observation import stable_observation_identity
from photomatagent.scientific.loop.scoring import compute_score
from photomatagent.scientific.loop.target import (
    ConstraintCheck,
    ConstraintOutcome,
    ConstraintSpec,
    ConstraintViolation,
    TargetSpec,
    evaluate_constraint,
    normalize_operating_conditions,
)
from photomatagent.scientific.state import (
    DEFAULT_TRUSTED_EVIDENCE_TOOLS,
    ScientificState,
)
from photomatagent.scientific.discovery.composition import normalize_composition

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
    "formation_energy": {
        "formation_energy",
        "formation_energy_eV_per_atom",
        "formation_energy_meV_per_atom",
    },
    "energy_above_hull": {
        "energy_above_hull",
        "energy_above_hull_eV_per_atom",
        "energy_above_hull_meV_per_atom",
    },
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
    "_mev_per_atom": "meV/atom",
    "_ev_per_atom": "eV/atom",
    "_mev": "meV",
    "_ev": "eV",
    "_um": "um",
    "_a_w": "A/W",
    "_g_cm3": "g/cm3",
    "_k": "K",
    "_jones": "cm Hz^1/2/W",
}

_DEVICE_ONLY_PROPERTIES = frozenset(
    {"responsivity", "detectivity", "dark_current", "quantum_efficiency"}
)
_DEVICE_REQUIRED_CONDITIONS = (
    "wavelength_um",
    "bias_v",
    "temperature_k",
    "measurement_definition",
)
_MAX_EXCLUSION_REASONS = 8
_MAX_REASON_CHARS = 512
_MAX_CONTRADICTION_REASONS = 8


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
    composition_identity: tuple[tuple[str, int], ...] | None = None


@dataclass(frozen=True)
class EvidenceEvaluationPolicy:
    """Host-owned evaluator switches used by isolated deterministic tests."""

    allow_synthetic_evidence: bool = False
    trusted_attestation_tools: frozenset[str] = DEFAULT_TRUSTED_EVIDENCE_TOOLS


@dataclass(frozen=True)
class AcceptedEvidenceRecord:
    """Evaluator-owned evidence adoption record.

    Raw evidence identifiers are intentionally kept in this runtime-only
    record.  They are needed to project a report back onto the authoritative
    ``ScientificState`` but must never become part of a serialized report.
    """

    evidence_id: str
    property: str
    outcome: PropertyResult
    observation_identity: str
    attestation_authority: str
    attestation_origin: str
    attestation_tool_name: str
    attestation_tool_call_id: str


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
    _accepted_evidence_manifest: tuple[AcceptedEvidenceRecord, ...] = PrivateAttr(
        default=()
    )

    @property
    def accepted_evidence_manifest(self) -> tuple[AcceptedEvidenceRecord, ...]:
        """Return the non-serializable manifest produced by the evaluator."""

        return self._accepted_evidence_manifest

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> EvaluationReport:
        copied = super().model_copy(update=update, deep=deep)
        copied._accepted_evidence_manifest = ()
        return copied

    def __copy__(self) -> EvaluationReport:
        copied = super().__copy__()
        copied._accepted_evidence_manifest = ()
        return copied

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> EvaluationReport:
        copied = super().__deepcopy__(memo)
        copied._accepted_evidence_manifest = ()
        if memo is not None:
            memo[id(self)] = copied
        return copied

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
        accepted_manifest: list[AcceptedEvidenceRecord] = []
        contradictions: list[str] = []
        used_confidences: list[float] = []
        for constraint in self.target.constraints:
            outcome = self._evaluate_constraint(
                constraint,
                candidate,
                scientific,
                accepted_manifest=accepted_manifest,
            )
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
        report = EvaluationReport(
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
        report._accepted_evidence_manifest = tuple(accepted_manifest)
        return report

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
        *,
        accepted_manifest: list[AcceptedEvidenceRecord] | None = None,
    ) -> ConstraintOutcome:
        if not _constraint_target_is_finite(constraint):
            return ConstraintOutcome(
                property=constraint.property,
                operator=constraint.operator,
                target_value=constraint.value,
                unit=constraint.unit,
                severity=constraint.severity,
                result="UNKNOWN",
                reason="CONSTRAINT_TARGET_INVALID",
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
                reason="EVIDENCE_REQUIREMENTS_INVALID",
            )
        resolved, exclusions = self._resolve_evidence(
            constraint, candidate, scientific, requirements
        )
        if resolved is None:
            reason = "NO_APPLICABLE_EVIDENCE"
            if exclusions:
                unique = list(dict.fromkeys(exclusions))[:_MAX_EXCLUSION_REASONS]
                reason += ":" + ",".join(unique)
            reason = reason[:_MAX_REASON_CHARS]
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
            self._append_manifest_record(
                accepted_manifest,
                scientific,
                resolved,
                constraint.property,
                "UNKNOWN",
            )
            return ConstraintOutcome(
                property=constraint.property,
                operator=constraint.operator,
                observed_value=resolved.value,
                target_value=constraint.value,
                unit=resolved.unit or constraint.unit,
                severity=constraint.severity,
                result="UNKNOWN",
                evidence_found=True,
                evidence_ids=[opaque_evidence_ref(resolved.evidence_id)],
                fidelity=resolved.fidelity,
                confidence=resolved.confidence,
                reason="CONSTRAINT_UNUSABLE",
            )
        result: PropertyResult = "PASS" if check.passed else "FAIL"
        self._append_manifest_record(
            accepted_manifest,
            scientific,
            resolved,
            constraint.property,
            result,
        )
        return ConstraintOutcome(
            property=constraint.property,
            operator=constraint.operator,
            observed_value=resolved.value,
            target_value=constraint.value,
            unit=resolved.unit or constraint.unit,
            severity=constraint.severity,
            result=result,
            evidence_found=True,
            evidence_ids=[opaque_evidence_ref(resolved.evidence_id)],
            fidelity=resolved.fidelity,
            confidence=resolved.confidence,
            soft_score=check.soft_score,
            reason="CONSTRAINT_PASS" if check.passed else "CONSTRAINT_FAIL",
        )

    def _append_manifest_record(
        self,
        accepted_manifest: list[AcceptedEvidenceRecord] | None,
        scientific: ScientificState,
        resolved: _ResolvedEvidence,
        property_name: str,
        outcome: PropertyResult,
    ) -> None:
        if accepted_manifest is None:
            return
        attestation = scientific.verified_attestation(resolved.evidence_id)
        matching = [
            evidence
            for evidence in scientific.evidence
            if evidence.id == resolved.evidence_id
        ]
        # Synthetic source fallback may satisfy the evaluator's legacy policy
        # without a runtime attestation.  It is deliberately not progress.
        if attestation is None or len(matching) != 1:
            return
        evidence = matching[0]
        accepted_manifest.append(
            AcceptedEvidenceRecord(
                evidence_id=resolved.evidence_id,
                property=property_name,
                outcome=outcome,
                observation_identity=stable_observation_identity(evidence),
                attestation_authority=attestation.authority,
                attestation_origin=attestation.origin,
                attestation_tool_name=attestation.tool_name,
                attestation_tool_call_id=attestation.tool_call_id,
            )
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
            authority = self._evidence_authority(evidence.id, scientific)
            resolved, exclusion = self._evidence_for_property(
                evidence, constraint, candidate, requirements, authority
            )
            if resolved is not None:
                candidates.append(resolved)
            elif exclusion:
                exclusions.append(exclusion)
        declarations = candidate.representation.get("properties")
        if isinstance(declarations, dict) and constraint.property in declarations:
            exclusions.append("CANDIDATE_DECLARED_PROPERTY")
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
        authority: EvidenceAuthority,
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
            authority=authority,
            allow_synthetic_evidence=self.policy.allow_synthetic_evidence,
        )
        if not applicable:
            return None, reason
        try:
            normalized_value, normalized_unit = _normalize_property_value(
                value, unit, constraint.unit
            )
        except ValueError:
            return None, "UNIT_OR_VALUE_UNUSABLE"
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
                composition_identity=_composition_identity(subject),
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
            parsed = EvidenceRequirements.model_validate(raw)
        except ValidationError as exc:
            return EvidenceRequirements(), (
                f"invalid evidence requirements for property {property_name!r}: {exc.errors()[0]['msg']}"
            )
        if property_name in _DEVICE_ONLY_PROPERTIES:
            operating_diagnostics = self.target.metadata.get(
                "operating_condition_diagnostics", []
            )
            normalized_conditions, current_diagnostics = normalize_operating_conditions(
                self.target.operating_conditions
            )
            if operating_diagnostics or current_diagnostics:
                return EvidenceRequirements(), "invalid target operating conditions"
            conditions = dict(parsed.conditions)
            condition_ranges = dict(parsed.condition_ranges)
            target_temperature = normalized_conditions.get("temperature_k")
            if (
                "temperature_k" in normalized_conditions
                and not _finite_number(target_temperature)
            ):
                return EvidenceRequirements(), "invalid target operating temperature"
            if _finite_number(target_temperature):
                conditions["temperature_k"] = target_temperature
            spectral_range = normalized_conditions.get("spectral_range_um")
            if "spectral_range_um" in normalized_conditions and not (
                isinstance(spectral_range, (list, tuple))
                and len(spectral_range) == 2
                and all(_finite_number(item) for item in spectral_range)
                and float(spectral_range[0]) <= float(spectral_range[1])
            ):
                return EvidenceRequirements(), "invalid target spectral range"
            if (
                isinstance(spectral_range, (list, tuple))
                and len(spectral_range) == 2
                and all(_finite_number(item) for item in spectral_range)
                and float(spectral_range[0]) <= float(spectral_range[1])
            ):
                condition_ranges["wavelength_um"] = (
                    float(spectral_range[0]),
                    float(spectral_range[1]),
                )
            parsed = parsed.model_copy(
                update={
                    "scope": "device",
                    "conditions": conditions,
                    "condition_ranges": condition_ranges,
                    "required_conditions": tuple(
                        dict.fromkeys(
                            (*_DEVICE_REQUIRED_CONDITIONS, *parsed.required_conditions)
                        )
                    ),
                }
            )
        return parsed, ""

    def _evidence_authority(
        self, evidence_id: str, scientific: ScientificState
    ) -> EvidenceAuthority:
        attestation = scientific.verified_attestation(evidence_id)
        if attestation is not None:
            if (
                attestation.origin == "trusted_builtin"
                and attestation.tool_name in self.policy.trusted_attestation_tools
            ):
                return "observation"
            if (
                attestation.origin == "synthetic_test"
                and self.policy.allow_synthetic_evidence
            ):
                return "synthetic"
        if self.policy.allow_synthetic_evidence:
            evidence = next(
                (item for item in scientific.evidence if item.id == evidence_id), None
            )
            source = str(getattr(evidence, "source", "")).strip().casefold()
            if source == "synthetic" or source.startswith(("synthetic:", "test-only")):
                return "synthetic"
        return "background"

    def _detect_contradictions(
        self, candidate: CandidateState, scientific: ScientificState
    ) -> list[str]:
        """Same-property evidence that disagrees beyond a small tolerance."""
        by_property: dict[str, list[_ResolvedEvidence]] = {}
        constraints_by_property: dict[str, list[ConstraintSpec]] = {}
        for constraint in self.target.constraints:
            constraints_by_property.setdefault(constraint.property, []).append(constraint)
        for property_name, constraints in constraints_by_property.items():
            requirements, error = self._requirements_for(property_name)
            if error:
                continue
            canonical_unit = _canonical_property_unit(property_name, constraints)
            canonical_constraint = constraints[0].model_copy(
                update={"unit": canonical_unit}
            )
            unique: dict[str, _ResolvedEvidence] = {}
            for evidence in scientific.evidence:
                authority = self._evidence_authority(evidence.id, scientific)
                resolved, _ = self._evidence_for_property(
                    evidence,
                    canonical_constraint,
                    candidate,
                    requirements,
                    authority,
                )
                if resolved is not None:
                    unique.setdefault(evidence.id, resolved)
            by_property[property_name] = list(unique.values())
        contradictions: list[str] = []
        for property_name, items in by_property.items():
            by_scope: dict[tuple[str, str, str], list[_ResolvedEvidence]] = {}
            for item in items:
                if (
                    item.composition_identity is None
                    or not item.method.strip()
                    or not item.structure_hash
                    or not item.conditions
                ):
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
                seen: list[float] = []
                for item in comparable:
                    if isinstance(item.value, bool) or not isinstance(item.value, (int, float)):
                        continue
                    value = float(item.value)
                    if not math.isfinite(value):
                        continue
                    for prior_value in seen:
                        scale = max(abs(value), abs(prior_value), 1e-12)
                        if abs(value - prior_value) / scale > 0.05:
                            reason = (
                                f"{_bounded_property_ref(property_name)}:"
                                "COMPARABLE_VALUES_DISAGREE"
                            )
                            total_chars = sum(len(item) for item in contradictions)
                            if (
                                len(contradictions) >= _MAX_CONTRADICTION_REASONS
                                or total_chars + len(reason) > _MAX_REASON_CHARS
                            ):
                                return contradictions
                            contradictions.append(reason)
                            break
                    if contradictions and contradictions[-1].startswith(
                        f"{_bounded_property_ref(property_name)}:"
                    ):
                        break
                    seen.append(value)
                if contradictions and contradictions[-1].startswith(
                    f"{_bounded_property_ref(property_name)}:"
                ):
                    break
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


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
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
    parts = [
        f"verdict={verdict}",
        f"passed_count={sum(o.result == 'PASS' for o in outcomes)}",
        f"failed_count={sum(o.result == 'FAIL' for o in outcomes)}",
        f"unknown_count={len(gaps)}",
        f"contradiction_count={len(contradictions)}",
    ]
    return ";".join(parts)[:_MAX_REASON_CHARS]


def _parse_json_payload(content: str) -> Any:
    """Deterministic JSON extraction shared with candidate extraction."""
    return extract_json_payload(content)


def _infer_unit(key: str) -> str:
    lowered = key.lower()
    for suffix, unit in _UNIT_SUFFIX.items():
        if lowered.endswith(suffix.lower()):
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
    return f"evidence:{evidence.id[:64]}"


def _bounded_property_ref(property_name: str) -> str:
    if (
        len(property_name) <= 64
        and property_name
        and all(character.isalnum() or character in "_.-" for character in property_name)
    ):
        return property_name
    digest = hashlib.sha256(property_name.encode("utf-8")).hexdigest()[:16]
    return f"property_{digest}"


def _canonical_property_unit(
    property_name: str, constraints: list[ConstraintSpec]
) -> str:
    known = {
        "band_gap": "eV",
        "formation_energy": "eV/atom",
        "energy_above_hull": "eV/atom",
        "cutoff_wavelength": "um",
    }
    return known.get(property_name, constraints[0].unit)


def _composition_identity(subject: str | None) -> tuple[tuple[str, int], ...] | None:
    if not subject:
        return None
    try:
        return normalize_composition(subject)
    except (ValueError, RuntimeError):
        return None


def _source_priority(source: str) -> int:
    """Tie-break: real scientific evidence beats candidate-declared predictions."""
    if source == "candidate_declared":
        return 0
    if source.startswith("evidence:"):
        return 2
    return 1
