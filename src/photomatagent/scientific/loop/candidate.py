"""CandidateState: candidates become first-class citizens of the loop.

Candidates are built from structured scientific output (generation-tool
evidence, CalculationRecord payloads, ScientificEvidence) -- never from
free-text conversation parsing. Provenance reuses the existing
:class:`CandidateLineage` chain instead of inventing a parallel system.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.capabilities.generation.lineage import (
    CandidateLineage,
)
from photomatagent.scientific.discovery.composition import (
    CompositionCapabilityError,
    composition_key,
    normalize_composition,
)
from photomatagent.scientific.discovery.structures import StructureDerivation
from photomatagent.scientific.evidence import Evidence
from photomatagent.scientific.state import ScientificState

CandidateStatus = Literal[
    "PROPOSED",
    "EVALUATING",
    "PASS",
    "FAIL",
    "REVISE",
    "INCONCLUSIVE",
    "REJECTED",
]

# Evidence properties that carry a structured candidate proposal.
_CANDIDATE_PROPERTIES = ("proposed_formula", "candidate_formula", "formula")


class CandidateState(BaseModel):
    """One proposed material candidate tracked by the scientific outer loop."""

    candidate_id: str = Field(default_factory=lambda: f"cand_{uuid4().hex[:10]}")
    parent_candidate_id: str | None = None
    label: str = ""
    candidate_type: str = ""
    representation: dict[str, Any] = Field(default_factory=dict)
    generation_method: str = ""
    generation_parameters: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    score: float | None = None
    status: CandidateStatus = "PROPOSED"
    rejection_reasons: list[str] = Field(default_factory=list)
    created_iteration: int = 0
    lineage: CandidateLineage | None = None

    @property
    def formula(self) -> str:
        value = self.representation.get("formula", "")
        return str(value) if value else ""

    @property
    def fingerprint(self) -> str:
        return candidate_fingerprint(self)


class CandidateProjectionDiagnostic(BaseModel):
    """One recoverable legacy-record problem encountered during projection."""

    code: Literal["INVALID_LEGACY_CANDIDATE", "COMPOSITION_CAPABILITY_UNAVAILABLE"]
    evidence_id: str
    message: str


class CandidateProjectionResult(BaseModel):
    candidates: list[CandidateState] = Field(default_factory=list)
    diagnostics: list[CandidateProjectionDiagnostic] = Field(default_factory=list)


def _canonical_formula(formula: str) -> str:
    """Normalize a formula to a canonical element-sorted string.

    ``HgTe``, ``hg Te``, ``TeHg`` all normalize to ``Hg1Te1``.
    """
    text = "".join(formula.split())
    tokens = re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", text)
    if not tokens or "".join(f"{element}{count or ''}" for element, count in tokens) != text:
        return "".join(sorted(text.casefold()))
    counts = {element: float(count or 1.0) for element, count in tokens}
    return "".join(f"{element}{counts[element]:g}" for element in sorted(counts))


def candidate_fingerprint(candidate: CandidateState) -> str:
    """Deterministic, stable identity for repetition detection.

    Built only from the normalized representation (formula / composition /
    structure identifier), so generating the same formula again produces the
    same fingerprint and never counts as a new iteration.
    """
    representation = candidate.representation or {}
    signature: dict[str, Any] = {}
    formula = representation.get("formula")
    if formula:
        try:
            signature["composition"] = normalize_composition(str(formula))
        except (ValueError, RuntimeError):
            signature["formula"] = _canonical_formula(str(formula))
    composition = representation.get("composition")
    if "composition" not in signature and isinstance(composition, dict):
        signature["composition"] = sorted(
            (str(element), _rounded(value))
            for element, value in composition.items()
        )
    elif "composition" not in signature and formula:
        signature["composition"] = _composition_from_formula(str(formula))
    for key in ("structure_identifier", "structure_id", "cif_hash"):
        if representation.get(key):
            signature[key] = str(representation[key])
    if not signature:
        signature["representation"] = _sorted_json(representation)
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _composition_from_formula(formula: str) -> list[tuple[str, float]]:
    tokens = re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", formula)
    counts: dict[str, float] = {}
    for element, count in tokens:
        counts[element] = counts.get(element, 0.0) + float(count or 1.0)
    return sorted((element, counts[element]) for element in counts)


def _rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    return value


def _sorted_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sorted_json(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_sorted_json(item) for item in value]
    return value


def _trusted_file_hash(value: Any) -> str:
    text = str(value or "")
    return text if re.fullmatch(r"[0-9a-f]{64}", text) else ""


def candidate_from_formula(
    formula: str,
    *,
    parent_candidate_id: str | None = None,
    candidate_type: str = "composition",
    generation_method: str = "",
    generation_parameters: dict[str, Any] | None = None,
    extra_representation: dict[str, Any] | None = None,
    created_iteration: int = 0,
    lineage: CandidateLineage | None = None,
) -> CandidateState:
    """Build a candidate from a structured formula (composition proposal)."""
    representation: dict[str, Any] = {"formula": formula}
    if extra_representation:
        representation.update(extra_representation)
    return CandidateState(
        parent_candidate_id=parent_candidate_id,
        label=formula,
        candidate_type=candidate_type,
        representation=representation,
        generation_method=generation_method,
        generation_parameters=generation_parameters or {},
        created_iteration=created_iteration,
        lineage=lineage,
    )


def extract_json_payload(content: str) -> Any:
    """Deterministically extract a JSON object from tool-output text.

    Accepts a pure JSON document, or a JSON object embedded in otherwise
    prose-wrapped output (e.g. mock tool results). Free prose is never
    guessed at -- only a parseable JSON object is returned.
    """
    stripped = content.strip()
    try:
        return json.loads(stripped)
    except (TypeError, json.JSONDecodeError):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None


def _as_formula(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list) and value and isinstance(value[0], str):
        return value[0].strip()
    return None


def _evidence_formula(evidence: Evidence | ScientificEvidence) -> str | None:
    property_name = getattr(evidence, "property", "")
    if property_name in _CANDIDATE_PROPERTIES:
        formula = _as_formula(getattr(evidence, "value", None))
        if formula:
            return formula
    structure_hash = str(getattr(evidence, "structure_hash", "") or "")
    if structure_hash:
        subject = _as_formula(getattr(evidence, "subject", ""))
        if subject:
            try:
                normalize_composition(subject)
            except (ValueError, RuntimeError):
                pass
            else:
                return subject
    if isinstance(evidence, Evidence):
        # Structured JSON payloads (e.g. mock.run_calculation results) carry
        # the material/formula the maker actually worked on.
        payload = extract_json_payload(evidence.content)
        if not isinstance(payload, dict):
            return None
        for key in ("formula", "material"):
            formula = _as_formula(payload.get(key))
            if formula:
                return formula
    return None


def extract_candidate_from_state(
    scientific: ScientificState,
    iteration: int = 0,
    generation_method: str = "",
) -> CandidateState | None:
    """Build the round's primary candidate from structured scientific state.

    Resolution order:
      1. ScientificEvidence / Evidence carrying a proposed candidate formula
         (generation tools, retrieval tools, structured payloads);
      2. JSON-payload Evidence that names the material under investigation.

    Returns ``None`` when no structured candidate exists yet -- the evaluator
    then reports INCONCLUSIVE (unknown != pass).
    """
    for evidence in reversed(scientific.evidence):
        formula = _evidence_formula(evidence)
        if not formula:
            continue
        return _candidate_from_evidence(
            evidence,
            formula,
            iteration=iteration,
            generation_method=generation_method,
        )
    return None


def _candidate_from_evidence(
    evidence: Evidence | ScientificEvidence,
    formula: str,
    *,
    iteration: int,
    generation_method: str = "",
) -> CandidateState:
    provenance = getattr(evidence, "provenance", {}) or {}
    method = generation_method or str(provenance.get("tool", ""))
    structure_hash = str(getattr(evidence, "structure_hash", "") or "")
    parent_structure_hash = str(
        provenance.get("parent_structure_hash", "") or ""
    )
    parent_candidate_id = str(
        provenance.get("parent_candidate_id", "") or ""
    )
    if not re.fullmatch(r"[0-9a-f]{64}", parent_structure_hash):
        parent_structure_hash = ""
    expected_parent_id = (
        f"cand_{parent_structure_hash[:24]}" if parent_structure_hash else ""
    )
    current_candidate_id = f"cand_{structure_hash[:24]}" if structure_hash else ""
    if (
        parent_candidate_id != expected_parent_id
        or parent_structure_hash == structure_hash
        or parent_candidate_id == current_candidate_id
    ):
        parent_candidate_id = ""
    output_sha = _trusted_file_hash(
        provenance.get("output_sha256") or provenance.get("input_sha256")
    )
    structure_representation: dict[str, Any] = {}
    if structure_hash:
        structure_representation = {
            "structure_identifier": structure_hash,
            "structure_hash": structure_hash,
            "path": str(
                provenance.get("path")
                or provenance.get("input_structure_path")
                or provenance.get("input_path")
                or ""
            ),
            "hypothesis_ids": list(provenance.get("hypothesis_ids", []))
            if isinstance(provenance.get("hypothesis_ids", []), list)
            else [],
        }
    lineage = CandidateLineage(
        generated_by=method or "unknown",
        generation_parameters=provenance,
        source_artifacts=[evidence.id],
        validation_status="UNVALIDATED_GENERATED_STRUCTURE",
    )
    candidate = candidate_from_formula(
        formula,
        parent_candidate_id=parent_candidate_id or None,
        generation_method=method,
        generation_parameters=provenance,
        extra_representation={
            "evidence_ids": [evidence.id],
            "evidence_type": str(getattr(evidence, "source_type", "")),
            "cif_hash": output_sha,
            **structure_representation,
        },
        created_iteration=iteration,
        lineage=lineage.model_copy(
            update={"parent_candidate_id": parent_candidate_id or None}
        ),
    )
    candidate.candidate_id = (
        f"cand_{structure_hash[:24]}"
        if structure_hash
        else f"cand_{composition_key(formula)[:24]}"
    )
    if structure_hash:
        candidate.candidate_type = "structure"
    candidate.evidence_ids = [evidence.id]
    return candidate


def extract_candidates_from_state(
    scientific: ScientificState,
    iteration: int = 0,
) -> list[CandidateState]:
    """Compatibility list projection for callers that do not consume diagnostics."""

    return project_candidates_from_state(scientific, iteration).candidates


def _legacy_source(evidence: Evidence | ScientificEvidence) -> dict[str, Any]:
    return {
        "evidence_id": evidence.id,
        "source": str(getattr(evidence, "source", "")),
        "source_type": str(getattr(evidence, "source_type", "")),
        "method": str(getattr(evidence, "method", "")),
        "provenance": dict(getattr(evidence, "provenance", {}) or {}),
    }


def _attach_legacy_provenance(
    candidate: CandidateState,
    *,
    sources: list[dict[str, Any]],
    evidence_ids: list[str],
) -> CandidateState:
    candidate.evidence_ids = list(evidence_ids)
    candidate.generation_parameters = {
        **candidate.generation_parameters,
        "legacy_sources": sources,
    }
    if candidate.lineage is not None:
        lineage_parameters = {
            **candidate.lineage.generation_parameters,
            "legacy_sources": sources,
        }
        source_artifacts = [
            artifact
            for artifact in candidate.lineage.source_artifacts
            if artifact not in evidence_ids
        ]
        source_artifacts.extend(evidence_ids)
        candidate.lineage = candidate.lineage.model_copy(
            update={
                "generation_parameters": lineage_parameters,
                "source_artifacts": source_artifacts,
            }
        )
    return candidate


def _formula_from_normalized_composition(
    composition: tuple[tuple[str, int], ...],
) -> str:
    return "".join(
        symbol if amount == 1 else f"{symbol}{amount}"
        for symbol, amount in composition
    )


def _structure_candidate(
    derivation: StructureDerivation,
    *,
    iteration: int,
) -> CandidateState:
    """Project one trusted derivation without using its filename as identity."""

    formula = _formula_from_normalized_composition(
        tuple(derivation.normalized_composition)
    )
    candidate = candidate_from_formula(
        formula,
        parent_candidate_id=derivation.parent_candidate_id,
        candidate_type="structure",
        generation_method=derivation.operation,
        generation_parameters={
            "operation": derivation.operation,
            "parameters": dict(derivation.parameters),
            "derivation_ids": [derivation.id],
            "sources": [
                {
                    "derivation_id": derivation.id,
                    "path": derivation.output_path,
                    "origin": dict(derivation.origin),
                }
            ],
        },
        extra_representation={
            "structure_identifier": derivation.structure_hash,
            "structure_hash": derivation.structure_hash,
            "cif_hash": _trusted_file_hash(
                derivation.origin.get("output_sha256", "")
            ),
            "path": derivation.output_path,
            "hypothesis_ids": (
                [derivation.hypothesis_id] if derivation.hypothesis_id else []
            ),
        },
        created_iteration=iteration,
        lineage=derivation.lineage,
    )
    candidate.candidate_id = derivation.candidate_id
    return candidate


def _merge_structure_derivation(
    candidate: CandidateState,
    derivation: StructureDerivation,
) -> None:
    """Retain all source records while keeping one hash-derived candidate."""

    representation = candidate.representation
    hypothesis_ids = representation.setdefault("hypothesis_ids", [])
    if derivation.hypothesis_id and derivation.hypothesis_id not in hypothesis_ids:
        hypothesis_ids.append(derivation.hypothesis_id)
    derivation_ids = candidate.generation_parameters.setdefault("derivation_ids", [])
    if derivation.id not in derivation_ids:
        derivation_ids.append(derivation.id)
    sources = candidate.generation_parameters.setdefault("sources", [])
    source = {
        "derivation_id": derivation.id,
        "path": derivation.output_path,
        "origin": dict(derivation.origin),
    }
    if source not in sources:
        sources.append(source)


def project_candidates_from_state(
    scientific: ScientificState,
    iteration: int = 0,
) -> CandidateProjectionResult:
    """Project all registered hypotheses and legacy evidence into candidates.

    Composition hypotheses retain composition identity; structure derivations
    use their trusted geometry hash as identity. Multiple sources for one
    structure merge while distinct geometries remain separate. Legacy
    structured evidence is retained and may coexist with mechanism candidates.
    """

    candidates: list[CandidateState] = []
    diagnostics: list[CandidateProjectionDiagnostic] = []
    by_id: dict[str, CandidateState] = {}
    index_by_id: dict[str, int] = {}

    for hypothesis in scientific.material_hypotheses:
        candidate = by_id.get(hypothesis.candidate_id)
        if candidate is None:
            composition = dict(hypothesis.normalized_composition)
            candidate = candidate_from_formula(
                hypothesis.proposal.formula,
                generation_method="mechanism_reasoning",
                extra_representation={
                    "composition": composition,
                    "hypothesis_ids": [hypothesis.id],
                },
                created_iteration=iteration,
                lineage=hypothesis.lineage,
            )
            candidate.candidate_id = hypothesis.candidate_id
            index_by_id[candidate.candidate_id] = len(candidates)
            candidates.append(candidate)
            by_id[candidate.candidate_id] = candidate
        else:
            hypothesis_ids = candidate.representation["hypothesis_ids"]
            if hypothesis.id not in hypothesis_ids:
                hypothesis_ids.append(hypothesis.id)

    for derivation in scientific.structure_derivations:
        identity = derivation.candidate_id
        existing = by_id.get(identity)
        if existing is None:
            candidate = _structure_candidate(derivation, iteration=iteration)
            index_by_id[identity] = len(candidates)
            candidates.append(candidate)
            by_id[identity] = candidate
        elif existing.candidate_type == "structure":
            _merge_structure_derivation(existing, derivation)

    for evidence in scientific.evidence:
        formula = _evidence_formula(evidence)
        if not formula:
            continue
        try:
            identity = f"cand_{composition_key(formula)[:24]}"
        except CompositionCapabilityError as exc:
            diagnostics.append(
                CandidateProjectionDiagnostic(
                    code="COMPOSITION_CAPABILITY_UNAVAILABLE",
                    evidence_id=evidence.id,
                    message=str(exc)[:300],
                )
            )
            continue
        except ValueError as exc:
            diagnostics.append(
                CandidateProjectionDiagnostic(
                    code="INVALID_LEGACY_CANDIDATE",
                    evidence_id=evidence.id,
                    message=str(exc)[:300],
                )
            )
            continue
        evidence_structure_hash = str(
            getattr(evidence, "structure_hash", "") or ""
        )
        if evidence_structure_hash:
            identity = f"cand_{evidence_structure_hash[:24]}"
            evidence_identity = str(getattr(evidence, "candidate_id", "") or "")
            expected_identity = identity
            if evidence_identity and evidence_identity != expected_identity:
                diagnostics.append(
                    CandidateProjectionDiagnostic(
                        code="INVALID_LEGACY_CANDIDATE",
                        evidence_id=evidence.id,
                        message=(
                            "ignored mismatched evidence candidate_id; derived "
                            f"trusted identity {expected_identity} from structure_hash"
                        ),
                    )
                )
        existing = by_id.get(identity)
        source = _legacy_source(evidence)
        if existing is not None and existing.candidate_type == "structure":
            existing.evidence_ids.append(evidence.id)
            _attach_legacy_provenance(
                existing,
                sources=[
                    *list(existing.generation_parameters.get("legacy_sources", [])),
                    source,
                ],
                evidence_ids=list(existing.evidence_ids),
            )
            continue
        if existing is not None and existing.generation_method == "mechanism_reasoning":
            prior_sources = list(
                existing.generation_parameters.get("legacy_sources", [])
            )
            _attach_legacy_provenance(
                existing,
                sources=[*prior_sources, source],
                evidence_ids=[*existing.evidence_ids, evidence.id],
            )
            continue
        candidate = _candidate_from_evidence(
            evidence,
            formula,
            iteration=iteration,
        )
        if existing is None:
            _attach_legacy_provenance(
                candidate,
                sources=[source],
                evidence_ids=[evidence.id],
            )
            index_by_id[identity] = len(candidates)
            candidates.append(candidate)
        else:
            prior_sources = list(
                existing.generation_parameters.get("legacy_sources", [])
            )
            _attach_legacy_provenance(
                candidate,
                sources=[*prior_sources, source],
                evidence_ids=[*existing.evidence_ids, evidence.id],
            )
            candidates[index_by_id[identity]] = candidate
        by_id[identity] = candidate

    return CandidateProjectionResult(candidates=candidates, diagnostics=diagnostics)
