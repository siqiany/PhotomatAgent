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
from photomatagent.scientific.evidence import Evidence
from photomatagent.scientific.state import ScientificState

CandidateStatus = Literal[
    "PROPOSED", "EVALUATING", "PASS", "FAIL", "REVISE", "REJECTED"
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
        signature["formula"] = _canonical_formula(str(formula))
    composition = representation.get("composition")
    if isinstance(composition, dict):
        signature["composition"] = sorted(
            (str(element), _rounded(value))
            for element, value in composition.items()
        )
    elif formula:
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
        provenance = getattr(evidence, "provenance", {}) or {}
        method = generation_method or str(provenance.get("tool", ""))
        lineage = CandidateLineage(
            generated_by=method or "unknown",
            generation_parameters=provenance,
            source_artifacts=[evidence.id],
            validation_status="UNVALIDATED_GENERATED_STRUCTURE",
        )
        return candidate_from_formula(
            formula,
            generation_method=method,
            generation_parameters=provenance,
            extra_representation={
                "evidence_ids": [evidence.id],
                "evidence_type": str(getattr(evidence, "source_type", "")),
            },
            created_iteration=iteration,
            lineage=lineage,
        )
    return None