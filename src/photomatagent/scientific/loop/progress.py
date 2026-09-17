"""Stable identities for scientific validation progress.

The evaluator deliberately exposes opaque evidence references.  The loop must
therefore derive progress from the accepted outcome and the observation's
content identity, never from producer supplied ids or request metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.evidence import Evidence
from photomatagent.scientific.evidence_refs import matches_evidence_ref
from photomatagent.scientific.loop.candidate import CandidateState
from photomatagent.scientific.loop.evaluation import EvaluationReport
from photomatagent.scientific.state import ScientificState


@dataclass(frozen=True)
class ValidationProgress:
    """Accepted validation work discovered in one evaluation."""

    candidate_id: str
    resolved_questions: tuple[str, ...]
    observation_keys: tuple[str, ...]


_HASH_KEYS = frozenset(
    {
        "artifact_hash",
        "artifact_sha",
        "artifact_sha256",
        "content_hash",
        "content_sha",
        "content_sha256",
        "sha256",
    }
)
_VOLATILE_KEYS = frozenset(
    {
        "artifact",
        "artifact_path",
        "created_at",
        "evidence_id",
        "filename",
        "file_name",
        "input_path",
        "output_name",
        "output_path",
        "path",
        "reason",
        "request_id",
        "timestamp",
        "tool_call_id",
    }
)
_NUMERIC_QUANTUM = 1e-3


def progress_from_evaluation(
    candidate: CandidateState,
    report: EvaluationReport,
    scientific: ScientificState,
    *,
    # Retained for callers from the round-1 API.  Acceptance is exclusively
    # evaluator-owned; this compatibility value has no effect.
    allow_synthetic_evidence: bool = False,
) -> ValidationProgress:
    """Project accepted evaluator outcomes onto stable progress identities.

    An outcome is eligible only when the evaluator recorded adoption of the
    exact raw evidence item in its runtime-only manifest.  Hand-built,
    deserialized, or model-produced reports have no manifest and therefore
    cannot create progress.  Public outcome/reference fields and the current
    authoritative state are cross-checked before deriving a stable identity.
    """

    if not report.candidate_id or report.candidate_id != candidate.candidate_id:
        return ValidationProgress(
            candidate_id=candidate.candidate_id,
            resolved_questions=(),
            observation_keys=(),
        )
    resolved: set[str] = set()
    observations: set[str] = set()
    state_by_id = {str(evidence.id): evidence for evidence in scientific.evidence}
    public_results = {
        (result.property, result.result): result
        for result in report.constraint_results
    }
    for record in report.accepted_evidence_manifest:
        result = public_results.get((record.property, record.outcome))
        evidence = state_by_id.get(record.evidence_id)
        if result is None or evidence is None or not result.evidence_ids:
            continue
        if not any(
            matches_evidence_ref(reference, record.evidence_id)
            for reference in result.evidence_ids
        ):
            continue
        resolved.add(record.property)
        observations.add(_observation_key(evidence, record.property, record.outcome))
    return ValidationProgress(
        candidate_id=candidate.candidate_id,
        resolved_questions=tuple(sorted(resolved)),
        observation_keys=tuple(sorted(observations)),
    )

def _observation_key(
    evidence: Evidence | ScientificEvidence, property_name: str, outcome: str
) -> str:
    return (
        f"question:{property_name}|outcome:{outcome}|"
        f"observation:{_observation_identity(evidence)}"
    )


def _observation_identity(evidence: Evidence | ScientificEvidence) -> str:
    explicit = _hash_values(getattr(evidence, "provenance", {}))
    if isinstance(evidence, Evidence):
        try:
            content_payload = json.loads(evidence.content)
        except (TypeError, json.JSONDecodeError):
            content_payload = {}
        explicit.update(_hash_values(content_payload))
    if explicit:
        payload: Any = {
            "hashes": sorted(explicit),
            "fidelity": getattr(evidence, "fidelity", ""),
            "structure_hash": getattr(evidence, "structure_hash", ""),
            "conditions": _stable_value(getattr(evidence, "conditions", {})),
        }
    elif isinstance(evidence, ScientificEvidence):
        payload = {
            "property": evidence.property,
            "value": _stable_value(evidence.value),
            "unit": evidence.unit,
            "source_type": evidence.source_type,
            "fidelity": evidence.fidelity,
            "method": evidence.method,
            "structure_hash": evidence.structure_hash,
            "conditions": _stable_value(evidence.conditions),
        }
    else:
        payload = {
            "type": evidence.type,
            "content": _stable_content(evidence.content),
            "confidence": _stable_value(evidence.confidence),
            "provenance": _stable_value(evidence.provenance),
        }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def _hash_values(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in _HASH_KEYS and item not in (None, ""):
                found.add(f"{normalized}:{item}")
            found.update(_hash_values(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_hash_values(item))
    return found


def _stable_content(content: str) -> Any:
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return " ".join(content.split())
    return _stable_value(parsed)


def _stable_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return round(float(value) / _NUMERIC_QUANTUM) * _NUMERIC_QUANTUM
    if isinstance(value, dict):
        return {
            str(key): _stable_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            if str(key).casefold() not in _VOLATILE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    return str(value)


__all__ = ["ValidationProgress", "progress_from_evaluation"]
