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
from photomatagent.scientific.loop.evaluation import (
    DEFAULT_PROPERTY_ALIASES,
    EvaluationReport,
)
from photomatagent.scientific.discovery.composition import normalize_composition
from photomatagent.scientific.state import (
    DEFAULT_TRUSTED_EVIDENCE_TOOLS,
    ScientificState,
)


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
    allow_synthetic_evidence: bool = False,
) -> ValidationProgress:
    """Project accepted evaluator outcomes onto stable progress identities.

    An outcome is eligible only when its evidence reference resolves to an
    evidence item in ``scientific``.  Opaque ids, request ids, reasons and
    artifact filenames are intentionally excluded from the identity.  When a
    producer supplies a content/artifact hash it is authoritative; otherwise a
    bounded semantic digest is used with small numeric changes quantized.
    """

    if not report.candidate_id or report.candidate_id != candidate.candidate_id:
        return ValidationProgress(
            candidate_id=candidate.candidate_id,
            resolved_questions=(),
            observation_keys=(),
        )
    resolved: set[str] = set()
    observations: set[str] = set()
    evidence_by_result = {
        result.property: result
        for result in report.constraint_results
        if result.result in {"PASS", "FAIL", "UNKNOWN"} and result.evidence_ids
    }
    for property_name, result in evidence_by_result.items():
        matches = _accepted_evidence(
            property_name=property_name,
            references=result.evidence_ids,
            candidate=candidate,
            scientific=scientific,
            allow_synthetic_evidence=allow_synthetic_evidence,
        )
        if not matches:
            continue
        for evidence in matches:
            resolved.add(property_name)
            observations.add(_observation_key(evidence, property_name, result.result))
    return ValidationProgress(
        candidate_id=candidate.candidate_id,
        resolved_questions=tuple(sorted(resolved)),
        observation_keys=tuple(sorted(observations)),
    )


def _accepted_evidence(
    *,
    property_name: str,
    references: list[str],
    candidate: CandidateState,
    scientific: ScientificState,
    allow_synthetic_evidence: bool,
) -> list[Evidence | ScientificEvidence]:
    accepted: list[Evidence | ScientificEvidence] = []
    aliases = {property_name, *DEFAULT_PROPERTY_ALIASES.get(property_name, set())}
    for evidence in scientific.evidence:
        if not any(matches_evidence_ref(reference, evidence.id) for reference in references):
            continue
        attestation = scientific.verified_attestation(evidence.id)
        if attestation is None or not _accepted_attestation(
            attestation, allow_synthetic_evidence=allow_synthetic_evidence
        ):
            continue
        if not _property_matches(evidence, aliases):
            continue
        if not _subject_matches(evidence, candidate):
            continue
        evidence_candidate_id = str(getattr(evidence, "candidate_id", ""))
        if evidence_candidate_id and evidence_candidate_id != candidate.candidate_id:
            continue
        evidence_structure = str(getattr(evidence, "structure_hash", ""))
        candidate_structure = _candidate_structure_hash(candidate)
        if evidence_structure and candidate_structure and evidence_structure != candidate_structure:
            continue
        accepted.append(evidence)
    return accepted


def _accepted_attestation(
    attestation: Any, *, allow_synthetic_evidence: bool
) -> bool:
    return (
        attestation.authority == "observation"
        and attestation.origin == "trusted_builtin"
        and attestation.tool_name in DEFAULT_TRUSTED_EVIDENCE_TOOLS
    ) or (allow_synthetic_evidence and (
        attestation.authority == "synthetic"
        and attestation.origin == "synthetic_test"
    ))


def _property_matches(
    evidence: Evidence | ScientificEvidence, aliases: set[str]
) -> bool:
    if isinstance(evidence, ScientificEvidence):
        return str(evidence.property) in aliases and evidence.value is not None
    payload = _json_payload(evidence.content)
    return isinstance(payload, dict) and any(
        str(key) in aliases and value is not None for key, value in payload.items()
    )


def _subject_matches(
    evidence: Evidence | ScientificEvidence, candidate: CandidateState
) -> bool:
    subject: str | object
    if isinstance(evidence, ScientificEvidence):
        subject = evidence.subject
    else:
        payload = _json_payload(evidence.content)
        subject = (
            payload.get("material") or payload.get("formula")
            if isinstance(payload, dict)
            else ""
        )
    if not isinstance(subject, str) or not subject.strip() or not candidate.formula.strip():
        return False
    try:
        return normalize_composition(subject) == normalize_composition(candidate.formula)
    except (ValueError, RuntimeError):
        return False


def _candidate_structure_hash(candidate: CandidateState) -> str:
    for key in ("structure_hash", "cif_hash", "structure_identifier", "structure_id"):
        value = candidate.representation.get(key)
        if value:
            return str(value)
    return ""


def _json_payload(content: str) -> Any:
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed


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
