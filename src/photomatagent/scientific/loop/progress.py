"""Stable identities for scientific validation progress.

The evaluator deliberately exposes opaque evidence references.  The loop must
therefore derive progress from the accepted outcome and the observation's
content identity, never from producer supplied ids or request metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

from photomatagent.scientific.evidence_refs import matches_evidence_ref
from photomatagent.scientific.loop.candidate import CandidateState
from photomatagent.scientific.loop.evaluation import EvaluationReport
from photomatagent.scientific.loop.observation import (
    observation_key,
    stable_observation_identity,
)
from photomatagent.scientific.state import ScientificState


@dataclass(frozen=True)
class ValidationProgress:
    """Accepted validation work discovered in one evaluation."""

    candidate_id: str
    resolved_questions: tuple[str, ...]
    observation_keys: tuple[str, ...]


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
    public_results = {
        (result.property, result.result): result
        for result in report.constraint_results
    }
    for record in report.accepted_evidence_manifest:
        result = public_results.get((record.property, record.outcome))
        matching = [
            evidence
            for evidence in scientific.evidence
            if str(evidence.id) == record.evidence_id
        ]
        if result is None or len(matching) != 1 or not result.evidence_ids:
            continue
        if not any(
            matches_evidence_ref(reference, record.evidence_id)
            for reference in result.evidence_ids
        ):
            continue
        evidence = matching[0]
        attestation = scientific.verified_attestation(record.evidence_id)
        if attestation is None:
            continue
        if (
            attestation.authority != record.attestation_authority
            or attestation.origin != record.attestation_origin
            or attestation.tool_name != record.attestation_tool_name
            or attestation.tool_call_id != record.attestation_tool_call_id
        ):
            continue
        if stable_observation_identity(evidence) != record.observation_identity:
            continue
        resolved.add(record.property)
        observations.add(observation_key(evidence, record.property, record.outcome))
    return ValidationProgress(
        candidate_id=candidate.candidate_id,
        resolved_questions=tuple(sorted(resolved)),
        observation_keys=tuple(sorted(observations)),
    )

__all__ = ["ValidationProgress", "progress_from_evaluation"]
