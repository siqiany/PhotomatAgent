"""Conservative structured-evidence carry-forward between fresh episodes."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from datetime import UTC, datetime

from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.evidence import Evidence
from photomatagent.scientific.state import ScientificState

CarryableEvidence = Evidence | ScientificEvidence
_MODEL_SOURCE_TYPES = frozenset({"model", "generative_model"})


@dataclass(frozen=True, slots=True)
class EvidenceCarryDecision:
    """Auditable reason why one prior evidence item was carried or rejected."""

    evidence_id: str
    carried: bool
    reason: str


def select_carry_forward_evidence(
    previous_state: ScientificState,
    *,
    invalidated_evidence_ids: Collection[str] = (),
    subject: str | None = None,
) -> tuple[list[CarryableEvidence], tuple[EvidenceCarryDecision, ...]]:
    """Select only structured evidence safe to expose to a new episode."""

    invalidated = frozenset(invalidated_evidence_ids)
    selected: list[CarryableEvidence] = []
    decisions: list[EvidenceCarryDecision] = []
    seen_ids: set[str] = set()
    for item in previous_state.evidence:
        reason = _rejection_reason(
            item,
            invalidated=invalidated,
            subject=subject,
            seen_ids=seen_ids,
        )
        carried = reason is None
        decisions.append(
            EvidenceCarryDecision(
                evidence_id=item.id,
                carried=carried,
                reason=reason or "validated structured evidence",
            )
        )
        if carried:
            selected.append(item)
            seen_ids.add(item.id)
    return selected, tuple(decisions)


def build_inherited_scientific_state(
    previous_state: ScientificState,
    *,
    source_episode: str,
    invalidated_evidence_ids: Collection[str] = (),
    subject: str | None = None,
    inherited_at: datetime | None = None,
) -> tuple[ScientificState, tuple[EvidenceCarryDecision, ...]]:
    """Build a fresh state containing only eligible copied evidence."""

    selected, decisions = select_carry_forward_evidence(
        previous_state,
        invalidated_evidence_ids=invalidated_evidence_ids,
        subject=subject,
    )
    timestamp = (inherited_at or datetime.now(UTC)).astimezone(UTC)
    inherited_timestamp = timestamp.isoformat().replace("+00:00", "Z")
    copied = [
        item.model_copy(
            deep=True,
            update={
                "provenance": {
                    **item.provenance,
                    "inherited_from_episode": source_episode,
                    "inherited_at": inherited_timestamp,
                }
            },
        )
        for item in selected
    ]
    return ScientificState(goal=previous_state.goal, evidence=copied), decisions


def _rejection_reason(
    item: CarryableEvidence,
    *,
    invalidated: frozenset[str],
    subject: str | None,
    seen_ids: set[str],
) -> str | None:
    if not item.id.strip():
        return "missing stable evidence ID"
    if item.id in seen_ids:
        return "duplicate evidence ID"
    if item.id in invalidated:
        return "invalidated by confirmed revision"
    if isinstance(item, ScientificEvidence):
        if item.provenance.get("validated") is False:
            return "structured evidence is explicitly unvalidated"
        if item.source_type in _MODEL_SOURCE_TYPES:
            return "model-generated evidence is not carried"
        if item.fidelity == "ml_generated":
            return "ml-generated evidence is not carried"
        if not item.source.strip() or not item.method.strip():
            return "structured evidence lacks source or method provenance"
        evidence_subject = item.subject
    else:
        if item.provenance.get("validated") is not True:
            return "ordinary evidence lacks explicit validation"
        evidence_subject = str(item.provenance.get("subject", ""))
    if subject is not None and not _subjects_compatible(subject, evidence_subject):
        return "evidence subject does not match the current subject"
    return None


def _subjects_compatible(expected: str, actual: str) -> bool:
    return bool(actual.strip()) and (
        _normalize_subject(expected) == _normalize_subject(actual)
    )


def _normalize_subject(value: str) -> str:
    return "".join(value.casefold().split())


__all__ = [
    "EvidenceCarryDecision",
    "build_inherited_scientific_state",
    "select_carry_forward_evidence",
]
