"""Validation and deterministic construction for hypothesis registration."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from photomatagent.scientific.capabilities.generation.lineage import CandidateLineage
from photomatagent.scientific.discovery.composition import (
    composition_key,
    normalize_composition,
)
from photomatagent.scientific.discovery.models import (
    HypothesisOrigin,
    HypothesisProposal,
    ScientificHypothesis,
)
from photomatagent.scientific.state import ScientificState


def _canonical_payload(proposal: HypothesisProposal) -> dict[str, Any]:
    payload = proposal.model_dump(mode="json")
    payload["formula"] = [list(item) for item in normalize_composition(proposal.formula)]
    return payload


def proposal_payload_sha256(proposal: HypothesisProposal) -> str:
    encoded = json.dumps(
        _canonical_payload(proposal),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hypothesis_id(proposal: HypothesisProposal) -> str:
    return f"hyp_{proposal_payload_sha256(proposal)[:24]}"


def candidate_id(proposal: HypothesisProposal) -> str:
    return f"cand_{composition_key(proposal.formula)[:24]}"


def validate_proposal(
    proposal: HypothesisProposal, state: ScientificState
) -> list[str]:
    """Validate state references and return non-fatal registration diagnostics."""

    # Canonicalization is a validation boundary as well as an identity function.
    normalize_composition(proposal.formula)
    known_hypotheses = {record.id for record in state.material_hypotheses}
    unknown_parents = sorted(set(proposal.parent_hypothesis_ids) - known_hypotheses)
    if unknown_parents:
        raise ValueError(f"unknown parent hypothesis id(s): {', '.join(unknown_parents)}")

    known_evidence = {item.id for item in state.evidence}
    unknown_evidence = sorted(
        {item.evidence_id for item in proposal.basis} - known_evidence
    )
    if unknown_evidence:
        raise ValueError(f"unknown evidence id(s): {', '.join(unknown_evidence)}")

    payload_hash = proposal_payload_sha256(proposal)
    existing = next(
        (
            record
            for record in state.material_hypotheses
            if record.proposal.request_id == proposal.request_id
        ),
        None,
    )
    if existing is not None and existing.request_payload_sha256 != payload_hash:
        raise ValueError(
            f"request_id {proposal.request_id!r} conflicts with an existing payload"
        )

    diagnostics: list[str] = []
    if not proposal.basis:
        diagnostics.append("NO_EXTERNAL_BASIS")
    if existing is not None:
        diagnostics.append("ALREADY_REGISTERED")
    return diagnostics


def build_hypothesis(
    proposal: HypothesisProposal, origin: HypothesisOrigin
) -> ScientificHypothesis:
    """Build an immutable hypothesis record from model input and runtime origin."""

    normalized = normalize_composition(proposal.formula)
    identity = candidate_id(proposal)
    return ScientificHypothesis(
        id=hypothesis_id(proposal),
        candidate_id=identity,
        proposal=proposal,
        normalized_composition=normalized,
        request_payload_sha256=proposal_payload_sha256(proposal),
        lineage=CandidateLineage(
            candidate_id=identity,
            generated_by="mechanism_reasoning",
            generation_parameters={
                "parent_hypothesis_ids": list(proposal.parent_hypothesis_ids)
            },
            transformation=proposal.design_operation,
            validation_status="UNVALIDATED_HYPOTHESIS",
        ),
        origin=origin,
        created_at=datetime.now(UTC),
    )

