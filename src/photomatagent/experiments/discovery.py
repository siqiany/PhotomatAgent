"""Deterministic metrics and controlled comparisons for discovery pilots.

The metrics deliberately consume the structured scientific state.  Answer text
is presentation only and cannot establish a candidate, evidence, or success.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from photomatagent.observability.analyzer import SessionSummary
from photomatagent.runtime.events import RuntimeEvent
from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.state import ScientificState


class DiscoveryMetrics(BaseModel):
    """Counts derived from hypotheses, evidence, events, and usage metadata."""

    model_config = ConfigDict(extra="forbid")

    proposal_count: int = Field(default=0, ge=0)
    unique_composition_count: int = Field(default=0, ge=0)
    traceable_basis_count: int = Field(default=0, ge=0)
    basis_count: int = Field(default=0, ge=0)
    unknown_property_count: int = Field(default=0, ge=0)
    unsupported_validation_count: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    scientific_validity_rate: float | None = Field(default=None, ge=0, le=1)


class EvidenceFixture(BaseModel):
    """Immutable, offline fixture input for one pilot task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fixture_id: str = Field(min_length=1)
    evidence: tuple[ScientificEvidence, ...] = ()
    open_questions: tuple[str, ...] = ()

    @property
    def content_sha256(self) -> str:
        payload = {
            "fixture_id": self.fixture_id,
            "evidence": [item.model_dump(mode="json", exclude={"created_at"}) for item in self.evidence],
            "open_questions": list(self.open_questions),
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def _fixture_evidence(
    evidence_id: str,
    subject: str,
    *,
    property: str = "band_gap",
    value: object = None,
    source_type: str = "literature",
    fidelity: str = "analytical",
    source: str = "pilot-fixture",
) -> ScientificEvidence:
    return ScientificEvidence(
        id=evidence_id,
        assessment_role="prior",
        subject=subject,
        property=property,
        value=value,
        unit="eV" if property == "band_gap" else "",
        source=source,
        source_type=source_type,  # type: ignore[arg-type]
        fidelity=fidelity,  # type: ignore[arg-type]
        method="fixed offline fixture",
        limitations="Prior evidence only; not an independent candidate validation.",
    )


_FIXTURES: dict[str, EvidenceFixture] = {
    "vae-none": EvidenceFixture(fixture_id="vae-none", open_questions=("No external evidence is available.",)),
    "mechanism-endmember-v1": EvidenceFixture(
        fixture_id="mechanism-endmember-v1",
        evidence=(_fixture_evidence("prior-nabis2", "NaBiS2", value=1.2),),
    ),
    "known-materials-v1": EvidenceFixture(
        fixture_id="known-materials-v1",
        evidence=(_fixture_evidence("prior-gaas", "GaAs", value=1.42),),
    ),
    "empty-literature-v1": EvidenceFixture(fixture_id="empty-literature-v1", open_questions=("Literature snapshot is empty.",)),
    "na-ag-bi-s-v1": EvidenceFixture(
        fixture_id="na-ag-bi-s-v1",
        evidence=(_fixture_evidence("prior-nabis2-endmember", "NaBiS2", value=1.2),),
    ),
    "polymorph-v1": EvidenceFixture(
        fixture_id="polymorph-v1",
        evidence=(_fixture_evidence("prior-polymorph", "NaBiS2", value=1.2),),
        open_questions=("Structure-specific evidence is missing.",),
    ),
    "unstable-substitution-v1": EvidenceFixture(
        fixture_id="unstable-substitution-v1",
        evidence=(_fixture_evidence("prior-unstable", "AgBiS2", value=0.9),),
        open_questions=("Phase competition remains unresolved.",),
    ),
    "bandgap-wavelength-v1": EvidenceFixture(
        fixture_id="bandgap-wavelength-v1",
        evidence=(_fixture_evidence("prior-bandgap", "NaBiS2", value=1.35),),
        open_questions=("Mechanism and wavelength compatibility require checking.",),
    ),
    "process-limits-v1": EvidenceFixture(
        fixture_id="process-limits-v1",
        open_questions=("Synthesis feasibility under the stated temperature is unknown.",),
    ),
    "missing-competitors-v1": EvidenceFixture(
        fixture_id="missing-competitors-v1",
        open_questions=("Competing phase inventory is incomplete.",),
    ),
    "followup-evidence-v1": EvidenceFixture(
        fixture_id="followup-evidence-v1",
        evidence=(_fixture_evidence("prior-followup", "NaBiS2", value=1.2),),
        open_questions=("A candidate-specific independent calculation is required.",),
    ),
    "forged-fidelity-v1": EvidenceFixture(
        fixture_id="forged-fidelity-v1",
        evidence=(_fixture_evidence("prior-forged", "NaBiS2", value=1.2, source_type="model", fidelity="ml_generated"),),
        open_questions=("Claimed DFT fidelity and citation must be independently checked.",),
    ),
}


def get_evidence_fixture(fixture_id: str) -> EvidenceFixture:
    if not fixture_id:
        return EvidenceFixture(fixture_id="empty")
    try:
        # Never expose the authoritative registry object.  ``ScientificEvidence``
        # has mutable nested provenance, so a shallow Pydantic copy is unsafe.
        return _FIXTURES[fixture_id].model_copy(deep=True)
    except KeyError as exc:
        raise ValueError(f"unknown evidence fixture: {fixture_id}") from exc


def inject_evidence_fixture(state: ScientificState, fixture_id: str) -> EvidenceFixture:
    """Copy fixed prior inputs into a fresh state without runtime authority."""
    fixture = get_evidence_fixture(fixture_id)
    for evidence in fixture.evidence:
        state.add_evidence(evidence.model_copy(deep=True))
    state.open_questions.extend(fixture.open_questions)
    return fixture


def fixture_snapshot_sha256(fixture_ids: Sequence[str]) -> str:
    # Hash the private authoritative templates, independent of caller-owned
    # copies returned by ``get_evidence_fixture`` or ``inject_evidence_fixture``.
    unknown = [identifier for identifier in fixture_ids if identifier and identifier not in _FIXTURES]
    if unknown:
        raise ValueError(f"unknown evidence fixture: {unknown[0]}")
    fixtures = [
        _FIXTURES[identifier] if identifier else EvidenceFixture(fixture_id="empty")
        for identifier in fixture_ids
    ]
    payload = [(fixture.fixture_id, fixture.content_sha256) for fixture in fixtures]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class AblationSpec(BaseModel):
    """The only supported ablation changes are explicitly named workflow arms."""

    model_config = ConfigDict(extra="forbid")

    treatment: Literal["workflow"]
    arms: list[str] = Field(min_length=2)
    controlled_fields: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_control_fields(self) -> "AblationSpec":
        if len(set(self.arms)) != len(self.arms):
            raise ValueError("ablation arms must be unique")
        allowed = {
            "provider", "model", "task_set", "budget", "evidence_snapshot", "workflow",
            "system_prompt", "skill_index", "context_builder", "context_engine", "tool_surface",
        }
        unknown = set(self.controlled_fields) - allowed
        if unknown:
            raise ValueError(f"unsupported controlled fields: {', '.join(sorted(unknown))}")
        required = {"provider", "model", "task_set", "budget", "evidence_snapshot"}
        missing = required - set(self.controlled_fields)
        if missing:
            raise ValueError("ablation must control: " + ", ".join(sorted(missing)))
        return self


def compute_discovery_metrics(
    state: ScientificState,
    *,
    events: Iterable[RuntimeEvent] = (),
    session_summary: SessionSummary | None = None,
    tool_calls: int | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> DiscoveryMetrics:
    """Calculate discovery metrics from structured state and runtime metadata.

    A prior or proposal is useful for traceability but is not an independent
    scientific observation.  Therefore validity remains unavailable until a
    matching observation from a calculation or experiment exists.
    """
    hypotheses = state.material_hypotheses
    evidence = list(state.evidence)
    evidence_ids = {item.id for item in evidence}
    by_candidate_property: dict[tuple[str, str], list[ScientificEvidence]] = {}
    for item in evidence:
        if not isinstance(item, ScientificEvidence):
            continue
        if item.assessment_role != "observation":
            continue
        by_candidate_property.setdefault((item.candidate_id, item.property), []).append(item)

    unknown_properties = 0
    unsupported_validation = 0
    for hypothesis in hypotheses:
        candidate = hypothesis.candidate_id
        for effect in hypothesis.proposal.expected_effects:
            if not by_candidate_property.get((candidate, effect.property)):
                unknown_properties += 1
        for _question in hypothesis.proposal.validation_questions:
            if not any(item.candidate_id == candidate for item in _observations(evidence)):
                unsupported_validation += 1

    summary = session_summary
    event_tool_calls = sum(1 for event in events if event.kind == "tool_completed")
    return DiscoveryMetrics(
        proposal_count=len(hypotheses),
        unique_composition_count=len(composition_identities(state)),
        traceable_basis_count=sum(
            1 for item in hypotheses for basis in item.proposal.basis if basis.evidence_id in evidence_ids
        ),
        basis_count=sum(len(item.proposal.basis) for item in hypotheses),
        unknown_property_count=unknown_properties,
        unsupported_validation_count=unsupported_validation,
        tool_calls=tool_calls if tool_calls is not None else (summary.tool_calls if summary else event_tool_calls),
        input_tokens=input_tokens if input_tokens is not None else (summary.input_tokens if summary else None),
        output_tokens=output_tokens if output_tokens is not None else (summary.output_tokens if summary else None),
        # P0 has no public deterministic checker result contract to consume.
        scientific_validity_rate=None,
    )


def _observations(evidence: Sequence[object]) -> list[ScientificEvidence]:
    return [
        item for item in evidence
        if isinstance(item, ScientificEvidence) and item.assessment_role == "observation"
    ]


def composition_identities(state: ScientificState) -> set[tuple[tuple[str, int], ...]]:
    """Return canonical composition identities already produced by registration.

    Registration owns parsing and normalization; metrics only consume that
    immutable contract and never reimplement chemistry parsing.
    """
    return {tuple(record.normalized_composition) for record in state.material_hypotheses}
