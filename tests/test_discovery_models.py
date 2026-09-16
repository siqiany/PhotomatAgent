from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from photomatagent.scientific.capabilities.generation.lineage import CandidateLineage
from photomatagent.scientific.discovery.models import (
    BasisReference,
    DiscoveryConstraints,
    ExpectedEffect,
    HypothesisOrigin,
    HypothesisProposal,
    HypothesisRegistration,
    ScientificHypothesis,
)


def _proposal_payload() -> dict[str, object]:
    return {
        "request_id": "r1",
        "formula": "NaBiS2",
        "statement": "Test a parent composition",
        "design_operation": "prototype_transfer",
        "validation_questions": ["Does the parent structure remain stable?"],
    }


def _scientific_hypothesis(
    *,
    proposal: HypothesisProposal | None = None,
    lineage: CandidateLineage | None = None,
    normalized_composition: object = (("Bi", 1), ("Na", 1), ("S", 2)),
) -> ScientificHypothesis:
    return ScientificHypothesis.model_validate({
        "id": "hyp_1",
        "candidate_id": "cand_1",
        "proposal": proposal or HypothesisProposal(**_proposal_payload()),
        "normalized_composition": normalized_composition,
        "request_payload_sha256": "a" * 64,
        "lineage": lineage or CandidateLineage(
            candidate_id="cand_1",
            generated_by="mechanism_reasoning",
            validation_status="UNVALIDATED_HYPOTHESIS",
        ),
        "origin": HypothesisOrigin(
            tool_name="hypothesis.register",
            tool_call_id="call_1",
            session_id="session_1",
            run_id="run_1",
            provider="test",
            model="test-model",
        ),
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
    })


def test_model_cannot_submit_validation_status() -> None:
    payload = _proposal_payload()
    payload["status"] = "PASS"

    with pytest.raises(ValidationError):
        HypothesisProposal.model_validate(payload)


def test_proposal_rejects_unexpanded_composition_variable() -> None:
    payload = _proposal_payload()
    payload["formula"] = "Na1-xAgxBiS2"

    with pytest.raises(ValidationError):
        HypothesisProposal.model_validate(payload)


@pytest.mark.parametrize("field", ["status", "fidelity", "confidence", "origin"])
def test_proposal_rejects_runtime_owned_or_conclusion_fields(field: str) -> None:
    payload = _proposal_payload()
    payload[field] = "forbidden"

    with pytest.raises(ValidationError):
        HypothesisProposal.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("request_id", ""),
        ("request_id", "r" * 129),
        ("formula", "f" * 257),
        ("statement", "s" * 2001),
        ("parent_hypothesis_ids", [f"h{i}" for i in range(9)]),
        ("basis", [{"evidence_id": f"e{i}", "relation": "supports"} for i in range(17)]),
        ("assumptions", ["a"] * 13),
        ("assumptions", ["a" * 1001]),
        (
            "expected_effects",
            [{"property": f"p{i}", "direction": "change", "rationale": "r"} for i in range(13)],
        ),
        ("counter_hypotheses", ["c"] * 9),
        ("counter_hypotheses", ["c" * 1001]),
        ("validation_questions", []),
        ("validation_questions", ["q"] * 13),
        ("validation_questions", ["q" * 1001]),
        ("synthesis_notes", ["n"] * 9),
        ("synthesis_notes", ["n" * 1501]),
    ],
)
def test_proposal_enforces_design_limits(field: str, value: object) -> None:
    payload = _proposal_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        HypothesisProposal.model_validate(payload)


def test_proposal_accepts_complete_bounded_contract() -> None:
    proposal = HypothesisProposal(
        request_id="r1",
        formula="NaBiS2",
        statement="Test a parent composition",
        design_operation="isovalent_substitution",
        parent_hypothesis_ids=["hyp_parent"],
        basis=[BasisReference(evidence_id="ev1", relation="analogue", anchor="table 1")],
        assumptions=["The analogue remains relevant."],
        expected_effects=[
            ExpectedEffect(property="band_gap", direction="change", rationale="Alloy disorder")
        ],
        counter_hypotheses=["Phase separation dominates."],
        validation_questions=["Is the structure dynamically stable?"],
        synthesis_notes=["Assumption: attempt a sulfur-rich condition."],
    )

    assert proposal.basis[0].relation == "analogue"
    assert proposal.expected_effects[0].direction == "change"


def test_registration_contains_only_a_proposal() -> None:
    registration = HypothesisRegistration(proposal=HypothesisProposal(**_proposal_payload()))
    assert registration.proposal.request_id == "r1"

    with pytest.raises(ValidationError):
        HypothesisRegistration.model_validate({
            "proposal": _proposal_payload(),
            "origin": {"provider": "model-controlled"},
        })


def test_scientific_hypothesis_reuses_candidate_lineage() -> None:
    lineage = CandidateLineage(
        candidate_id="cand_1",
        generated_by="mechanism_reasoning",
        validation_status="UNVALIDATED_HYPOTHESIS",
    )
    record = ScientificHypothesis(
        id="hyp_1",
        candidate_id="cand_1",
        proposal=HypothesisProposal(**_proposal_payload()),
        normalized_composition=(("Bi", 1), ("Na", 1), ("S", 2)),
        request_payload_sha256="a" * 64,
        lineage=lineage,
        origin=HypothesisOrigin(
            tool_name="hypothesis.register",
            tool_call_id="call_1",
            session_id="session_1",
            run_id="run_1",
            provider="test",
            model="test-model",
        ),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(record.lineage, CandidateLineage)
    with pytest.raises(ValidationError):
        ScientificHypothesis.model_validate({**record.model_dump(), "status": "PASS"})


@pytest.mark.parametrize(
    "lineage",
    [
        CandidateLineage(
            candidate_id="cand_1",
            generated_by="mattergen",
            validation_status="UNVALIDATED_HYPOTHESIS",
        ),
        CandidateLineage(
            candidate_id="cand_1",
            generated_by="mechanism_reasoning",
            validation_status="PASS",
        ),
    ],
)
def test_scientific_hypothesis_requires_unvalidated_mechanism_lineage(
    lineage: CandidateLineage,
) -> None:
    with pytest.raises(ValidationError):
        ScientificHypothesis(
            id="hyp_1",
            candidate_id="cand_1",
            proposal=HypothesisProposal(**_proposal_payload()),
            normalized_composition=(("Bi", 1), ("Na", 1), ("S", 2)),
            request_payload_sha256="a" * 64,
            lineage=lineage,
            origin=HypothesisOrigin(
                tool_name="hypothesis.register",
                tool_call_id="call_1",
                session_id="session_1",
                run_id="run_1",
                provider="test",
                model="test-model",
            ),
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )


def test_scientific_hypothesis_takes_deeply_immutable_nested_snapshots() -> None:
    proposal = HypothesisProposal(
        **_proposal_payload(),
        assumptions=["Initial assumption"],
        basis=[BasisReference(evidence_id="ev1", relation="supports")],
    )
    lineage = CandidateLineage(
        candidate_id="cand_1",
        generated_by="mechanism_reasoning",
        validation_status="UNVALIDATED_HYPOTHESIS",
        generation_parameters={"nested": {"source": "initial"}},
        source_artifacts=["source.json"],
    )
    record = _scientific_hypothesis(proposal=proposal, lineage=lineage)

    assert record.proposal is not proposal
    assert record.lineage is not lineage
    with pytest.raises(ValidationError):
        record.proposal.statement = "rewritten"
    with pytest.raises(TypeError):
        record.proposal.assumptions.append("rewritten")
    with pytest.raises(ValidationError):
        record.proposal.basis[0].anchor = "rewritten"
    with pytest.raises(ValidationError):
        record.lineage.validation_status = "PASS"
    with pytest.raises(TypeError):
        record.lineage.source_artifacts.append("rewritten.json")
    with pytest.raises(TypeError):
        record.lineage.generation_parameters["changed"] = True
    with pytest.raises(TypeError):
        record.lineage.generation_parameters["nested"]["source"] = "rewritten"

    proposal.statement = "externally rewritten"
    proposal.assumptions.append("externally rewritten")
    lineage.validation_status = "PASS"
    lineage.source_artifacts.append("external.json")
    lineage.generation_parameters["nested"]["source"] = "externally rewritten"

    assert record.proposal.statement == "Test a parent composition"
    assert record.proposal.assumptions == ["Initial assumption"]
    assert record.lineage.validation_status == "UNVALIDATED_HYPOTHESIS"
    assert record.lineage.source_artifacts == ["source.json"]
    assert record.lineage.generation_parameters == {"nested": {"source": "initial"}}
    assert record.model_copy(deep=True) == record


@pytest.mark.parametrize(
    "normalized_composition",
    [
        (("Na", 1), ("Na", 1)),
        (("Bogus", 1),),
        (("Na", True),),
        (("Na", 1.0),),
        (("Na", "1"),),
        (("Na", 1), ("Cl", 1)),
        (("Na", 0),),
        (("Na", -1),),
        (("Cl", 2), ("Na", 2)),
    ],
)
def test_scientific_hypothesis_rejects_noncanonical_composition_domain(
    normalized_composition: object,
) -> None:
    with pytest.raises(ValidationError):
        _scientific_hypothesis(normalized_composition=normalized_composition)


def test_discovery_constraints_have_neutral_defaults() -> None:
    constraints = DiscoveryConstraints()

    assert constraints.required_elements == []
    assert constraints.forbidden_elements == []
    assert constraints.allow_isovalent_alloy is None
    assert constraints.allow_donor_acceptor_doping is None
