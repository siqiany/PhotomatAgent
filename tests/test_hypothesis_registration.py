from __future__ import annotations

from datetime import UTC, datetime

import pytest

from photomatagent.scientific.capabilities.generation.hypotheses import (
    RegisterHypothesisTool,
)
from photomatagent.scientific.capabilities.generation.tools import (
    GenerationCapabilitiesTool,
)
from photomatagent.scientific.discovery.models import (
    HypothesisOrigin,
    HypothesisProposal,
)
from photomatagent.scientific.discovery.registration import (
    build_hypothesis,
    validate_proposal,
)
from photomatagent.scientific.evidence import Evidence
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.base import ToolResult
from photomatagent.tools.exposure import ToolExposure


def proposal(**changes: object) -> HypothesisProposal:
    payload: dict[str, object] = {
        "request_id": "r1",
        "formula": "Na0.75Ag0.25BiS2",
        "statement": "Explore isovalent substitution",
        "design_operation": "isovalent_substitution",
        "validation_questions": ["Is phase separation preferred?"],
    }
    payload.update(changes)
    return HypothesisProposal.model_validate(payload)


def origin() -> HypothesisOrigin:
    return HypothesisOrigin(
        tool_name="generation.register_hypothesis",
        tool_call_id="call-1",
        session_id="session-1",
        run_id="run-1",
        provider="fake",
        model="fake",
    )


@pytest.mark.asyncio
async def test_registration_tool_returns_update_without_mutating_state() -> None:
    state = ScientificState()

    result = await RegisterHypothesisTool(state).execute(proposal().model_dump())

    assert isinstance(result, ToolResult)
    assert not result.is_error
    assert state.material_hypotheses == []
    assert state.evidence == []
    assert len(result.state_updates) == 1
    assert result.state_updates[0].proposal.request_id == "r1"
    assert "NO_EXTERNAL_BASIS" in result.data["diagnostics"]


@pytest.mark.asyncio
async def test_registration_tool_fails_soft_without_live_state() -> None:
    result = await RegisterHypothesisTool(None).execute(proposal().model_dump())

    assert result.is_error
    assert result.data["error_type"] == "STATE_UNAVAILABLE"
    assert result.state_updates == []


@pytest.mark.asyncio
async def test_generation_capabilities_advertises_mechanism_registration() -> None:
    result = await GenerationCapabilitiesTool().execute({})

    assert result.data["mechanism_reasoning"]["status"] == "AVAILABLE"
    assert result.data["mechanism_reasoning"]["validation_status"] == (
        "UNVALIDATED_HYPOTHESIS"
    )


def test_registration_tool_is_deferred_and_schema_forbids_runtime_origin() -> None:
    tool = RegisterHypothesisTool(ScientificState())

    assert tool.name == "generation.register_hypothesis"
    assert tool.exposure is ToolExposure.DEFERRED
    assert "origin" not in tool.input_schema["properties"]
    assert tool.input_schema["additionalProperties"] is False


def test_validate_proposal_checks_basis_parent_and_request_conflicts() -> None:
    state = ScientificState(
        evidence=[
            Evidence(
                id="ev-1",
                type="literature",
                source="paper",
                content="analogue",
                confidence=0.8,
            )
        ]
    )
    parent = build_hypothesis(proposal(request_id="parent"), origin())
    state.add_material_hypothesis(parent)
    child = proposal(
        request_id="child",
        parent_hypothesis_ids=[parent.id],
        basis=[{"evidence_id": "ev-1", "relation": "analogue"}],
    )

    assert validate_proposal(child, state) == []
    with pytest.raises(ValueError, match="unknown parent"):
        validate_proposal(
            proposal(request_id="bad-parent", parent_hypothesis_ids=["missing"]), state
        )
    with pytest.raises(ValueError, match="unknown evidence"):
        validate_proposal(
            proposal(
                request_id="bad-evidence",
                basis=[{"evidence_id": "missing", "relation": "supports"}],
            ),
            state,
        )
    with pytest.raises(ValueError, match="request_id.*conflict"):
        validate_proposal(
            proposal(request_id="parent", statement="changed payload"), state
        )


def test_build_and_state_append_are_stable_idempotent_and_immutable() -> None:
    state = ScientificState()
    first = build_hypothesis(proposal(), origin())
    equivalent = build_hypothesis(
        proposal(formula="Na3AgBi4S8"),
        origin().model_copy(update={"tool_call_id": "call-2"}),
    )

    assert first.candidate_id == equivalent.candidate_id
    assert first.id == equivalent.id
    assert first.created_at.tzinfo is UTC or first.created_at.utcoffset() is not None
    assert first.lineage.generated_by == "mechanism_reasoning"
    assert first.lineage.validation_status == "UNVALIDATED_HYPOTHESIS"
    assert state.add_material_hypothesis(first) is first
    retry = build_hypothesis(proposal(), origin())
    assert state.add_material_hypothesis(retry) is first
    assert len(state.material_hypotheses) == 1

    conflicting = build_hypothesis(
        proposal(statement="changed payload"),
        origin().model_copy(update={"tool_call_id": "call-3"}),
    )
    with pytest.raises(ValueError, match="request_id.*conflict"):
        state.add_material_hypothesis(conflicting)
    assert state.material_hypotheses == [first]
