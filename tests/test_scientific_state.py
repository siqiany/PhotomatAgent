from __future__ import annotations

import pytest

from photomatagent.runtime.evidence_attestation import _RuntimeEvidenceAuthority
from photomatagent.scientific.calculations import CalculationRecord
from photomatagent.scientific.claims import ScientificClaim
from photomatagent.scientific.evidence import Evidence
from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.discovery.models import HypothesisOrigin, HypothesisProposal
from photomatagent.scientific.discovery.registration import build_hypothesis
from photomatagent.scientific.state import EvidenceAttestation, ScientificState
from photomatagent.scientific.tasks import ScientificTask
from photomatagent.tools.registry import ToolRegistry
from photomatagent.tools.base import ToolError
from photomatagent.tools.scientific_state_inspect import ScientificStateInspectTool


def test_evidence_roundtrip():
    ev = Evidence(
        type="calculation",
        source="mock",
        content="band gap 0.31 eV",
        confidence=0.5,
        provenance={"tool": "mock.run_calculation"},
    )
    restored = Evidence.model_validate_json(ev.model_dump_json())
    assert restored == ev
    assert restored.provenance["tool"] == "mock.run_calculation"


def test_claim_roundtrip():
    claim = ScientificClaim(
        statement="GaAs has a direct band gap",
        confidence=0.7,
        supporting_evidence=["ev_1"],
        status="supported",
    )
    restored = ScientificClaim.model_validate_json(claim.model_dump_json())
    assert restored == claim


def test_calculation_record_roundtrip():
    record = CalculationRecord(
        backend="mock",
        task_type="band_structure",
        status="completed",
        input_reference={"material": "GaAs"},
        output_reference="mock://GaAs/band_structure",
        metadata={"band_gap": 0.31},
    )
    restored = CalculationRecord.model_validate_json(record.model_dump_json())
    assert restored == record


def test_task_statuses():
    task = ScientificTask(backend="mock", status="RUNNING")
    assert task.status == "RUNNING"
    task.status = "COMPLETED"
    assert task.status == "COMPLETED"


def test_state_collections():
    state = ScientificState(goal="understand GaAs")
    state.hypotheses.append("GaAs is suitable for IR detection")
    ev = state.add_evidence(
        Evidence(type="calculation", source="mock", content="gap 0.31", confidence=0.5)
    )
    state.add_claim(ScientificClaim(statement="gap is direct", supporting_evidence=[ev.id]))
    state.add_calculation(
        CalculationRecord(backend="mock", task_type="band_structure", status="completed")
    )
    state.add_task(ScientificTask(backend="mock"))
    assert len(state.evidence) == 1
    assert len(state.claims) == 1
    assert len(state.calculations) == 1
    assert len(state.pending_tasks) == 1
    assert state.goal == "understand GaAs"


def test_state_serializes_to_json():
    state = ScientificState(goal="x")
    payload = state.model_dump_json()
    restored = ScientificState.model_validate_json(payload)
    assert restored.goal == "x"


def test_evidence_attestation_roundtrip_and_legacy_default() -> None:
    evidence = ScientificEvidence(
        subject="HgTe", property="band_gap", value=0.2, unit="eV"
    )
    state = ScientificState(evidence=[evidence])
    assert state.evidence_attestations == {}

    authority = _RuntimeEvidenceAuthority()
    authority.bind(state)
    authority.attest(
        state,
        EvidenceAttestation(
            evidence_id=evidence.id,
            authority="observation",
            origin="trusted_builtin",
            tool_name="electronic.band_summary",
            tool_call_id="call-1",
        ),
    )
    payload = state.model_dump_json()
    restored = ScientificState.model_validate_json(payload)

    assert "host_proof" not in payload
    assert "runtime_authority" not in payload
    assert restored.evidence_attestations[evidence.id].authority == "observation"
    assert restored.verified_attestation(evidence.id) is None
    assert state == restored

    restored.goal = "different durable content"
    assert state != restored


def test_runtime_authority_is_not_a_public_scientific_state_api() -> None:
    state = ScientificState()

    assert not hasattr(state, "bind_runtime_authority")
    assert not hasattr(state, "attest_evidence")
    assert not hasattr(state, "clear_runtime_authority")


@pytest.mark.parametrize(
    "attestation_values",
    [
        {
            "evidence_id": "different-id",
            "authority": "observation",
            "origin": "trusted_builtin",
            "tool_name": "electronic.band_summary",
            "tool_call_id": "call-1",
        },
        {
            "authority": "observation",
            "origin": "untrusted_tool",
            "tool_name": "scientific.untrusted",
            "tool_call_id": "call-1",
        },
    ],
)
def test_tampered_snapshot_attestations_are_dropped_on_deserialization(
    attestation_values: dict[str, str],
) -> None:
    evidence = ScientificEvidence(
        id="sev-real", subject="HgTe", property="band_gap", value=0.1, unit="eV"
    )
    values = {"evidence_id": evidence.id, **attestation_values}
    state = ScientificState.model_validate(
        {
            "evidence": [evidence.model_dump(mode="json")],
            "evidence_attestations": {evidence.id: values},
        }
    )

    assert state.evidence_attestations == {}


def test_duplicate_evidence_ids_cannot_reuse_one_attestation() -> None:
    evidence = ScientificEvidence(
        id="sev-duplicate",
        subject="HgTe",
        property="band_gap",
        value=0.1,
        unit="eV",
    )
    state = ScientificState.model_validate(
        {
            "evidence": [evidence, evidence.model_copy()],
            "evidence_attestations": {
                evidence.id: {
                    "evidence_id": evidence.id,
                    "authority": "observation",
                    "origin": "trusted_builtin",
                    "tool_name": "electronic.band_summary",
                    "tool_call_id": "call-1",
                }
            },
        }
    )

    assert state.evidence_attestations == {}


def test_legacy_hypothesis_strings_remain_unchanged():
    restored = ScientificState.model_validate(
        {
            "goal": "old goal",
            "hypotheses": ["a historical free-text hypothesis"],
        }
    )

    assert restored.hypotheses == ["a historical free-text hypothesis"]
    assert restored.material_hypotheses == []


def _registered_hypothesis(index: int):
    return build_hypothesis(
        HypothesisProposal(
            request_id=f"inspect-{index}",
            formula=f"Na{index + 1}BiS2",
            statement=f"inspect statement {index}",
            design_operation="isovalent_substitution",
            basis=[
                {
                    "evidence_id": f"ev-{index}",
                    "relation": "supports",
                    "anchor": f"FULL BASIS {index}",
                }
            ],
            validation_questions=[f"inspect gap {index}"],
        ),
        HypothesisOrigin(
            tool_name="generation.register_hypothesis",
            tool_call_id=f"call-{index}",
            session_id="session",
            run_id="run",
            provider="fake",
            model="fake",
        ),
    )


@pytest.mark.asyncio
async def test_hypothesis_inspection_is_paginated_and_omits_basis_body():
    state = ScientificState(
        material_hypotheses=[_registered_hypothesis(index) for index in range(60)]
    )
    tool = ScientificStateInspectTool(state)

    first = await tool.execute({"section": "hypotheses"})
    second = await tool.execute({"section": "hypotheses", "offset": 10, "limit": 2})

    assert first.data["offset"] == 0
    assert first.data["limit"] == 10
    assert first.data["total"] == 60
    assert len(first.data["items"]) == 10
    assert second.data["offset"] == 10
    assert len(second.data["items"]) == 2
    assert state.material_hypotheses[10].id in second.output
    assert "FULL BASIS" not in first.output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"section": "hypotheses", "offset": True},
        {"section": "hypotheses", "offset": "1"},
        {"section": "hypotheses", "offset": -1},
        {"section": "hypotheses", "limit": False},
        {"section": "hypotheses", "limit": 1.5},
        {"section": "hypotheses", "limit": 0},
        {"section": "hypotheses", "limit": 51},
        {"section": "structures"},
        {"section": 1},
    ],
)
async def test_hypothesis_inspect_execute_rejects_invalid_boundaries(arguments):
    state = ScientificState(
        material_hypotheses=[_registered_hypothesis(index) for index in range(60)]
    )

    result = await ScientificStateInspectTool(state).execute(arguments)

    assert result.is_error is True
    assert result.data["error_type"] == "INVALID_INSPECT_ARGUMENTS"
    assert result.state_updates == []
    assert "FULL BASIS" not in result.output
    assert state.material_hypotheses[-1].id not in result.output


def test_hypothesis_inspection_bounds_offset_limit_and_excludes_structures():
    registry = ToolRegistry()
    registry.register(ScientificStateInspectTool(ScientificState()))

    assert "hypotheses" in ScientificStateInspectTool.input_schema["properties"][
        "section"
    ]["enum"]
    assert "structures" not in ScientificStateInspectTool.input_schema["properties"][
        "section"
    ]["enum"]
    registry.validate_arguments(
        "scientific_state_inspect",
        {"section": "hypotheses", "offset": 0, "limit": 50},
    )
    for invalid in (
        {"section": "hypotheses", "offset": -1},
        {"section": "hypotheses", "limit": 0},
        {"section": "hypotheses", "limit": 51},
        {"section": "structures"},
    ):
        with pytest.raises(ToolError):
            registry.validate_arguments("scientific_state_inspect", invalid)
