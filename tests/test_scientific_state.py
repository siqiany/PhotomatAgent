from __future__ import annotations

from photomatagent.scientific.calculations import CalculationRecord
from photomatagent.scientific.claims import ScientificClaim
from photomatagent.scientific.evidence import Evidence
from photomatagent.scientific.state import ScientificState
from photomatagent.scientific.tasks import ScientificTask


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
