from __future__ import annotations

import pytest

from photomatagent.scientific.loop.candidate import candidate_from_formula
from photomatagent.scientific.loop.policy import (
    ScientificLoopDecision,
    ScientificLoopPolicy,
    ScientificLoopState,
    ScientificLoopSummary,
)
from photomatagent.scientific.loop.stagnation import StagnationDetector
from photomatagent.scientific.loop.target import (
    ConstraintSpec,
    TargetSpec,
)
from photomatagent.scientific.loop.evaluation import (
    EvidenceEvaluationPolicy,
    EvaluationReport,
    ScientificEvaluator,
)
from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.state import ScientificState


def _target() -> TargetSpec:
    return TargetSpec(
        goal="LWIR detector",
        constraints=[
            ConstraintSpec(property="band_gap", operator="le", value=0.155, unit="eV"),
            ConstraintSpec(property="responsivity", operator="ge", value=1.0, unit="A/W"),
        ],
    )


def _passing_report() -> EvaluationReport:
    candidate = candidate_from_formula(
        "HgTe", extra_representation={"structure_hash": "synthetic:device"}
    )
    evaluator = ScientificEvaluator(
        _target(), policy=EvidenceEvaluationPolicy(allow_synthetic_evidence=True)
    )
    state = ScientificState()
    state.add_evidence(
        ScientificEvidence(
            subject="HgTe", property="band_gap", value=0.14, unit="eV",
            source="synthetic", source_type="dft_calculation", fidelity="dft",
            method="synthetic method", structure_hash="synthetic:device",
            conditions={"temperature_k": 77},
        )
    )
    state.add_evidence(
        ScientificEvidence(
            subject="HgTe", property="responsivity", value=1.4, unit="A/W",
            source="synthetic", source_type="experimental", fidelity="experimental",
            method="synthetic device measurement", structure_hash="synthetic:device",
            conditions={"wavelength_um": 10.0, "bias_v": 0.1, "temperature_k": 77,
                        "measurement_definition": "synthetic calibrated response"},
        )
    )
    return evaluator.evaluate(candidate, state)


def _state(rounds: int = 1) -> ScientificLoopState:
    return ScientificLoopState(target=_target())


def test_policy_success_when_all_hard_pass_and_confidence_high():
    report = _passing_report()
    assert report.verdict == "PASS"
    state = _state()
    state.round = 1
    state.best_score = report.score
    state.best_candidate_id = "cand_1"
    decision = ScientificLoopPolicy().decide(
        evaluation=report,
        state=state,
        stagnation=StagnationDetector(),
        max_rounds=6,
        max_candidates=12,
        min_confidence=0.6,
    )
    assert decision.action == "SUCCESS"
    assert decision.best_candidate_id == "cand_1"


def test_policy_success_requires_confidence_threshold():
    report = _passing_report()
    state = _state()
    decision = ScientificLoopPolicy().decide(
        evaluation=report,
        state=state,
        stagnation=StagnationDetector(),
        max_rounds=6,
        max_candidates=12,
        min_confidence=0.99,
    )
    assert decision.action == "CONTINUE"


def test_policy_does_not_finish_before_pending_candidates_are_evaluated():
    report = _passing_report()
    state = _state()
    state.pending_candidate_ids = ["cand_pending"]

    decision = ScientificLoopPolicy().decide(
        evaluation=report,
        state=state,
        stagnation=StagnationDetector(),
        max_rounds=6,
        max_candidates=12,
        min_confidence=0.6,
    )

    assert decision.action == "CONTINUE"


def test_policy_continues_on_resolvable_gap():
    candidate = candidate_from_formula("HgTe")
    evaluator = ScientificEvaluator(_target())
    report = evaluator.evaluate(candidate, ScientificState())
    state = _state()
    state.round = 1
    decision = ScientificLoopPolicy().decide(
        evaluation=report,
        state=state,
        stagnation=StagnationDetector(),
        max_rounds=6,
        max_candidates=12,
        min_confidence=0.6,
    )
    assert decision.action == "CONTINUE"


def _inconclusive_report() -> EvaluationReport:
    candidate = candidate_from_formula("HgTe")
    return ScientificEvaluator(_target()).evaluate(candidate, ScientificState())


def test_policy_escalates_on_request():
    report = _inconclusive_report()
    decision = ScientificLoopPolicy().decide(
        evaluation=report,
        state=_state(),
        stagnation=StagnationDetector(),
        max_rounds=6,
        max_candidates=12,
        min_confidence=0.6,
        escalate_requested=True,
    )
    assert decision.action == "ESCALATE"


def test_policy_inconclusive_with_reason():
    report = _inconclusive_report()
    decision = ScientificLoopPolicy().decide(
        evaluation=report,
        state=_state(),
        stagnation=StagnationDetector(),
        max_rounds=6,
        max_candidates=12,
        min_confidence=0.6,
        inconclusive_reason="no capability available",
    )
    assert decision.action == "INCONCLUSIVE"


def test_policy_inconclusive_without_evaluation():
    decision = ScientificLoopPolicy().decide(
        evaluation=None,
        state=_state(),
        stagnation=StagnationDetector(),
        max_rounds=6,
        max_candidates=12,
        min_confidence=0.6,
    )
    assert decision.action == "INCONCLUSIVE"


def test_inconclusive_candidate_is_not_failed():
    state = ScientificLoopState(target=TargetSpec(goal="check a hypothesis"))
    candidate = candidate_from_formula("NaBiS2")
    state.add_candidate(
        candidate,
        EvaluationReport(
            candidate_id=candidate.candidate_id,
            verdict="INCONCLUSIVE",
            score=0,
        ),
    )

    assert state.candidates[0].status == "INCONCLUSIVE"
    assert state.best_candidate_id == candidate.candidate_id


def test_loop_state_defaults_queue_fields_for_legacy_snapshots():
    restored = ScientificLoopState.model_validate(
        {"target": TargetSpec(goal="legacy").model_dump(mode="json")}
    )

    assert restored.pending_candidate_ids == []
    assert restored.active_candidate_id is None


def test_candidate_projection_is_updated_while_evaluations_remain_history():
    state = ScientificLoopState(target=TargetSpec(goal="rank candidates"))
    first = candidate_from_formula("HgTe")
    second = candidate_from_formula("PbTe")
    state.add_candidate(
        first,
        EvaluationReport(candidate_id=first.candidate_id, verdict="PASS", score=0.9),
    )
    state.add_candidate(
        second,
        EvaluationReport(candidate_id=second.candidate_id, verdict="PASS", score=0.6),
    )
    contradicted = first.model_copy(update={"label": "HgTe re-evaluated"})

    state.add_candidate(
        contradicted,
        EvaluationReport(candidate_id=first.candidate_id, verdict="FAIL", score=0.1),
    )

    assert len(state.candidates) == 2
    assert len(state.evaluations) == 3
    assert state.candidates[0].label == "HgTe re-evaluated"
    assert state.candidates[0].status == "FAIL"
    assert state.best_candidate_id == second.candidate_id
    assert state.best_score == 0.6


def test_policy_budget_exhausted_on_rounds():
    state = _state()
    state.round = 7
    decision = ScientificLoopPolicy().decide(
        evaluation=_passing_report(),
        state=state,
        stagnation=StagnationDetector(),
        max_rounds=6,
        max_candidates=12,
        min_confidence=0.6,
    )
    assert decision.action == "BUDGET_EXHAUSTED"


def test_policy_budget_exhausted_on_candidate_count():
    state = _state()
    state.candidates = [candidate_from_formula(f"M{i}" ) for i in range(13)]
    decision = ScientificLoopPolicy().decide(
        evaluation=None,
        state=state,
        stagnation=StagnationDetector(),
        max_rounds=6,
        max_candidates=12,
        min_confidence=0.6,
    )
    assert decision.action == "BUDGET_EXHAUSTED"


def test_policy_stalled_when_stagnation_detected():
    state = _state()
    detector = StagnationDetector(patience=3)
    candidate = candidate_from_formula("HgTe")
    evaluator = ScientificEvaluator(_target())
    report = evaluator.evaluate(candidate, ScientificState())  # score 0.0
    for _ in range(4):
        detector.record(candidate, report)
    assert detector.stalled
    decision = ScientificLoopPolicy().decide(
        evaluation=report,
        state=state,
        stagnation=detector,
        max_rounds=12,
        max_candidates=12,
        min_confidence=0.6,
    )
    assert decision.action == "STALLED"


def test_loop_decision_and_summary_models():
    decision = ScientificLoopDecision(action="SUCCESS", reason="done", best_candidate_id="cand_1")
    assert decision.action == "SUCCESS"
    summary = ScientificLoopSummary(
        status="SUCCESS",
        rounds=3,
        candidate_count=2,
        best_candidate_id="cand_1",
        best_score=0.9,
        final_evaluation=None,
        termination_reason="done",
    )
    assert summary.candidate_count == 2
    assert summary.final_evaluation is None
