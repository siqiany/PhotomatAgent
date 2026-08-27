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
    candidate = candidate_from_formula("HgTe")
    evaluator = ScientificEvaluator(_target())
    state = ScientificState()
    state.add_evidence(
        ScientificEvidence(
            subject="HgTe", property="band_gap", value=0.14, unit="eV",
            source="s", source_type="dft_calculation", fidelity="dft",
        )
    )
    state.add_evidence(
        ScientificEvidence(
            subject="HgTe", property="responsivity", value=1.4, unit="A/W",
            source="s2", source_type="experimental", fidelity="experimental",
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