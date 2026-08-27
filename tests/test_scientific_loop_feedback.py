from __future__ import annotations

from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.loop.candidate import candidate_from_formula
from photomatagent.scientific.loop.evaluation import ScientificEvaluator
from photomatagent.scientific.loop.feedback import (
    build_feedback,
    format_feedback_for_model,
)
from photomatagent.scientific.loop.target import (
    ConstraintSpec,
    TargetSpec,
)
from photomatagent.scientific.state import ScientificState


def _target() -> TargetSpec:
    return TargetSpec(
        goal="LWIR detector",
        constraints=[
            ConstraintSpec(
                property="band_gap", operator="le", value=0.155, unit="eV"
            ),
            ConstraintSpec(
                property="responsivity", operator="ge", value=1.0, unit="A/W"
            ),
        ],
    )


def _evaluate(candidate, *evidence) -> object:
    evaluator = ScientificEvaluator(_target())
    state = ScientificState()
    for item in evidence:
        state.add_evidence(item)  # type: ignore[arg-type]
    return evaluator.evaluate(candidate, state)


def _gap(value: float, *, fidelity: str = "dft") -> ScientificEvidence:
    return ScientificEvidence(
        subject="HgTe",
        property="band_gap",
        value=value,
        unit="eV",
        source="synthetic",
        source_type="dft_calculation",
        fidelity=fidelity,
    )


def test_pass_produces_no_feedback():
    candidate = candidate_from_formula("HgTe")
    report = _evaluate(
        candidate,
        _gap(0.14),
        ScientificEvidence(
            subject="HgTe",
            property="responsivity",
            value=1.4,
            unit="A/W",
            source="synthetic",
            source_type="experimental",
            fidelity="experimental",
        ),
    )
    signal = build_feedback(_target(), candidate, report, [])
    assert signal is None


def test_hard_violation_prioritizes_revision():
    candidate = candidate_from_formula("HgTe")
    report = _evaluate(candidate, _gap(0.21))
    signal = build_feedback(_target(), candidate, report, [])
    assert signal is not None
    assert signal.decision == "REVISE"
    assert any(a.target_property == "band_gap" for a in signal.recommended_actions)
    assert "band_gap" in signal.summary


def test_missing_evidence_drives_continue_and_calculate():
    candidate = candidate_from_formula("HgTe")
    report = _evaluate(candidate)
    signal = build_feedback(_target(), candidate, report, [])
    assert signal is not None
    assert signal.decision == "CONTINUE"
    actions = {a.target_property for a in signal.recommended_actions}
    assert {"band_gap", "responsivity"} <= actions
    assert signal.evidence_gaps == ["band_gap", "responsivity"]


def test_low_fidelity_critical_escalates():
    candidate = candidate_from_formula("HgTe")
    report = _evaluate(candidate, _gap(0.14, fidelity="empirical"))
    signal = build_feedback(_target(), candidate, report, [])
    assert signal is not None
    assert signal.decision == "ESCALATE"
    assert any(
        a.action_type == "ESCALATE_FIDELITY" for a in signal.recommended_actions
    )


def test_repeated_candidate_is_rejected_with_change_strategy():
    first = candidate_from_formula("HgTe")
    second = candidate_from_formula("HgTe")  # identical fingerprint
    report = _evaluate(second, _gap(0.21))
    signal = build_feedback(_target(), second, report, [first])
    assert signal is not None
    assert signal.decision == "REJECT"
    assert any(
        a.action_type == "CHANGE_STRATEGY" for a in signal.recommended_actions
    )
    assert signal.prohibited_repeats == ["HgTe"]


def test_format_feedback_is_a_research_instruction():
    candidate = candidate_from_formula("HgTe")
    report = _evaluate(candidate, _gap(0.21))
    signal = build_feedback(_target(), candidate, report, [])
    assert signal is not None
    text = format_feedback_for_model(signal, round_number=2)
    assert "Scientific feedback from round 2" in text
    assert "band_gap = 0.21" in text
    assert "responsivity" in text
    assert "Do not claim completion" in text


def test_contradictions_produce_validate_action():
    candidate = candidate_from_formula("HgTe")
    evaluator = ScientificEvaluator(_target())
    state = ScientificState()
    state.add_evidence(_gap(0.14, fidelity="dft"))
    state.add_evidence(
        ScientificEvidence(
            subject="HgTe",
            property="band_gap",
            value=0.4,
            unit="eV",
            source="other",
            source_type="experimental",
            fidelity="experimental",
        )
    )
    report = evaluator.evaluate(candidate, state)
    signal = build_feedback(_target(), candidate, report, [])
    assert signal is not None
    assert any(a.action_type == "VALIDATE" for a in signal.recommended_actions)