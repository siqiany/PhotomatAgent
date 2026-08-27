from __future__ import annotations

import pytest

from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.evidence import Evidence
from photomatagent.scientific.loop.candidate import candidate_from_formula
from photomatagent.scientific.loop.evaluation import ScientificEvaluator
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


def _candidate(formula: str = "HgTe") -> object:
    return candidate_from_formula(formula)


def _scientific(*evidence: object) -> ScientificState:
    state = ScientificState()
    for item in evidence:
        state.add_evidence(item)  # type: ignore[arg-type]
    return state


def _gap_evidence(value: float, *, fidelity: str = "dft", subject: str = "HgTe") -> ScientificEvidence:
    return ScientificEvidence(
        subject=subject,
        property="band_gap",
        value=value,
        unit="eV",
        source="synthetic",
        source_type="dft_calculation",
        fidelity=fidelity,
        summary=f"band gap {value} eV",
    )


def test_hard_constraint_pass_with_evidence():
    evaluator = ScientificEvaluator(_target())
    report = evaluator.evaluate(
        _candidate(), _scientific(_gap_evidence(0.14))
    )
    band_gap = report.violation_for("band_gap")
    assert band_gap is None
    result = next(r for r in report.constraint_results if r.property == "band_gap")
    assert result.result == "PASS"
    assert result.observed_value == 0.14
    assert report.verdict == "INCONCLUSIVE"  # responsivity evidence missing
    assert report.critical_evidence_gaps == ["responsivity"]


def test_hard_constraint_fail_produces_violation():
    evaluator = ScientificEvaluator(_target())
    report = evaluator.evaluate(_candidate(), _scientific(_gap_evidence(0.21)))
    violation = report.violation_for("band_gap")
    assert violation is not None
    assert violation.observed_value == 0.21
    assert violation.target_value == 0.155
    assert report.verdict == "FAIL"
    assert report.hard_constraints_passed is False


def test_missing_evidence_is_unknown_never_pass():
    evaluator = ScientificEvaluator(_target())
    report = evaluator.evaluate(_candidate(), _scientific())
    results = {r.property: r for r in report.constraint_results}
    assert results["band_gap"].result == "UNKNOWN"
    assert results["responsivity"].result == "UNKNOWN"
    assert report.verdict == "INCONCLUSIVE"
    assert report.hard_constraints_passed is False
    assert report.critical_evidence_gaps == ["band_gap", "responsivity"]


def test_all_constraints_pass_verdict():
    evaluator = ScientificEvaluator(_target())
    state = _scientific(
        _gap_evidence(0.14),
        ScientificEvidence(
            subject="HgTe",
            property="responsivity",
            value=1.4,
            unit="A/W",
            source="synthetic",
            source_type="analytical_model",
            fidelity="analytical",
        ),
    )
    report = evaluator.evaluate(_candidate(), state)
    assert report.verdict == "PASS"
    assert report.hard_constraints_passed is True
    assert report.violations == []
    assert report.critical_evidence_gaps == []
    assert report.score > 0.0


def test_json_payload_evidence_from_free_text_evidence():
    """mock.run_calculation stores JSON payloads in Evidence.content."""
    evidence = Evidence(
        type="calculation",
        source="mock",
        content='{"material": "HgTe", "band_gap": 0.31, "gap_type": "direct"}',
        confidence=0.5,
    )
    evaluator = ScientificEvaluator(_target())
    report = evaluator.evaluate(_candidate(), _scientific(evidence))
    result = next(r for r in report.constraint_results if r.property == "band_gap")
    assert result.result == "FAIL"
    assert result.observed_value == 0.31
    assert report.verdict == "FAIL"


def test_evidence_bound_to_a_different_material_is_not_used():
    evaluator = ScientificEvaluator(_target())
    report = evaluator.evaluate(
        _candidate("PbTe"), _scientific(_gap_evidence(0.14, subject="HgTe"))
    )
    result = next(r for r in report.constraint_results if r.property == "band_gap")
    assert result.result == "UNKNOWN"


def test_candidate_declared_properties_are_low_fidelity():
    candidate = candidate_from_formula(
        "HgTe", extra_representation={"properties": {"band_gap": {"value": 0.1, "unit": "eV"}}}
    )
    evaluator = ScientificEvaluator(_target())
    report = evaluator.evaluate(candidate, _scientific())
    result = next(r for r in report.constraint_results if r.property == "band_gap")
    assert result.result == "PASS"
    assert report.confidence <= 0.25  # ml_generated confidence


def test_evaluation_without_candidate_is_inconclusive():
    evaluator = ScientificEvaluator(_target())
    report = evaluator.evaluate(None, _scientific())
    assert report.verdict == "INCONCLUSIVE"
    assert set(report.critical_evidence_gaps) == {"band_gap", "responsivity"}


def test_contradicting_evidence_detected():
    evaluator = ScientificEvaluator(_target())
    state = _scientific(
        _gap_evidence(0.14, subject="HgTe", fidelity="dft"),
        _gap_evidence(0.31, subject="HgTe", fidelity="experimental"),
    )
    report = evaluator.evaluate(_candidate(), state)
    assert any("band_gap" in item for item in report.contradictions)


def test_soft_constraint_unknown_does_not_block_pass():
    target = TargetSpec(
        goal="demo",
        constraints=[
            ConstraintSpec(property="band_gap", operator="le", value=0.155, unit="eV"),
            ConstraintSpec(
                property="cost", operator="le", value=10, severity="SOFT"
            ),
        ],
    )
    evaluator = ScientificEvaluator(target)
    report = evaluator.evaluate(
        _candidate(), _scientific(_gap_evidence(0.14))
    )
    assert report.verdict == "PASS"
    assert report.critical_evidence_gaps == []
    assert "cost" in report.evidence_gaps