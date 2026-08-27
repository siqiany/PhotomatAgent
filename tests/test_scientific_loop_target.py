from __future__ import annotations

import pytest

from photomatagent.scientific.loop.candidate import (
    candidate_from_formula,
    candidate_fingerprint,
)
from photomatagent.scientific.loop.target import (
    ConstraintOutcome,
    ConstraintSpec,
    ConstraintViolation,
    TargetSpec,
    canonical_lwir_detector_target,
    evaluate_constraint,
)


def test_hard_constraint_pass():
    constraint = ConstraintSpec(
        property="band_gap", operator="le", value=0.155, unit="eV"
    )
    check = evaluate_constraint(constraint, 0.14)
    assert check.passed is True
    assert check.evaluable


def test_hard_constraint_fail():
    constraint = ConstraintSpec(
        property="band_gap", operator="le", value=0.155, unit="eV"
    )
    check = evaluate_constraint(constraint, 0.21)
    assert check.passed is False
    assert check.soft_score < 1.0


def test_missing_value_is_unknown_not_fail():
    constraint = ConstraintSpec(
        property="responsivity", operator="ge", value=1.0, unit="A/W"
    )
    check = evaluate_constraint(constraint, None)
    assert check.passed is None
    assert not check.evaluable


def test_between_and_equality_operators():
    between = ConstraintSpec(
        property="spectral", operator="between", value=[8.0, 14.0], unit="um"
    )
    assert evaluate_constraint(between, 12.0).passed is True
    assert evaluate_constraint(between, 15.0).passed is False
    equal = ConstraintSpec(property="x", operator="eq", value=1.5)
    assert evaluate_constraint(equal, 1.5).passed is True
    assert evaluate_constraint(equal, 1.6).passed is False


def test_constraint_violation_structured():
    constraint = ConstraintSpec(
        property="band_gap",
        operator="le",
        value=0.155,
        unit="eV",
        severity="HARD",
    )
    violation = ConstraintViolation.from_constraint(
        constraint, observed_value=0.21, evidence_ids=["sev_1"]
    )
    assert violation.property == "band_gap"
    assert violation.observed_value == 0.21
    assert violation.target_value == 0.155
    assert violation.severity == "HARD"
    assert violation.evidence_ids == ["sev_1"]
    assert "0.21" in violation.message


def test_target_spec_hard_and_soft_split():
    target = TargetSpec(
        goal="demo",
        constraints=[
            ConstraintSpec(property="a", operator="le", value=1, severity="HARD"),
            ConstraintSpec(property="b", operator="ge", value=2, severity="SOFT"),
        ],
    )
    assert [c.property for c in target.hard_constraints()] == ["a"]
    assert [c.property for c in target.soft_constraints()] == ["b"]
    assert target.constraint("a") is not None
    assert target.constraint("missing") is None


def test_canonical_lwir_target():
    target = canonical_lwir_detector_target()
    assert target.goal
    assert {c.property for c in target.hard_constraints()} == {"band_gap", "responsivity"}
    assert target.operating_conditions["temperature_k"] == 77


def test_candidate_fingerprint_detects_identical_formulas():
    a = candidate_from_formula("HgTe")
    b = candidate_from_formula("TeHg")
    c = candidate_from_formula("HgTe", extra_representation={"notes": "regenerated"})
    d = candidate_from_formula("Hg0.7Cd0.3Te")
    assert candidate_fingerprint(a) == candidate_fingerprint(b)
    assert candidate_fingerprint(a) == candidate_fingerprint(c)
    assert candidate_fingerprint(a) != candidate_fingerprint(d)


def test_constraint_outcome_model():
    outcome = ConstraintOutcome(
        property="responsivity",
        operator="ge",
        observed_value=0.5,
        target_value=1.0,
        unit="A/W",
        severity="HARD",
        result="FAIL",
        evidence_found=True,
    )
    assert outcome.result == "FAIL"
    assert outcome.evidence_found is True