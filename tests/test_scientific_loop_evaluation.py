from __future__ import annotations

import pytest

from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.evidence import Evidence
from photomatagent.scientific.loop.candidate import candidate_from_formula
from photomatagent.scientific.loop.evaluation import (
    EvidenceEvaluationPolicy,
    ScientificEvaluator,
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
        method="synthetic threshold fixture",
        fidelity=fidelity,
        summary=f"band gap {value} eV",
    )


def _test_evaluator(target: TargetSpec | None = None) -> ScientificEvaluator:
    """Enable synthetic evidence only in this deterministic algorithm fixture."""
    return ScientificEvaluator(
        target or _target(),
        policy=EvidenceEvaluationPolicy(allow_synthetic_evidence=True),
    )


def test_hard_constraint_pass_with_evidence():
    evaluator = _test_evaluator()
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
    evaluator = _test_evaluator()
    report = evaluator.evaluate(_candidate(), _scientific(_gap_evidence(0.21)))
    violation = report.violation_for("band_gap")
    assert violation is not None
    assert violation.observed_value == 0.21
    assert violation.target_value == 0.155
    assert report.verdict == "FAIL"
    assert report.hard_constraints_passed is False


def test_missing_evidence_is_unknown_never_pass():
    evaluator = _test_evaluator()
    report = evaluator.evaluate(_candidate(), _scientific())
    results = {r.property: r for r in report.constraint_results}
    assert results["band_gap"].result == "UNKNOWN"
    assert results["responsivity"].result == "UNKNOWN"
    assert report.verdict == "INCONCLUSIVE"
    assert report.hard_constraints_passed is False
    assert report.critical_evidence_gaps == ["band_gap", "responsivity"]


def test_all_constraints_pass_verdict():
    target = TargetSpec(
        goal="two supported-unit constraints",
        constraints=[
            ConstraintSpec(property="band_gap", operator="le", value=0.155, unit="eV"),
            ConstraintSpec(
                property="cutoff_wavelength", operator="ge", value=10.0, unit="um"
            ),
        ],
    )
    evaluator = _test_evaluator(target)
    state = _scientific(
        _gap_evidence(0.14),
        ScientificEvidence(
            subject="HgTe",
            property="cutoff_wavelength",
            value=10_500.0,
            unit="nm",
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


def test_matching_nonconvertible_units_do_not_require_conversion() -> None:
    evidence = ScientificEvidence(
        subject="HgTe",
        property="responsivity",
        value=1.4,
        unit="A/W",
        source="independent measurement archive",
        source_type="experimental",
        method="calibrated measurement",
        fidelity="experimental",
    )

    report = ScientificEvaluator(_target()).evaluate(
        _candidate(), _scientific(evidence)
    )

    result = next(r for r in report.constraint_results if r.property == "responsivity")
    assert result.result == "PASS"
    assert result.observed_value == 1.4


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
    assert result.result == "UNKNOWN"
    assert result.observed_value is None
    assert report.verdict == "INCONCLUSIVE"


def test_evidence_bound_to_a_different_material_is_not_used():
    evaluator = _test_evaluator()
    report = evaluator.evaluate(
        _candidate("PbTe"), _scientific(_gap_evidence(0.14, subject="HgTe"))
    )
    result = next(r for r in report.constraint_results if r.property == "band_gap")
    assert result.result == "UNKNOWN"


def test_candidate_declared_properties_cannot_validate_a_constraint():
    candidate = candidate_from_formula(
        "HgTe", extra_representation={"properties": {"band_gap": {"value": 0.1, "unit": "eV"}}}
    )
    evaluator = ScientificEvaluator(_target())
    report = evaluator.evaluate(candidate, _scientific())
    result = next(r for r in report.constraint_results if r.property == "band_gap")
    assert result.result == "UNKNOWN"
    assert report.verdict == "INCONCLUSIVE"


def test_declared_dft_is_not_measurement():
    target = TargetSpec(
        goal="check gap",
        constraints=[
            ConstraintSpec(
                property="band_gap", operator="le", value=1.2, unit="eV"
            )
        ],
    )
    candidate = candidate_from_formula(
        "Na0.75Ag0.25BiS2",
        extra_representation={
            "properties": {
                "band_gap": {"value": 1.1, "fidelity": "dft", "confidence": 1}
            }
        },
    )

    report = ScientificEvaluator(target).evaluate(candidate, ScientificState())

    assert report.constraint_results[0].result == "UNKNOWN"
    assert report.verdict == "INCONCLUSIVE"


def test_evaluation_without_candidate_is_inconclusive():
    evaluator = _test_evaluator()
    report = evaluator.evaluate(None, _scientific())
    assert report.verdict == "INCONCLUSIVE"
    assert set(report.critical_evidence_gaps) == {"band_gap", "responsivity"}


def test_contradicting_evidence_detected():
    evaluator = _test_evaluator()
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
    evaluator = _test_evaluator(target)
    report = evaluator.evaluate(
        _candidate(), _scientific(_gap_evidence(0.14))
    )
    assert report.verdict == "PASS"
    assert report.critical_evidence_gaps == []
    assert "cost" in report.evidence_gaps


def test_equivalent_energy_units_are_converted_before_threshold_comparison():
    target = TargetSpec(
        goal="gap",
        constraints=[
            ConstraintSpec(property="band_gap", operator="le", value=1.0, unit="eV")
        ],
    )
    evidence = ScientificEvidence(
        subject="HgTe",
        property="band_gap",
        value=1000.0,
        unit="meV",
        source="independent calculation archive",
        source_type="dft_calculation",
        method="PBE",
        fidelity="dft",
    )

    report = ScientificEvaluator(target).evaluate(_candidate(), _scientific(evidence))

    assert report.constraint_results[0].result == "PASS"
    assert report.constraint_results[0].observed_value == 1.0
    assert report.constraint_results[0].unit == "eV"


@pytest.mark.parametrize(
    ("value", "unit"),
    [(1.0, "A/cm2"), (1.0, ""), (float("nan"), "eV"), (float("inf"), "eV")],
)
def test_unusable_units_and_non_finite_values_remain_unknown(
    value: float, unit: str
) -> None:
    target = TargetSpec(
        goal="bounded evaluation",
        constraints=[
            ConstraintSpec(property="band_gap", operator="le", value=1.0, unit="eV")
        ],
    )
    evidence = ScientificEvidence(
        subject="HgTe",
        property="band_gap",
        value=value,
        unit=unit,
        source="independent calculation archive",
        source_type="dft_calculation",
        method="PBE",
        fidelity="dft",
    )

    report = ScientificEvaluator(target).evaluate(_candidate(), _scientific(evidence))

    assert report.constraint_results[0].result == "UNKNOWN"
    assert report.verdict == "INCONCLUSIVE"


def test_non_finite_constraint_limit_remains_unknown() -> None:
    target = TargetSpec(
        goal="reject an unbounded threshold",
        constraints=[
            ConstraintSpec(
                property="band_gap", operator="le", value=float("inf"), unit="eV"
            )
        ],
    )
    evidence = ScientificEvidence(
        subject="HgTe",
        property="band_gap",
        value=0.2,
        unit="eV",
        source="independent calculation archive",
        source_type="dft_calculation",
        method="PBE",
        fidelity="dft",
    )

    report = ScientificEvaluator(target).evaluate(_candidate(), _scientific(evidence))

    assert report.constraint_results[0].result == "UNKNOWN"
    assert "finite" in report.constraint_results[0].reason


def test_requirements_come_from_target_metadata_not_candidate_representation() -> None:
    target = TargetSpec(
        goal="structure-specific gap",
        constraints=[
            ConstraintSpec(property="band_gap", operator="le", value=1.0, unit="eV")
        ],
        metadata={
            "evidence_requirements": {
                "band_gap": {"scope": "structure", "allowed_fidelities": ["dft"]}
            }
        },
    )
    candidate = candidate_from_formula(
        "HgTe",
        extra_representation={
            "structure_hash": "sha256:target",
            "evidence_requirements": {"band_gap": {"scope": "composition"}},
        },
    )
    evidence = ScientificEvidence(
        subject="HgTe",
        property="band_gap",
        value=0.2,
        unit="eV",
        source="independent calculation archive",
        source_type="dft_calculation",
        method="PBE",
        fidelity="dft",
    )

    report = ScientificEvaluator(target).evaluate(candidate, _scientific(evidence))

    assert report.constraint_results[0].result == "UNKNOWN"
    assert "structure" in report.constraint_results[0].reason


def test_prior_is_background_only_even_when_value_would_pass() -> None:
    evidence = ScientificEvidence(
        subject="HgTe",
        property="band_gap",
        value=0.1,
        unit="eV",
        source="review article",
        source_type="literature",
        fidelity="experimental",
        assessment_role="prior",
    )

    report = ScientificEvaluator(_target()).evaluate(
        _candidate(), _scientific(evidence)
    )

    result = next(r for r in report.constraint_results if r.property == "band_gap")
    assert result.result == "UNKNOWN"
    assert "prior" in result.reason


def test_different_conditions_and_methods_do_not_create_false_contradictions() -> None:
    first = ScientificEvidence(
        subject="HgTe",
        property="band_gap",
        value=0.1,
        unit="eV",
        source="archive-1",
        source_type="dft_calculation",
        method="PBE",
        fidelity="dft",
        conditions={"temperature_k": 77},
    )
    different_condition = first.model_copy(
        update={"id": "condition-2", "value": 0.3, "conditions": {"temperature_k": 300}}
    )
    different_method = first.model_copy(
        update={"id": "method-2", "value": 0.4, "method": "HSE06"}
    )

    report = ScientificEvaluator(_target()).evaluate(
        _candidate(), _scientific(first, different_condition, different_method)
    )

    assert report.contradictions == []


def test_same_method_conditions_and_units_can_form_a_contradiction() -> None:
    first = ScientificEvidence(
        subject="HgTe",
        property="band_gap",
        value=0.1,
        unit="eV",
        source="archive-1",
        source_type="dft_calculation",
        method="PBE",
        fidelity="dft",
        conditions={"temperature_k": 77},
    )
    second = first.model_copy(
        update={"id": "same-scope-2", "source": "archive-2", "value": 300.0, "unit": "meV"}
    )

    report = ScientificEvaluator(_target()).evaluate(
        _candidate(), _scientific(first, second)
    )

    assert len(report.contradictions) == 1
    assert "band_gap" in report.contradictions[0]


def test_numerically_equal_conditions_share_conflict_scope() -> None:
    first = ScientificEvidence(
        subject="HgTe",
        property="band_gap",
        value=0.1,
        unit="eV",
        source="archive-1",
        source_type="dft_calculation",
        method="PBE",
        fidelity="dft",
        conditions={"temperature_k": 77},
    )
    second = first.model_copy(
        update={
            "id": "numeric-condition-2",
            "source": "archive-2",
            "value": 0.3,
            "conditions": {"temperature_k": 77.0},
        }
    )

    report = ScientificEvaluator(_target()).evaluate(
        _candidate(), _scientific(first, second)
    )

    assert len(report.contradictions) == 1
