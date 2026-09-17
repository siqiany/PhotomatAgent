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
    canonical_lwir_detector_target,
)
from photomatagent.scientific.state import EvidenceAttestation, ScientificState


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


def _attested_scientific(*evidence: object) -> ScientificState:
    state = _scientific(*evidence)
    for item in state.evidence:
        state.evidence_attestations[item.id] = EvidenceAttestation.host_create(
            evidence_id=item.id,
            authority="observation",
            origin="trusted_builtin",
            tool_name="electronic.band_summary",
            tool_call_id=f"call-{item.id}",
        )
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
    target = TargetSpec(
        goal="density",
        constraints=[
            ConstraintSpec(property="density", operator="ge", value=5.0, unit="g/cm3")
        ],
    )
    evidence = ScientificEvidence(
        subject="HgTe",
        property="density",
        value=8.1,
        unit="g/cm3",
        source="structure parser",
        source_type="calculation",
        method="pymatgen density",
        fidelity="analytical",
    )

    report = ScientificEvaluator(target).evaluate(
        _candidate(), _attested_scientific(evidence)
    )

    result = report.constraint_results[0]
    assert result.result == "PASS"
    assert result.observed_value == 8.1


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


def test_fully_laundered_unattested_evidence_cannot_validate() -> None:
    evidence = ScientificEvidence(
        subject="HgTe",
        property="band_gap",
        value=0.1,
        unit="eV",
        source="innocent independent archive",
        source_type="dft_calculation",
        method="claimed PBE",
        fidelity="dft",
        assessment_role="observation",
        provenance={"tool": "innocent.validator", "trusted": True},
    )

    report = ScientificEvaluator(_target()).evaluate(
        _candidate(), _scientific(evidence)
    )

    assert report.constraint_results[0].result == "UNKNOWN"
    assert "UNATTESTED" in report.constraint_results[0].reason


@pytest.mark.parametrize("tamper", ["mismatched_id", "incompatible_origin", "proof"])
def test_directly_tampered_attestation_remains_unknown(tamper: str) -> None:
    evidence = ScientificEvidence(
        id="sev-tampered",
        subject="HgTe",
        property="band_gap",
        value=0.1,
        unit="eV",
        source="archive",
        source_type="dft_calculation",
        method="PBE",
        fidelity="dft",
    )
    state = ScientificState(evidence=[evidence])
    attestation = EvidenceAttestation.host_create(
        evidence_id=evidence.id,
        authority="observation",
        origin="trusted_builtin",
        tool_name="electronic.band_summary",
        tool_call_id="call-1",
    )
    if tamper == "mismatched_id":
        attestation.evidence_id = "different-id"
    elif tamper == "incompatible_origin":
        attestation.origin = "untrusted_tool"
    else:
        attestation.host_proof = "forged"
    state.evidence_attestations[evidence.id] = attestation

    report = ScientificEvaluator(_target()).evaluate(_candidate(), state)

    assert report.constraint_results[0].result == "UNKNOWN"


def test_host_shaped_attestation_from_untrusted_tool_remains_unknown() -> None:
    evidence = ScientificEvidence(
        id="sev-untrusted-tool",
        subject="HgTe",
        property="band_gap",
        value=0.1,
        unit="eV",
        source="archive",
        source_type="dft_calculation",
        method="PBE",
        fidelity="dft",
    )
    state = ScientificState(evidence=[evidence])
    state.attest_evidence(
        EvidenceAttestation.host_create(
            evidence_id=evidence.id,
            authority="observation",
            origin="trusted_builtin",
            tool_name="renamed.untrusted",
            tool_call_id="call-1",
        )
    )

    report = ScientificEvaluator(_target()).evaluate(_candidate(), state)

    assert report.constraint_results[0].result == "UNKNOWN"


def test_evaluation_without_candidate_is_inconclusive():
    evaluator = _test_evaluator()
    report = evaluator.evaluate(None, _scientific())
    assert report.verdict == "INCONCLUSIVE"
    assert set(report.critical_evidence_gaps) == {"band_gap", "responsivity"}


def test_missing_structure_and_conditions_prevent_contradiction():
    evaluator = _test_evaluator()
    state = _scientific(
        _gap_evidence(0.14, subject="HgTe", fidelity="dft"),
        _gap_evidence(0.31, subject="HgTe", fidelity="experimental"),
    )
    report = evaluator.evaluate(_candidate(), state)
    assert report.contradictions == []


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

    report = ScientificEvaluator(target).evaluate(
        _candidate(), _attested_scientific(evidence)
    )

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
    assert report.constraint_results[0].reason == "CONSTRAINT_TARGET_INVALID"


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

    report = ScientificEvaluator(target).evaluate(
        candidate, _attested_scientific(evidence)
    )

    assert report.constraint_results[0].result == "UNKNOWN"
    assert "EVIDENCE_STRUCTURE_MISSING" in report.constraint_results[0].reason


@pytest.mark.parametrize(
    "missing",
    [
        "structure_hash",
        "wavelength_um",
        "bias_v",
        "temperature_k",
        "measurement_definition",
    ],
)
def test_device_property_safe_defaults_require_complete_scope(missing: str) -> None:
    candidate = candidate_from_formula(
        "HgTe", extra_representation={"structure_hash": "sha256:device"}
    )
    conditions: dict[str, object] = {
        "wavelength_um": 10.0,
        "bias_v": 0.1,
        "temperature_k": 77,
        "measurement_definition": "external quantum efficiency calibrated",
    }
    evidence_values: dict[str, object] = {
        "subject": "HgTe",
        "property": "responsivity",
        "value": 1.4,
        "unit": "A/W",
        "source": "device measurement archive",
        "source_type": "experimental",
        "method": "calibrated device measurement",
        "fidelity": "experimental",
        "structure_hash": "sha256:device",
        "conditions": conditions,
    }
    if missing == "structure_hash":
        evidence_values["structure_hash"] = ""
    else:
        conditions.pop(missing)
    evidence = ScientificEvidence(**evidence_values)

    report = ScientificEvaluator(_target()).evaluate(
        candidate, _attested_scientific(evidence)
    )

    result = next(item for item in report.constraint_results if item.property == "responsivity")
    assert result.result == "UNKNOWN"
    expected_code = (
        "EVIDENCE_STRUCTURE_MISSING"
        if missing == "structure_hash"
        else f"CONDITION_MISSING:{missing}"
    )
    assert expected_code in result.reason


def test_fully_scoped_attested_device_observation_can_pass() -> None:
    candidate = candidate_from_formula(
        "HgTe", extra_representation={"structure_hash": "sha256:device"}
    )
    evidence = ScientificEvidence(
        subject="HgTe",
        property="responsivity",
        value=1.4,
        unit="A/W",
        source="device measurement archive",
        source_type="experimental",
        method="calibrated device measurement",
        fidelity="experimental",
        structure_hash="sha256:device",
        conditions={
            "wavelength_um": 10.0,
            "bias_v": 0.1,
            "temperature_k": 77,
            "measurement_definition": "external quantum efficiency calibrated",
        },
    )

    report = ScientificEvaluator(_target()).evaluate(
        candidate, _attested_scientific(evidence)
    )

    result = next(item for item in report.constraint_results if item.property == "responsivity")
    assert result.result == "PASS"


def test_target_metadata_cannot_relax_device_safe_defaults() -> None:
    target = TargetSpec(
        goal="device",
        constraints=[
            ConstraintSpec(property="responsivity", operator="ge", value=1.0, unit="A/W")
        ],
        metadata={"evidence_requirements": {"responsivity": {"scope": "composition"}}},
    )
    evidence = ScientificEvidence(
        subject="HgTe",
        property="responsivity",
        value=2.0,
        unit="A/W",
        source="archive",
        source_type="experimental",
        method="measurement",
        fidelity="experimental",
    )

    report = ScientificEvaluator(target).evaluate(
        _candidate(), _attested_scientific(evidence)
    )

    assert report.constraint_results[0].result == "UNKNOWN"
    assert "CANDIDATE_STRUCTURE_MISSING" in report.constraint_results[0].reason


def test_target_metadata_can_tighten_device_operating_conditions() -> None:
    target = TargetSpec(
        goal="device",
        constraints=[
            ConstraintSpec(property="responsivity", operator="ge", value=1.0, unit="A/W")
        ],
        metadata={
            "evidence_requirements": {
                "responsivity": {"conditions": {"temperature_k": 77}}
            }
        },
    )
    candidate = candidate_from_formula(
        "HgTe", extra_representation={"structure_hash": "sha256:device"}
    )
    evidence = ScientificEvidence(
        subject="HgTe",
        property="responsivity",
        value=2.0,
        unit="A/W",
        source="archive",
        source_type="experimental",
        method="measurement",
        fidelity="experimental",
        structure_hash="sha256:device",
        conditions={
            "wavelength_um": 10.0,
            "bias_v": 0.1,
            "temperature_k": 300,
            "measurement_definition": "external quantum efficiency calibrated",
        },
    )

    report = ScientificEvaluator(target).evaluate(
        candidate, _attested_scientific(evidence)
    )

    assert report.constraint_results[0].result == "UNKNOWN"
    assert "CONDITION_MISMATCH:temperature_k" in report.constraint_results[0].reason


@pytest.mark.parametrize(
    ("condition_update", "expected_code"),
    [
        ({"temperature_k": 300}, "CONDITION_MISMATCH:temperature_k"),
        ({"wavelength_um": 1.0}, "CONDITION_OUT_OF_RANGE:wavelength_um"),
        ({"temperature_k": None}, "CONDITION_INVALID:temperature_k"),
        ({"wavelength_um": float("nan")}, "CONDITION_INVALID:wavelength_um"),
        ({"bias_v": float("inf")}, "CONDITION_INVALID:bias_v"),
        ({"measurement_definition": "   "}, "CONDITION_INVALID:measurement_definition"),
    ],
)
def test_canonical_device_target_rejects_invalid_or_out_of_domain_conditions(
    condition_update: dict[str, object], expected_code: str
) -> None:
    candidate = candidate_from_formula(
        "HgTe", extra_representation={"structure_hash": "sha256:device"}
    )
    conditions: dict[str, object] = {
        "wavelength_um": 10.0,
        "bias_v": 0.1,
        "temperature_k": 77,
        "measurement_definition": "calibrated device responsivity",
    }
    conditions.update(condition_update)
    evidence = ScientificEvidence(
        subject="HgTe",
        property="responsivity",
        value=2.0,
        unit="A/W",
        source="archive",
        source_type="experimental",
        method="measurement",
        fidelity="experimental",
        structure_hash="sha256:device",
        conditions=conditions,
    )

    report = ScientificEvaluator(canonical_lwir_detector_target()).evaluate(
        candidate, _attested_scientific(evidence)
    )
    result = next(item for item in report.constraint_results if item.property == "responsivity")

    assert result.result == "UNKNOWN"
    assert expected_code in result.reason


def test_canonical_device_target_accepts_valid_operating_domain() -> None:
    candidate = candidate_from_formula(
        "HgTe", extra_representation={"structure_hash": "sha256:device"}
    )
    evidence = ScientificEvidence(
        subject="HgTe",
        property="responsivity",
        value=2.0,
        unit="A/W",
        source="archive",
        source_type="experimental",
        method="measurement",
        fidelity="experimental",
        structure_hash="sha256:device",
        conditions={
            "wavelength_um": 10.0,
            "bias_v": 0.1,
            "temperature_k": 77,
            "measurement_definition": "calibrated device responsivity",
        },
    )

    report = ScientificEvaluator(canonical_lwir_detector_target()).evaluate(
        candidate, _attested_scientific(evidence)
    )

    result = next(item for item in report.constraint_results if item.property == "responsivity")
    assert result.result == "PASS"


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
        _candidate(), _attested_scientific(evidence)
    )

    result = next(r for r in report.constraint_results if r.property == "band_gap")
    assert result.result == "UNKNOWN"
    assert "ROLE_BACKGROUND" in result.reason


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
        structure_hash="sha256:phase-a",
    )
    second = first.model_copy(
        update={"id": "same-scope-2", "source": "archive-2", "value": 300.0, "unit": "meV"}
    )

    report = ScientificEvaluator(_target()).evaluate(
        candidate_from_formula(
            "HgTe", extra_representation={"structure_hash": "sha256:phase-a"}
        ),
        _attested_scientific(first, second),
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
        structure_hash="sha256:phase-a",
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
        candidate_from_formula(
            "HgTe", extra_representation={"structure_hash": "sha256:phase-a"}
        ),
        _attested_scientific(first, second),
    )

    assert len(report.contradictions) == 1


@pytest.mark.parametrize("missing", ["structure", "conditions"])
def test_missing_comparability_fields_do_not_form_contradictions(missing: str) -> None:
    values: dict[str, object] = {
        "subject": "HgTe",
        "property": "band_gap",
        "value": 0.1,
        "unit": "eV",
        "source": "archive-1",
        "source_type": "dft_calculation",
        "method": "PBE",
        "fidelity": "dft",
        "structure_hash": "sha256:phase-a",
        "conditions": {"temperature_k": 77},
    }
    if missing == "structure":
        values["structure_hash"] = ""
    else:
        values["conditions"] = {}
    first = ScientificEvidence(**values)
    second = first.model_copy(update={"id": "second", "value": 0.3})

    report = ScientificEvaluator(_target()).evaluate(
        _candidate(), _attested_scientific(first, second)
    )

    assert report.contradictions == []


def test_opaque_subjects_do_not_form_contradictions() -> None:
    first = ScientificEvidence(
        subject="sample alpha",
        property="band_gap",
        value=0.1,
        unit="eV",
        source="archive-1",
        source_type="dft_calculation",
        method="PBE",
        fidelity="dft",
        structure_hash="sha256:phase-a",
        conditions={"temperature_k": 77},
    )
    second = first.model_copy(update={"id": "opaque-2", "value": 0.3})

    report = ScientificEvaluator(_target()).evaluate(
        _candidate(), _attested_scientific(first, second)
    )

    assert report.contradictions == []


def test_exclusion_reason_is_bounded_and_does_not_echo_untrusted_fields() -> None:
    secret = "SECRET_TOKEN_" + "x" * 1000
    state = ScientificState(
        evidence=[
            ScientificEvidence(
                subject="HgTe",
                property="band_gap",
                value=0.1,
                unit="eV",
                source=secret,
                source_type="dft_calculation",
                method=secret,
                fidelity="dft",
                provenance={"tool": secret},
            )
            for _ in range(100)
        ]
    )

    report = ScientificEvaluator(_target()).evaluate(_candidate(), state)
    reason = report.constraint_results[0].reason

    assert "SECRET_TOKEN" not in reason
    assert len(reason) <= 512


def test_contradiction_diagnostics_are_categorical_and_bounded() -> None:
    evidence = [
        ScientificEvidence(
            id=f"SECRET_TOKEN_{index}_" + "x" * 200,
            subject="HgTe",
            property="band_gap",
            value=0.1 + index,
            unit="eV",
            source="synthetic",
            source_type="dft_calculation",
            method="PBE",
            fidelity="dft",
            structure_hash="sha256:phase-a",
            conditions={"temperature_k": 77},
        )
        for index in range(30)
    ]
    candidate = candidate_from_formula(
        "HgTe", extra_representation={"structure_hash": "sha256:phase-a"}
    )

    report = _test_evaluator().evaluate(candidate, _scientific(*evidence))

    assert report.contradictions == ["band_gap:COMPARABLE_VALUES_DISAGREE"]
    assert "SECRET_TOKEN" not in report.rationale


def test_one_evidence_record_cannot_contradict_itself_across_target_units() -> None:
    target = TargetSpec(
        goal="same observation in two units",
        constraints=[
            ConstraintSpec(property="band_gap", operator="le", value=1.0, unit="eV"),
            ConstraintSpec(property="band_gap", operator="ge", value=500.0, unit="meV"),
        ],
    )
    evidence = ScientificEvidence(
        subject="HgTe",
        property="band_gap",
        value=0.8,
        unit="eV",
        source="archive",
        source_type="dft_calculation",
        method="PBE",
        fidelity="dft",
        structure_hash="sha256:phase-a",
        conditions={"temperature_k": 77},
    )

    report = ScientificEvaluator(target).evaluate(
        candidate_from_formula(
            "HgTe", extra_representation={"structure_hash": "sha256:phase-a"}
        ),
        _attested_scientific(evidence),
    )

    assert [item.result for item in report.constraint_results] == ["PASS", "PASS"]
    assert report.contradictions == []


def test_accepted_evidence_uses_opaque_id_and_categorical_reason() -> None:
    raw_id = "/private/raw/path/SECRET_TOKEN_" + "x" * 2000
    evidence = ScientificEvidence(
        id=raw_id,
        subject="HgTe",
        property="band_gap",
        value=0.123456789,
        unit="eV",
        source="archive",
        source_type="dft_calculation",
        method="PBE",
        fidelity="dft",
    )

    report = ScientificEvaluator(_target()).evaluate(
        _candidate(), _attested_scientific(evidence)
    )
    result = next(item for item in report.constraint_results if item.property == "band_gap")

    assert result.result == "PASS"
    assert result.reason == "CONSTRAINT_PASS"
    assert result.evidence_ids[0].startswith("eref_")
    assert len(result.evidence_ids[0]) <= 32
    assert "SECRET_TOKEN" not in report.model_dump_json()
    assert "0.123456789 le" not in report.model_dump_json()


def test_rationale_is_bounded_for_many_long_property_names() -> None:
    long_name = "very_long_property_" + "x" * 5000
    target = TargetSpec(
        goal="bounded rationale",
        constraints=[
            ConstraintSpec(
                property=f"{long_name}_{index}", operator="le", value=1.0, unit="eV"
            )
            for index in range(20)
        ],
    )

    report = ScientificEvaluator(target).evaluate(_candidate(), ScientificState())

    assert len(report.rationale) <= 512
    assert long_name not in report.rationale


@pytest.mark.parametrize(
    ("property_name", "payload_key", "payload_value", "target_unit", "expected"),
    [
        ("band_gap", "band_gap_eV", 0.2, "eV", 0.2),
        (
            "formation_energy",
            "formation_energy_meV_per_atom",
            200.0,
            "eV/atom",
            0.2,
        ),
    ],
)
def test_attested_legacy_explicit_unit_suffixes_are_inferred(
    property_name: str,
    payload_key: str,
    payload_value: float,
    target_unit: str,
    expected: float,
) -> None:
    target = TargetSpec(
        goal="legacy",
        constraints=[
            ConstraintSpec(
                property=property_name, operator="le", value=expected, unit=target_unit
            )
        ],
    )
    evidence = Evidence(
        type="calculation",
        source="host verified legacy adapter",
        content=(
            '{"material":"HgTe","'
            + payload_key
            + '":'
            + str(payload_value)
            + "}"
        ),
        confidence=0.8,
    )

    report = ScientificEvaluator(target).evaluate(
        _candidate(), _attested_scientific(evidence)
    )

    assert report.constraint_results[0].result == "PASS"
    assert report.constraint_results[0].observed_value == expected


@pytest.mark.parametrize("legacy_type", ["candidate_prediction", "model", "literature"])
def test_unattested_legacy_records_cannot_validate(legacy_type: str) -> None:
    evidence = Evidence(
        type=legacy_type,
        source="renamed innocent source",
        content='{"material":"HgTe","band_gap_eV":0.1}',
        confidence=1.0,
    )

    report = ScientificEvaluator(_target()).evaluate(
        _candidate(), _scientific(evidence)
    )

    assert report.constraint_results[0].result == "UNKNOWN"
