from __future__ import annotations

import math

import pytest

from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.loop.candidate import candidate_from_formula
from photomatagent.scientific.loop.evidence_scope import (
    EvidenceRequirements,
    convert_property_value,
    evidence_applicable,
)


def _evidence(**overrides: object) -> ScientificEvidence:
    values: dict[str, object] = {
        "subject": "Na3AgBi4S8",
        "property": "band_gap",
        "value": 1.0,
        "unit": "eV",
        "source": "independent calculation archive",
        "source_type": "dft_calculation",
        "method": "PBE",
        "fidelity": "dft",
    }
    values.update(overrides)
    return ScientificEvidence(**values)


def test_scientific_evidence_scope_fields_have_legacy_defaults() -> None:
    evidence = _evidence()

    assert evidence.assessment_role == "observation"
    assert evidence.candidate_id == ""
    assert evidence.structure_hash == ""
    assert evidence.conditions == {}


@pytest.mark.parametrize("role", ["proposal", "prior"])
def test_non_observation_roles_cannot_validate_constraints(role: str) -> None:
    applicable, reason = evidence_applicable(
        _evidence(assessment_role=role),
        candidate_from_formula("Na0.75Ag0.25BiS2"),
        EvidenceRequirements(),
    )

    assert applicable is False
    assert role in reason


@pytest.mark.parametrize(
    ("overrides", "reason_fragment"),
    [
        ({"source_type": "generative_model", "assessment_role": "observation"}, "generative"),
        ({"fidelity": "ml_generated", "assessment_role": "observation"}, "generated"),
        (
            {
                "source": "candidate_declared",
                "assessment_role": "observation",
                "fidelity": "dft",
            },
            "candidate-declared",
        ),
        (
            {
                "source": "generation",
                "assessment_role": "observation",
                "fidelity": "experimental",
            },
            "generation",
        ),
        ({"source": "synthetic", "assessment_role": "observation"}, "synthetic"),
        ({"source": "mock", "provenance": {"tool": "mock.run_calculation"}}, "mock"),
    ],
)
def test_generation_and_mock_sources_cannot_self_promote_to_observation(
    overrides: dict[str, object], reason_fragment: str
) -> None:
    applicable, reason = evidence_applicable(
        _evidence(**overrides),
        candidate_from_formula("Na0.75Ag0.25BiS2"),
        EvidenceRequirements(),
    )

    assert applicable is False
    assert reason_fragment in reason.lower()


def test_applicability_rejects_role_before_identity_conditions_and_unit() -> None:
    applicable, reason = evidence_applicable(
        _evidence(
            assessment_role="proposal",
            subject="NaBiS2",
            structure_hash="sha256:wrong",
            conditions={},
            unit="",
        ),
        candidate_from_formula(
            "Na3AgBi4S8", extra_representation={"structure_hash": "sha256:target"}
        ),
        EvidenceRequirements(
            scope="device", conditions={"temperature_k": 77}, require_unit=True
        ),
    )

    assert applicable is False
    assert "proposal" in reason


def test_applicability_rejects_identity_before_conditions_and_unit() -> None:
    applicable, reason = evidence_applicable(
        _evidence(subject="NaBiS2", conditions={}, unit=""),
        candidate_from_formula("Na3AgBi4S8"),
        EvidenceRequirements(
            scope="composition", conditions={"temperature_k": 77}, require_unit=True
        ),
    )

    assert applicable is False
    assert "composition" in reason


def test_unknown_legacy_subject_cannot_establish_device_identity() -> None:
    candidate = candidate_from_formula(
        "Na3AgBi4S8", extra_representation={"structure_hash": "sha256:device"}
    )

    applicable, reason = evidence_applicable(
        _evidence(
            subject="legacy sample A",
            structure_hash="sha256:device",
            conditions={"temperature_k": 77},
        ),
        candidate,
        EvidenceRequirements(scope="device", conditions={"temperature_k": 77}),
    )

    assert applicable is False
    assert "legacy" in reason


def test_explicit_candidate_id_must_match() -> None:
    candidate = candidate_from_formula("Na3AgBi4S8")

    applicable, reason = evidence_applicable(
        _evidence(candidate_id="cand_other"), candidate, EvidenceRequirements()
    )

    assert applicable is False
    assert "candidate_id" in reason


def test_decimal_and_integer_formulas_share_composition_identity() -> None:
    applicable, reason = evidence_applicable(
        _evidence(subject="Na3AgBi4S8"),
        candidate_from_formula("Na0.75Ag0.25BiS2"),
        EvidenceRequirements(scope="composition"),
    )

    assert (applicable, reason) == (True, "applicable")


def test_endpoint_literature_does_not_validate_a_different_composition() -> None:
    applicable, reason = evidence_applicable(
        _evidence(subject="NaBiS2", source_type="literature"),
        candidate_from_formula("Na0.75Ag0.25BiS2"),
        EvidenceRequirements(scope="composition"),
    )

    assert applicable is False
    assert "composition" in reason


def test_structure_scope_requires_matching_explicit_structure_hash() -> None:
    candidate = candidate_from_formula(
        "Na3AgBi4S8", extra_representation={"structure_hash": "sha256:target"}
    )

    missing, missing_reason = evidence_applicable(
        _evidence(), candidate, EvidenceRequirements(scope="structure")
    )
    wrong, wrong_reason = evidence_applicable(
        _evidence(structure_hash="sha256:other"),
        candidate,
        EvidenceRequirements(scope="structure"),
    )
    matching, matching_reason = evidence_applicable(
        _evidence(structure_hash="sha256:target"),
        candidate,
        EvidenceRequirements(scope="structure"),
    )

    assert missing is False and "structure" in missing_reason
    assert wrong is False and "structure" in wrong_reason
    assert (matching, matching_reason) == (True, "applicable")


def test_device_scope_requires_structure_and_all_declared_conditions() -> None:
    candidate = candidate_from_formula(
        "Na3AgBi4S8", extra_representation={"structure_hash": "sha256:device"}
    )
    requirements = EvidenceRequirements(
        scope="device",
        conditions={"temperature_k": 77, "bias_v": 0.1, "wavelength_um": 10.0},
    )

    applicable, reason = evidence_applicable(
        _evidence(
            structure_hash="sha256:device",
            conditions={"temperature_k": 77, "bias_v": 0.1},
        ),
        candidate,
        requirements,
    )

    assert applicable is False
    assert "wavelength_um" in reason


def test_allowed_fidelity_and_unit_presence_are_checked_after_identity() -> None:
    candidate = candidate_from_formula("Na3AgBi4S8")

    fidelity_ok, fidelity_reason = evidence_applicable(
        _evidence(fidelity="analytical"),
        candidate,
        EvidenceRequirements(allowed_fidelities=("dft",)),
    )
    unit_ok, unit_reason = evidence_applicable(
        _evidence(unit=""), candidate, EvidenceRequirements(require_unit=True)
    )

    assert fidelity_ok is False and "fidelity" in fidelity_reason
    assert unit_ok is False and "unit" in unit_reason


@pytest.mark.parametrize(
    ("value", "from_unit", "to_unit", "expected"),
    [
        (1000.0, "meV", "eV", 1.0),
        (1.0, "eV", "meV", 1000.0),
        (1000.0, "nm", "um", 1.0),
        (1.0, "um", "nm", 1000.0),
        (1000.0, "meV/atom", "eV/atom", 1.0),
        (1.0, "eV/atom", "meV/atom", 1000.0),
    ],
)
def test_convert_property_value_supports_only_the_designed_pairs(
    value: float, from_unit: str, to_unit: str, expected: float
) -> None:
    assert convert_property_value(value, from_unit, to_unit) == expected


@pytest.mark.parametrize(
    ("from_unit", "to_unit"),
    [("A", "A/cm2"), ("eV", "um"), ("K", "K"), ("", "eV")],
)
def test_convert_property_value_rejects_unknown_or_incompatible_units(
    from_unit: str, to_unit: str
) -> None:
    with pytest.raises(ValueError):
        convert_property_value(1.0, from_unit, to_unit)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_convert_property_value_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError):
        convert_property_value(value, "eV", "meV")
