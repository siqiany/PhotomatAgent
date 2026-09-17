from __future__ import annotations

import copy

from photomatagent.runtime.evidence_attestation import _RuntimeEvidenceAuthority
from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.loop.candidate import candidate_from_formula
from photomatagent.scientific.loop.evaluation import (
    EvidenceEvaluationPolicy,
    EvaluationReport,
    PropertyEvaluation,
    ScientificEvaluator,
)
from photomatagent.scientific.loop.feedback import build_feedback
from photomatagent.scientific.loop.policy import ScientificLoopState
from photomatagent.scientific.loop.observation import stable_observation_identity
from photomatagent.scientific.loop.progress import progress_from_evaluation
from photomatagent.scientific.loop.target import ConstraintSpec, TargetSpec
from photomatagent.scientific.state import EvidenceAttestation, ScientificState


def _target() -> TargetSpec:
    return TargetSpec(
        goal="band gap",
        constraints=[
            ConstraintSpec(property="band_gap", operator="ge", value=0.2, unit="eV")
        ],
    )


def _state(evidence: ScientificEvidence, *, synthetic: bool = False) -> ScientificState:
    state = ScientificState(evidence=[evidence])
    authority = _RuntimeEvidenceAuthority()
    authority.bind(state)
    authority.attest(
        state,
        EvidenceAttestation(
            evidence_id=evidence.id,
            authority="synthetic" if synthetic else "observation",
            origin="synthetic_test" if synthetic else "trusted_builtin",
            tool_name="test.report_property" if synthetic else "electronic.band_summary",
            tool_call_id="call-1",
        ),
    )
    return state


def _evidence(value: float = 0.1, *, evidence_id: str = "raw-a") -> ScientificEvidence:
    return ScientificEvidence(
        id=evidence_id,
        subject="HgTe",
        property="band_gap",
        value=value,
        unit="eV",
        source="measurement",
        source_type="experimental",
        method="spectroscopy",
        fidelity="experimental",
        provenance={"content_sha256": evidence_id},
    )


def test_only_evaluator_manifest_can_create_progress_and_is_not_serialized():
    candidate = candidate_from_formula("HgTe")
    state = _state(_evidence())
    report = ScientificEvaluator(_target()).evaluate(candidate, state)

    assert progress_from_evaluation(candidate, report, state).observation_keys
    restored = EvaluationReport.model_validate_json(report.model_dump_json())
    assert progress_from_evaluation(candidate, restored, state).observation_keys == ()


def test_same_hash_still_binds_canonical_scientific_fields():
    base = _evidence(0.1).model_copy(
        update={"provenance": {"content_sha256": "same-artifact"}}
    )
    tiny = base.model_copy(update={"value": 0.1004})
    repost = base.model_copy(
        update={
            "id": "reposted",
            "provenance": {
                "content_sha256": "same-artifact",
                "request_id": "new-request",
                "reason": "retry",
                "artifact": {"path": "renamed.cif"},
            },
        }
    )
    changed_value = base.model_copy(update={"value": 0.9})
    changed_property = base.model_copy(update={"property": "responsivity", "unit": "A/W"})
    changed_method = base.model_copy(update={"method": "different method"})

    assert stable_observation_identity(base) == stable_observation_identity(tiny)
    assert stable_observation_identity(base) == stable_observation_identity(repost)
    assert stable_observation_identity(base) != stable_observation_identity(changed_value)
    assert stable_observation_identity(base) != stable_observation_identity(changed_property)
    assert stable_observation_identity(base) != stable_observation_identity(changed_method)


def test_forged_public_report_and_untrusted_policy_do_not_create_progress():
    candidate = candidate_from_formula("HgTe")
    evidence = _evidence()
    state = _state(evidence)
    forged = EvaluationReport(
        candidate_id=candidate.candidate_id,
        constraint_results=[
            PropertyEvaluation(
                property="band_gap", result="PASS", evidence_ids=[evidence.id]
            )
        ],
    )
    assert progress_from_evaluation(candidate, forged, state).observation_keys == ()

    synthetic_state = _state(evidence.model_copy(update={"id": "synthetic"}), synthetic=True)
    synthetic_report = ScientificEvaluator(_target()).evaluate(candidate, synthetic_state)
    assert progress_from_evaluation(candidate, synthetic_report, synthetic_state).observation_keys == ()


def test_real_evaluator_rejections_have_no_manifest():
    candidate = candidate_from_formula("HgTe")
    device_target = TargetSpec(
        goal="device responsivity",
        constraints=[
            ConstraintSpec(property="responsivity", operator="ge", value=1.0, unit="A/W")
        ],
        operating_conditions={"temperature_k": 77.0, "spectral_range_um": [8.0, 14.0]},
    )

    missing_structure = ScientificEvidence(
        subject="HgTe", property="responsivity", value=2.0, unit="A/W",
        source="measurement", source_type="experimental", fidelity="experimental",
        conditions={"temperature_k": 77.0, "wavelength_um": 10.0, "bias_v": 0.1,
                    "measurement_definition": "current sweep"},
    )
    state = _state(missing_structure)
    report = ScientificEvaluator(device_target).evaluate(candidate, state)
    assert report.constraint_results[0].result == "UNKNOWN"
    assert report.accepted_evidence_manifest == ()

    wrong_temperature = missing_structure.model_copy(
        update={"id": "wrong-temperature", "structure_hash": "device-hash",
                "conditions": {"temperature_k": 300.0, "wavelength_um": 10.0,
                                "bias_v": 0.1, "measurement_definition": "current sweep"}}
    )
    structured = candidate.model_copy(update={"representation": {"structure_hash": "device-hash"}})
    wrong_state = _state(wrong_temperature)
    wrong_report = ScientificEvaluator(device_target).evaluate(structured, wrong_state)
    assert wrong_report.constraint_results[0].result == "UNKNOWN"
    assert wrong_report.accepted_evidence_manifest == ()

    prior = missing_structure.model_copy(update={"id": "prior", "assessment_role": "prior"})
    prior_report = ScientificEvaluator(device_target).evaluate(candidate, _state(prior))
    assert prior_report.constraint_results[0].result == "UNKNOWN"
    assert prior_report.accepted_evidence_manifest == ()

    custom_policy_report = ScientificEvaluator(
        _target(), policy=EvidenceEvaluationPolicy(trusted_attestation_tools=frozenset())
    ).evaluate(candidate, _state(_evidence()))
    assert custom_policy_report.constraint_results[0].result == "UNKNOWN"
    assert custom_policy_report.accepted_evidence_manifest == ()


def test_report_and_state_copies_clear_runtime_authority():
    candidate = candidate_from_formula("HgTe")
    state = _state(_evidence())
    report = ScientificEvaluator(_target()).evaluate(candidate, state)
    assert progress_from_evaluation(candidate, report, state).observation_keys

    for copied_report in (
        report.model_copy(),
        report.model_copy(deep=True),
        copy.copy(report),
        copy.deepcopy(report),
    ):
        assert copied_report.accepted_evidence_manifest == ()
        assert progress_from_evaluation(candidate, copied_report, state).observation_keys == ()

    for copied_state in (state.model_copy(), state.model_copy(deep=True), copy.copy(state), copy.deepcopy(state)):
        assert copied_state.verified_attestation(state.evidence[0].id) is None
        assert progress_from_evaluation(candidate, report, copied_state).observation_keys == ()

    loop_state = ScientificLoopState(target=_target())
    loop_state.add_candidate(candidate, report)
    assert loop_state.historical_candidate_evaluations
    for copied_loop_state in (
        loop_state.model_copy(),
        loop_state.model_copy(deep=True),
        copy.copy(loop_state),
        copy.deepcopy(loop_state),
    ):
        assert copied_loop_state.historical_candidate_evaluations == ()


def test_mutated_or_replaced_current_evidence_cannot_reuse_manifest():
    candidate = candidate_from_formula("HgTe")
    evidence = _evidence()
    state = _state(evidence)
    report = ScientificEvaluator(_target()).evaluate(candidate, state)

    evidence.provenance["content_sha256"] = "changed-content"
    assert progress_from_evaluation(candidate, report, state).observation_keys == ()

    duplicate = evidence.model_copy(update={"id": evidence.id})
    state.evidence.append(duplicate)
    assert state.verified_attestation(evidence.id) is None
    assert progress_from_evaluation(candidate, report, state).observation_keys == ()


def test_same_candidate_id_is_continuation_but_different_id_requires_new_observation():
    candidate = candidate_from_formula("HgTe")
    state = _state(_evidence())
    evaluator = ScientificEvaluator(_target())
    prior = evaluator.evaluate(candidate, state)
    current = evaluator.evaluate(candidate.model_copy(deep=True), state)
    continuation = build_feedback(
        _target(),
        candidate,
        current,
        [],
        scientific=state,
        prior_evaluations=[(candidate.model_copy(update={"evidence_ids": []}), prior)],
    )
    assert continuation is not None
    assert continuation.decision != "REJECT"

    prior_candidate = candidate.model_copy(update={"candidate_id": "prior-candidate", "evidence_ids": []})
    prior_report = evaluator.evaluate(prior_candidate, state)
    duplicate = build_feedback(
        _target(),
        candidate,
        current,
        [],
        scientific=state,
        prior_evaluations=[(prior_candidate, prior_report)],
    )
    assert duplicate is not None
    assert duplicate.decision == "REJECT"
