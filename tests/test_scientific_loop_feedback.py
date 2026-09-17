from __future__ import annotations

from photomatagent.runtime.evidence_attestation import _RuntimeEvidenceAuthority
from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.loop.candidate import candidate_from_formula
from photomatagent.scientific.loop.evaluation import (
    EvidenceEvaluationPolicy,
    ScientificEvaluator,
)
from photomatagent.scientific.loop.feedback import (
    build_feedback,
    format_feedback_for_model,
)
from photomatagent.scientific.loop.target import (
    ConstraintSpec,
    TargetSpec,
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


def _evaluate(candidate, *evidence) -> object:
    evaluator = ScientificEvaluator(
        _target(), policy=EvidenceEvaluationPolicy(allow_synthetic_evidence=True)
    )
    candidate.representation.setdefault("structure_hash", "synthetic:device")
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
        method="synthetic method",
        fidelity=fidelity,
        structure_hash="synthetic:device",
        conditions={"temperature_k": 77},
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
            method="synthetic device measurement",
            fidelity="experimental",
            structure_hash="synthetic:device",
            conditions={
                "wavelength_um": 10.0,
                "bias_v": 0.1,
                "temperature_k": 77,
                "measurement_definition": "synthetic calibrated response",
            },
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


def test_same_composition_with_forged_evidence_id_is_rejected():
    first = candidate_from_formula("HgTe")
    second = candidate_from_formula("HgTe")
    second.evidence_ids = ["new-evidence-id"]
    report = _evaluate(second, _gap(0.14))
    signal = build_feedback(_target(), second, report, [first])
    assert signal is not None
    assert signal.decision == "REJECT"


def test_same_composition_with_new_attested_accepted_observation_is_not_rejected():
    first = candidate_from_formula("HgTe")
    second = candidate_from_formula("HgTe")
    old = _gap(0.14).model_copy(
        update={"id": "old-evidence", "provenance": {"content_sha256": "old"}}
    )
    new = _gap(0.13, fidelity="experimental").model_copy(
        update={
            "id": "new-evidence",
            "source_type": "experimental",
            "provenance": {"content_sha256": "new"},
        }
    )
    first.evidence_ids = [old.id]
    second.evidence_ids = [new.id]
    state = ScientificState(evidence=[old, new])
    authority = _RuntimeEvidenceAuthority()
    authority.bind(state)
    for item in state.evidence:
        authority.attest(
            state,
            EvidenceAttestation(
                evidence_id=item.id,
                authority="observation",
                origin="trusted_builtin",
                tool_name="electronic.band_summary",
                tool_call_id=f"call-{item.id}",
            ),
        )
    evaluator = ScientificEvaluator(
        _target(), policy=EvidenceEvaluationPolicy(allow_synthetic_evidence=True)
    )
    second.representation["structure_hash"] = "synthetic:device"
    report = evaluator.evaluate(second, state)
    prior = evaluator.evaluate(first, ScientificState(evidence=[old]))
    signal = build_feedback(
        _target(), second, report, [first], scientific=state,
        prior_evaluations=[(first, prior)],
    )
    assert signal is not None
    assert signal.decision != "REJECT"


def test_same_content_hash_with_new_id_is_still_rejected():
    first = candidate_from_formula("HgTe")
    second = candidate_from_formula("HgTe")
    old = _gap(0.14).model_copy(
        update={"id": "old-evidence", "provenance": {"content_sha256": "same"}}
    )
    new = _gap(0.14).model_copy(
        update={"id": "new-evidence", "provenance": {"content_sha256": "same"}}
    )
    first.evidence_ids = [old.id]
    second.evidence_ids = [new.id]
    state = ScientificState(evidence=[old, new])
    authority = _RuntimeEvidenceAuthority()
    authority.bind(state)
    for item in state.evidence:
        authority.attest(
            state,
            EvidenceAttestation(
                evidence_id=item.id,
                authority="observation",
                origin="trusted_builtin",
                tool_name="electronic.band_summary",
                tool_call_id=f"call-{item.id}",
            ),
        )
    report = _evaluate(second, new)
    prior = _evaluate(first, old)
    signal = build_feedback(
        _target(), second, report, [first], scientific=state,
        prior_evaluations=[(first, prior)],
    )
    assert signal is not None
    assert signal.decision == "REJECT"


def test_inconclusive_feedback_says_not_yet_verified():
    candidate = candidate_from_formula("HgTe")
    report = _evaluate(candidate)
    signal = build_feedback(_target(), candidate, report, [])
    assert signal is not None
    assert "not yet" in signal.summary.lower()


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
    evaluator = ScientificEvaluator(
        _target(), policy=EvidenceEvaluationPolicy(allow_synthetic_evidence=True)
    )
    candidate.representation["structure_hash"] = "synthetic:device"
    state = ScientificState()
    state.add_evidence(_gap(0.14, fidelity="dft"))
    state.add_evidence(
        ScientificEvidence(
            subject="HgTe",
            property="band_gap",
            value=0.4,
            unit="eV",
            source="synthetic:other",
            source_type="experimental",
            method="synthetic method",
            fidelity="experimental",
            structure_hash="synthetic:device",
            conditions={"temperature_k": 77},
        )
    )
    report = evaluator.evaluate(candidate, state)
    signal = build_feedback(_target(), candidate, report, [])
    assert signal is not None
    assert any(a.action_type == "VALIDATE" for a in signal.recommended_actions)
