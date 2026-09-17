from __future__ import annotations

from photomatagent.scientific.loop.candidate import candidate_from_formula
from photomatagent.scientific.loop.evaluation import EvaluationReport
from photomatagent.scientific.loop.progress import (
    ValidationProgress,
    progress_from_evaluation,
)
from photomatagent.runtime.evidence_attestation import _RuntimeEvidenceAuthority
from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.loop.evaluation import PropertyEvaluation
from photomatagent.scientific.state import EvidenceAttestation, ScientificState
from photomatagent.scientific.loop.stagnation import (
    StagnationDetector,
    gap_signature,
    violation_signature,
)
from photomatagent.scientific.loop.target import ConstraintViolation


def _report(score: float = 0.5, violations: list[str] | None = None) -> EvaluationReport:
    return EvaluationReport(
        candidate_id="cand_x",
        score=score,
        verdict="FAIL" if violations else "INCONCLUSIVE",
        violations=[
            ConstraintViolation(
                property=name,
                observed_value=0.3,
                target_value=0.155,
                unit="eV",
                severity="HARD",
                message=f"{name} violated",
            )
            for name in (violations or [])
        ],
        critical_evidence_gaps=[],
    )


def test_identical_fingerprint_never_counts_as_progress():
    detector = StagnationDetector(patience=3)
    candidate = candidate_from_formula("HgTe")
    report = _report(score=0.5)
    for _ in range(4):
        detector.record(candidate, report)
    assert detector.stalled
    assert detector.repeated_candidate_ids == [candidate.candidate_id] * 3


def test_score_improvement_resets_stagnation():
    detector = StagnationDetector(patience=3)
    candidate = candidate_from_formula("HgTe")
    detector.record(candidate, _report(score=0.5))
    detector.record(candidate, _report(score=0.5))
    assert not detector.stalled
    assert detector.no_progress_rounds == 1
    detector.record(candidate, _report(score=0.9))  # improvement
    assert detector.no_progress_rounds == 0
    detector.record(candidate, _report(score=0.9))
    detector.record(candidate, _report(score=0.9))
    detector.record(candidate, _report(score=0.9))
    assert detector.stalled


def test_below_epsilon_improvement_does_not_reset():
    detector = StagnationDetector(patience=2, epsilon=0.1)
    candidate = candidate_from_formula("HgTe")
    detector.record(candidate, _report(score=0.5))
    detector.record(candidate, _report(score=0.55))  # +0.05 < epsilon 0.1
    assert detector.no_progress_rounds == 1
    detector.record(candidate, _report(score=0.551))
    assert detector.stalled


def test_distinct_improving_candidates_reset_stagnation():
    detector = StagnationDetector(patience=3)
    detector.record(candidate_from_formula("HgTe"), _report(score=0.4))
    detector.record(candidate_from_formula("HgCdTe"), _report(score=0.6))
    detector.record(candidate_from_formula("PbSnTe"), _report(score=0.8))
    assert not detector.stalled
    assert detector.no_progress_rounds == 0


def test_signatures_are_deterministic():
    a = _report(score=0.5, violations=["band_gap", "responsivity"])
    b = _report(score=0.5, violations=["responsivity", "band_gap"])
    expected = ("band_gap:0.3", "responsivity:0.3")
    assert violation_signature(a) == violation_signature(b) == expected
    assert gap_signature(a) == ()
    assert gap_signature(_report(score=0.3)) == ()


def test_is_duplicate_flags_repeats():
    detector = StagnationDetector()
    first = candidate_from_formula("HgTe")
    second = candidate_from_formula("TeHg")
    assert not detector.is_duplicate(first)
    detector.record(first, _report())
    assert detector.is_duplicate(second)


def test_new_resolved_question_counts_without_score_gain():
    detector = StagnationDetector(patience=2)
    candidate = candidate_from_formula("NaBiS2")
    report = EvaluationReport(candidate_id=candidate.candidate_id, score=0)
    detector.record(candidate, report)
    detector.record(
        candidate,
        report,
        progress=ValidationProgress(
            candidate_id=candidate.candidate_id,
            resolved_questions=("band_gap",),
            observation_keys=("artifact:abc",),
        ),
    )
    assert detector.no_progress_rounds == 0


def test_same_observation_with_new_evidence_id_does_not_count():
    detector = StagnationDetector(patience=2)
    candidate = candidate_from_formula("NaBiS2")
    report = EvaluationReport(candidate_id=candidate.candidate_id, score=0)
    first = ValidationProgress(
        candidate_id=candidate.candidate_id,
        resolved_questions=("band_gap",),
        observation_keys=("content:stable",),
    )
    detector.record(candidate, report, progress=first)
    detector.record(
        candidate,
        report,
        progress=ValidationProgress(
            candidate_id=candidate.candidate_id,
            resolved_questions=("band_gap",),
            observation_keys=("content:stable",),
        ),
    )
    assert detector.no_progress_rounds == 1


def test_accepted_failure_is_validation_progress():
    candidate = candidate_from_formula("NaBiS2")
    evidence = ScientificEvidence(
        id="evidence-old-id",
        subject="NaBiS2",
        property="band_gap",
        value=0.8,
        unit="eV",
        source="dft",
        source_type="dft_calculation",
        fidelity="dft",
        provenance={"content_sha256": "stable-content"},
    )
    report = EvaluationReport(
        candidate_id=candidate.candidate_id,
        verdict="FAIL",
        constraint_results=[
            PropertyEvaluation(
                property="band_gap",
                result="FAIL",
                evidence_ids=["evidence-old-id"],
            )
        ],
    )
    state = ScientificState(evidence=[evidence])
    authority = _RuntimeEvidenceAuthority()
    authority.bind(state)
    authority.attest(
        state,
        EvidenceAttestation(
            evidence_id=evidence.id,
            authority="observation",
            origin="trusted_builtin",
            tool_name="electronic.band_summary",
            tool_call_id="call-failure",
        ),
    )
    progress = progress_from_evaluation(candidate, report, state)
    assert progress.resolved_questions == ("band_gap",)
    assert progress.observation_keys


def test_accepted_unknown_is_progress_but_unresolved_gap_is_not():
    candidate = candidate_from_formula("NaBiS2")
    evidence = ScientificEvidence(
        id="evidence-id",
        subject="NaBiS2",
        property="band_gap",
        value="unparseable",
        unit="eV",
        source="dft",
        source_type="dft_calculation",
        fidelity="dft",
        provenance={"artifact_sha256": "artifact-content"},
    )
    accepted = EvaluationReport(
        candidate_id=candidate.candidate_id,
        constraint_results=[
            PropertyEvaluation(
                property="band_gap",
                result="UNKNOWN",
                evidence_ids=["evidence-id"],
            )
        ],
    )
    empty = EvaluationReport(
        candidate_id=candidate.candidate_id,
        constraint_results=[PropertyEvaluation(property="band_gap", result="UNKNOWN")],
    )
    state = ScientificState(evidence=[evidence])
    authority = _RuntimeEvidenceAuthority()
    authority.bind(state)
    authority.attest(
        state,
        EvidenceAttestation(
            evidence_id=evidence.id,
            authority="observation",
            origin="trusted_builtin",
            tool_name="electronic.band_summary",
            tool_call_id="call-unknown",
        ),
    )
    assert progress_from_evaluation(candidate, accepted, state).resolved_questions == (
        "band_gap",
    )
    assert progress_from_evaluation(candidate, empty, state).resolved_questions == ()


def test_content_hash_survives_id_and_cif_filename_changes():
    candidate = candidate_from_formula("NaBiS2")
    first = ScientificEvidence(
        id="raw-id-a",
        subject="NaBiS2",
        property="band_gap",
        value=0.1000001,
        unit="eV",
        source="dft",
        source_type="dft_calculation",
        fidelity="dft",
        provenance={
            "content_sha256": "same-bytes",
            "request_id": "request-a",
            "artifact": {"path": "first-name.cif"},
        },
    )
    second = first.model_copy(
        update={
            "id": "raw-id-b",
            "value": 0.1000002,
            "provenance": {
                "content_sha256": "same-bytes",
                "request_id": "request-b",
                "artifact": {"path": "renamed.cif"},
            },
        }
    )
    report_a = EvaluationReport(
        candidate_id=candidate.candidate_id,
        constraint_results=[
            PropertyEvaluation(property="band_gap", result="PASS", evidence_ids=[first.id])
        ],
    )
    report_b = report_a.model_copy(
        update={
            "constraint_results": [
                PropertyEvaluation(
                    property="band_gap", result="PASS", evidence_ids=[second.id]
                )
            ]
        }
    )
    def attested(item: ScientificEvidence) -> ScientificState:
        state = ScientificState(evidence=[item])
        authority = _RuntimeEvidenceAuthority()
        authority.bind(state)
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
        return state

    progress_a = progress_from_evaluation(candidate, report_a, attested(first))
    progress_b = progress_from_evaluation(candidate, report_b, attested(second))
    assert progress_a.observation_keys == progress_b.observation_keys


def test_progress_ignores_report_for_another_candidate():
    candidate = candidate_from_formula("NaBiS2")
    evidence = ScientificEvidence(
        id="evidence-id",
        subject="NaBiS2",
        property="band_gap",
        value=0.1,
        unit="eV",
        source="dft",
        source_type="dft_calculation",
        fidelity="dft",
    )
    report = EvaluationReport(
        candidate_id="different-candidate",
        constraint_results=[
            PropertyEvaluation(property="band_gap", result="PASS", evidence_ids=[evidence.id])
        ],
    )
    progress = progress_from_evaluation(candidate, report, ScientificState(evidence=[evidence]))
    assert progress.observation_keys == ()


def test_progress_requires_nonempty_exact_report_candidate_id():
    candidate = candidate_from_formula("NaBiS2")
    evidence = ScientificEvidence(
        id="evidence-id",
        subject="NaBiS2",
        property="band_gap",
        value=0.1,
        unit="eV",
        source="dft",
        source_type="dft_calculation",
        fidelity="dft",
    )
    state = ScientificState(evidence=[evidence])
    authority = _RuntimeEvidenceAuthority()
    authority.bind(state)
    authority.attest(
        state,
        EvidenceAttestation(
            evidence_id=evidence.id,
            authority="observation",
            origin="trusted_builtin",
            tool_name="electronic.band_summary",
            tool_call_id="call-candidate-id",
        ),
    )
    report = EvaluationReport(
        candidate_id="",
        constraint_results=[
            PropertyEvaluation(property="band_gap", result="PASS", evidence_ids=[evidence.id])
        ],
    )
    assert progress_from_evaluation(candidate, report, state).observation_keys == ()


def test_progress_rejects_unattested_wrong_property_and_subject():
    candidate = candidate_from_formula("NaBiS2")
    wrong_property = ScientificEvidence(
        id="wrong-property",
        subject="NaBiS2",
        property="responsivity",
        value=1.1,
        unit="A/W",
        source="dft",
        source_type="dft_calculation",
        fidelity="dft",
    )
    wrong_subject = ScientificEvidence(
        id="wrong-subject",
        subject="HgTe",
        property="band_gap",
        value=0.1,
        unit="eV",
        source="dft",
        source_type="dft_calculation",
        fidelity="dft",
    )
    state = ScientificState(evidence=[wrong_property, wrong_subject])
    authority = _RuntimeEvidenceAuthority()
    authority.bind(state)
    authority.attest(
        state,
        EvidenceAttestation(
            evidence_id=wrong_property.id,
            authority="background",
            origin="untrusted_tool",
            tool_name="untrusted",
            tool_call_id="call-background",
        ),
    )
    authority.attest(
        state,
        EvidenceAttestation(
            evidence_id=wrong_subject.id,
            authority="observation",
            origin="trusted_builtin",
            tool_name="electronic.band_summary",
            tool_call_id="call-wrong-subject",
        ),
    )
    report = EvaluationReport(
        candidate_id=candidate.candidate_id,
        constraint_results=[
            PropertyEvaluation(
                property="band_gap", result="PASS", evidence_ids=[wrong_property.id]
            ),
            PropertyEvaluation(
                property="band_gap", result="PASS", evidence_ids=[wrong_subject.id]
            ),
        ],
    )
    assert progress_from_evaluation(candidate, report, state).observation_keys == ()


def test_synthetic_attestation_requires_evaluator_policy_opt_in():
    candidate = candidate_from_formula("NaBiS2")
    evidence = ScientificEvidence(
        id="synthetic-evidence",
        subject="NaBiS2",
        property="band_gap",
        value=0.1,
        unit="eV",
        source="test-only provider",
        source_type="dft_calculation",
        fidelity="dft",
    )
    state = ScientificState(evidence=[evidence])
    authority = _RuntimeEvidenceAuthority()
    authority.bind(state)
    authority.attest(
        state,
        EvidenceAttestation(
            evidence_id=evidence.id,
            authority="synthetic",
            origin="synthetic_test",
            tool_name="test.report_property",
            tool_call_id="call-synthetic",
        ),
    )
    report = EvaluationReport(
        candidate_id=candidate.candidate_id,
        constraint_results=[
            PropertyEvaluation(property="band_gap", result="PASS", evidence_ids=[evidence.id])
        ],
    )
    assert progress_from_evaluation(candidate, report, state).observation_keys == ()
    assert progress_from_evaluation(
        candidate, report, state, allow_synthetic_evidence=True
    ).observation_keys
