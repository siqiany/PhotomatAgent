from __future__ import annotations

from photomatagent.scientific.loop.candidate import candidate_from_formula
from photomatagent.scientific.loop.evaluation import EvaluationReport
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