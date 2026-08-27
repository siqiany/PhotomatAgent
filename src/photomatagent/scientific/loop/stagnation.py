"""StagnationDetector: prove that the loop is actually converging.

Three independent no-progress signals are tracked:
  * best-score improvement below ``epsilon`` for ``patience`` rounds;
  * the candidate fingerprint (identical proposals never count as progress);
  * the violation / evidence-gap signatures (same unsolved problem, same
    missing proof).

``HgTe, HgTe, HgTe, HgTe`` is one iteration, not four.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from photomatagent.scientific.loop.candidate import CandidateState
from photomatagent.scientific.loop.evaluation import EvaluationReport


def violation_signature(report: EvaluationReport) -> tuple[str, ...]:
    """Sorted identity of the currently failing properties."""
    return tuple(
        sorted(
            f"{v.property}:{round(float(v.observed_value), 4) if v.observed_value is not None else '?'}"
            for v in report.violations
        )
    )


def gap_signature(report: EvaluationReport) -> tuple[str, ...]:
    """Sorted identity of the currently missing evidence."""
    return tuple(sorted(report.critical_evidence_gaps))


@dataclass
class StagnationDetector:
    """Tracks a bounded history of loop progress signals."""

    patience: int = 3
    epsilon: float = 1e-3

    _best_score: float = 0.0
    _no_progress_rounds: int = 0
    _fingerprints: set[str] = field(default_factory=set)
    _last_violations: tuple[str, ...] = ()
    _last_gaps: tuple[str, ...] = ()
    _repeated_candidate_ids: list[str] = field(default_factory=list)

    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def no_progress_rounds(self) -> int:
        return self._no_progress_rounds

    @property
    def stalled(self) -> bool:
        return self._no_progress_rounds >= self.patience

    @property
    def repeated_candidate_ids(self) -> list[str]:
        return list(self._repeated_candidate_ids)

    def reset(self) -> None:
        self._best_score = 0.0
        self._no_progress_rounds = 0
        self._fingerprints = set()
        self._last_violations = ()
        self._last_gaps = ()
        self._repeated_candidate_ids = []

    def is_duplicate(self, candidate: CandidateState) -> bool:
        """True when this exact candidate (fingerprint) was already proposed."""
        return candidate.fingerprint in self._fingerprints

    def record(
        self,
        candidate: CandidateState,
        report: EvaluationReport,
    ) -> None:
        """Feed one evaluated candidate; updates the no-progress counter.

        A round counts as progress only when the best score improves by at
        least ``epsilon``. Identical fingerprints, or identical violation and
        evidence-gap signatures with no score improvement, never count.
        """
        fingerprint = candidate.fingerprint
        if fingerprint in self._fingerprints:
            self._repeated_candidate_ids.append(candidate.candidate_id)
        else:
            self._fingerprints.add(fingerprint)

        score = float(report.score or 0.0)
        if score - self._best_score >= self.epsilon:
            self._best_score = score
            self._no_progress_rounds = 0
        else:
            self._no_progress_rounds += 1
        self._last_violations = violation_signature(report)
        self._last_gaps = gap_signature(report)