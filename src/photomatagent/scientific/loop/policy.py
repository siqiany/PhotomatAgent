"""ScientificLoopPolicy: scientific convergence, separate from runtime StopPolicy.

``runtime/stop_policy.py`` keeps deciding runtime-level stop (fatal error,
max iterations, provider completion). This policy decides whether the science
has converged -- SUCCESS, CONTINUE, ESCALATE, STALLED, INCONCLUSIVE or
BUDGET_EXHAUSTED -- from evidence, constraints, confidence and budgets.

The optional LLM Scientific Judge is strictly advisory here: it can only hold
back SUCCESS when it raises concerns about an otherwise deterministic pass
(``judge_min_quality`` / ``require_judge``). It can never turn a deterministic
FAIL or UNKNOWN into a PASS, and it never rescinds a hard-constraint
violation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, PrivateAttr

from photomatagent.scientific.loop.candidate import CandidateState
from photomatagent.scientific.loop.evaluation import EvaluationReport
from photomatagent.scientific.loop.feedback import FeedbackSignal
from photomatagent.scientific.loop.judge import JudgeReport
from photomatagent.scientific.loop.stagnation import StagnationDetector
from photomatagent.scientific.loop.target import (
    ConstraintViolation,
    TargetSpec,
)

LoopAction = Literal[
    "SUCCESS",
    "CONTINUE",
    "ESCALATE",
    "STALLED",
    "INCONCLUSIVE",
    "BUDGET_EXHAUSTED",
]


class ScientificLoopDecision(BaseModel):
    action: LoopAction
    reason: str
    best_candidate_id: str | None = None


class ScientificLoopState(BaseModel):
    """Where the optimization/investigation loop currently is.

    Kept separate from ConversationState and ScientificState on purpose:
    ScientificState is *what we know scientifically*; this is *where the loop
    is* in its search over candidates.
    """

    target: TargetSpec
    candidates: list[CandidateState] = Field(default_factory=list)
    evaluations: list[EvaluationReport] = Field(default_factory=list)
    feedback_history: list[FeedbackSignal] = Field(default_factory=list)
    pending_candidate_ids: list[str] = Field(default_factory=list)
    active_candidate_id: str | None = None
    best_candidate_id: str | None = None
    best_score: float = 0.0
    round: int = 0
    no_progress_rounds: int = 0
    status: str = "RUNNING"
    _candidate_evaluation_history: list[tuple[CandidateState, EvaluationReport]] = PrivateAttr(
        default_factory=list
    )

    @property
    def historical_candidate_evaluations(
        self,
    ) -> tuple[tuple[CandidateState, EvaluationReport], ...]:
        """Runtime-only candidate/evaluation pairs in chronological order.

        The public state snapshot intentionally remains serializable without
        raw evaluator provenance.  Restoring an old snapshot therefore
        yields an empty history and duplicate suppression fails closed.
        """

        return tuple(self._candidate_evaluation_history)

    def register_candidates(self, candidates: list[CandidateState]) -> None:
        """Upsert current projections and queue newly discovered identities."""

        indexes = {
            candidate.candidate_id: index
            for index, candidate in enumerate(self.candidates)
        }
        for candidate in candidates:
            index = indexes.get(candidate.candidate_id)
            if index is None:
                indexes[candidate.candidate_id] = len(self.candidates)
                self.candidates.append(candidate)
                if (
                    candidate.candidate_id != self.active_candidate_id
                    and candidate.candidate_id not in self.pending_candidate_ids
                ):
                    self.pending_candidate_ids.append(candidate.candidate_id)
                continue
            previous = self.candidates[index]
            candidate.score = previous.score
            candidate.status = previous.status
            candidate.rejection_reasons = list(previous.rejection_reasons)
            self.candidates[index] = candidate

    def candidate(self, candidate_id: str | None) -> CandidateState | None:
        if candidate_id is None:
            return None
        return next(
            (
                candidate
                for candidate in self.candidates
                if candidate.candidate_id == candidate_id
            ),
            None,
        )

    def add_candidate(
        self, candidate: CandidateState | None, evaluation: EvaluationReport
    ) -> None:
        if candidate is None:
            # Round produced no structured candidate: keep the evaluation for
            # the trajectory, but no candidate enters the ranking.
            self.evaluations.append(evaluation)
            return
        candidate.score = evaluation.score
        status_by_verdict = {
            "PASS": "PASS",
            "FAIL": "FAIL",
            "REVISE": "REVISE",
            "INCONCLUSIVE": "INCONCLUSIVE",
        }
        candidate.status = status_by_verdict[evaluation.verdict]  # type: ignore[assignment]
        existing_index = next(
            (
                index
                for index, existing in enumerate(self.candidates)
                if existing.candidate_id == candidate.candidate_id
            ),
            None,
        )
        if existing_index is None:
            self.candidates.append(candidate)
        else:
            self.candidates[existing_index] = candidate
        self.evaluations.append(evaluation)
        self._candidate_evaluation_history.append(
            (candidate.model_copy(deep=True), evaluation)
        )
        self._recompute_best_candidate()

    def _recompute_best_candidate(self) -> None:
        latest_scores = {
            evaluation.candidate_id: evaluation.score
            for evaluation in self.evaluations
            if evaluation.candidate_id
        }
        ranked = [
            candidate
            for candidate in self.candidates
            if candidate.candidate_id in latest_scores
        ]
        if not ranked:
            self.best_candidate_id = None
            self.best_score = 0.0
            return
        best = max(ranked, key=lambda item: latest_scores[item.candidate_id])
        self.best_candidate_id = best.candidate_id
        self.best_score = latest_scores[best.candidate_id]


class ScientificLoopSummary(BaseModel):
    """Structured loop outcome shared by CLI, experiments and event logs."""

    status: Literal["SUCCESS", "STALLED", "INCONCLUSIVE", "BUDGET_EXHAUSTED"]
    rounds: int
    candidate_count: int
    best_candidate_id: str | None
    best_score: float
    final_evaluation: EvaluationReport | None
    unresolved_violations: list[ConstraintViolation] = Field(default_factory=list)
    unresolved_evidence_gaps: list[str] = Field(default_factory=list)
    termination_reason: str = ""
    judge_report: JudgeReport | None = None
    projection_diagnostics: list[str] = Field(default_factory=list)


class ScientificLoopPolicy:
    """Deterministic scientific convergence policy.

    ``judge_min_quality`` and ``require_judge`` only gate the SUCCESS path:
    they never create a SUCCESS and never change a FAIL/UNKNOWN verdict.
    """

    def __init__(
        self,
        *,
        judge_min_quality: float = 0.6,
        require_judge: bool = False,
    ) -> None:
        self.judge_min_quality = judge_min_quality
        self.require_judge = require_judge

    def decide(
        self,
        *,
        evaluation: EvaluationReport | None,
        state: ScientificLoopState,
        stagnation: StagnationDetector,
        max_rounds: int,
        max_candidates: int,
        min_confidence: float,
        escalate_requested: bool = False,
        inconclusive_reason: str | None = None,
        judge: JudgeReport | None = None,
    ) -> ScientificLoopDecision:
        if state.round > max_rounds or len(state.candidates) > max_candidates:
            return ScientificLoopDecision(
                action="BUDGET_EXHAUSTED",
                reason=(
                    f"max rounds {max_rounds} / max candidates "
                    f"{max_candidates} exceeded"
                ),
                best_candidate_id=state.best_candidate_id,
            )
        if state.pending_candidate_ids and evaluation is not None:
            return ScientificLoopDecision(
                action="CONTINUE",
                reason="registered candidates remain pending evaluation",
                best_candidate_id=state.best_candidate_id,
            )
        if stagnation.stalled:
            return ScientificLoopDecision(
                action="STALLED",
                reason=(
                    f"no score improvement for {stagnation.patience} consecutive "
                    "rounds (same candidates or unsolved signatures)"
                ),
                best_candidate_id=state.best_candidate_id,
            )
        if evaluation is None:
            return ScientificLoopDecision(
                action="INCONCLUSIVE",
                reason="no candidate could be constructed from structured state",
                best_candidate_id=state.best_candidate_id,
            )
        deterministic_ok = (
            evaluation.verdict == "PASS"
            and not evaluation.critical_evidence_gaps
            and evaluation.confidence >= min_confidence
        )
        if not deterministic_ok:
            # The judge is advisory; regardless of what it says, deterministic
            # FAIL/UNKNOWN/low confidence can never become SUCCESS.
            if escalate_requested:
                return ScientificLoopDecision(
                    action="ESCALATE",
                    reason="key constraints need higher-fidelity evidence",
                    best_candidate_id=state.best_candidate_id,
                )
            if inconclusive_reason:
                return ScientificLoopDecision(
                    action="INCONCLUSIVE",
                    reason=inconclusive_reason,
                    best_candidate_id=state.best_candidate_id,
                )
            return ScientificLoopDecision(
                action="CONTINUE",
                reason=(
                    f"verdict={evaluation.verdict}; resolvable violations or "
                    "evidence gaps remain"
                ),
                best_candidate_id=state.best_candidate_id,
            )
        judge_ok = self._judge_ok(judge)
        if judge_ok:
            return ScientificLoopDecision(
                action="SUCCESS",
                reason=(
                    f"all hard constraints satisfied (score {evaluation.score:.3f}, "
                    f"confidence {evaluation.confidence:.3f} >= {min_confidence})"
                )
                + self._judge_reason_suffix(judge),
                best_candidate_id=state.best_candidate_id,
            )
        return ScientificLoopDecision(
            action="CONTINUE",
            reason=(
                "deterministic constraints pass, but the scientific judge "
                f"raised concerns: {judge.summary_line() if judge else 'judge required but unavailable'}"
            ),
            best_candidate_id=state.best_candidate_id,
        )

    def _judge_ok(self, judge: JudgeReport | None) -> bool:
        if judge is None or judge.status == "UNAVAILABLE":
            # Missing or failed judge: SUCCESS is blocked only when a judge is
            # strictly required; otherwise the judge stays advisory.
            return not self.require_judge
        return judge.scientific_quality >= self.judge_min_quality

    def _judge_reason_suffix(self, judge: JudgeReport | None) -> str:
        if judge is None:
            return ""
        return f"; judge quality {judge.scientific_quality:.2f}"
