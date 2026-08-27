"""ScientificLoopPolicy: scientific convergence, separate from runtime StopPolicy.

``runtime/stop_policy.py`` keeps deciding runtime-level stop (fatal error,
max iterations, provider completion). This policy decides whether the science
has converged -- SUCCESS, CONTINUE, ESCALATE, STALLED, INCONCLUSIVE or
BUDGET_EXHAUSTED -- from evidence, constraints, confidence and budgets.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from photomatagent.scientific.loop.candidate import CandidateState
from photomatagent.scientific.loop.evaluation import EvaluationReport
from photomatagent.scientific.loop.feedback import FeedbackSignal
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
    best_candidate_id: str | None = None
    best_score: float = 0.0
    round: int = 0
    no_progress_rounds: int = 0
    status: str = "RUNNING"

    def add_candidate(
        self, candidate: CandidateState | None, evaluation: EvaluationReport
    ) -> None:
        if candidate is None:
            # Round produced no structured candidate: keep the evaluation for
            # the trajectory, but no candidate enters the ranking.
            self.evaluations.append(evaluation)
            return
        candidate.score = evaluation.score
        candidate.status = (
            "PASS"
            if evaluation.verdict == "PASS"
            else "REVISE"
            if evaluation.verdict == "REVISE"
            else "FAIL"
        )
        self.candidates.append(candidate)
        self.evaluations.append(evaluation)
        if evaluation.score > self.best_score:
            self.best_score = evaluation.score
            self.best_candidate_id = candidate.candidate_id


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


class ScientificLoopPolicy:
    """Deterministic scientific convergence policy."""

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
        if (
            evaluation.verdict == "PASS"
            and not evaluation.critical_evidence_gaps
            and evaluation.confidence >= min_confidence
        ):
            return ScientificLoopDecision(
                action="SUCCESS",
                reason=(
                    f"all hard constraints satisfied (score {evaluation.score:.3f}, "
                    f"confidence {evaluation.confidence:.3f} >= {min_confidence})"
                ),
                best_candidate_id=state.best_candidate_id,
            )
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
            reason=f"verdict={evaluation.verdict}; resolvable violations or evidence gaps remain",
            best_candidate_id=state.best_candidate_id,
        )