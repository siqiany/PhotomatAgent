"""Evidence-Guided Scientific Feedback Loop.

The outer scientific loop (`ScientificLoopController`) wraps the existing
AgentRuntime (the Maker) with a deterministic Checker
(`ScientificEvaluator`), structured feedback (`FeedbackSignal`), a
convergence policy (`ScientificLoopPolicy`) and stagnation detection. It
exists to enforce one architectural invariant: a model "final answer" is not
scientific success -- success is decided by evidence, constraints and the
independent evaluator.
"""

from __future__ import annotations

from photomatagent.scientific.loop.candidate import (
    CandidateState,
    candidate_fingerprint,
    extract_candidate_from_state,
)
from photomatagent.scientific.loop.controller import (
    ScientificLoopConfig,
    ScientificLoopController,
    format_loop_state_snapshot,
)
from photomatagent.scientific.loop.evaluation import (
    EvaluationReport,
    PropertyEvaluation,
    ScientificEvaluator,
    fidelity_rank,
)
from photomatagent.scientific.loop.feedback import (
    FeedbackSignal,
    RecommendedAction,
    build_feedback,
    format_feedback_for_model,
)
from photomatagent.scientific.loop.policy import (
    ScientificLoopDecision,
    ScientificLoopPolicy,
    ScientificLoopState,
    ScientificLoopSummary,
)
from photomatagent.scientific.loop.scoring import compute_score, score_breakdown
from photomatagent.scientific.loop.stagnation import (
    StagnationDetector,
    gap_signature,
    violation_signature,
)
from photomatagent.scientific.loop.target import (
    ConstraintCheck,
    ConstraintOutcome,
    ConstraintSpec,
    ConstraintViolation,
    TargetSpec,
    canonical_lwir_detector_target,
    evaluate_constraint,
)

__all__ = [
    "CandidateState",
    "ConstraintCheck",
    "ConstraintOutcome",
    "ConstraintSpec",
    "ConstraintViolation",
    "EvaluationReport",
    "FeedbackSignal",
    "PropertyEvaluation",
    "RecommendedAction",
    "ScientificEvaluator",
    "ScientificLoopConfig",
    "ScientificLoopController",
    "ScientificLoopDecision",
    "ScientificLoopPolicy",
    "ScientificLoopState",
    "ScientificLoopSummary",
    "StagnationDetector",
    "TargetSpec",
    "build_feedback",
    "candidate_fingerprint",
    "canonical_lwir_detector_target",
    "compute_score",
    "evaluate_constraint",
    "extract_candidate_from_state",
    "fidelity_rank",
    "format_feedback_for_model",
    "format_loop_state_snapshot",
    "gap_signature",
    "score_breakdown",
    "violation_signature",
]