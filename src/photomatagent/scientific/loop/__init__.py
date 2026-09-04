"""Evidence-guided scientific feedback loop with lazy public exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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
    from photomatagent.scientific.loop.judge import (
        JudgeIssue,
        JudgeReport,
        ScientificJudge,
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

_EXPORT_MODULE = {
    "CandidateState": "candidate",
    "candidate_fingerprint": "candidate",
    "extract_candidate_from_state": "candidate",
    "ScientificLoopConfig": "controller",
    "ScientificLoopController": "controller",
    "format_loop_state_snapshot": "controller",
    "EvaluationReport": "evaluation",
    "PropertyEvaluation": "evaluation",
    "ScientificEvaluator": "evaluation",
    "fidelity_rank": "evaluation",
    "FeedbackSignal": "feedback",
    "RecommendedAction": "feedback",
    "build_feedback": "feedback",
    "format_feedback_for_model": "feedback",
    "JudgeIssue": "judge",
    "JudgeReport": "judge",
    "ScientificJudge": "judge",
    "ScientificLoopDecision": "policy",
    "ScientificLoopPolicy": "policy",
    "ScientificLoopState": "policy",
    "ScientificLoopSummary": "policy",
    "compute_score": "scoring",
    "score_breakdown": "scoring",
    "StagnationDetector": "stagnation",
    "gap_signature": "stagnation",
    "violation_signature": "stagnation",
    "ConstraintCheck": "target",
    "ConstraintOutcome": "target",
    "ConstraintSpec": "target",
    "ConstraintViolation": "target",
    "TargetSpec": "target",
    "canonical_lwir_detector_target": "target",
    "evaluate_constraint": "target",
}

__all__ = list(_EXPORT_MODULE)


def __getattr__(name: str) -> Any:
    try:
        relative_module = _EXPORT_MODULE[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    module = import_module(f"photomatagent.scientific.loop.{relative_module}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
