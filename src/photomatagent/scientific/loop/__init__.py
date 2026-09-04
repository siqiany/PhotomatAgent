"""Evidence-guided scientific feedback loop with lazy public exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

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
