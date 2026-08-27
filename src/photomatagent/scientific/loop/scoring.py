"""Deterministic, transparent candidate scoring.

First version is intentionally simple: hard-constraint satisfaction, soft
constraint satisfaction, evidence completeness and confidence are combined
with fixed, documented weights. No ML, no hidden hyperparameters.
"""

from __future__ import annotations

from typing import Any

from photomatagent.scientific.loop.target import (
    ConstraintOutcome,
    TargetSpec,
)

# Documented score mix (0..1); deterministic and explainable.
HARD_PASS_WEIGHT = 0.50
SOFT_SCORE_WEIGHT = 0.20
EVIDENCE_WEIGHT = 0.20
CONFIDENCE_WEIGHT = 0.10


def score_breakdown(
    *,
    target: TargetSpec,
    outcomes: list[ConstraintOutcome],
    overall_confidence: float,
) -> dict[str, float]:
    """Return the individual components of a candidate score."""
    hard = [o for o in outcomes if o.severity == "HARD"]
    soft = [o for o in outcomes if o.severity == "SOFT"]
    hard_count = len(target.hard_constraints())
    hard_pass = sum(1 for o in hard if o.result == "PASS")
    hard_pass_ratio = (hard_pass / hard_count) if hard_count else 1.0
    soft_score = (
        sum(o.soft_score for o in soft) / len(soft) if soft else 1.0
    )
    evidence_completeness = (
        sum(1 for o in outcomes if o.evidence_found) / len(outcomes)
        if outcomes
        else 0.0
    )
    return {
        "hard_pass_ratio": round(hard_pass_ratio, 6),
        "soft_score": round(soft_score, 6),
        "evidence_completeness": round(evidence_completeness, 6),
        "confidence": round(float(overall_confidence), 6),
    }


def compute_score(
    *,
    target: TargetSpec,
    outcomes: list[ConstraintOutcome],
    overall_confidence: float,
) -> float:
    """Composite 0..1 score:
    ``0.50 * hard_pass_ratio + 0.20 * soft_score + 0.20 * evidence_completeness
    + 0.10 * confidence``.
    """
    parts = score_breakdown(
        target=target,
        outcomes=outcomes,
        overall_confidence=overall_confidence,
    )
    return round(
        HARD_PASS_WEIGHT * parts["hard_pass_ratio"]
        + SOFT_SCORE_WEIGHT * parts["soft_score"]
        + EVIDENCE_WEIGHT * parts["evidence_completeness"]
        + CONFIDENCE_WEIGHT * parts["confidence"],
        6,
    )


def summarize_score(parts: dict[str, Any]) -> str:
    """Short human-readable score explanation used in reports and feedback."""
    return (
        f"hard={parts['hard_pass_ratio']:.2f} soft={parts['soft_score']:.2f} "
        f"evidence={parts['evidence_completeness']:.2f} "
        f"confidence={parts['confidence']:.2f}"
    )