"""FeedbackEngine: turns an EvaluationReport into actionable next-round signals.

Feedback is never "try again". It states: what failed, why, what evidence is
missing, what to do next and in what order, and what must not be repeated.

An optional advisory ``JudgeReport`` (LLM Scientific Judge) is embedded into
the signal: its significant concerns become validation actions and priority
lines. Judge concerns can keep the loop investigating, but they never remove
deterministic violations or gaps, and they never convert a PASS into a FAIL.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from photomatagent.scientific.loop.candidate import (
    CandidateState,
    candidate_fingerprint,
)
from photomatagent.scientific.loop.evaluation import EvaluationReport
from photomatagent.scientific.loop.judge import JudgeIssue, JudgeReport
from photomatagent.scientific.loop.progress import progress_from_evaluation
from photomatagent.scientific.state import ScientificState
from photomatagent.scientific.loop.target import (
    ConstraintSpec,
    ConstraintViolation,
    TargetSpec,
)

ActionType = Literal[
    "SEARCH",
    "CALCULATE",
    "GENERATE",
    "VALIDATE",
    "ESCALATE_FIDELITY",
    "CHANGE_STRATEGY",
]
FeedbackDecision = Literal["CONTINUE", "REVISE", "ESCALATE", "REJECT"]


class RecommendedAction(BaseModel):
    action_type: ActionType
    description: str
    target_property: str | None = None
    preferred_capability: str | None = None
    priority: int = 0


class FeedbackSignal(BaseModel):
    candidate_id: str
    decision: FeedbackDecision = "CONTINUE"
    violations: list[ConstraintViolation] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    recommended_actions: list[RecommendedAction] = Field(default_factory=list)
    prohibited_repeats: list[str] = Field(default_factory=list)
    summary: str = ""
    judge: JudgeReport | None = None


def build_feedback(
    target: TargetSpec,
    candidate: CandidateState,
    evaluation: EvaluationReport,
    history: list[CandidateState],
    judge: JudgeReport | None = None,
    *,
    scientific: ScientificState | None = None,
    prior_evaluations: Sequence[tuple[CandidateState, EvaluationReport]] | None = None,
    # Kept for source compatibility with the round-1 API.  Evidence policy is
    # evaluator-owned now, so this flag is intentionally not consulted.
    allow_synthetic_evidence: bool = False,
) -> FeedbackSignal | None:
    """Build the next-round feedback for one evaluated candidate.

    Returns ``None`` when there is nothing actionable: a deterministic PASS
    with no significant judge concerns needs no feedback -- the loop policy
    decides SUCCESS instead.

    The judge NEVER overrides hard constraints: deterministic violations and
    gaps are always kept; judge concerns are appended as advisory actions.
    """
    judge_concerns = (
        list(judge.significant_issues) if judge is not None and judge.available else []
    )
    if evaluation.verdict == "PASS" and not judge_concerns:
        return None

    prohibited: list[str] = []
    comparison_candidates = (
        [previous for previous, _ in prior_evaluations]
        if prior_evaluations is not None
        else history
    )
    for previous in comparison_candidates:
        if (
            previous.candidate_id != candidate.candidate_id
            and candidate_fingerprint(previous) == candidate.fingerprint
        ):
            prohibited.append(candidate.label or candidate.candidate_id)

    supplemental_evidence = _has_supplemental_evidence(
        candidate,
        evaluation,
        scientific,
        prior_evaluations=prior_evaluations,
    )
    duplicate = bool(prohibited) and not supplemental_evidence
    if supplemental_evidence:
        prohibited = []
    hard_violations = [v for v in evaluation.violations if v.severity == "HARD"]
    soft_violations = [v for v in evaluation.violations if v.severity == "SOFT"]
    low_fidelity_critical = _low_fidelity_critical(target, evaluation)

    if hard_violations or duplicate:
        decision: FeedbackDecision = "REJECT" if duplicate else "REVISE"
    elif low_fidelity_critical:
        # Key constraint rests only on cheap evidence: escalate fidelity.
        decision = "ESCALATE"
    elif evaluation.critical_evidence_gaps:
        decision = "CONTINUE"
    elif evaluation.contradictions:
        decision = "CONTINUE"
    else:
        decision = "REVISE"

    if evaluation.verdict == "FAIL":
        decision = "REJECT" if duplicate else "REVISE"
    elif evaluation.verdict == "INCONCLUSIVE" and decision == "REVISE":
        decision = "CONTINUE"
    elif evaluation.verdict == "PASS" and judge_concerns:
        # Deterministic pass but the judge wants more certainty: keep going.
        decision = "CONTINUE"

    actions = _recommended_actions(
        target=target,
        candidate=candidate,
        evaluation=evaluation,
        hard_violations=hard_violations,
        soft_violations=soft_violations,
        low_fidelity_critical=low_fidelity_critical,
        duplicate=duplicate,
        judge_concerns=judge_concerns,
    )
    summary = _summarize(
        target=target,
        candidate=candidate,
        evaluation=evaluation,
        duplicate=duplicate,
        prohibited=prohibited,
        actions=actions,
        judge=judge,
    )
    return FeedbackSignal(
        candidate_id=candidate.candidate_id,
        decision=decision,
        violations=evaluation.violations,
        evidence_gaps=evaluation.critical_evidence_gaps,
        contradictions=evaluation.contradictions,
        recommended_actions=actions,
        prohibited_repeats=prohibited,
        summary=summary,
        judge=judge,
    )


def _low_fidelity_critical(
    target: TargetSpec, evaluation: EvaluationReport
) -> bool:
    """Any HARD constraint with evidence below the escalation threshold."""
    for result in evaluation.constraint_results:
        constraint = target.constraint(result.property)
        if constraint is None or constraint.severity != "HARD":
            continue
        if result.result in {"PASS", "FAIL"} and result.confidence > 0.0:
            if result.confidence <= 0.55:  # analytical/empirical or below
                return True
    return False


def _recommended_actions(
    *,
    target: TargetSpec,
    candidate: CandidateState,
    evaluation: EvaluationReport,
    hard_violations: list[ConstraintViolation],
    soft_violations: list[ConstraintViolation],
    low_fidelity_critical: bool,
    duplicate: bool,
    judge_concerns: list[JudgeIssue] | None = None,
) -> list[RecommendedAction]:
    if judge_concerns is None:
        judge_concerns = []
    actions: list[RecommendedAction] = []
    priority = 1

    if duplicate:
        actions.append(
            RecommendedAction(
                action_type="CHANGE_STRATEGY",
                description=(
                    "do not repeat the identical candidate; vary composition, "
                    "structure family or generation conditions"
                ),
                target_property=None,
                priority=priority,
            )
        )
        priority += 1

    for gap in evaluation.critical_evidence_gaps:
        constraint = target.constraint(gap)
        actions.append(
            RecommendedAction(
                action_type="CALCULATE",
                description=f"obtain evidence for {gap}",
                target_property=gap,
                preferred_capability=_preferred_capability(constraint),
                priority=priority,
            )
        )
        priority += 1

    for violation in hard_violations:
        actions.append(
            RecommendedAction(
                action_type="GENERATE",
                description=(
                    f"resolve hard constraint on {violation.property}: "
                    f"{violation.message}"
                ),
                target_property=violation.property,
                preferred_capability="generation",
                priority=priority,
            )
        )
        priority += 1

    for violation in soft_violations:
        actions.append(
            RecommendedAction(
                action_type="GENERATE",
                description=f"improve soft constraint: {violation.message}",
                target_property=violation.property,
                priority=priority,
            )
        )
        priority += 1

    if low_fidelity_critical:
        actions.append(
            RecommendedAction(
                action_type="ESCALATE_FIDELITY",
                description=(
                    "critical constraints currently rest on analytical/empirical "
                    "evidence; escalate to a higher-fidelity calculation before "
                    "claiming success"
                ),
                priority=priority,
            )
        )
        priority += 1

    for contradiction in evaluation.contradictions:
        actions.append(
            RecommendedAction(
                action_type="VALIDATE",
                description=f"resolve contradictory evidence: {contradiction}",
                priority=priority,
            )
        )
        priority += 1

    # Advisory judge concerns: always after the deterministic actions, so they
    # can never crowd out hard-constraint work.
    for issue in judge_concerns:
        if issue.severity == "LOW":
            continue
        if issue.category == "evidence_gap":
            action_type: ActionType = "CALCULATE"
        elif issue.category == "evidence_quality":
            action_type = "ESCALATE_FIDELITY"
        else:
            action_type = "VALIDATE"
        actions.append(
            RecommendedAction(
                action_type=action_type,
                description=f"judge concern ({issue.category}): {issue.description}",
                target_property=issue.property,
                priority=priority,
            )
        )
        priority += 1

    if not actions:
        actions.append(
            RecommendedAction(
                action_type="CALCULATE",
                description="collect the missing evidence listed in the evaluation",
                priority=priority,
            )
        )
    return actions


def _summarize(
    *,
    target: TargetSpec,
    candidate: CandidateState,
    evaluation: EvaluationReport,
    duplicate: bool,
    prohibited: list[str],
    actions: list[RecommendedAction],
    judge: JudgeReport | None = None,
) -> str:
    if evaluation.verdict == "INCONCLUSIVE":
        opening = "has not yet been verified against"
    elif evaluation.verdict == "FAIL":
        opening = "fails"
    else:
        opening = "does not satisfy"
    lines: list[str] = [
        f"Candidate {candidate.label or candidate.candidate_id} {opening} "
        f"the target (verdict={evaluation.verdict}, score={evaluation.score:.3f})."
    ]
    if evaluation.violations:
        lines.append("Violations: " + "; ".join(v.short() for v in evaluation.violations))
    if evaluation.critical_evidence_gaps:
        lines.append("Missing evidence: " + ", ".join(evaluation.critical_evidence_gaps))
    if evaluation.contradictions:
        lines.append("Contradictions: " + "; ".join(evaluation.contradictions))
    if duplicate:
        lines.append("This candidate repeats an already-proposed candidate.")
    if prohibited:
        lines.append("Do not repeat: " + ", ".join(prohibited))
    if judge is not None and judge.available:
        concerns = judge.significant_issues
        if concerns:
            lines.append(
                "Judge concerns: "
                + "; ".join(
                    f"[{issue.severity}] {issue.description}" for issue in concerns
                )
            )
        if judge.recommendations:
            lines.append(
                "Judge recommendations: " + "; ".join(judge.recommendations)
            )
    if actions:
        lines.append(
            "Priority: " + "; ".join(f"{a.priority}. {a.description}" for a in actions)
        )
    return " ".join(lines)


def _has_supplemental_evidence(
    candidate: CandidateState,
    evaluation: EvaluationReport,
    scientific: ScientificState | None,
    *,
    prior_evaluations: Sequence[tuple[CandidateState, EvaluationReport]] | None,
) -> bool:
    """Recognize only a newly accepted observation from evaluator history.

    A missing prior manifest is treated as unknown provenance and keeps the
    candidate rejected.  This is deliberately fail-closed for old snapshots
    and hand-built historical reports.
    """
    if scientific is None or prior_evaluations is None:
        return False
    current_progress = progress_from_evaluation(
        candidate,
        evaluation,
        scientific,
    )
    current_observations = {
        key.rsplit("observation:", 1)[-1]
        for key in current_progress.observation_keys
    }
    if not current_observations:
        return False
    previous_observations: set[str] = set()
    matching_history = [
        (previous, previous_evaluation)
        for previous, previous_evaluation in prior_evaluations
        if (
            previous.candidate_id != candidate.candidate_id
            and candidate_fingerprint(previous) == candidate.fingerprint
        )
    ]
    if not matching_history:
        return False
    for previous, previous_evaluation in matching_history:
        previous_progress = progress_from_evaluation(
            previous,
            previous_evaluation,
            scientific,
        )
        if not previous_evaluation.accepted_evidence_manifest:
            return False
        if not previous_progress.observation_keys:
            return False
        previous_observations.update(
            key.rsplit("observation:", 1)[-1]
            for key in previous_progress.observation_keys
        )
    return bool(current_observations - previous_observations)


def _preferred_capability(constraint: ConstraintSpec | None) -> str:
    """Map a property to the capability most likely to produce its evidence."""
    if constraint is None:
        return "scientific"
    return {
        "band_gap": "electronic",
        "formation_energy": "materials",
        "responsivity": "photodetector",
        "quantum_efficiency": "photodetector",
        "detectivity": "device",
        "dark_current": "device",
        "density": "materials",
    }.get(constraint.property, "scientific")


def format_feedback_for_model(signal: FeedbackSignal, *, round_number: int) -> str:
    """Render one FeedbackSignal as the next round's research instruction."""
    lines = [
        f"Scientific feedback from round {round_number}",
        "",
        f"Decision: {signal.decision}",
    ]
    if "verdict=INCONCLUSIVE" in signal.summary:
        lines.append("Status: the target has not yet been verified.")
    elif "verdict=FAIL" in signal.summary:
        lines.append("Status: the candidate fails at least one target constraint.")
    if signal.violations:
        lines.append("Hard constraint violations:")
        lines += [f"- {v.message}" for v in signal.violations if v.severity == "HARD"]
        lines += [f"- (soft) {v.message}" for v in signal.violations if v.severity == "SOFT"]
    if signal.evidence_gaps:
        lines.append("Missing evidence:")
        lines += [f"- {gap}" for gap in signal.evidence_gaps]
    if signal.contradictions:
        lines.append("Contradictions:")
        lines += [f"- {item}" for item in signal.contradictions]
    if signal.judge is not None and signal.judge.available:
        concerns = signal.judge.significant_issues
        if concerns:
            lines.append("Judge concerns (advisory):")
            lines += [f"- {issue.description}" for issue in concerns]
    if signal.prohibited_repeats:
        lines.append("Do not repeat:")
        lines += [f"- {item}" for item in signal.prohibited_repeats]
    if signal.recommended_actions:
        lines.append("Priority for the next round:")
        for action in sorted(signal.recommended_actions, key=lambda a: a.priority):
            lines.append(f"{action.priority}. [{action.action_type}] {action.description}")
    lines.append(
        "Do not claim completion until the scientific evaluator reports all hard "
        "constraints satisfied or the loop policy declares the task inconclusive."
    )
    return "\n".join(lines)
