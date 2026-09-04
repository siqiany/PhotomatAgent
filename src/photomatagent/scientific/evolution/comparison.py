"""Deterministic, evidence-preserving comparison of adjacent episodes."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from photomatagent.redaction import redact_secrets
from photomatagent.scientific.evolution.models import (
    AcceptanceResult,
    AcceptanceStatus,
    ArtifactDiff,
    ComparisonReport,
    ConstraintChangeSummary,
    CostDelta,
    EpisodeRecord,
    EvidenceChangeSummary,
    ExpertFeedbackRecord,
    FeedbackDelta,
    FidelityChangeSummary,
    RevisionPlan,
    RubricScoreDelta,
    validate_managed_id,
)
from photomatagent.scientific.evolution.rubric import expert_utility
from photomatagent.scientific.loop.evaluation import fidelity_rank
from photomatagent.scientific.state import ScientificState

_DIMENSIONS = (
    "scientific_correctness",
    "evidence_sufficiency",
    "novelty",
    "actionability",
    "overall",
)
_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("expert_utility_delta", 0.45),
    ("closure_rate", 0.25),
    ("recurrence_rate", -0.15),
    ("new_issue_rate", -0.10),
    ("normalized_cost_increase", -0.05),
)
_SEVERITY_WEIGHT = {"LOW": 0.25, "MEDIUM": 0.5, "HIGH": 0.75, "CRITICAL": 1.0}


def compare_episodes(
    *,
    previous: EpisodeRecord,
    current: EpisodeRecord,
    previous_plan: RevisionPlan,
    previous_feedback: ExpertFeedbackRecord | None = None,
    current_feedback: ExpertFeedbackRecord | None = None,
    previous_items: Sequence[FeedbackDelta] | None = None,
    current_items: Sequence[FeedbackDelta] | None = None,
    machine_results: Mapping[str, bool | str | AcceptanceResult] | None = None,
    previous_state: ScientificState | None = None,
    current_state: ScientificState | None = None,
    comparison_id: str | None = None,
) -> ComparisonReport:
    """Compare exactly one adjacent episode pair without model judgment.

    Missing machine results and missing later expert feedback are unknown.  In
    particular, silence never closes an issue or becomes a zero recurrence
    rate.  ``current_items`` must therefore be ``None`` when no later compiled
    review exists, rather than an invented empty review.
    """

    _validate_inputs(
        previous=previous,
        current=current,
        previous_plan=previous_plan,
        previous_feedback=previous_feedback,
        current_feedback=current_feedback,
    )
    prior = tuple(previous_items or ())
    later = None if current_items is None else tuple(current_items)
    score_deltas = _score_deltas(previous_feedback, current_feedback)
    acceptance, closed = _acceptance_results(
        previous_plan=previous_plan,
        previous_items=prior,
        current_items=later,
        machine_results=machine_results or {},
    )
    evaluated = [item for item in acceptance if item.status in {"PASS", "FAIL"}]
    closure_rate = (
        sum(item.status == "PASS" for item in evaluated) / len(evaluated)
        if evaluated
        else None
    )
    recurring, new, recurrence_rate, new_issue_rate = _issue_changes(prior, later)
    constraint_changes = _constraint_changes(previous, current)
    evidence_changes, fidelity_changes = _evidence_changes(
        previous,
        current,
        previous_plan,
        previous_state,
        current_state,
    )
    artifact_diff = _artifact_diff(previous, current)
    cost_delta = _cost_delta(previous, current)
    normalized_cost = _normalized_cost_increase(previous, current)
    utility_delta = (
        round(
            expert_utility(current_feedback.scores)
            - expert_utility(previous_feedback.scores),
            6,
        )
        if previous_feedback is not None and current_feedback is not None
        else None
    )
    reward, components = compute_learning_signal(
        expert_utility_delta=utility_delta,
        closure_rate=closure_rate,
        recurrence_rate=recurrence_rate,
        new_issue_rate=new_issue_rate,
        normalized_cost_increase=normalized_cost,
    )
    closed_ids = list(
        dict.fromkeys(
            item.acceptance_id for item in acceptance if item.status == "PASS"
        )
    )
    # Acceptance IDs are the source issue IDs whenever an issue can be linked.
    # Keep the explicit set returned by the matcher to avoid classifying an
    # unlinked plan-level check as an expert issue.
    closed_ids = [value for value in closed_ids if value in closed]
    return ComparisonReport(
        comparison_id=comparison_id or f"cmp_{previous.version}_{current.version}",
        evolution_id=previous.evolution_id,
        previous_version=previous.version,
        current_version=current.version,
        score_deltas=score_deltas,
        acceptance_results=acceptance,
        closed_issue_ids=closed_ids,
        recurring_issue_ids=[item.item_id for item in recurring if item.item_id],
        new_issue_ids=[item.item_id for item in new if item.item_id],
        closure_rate=closure_rate,
        recurrence_rate=recurrence_rate,
        new_issue_rate=new_issue_rate,
        constraint_changes=constraint_changes,
        evidence_changes=evidence_changes,
        fidelity_changes=fidelity_changes,
        artifact_diff=artifact_diff,
        cost_delta=cost_delta,
        unresolved_human_checks=[
            item.detail
            for item in acceptance
            if item.status == "NEEDS_HUMAN_REVIEW"
        ],
        expert_utility_delta=utility_delta,
        normalized_cost_increase=normalized_cost,
        reward=reward,
        components_used=components,
        module_credit=_module_credit(prior, later, set(closed)),
        created_at=current.completed_at or current.created_at,
    )


def compute_learning_signal(
    *,
    expert_utility_delta: float | None,
    closure_rate: float | None,
    recurrence_rate: float | None,
    new_issue_rate: float | None,
    normalized_cost_increase: float | None,
) -> tuple[float | None, list[str]]:
    """Apply the approved fixed reward with missing-weight renormalization."""

    values = {
        "expert_utility_delta": expert_utility_delta,
        "closure_rate": closure_rate,
        "recurrence_rate": recurrence_rate,
        "new_issue_rate": new_issue_rate,
        "normalized_cost_increase": normalized_cost_increase,
    }
    available = [
        (name, weight, value)
        for name, weight in _WEIGHTS
        if (value := values[name]) is not None
    ]
    if not available:
        return None, []
    used = [name for name, _weight, _value in available]
    denominator = sum(abs(weight) for _name, weight, _value in available)
    raw = sum(weight * value for _name, weight, value in available)
    return round(max(-1.0, min(1.0, raw / denominator)), 6), used


def _validate_inputs(
    *,
    previous: EpisodeRecord,
    current: EpisodeRecord,
    previous_plan: RevisionPlan,
    previous_feedback: ExpertFeedbackRecord | None,
    current_feedback: ExpertFeedbackRecord | None,
) -> None:
    if previous.status != "COMPLETED" or current.status != "COMPLETED":
        raise ValueError("episode comparison requires two completed episodes")
    if previous.evolution_id != current.evolution_id:
        raise ValueError("episode comparison requires one evolution task")
    if current.parent_version != previous.version:
        raise ValueError("current episode parent does not match previous version")
    if (
        previous_plan.evolution_id != previous.evolution_id
        or previous_plan.source_version != previous.version
        or current.revision_plan_id != previous_plan.revision_id
        or not previous_plan.confirmed
    ):
        raise ValueError(
            "revision plan is not the confirmed plan applied to current episode"
        )
    for feedback, episode, label in (
        (previous_feedback, previous, "previous"),
        (current_feedback, current, "current"),
    ):
        if feedback is None:
            continue
        if (
            feedback.evolution_id != episode.evolution_id
            or feedback.episode_version != episode.version
            or (
                episode.artifact is not None
                and feedback.result_sha256 != episode.artifact.sha256
            )
        ):
            raise ValueError(f"{label} feedback is not bound to its episode artifact")
    if (
        previous_feedback is not None
        and previous_plan.feedback_id != previous_feedback.feedback_id
    ):
        raise ValueError("revision plan is not bound to previous feedback")


def _score_deltas(
    previous: ExpertFeedbackRecord | None,
    current: ExpertFeedbackRecord | None,
) -> list[RubricScoreDelta]:
    if previous is None or current is None:
        return []
    values: list[RubricScoreDelta] = []
    for dimension in _DIMENSIONS:
        left = getattr(previous.scores, dimension)
        right = getattr(current.scores, dimension)
        delta = right - left
        values.append(
            RubricScoreDelta(
                dimension=dimension,  # type: ignore[arg-type]
                previous=left,
                current=right,
                delta=delta,
                normalized_delta=delta / 4,
            )
        )
    return values


def _negative(items: Sequence[FeedbackDelta]) -> list[FeedbackDelta]:
    return [item for item in items if item.status != "POSITIVE_SIGNAL"]


def _signature(item: FeedbackDelta) -> tuple[str, str]:
    return item.category, item.responsible_module.strip().casefold()


def _issue_changes(
    previous: Sequence[FeedbackDelta],
    current: Sequence[FeedbackDelta] | None,
) -> tuple[list[FeedbackDelta], list[FeedbackDelta], float | None, float | None]:
    if current is None:
        return [], [], None, None
    prior = _negative(previous)
    later = _negative(current)
    old_signatures = {_signature(item) for item in prior}
    recurring = [item for item in later if _signature(item) in old_signatures]
    new = [item for item in later if _signature(item) not in old_signatures]
    recurrence_rate = (
        len({_signature(item) for item in recurring}) / len(old_signatures)
        if old_signatures
        else 0.0
    )
    later_signatures = {_signature(item) for item in later}
    new_signatures = {_signature(item) for item in new}
    new_issue_rate = (
        len(new_signatures) / len(later_signatures)
        if later_signatures
        else 0.0
    )
    return recurring, new, recurrence_rate, new_issue_rate


def _acceptance_results(
    *,
    previous_plan: RevisionPlan,
    previous_items: Sequence[FeedbackDelta],
    current_items: Sequence[FeedbackDelta] | None,
    machine_results: Mapping[str, bool | str | AcceptanceResult],
) -> tuple[list[AcceptanceResult], set[str]]:
    results: list[AcceptanceResult] = []
    closed: set[str] = set()
    items_by_test = {
        item.acceptance_test: item
        for item in _negative(previous_items)
        if item.acceptance_test
    }
    linked_ids: set[str] = set()
    for test in previous_plan.machine_acceptance_tests:
        issue = items_by_test.get(test)
        acceptance_id = _acceptance_id(issue, test)
        if issue is not None and issue.item_id:
            linked_ids.add(issue.item_id)
        status = _machine_status(machine_results.get(test))
        results.append(
            AcceptanceResult(
                acceptance_id=acceptance_id,
                status=status,
                detail=test,
            )
        )
        if status == "PASS" and issue is not None and issue.item_id:
            closed.add(issue.item_id)

    positive_signatures = (
        {_signature(item) for item in current_items if item.status == "POSITIVE_SIGNAL"}
        if current_items is not None
        else set()
    )
    human_candidates = [
        item
        for item in _negative(previous_items)
        if item.item_id not in linked_ids
    ]
    for index, test in enumerate(previous_plan.human_acceptance_tests):
        issue = human_candidates[index] if index < len(human_candidates) else None
        acceptance_id = _acceptance_id(issue, test)
        confirmed = issue is not None and _signature(issue) in positive_signatures
        human_status: AcceptanceStatus = (
            "PASS" if confirmed else "NEEDS_HUMAN_REVIEW"
        )
        results.append(
            AcceptanceResult(
                acceptance_id=acceptance_id,
                status=human_status,
                detail=test,
            )
        )
        if confirmed and issue is not None and issue.item_id:
            closed.add(issue.item_id)

    # A structured positive signal is an explicit expert confirmation even if
    # the plan did not render a separate human-only test for that item.
    represented = {item.acceptance_id for item in results}
    for issue in _negative(previous_items):
        if (
            issue.item_id
            and issue.item_id not in represented
            and _signature(issue) in positive_signatures
        ):
            results.append(
                AcceptanceResult(
                    acceptance_id=issue.item_id,
                    status="PASS",
                    detail="later expert review explicitly confirmed this issue",
                )
            )
            closed.add(issue.item_id)
    return results, closed


def _acceptance_id(item: FeedbackDelta | None, test: str) -> str:
    if item is not None and item.item_id is not None:
        return item.item_id
    digest = hashlib.sha256(test.encode("utf-8")).hexdigest()[:12]
    return f"accept_{digest}"


def _machine_status(
    value: bool | str | AcceptanceResult | None,
) -> AcceptanceStatus:
    if isinstance(value, AcceptanceResult):
        return (
            value.status
            if value.status in {"PASS", "FAIL"}
            else "NEEDS_HUMAN_REVIEW"
        )
    if value is True or value == "PASS":
        return "PASS"
    if value is False or value == "FAIL":
        return "FAIL"
    return "NEEDS_HUMAN_REVIEW"


def _outcomes(episode: EpisodeRecord) -> dict[str, object]:
    if episode.summary is None or episode.summary.final_evaluation is None:
        return {}
    return {
        item.property: item
        for item in episode.summary.final_evaluation.constraint_results
    }


def _constraint_changes(
    previous: EpisodeRecord,
    current: EpisodeRecord,
) -> ConstraintChangeSummary:
    old = _outcomes(previous)
    new = _outcomes(current)
    newly_passed: list[str] = []
    newly_failed: list[str] = []
    newly_unknown: list[str] = []
    still_failed: list[str] = []
    still_unknown: list[str] = []
    for name in sorted(new):
        current_result = getattr(new[name], "result")
        previous_result = getattr(old.get(name), "result", None)
        if current_result == "PASS" and previous_result != "PASS":
            newly_passed.append(name)
        elif current_result == "FAIL":
            (still_failed if previous_result == "FAIL" else newly_failed).append(name)
        elif current_result == "UNKNOWN":
            target = still_unknown if previous_result == "UNKNOWN" else newly_unknown
            target.append(name)
    return ConstraintChangeSummary(
        newly_passed=newly_passed,
        newly_failed=newly_failed,
        newly_unknown=newly_unknown,
        still_failed=still_failed,
        still_unknown=still_unknown,
    )


def _evidence_map(
    episode: EpisodeRecord,
    state: ScientificState | None,
) -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    if state is not None:
        for item in state.evidence:
            try:
                evidence_id = validate_managed_id(item.id)
            except (TypeError, ValueError):
                continue
            values[evidence_id] = getattr(item, "fidelity", "empirical")
        return values
    for outcome in _outcomes(episode).values():
        for raw_id in getattr(outcome, "evidence_ids"):
            try:
                evidence_id = validate_managed_id(raw_id)
            except (TypeError, ValueError):
                continue
            values[evidence_id] = getattr(outcome, "fidelity")
    return values


def _evidence_changes(
    previous: EpisodeRecord,
    current: EpisodeRecord,
    plan: RevisionPlan,
    previous_state: ScientificState | None,
    current_state: ScientificState | None,
) -> tuple[EvidenceChangeSummary, FidelityChangeSummary]:
    old = _evidence_map(previous, previous_state)
    new = _evidence_map(current, current_state)
    old_ids = set(old)
    new_ids = set(new)
    shared = old_ids & new_ids
    upgraded: list[str] = []
    downgraded: list[str] = []
    unchanged: list[str] = []
    for evidence_id in sorted(shared):
        left = fidelity_rank(old[evidence_id])
        right = fidelity_rank(new[evidence_id])
        if right > left:
            upgraded.append(evidence_id)
        elif right < left:
            downgraded.append(evidence_id)
        else:
            unchanged.append(evidence_id)
    old_gaps = set(
        previous.summary.unresolved_evidence_gaps if previous.summary else ()
    )
    new_gaps = set(current.summary.unresolved_evidence_gaps if current.summary else ())
    return (
        EvidenceChangeSummary(
            added_ids=sorted(new_ids - old_ids),
            removed_ids=sorted(old_ids - new_ids),
            carried_ids=sorted(shared),
            invalidated_ids=sorted(set(plan.invalidated_evidence_ids)),
            resolved_gaps=sorted(old_gaps - new_gaps),
            new_gaps=sorted(new_gaps - old_gaps),
        ),
        FidelityChangeSummary(
            upgraded_ids=upgraded,
            downgraded_ids=downgraded,
            unchanged_ids=unchanged,
        ),
    )


def _artifact_diff(
    previous: EpisodeRecord,
    current: EpisodeRecord,
) -> ArtifactDiff | None:
    if previous.artifact is None or current.artifact is None:
        return None
    return ArtifactDiff(
        previous_sha256=previous.artifact.sha256,
        current_sha256=current.artifact.sha256,
        changed=previous.artifact.sha256 != current.artifact.sha256,
        size_bytes_delta=current.artifact.size_bytes - previous.artifact.size_bytes,
        summary="primary result bytes changed"
        if previous.artifact.sha256 != current.artifact.sha256
        else "primary result bytes unchanged",
    )


def _cost_delta(previous: EpisodeRecord, current: EpisodeRecord) -> CostDelta:
    left = previous.cost
    right = current.cost
    hpc = (
        right.hpc_cost - left.hpc_cost
        if left.hpc_cost is not None and right.hpc_cost is not None
        else None
    )
    return CostDelta(
        input_tokens=right.input_tokens - left.input_tokens,
        output_tokens=right.output_tokens - left.output_tokens,
        tool_calls=right.tool_calls - left.tool_calls,
        runtime_seconds=right.runtime_seconds - left.runtime_seconds,
        hpc_cost=hpc,
    )


def _normalized_cost_increase(previous: EpisodeRecord, current: EpisodeRecord) -> float:
    pairs: list[tuple[float, float]] = [
        (float(previous.cost.input_tokens), float(current.cost.input_tokens)),
        (float(previous.cost.output_tokens), float(current.cost.output_tokens)),
        (float(previous.cost.tool_calls), float(current.cost.tool_calls)),
        (previous.cost.runtime_seconds, current.cost.runtime_seconds),
    ]
    if previous.cost.hpc_cost is not None and current.cost.hpc_cost is not None:
        pairs.append((previous.cost.hpc_cost, current.cost.hpc_cost))
    changes = [
        max(-1.0, min(1.0, (right - left) / max(left, 1.0)))
        for left, right in pairs
    ]
    return round(sum(changes) / len(changes), 6)


def _module_credit(
    previous: Sequence[FeedbackDelta],
    current: Sequence[FeedbackDelta] | None,
    closed_ids: set[str],
) -> dict[str, float]:
    scores: dict[str, list[float]] = {}
    prior_signatures = {_signature(item) for item in _negative(previous)}
    for item in _negative(previous):
        if item.item_id in closed_ids:
            scores.setdefault(_safe_module(item.responsible_module), []).append(
                _SEVERITY_WEIGHT[item.severity]
            )
    if current is not None:
        for item in _negative(current):
            penalty = -_SEVERITY_WEIGHT[item.severity]
            module = _safe_module(item.responsible_module)
            scores.setdefault(module, []).append(penalty)
            if _signature(item) in prior_signatures:
                # Recurrence is deliberately visible as a second penalty while
                # still remaining one episode-pair observation.
                scores[module].append(penalty)
    return {
        module: round(max(-1.0, min(1.0, sum(values) / len(values))), 6)
        for module, values in sorted(scores.items())
        if values
    }


def _safe_module(value: str) -> str:
    redacted = redact_secrets({value: value})[value]
    safe = str(redacted)
    return safe if len(safe) <= 200 else safe[:199] + "…"


__all__ = ["compare_episodes", "compute_learning_signal"]
