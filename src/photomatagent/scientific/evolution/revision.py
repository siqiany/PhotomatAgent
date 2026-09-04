"""Deterministic revision planning from validated feedback compilations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence

from photomatagent.redaction import redact_text
from photomatagent.scientific.evolution.models import (
    ExpertFeedbackRecord,
    FeedbackCompilation,
    FeedbackDelta,
    RevisionPlan,
    StrategyArm,
)

_MAX_ITEM_TEXT = 500
_MAX_PLAN_ITEMS = 100
_MAX_INSTRUCTION_CHARS = 16_000
_MAX_INSTRUCTION_ITEMS = 40
_EVIDENCE_MODULE_TERMS = (
    "evidence",
    "retriev",
    "source",
    "citation",
    "literature",
    "fidelity",
)
_OUTPUT_MODULE_TERMS = ("output", "report", "schema", "deliverable", "render")
_TARGET_MODULE_TERMS = ("target", "constraint", "contract", "task")
_DIVERSITY_MODULE_TERMS = ("innovation", "novel", "divers", "candidate", "search")
_UNCERTAINTY_TERMS = ("uncertain", "uncertainty", "ambig", "unknown")


def build_revision_plan(
    *,
    feedback: ExpertFeedbackRecord,
    compilation: FeedbackCompilation,
) -> RevisionPlan:
    """Compile one AVAILABLE, identity-bound compilation into a safe plan.

    Only validated compilation fields are routed. The feedback ``raw_input`` and
    delta ``source_span`` provenance are deliberately never copied.
    """

    _validate_context(feedback, compilation)
    contract_changes: list[str] = []
    evidence_requirements: list[str] = []
    output_schema_requirements: list[str] = []
    preserved_facts: list[str] = []
    prohibited_repeats: list[str] = []
    invalidated_conclusions: list[str] = []
    machine_tests: list[str] = []
    human_tests: list[str] = []
    ambiguities: list[str] = []

    for index, item in enumerate(compilation.items[:_MAX_PLAN_ITEMS], start=1):
        item_id = item.item_id or f"item_{index:03d}"
        actions = [_bounded(value) for value in item.requested_actions]
        preserve = [_bounded(value) for value in item.preserve]
        module = item.responsible_module.lower()

        if item.status == "POSITIVE_SIGNAL":
            preserved_facts.extend(preserve or [_bounded(item.problem)])
        else:
            preserved_facts.extend(preserve)
            destination = _route_destination(item.category, module)
            if destination == "evidence":
                evidence_requirements.extend(actions)
            elif destination == "output":
                output_schema_requirements.extend(actions)
            else:
                contract_changes.extend(actions)

            if item.status == "CORRECTION":
                prohibited_repeats.append(_bounded(item.problem))
                if item.category == "SCIENTIFIC_CORRECTNESS":
                    invalidated_conclusions.append(_bounded(item.problem))

            if item.acceptance_test:
                acceptance = _bounded(item.acceptance_test)
                if item.status == "QUERY":
                    human_tests.append(f"QUERY {item_id}: {acceptance}")
                else:
                    machine_tests.append(acceptance)
            elif item.status == "QUERY":
                human_tests.append(
                    f"QUERY {item_id}: resolve uncertainty before treating it as fact"
                )

            if (
                item.severity == "CRITICAL"
                and not item.requested_actions
                and not item.acceptance_test
            ):
                ambiguities.append(
                    f"CRITICAL {item_id} needs a requested action or acceptance test"
                )

    arm, reason = _fixed_strategy_choice(compilation.items)
    identity_payload = {
        "compilation_id": compilation.compilation_id,
        "evolution_id": compilation.evolution_id,
        "feedback_id": compilation.feedback_id,
        "episode_version": compilation.episode_version,
        "items": [
            item.model_dump(mode="json", exclude={"source_span"})
            for item in compilation.items
        ],
        "warnings": list(compilation.warnings),
    }
    digest = hashlib.sha256(
        json.dumps(
            identity_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return RevisionPlan(
        revision_id=f"rp_{digest[:10]}",
        evolution_id=feedback.evolution_id,
        source_version=feedback.episode_version,
        feedback_id=feedback.feedback_id,
        contract_changes=_dedupe(contract_changes),
        evidence_requirements=_dedupe(evidence_requirements),
        output_schema_requirements=_dedupe(output_schema_requirements),
        preserved_facts=_dedupe(preserved_facts),
        prohibited_repeats=_dedupe(prohibited_repeats),
        invalidated_conclusions=_dedupe(invalidated_conclusions),
        machine_acceptance_tests=_dedupe(machine_tests),
        human_acceptance_tests=_dedupe(human_tests),
        strategy_arm=arm,
        strategy_reason=reason,
        warnings=_dedupe(_bounded(value) for value in compilation.warnings),
        unresolved_ambiguities=_dedupe(ambiguities),
        has_blocking_ambiguity=bool(ambiguities),
        created_at=compilation.created_at,
    )


def format_revision_instruction(
    plan: RevisionPlan,
    *,
    strategy: StrategyArm,
) -> str:
    """Render a bounded dynamic instruction without feedback provenance prose."""

    if strategy != plan.strategy_arm and plan.strategy_arm != "STATIC":
        # Explicit callers may override STATIC for controlled evaluation, but a
        # feedback-selected non-baseline arm must remain bound to its plan.
        raise ValueError("strategy does not match the confirmed revision plan")
    sections: list[tuple[str, Sequence[str]]] = [
        ("Contract changes", plan.contract_changes),
        ("Evidence requirements", plan.evidence_requirements),
        ("Output schema requirements", plan.output_schema_requirements),
        ("Preserve", plan.preserved_facts),
        ("Preserved evidence IDs", plan.preserved_evidence_ids),
        ("Prohibited repeats", plan.prohibited_repeats),
        ("Invalidated conclusions", plan.invalidated_conclusions),
        ("Invalidated evidence IDs", plan.invalidated_evidence_ids),
        ("Machine acceptance tests", plan.machine_acceptance_tests),
        ("Human acceptance tests", plan.human_acceptance_tests),
        ("Warnings", plan.warnings),
        ("Unresolved ambiguities", plan.unresolved_ambiguities),
    ]
    lines = [
        "Revision requirements",
        f"Strategy: {strategy}",
        "Do not override deterministic constraints or treat UNKNOWN as PASS.",
        "QUERY acceptance items remain questions until explicitly resolved.",
    ]
    for heading, values in sections:
        if not values:
            continue
        lines.append(f"\n{heading}:")
        lines.extend(
            f"- {_bounded(value)}" for value in values[:_MAX_INSTRUCTION_ITEMS]
        )
        omitted = len(values) - min(len(values), _MAX_INSTRUCTION_ITEMS)
        if omitted:
            lines.append(f"- [{omitted} additional bounded items omitted]")
    rendered = "\n".join(lines)
    return rendered[:_MAX_INSTRUCTION_CHARS]


def _validate_context(
    feedback: ExpertFeedbackRecord,
    compilation: FeedbackCompilation,
) -> None:
    if compilation.status != "AVAILABLE":
        raise ValueError("revision planning requires an AVAILABLE compilation")
    if (
        compilation.evolution_id != feedback.evolution_id
        or compilation.feedback_id != feedback.feedback_id
        or compilation.episode_version != feedback.episode_version
    ):
        raise ValueError("feedback compilation identity mismatch")


def _route_destination(category: str, module: str) -> str:
    if category == "EVIDENCE_SUFFICIENCY" or any(
        term in module for term in _EVIDENCE_MODULE_TERMS
    ):
        return "evidence"
    if category == "DELIVERABLE_COMPLETENESS" or any(
        term in module for term in _OUTPUT_MODULE_TERMS
    ):
        return "output"
    if category == "ACTIONABILITY" and not any(
        term in module for term in _TARGET_MODULE_TERMS
    ):
        return "output"
    return "contract"


def _fixed_strategy_choice(
    items: Sequence[FeedbackDelta],
) -> tuple[StrategyArm, str]:
    effective = [item for item in items if item.status != "POSITIVE_SIGNAL"]
    evidence = [
        item
        for item in effective
        if item.severity in {"HIGH", "CRITICAL"}
        and (
            item.category == "EVIDENCE_SUFFICIENCY"
            or any(
                term in item.responsible_module.lower()
                for term in _EVIDENCE_MODULE_TERMS
            )
        )
    ]
    diversity = [
        item
        for item in effective
        if item.severity in {"HIGH", "CRITICAL"}
        and (
            item.category == "NOVELTY"
            or any(
                term in item.responsible_module.lower()
                for term in _DIVERSITY_MODULE_TERMS
            )
        )
    ]
    uncertainty = [
        item
        for item in effective
        if item.status == "QUERY"
        or any(
            term in f"{item.responsible_module} {item.problem}".lower()
            for term in _UNCERTAINTY_TERMS
        )
    ]
    ranked: tuple[tuple[StrategyArm, str, list[FeedbackDelta]], ...] = (
        ("EVIDENCE_FIRST", "HIGH/CRITICAL evidence issue", evidence),
        ("DIVERSITY_FIRST", "HIGH/CRITICAL innovation/diversity issue", diversity),
        ("UNCERTAINTY_FIRST", "QUERY or uncertainty issue", uncertainty),
    )
    for arm, label, candidates in ranked:
        if candidates:
            critical = sum(item.severity == "CRITICAL" for item in candidates)
            high = sum(item.severity == "HIGH" for item in candidates)
            first = min(items.index(item) for item in candidates)
            item_id = items[first].item_id or f"item_{first + 1:03d}"
            reason = (
                f"fixed-v1: {label}; critical={critical}; high={high}; "
                f"first={item_id}"
            )
            return arm, reason
    return "STATIC", "fixed-v1: no effective negative item"


def _bounded(value: str) -> str:
    safe = redact_text(value).strip()
    if len(safe) <= _MAX_ITEM_TEXT:
        return safe
    return safe[: _MAX_ITEM_TEXT - 1] + "…"


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


__all__ = ["build_revision_plan", "format_revision_instruction"]
