"""Bind compiled expert feedback to explicit, trusted QD step provenance."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Literal, cast

from photomatagent.scientific.evolution.models import (
    ExpertFeedbackRecord, FeedbackCompilation, RevisionPlan,
)
from photomatagent.scientific.evolution.qd.models import RevisionIssue


def issues_from_revision(
    plan: RevisionPlan,
    *,
    feedback: ExpertFeedbackRecord,
    compilation: FeedbackCompilation,
    module_steps: Mapping[str, tuple[str, ...]],
    source_result_sha256: str,
) -> tuple[RevisionIssue, ...]:
    """Create issues only with independently supplied source and step bindings.

    ``RevisionPlan`` is a lossy aggregate: it has neither the result digest nor
    per-item modules/status. The caller must supply the confirmed feedback,
    its compilation, the current source artifact digest, and a trusted mapping
    from module names to frozen Genome step IDs. No fallback step is inferred.
    """

    if not plan.confirmed:
        raise ValueError("REVISION_NOT_CONFIRMED")
    if plan.has_blocking_ambiguity or plan.unresolved_ambiguities:
        raise ValueError("REVISION_BLOCKED")
    if compilation.status != "AVAILABLE":
        raise ValueError("COMPILATION_UNAVAILABLE")
    identity = (plan.evolution_id, plan.feedback_id, plan.source_version)
    if identity != (feedback.evolution_id, feedback.feedback_id, feedback.episode_version):
        raise ValueError("REVISION_SOURCE_MISMATCH")
    if identity != (compilation.evolution_id, compilation.feedback_id, compilation.episode_version):
        raise ValueError("REVISION_SOURCE_MISMATCH")
    if not re.fullmatch(r"[0-9a-f]{64}", source_result_sha256) or source_result_sha256 != feedback.result_sha256:
        raise ValueError("SOURCE_RESULT_HASH_MISMATCH")

    actions_in_plan = set(plan.contract_changes) | set(plan.evidence_requirements) | set(plan.output_schema_requirements)
    preserves_in_plan = set(plan.preserved_facts)
    machine_ids = {check.acceptance_id for check in plan.machine_acceptance_checks}
    issues: list[RevisionIssue] = []
    seen: set[str] = set()
    for item in compilation.items:
        if item.item_id is None or item.item_id in seen:
            raise ValueError("FEEDBACK_ITEM_ID_INVALID")
        seen.add(item.item_id)
        if item.status != "POSITIVE_SIGNAL" and any(action.strip() not in actions_in_plan for action in item.requested_actions):
            raise ValueError("REVISION_COMPILATION_MISMATCH")
        if any(value.strip() not in preserves_in_plan for value in item.preserve):
            raise ValueError("REVISION_COMPILATION_MISMATCH")
        if item.acceptance_test and item.status != "QUERY" and item.item_id not in machine_ids:
            raise ValueError("REVISION_COMPILATION_MISMATCH")
        if item.status == "QUERY" and not any(test.startswith(f"QUERY {item.item_id}:") for test in plan.human_acceptance_tests):
            raise ValueError("REVISION_COMPILATION_MISMATCH")
        step_ids = tuple(module_steps.get(item.responsible_module, ()))
        if item.status != "POSITIVE_SIGNAL" and not step_ids:
            raise ValueError("MODULE_STEP_UNMAPPED")
        if len(step_ids) != len(set(step_ids)) or any(not step_id for step_id in step_ids):
            raise ValueError("MODULE_STEP_MAPPING_INVALID")
        issues.append(RevisionIssue(
            issue_id=f"{feedback.feedback_id}:{item.item_id}",
            source_kind="human", source_ref=feedback.feedback_id,
            artifact_sha256=feedback.result_sha256,
            status="resolved" if item.status == "POSITIVE_SIGNAL" else "open",
            severity=cast(Literal["low", "medium", "high", "critical"], item.severity.lower()), step_ids=step_ids,
            requested_actions=(
                () if item.status == "POSITIVE_SIGNAL" else
                (f"QUERY: {item.problem}", *item.requested_actions) if item.status == "QUERY" else
                tuple(item.requested_actions)
            ),
            preserve=tuple(item.preserve or ((item.problem,) if item.status == "POSITIVE_SIGNAL" else ())),
            acceptance_check_ids=(item.item_id,) if item.item_id in machine_ids and item.status != "QUERY" else (),
        ))
    return tuple(issues)
