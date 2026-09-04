from __future__ import annotations

import pytest

from photomatagent.scientific.evolution.models import (
    ExpertFeedbackRecord,
    FeedbackCompilation,
    FeedbackDelta,
    FeedbackItemStatus,
    RevisionPlan,
    RubricScores,
)
from photomatagent.scientific.evolution.revision import (
    build_revision_plan,
    format_revision_instruction,
)
from photomatagent.scientific.loop import (
    ConstraintSpec,
    EvaluationReport,
    PropertyEvaluation,
    ScientificLoopSummary,
    TargetSpec,
)


def _feedback(*, raw_input: str = "RAW EXPERT PROSE 9f0b") -> ExpertFeedbackRecord:
    return ExpertFeedbackRecord(
        feedback_id="fb_revision_unit",
        evolution_id="evo_revision_unit",
        episode_version="v001",
        result_sha256="a" * 64,
        rubric_version="expert-review-v1",
        raw_input=raw_input,
        scores=RubricScores(
            scientific_correctness=3,
            evidence_sufficiency=2,
            novelty=3,
            actionability=3,
            overall=3,
        ),
    )


def _compilation(*items: FeedbackDelta) -> FeedbackCompilation:
    return FeedbackCompilation(
        compilation_id="comp_revision_unit",
        evolution_id="evo_revision_unit",
        feedback_id="fb_revision_unit",
        episode_version="v001",
        status="AVAILABLE",
        items=items,
        warnings=("compiler warning",),
        provider="fake",
        model="fake",
    )


def test_revision_plan_routes_evidence_query_without_promoting_it_to_fact() -> None:
    feedback = _feedback()
    compilation = _compilation(
        FeedbackDelta(
            item_id="item_001",
            category="EVIDENCE_SUFFICIENCY",
            status="QUERY",
            severity="HIGH",
            responsible_module="retrieval_planner",
            problem="Is abstract-only support sufficient?",
            requested_actions=("Read the full text",),
            acceptance_test="Bind each core claim to full-text evidence",
            preserve=(),
            confidence=0.9,
            source_span="expert source span that must not be rendered",
        )
    )

    plan = build_revision_plan(feedback=feedback, compilation=compilation)

    assert plan.evidence_requirements == ["Read the full text"]
    assert plan.machine_acceptance_tests == []
    assert plan.human_acceptance_tests == [
        "QUERY item_001: Bind each core claim to full-text evidence"
    ]
    assert plan.invalidated_conclusions == []
    assert plan.strategy_arm == "EVIDENCE_FIRST"
    assert plan.warnings == ["compiler warning"]


def test_revision_instruction_is_bounded_structured_and_excludes_raw_provenance() -> None:
    feedback = _feedback()
    compilation = _compilation(
        FeedbackDelta(
            item_id="item_001",
            category="DELIVERABLE_COMPLETENESS",
            status="CORRECTION",
            severity="MEDIUM",
            responsible_module="report_renderer",
            problem="Missing limitations section",
            requested_actions=("Add a limitations section",),
            acceptance_test="Report contains limitations",
            preserve=("Keep the verified table",),
            confidence=1.0,
            source_span="RAW SOURCE SPAN 16c0",
        )
    )
    plan = build_revision_plan(feedback=feedback, compilation=compilation)

    text = format_revision_instruction(plan, strategy="STATIC")

    assert feedback.raw_input not in text
    assert "RAW SOURCE SPAN 16c0" not in text
    assert "Revision requirements" in text
    assert "Do not override deterministic constraints" in text
    assert "Add a limitations section" in text
    assert "compiler warning" in text
    assert len(text) <= 16_000


def test_critical_item_without_action_or_test_creates_blocking_ambiguity() -> None:
    plan = build_revision_plan(
        feedback=_feedback(),
        compilation=_compilation(
            FeedbackDelta(
                item_id="item_007",
                category="SCIENTIFIC_CORRECTNESS",
                status="CORRECTION",
                severity="CRITICAL",
                responsible_module="scientific_checker",
                problem="The conclusion may be wrong",
                confidence=0.8,
                source_span="The conclusion may be wrong",
            )
        ),
    )

    assert plan.has_blocking_ambiguity is True
    assert plan.unresolved_ambiguities == [
        "CRITICAL item_007 needs a requested action or acceptance test"
    ]


def test_revision_plan_identity_and_routing_are_deterministic() -> None:
    feedback = _feedback()
    compilation = _compilation(
        FeedbackDelta(
            item_id="item_001",
            category="TASK_DEFINITION",
            status="CORRECTION",
            severity="HIGH",
            responsible_module="target_contract",
            problem="Wrong operating condition",
            requested_actions=("Use the declared temperature",),
            acceptance_test="Temperature equals target snapshot",
            preserve=("Keep the wavelength range",),
            confidence=0.95,
            source_span="Wrong operating condition",
        )
    )

    first = build_revision_plan(feedback=feedback, compilation=compilation)
    second = build_revision_plan(feedback=feedback, compilation=compilation)

    assert first == second
    assert first.revision_id.startswith("rp_")
    assert len(first.revision_id) == 13
    assert first.contract_changes == ["Use the declared temperature"]
    assert first.preserved_facts == ["Keep the wavelength range"]


def test_revision_planning_contract_is_exported_from_package() -> None:
    from photomatagent.scientific.evolution import (
        FixedStrategySelector,
        build_revision_plan as exported_build,
        format_revision_instruction as exported_format,
    )

    assert exported_build is build_revision_plan
    assert exported_format is format_revision_instruction
    assert FixedStrategySelector.__name__ == "FixedStrategySelector"


def _strategy_delta(
    item_id: str,
    *,
    category: str,
    status: FeedbackItemStatus,
    severity: str,
    module: str,
) -> FeedbackDelta:
    return FeedbackDelta(
        item_id=item_id,
        category=category,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        severity=severity,  # type: ignore[arg-type]
        responsible_module=module,
        problem=f"structured {item_id}",
        requested_actions=(f"action {item_id}",),
        acceptance_test=f"test {item_id}",
        confidence=1.0,
        source_span=f"source {item_id}",
    )


def test_fixed_strategy_priority_is_evidence_then_diversity_then_uncertainty() -> None:
    evidence = _strategy_delta(
        "item_001",
        category="EVIDENCE_SUFFICIENCY",
        status="CORRECTION",
        severity="HIGH",
        module="retrieval",
    )
    diversity = _strategy_delta(
        "item_002",
        category="NOVELTY",
        status="CORRECTION",
        severity="CRITICAL",
        module="candidate_generator",
    )
    query = _strategy_delta(
        "item_003",
        category="OTHER",
        status="QUERY",
        severity="CRITICAL",
        module="checker",
    )

    assert build_revision_plan(
        feedback=_feedback(), compilation=_compilation(evidence, diversity, query)
    ).strategy_arm == "EVIDENCE_FIRST"
    assert build_revision_plan(
        feedback=_feedback(), compilation=_compilation(diversity, query)
    ).strategy_arm == "DIVERSITY_FIRST"
    assert build_revision_plan(
        feedback=_feedback(), compilation=_compilation(query)
    ).strategy_arm == "UNCERTAINTY_FIRST"


def test_fixed_strategy_ignores_positive_signals_and_has_stable_tie_metadata() -> None:
    positive = _strategy_delta(
        "item_001",
        category="EVIDENCE_SUFFICIENCY",
        status="POSITIVE_SIGNAL",
        severity="CRITICAL",
        module="retrieval",
    )
    assert build_revision_plan(
        feedback=_feedback(), compilation=_compilation(positive)
    ).strategy_arm == "STATIC"

    high = _strategy_delta(
        "item_004",
        category="EVIDENCE_SUFFICIENCY",
        status="CORRECTION",
        severity="HIGH",
        module="retrieval",
    )
    critical = _strategy_delta(
        "item_005",
        category="EVIDENCE_SUFFICIENCY",
        status="CORRECTION",
        severity="CRITICAL",
        module="retrieval",
    )
    reason = build_revision_plan(
        feedback=_feedback(), compilation=_compilation(high, critical)
    ).strategy_reason
    assert reason.endswith("critical=1; high=1; first=item_004")


def test_revision_plan_uses_frozen_target_and_known_previous_evidence_ids() -> None:
    target = TargetSpec(
        goal="Preserve the LWIR target",
        constraints=[
            ConstraintSpec(
                property="band_gap",
                operator="le",
                value=0.2,
                unit="eV",
            )
        ],
    )
    summary = ScientificLoopSummary(
        status="INCONCLUSIVE",
        rounds=1,
        candidate_count=1,
        best_candidate_id="cand_1",
        best_score=0.2,
        final_evaluation=EvaluationReport(
            candidate_id="cand_1",
            constraint_results=[
                PropertyEvaluation(
                    property="band_gap",
                    result="PASS",
                    evidence_ids=["sev_known"],
                )
            ],
        ),
    )
    compilation = _compilation(
        FeedbackDelta(
            item_id="item_001",
            category="TASK_DEFINITION",
            status="CORRECTION",
            severity="HIGH",
            responsible_module="target_contract",
            problem="Operating condition needs correction",
            requested_actions=(
                " Preserve evidence_id:sev_known   and update temperature ",
                "Invalidate evidence_id:sev_known",
            ),
            acceptance_test="  Verify target band_gap contract  ",
            preserve=(" Keep evidence_id:sev_known ",),
            confidence=1.0,
            source_span="source",
        )
    )

    plan = build_revision_plan(
        feedback=_feedback(),
        compilation=compilation,
        target=target,
        previous_summary=summary,
    )

    assert plan.preserved_evidence_ids == ["sev_known"]
    assert plan.invalidated_evidence_ids == ["sev_known"]
    assert plan.contract_changes == [
        "Preserve evidence_id:sev_known and update temperature",
        "Invalidate evidence_id:sev_known",
    ]
    assert plan.machine_acceptance_tests == ["Verify target band_gap contract"]
    assert plan.has_blocking_ambiguity is True
    assert any(
        "preserved and invalidated" in value
        for value in plan.unresolved_ambiguities
    )

    other_target = target.model_copy(update={"goal": "Different immutable target"})
    other = build_revision_plan(
        feedback=_feedback(),
        compilation=compilation,
        target=other_target,
        previous_summary=summary,
    )
    assert other.revision_id != plan.revision_id


def test_unknown_explicit_evidence_reference_is_warned_and_never_added() -> None:
    summary = ScientificLoopSummary(
        status="INCONCLUSIVE",
        rounds=1,
        candidate_count=0,
        best_candidate_id=None,
        best_score=0.0,
        final_evaluation=None,
    )
    plan = build_revision_plan(
        feedback=_feedback(),
        compilation=_compilation(
            FeedbackDelta(
                item_id="item_009",
                category="EVIDENCE_SUFFICIENCY",
                status="CORRECTION",
                severity="HIGH",
                responsible_module="retrieval",
                problem="Invalidate evidence_id:sev_unknown",
                requested_actions=("Invalidate evidence_id:sev_unknown",),
                acceptance_test="Replacement evidence is present",
                preserve=("Keep evidence_id:sev_unknown",),
                confidence=1.0,
                source_span="source",
            )
        ),
        target=TargetSpec(goal="Target"),
        previous_summary=summary,
    )

    assert plan.preserved_evidence_ids == []
    assert plan.invalidated_evidence_ids == []
    assert any("Unknown evidence ID sev_unknown" in value for value in plan.warnings)


def _summary_with_known_evidence() -> ScientificLoopSummary:
    return ScientificLoopSummary(
        status="INCONCLUSIVE",
        rounds=1,
        candidate_count=1,
        best_candidate_id="cand_1",
        best_score=0.2,
        final_evaluation=EvaluationReport(
            candidate_id="cand_1",
            constraint_results=[
                PropertyEvaluation(
                    property="band_gap",
                    result="PASS",
                    evidence_ids=["sev_known"],
                )
            ],
        ),
    )


def _plan_for_invalidation_case(
    *,
    status: str,
    action: str,
) -> RevisionPlan:
    return build_revision_plan(
        feedback=_feedback(),
        compilation=_compilation(
            FeedbackDelta(
                item_id="item_invalidation",
                category="EVIDENCE_SUFFICIENCY",
                status=status,
                severity="HIGH",
                responsible_module="retrieval",
                problem="Review the evidence disposition",
                requested_actions=(action,),
                acceptance_test="Disposition is explicitly resolved",
                confidence=1.0,
                source_span="source",
            )
        ),
        previous_summary=_summary_with_known_evidence(),
    )


def test_query_evidence_reference_remains_non_authoritative() -> None:
    plan = _plan_for_invalidation_case(
        status="QUERY",
        action="Invalidate evidence_id:sev_known",
    )

    assert plan.invalidated_evidence_ids == []
    assert plan.evidence_requirements == ["Invalidate evidence_id:sev_known"]
    assert plan.human_acceptance_tests == [
        "QUERY item_invalidation: Disposition is explicitly resolved"
    ]


def test_positive_signal_evidence_reference_never_invalidates() -> None:
    plan = _plan_for_invalidation_case(
        status="POSITIVE_SIGNAL",
        action="Invalidate evidence_id:sev_known",
    )

    assert plan.invalidated_evidence_ids == []


def test_uncertain_invalidation_phrase_never_invalidates() -> None:
    plan = _plan_for_invalidation_case(
        status="CORRECTION",
        action="Uncertain whether to invalidate evidence_id:sev_known",
    )

    assert plan.invalidated_evidence_ids == []


@pytest.mark.parametrize(
    "action",
    [
        "Do not invalidate evidence_id:sev_known",
        "不要作废 evidence_id:sev_known",
        "不得删除 evidence_id:sev_known",
    ],
)
def test_negated_invalidation_phrase_never_invalidates(action: str) -> None:
    plan = _plan_for_invalidation_case(status="CORRECTION", action=action)

    assert plan.invalidated_evidence_ids == []


def test_explicit_correction_disposition_invalidates_known_evidence() -> None:
    plan = _plan_for_invalidation_case(
        status="CORRECTION",
        action="Invalidate evidence_id:sev_known",
    )

    assert plan.invalidated_evidence_ids == ["sev_known"]


def test_whitespace_only_critical_positive_signal_is_blocking() -> None:
    plan = build_revision_plan(
        feedback=_feedback(),
        compilation=_compilation(
            FeedbackDelta(
                item_id="item_010",
                category="OTHER",
                status="POSITIVE_SIGNAL",
                severity="CRITICAL",
                responsible_module="report",
                problem="Preserve a claimed strength",
                requested_actions=("   \n\t  ",),
                acceptance_test="  ",
                preserve=("  Keep   verified   section  ",),
                confidence=1.0,
                source_span="source",
            )
        ),
    )

    assert plan.preserved_facts == ["Keep verified section"]
    assert plan.has_blocking_ambiguity is True
    assert plan.unresolved_ambiguities == [
        "CRITICAL item_010 needs a requested action or acceptance test"
    ]
