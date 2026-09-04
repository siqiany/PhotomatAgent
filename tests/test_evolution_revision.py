from __future__ import annotations

from photomatagent.scientific.evolution.models import (
    ExpertFeedbackRecord,
    FeedbackCompilation,
    FeedbackDelta,
    RubricScores,
)
from photomatagent.scientific.evolution.revision import (
    build_revision_plan,
    format_revision_instruction,
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
    status: str,
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
