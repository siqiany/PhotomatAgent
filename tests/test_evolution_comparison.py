from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from photomatagent.cli.app import app
from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.evolution.comparison import (
    compare_episodes,
    compute_learning_signal,
    evaluate_machine_acceptance,
)
from photomatagent.scientific.evolution.experience import (
    ExperienceEvidence,
    ExperiencePromotionError,
    ExperienceRecord,
    create_experience,
    derive_experience_id,
    promote_experience,
)
from photomatagent.scientific.evolution.models import (
    ArtifactRef,
    ComparisonReport,
    CostDelta,
    CostSnapshot,
    EpisodeRecord,
    EvolutionTask,
    ExpertFeedbackDraft,
    ExpertFeedbackRecord,
    FeedbackCompilation,
    FeedbackDelta,
    RevisionPlan,
    RubricScores,
)
from photomatagent.scientific.evolution.service import (
    ArtifactMismatchError,
    EvolutionOperationConflict,
    EvolutionService,
    InvalidEvolutionTransition,
)
from photomatagent.scientific.evolution.revision import build_revision_plan
from photomatagent.scientific.evolution.store import (
    EvolutionAlreadyExistsError,
    EvolutionStore,
)
from photomatagent.scientific.evolution.strategy import FixedStrategySelector
from photomatagent.scientific.loop import ScientificLoopSummary, TargetSpec
from photomatagent.scientific.loop.evaluation import (
    EvaluationReport,
    PropertyEvaluation,
)
from photomatagent.scientific.state import ScientificState
from photomatagent.workspace import Workspace


def _artifact(path: str, content: bytes) -> ArtifactRef:
    return ArtifactRef(
        path=path,
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _summary(
    *outcomes: tuple[str, str, str | None, tuple[str, ...]],
    gaps: tuple[str, ...] = (),
) -> ScientificLoopSummary:
    constraint_results = [
        PropertyEvaluation(
            property=name,
            result=result,  # type: ignore[arg-type]
            evidence_ids=list(evidence_ids),
        )
        for name, result, fidelity, evidence_ids in outcomes
    ]
    return ScientificLoopSummary(
        status="INCONCLUSIVE",
        rounds=1,
        candidate_count=1,
        best_candidate_id="candidate_1",
        best_score=0.5,
        final_evaluation=EvaluationReport(
            verdict="REVISE",
            score=0.5,
            constraint_results=constraint_results,
            evidence_gaps=list(gaps),
            critical_evidence_gaps=list(gaps),
        ),
        unresolved_evidence_gaps=list(gaps),
    )


def _episode(
    version: str,
    *,
    summary: ScientificLoopSummary | None = None,
    artifact: ArtifactRef | None = None,
    cost: CostSnapshot | None = None,
) -> EpisodeRecord:
    index = int(version[1:])
    return EpisodeRecord(
        evolution_id="evo_compare",
        episode_id=f"ep_{version}",
        version=version,  # type: ignore[arg-type]
        status="COMPLETED",
        parent_version=(f"v{index - 1:03d}" if index > 1 else None),
        revision_plan_id="rp_compare" if index > 1 else None,
        task_snapshot={"task_group_id": "group_compare"},
        target_snapshot=TargetSpec(goal="compare safely"),
        summary=summary,
        artifact=artifact,
        cost=cost or CostSnapshot(),
    )


def _feedback(version: str, value: int, *, feedback_id: str) -> ExpertFeedbackRecord:
    artifact_hash = "a" * 64 if version == "v001" else "b" * 64
    return ExpertFeedbackRecord(
        feedback_id=feedback_id,
        evolution_id="evo_compare",
        episode_version=version,  # type: ignore[arg-type]
        result_sha256=artifact_hash,
        rubric_version="expert-review-v1",
        raw_input="expert review",
        scores=RubricScores(
            scientific_correctness=value,
            evidence_sufficiency=value,
            novelty=value,
            actionability=value,
            overall=value,
        ),
    )


def _compilation(
    version: str,
    *,
    feedback_id: str,
    items: tuple[FeedbackDelta, ...] = (),
) -> FeedbackCompilation:
    return FeedbackCompilation(
        compilation_id=f"comp_{version}",
        evolution_id="evo_compare",
        feedback_id=feedback_id,
        episode_version=version,  # type: ignore[arg-type]
        status="AVAILABLE",
        items=items,
        provider="fake",
        model="fake",
    )


def _delta(
    item_id: str,
    *,
    category: str = "ACTIONABILITY",
    module: str = "process",
    status: str = "CORRECTION",
    severity: str = "HIGH",
    acceptance_test: str | None = None,
) -> FeedbackDelta:
    return FeedbackDelta(
        item_id=item_id,
        category=category,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        severity=severity,  # type: ignore[arg-type]
        responsible_module=module,
        problem=f"problem {item_id}",
        acceptance_test=acceptance_test,
        confidence=1.0,
        source_span="expert statement",
    )


def test_missing_repeated_comment_does_not_auto_close_human_check() -> None:
    previous = _episode("v001")
    current = _episode("v002")
    plan = RevisionPlan(
        revision_id="rp_compare",
        evolution_id="evo_compare",
        source_version="v001",
        feedback_id="fb_v1",
        human_acceptance_tests=["Expert confirms process is reproducible"],
        confirmed=True,
    )

    report = compare_episodes(
        previous=previous,
        current=current,
        previous_plan=plan,
        previous_feedback=_feedback("v001", 2, feedback_id="fb_v1"),
        current_feedback=None,
        previous_items=(_delta("issue_process", acceptance_test=None),),
    )

    assert report.acceptance_results[0].status == "NEEDS_HUMAN_REVIEW"
    assert report.closure_rate is None
    assert report.closed_issue_ids == []


def test_parent_child_comparison_allows_failed_version_gap() -> None:
    current = _episode("v003").model_copy(update={"parent_version": "v001"})

    report = compare_episodes(
        previous=_episode("v001"),
        current=current,
        previous_plan=RevisionPlan(
            revision_id="rp_compare",
            evolution_id="evo_compare",
            source_version="v001",
            feedback_id="fb_v1",
            confirmed=True,
        ),
    )

    assert report.previous_version == "v001"
    assert report.current_version == "v003"


def test_machine_acceptance_pass_closes_issue_and_fail_enters_denominator() -> None:
    plan = RevisionPlan(
        revision_id="rp_compare",
        evolution_id="evo_compare",
        source_version="v001",
        feedback_id="fb_v1",
        machine_acceptance_tests=["process_steps_complete", "safety_check"],
        confirmed=True,
    )
    report = compare_episodes(
        previous=_episode("v001"),
        current=_episode("v002"),
        previous_plan=plan,
        previous_feedback=_feedback("v001", 2, feedback_id="fb_v1"),
        current_feedback=None,
        previous_items=(
            _delta("issue_process", acceptance_test="process_steps_complete"),
            _delta(
                "issue_safety",
                category="SAFETY",
                module="safety",
                severity="CRITICAL",
                acceptance_test="safety_check",
            ),
        ),
        machine_results={"process_steps_complete": True, "safety_check": False},
    )

    assert report.closed_issue_ids == ["issue_process"]
    assert report.closure_rate == pytest.approx(0.5)
    assert [item.status for item in report.acceptance_results] == ["PASS", "FAIL"]


def test_scores_require_two_reviews_and_signature_issue_changes() -> None:
    prior_items = (
        _delta("old_evidence", category="EVIDENCE_SUFFICIENCY", module="retrieval"),
        _delta("old_process", category="ACTIONABILITY", module="process"),
    )
    current_items = (
        _delta("again_evidence", category="EVIDENCE_SUFFICIENCY", module="retrieval"),
        _delta("new_safety", category="SAFETY", module="safety"),
        _delta("new_safety_duplicate", category="SAFETY", module="safety"),
    )
    without_review = compare_episodes(
        previous=_episode("v001"),
        current=_episode("v002"),
        previous_plan=RevisionPlan(
            revision_id="rp_compare",
            evolution_id="evo_compare",
            source_version="v001",
            feedback_id="fb_v1",
            confirmed=True,
        ),
        previous_feedback=_feedback("v001", 2, feedback_id="fb_v1"),
        current_feedback=None,
        previous_items=prior_items,
    )
    current_feedback = _feedback("v002", 4, feedback_id="fb_v2")
    report = compare_episodes(
        previous=_episode("v001"),
        current=_episode("v002"),
        previous_plan=RevisionPlan(
            revision_id="rp_compare",
            evolution_id="evo_compare",
            source_version="v001",
            feedback_id="fb_v1",
            confirmed=True,
        ),
        previous_feedback=_feedback("v001", 2, feedback_id="fb_v1"),
        current_feedback=current_feedback,
        current_compilation=_compilation(
            "v002",
            feedback_id=current_feedback.feedback_id,
            items=current_items,
        ),
        previous_items=prior_items,
    )

    assert without_review.score_deltas == []
    assert [item.delta for item in report.score_deltas] == [2, 2, 2, 2, 2]
    assert report.recurring_issue_ids == ["again_evidence"]
    assert report.new_issue_ids == ["new_safety", "new_safety_duplicate"]
    assert report.recurrence_rate == pytest.approx(0.5)
    assert report.new_issue_rate == pytest.approx(0.5)


def test_comparison_reports_scientific_artifact_and_cost_changes() -> None:
    previous_content = b"previous"
    current_content = b"current report"
    previous = _episode(
        "v001",
        summary=_summary(
            ("band_gap", "FAIL", "analytical", ("sev_shared",)),
            ("responsivity", "UNKNOWN", None, ()),
            gaps=("responsivity",),
        ),
        artifact=_artifact("user_output/evo_compare/v001/result.md", previous_content),
        cost=CostSnapshot(
            input_tokens=100,
            output_tokens=50,
            tool_calls=2,
            runtime_seconds=10,
            hpc_cost=2,
        ),
    )
    current = _episode(
        "v002",
        summary=_summary(
            ("band_gap", "PASS", "dft", ("sev_shared",)),
            ("responsivity", "PASS", "experimental", ("sev_new",)),
            ("dark_current", "UNKNOWN", None, ()),
            gaps=("dark_current",),
        ),
        artifact=_artifact("user_output/evo_compare/v002/result.md", current_content),
        cost=CostSnapshot(
            input_tokens=150,
            output_tokens=40,
            tool_calls=3,
            runtime_seconds=20,
            hpc_cost=3,
        ),
    )

    report = compare_episodes(
        previous=previous,
        current=current,
        previous_plan=RevisionPlan(
            revision_id="rp_compare",
            evolution_id="evo_compare",
            source_version="v001",
            feedback_id="fb_v1",
            invalidated_evidence_ids=["sev_removed"],
            confirmed=True,
        ),
        previous_state=ScientificState(
            evidence=[
                ScientificEvidence(
                    id="sev_shared",
                    subject="InAs",
                    property="band_gap",
                    fidelity="analytical",
                ),
                ScientificEvidence(
                    id="sev_removed",
                    subject="InAs",
                    property="stability",
                    fidelity="empirical",
                ),
            ]
        ),
        current_state=ScientificState(
            evidence=[
                ScientificEvidence(
                    id="sev_shared",
                    subject="InAs",
                    property="band_gap",
                    fidelity="dft",
                ),
                ScientificEvidence(
                    id="sev_new",
                    subject="InAs",
                    property="responsivity",
                    fidelity="experimental",
                ),
            ]
        ),
    )

    assert report.constraint_changes.newly_passed == ["band_gap", "responsivity"]
    assert report.constraint_changes.newly_unknown == ["dark_current"]
    assert report.evidence_changes.added_ids == ["sev_new"]
    assert report.evidence_changes.removed_ids == ["sev_removed"]
    assert report.evidence_changes.carried_ids == ["sev_shared"]
    assert report.evidence_changes.invalidated_ids == ["sev_removed"]
    assert report.evidence_changes.resolved_gaps == ["responsivity"]
    assert report.evidence_changes.new_gaps == ["dark_current"]
    assert report.fidelity_changes.upgraded_ids == ["sev_shared"]
    assert report.artifact_diff is not None and report.artifact_diff.changed
    assert report.artifact_diff.size_bytes_delta == (
        len(current_content) - len(previous_content)
    )
    assert report.cost_delta.input_tokens == 50
    assert report.cost_delta.output_tokens == -10
    assert report.cost_delta.hpc_cost == pytest.approx(1.0)


def test_learning_signal_renormalizes_missing_components_and_records_them() -> None:
    reward, used = compute_learning_signal(
        expert_utility_delta=0.5,
        closure_rate=None,
        recurrence_rate=0.25,
        new_issue_rate=None,
        normalized_cost_increase=0.5,
    )

    # (0.45*0.5 - 0.15*0.25 - 0.05*0.5) / (0.45+0.15+0.05)
    assert reward == pytest.approx(0.25)
    assert used == [
        "expert_utility_delta",
        "recurrence_rate",
        "normalized_cost_increase",
    ]


def test_comparison_report_rejects_unbounded_or_inconsistent_learning_fields() -> None:
    base = {
        "comparison_id": "cmp_bounded",
        "evolution_id": "evo_compare",
        "previous_version": "v001",
        "current_version": "v002",
    }
    with pytest.raises(ValidationError, match="module credit"):
        ComparisonReport(**base, module_credit={"planner": 1.1})
    with pytest.raises(ValidationError, match="reward components"):
        ComparisonReport(
            **base,
            expert_utility_delta=0.5,
            reward=0.5,
            components_used=["expert_utility_delta", "expert_utility_delta"],
        )
    with pytest.raises(ValidationError, match="reward components"):
        ComparisonReport(
            **base,
            expert_utility_delta=0.5,
            closure_rate=1.0,
            reward=0.678571,
            components_used=["closure_rate", "expert_utility_delta"],
            acceptance_results=[
                {
                    "acceptance_id": "issue_closed",
                    "status": "PASS",
                }
            ],
        )
    report = ComparisonReport(
        **base,
        expert_utility_delta=0.5,
        reward=0.5,
        components_used=["expert_utility_delta"],
        module_credit={"planner": 0.5},
    )
    with pytest.raises(ValidationError):
        report.reward = 0.2
    with pytest.raises(TypeError):
        report.module_credit["planner"] = 0.2


def test_module_credit_is_bounded_and_does_not_create_extra_observations() -> None:
    current_feedback = _feedback("v002", 3, feedback_id="fb_v2")
    current_items = (
        _delta(
            "retrieval_again",
            category="EVIDENCE_SUFFICIENCY",
            module="retrieval",
        ),
        _delta("novelty_new", category="NOVELTY", module="search"),
    )
    report = compare_episodes(
        previous=_episode("v001"),
        current=_episode("v002"),
        previous_plan=RevisionPlan(
            revision_id="rp_compare",
            evolution_id="evo_compare",
            source_version="v001",
            feedback_id="fb_v1",
            confirmed=True,
        ),
        previous_items=(
            _delta("retrieval_1", category="EVIDENCE_SUFFICIENCY", module="retrieval"),
            _delta("retrieval_2", category="EVIDENCE_SUFFICIENCY", module="retrieval"),
        ),
        current_feedback=current_feedback,
        current_compilation=_compilation(
            "v002",
            feedback_id=current_feedback.feedback_id,
            items=current_items,
        ),
    )
    experience = create_experience(report, task_group_id="group_compare")

    assert set(report.module_credit) == {"retrieval", "search"}
    assert all(-1.0 <= value <= 1.0 for value in report.module_credit.values())
    assert len(experience.observations) == 1


def test_later_positive_signal_explicitly_closes_human_issue() -> None:
    current_feedback = _feedback("v002", 3, feedback_id="fb_v2").model_copy(
        update={"resolved_issue_ids": ["issue_evidence"]}
    )
    current_items = (
        _delta(
            "confirmation",
            category="EVIDENCE_SUFFICIENCY",
            module="retrieval",
            status="POSITIVE_SIGNAL",
        ),
    )
    report = compare_episodes(
        previous=_episode("v001"),
        current=_episode("v002"),
        previous_plan=RevisionPlan(
            revision_id="rp_compare",
            evolution_id="evo_compare",
            source_version="v001",
            feedback_id="fb_v1",
            human_acceptance_tests=["Expert confirms evidence chain"],
            confirmed=True,
        ),
        previous_items=(
            _delta(
                "issue_evidence",
                category="EVIDENCE_SUFFICIENCY",
                module="retrieval",
            ),
        ),
        current_feedback=current_feedback,
        current_compilation=_compilation(
            "v002",
            feedback_id=current_feedback.feedback_id,
            items=current_items,
        ),
    )

    assert report.closed_issue_ids == ["issue_evidence"]
    assert report.closure_rate is None
    assert report.acceptance_results[0].status == "PASS"
    assert report.acceptance_results[0].kind == "HUMAN"


def test_unrelated_positive_signal_never_human_closes_issue() -> None:
    current_feedback = _feedback("v002", 3, feedback_id="fb_v2")
    current_items = (
        _delta(
            "praise_only",
            category="EVIDENCE_SUFFICIENCY",
            module="retrieval",
            status="POSITIVE_SIGNAL",
        ),
    )
    report = compare_episodes(
        previous=_episode("v001"),
        current=_episode("v002"),
        previous_plan=RevisionPlan(
            revision_id="rp_compare",
            evolution_id="evo_compare",
            source_version="v001",
            feedback_id="fb_v1",
            human_acceptance_tests=["Expert confirms evidence chain"],
            confirmed=True,
        ),
        previous_items=(
            _delta("issue_evidence", category="EVIDENCE_SUFFICIENCY", module="retrieval"),
        ),
        current_feedback=current_feedback,
        current_compilation=_compilation(
            "v002",
            feedback_id=current_feedback.feedback_id,
            items=current_items,
        ),
    )

    assert report.closed_issue_ids == []
    assert report.acceptance_results[0].status == "NEEDS_HUMAN_REVIEW"


def test_human_check_uses_its_exact_query_issue_reference() -> None:
    current_feedback = _feedback("v002", 3, feedback_id="fb_v2").model_copy(
        update={"resolved_issue_ids": ["issue_query"]}
    )
    report = compare_episodes(
        previous=_episode("v001"),
        current=_episode("v002"),
        previous_plan=RevisionPlan(
            revision_id="rp_compare",
            evolution_id="evo_compare",
            source_version="v001",
            feedback_id="fb_v1",
            human_acceptance_tests=["QUERY issue_query: Expert confirms the answer"],
            confirmed=True,
        ),
        previous_items=(
            _delta("issue_unrelated"),
            _delta("issue_query", status="QUERY"),
        ),
        current_feedback=current_feedback,
        current_compilation=_compilation(
            "v002",
            feedback_id=current_feedback.feedback_id,
        ),
    )

    assert report.closed_issue_ids == ["issue_query"]
    assert report.acceptance_results[0].acceptance_id == "issue_query"
    assert report.acceptance_results[0].status == "PASS"


def test_module_credit_redacts_sensitive_module_names() -> None:
    current_feedback = _feedback("v002", 3, feedback_id="fb_v2")
    current_items = (
        _delta(
            "unsafe_module_name",
            module="Authorization: Bearer module-secret",
        ),
    )
    report = compare_episodes(
        previous=_episode("v001"),
        current=_episode("v002"),
        previous_plan=RevisionPlan(
            revision_id="rp_compare",
            evolution_id="evo_compare",
            source_version="v001",
            feedback_id="fb_v1",
            confirmed=True,
        ),
        current_feedback=current_feedback,
        current_compilation=_compilation(
            "v002",
            feedback_id=current_feedback.feedback_id,
            items=current_items,
        ),
    )

    assert "module-secret" not in str(report.module_credit)
    assert "[REDACTED]" in report.module_credit


def _experience_evidence(
    index: int,
    reward: float,
    *,
    unsafe: bool = False,
) -> ExperienceEvidence:
    return ExperienceEvidence(
        comparison_id=f"cmp_{index}",
        task_group_id=f"task_{index}",
        reward=reward,
        safety_or_fabrication_failure=unsafe,
    )


def test_experience_maturity_uses_distinct_tasks_safety_and_user_approval() -> None:
    comparison = ComparisonReport(
        comparison_id="cmp_seed",
        evolution_id="evo_compare",
        previous_version="v001",
        current_version="v002",
        expert_utility_delta=0.5,
        reward=0.5,
        components_used=["expert_utility_delta"],
    )
    observation = create_experience(comparison, task_group_id="task_0")
    assert observation.maturity == "OBSERVATION"
    assert observation.evolution_id == comparison.evolution_id

    with pytest.raises(ExperiencePromotionError, match="distinct task"):
        promote_experience(
            observation,
            to="HYPOTHESIS",
            evidence=[
                ExperienceEvidence(
                    comparison_id="cmp_same_task",
                    task_group_id="task_0",
                    reward=0.25,
                )
            ],
        )

    hypothesis = promote_experience(
        observation,
        to="HYPOTHESIS",
        evidence=[_experience_evidence(1, 0.25)],
    )
    assert hypothesis.maturity == "HYPOTHESIS"

    validation_evidence = [_experience_evidence(i, 0.2) for i in range(2, 6)]
    with pytest.raises(ExperiencePromotionError, match="maturity transition"):
        promote_experience(
            observation,
            to="VALIDATED_EXPERIENCE",
            evidence=validation_evidence,
        )
    validated = promote_experience(
        hypothesis,
        to="VALIDATED_EXPERIENCE",
        evidence=validation_evidence,
    )
    assert validated.maturity == "VALIDATED_EXPERIENCE"

    with pytest.raises(ExperiencePromotionError, match="user approval"):
        promote_experience(
            validated,
            to="REUSABLE_SKILL",
            evidence=[],
        )
    reusable = promote_experience(
        validated,
        to="REUSABLE_SKILL",
        evidence=[],
        user_approved=True,
    )
    assert reusable.maturity == "REUSABLE_SKILL"
    assert reusable.user_approved_for_reuse

    with pytest.raises(ExperiencePromotionError, match="safety"):
        promote_experience(
            hypothesis,
            to="VALIDATED_EXPERIENCE",
            evidence=[
                *validation_evidence[:3],
                _experience_evidence(5, 0.2, unsafe=True),
            ],
        )


def test_experience_model_rejects_forged_id_and_bypassed_maturity() -> None:
    observation = _experience_evidence(1, 0.5)
    with pytest.raises(ValidationError, match="experience_id"):
        ExperienceRecord(
            experience_id="exp_forged",
            evolution_id="evo_compare",
            maturity="OBSERVATION",
            observations=(observation,),
            created_at=_episode("v001").created_at,
        )
    with pytest.raises(ValidationError, match="HYPOTHESIS"):
        ExperienceRecord(
            experience_id="exp_forged",
            evolution_id="evo_compare",
            base_experience_id="exp_parent",
            previous_maturity="OBSERVATION",
            maturity="HYPOTHESIS",
            observations=(observation,),
            created_at=_episode("v001").created_at,
        )


def test_grouped_rewards_are_averaged_once_per_task_group() -> None:
    comparison = ComparisonReport(
        comparison_id="cmp_seed",
        evolution_id="evo_compare",
        previous_version="v001",
        current_version="v002",
        expert_utility_delta=0.1,
        reward=0.1,
        components_used=["expert_utility_delta"],
    )
    observation = create_experience(comparison, task_group_id="task_0")
    hypothesis = promote_experience(
        observation,
        to="HYPOTHESIS",
        evidence=[_experience_evidence(1, 0.1)],
    )
    # Duplicate positive samples in one group receive only one group-level vote.
    grouped = [
        ExperienceEvidence(
            comparison_id=f"cmp_group_a_{index}",
            task_group_id="task_0",
            reward=1.0,
        )
        for index in range(4)
    ]
    grouped.extend(
        ExperienceEvidence(
            comparison_id=f"cmp_group_{index}",
            task_group_id=f"task_{index}",
            reward=-0.3,
        )
        for index in range(2, 6)
    )
    with pytest.raises(ExperiencePromotionError, match="positive average"):
        promote_experience(
            hypothesis,
            to="VALIDATED_EXPERIENCE",
            evidence=grouped,
        )


def test_store_revalidates_maturity_when_model_construction_is_bypassed(
    tmp_path: Path,
) -> None:
    service, task = _persisted_pair(tmp_path)
    comparison = ComparisonReport(
        comparison_id="cmp_direct",
        evolution_id=task.evolution_id,
        previous_version="v001",
        current_version="v002",
        expert_utility_delta=0.5,
        reward=0.5,
        components_used=["expert_utility_delta"],
    )
    observation = create_experience(comparison, task_group_id="only_group")
    service.store.write_experience(observation)
    observations = observation.observations
    forged = ExperienceRecord.model_construct(
        schema_version=1,
        experience_id=derive_experience_id(
            evolution_id=task.evolution_id,
            maturity="HYPOTHESIS",
            observations=observations,
            base_experience_id=observation.experience_id,
            previous_maturity="OBSERVATION",
            created_at=observation.created_at,
        ),
        evolution_id=task.evolution_id,
        base_experience_id=observation.experience_id,
        previous_maturity="OBSERVATION",
        maturity="HYPOTHESIS",
        observations=observations,
        user_approved_for_reuse=False,
        created_at=observation.created_at,
    )

    with pytest.raises(ValidationError, match="HYPOTHESIS"):
        service.store.write_experience(forged)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_store_revalidates_non_finite_comparison_cost(
    tmp_path: Path,
    value: float,
) -> None:
    service, task = _persisted_pair(tmp_path)
    valid = ComparisonReport(
        comparison_id="cmp_nonfinite",
        evolution_id=task.evolution_id,
        previous_version="v001",
        current_version="v002",
    )
    forged = valid.model_copy(
        update={
            "cost_delta": CostDelta.model_construct(
                input_tokens=0,
                output_tokens=0,
                tool_calls=0,
                runtime_seconds=value,
                hpc_cost=None,
            )
        }
    )

    with pytest.raises(ValidationError):
        service.store.write_comparison(forged)


def _persisted_pair(
    tmp_path: Path,
    *,
    forge_plan: bool = False,
    current_overrides: dict[str, object] | None = None,
) -> tuple[EvolutionService, EvolutionTask]:
    workspace = Workspace(tmp_path)
    store = EvolutionStore(workspace)
    service = EvolutionService(store)
    target = TargetSpec(goal="compare safely")
    previous_content = b"previous"
    previous_ref = _artifact(
        "user_output/evo_compare/v001/result.md",
        previous_content,
    )
    feedback = _feedback("v001", 2, feedback_id="fb_v1").model_copy(
        update={"result_sha256": previous_ref.sha256}
    )
    compilation = FeedbackCompilation(
        compilation_id="comp_v1",
        evolution_id="evo_compare",
        feedback_id="fb_v1",
        episode_version="v001",
        status="AVAILABLE",
        items=(
            _delta(
                "issue_secret",
                module="Authorization_Bearer_cli-secret",
                acceptance_test="machine_check",
            ),
        ),
        provider="fake",
        model="fake",
    )
    plan = build_revision_plan(
        feedback=feedback,
        compilation=compilation,
        target=target,
        previous_summary=None,
    ).model_copy(update={"confirmed": True})
    if forge_plan:
        plan = plan.model_copy(update={"contract_changes": ["forged comparison input"]})
    task = EvolutionTask(
        evolution_id="evo_compare",
        goal="compare safely",
        target=target,
        task_group_id="group_compare",
        input_sha256="c" * 64,
        status="AWAITING_EXPERT_FEEDBACK",
        current_version="v002",
        last_completed_version="v002",
        episode_ids=[
            EvolutionService._episode_id("evo_compare", "v001"),
            EvolutionService._episode_id("evo_compare", "v002"),
        ],
        feedback_ids=["fb_v1"],
        compilation_ids=["comp_v1"],
        revision_ids=[plan.revision_id],
    )
    strategy = FixedStrategySelector().select(task, plan)
    task = task.model_copy(update={"strategy_ids": [strategy.strategy_id]})
    store.create_task(task)
    for version, content in (("v001", previous_content), ("v002", b"current")):
        relative = f"user_output/evo_compare/{version}/result.md"
        path = workspace.resolve(relative, must_exist=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        state_path = store.write_scientific_state(
            "evo_compare",
            version,
            ScientificState(),
        )
        episode_updates: dict[str, object] = {
            "scientific_state_path": workspace.relative(state_path),
            "task_snapshot": task.model_dump(mode="json"),
            "episode_id": EvolutionService._episode_id(
                "evo_compare", version  # type: ignore[arg-type]
            ),
            "revision_plan_id": plan.revision_id if version == "v002" else None,
            "applied_feedback_id": plan.feedback_id if version == "v002" else None,
            "strategy_id": strategy.strategy_id if version == "v002" else None,
            "strategy_arm": strategy.arm if version == "v002" else "STATIC",
        }
        if version == "v002" and current_overrides:
            episode_updates.update(current_overrides)
        persisted_episode = _episode(
            version,
            artifact=_artifact(relative, content),
        ).model_copy(update=episode_updates)
        if version == "v002" and "acceptance_results" not in (
            current_overrides or {}
        ):
            persisted_episode = persisted_episode.model_copy(
                update={
                    "acceptance_results": evaluate_machine_acceptance(
                        plan=plan,
                        episode=persisted_episode,
                        state=ScientificState(),
                    )
                }
            )
        store.write_episode(
            persisted_episode
        )
    store.write_feedback(feedback)
    store.write_compilation(compilation)
    store.write_revision(plan)
    store.write_strategy(strategy)
    return service, task


def test_service_comparison_is_immutable_and_idempotent(tmp_path: Path) -> None:
    service, task = _persisted_pair(tmp_path)

    first = service.compare(task.evolution_id, "v001", "v002")
    second = service.compare(task.evolution_id, "v001", "v002")

    assert second.entity == first.entity
    stored_task = service.get(task.evolution_id)
    assert stored_task.comparison_ids == [first.entity.comparison_id]
    assert first.entity.phase == "PRE_FEEDBACK"
    assert stored_task.experience_ids == []
    assert service.store.load_comparison(
        task.evolution_id, first.entity.comparison_id
    ) == first.entity
    with pytest.raises(EvolutionAlreadyExistsError):
        service.store.write_comparison(first.entity)


def test_preliminary_then_reviewed_comparison_has_distinct_snapshot_and_one_sample(
    tmp_path: Path,
) -> None:
    service, task = _persisted_pair(tmp_path)
    preliminary = service.compare(task.evolution_id, "v001", "v002").entity
    current = service.store.load_episode(task.evolution_id, "v002")
    assert current.artifact is not None
    service.attach_feedback(
        task.evolution_id,
        "v002",
        feedback_id="fb_v2",
        draft=ExpertFeedbackDraft(
            scores=RubricScores(
                scientific_correctness=4,
                evidence_sufficiency=4,
                novelty=4,
                actionability=4,
                overall=4,
            ),
            resolved_issue_ids=["issue_secret"],
        ),
        result_sha256=current.artifact.sha256,
        raw_input="reviewed",
    )

    with pytest.raises(InvalidEvolutionTransition, match="compile.*current feedback"):
        service.compare(task.evolution_id, "v001", "v002")
    incomplete = service.get(task.evolution_id)
    assert incomplete.comparison_ids == [preliminary.comparison_id]
    assert incomplete.experience_ids == []

    current_feedback = service.store.load_feedback(task.evolution_id, "fb_v2")
    compilation = _compilation(
        "v002",
        feedback_id=current_feedback.feedback_id,
    )
    first_compilation = service.save_compilation(task.evolution_id, compilation)
    retried_compilation = service.save_compilation(task.evolution_id, compilation)
    assert retried_compilation.entity == first_compilation.entity

    reviewed = service.compare(task.evolution_id, "v001", "v002").entity
    retried = service.compare(task.evolution_id, "v001", "v002").entity

    assert preliminary.phase == "PRE_FEEDBACK"
    assert reviewed.phase == "POST_FEEDBACK"
    assert preliminary.comparison_id != reviewed.comparison_id
    assert reviewed.current_feedback_id == "fb_v2"
    assert reviewed.current_compilation_id == "comp_v002"
    assert reviewed.current_compilation_sha256 is not None
    assert retried == reviewed
    stored = service.get(task.evolution_id)
    assert stored.comparison_ids == [
        preliminary.comparison_id,
        reviewed.comparison_id,
    ]
    assert len(stored.experience_ids) == 1


def test_service_revalidates_physical_artifacts(tmp_path: Path) -> None:
    service, task = _persisted_pair(tmp_path)
    current = service.store.load_episode(task.evolution_id, "v002")
    assert current.artifact is not None
    service.store.workspace.resolve(current.artifact.path).write_text(
        "tampered", encoding="utf-8"
    )

    with pytest.raises(ArtifactMismatchError):
        service.compare(task.evolution_id, "v001", "v002")

    assert service.get(task.evolution_id).comparison_ids == []


def test_service_rebuilds_canonical_revision_before_comparing(tmp_path: Path) -> None:
    service, task = _persisted_pair(tmp_path, forge_plan=True)

    with pytest.raises(EvolutionOperationConflict, match="canonical"):
        service.compare(task.evolution_id, "v001", "v002")

    assert service.get(task.evolution_id).comparison_ids == []


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"applied_feedback_id": "fb_wrong"}, "feedback"),
        ({"revision_plan_id": "rp_wrong"}, "revision plan"),
        ({"strategy_id": "strategy_wrong"}, "strategy"),
        ({"strategy_arm": "DIVERSITY_FIRST"}, "strategy"),
        ({"parent_version": "v000"}, "parent/child"),
        (
            {
                "acceptance_results": [
                    {
                        "acceptance_id": "issue_secret",
                        "status": "PASS",
                    }
                ]
            },
            "acceptance",
        ),
    ],
)
def test_service_rejects_each_tampered_comparison_provenance_link(
    tmp_path: Path,
    overrides: dict[str, object],
    match: str,
) -> None:
    service, task = _persisted_pair(tmp_path, current_overrides=overrides)

    with pytest.raises(
        (EvolutionOperationConflict, InvalidEvolutionTransition), match=match
    ):
        service.compare(task.evolution_id, "v001", "v002")

    assert service.get(task.evolution_id).comparison_ids == []


def test_compare_cli_is_bounded_redacted_and_does_not_construct_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, task = _persisted_pair(tmp_path)
    monkeypatch.setattr(
        "photomatagent.cli.chat.build_runtime",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError(f"compare constructed runtime: {kwargs}")
        ),
    )

    result = CliRunner().invoke(
        app,
        [
            "evolve",
            "compare",
            task.evolution_id,
            "v001",
            "v002",
            "--workspace",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "v001" in result.output and "v002" in result.output
    assert "NEEDS_HUMAN_REVIEW" in result.output
    assert "Primary artifact delta" in result.output
    assert "Previous SHA-256" in result.output
    assert "Input tokens" in result.output
    assert "Unresolved human checks" in result.output
    assert "cli-secret" not in result.output
    assert len(service.get(task.evolution_id).comparison_ids) == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_compare_cli_rejects_non_finite_persisted_episode_cost(
    tmp_path: Path,
    value: float,
) -> None:
    service, task = _persisted_pair(tmp_path)
    episode_path = service.store.workspace.resolve(
        f".photomatagent/evolutions/{task.evolution_id}/episodes/v002.json"
    )
    payload = json.loads(episode_path.read_text(encoding="utf-8"))
    payload["cost"]["runtime_seconds"] = value
    episode_path.write_text(json.dumps(payload), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "evolve",
            "compare",
            task.evolution_id,
            "v001",
            "v002",
            "--workspace",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 2
    assert "corrupt evolution record" in result.output
    assert len(result.output) <= 1_200
