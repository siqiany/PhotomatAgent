from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from photomatagent.cli.app import app
from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.evolution.comparison import (
    compare_episodes,
    compute_learning_signal,
)
from photomatagent.scientific.evolution.experience import (
    ExperienceEvidence,
    ExperiencePromotionError,
    create_experience,
    promote_experience,
)
from photomatagent.scientific.evolution.models import (
    ArtifactRef,
    ComparisonReport,
    CostSnapshot,
    EpisodeRecord,
    EvolutionTask,
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
)
from photomatagent.scientific.evolution.revision import build_revision_plan
from photomatagent.scientific.evolution.store import (
    EvolutionAlreadyExistsError,
    EvolutionStore,
)
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
        current_feedback=_feedback("v002", 4, feedback_id="fb_v2"),
        previous_items=prior_items,
        current_items=current_items,
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


def test_module_credit_is_bounded_and_does_not_create_extra_observations() -> None:
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
        current_items=(
            _delta(
                "retrieval_again",
                category="EVIDENCE_SUFFICIENCY",
                module="retrieval",
            ),
            _delta("novelty_new", category="NOVELTY", module="search"),
        ),
    )
    experience = create_experience(report, task_group_id="group_compare")

    assert set(report.module_credit) == {"retrieval", "search"}
    assert all(-1.0 <= value <= 1.0 for value in report.module_credit.values())
    assert len(experience.observations) == 1


def test_later_positive_signal_explicitly_closes_human_issue() -> None:
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
        current_items=(
            _delta(
                "confirmation",
                category="EVIDENCE_SUFFICIENCY",
                module="retrieval",
                status="POSITIVE_SIGNAL",
            ),
        ),
    )

    assert report.closed_issue_ids == ["issue_evidence"]
    assert report.closure_rate == pytest.approx(1.0)
    assert report.acceptance_results[0].status == "PASS"


def test_module_credit_redacts_sensitive_module_names() -> None:
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
        current_items=(
            _delta(
                "unsafe_module_name",
                module="Authorization: Bearer module-secret",
            ),
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
        reward=0.5,
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


def _persisted_pair(
    tmp_path: Path,
    *,
    forge_plan: bool = False,
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
        store.write_episode(
            _episode(version, artifact=_artifact(relative, content)).model_copy(
                update={
                    "scientific_state_path": workspace.relative(state_path),
                    "task_snapshot": task.model_dump(mode="json"),
                    "episode_id": EvolutionService._episode_id(
                        "evo_compare", version  # type: ignore[arg-type]
                    ),
                    "revision_plan_id": (
                        plan.revision_id if version == "v002" else None
                    ),
                }
            )
        )
    store.write_feedback(feedback)
    store.write_compilation(compilation)
    store.write_revision(plan)
    return service, task


def test_service_comparison_is_immutable_and_idempotent(tmp_path: Path) -> None:
    service, task = _persisted_pair(tmp_path)

    first = service.compare(task.evolution_id, "v001", "v002")
    second = service.compare(task.evolution_id, "v001", "v002")

    assert second.entity == first.entity
    stored_task = service.get(task.evolution_id)
    assert stored_task.comparison_ids == [first.entity.comparison_id]
    assert len(stored_task.experience_ids) == 1
    assert service.store.load_comparison(
        task.evolution_id, first.entity.comparison_id
    ) == first.entity
    experience = service.store.load_experience(
        task.evolution_id, stored_task.experience_ids[0]
    )
    assert len(experience.observations) == 1
    with pytest.raises(EvolutionAlreadyExistsError):
        service.store.write_comparison(first.entity)


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
    assert "cli-secret" not in result.output
    assert len(service.get(task.evolution_id).comparison_ids) == 1
