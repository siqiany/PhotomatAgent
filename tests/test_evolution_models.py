from datetime import UTC, datetime, timedelta, timezone
import json
import re

from pydantic import ValidationError
import pytest

from photomatagent.scientific.evolution.models import (
    ArtifactDiff,
    ArtifactRef,
    ComparisonReport,
    ConstraintChangeSummary,
    CostDelta,
    CostSnapshot,
    EpisodeRecord,
    EvidenceChangeSummary,
    EvolutionTask,
    ExpertFeedbackDraft,
    ExpertFeedbackRecord,
    FeedbackCompilation,
    FidelityChangeSummary,
    RevisionPlan,
    RubricScoreDelta,
    RubricFlags,
    RubricScores,
    StrategyVersion,
    new_episode_id,
    new_evolution_id,
    new_feedback_id,
    new_revision_id,
    new_strategy_id,
    validate_managed_id,
)
from photomatagent.scientific.evolution.rubric import assess_hard_caps, expert_utility
from photomatagent.scientific.loop import ConstraintSpec, ScientificLoopSummary, TargetSpec


def _scores(value: int = 5) -> RubricScores:
    return RubricScores(
        scientific_correctness=value,
        evidence_sufficiency=value,
        novelty=value,
        actionability=value,
        overall=value,
    )


def _target() -> TargetSpec:
    return TargetSpec(goal="goal", metadata={"nested": {"value": 1}})


def _task(**updates: object) -> EvolutionTask:
    data: dict[str, object] = {
        "evolution_id": "evo_test",
        "goal": "goal",
        "target": _target(),
        "task_group_id": "task_group_1",
        "input_sha256": "a" * 64,
    }
    data.update(updates)
    return EvolutionTask.model_validate(data)


def _episode(**updates: object) -> EpisodeRecord:
    data: dict[str, object] = {
        "evolution_id": "evo_test",
        "episode_id": "ep_test",
        "version": "v001",
        "task_snapshot": {"nested": {"value": 1}},
        "target_snapshot": _target(),
        "provider": "fake",
        "model": "fake-model",
        "tool_surface_fingerprint": "c" * 64,
        "capability_fingerprint": "d" * 64,
        "data_source_fingerprints": {"catalog": "e" * 64},
    }
    data.update(updates)
    return EpisodeRecord.model_validate(data)


def test_feedback_scores_are_bounded_integers():
    with pytest.raises(ValidationError):
        RubricScores(
            scientific_correctness=6,
            evidence_sufficiency=3,
            novelty=3,
            actionability=3,
            overall=3,
        )


@pytest.mark.parametrize(
    "field",
    [
        "scientific_correctness",
        "evidence_sufficiency",
        "novelty",
        "actionability",
        "overall",
    ],
)
@pytest.mark.parametrize("value", [0, 6])
def test_every_feedback_score_enforces_both_bounds(field, value):
    values = _scores(3).model_dump()
    values[field] = value
    with pytest.raises(ValidationError):
        RubricScores.model_validate(values)


def test_hard_caps_are_suggested_without_rewriting_expert_input():
    scores = RubricScores(
        scientific_correctness=5,
        evidence_sufficiency=5,
        novelty=5,
        actionability=5,
        overall=5,
    )
    result = assess_hard_caps(
        scores,
        RubricFlags(fabricated_source=True),
    )
    assert scores.evidence_sufficiency == 5
    assert result.suggested_scores.evidence_sufficiency == 1
    assert result.suggested_scores.overall == 1
    assert result.reasons


def test_expert_utility_uses_approved_weights():
    scores = RubricScores(
        scientific_correctness=5,
        evidence_sufficiency=1,
        novelty=1,
        actionability=1,
        overall=5,
    )
    assert expert_utility(scores) == pytest.approx(0.35)


def test_generated_evolution_ids_are_path_safe():
    value = new_evolution_id()
    assert value.startswith("evo_")
    assert "/" not in value and ".." not in value


@pytest.mark.parametrize("value", ["3", 3.0, True])
def test_feedback_scores_reject_coerced_integer_inputs(value):
    with pytest.raises(ValidationError):
        RubricScores(
            scientific_correctness=value,
            evidence_sufficiency=3,
            novelty=3,
            actionability=3,
            overall=3,
        )


@pytest.mark.parametrize(
    ("flags", "field", "cap"),
    [
        ({"fabricated_source": True}, "evidence_sufficiency", 1),
        ({"fabricated_source": True}, "overall", 1),
        ({"conclusion_changing_error": True}, "scientific_correctness", 2),
        ({"conclusion_changing_error": True}, "overall", 2),
        ({"abstract_only_core_evidence": True}, "evidence_sufficiency", 2),
        ({"unsupported_novelty": True}, "novelty", 2),
        ({"process_parameters_only": True}, "actionability", 2),
    ],
)
def test_each_hard_cap_is_applied(flags, field, cap):
    assessment = assess_hard_caps(_scores(), RubricFlags(**flags))
    assert getattr(assessment.suggested_scores, field) == cap


def test_all_hard_caps_compose_using_the_strictest_limit():
    assessment = assess_hard_caps(
        _scores(),
        RubricFlags(
            fabricated_source=True,
            conclusion_changing_error=True,
            abstract_only_core_evidence=True,
            unsupported_novelty=True,
            process_parameters_only=True,
        ),
    )
    assert assessment.suggested_scores == RubricScores(
        scientific_correctness=2,
        evidence_sufficiency=1,
        novelty=2,
        actionability=2,
        overall=1,
    )
    assert len(assessment.reasons) == 5


@pytest.mark.parametrize("bad", ["", "../escape", "/tmp/escape", "a/b", "has space"])
def test_managed_ids_reject_unsafe_values(bad):
    with pytest.raises(ValueError):
        validate_managed_id(bad)
    with pytest.raises(ValidationError):
        _task(evolution_id=bad)


def test_managed_reference_ids_are_validated_in_optional_fields_and_lists():
    with pytest.raises(ValidationError):
        _task(episode_ids=["bad/id"])
    with pytest.raises(ValidationError):
        EpisodeRecord(
            evolution_id="evo_test",
            episode_id="ep_test",
            version="v001",
            applied_feedback_id="bad/id",
            task_snapshot={"goal": "goal"},
            target_snapshot=_target(),
        )
    with pytest.raises(ValidationError):
        RevisionPlan(
            revision_id="rp_test",
            evolution_id="evo_test",
            source_version="v001",
            feedback_id="fb_test",
            preserved_evidence_ids=["bad/id"],
        )


def test_generated_ids_follow_managed_patterns():
    assert re.fullmatch(r"evo_\d{8}T\d{12}Z_[0-9a-f]{6}", new_evolution_id())
    assert re.fullmatch(r"ep_[0-9a-f]{10}", new_episode_id())
    assert re.fullmatch(r"fb_[0-9a-f]{10}", new_feedback_id())
    assert re.fullmatch(r"rp_[0-9a-f]{10}", new_revision_id())
    assert re.fullmatch(r"strategy_[0-9a-f]{10}", new_strategy_id())


def test_schema_versions_versions_and_hashes_are_strict():
    with pytest.raises(ValidationError):
        _task(schema_version=2)
    with pytest.raises(ValidationError):
        EpisodeRecord(
            evolution_id="evo_test",
            episode_id="ep_test",
            version="version-one",
            task_snapshot={"goal": "goal"},
            target_snapshot=_target(),
        )
    with pytest.raises(ValidationError):
        ArtifactRef(path="result.md", size_bytes=1, sha256="abc")


def test_required_identity_fields_have_no_invalid_empty_defaults():
    with pytest.raises(ValidationError):
        EvolutionTask(
            evolution_id="evo_test",
            goal="goal",
            target=_target(),
            task_group_id="task_group_1",
        )
    with pytest.raises(ValidationError):
        ExpertFeedbackRecord(
            feedback_id="fb_test",
            evolution_id="evo_test",
            episode_version="v001",
            result_sha256="b" * 64,
            rubric_version="expert-review-v1",
            scores=_scores(3),
        )


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (EvolutionTask, {
            "evolution_id": "evo_test",
            "goal": "goal",
            "target": _target(),
            "task_group_id": "task_group_1",
            "input_sha256": "short",
        }),
        (EpisodeRecord, {
            "evolution_id": "evo_test",
            "episode_id": "ep_test",
            "version": "v001",
            "task_snapshot": {"goal": "goal"},
            "target_snapshot": _target(),
            "tool_surface_fingerprint": "short",
        }),
        (ExpertFeedbackRecord, {
            "feedback_id": "fb_test",
            "evolution_id": "evo_test",
            "episode_version": "v001",
            "result_sha256": "short",
            "rubric_version": "expert-review-v1",
            "raw_input": "raw",
            "scores": _scores(3),
        }),
        (StrategyVersion, {
            "strategy_id": "strategy_test",
            "evolution_id": "evo_test",
            "arm": "STATIC",
            "strategy_sha256": "short",
        }),
    ],
)
def test_every_hash_field_requires_an_exact_sha256(model, payload):
    with pytest.raises(ValidationError):
        model.model_validate(payload)


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (EvolutionTask, {
            "evolution_id": "evo_test",
            "goal": "goal",
            "target": _target(),
            "task_group_id": "task_group_1",
            "input_sha256": "a" * 64,
        }),
        (EpisodeRecord, {
            "evolution_id": "evo_test",
            "episode_id": "ep_test",
            "version": "v001",
            "task_snapshot": {"goal": "goal"},
            "target_snapshot": _target(),
        }),
        (FeedbackCompilation, {}),
        (ExpertFeedbackRecord, {
            "feedback_id": "fb_test",
            "evolution_id": "evo_test",
            "episode_version": "v001",
            "result_sha256": "b" * 64,
            "rubric_version": "expert-review-v1",
            "raw_input": "raw",
            "scores": _scores(3),
        }),
        (RevisionPlan, {
            "revision_id": "rp_test",
            "evolution_id": "evo_test",
            "source_version": "v001",
            "feedback_id": "fb_test",
        }),
        (StrategyVersion, {
            "strategy_id": "strategy_test",
            "evolution_id": "evo_test",
            "arm": "STATIC",
        }),
        (ComparisonReport, {
            "comparison_id": "comparison_test",
            "evolution_id": "evo_test",
            "previous_version": "v001",
            "current_version": "v002",
        }),
    ],
)
def test_every_persisted_schema_rejects_unknown_schema_versions(model, payload):
    with pytest.raises(ValidationError):
        model.model_validate({**payload, "schema_version": 2})


def test_models_forbid_unknown_fields():
    with pytest.raises(ValidationError):
        _task(unknown=True)


def test_task_identity_is_frozen_but_lifecycle_fields_are_mutable():
    task = _task()
    with pytest.raises(ValidationError):
        task.evolution_id = "evo_changed"
    with pytest.raises(ValidationError):
        task.goal = "changed"
    task.status = "RUNNING"
    assert task.status == "RUNNING"


def test_task_and_episode_snapshots_are_independent_of_caller_inputs():
    target = _target()
    task = _task(target=target)
    snapshot = {"nested": {"value": 1}}
    episode = EpisodeRecord(
        evolution_id="evo_test",
        episode_id="ep_test",
        version="v001",
        task_snapshot=snapshot,
        target_snapshot=target,
    )

    target.metadata["nested"]["value"] = 2
    snapshot["nested"]["value"] = 2

    assert task.target.metadata["nested"]["value"] == 1
    assert episode.target_snapshot is not None
    assert episode.target_snapshot.metadata["nested"]["value"] == 1
    assert episode.task_snapshot["nested"]["value"] == 1


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("task_snapshot", {"changed": True}),
        ("target_snapshot", TargetSpec(goal="changed")),
        ("provider", "other-provider"),
        ("model", "other-model"),
        ("tool_surface_fingerprint", "f" * 64),
        ("capability_fingerprint", "0" * 64),
        ("data_source_fingerprints", {"other": "1" * 64}),
    ],
)
def test_episode_execution_context_fields_cannot_be_reassigned(field, replacement):
    episode = _episode()
    with pytest.raises(ValidationError):
        setattr(episode, field, replacement)


def test_episode_defensively_copies_nested_context_and_fingerprint_mapping():
    task_snapshot = {"nested": {"value": 1}}
    target_snapshot = _target()
    data_sources = {"catalog": "e" * 64}
    episode = _episode(
        task_snapshot=task_snapshot,
        target_snapshot=target_snapshot,
        data_source_fingerprints=data_sources,
    )

    task_snapshot["nested"]["value"] = 2
    target_snapshot.metadata["nested"]["value"] = 2
    data_sources["catalog"] = "f" * 64

    assert episode.task_snapshot["nested"]["value"] == 1
    assert episode.target_snapshot.metadata["nested"]["value"] == 1
    assert episode.data_source_fingerprints == {"catalog": "e" * 64}


def test_episode_snapshot_values_reject_direct_nested_mutation():
    episode = _episode(
        task_snapshot={"nested": {"value": 1}, "items": ["a"]},
    )

    with pytest.raises(TypeError):
        episode.task_snapshot["nested"]["value"] = 2
    with pytest.raises(TypeError):
        episode.task_snapshot["items"].append("b")
    with pytest.raises(ValidationError):
        episode.target_snapshot.goal = "changed"
    with pytest.raises(TypeError):
        episode.target_snapshot.metadata["nested"]["value"] = 2
    with pytest.raises(TypeError):
        episode.data_source_fingerprints["catalog"] = "f" * 64


def test_target_snapshot_is_recursive_across_constraints_and_conditions():
    target = TargetSpec(
        goal="goal",
        constraints=[
            ConstraintSpec(property="phase", operator="eq", value={"allowed": ["A"]})
        ],
        objectives=["stability"],
        operating_conditions={"temperature": {"kelvin": 77}},
    )
    episode = _episode(target_snapshot=target)

    with pytest.raises(TypeError):
        episode.target_snapshot.constraints.append(
            ConstraintSpec(property="other", operator="eq", value=1)
        )
    with pytest.raises(ValidationError):
        episode.target_snapshot.constraints[0].property = "changed"
    with pytest.raises(TypeError):
        episode.target_snapshot.constraints[0].value["allowed"].append("B")
    with pytest.raises(TypeError):
        episode.target_snapshot.objectives.append("changed")
    with pytest.raises(TypeError):
        episode.target_snapshot.operating_conditions["temperature"]["kelvin"] = 300


def test_immutable_episode_snapshots_round_trip_as_clean_json():
    episode = _episode(
        task_snapshot={"nested": {"value": 1}, "items": ["a"]},
    )

    serialized = episode.model_dump_json()
    payload = json.loads(serialized)
    assert payload["task_snapshot"] == {
        "nested": {"value": 1},
        "items": ["a"],
    }
    assert payload["target_snapshot"]["goal"] == "goal"
    assert payload["data_source_fingerprints"] == {"catalog": "e" * 64}

    restored = EpisodeRecord.model_validate_json(serialized)
    assert restored == episode
    with pytest.raises(TypeError):
        restored.task_snapshot["nested"]["value"] = 2
    with pytest.raises(TypeError):
        restored.target_snapshot.metadata["nested"]["value"] = 2
    with pytest.raises(TypeError):
        restored.data_source_fingerprints["catalog"] = "f" * 64


def test_episode_task_snapshot_rejects_non_json_values_before_persistence():
    with pytest.raises(ValidationError):
        _episode(task_snapshot={"not_json": object()})


def test_immutable_episode_snapshots_support_safe_deep_copy_and_target_use():
    episode = _episode(
        task_snapshot={"nested": {"value": 1}, "items": ["a"]},
    )

    clone = episode.model_copy(deep=True)
    assert clone == episode
    assert isinstance(clone.target_snapshot, TargetSpec)

    mutable_target = clone.target_snapshot.to_target_spec()
    mutable_target.goal = "changed"
    mutable_target.metadata["nested"]["value"] = 2
    assert clone.target_snapshot.goal == "goal"
    assert clone.target_snapshot.metadata["nested"]["value"] == 1


def test_paused_task_requires_a_strict_resume_checkpoint():
    with pytest.raises(ValidationError, match="resume_status"):
        _task(status="STOPPED")
    with pytest.raises(ValidationError, match="resume_status"):
        _task(status="STOPPED", resume_status="RUNNING")

    stopped = _task(status="STOPPED", resume_status="FEEDBACK_RECORDED")
    assert stopped.resume_status == "FEEDBACK_RECORDED"


def test_active_task_cannot_retain_a_resume_checkpoint():
    with pytest.raises(ValidationError, match="resume_status"):
        _task(status="CREATED", resume_status="CREATED")


def test_episode_lifecycle_fields_remain_mutable():
    episode = _episode()
    started = datetime(2026, 9, 4, 2, 0, tzinfo=UTC)
    completed = datetime(2026, 9, 4, 2, 1, tzinfo=UTC)
    summary = ScientificLoopSummary(
        status="INCONCLUSIVE",
        rounds=1,
        candidate_count=0,
        best_candidate_id=None,
        best_score=0.0,
        final_evaluation=None,
    )
    cost = CostSnapshot(input_tokens=12, runtime_seconds=60.0)

    episode.status = "RUNNING"
    episode.created_at = started
    episode.started_at = started
    episode.completed_at = completed
    episode.summary = summary
    episode.cost = cost

    assert episode.status == "RUNNING"
    assert episode.created_at == started
    assert episode.started_at == started
    assert episode.completed_at == completed
    assert episode.summary == summary
    assert episode.cost == cost


def test_explicit_datetimes_reject_naive_values_and_normalize_to_utc():
    with pytest.raises(ValidationError):
        _task(created_at=datetime(2026, 9, 4, 10, 0))

    local_time = datetime(2026, 9, 4, 10, 0, tzinfo=timezone(timedelta(hours=8)))
    task = _task(created_at=local_time, updated_at=local_time)
    assert task.created_at == datetime(2026, 9, 4, 2, 0, tzinfo=UTC)
    assert task.created_at.tzinfo is UTC


def test_utc_datetimes_survive_json_round_trip():
    task = _task()
    restored = EvolutionTask.model_validate_json(task.model_dump_json())
    assert restored == task
    assert restored.created_at.tzinfo is UTC
    assert restored.updated_at.tzinfo is UTC


def test_feedback_record_requires_deterministic_cap_provenance():
    with pytest.raises(ValidationError):
        ExpertFeedbackRecord(
            feedback_id="fb_test",
            evolution_id="evo_test",
            episode_version="v001",
            result_sha256="b" * 64,
            rubric_version="expert-review-v1",
            raw_input="raw",
            scores=_scores(),
            flags=RubricFlags(fabricated_source=True),
        )


def test_feedback_record_requires_override_reason_above_suggested_cap():
    assessment = assess_hard_caps(_scores(), RubricFlags(fabricated_source=True))
    data = {
        "feedback_id": "fb_test",
        "evolution_id": "evo_test",
        "episode_version": "v001",
        "result_sha256": "b" * 64,
        "rubric_version": "expert-review-v1",
        "raw_input": "raw",
        "scores": _scores(),
        "flags": RubricFlags(fabricated_source=True),
        "suggested_scores": assessment.suggested_scores,
        "hard_cap_reasons": assessment.reasons,
    }
    with pytest.raises(ValidationError):
        ExpertFeedbackRecord.model_validate(data)

    record = ExpertFeedbackRecord.model_validate(
        {**data, "hard_cap_override_reason": "专家核验后保留原分数"}
    )
    assert record.scores == _scores()
    assert record.suggested_scores == assessment.suggested_scores


def test_feedback_record_copies_original_scores_before_persisting():
    scores = _scores(3)
    record = ExpertFeedbackRecord(
        feedback_id="fb_test",
        evolution_id="evo_test",
        episode_version="v001",
        result_sha256="b" * 64,
        rubric_version="expert-review-v1",
        raw_input="raw",
        scores=scores,
    )
    scores.overall = 1
    assert record.scores.overall == 3


def test_feedback_record_rejects_mismatched_cap_provenance():
    with pytest.raises(ValidationError):
        ExpertFeedbackRecord(
            feedback_id="fb_test",
            evolution_id="evo_test",
            episode_version="v001",
            result_sha256="b" * 64,
            rubric_version="expert-review-v1",
            raw_input="raw",
            scores=_scores(),
            flags=RubricFlags(fabricated_source=True),
            suggested_scores=_scores(1),
            hard_cap_reasons=["wrong reason"],
            hard_cap_override_reason="reviewed",
        )


def test_comparison_report_round_trips_typed_design_deltas():
    report = ComparisonReport(
        comparison_id="comparison_test",
        evolution_id="evo_test",
        previous_version="v001",
        current_version="v002",
        score_deltas=[
            RubricScoreDelta(
                dimension="scientific_correctness",
                previous=3,
                current=4,
                delta=1,
                normalized_delta=0.25,
            )
        ],
        constraint_changes=ConstraintChangeSummary(newly_passed=["band_gap"]),
        evidence_changes=EvidenceChangeSummary(added_ids=["evidence_2"]),
        fidelity_changes=FidelityChangeSummary(upgraded_ids=["evidence_1"]),
        artifact_diff=ArtifactDiff(
            previous_sha256="c" * 64,
            current_sha256="d" * 64,
            changed=True,
            size_bytes_delta=12,
        ),
        cost_delta=CostDelta(input_tokens=10, runtime_seconds=0.5),
        unresolved_human_checks=["确认工艺安全性"],
    )
    assert ComparisonReport.model_validate_json(report.model_dump_json()) == report
