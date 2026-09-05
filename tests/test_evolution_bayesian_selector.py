from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from photomatagent.cli.app import app
from photomatagent.scientific.evolution.experience import (
    StrategyObservation,
    create_experience,
    create_strategy_observation,
    strategy_observation_sha256,
)
from photomatagent.scientific.evolution.models import (
    ComparisonReport,
    EpisodeRecord,
    EvolutionTask,
    RevisionPlan,
)
from photomatagent.scientific.evolution.store import (
    EvolutionAlreadyExistsError,
    EvolutionConflictError,
    EvolutionCorruptRecordError,
    EvolutionStore,
)
from photomatagent.scientific.evolution.strategy import (
    FEATURE_SCHEMA,
    BayesianLinearStrategySelector,
    FixedStrategySelector,
    StrategyPosteriorSnapshot,
    TaskContext,
    posterior_sha256,
)
from photomatagent.scientific.loop import TargetSpec
from photomatagent.workspace import Workspace


def _observation(
    index: int,
    *,
    task_group_id: str,
    arm: str = "EVIDENCE_FIRST",
    reward: float = 0.5,
    context: TaskContext | None = None,
    expert_backed: bool = True,
) -> StrategyObservation:
    evolution_id = f"evo_bayes_{index}"
    task = EvolutionTask(
        evolution_id=evolution_id,
        goal="learn from reviewed comparison",
        target=TargetSpec(goal="learn from reviewed comparison"),
        task_group_id=task_group_id,
        input_sha256=f"{index % 16:x}" * 64,
    )
    plan = RevisionPlan(
        revision_id=f"rp_bayes_{index}",
        evolution_id=evolution_id,
        source_version="v001",
        feedback_id=f"fb_previous_{index}",
        strategy_arm=arm,  # type: ignore[arg-type]
        strategy_reason="reviewed revision",
    )
    strategy = FixedStrategySelector().select(task, plan)
    reward_fields = (
        {
            "expert_utility_delta": reward,
            "reward": reward,
            "components_used": ["expert_utility_delta"],
        }
        if expert_backed
        else {
            "normalized_cost_increase": -reward,
            "reward": reward,
            "components_used": ["normalized_cost_increase"],
        }
    )
    comparison = ComparisonReport(
        comparison_id=f"cmp_bayes_{index}",
        evolution_id=evolution_id,
        previous_version="v001",
        current_version="v002",
        phase="POST_FEEDBACK",
        current_feedback_id=f"fb_current_{index}",
        current_feedback_sha256="a" * 64,
        current_compilation_id=f"comp_current_{index}",
        current_compilation_sha256="b" * 64,
        **reward_fields,
    )
    experience = create_experience(comparison, task_group_id=task_group_id)
    return create_strategy_observation(
        task=task,
        comparison=comparison,
        experience=experience,
        strategy=strategy,
        context=context or TaskContext.from_target(task.target),
        source_execution_mode="CARRY_VERIFIED_EVIDENCE",
    )


def _observations(
    count: int,
    *,
    distinct_tasks: int,
    arm: str = "EVIDENCE_FIRST",
    reward: float = 0.5,
) -> list[StrategyObservation]:
    return [
        _observation(
            index,
            task_group_id=f"group_{index % distinct_tasks}",
            arm=arm,
            reward=reward,
        )
        for index in range(count)
    ]


def test_bayesian_selector_stays_disabled_below_distinct_task_gate() -> None:
    selector = BayesianLinearStrategySelector(seed=7)
    selector.fit(_observations(25, distinct_tasks=1))

    assert selector.enabled is False
    assert selector.diagnostics.observation_count == 25
    assert selector.diagnostics.distinct_tasks == 1


def test_selector_enables_at_twenty_observations_and_eight_tasks() -> None:
    selector = BayesianLinearStrategySelector(seed=7)
    selector.fit(_observations(20, distinct_tasks=8))

    assert selector.enabled is True
    assert selector.diagnostics.observation_count == 20
    assert selector.diagnostics.distinct_tasks == 8
    assert selector.diagnostics.effective_training_rows == 8


def test_disabled_selector_falls_back_to_exact_fixed_strategy() -> None:
    observation = _observation(1, task_group_id="one_group")
    selector = BayesianLinearStrategySelector(seed=7).fit([observation])
    task = EvolutionTask(
        evolution_id="evo_fallback",
        goal="fallback",
        target=TargetSpec(goal="fallback"),
        task_group_id="group_fallback",
        input_sha256="f" * 64,
    )
    plan = RevisionPlan(
        revision_id="rp_fallback",
        evolution_id=task.evolution_id,
        source_version="v001",
        feedback_id="fb_fallback",
        strategy_arm="UNCERTAINTY_FIRST",
        strategy_reason="fixed fallback",
    )

    selected = selector.select(task, plan, TaskContext.from_target(task.target))

    assert selected == FixedStrategySelector().select(task, plan)
    assert selected.parameters["selector"] == "fixed-v1"


def test_thompson_sampling_is_reproducible_without_shared_rng_state() -> None:
    observations = _observations(20, distinct_tasks=8)
    left = BayesianLinearStrategySelector(seed=23).fit(observations)
    right = BayesianLinearStrategySelector(seed=23).fit(reversed(observations))
    task = EvolutionTask(
        evolution_id="evo_reproducible",
        goal="reproducible",
        target=TargetSpec(goal="reproducible"),
        task_group_id="group_reproducible",
        input_sha256="d" * 64,
    )
    plan = RevisionPlan(
        revision_id="rp_reproducible",
        evolution_id=task.evolution_id,
        source_version="v001",
        feedback_id="fb_reproducible",
    )
    context = TaskContext.from_target(task.target)

    first = left.select(task, plan, context)
    retry = left.select(task, plan, context)
    independently_fitted = right.select(task, plan, context)
    next_revision = left.select(
        task,
        plan.model_copy(update={"revision_id": "rp_reproducible_next"}),
        context,
    )

    assert first == retry
    assert first.arm == independently_fitted.arm
    assert first.parameters["decision_seed"] == independently_fitted.parameters[
        "decision_seed"
    ]
    assert first.parameters["decision_seed"] != next_revision.parameters[
        "decision_seed"
    ]


def test_posterior_covariance_is_finite_symmetric_and_positive_definite() -> None:
    selector = BayesianLinearStrategySelector(seed=3).fit(
        _observations(24, distinct_tasks=8)
    )
    assert selector.posterior is not None
    covariance = np.asarray(selector.posterior.covariance)

    assert np.isfinite(covariance).all()
    assert np.allclose(covariance, covariance.T, atol=1e-12)
    assert np.linalg.eigvalsh(covariance).min() > 0.0
    assert selector.posterior.feature_schema == FEATURE_SCHEMA
    assert selector.posterior.prior_precision == 4.0
    assert selector.posterior.noise_variance == 0.25


@pytest.mark.parametrize("value", ["0.0", True])
def test_posterior_rejects_coerced_numeric_entries(value: object) -> None:
    selector = BayesianLinearStrategySelector(seed=3).fit(
        _observations(20, distinct_tasks=8)
    )
    assert selector.posterior is not None
    payload = selector.posterior.model_dump(mode="python")
    payload["mean"] = (value, *payload["mean"][1:])

    with pytest.raises(ValidationError, match="posterior arrays"):
        StrategyPosteriorSnapshot.model_validate(payload)


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), -float("inf")])
def test_strategy_observation_rejects_nonfinite_reward(reward: float) -> None:
    with pytest.raises(ValidationError):
        _observation(1, task_group_id="group_bad", reward=reward)


@pytest.mark.parametrize("reward", [True, "0.5"])
def test_strategy_observation_rejects_coerced_reward(reward: object) -> None:
    payload = _observation(1, task_group_id="group_bad").model_dump(mode="python")
    payload["reward"] = reward
    with pytest.raises(ValidationError):
        StrategyObservation.model_validate(payload)


def test_strategy_observation_rejects_forged_id_hash_and_is_immutable() -> None:
    observation = _observation(1, task_group_id="group_provenance")
    payload = observation.model_dump(mode="python")
    payload["observation_id"] = "obs_forged"
    with pytest.raises(ValidationError, match="observation_id"):
        StrategyObservation.model_validate(payload)
    payload = observation.model_dump(mode="python")
    payload["observation_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="observation_sha256"):
        StrategyObservation.model_validate(payload)
    with pytest.raises(ValidationError):
        observation.reward = 0.0


def test_pre_feedback_and_fresh_evaluation_cannot_be_training_observations() -> None:
    valid = _observation(1, task_group_id="group_reviewed")
    for field, value in (
        ("comparison_phase", "PRE_FEEDBACK"),
        ("source_execution_mode", "FRESH_EVALUATION"),
    ):
        payload = valid.model_dump(mode="python")
        payload[field] = value
        payload["observation_id"] = "obs_forged"
        with pytest.raises(ValidationError):
            StrategyObservation.model_validate(payload)


def test_machine_only_reward_cannot_become_strategy_training_signal() -> None:
    with pytest.raises(ValueError, match="expert utility"):
        _observation(
            1,
            task_group_id="group_machine_only",
            reward=-0.5,
            expert_backed=False,
        )


def test_duplicate_observation_is_idempotent_and_conflict_is_rejected() -> None:
    observations = _observations(20, distinct_tasks=8)
    selector = BayesianLinearStrategySelector(seed=5).fit(
        [*observations, observations[0]]
    )
    assert selector.diagnostics.observation_count == 20

    conflict = _observation(
        0,
        task_group_id=observations[0].task_group_id,
        reward=-0.5,
    )
    with pytest.raises(ValueError, match="conflicting.*comparison"):
        BayesianLinearStrategySelector(seed=5).fit([*observations, conflict])


def test_repeated_same_group_arm_context_is_aggregated_not_overweighted() -> None:
    base = _observations(20, distinct_tasks=8)
    repeated = [
        _observation(
            1_000 + index,
            task_group_id=base[0].task_group_id,
            reward=base[0].reward,
            context=base[0].context,
        )
        for index in range(100)
    ]
    baseline = BayesianLinearStrategySelector(seed=11).fit(base)
    crowded = BayesianLinearStrategySelector(seed=11).fit([*base, *repeated])
    assert baseline.posterior is not None
    assert crowded.posterior is not None

    assert crowded.diagnostics.observation_count == 120
    assert crowded.diagnostics.effective_training_rows == 8
    assert np.allclose(crowded.posterior.mean, baseline.posterior.mean)
    assert np.allclose(crowded.posterior.covariance, baseline.posterior.covariance)


def test_high_reward_arm_is_selected_more_often_across_independent_seeds() -> None:
    observations: list[StrategyObservation] = []
    index = 0
    for group in range(32):
        for arm, reward in (
            ("STATIC", -0.8),
            ("EVIDENCE_FIRST", 0.9),
            ("DIVERSITY_FIRST", -0.6),
            ("UNCERTAINTY_FIRST", -0.7),
        ):
            observations.append(
                _observation(
                    index,
                    task_group_id=f"balanced_{group}",
                    arm=arm,
                    reward=reward,
                )
            )
            index += 1
    task = EvolutionTask(
        evolution_id="evo_arm_frequency",
        goal="arm frequency",
        target=TargetSpec(goal="arm frequency"),
        task_group_id="group_arm_frequency",
        input_sha256="e" * 64,
    )
    plan = RevisionPlan(
        revision_id="rp_arm_frequency",
        evolution_id=task.evolution_id,
        source_version="v001",
        feedback_id="fb_arm_frequency",
    )
    context = TaskContext.from_target(task.target)

    choices = [
        BayesianLinearStrategySelector(seed=seed)
        .fit(observations)
        .select(task, plan, context)
        .arm
        for seed in range(80)
    ]

    assert choices.count("EVIDENCE_FIRST") > 60


def test_posterior_snapshot_round_trips_atomically_and_detects_tampering(
    tmp_path: Path,
) -> None:
    observations: list[StrategyObservation] = []
    store: EvolutionStore | None = None
    for index in range(20):
        store, observation = _persist_observation_sources(
            tmp_path,
            index=1_000 + index,
            task_group_id=f"group_posterior_{index % 8}",
        )
        store.write_strategy_observation(observation)
        observations.append(observation)
    assert store is not None
    selector = BayesianLinearStrategySelector(seed=13).fit(
        observations
    )
    assert selector.posterior is not None
    evolution_id = observations[0].evolution_id

    path = store.write_strategy_posterior(evolution_id, selector.posterior)

    assert store.load_strategy_posterior(
        evolution_id, selector.posterior.posterior_id
    ) == selector.posterior
    with pytest.raises(EvolutionAlreadyExistsError):
        store.write_strategy_posterior(evolution_id, selector.posterior)

    forged_payload = selector.posterior.model_dump(mode="python")
    forged_payload["mean"] = (
        forged_payload["mean"][0] + 0.1,
        *forged_payload["mean"][1:],
    )
    forged_draft = StrategyPosteriorSnapshot.model_construct(**forged_payload)
    forged_hash = posterior_sha256(forged_draft)
    forged_payload["posterior_sha256"] = forged_hash
    forged_payload["posterior_id"] = f"posterior_{forged_hash[:16]}"
    forged = StrategyPosteriorSnapshot.model_validate(forged_payload)
    with pytest.raises(EvolutionConflictError, match="authoritative observations"):
        store.write_strategy_posterior(evolution_id, forged)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["mean"][0] += 0.1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EvolutionCorruptRecordError):
        store.load_strategy_posterior(
            evolution_id, selector.posterior.posterior_id
        )
    with pytest.raises(ValueError):
        store.load_strategy_posterior(evolution_id, "../escape")


def _persist_observation_sources(
    tmp_path: Path,
    *,
    index: int = 901,
    task_group_id: str = "group_store_authority",
    include_experience_manifest: bool = True,
    execution_mode: str = "CARRY_VERIFIED_EVIDENCE",
) -> tuple[EvolutionStore, StrategyObservation]:
    observation = _observation(index, task_group_id=task_group_id)
    task = EvolutionTask(
        evolution_id=observation.evolution_id,
        goal="store authority",
        target=TargetSpec(goal="store authority"),
        task_group_id=observation.task_group_id,
        input_sha256=f"{index % 16:x}" * 64,
        episode_ids=[f"ep_store_previous_{index}", f"ep_store_current_{index}"],
        comparison_ids=[observation.comparison_id],
        experience_ids=(
            [observation.experience_id] if include_experience_manifest else []
        ),
        strategy_ids=[observation.strategy_id],
    )
    plan = RevisionPlan(
        revision_id=f"rp_bayes_{index}",
        evolution_id=task.evolution_id,
        source_version="v001",
        feedback_id=f"fb_previous_{index}",
        strategy_arm=observation.strategy_arm,
        strategy_reason="reviewed revision",
        created_at=observation.strategy_cutoff_at,
    )
    strategy = FixedStrategySelector().select(task, plan)
    comparison = ComparisonReport(
        comparison_id=observation.comparison_id,
        evolution_id=task.evolution_id,
        previous_version="v001",
        current_version="v002",
        phase="POST_FEEDBACK",
        current_feedback_id=observation.current_feedback_id,
        current_feedback_sha256=observation.current_feedback_sha256,
        current_compilation_id=observation.current_compilation_id,
        current_compilation_sha256=observation.current_compilation_sha256,
        expert_utility_delta=observation.reward,
        reward=observation.reward,
        components_used=["expert_utility_delta"],
        created_at=observation.created_at,
    )
    experience = create_experience(
        comparison, task_group_id=observation.task_group_id
    )
    assert strategy == FixedStrategySelector().select(task, plan)
    assert strategy.strategy_id == observation.strategy_id
    assert experience.experience_id == observation.experience_id
    store = EvolutionStore(Workspace(tmp_path))
    store.create_task(task)
    store.write_comparison(comparison)
    store.write_experience(experience)
    store.write_strategy(strategy)
    store.write_episode(
        EpisodeRecord(
            evolution_id=task.evolution_id,
            episode_id=f"ep_store_previous_{index}",
            version="v001",
            status="COMPLETED",
            execution_mode="NORMAL",
            task_snapshot=task.model_dump(mode="json"),
            target_snapshot=task.target,
        )
    )
    store.write_episode(
        EpisodeRecord(
            evolution_id=task.evolution_id,
            episode_id=f"ep_store_current_{index}",
            version="v002",
            status="COMPLETED",
            parent_version="v001",
            revision_plan_id=plan.revision_id,
            applied_feedback_id=plan.feedback_id,
            execution_mode=execution_mode,  # type: ignore[arg-type]
            strategy_id=strategy.strategy_id,
            strategy_arm=strategy.arm,
            strategy_sha256=strategy.strategy_sha256,
            strategy_cutoff_at=strategy.cutoff_at,
            task_snapshot=task.model_dump(mode="json"),
            target_snapshot=task.target,
        )
    )
    return store, observation


def test_store_rechecks_authoritative_observation_links_and_is_idempotent(
    tmp_path: Path,
) -> None:
    store, observation = _persist_observation_sources(tmp_path)

    first = store.write_strategy_observation(observation)
    second = store.write_strategy_observation(observation)

    assert second == first
    assert store.load_strategy_observation(
        observation.evolution_id, observation.observation_id
    ) == observation
    assert store.list_all_strategy_observations() == [observation]


def test_store_rejects_resigned_observation_with_forged_context(
    tmp_path: Path,
) -> None:
    store, observation = _persist_observation_sources(tmp_path)
    payload = observation.model_dump(mode="python")
    payload["context"] = TaskContext(
        hard_constraint_ratio=0.0,
        soft_constraint_ratio=0.0,
        objective_ratio=0.0,
        operating_condition_ratio=0.0,
        previous_critical_gap_ratio=0.1,
    )
    forged_draft = StrategyObservation.model_construct(**payload)
    digest = strategy_observation_sha256(forged_draft)
    payload["observation_sha256"] = digest
    payload["observation_id"] = f"obs_{digest[:16]}"
    forged = StrategyObservation.model_validate(payload)

    with pytest.raises(EvolutionConflictError, match="task context"):
        store.write_strategy_observation(forged)


def test_store_rejects_observation_without_manifest_experience_link(
    tmp_path: Path,
) -> None:
    store, observation = _persist_observation_sources(
        tmp_path, include_experience_manifest=False
    )

    with pytest.raises(EvolutionConflictError, match="experience.*manifest"):
        store.write_strategy_observation(observation)


def test_store_rejects_observation_execution_mode_claim_mismatch(
    tmp_path: Path,
) -> None:
    store, observation = _persist_observation_sources(
        tmp_path, execution_mode="NORMAL"
    )

    with pytest.raises(EvolutionConflictError, match="execution mode"):
        store.write_strategy_observation(observation)


def test_evolve_status_reports_only_fixed_or_enabled_selector_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    task = EvolutionTask(
        evolution_id="evo_selector_status",
        goal="status",
        target=TargetSpec(goal="status"),
        task_group_id="group_selector_status",
        input_sha256="6" * 64,
    )
    store.create_task(task)
    runner = CliRunner()

    fixed = runner.invoke(
        app,
        ["evolve", "status", task.evolution_id, "--workspace", str(tmp_path)],
    )
    assert fixed.exit_code == 0
    assert "fixed baseline" in fixed.stdout
    assert "learning complete" not in fixed.stdout.lower()

    monkeypatch.setattr(
        EvolutionStore,
        "list_all_strategy_observations",
        lambda self: _observations(20, distinct_tasks=8),
    )
    enabled = runner.invoke(
        app,
        ["evolve", "status", task.evolution_id, "--workspace", str(tmp_path)],
    )
    assert enabled.exit_code == 0
    assert "Bayesian enabled" in enabled.stdout
