from __future__ import annotations

from photomatagent.scientific.evolution.models import EvolutionTask, RevisionPlan
from photomatagent.scientific.evolution.strategy import (
    ARM_ORDER,
    FEATURE_DIMENSION,
    FEATURE_SCHEMA,
    FixedStrategySelector,
    TaskContext,
    feature_vector,
)
from photomatagent.scientific.loop import ConstraintSpec, TargetSpec


def _task() -> EvolutionTask:
    return EvolutionTask(
        evolution_id="evo_strategy_unit",
        goal="Improve report",
        target=TargetSpec(goal="Improve report"),
        task_group_id="group_strategy_unit",
        input_sha256="b" * 64,
    )


def _plan(arm: str, reason: str) -> RevisionPlan:
    return RevisionPlan(
        revision_id="rp_strategy_unit",
        evolution_id="evo_strategy_unit",
        source_version="v001",
        feedback_id="fb_strategy_unit",
        strategy_arm=arm,  # type: ignore[arg-type]
        strategy_reason=reason,
    )


def test_fixed_selector_materializes_exact_plan_choice_without_randomness() -> None:
    task = _task()
    plan = _plan(
        "EVIDENCE_FIRST",
        "fixed-v1: HIGH evidence issue; critical=0; high=1; first=item_002",
    )

    first = FixedStrategySelector().select(task, plan)
    second = FixedStrategySelector().select(task, plan)

    assert first == second
    assert first.arm == "EVIDENCE_FIRST"
    assert first.reason == plan.strategy_reason
    assert first.strategy_id.startswith("strategy_")
    assert len(first.strategy_id) == 19
    assert first.strategy_sha256 is not None
    assert len(first.strategy_sha256) == 64
    assert first.parameters == {
        "selector": "fixed-v1",
        "revision_id": "rp_strategy_unit",
    }


def test_fixed_selector_rejects_cross_task_plan() -> None:
    task = _task()
    plan = _plan("STATIC", "fixed-v1: no effective negative item").model_copy(
        update={"evolution_id": "evo_other"}
    )

    try:
        FixedStrategySelector().select(task, plan)
    except ValueError as exc:
        assert "same evolution task" in str(exc)
    else:  # pragma: no cover - explicit assertion without pytest dependency
        raise AssertionError("cross-task plan was accepted")


def test_task_context_and_feature_vector_have_fixed_bounded_schema() -> None:
    target = TargetSpec(
        goal="bounded context",
        constraints=[
            *[
                ConstraintSpec(property=f"hard_{index}", operator="ge")
                for index in range(12)
            ],
            *[
                ConstraintSpec(
                    property=f"soft_{index}", operator="ge", severity="SOFT"
                )
                for index in range(5)
            ],
        ],
        objectives=[f"objective {index}" for index in range(3)],
        operating_conditions={f"condition_{index}": index for index in range(20)},
    )

    context = TaskContext.from_target(target, previous_critical_gap_count=4)
    vector = feature_vector(context, "DIVERSITY_FIRST")

    assert context.values == (1.0, 1.0, 0.5, 0.3, 1.0, 0.4)
    assert ARM_ORDER == (
        "STATIC",
        "EVIDENCE_FIRST",
        "DIVERSITY_FIRST",
        "UNCERTAINTY_FIRST",
    )
    assert FEATURE_DIMENSION == 30
    assert len(FEATURE_SCHEMA) == FEATURE_DIMENSION
    assert vector == (
        1.0,
        1.0,
        0.5,
        0.3,
        1.0,
        0.4,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.5,
        0.0,
        0.0,
        0.0,
        0.3,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.4,
        0.0,
    )
