from __future__ import annotations

import json
from pathlib import Path

import pytest

from photomatagent.scientific.evolution.models import (
    EpisodeRecord,
    EvolutionTask,
    ExpertFeedbackRecord,
    RevisionPlan,
    RubricScores,
    StrategyVersion,
)
from photomatagent.scientific.evolution.store import (
    EvolutionAlreadyExistsError,
    EvolutionConflictError,
    EvolutionLockError,
    EvolutionStore,
)
from photomatagent.scientific.loop import TargetSpec
from photomatagent.scientific.state import ScientificState
from photomatagent.workspace import Workspace


def make_task(evolution_id: str = "evo_test") -> EvolutionTask:
    return EvolutionTask(
        evolution_id=evolution_id,
        goal="find a stable infrared absorber",
        target=TargetSpec(goal="find a stable infrared absorber"),
        task_group_id="group_test",
        input_sha256="a" * 64,
    )


def make_episode(evolution_id: str = "evo_test") -> EpisodeRecord:
    return EpisodeRecord(
        evolution_id=evolution_id,
        episode_id="ep_test",
        version="v001",
        task_snapshot={"goal": "find a stable infrared absorber"},
        target_snapshot=TargetSpec(goal="find a stable infrared absorber"),
    )


def make_feedback(evolution_id: str = "evo_test") -> ExpertFeedbackRecord:
    return ExpertFeedbackRecord(
        feedback_id="fb_test",
        evolution_id=evolution_id,
        episode_version="v001",
        result_sha256="b" * 64,
        rubric_version="expert-review-v1",
        raw_input="Authorization: Bearer super-secret-value",
        scores=RubricScores(
            scientific_correctness=3,
            evidence_sufficiency=3,
            novelty=3,
            actionability=3,
            overall=3,
        ),
    )


def test_store_round_trip_and_revision_conflict(tmp_path: Path) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    task = make_task()

    created = store.create_task(task)
    loaded = store.load_task(task.evolution_id)
    loaded.status = "RUNNING"
    saved = store.save_task(loaded, expected_revision=0)

    assert created == task
    assert saved.revision == 1
    assert saved.status == "RUNNING"
    assert saved.updated_at >= created.updated_at
    assert store.load_task(task.evolution_id) == saved
    with pytest.raises(EvolutionConflictError):
        store.save_task(loaded, expected_revision=0)


def test_store_refuses_to_overwrite_immutable_episode(tmp_path: Path) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    task = make_task()
    episode = make_episode()
    store.create_task(task)

    path = store.write_episode(episode)

    assert path == tmp_path / ".photomatagent/evolutions/evo_test/episodes/v001.json"
    assert EpisodeRecord.model_validate_json(path.read_text(encoding="utf-8")) == episode
    with pytest.raises(EvolutionAlreadyExistsError):
        store.write_episode(episode)


@pytest.mark.parametrize("bad", ["../escape", "/tmp/escape", "a/b"])
def test_store_rejects_unmanaged_ids(tmp_path: Path, bad: str) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    with pytest.raises(ValueError):
        store.load_task(bad)


def test_store_revalidates_record_ids_before_building_managed_paths(
    tmp_path: Path,
) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    store.create_task(make_task())
    unsafe_episode = make_episode().model_copy(update={"version": "../escape"})
    unsafe_feedback = make_feedback().model_copy(update={"feedback_id": "a/b"})

    with pytest.raises(ValueError):
        store.write_episode(unsafe_episode)
    with pytest.raises(ValueError):
        store.write_feedback(unsafe_feedback)

    assert not (store.root / "escape.json").exists()
    assert not (store.root / "evo_test/feedback/a/b.json").exists()


def test_store_reports_lock_timeout_without_removing_another_owner_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    store.create_task(make_task())
    lock_path = store.root / "evo_test/.lock"
    lock_path.write_text("held by another process", encoding="utf-8")
    monotonic_values = iter([10.0, 15.0])
    monkeypatch.setattr(
        "photomatagent.scientific.evolution.store.time.monotonic",
        lambda: next(monotonic_values),
    )

    with pytest.raises(EvolutionLockError):
        store.save_task(store.load_task("evo_test"), expected_revision=0)

    assert lock_path.read_text(encoding="utf-8") == "held by another process"


def test_interrupted_atomic_create_leaves_no_formal_or_temporary_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = EvolutionStore(Workspace(tmp_path))

    def fail_replace(source: str | Path, destination: str | Path) -> None:
        raise OSError("simulated interruption before replace")

    monkeypatch.setattr(
        "photomatagent.scientific.evolution.store.os.replace", fail_replace
    )

    with pytest.raises(OSError, match="simulated interruption"):
        store.create_task(make_task())

    task_dir = tmp_path / ".photomatagent/evolutions/evo_test"
    assert not (task_dir / "task.json").exists()
    assert list(task_dir.glob("*.tmp")) == []


def test_immutable_records_are_redacted_and_written_to_managed_paths(
    tmp_path: Path,
) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    store.create_task(make_task())
    feedback = make_feedback()
    revision = RevisionPlan(
        revision_id="rp_test",
        evolution_id="evo_test",
        source_version="v001",
        feedback_id="fb_test",
    )
    strategy = StrategyVersion(
        strategy_id="strategy_test",
        evolution_id="evo_test",
        arm="STATIC",
        parameters={"api_key": "must-not-persist"},
    )

    feedback_path = store.write_feedback(feedback)
    revision_path = store.write_revision(revision)
    strategy_path = store.write_strategy(strategy)

    assert feedback_path.name == "fb_test.json"
    assert feedback_path.parent.name == "feedback"
    assert revision_path.name == "rp_test.json"
    assert revision_path.parent.name == "revisions"
    assert strategy_path.name == "strategy_test.json"
    assert strategy_path.parent.name == "strategies"
    assert "super-secret-value" not in feedback_path.read_text(encoding="utf-8")
    payload = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert payload["parameters"]["api_key"] == "[REDACTED]"

    for write, record in (
        (store.write_feedback, feedback),
        (store.write_revision, revision),
        (store.write_strategy, strategy),
    ):
        with pytest.raises(EvolutionAlreadyExistsError):
            write(record)


def test_scientific_state_round_trips_at_episode_version_path(tmp_path: Path) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    store.create_task(make_task())
    state = ScientificState(
        goal="infrared detector",
        hypotheses=["candidate token sk-abcdefghijk"],
        open_questions=["Which validation is still missing?"],
    )

    path = store.write_scientific_state("evo_test", "v001", state)

    assert path == (
        tmp_path
        / ".photomatagent/evolutions/evo_test/episodes/v001.scientific.json"
    )
    assert "sk-abcdefghijk" not in path.read_text(encoding="utf-8")
    loaded = store.load_scientific_state("evo_test", "v001")
    assert loaded.goal == state.goal
    assert loaded.hypotheses == ["candidate token [REDACTED]"]
    with pytest.raises(EvolutionAlreadyExistsError):
        store.write_scientific_state("evo_test", "v001", state)


def test_list_tasks_is_sorted_and_ignores_non_task_entries(tmp_path: Path) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    second = make_task("evo_second")
    first = make_task("evo_first")
    store.create_task(second)
    store.create_task(first)
    (store.root / "notes.txt").write_text("not a task", encoding="utf-8")

    assert [task.evolution_id for task in store.list_tasks()] == [
        "evo_first",
        "evo_second",
    ]


def test_create_task_requires_initial_revision_and_is_immutable(tmp_path: Path) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    task = make_task()
    store.create_task(task)

    with pytest.raises(EvolutionAlreadyExistsError):
        store.create_task(task)
    with pytest.raises(ValueError, match="revision 0"):
        store.create_task(make_task("evo_nonzero").model_copy(update={"revision": 2}))
