from __future__ import annotations

import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from pydantic import ValidationError
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
    EvolutionCorruptRecordError,
    EvolutionLockError,
    EvolutionStore,
    EvolutionUnsupportedSchemaError,
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


def test_save_rejects_caller_revision_that_differs_from_expected(tmp_path: Path) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    task = make_task()
    store.create_task(task)
    forged = task.model_copy(update={"revision": 9})

    with pytest.raises(EvolutionConflictError, match="caller revision=9"):
        store.save_task(forged, expected_revision=0)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("goal", "different goal"),
        ("target", TargetSpec(goal="different target")),
        ("task_group_id", "different_group"),
        ("input_sha256", "b" * 64),
        ("created_at", "2026-09-04T00:00:00Z"),
    ],
)
def test_save_rejects_changes_to_authoritative_immutable_task_fields(
    tmp_path: Path, field: str, replacement: object
) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    task = make_task()
    store.create_task(task)
    forged = task.model_copy(update={field: replacement})

    with pytest.raises(EvolutionConflictError, match=field):
        store.save_task(forged, expected_revision=0)

    assert store.load_task(task.evolution_id) == task


def test_save_rebuilds_task_from_disk_and_ignores_caller_updated_at(
    tmp_path: Path,
) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    task = make_task()
    store.create_task(task)
    caller = task.model_copy(update={"updated_at": task.created_at, "status": "RUNNING"})

    saved = store.save_task(caller, expected_revision=0)

    assert saved.status == "RUNNING"
    assert saved.updated_at > task.created_at


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


@pytest.mark.parametrize("record_kind", ["episode", "feedback", "revision", "strategy", "state"])
def test_immutable_writes_use_atomic_no_replace_under_a_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_kind: str,
) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    store.create_task(make_task())
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
    )
    state = ScientificState(goal="winner")
    destinations = {
        "episode": store.root / "evo_test/episodes/v001.json",
        "feedback": store.root / "evo_test/feedback/fb_test.json",
        "revision": store.root / "evo_test/revisions/rp_test.json",
        "strategy": store.root / "evo_test/strategies/strategy_test.json",
        "state": store.root / "evo_test/episodes/v001.scientific.json",
    }
    destination = destinations[record_kind]
    original_link = os.link

    def competing_link(source: str | Path, target: str | Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text('{"winner": true}', encoding="utf-8")
        original_link(source, target)

    monkeypatch.setattr("photomatagent.scientific.evolution.store.os.link", competing_link)

    with pytest.raises(EvolutionAlreadyExistsError):
        if record_kind == "episode":
            store.write_episode(make_episode())
        elif record_kind == "feedback":
            store.write_feedback(make_feedback())
        elif record_kind == "revision":
            store.write_revision(revision)
        elif record_kind == "strategy":
            store.write_strategy(strategy)
        else:
            store.write_scientific_state("evo_test", "v001", state)

    assert json.loads(destination.read_text(encoding="utf-8")) == {"winner": True}
    assert list(destination.parent.glob("*.tmp")) == []


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


@pytest.mark.parametrize(
    "record_kind",
    ["create_task", "save_task", "episode", "feedback", "revision", "strategy", "state"],
)
def test_store_revalidates_bypassed_models_immediately_before_writing(
    tmp_path: Path, record_kind: str
) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    if record_kind != "create_task":
        store.create_task(make_task())

    with pytest.raises(ValidationError):
        if record_kind == "create_task":
            store.create_task(make_task().model_copy(update={"status": "INVALID"}))
        elif record_kind == "save_task":
            store.save_task(
                make_task().model_copy(update={"status": "INVALID"}),
                expected_revision=0,
            )
        elif record_kind == "episode":
            store.write_episode(make_episode().model_copy(update={"status": "INVALID"}))
        elif record_kind == "feedback":
            store.write_feedback(
                make_feedback().model_copy(update={"rubric_version": "INVALID"})
            )
        elif record_kind == "revision":
            store.write_revision(
                RevisionPlan(
                    revision_id="rp_test",
                    evolution_id="evo_test",
                    source_version="v001",
                    feedback_id="fb_test",
                ).model_copy(update={"source_version": "INVALID"})
            )
        elif record_kind == "strategy":
            store.write_strategy(
                StrategyVersion(
                    strategy_id="strategy_test",
                    evolution_id="evo_test",
                    arm="STATIC",
                ).model_copy(update={"arm": "INVALID"})
            )
        else:
            store.write_scientific_state(
                "evo_test",
                "v001",
                ScientificState(goal="goal").model_copy(update={"hypotheses": [1]}),
            )


def test_store_revalidates_redacted_payload_before_committing(tmp_path: Path) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    store.create_task(make_task())
    strategy = StrategyVersion(
        strategy_id="sk-abcdefgh",
        evolution_id="evo_test",
        arm="STATIC",
    )

    with pytest.raises(ValidationError):
        store.write_strategy(strategy)

    assert not (store.root / "evo_test/strategies/sk-abcdefgh.json").exists()


def test_store_rejects_unknown_fields_injected_by_model_copy(tmp_path: Path) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    bypassed = make_task().model_copy(update={"unknown_persisted_field": True})

    with pytest.raises(ValidationError):
        store.create_task(bypassed)

    assert not (store.root / "evo_test/task.json").exists()


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


def test_lock_release_preserves_a_replacement_owner(tmp_path: Path) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    store.create_task(make_task())
    task_dir = store.root / "evo_test"
    lock_path = task_dir / ".lock"

    with store._task_lock(task_dir):
        lock_path.unlink()
        lock_path.write_text("replacement-owner", encoding="utf-8")

    assert lock_path.read_text(encoding="utf-8") == "replacement-owner"


def test_replaced_lock_does_not_suppress_the_protected_operation_error(
    tmp_path: Path,
) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    store.create_task(make_task())
    task_dir = store.root / "evo_test"
    lock_path = task_dir / ".lock"

    with pytest.raises(RuntimeError, match="protected failure"):
        with store._task_lock(task_dir):
            lock_path.unlink()
            lock_path.write_text("replacement-owner", encoding="utf-8")
            raise RuntimeError("protected failure")

    assert lock_path.read_text(encoding="utf-8") == "replacement-owner"


def test_lock_is_removed_when_a_locked_operation_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    store.create_task(make_task())

    def fail_write(path: Path, payload: object) -> None:
        raise OSError("write failed")

    monkeypatch.setattr(store, "_write_json_atomic", fail_write)

    with pytest.raises(OSError, match="write failed"):
        store.save_task(store.load_task("evo_test"), expected_revision=0)

    assert not (store.root / "evo_test/.lock").exists()


def test_lock_is_removed_when_owner_token_initialization_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    store.create_task(make_task())

    def fail_token_write(descriptor: int, data: bytes) -> int:
        raise OSError("token write failed")

    monkeypatch.setattr("photomatagent.scientific.evolution.store.os.write", fail_token_write)

    with pytest.raises(OSError, match="token write failed"):
        store.save_task(store.load_task("evo_test"), expected_revision=0)

    assert not (store.root / "evo_test/.lock").exists()


def test_concurrent_saves_allow_exactly_one_revision_winner(tmp_path: Path) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    store.create_task(make_task())
    barrier = Barrier(2)

    def save(status: str) -> EvolutionTask | Exception:
        candidate = store.load_task("evo_test")
        candidate.status = status  # type: ignore[assignment]
        barrier.wait()
        try:
            return store.save_task(candidate, expected_revision=0)
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(save, ["RUNNING", "STOPPED"]))

    assert sum(isinstance(result, EvolutionTask) for result in results) == 1
    assert sum(isinstance(result, EvolutionConflictError) for result in results) == 1
    assert store.load_task("evo_test").revision == 1


def test_interrupted_atomic_create_leaves_no_formal_or_temporary_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    task = store.create_task(make_task())

    def fail_replace(source: str | Path, destination: str | Path) -> None:
        raise OSError("simulated interruption before replace")

    monkeypatch.setattr(
        "photomatagent.scientific.evolution.store.os.replace", fail_replace
    )

    with pytest.raises(OSError, match="simulated interruption"):
        store.save_task(task, expected_revision=0)

    task_dir = tmp_path / ".photomatagent/evolutions/evo_test"
    assert store.load_task("evo_test").revision == 0
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


def test_episode_versions_accept_ascii_digits_only(tmp_path: Path) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    store.create_task(make_task())

    with pytest.raises(ValueError, match="vNNN"):
        store.write_scientific_state("evo_test", "v１２３", ScientificState())


def test_load_task_reports_malformed_and_unsupported_records_with_context(
    tmp_path: Path,
) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    malformed_path = store.root / "evo_malformed/task.json"
    malformed_path.parent.mkdir()
    malformed_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(EvolutionCorruptRecordError) as malformed:
        store.load_task("evo_malformed")
    assert malformed.value.path == malformed_path
    assert "evo_malformed/task.json" in str(malformed.value)

    unsupported_path = store.root / "evo_unsupported/task.json"
    unsupported_path.parent.mkdir()
    unsupported_path.write_text('{"schema_version": 2}', encoding="utf-8")
    with pytest.raises(EvolutionUnsupportedSchemaError) as unsupported:
        store.load_task("evo_unsupported")
    assert unsupported.value.path == unsupported_path
    assert unsupported.value.schema_version == 2
    assert "schema_version=2" in str(unsupported.value)


def test_load_scientific_state_reports_corrupt_json_with_path(tmp_path: Path) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    store.create_task(make_task())
    state_path = store.root / "evo_test/episodes/v001.scientific.json"
    state_path.parent.mkdir()
    state_path.write_text("[]", encoding="utf-8")

    with pytest.raises(EvolutionCorruptRecordError) as caught:
        store.load_scientific_state("evo_test", "v001")
    assert caught.value.path == state_path
    assert "v001.scientific.json" in str(caught.value)


def test_list_tasks_surfaces_first_corrupt_task_in_sorted_order(tmp_path: Path) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    later = store.root / "evo_b/task.json"
    earlier = store.root / "evo_a/task.json"
    later.parent.mkdir()
    earlier.parent.mkdir()
    later.write_text('{"schema_version": 2}', encoding="utf-8")
    earlier.write_text("not-json", encoding="utf-8")

    with pytest.raises(EvolutionCorruptRecordError) as caught:
        store.list_tasks()

    assert caught.value.path == earlier


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
