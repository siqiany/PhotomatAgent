from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
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
    assert created.event_outbox[0]["kind"] == "evolution_task_created"
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


def test_store_reports_lock_timeout_while_another_owner_holds_advisory_lock(
    tmp_path: Path,
) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    store.create_task(make_task())
    contender = EvolutionStore(Workspace(tmp_path))
    contender.lock_timeout_seconds = 0.01
    task_dir = store.root / "evo_test"

    with store._task_lock(task_dir):
        with pytest.raises(EvolutionLockError):
            contender.save_task(contender.load_task("evo_test"), expected_revision=0)

    assert (task_dir / ".lock").is_file()


def test_lock_release_keeps_one_persistent_lock_inode(tmp_path: Path) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    store.create_task(make_task())
    task_dir = store.root / "evo_test"
    lock_path = task_dir / ".lock"

    with store._task_lock(task_dir):
        first_inode = lock_path.stat().st_ino
    with store._task_lock(task_dir):
        second_inode = lock_path.stat().st_ino

    assert first_inode == second_inode


def test_lock_release_does_not_suppress_the_protected_operation_error(
    tmp_path: Path,
) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    store.create_task(make_task())
    task_dir = store.root / "evo_test"
    lock_path = task_dir / ".lock"

    with pytest.raises(RuntimeError, match="protected failure"):
        with store._task_lock(task_dir):
            raise RuntimeError("protected failure")

    assert lock_path.is_file()
    with store._task_lock(task_dir):
        pass


def test_lock_is_released_when_a_locked_operation_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    store.create_task(make_task())

    def fail_write(path: Path, payload: object) -> None:
        raise OSError("write failed")

    monkeypatch.setattr(store, "_write_json_atomic", fail_write)

    with pytest.raises(OSError, match="write failed"):
        store.save_task(store.load_task("evo_test"), expected_revision=0)

    task_dir = store.root / "evo_test"
    assert (task_dir / ".lock").is_file()
    with store._task_lock(task_dir):
        pass


def test_process_death_releases_lock_and_reservation_reconciles(
    tmp_path: Path,
) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    store.create_task(make_task())
    child_program = """
import time
from pathlib import Path
from photomatagent.scientific.evolution.models import EpisodeRecord
from photomatagent.scientific.evolution.service import EvolutionService
from photomatagent.scientific.evolution.store import EvolutionStore
from photomatagent.scientific.loop import TargetSpec
from photomatagent.workspace import Workspace

store = EvolutionStore(Workspace(Path(__import__('sys').argv[1])))
with store.transaction('evo_test') as transaction:
    task = transaction.load_task()
    version = 'v001'
    transaction.write_episode(EpisodeRecord(
        evolution_id=task.evolution_id,
        episode_id=EvolutionService._episode_id(task.evolution_id, version),
        version=version,
        task_snapshot=task.model_dump(mode='json'),
        target_snapshot=TargetSpec.model_validate(task.target),
        execution_mode='NORMAL',
    ))
    print('READY', flush=True)
    time.sleep(60)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", child_program, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "READY"
        child.kill()
        child.wait(timeout=5)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)

    store.lock_timeout_seconds = 0.2
    from photomatagent.scientific.evolution.service import EvolutionService

    reconciled = EvolutionService(store).reserve_episode("evo_test", mode="NORMAL")

    assert reconciled.entity.version == "v001"
    assert reconciled.entity.status == "RESERVED"
    assert store.load_task("evo_test").status == "RUNNING"


def test_concurrent_saves_allow_exactly_one_revision_winner(tmp_path: Path) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    store.create_task(make_task())
    barrier = Barrier(2)

    def save(status: str) -> EvolutionTask | Exception:
        current = store.load_task("evo_test")
        updates: dict[str, object] = {"status": status}
        if status == "STOPPED":
            updates["resume_status"] = "CREATED"
        candidate = current.model_copy(update=updates)
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


def test_load_task_rejects_record_whose_identity_differs_from_directory(
    tmp_path: Path,
) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    task_path = store.root / "evo_requested/task.json"
    task_path.parent.mkdir()
    task_path.write_text(make_task("evo_stored").model_dump_json(), encoding="utf-8")

    with pytest.raises(EvolutionCorruptRecordError) as caught:
        store.load_task("evo_requested")

    assert caught.value.path == task_path
    assert "requested='evo_requested'" in str(caught.value)
    assert "stored='evo_stored'" in str(caught.value)


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


def test_list_tasks_deterministically_surfaces_mismatched_task_identity(
    tmp_path: Path,
) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    mismatch = store.root / "evo_a/task.json"
    mismatch.parent.mkdir()
    mismatch.write_text(make_task("evo_other").model_dump_json(), encoding="utf-8")
    store.create_task(make_task("evo_b"))

    with pytest.raises(EvolutionCorruptRecordError) as caught:
        store.list_tasks()

    assert caught.value.path == mismatch


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


def test_episode_load_and_legal_status_transitions_round_trip(tmp_path: Path) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    store.create_task(make_task())
    reserved = make_episode()
    store.write_episode(reserved)

    running = store.transition_episode(
        reserved.model_copy(
            update={"status": "RUNNING", "started_at": "2026-09-04T08:00:00Z"}
        ),
        expected_status="RESERVED",
    )
    completed = store.transition_episode(
        running.model_copy(update={"status": "COMPLETED"}),
        expected_status="RUNNING",
    )

    assert store.load_episode("evo_test", "v001") == completed
    assert completed.status == "COMPLETED"


def test_episode_transition_rejects_illegal_and_stale_statuses(tmp_path: Path) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    store.create_task(make_task())
    reserved = make_episode()
    store.write_episode(reserved)

    with pytest.raises(EvolutionConflictError, match="illegal episode transition"):
        store.transition_episode(
            reserved.model_copy(update={"status": "COMPLETED"}),
            expected_status="RESERVED",
        )

    running = store.transition_episode(
        reserved.model_copy(
            update={"status": "RUNNING", "started_at": "2026-09-04T08:00:00Z"}
        ),
        expected_status="RESERVED",
    )
    with pytest.raises(EvolutionConflictError, match="stored status=RUNNING"):
        store.transition_episode(
            running.model_copy(update={"status": "FAILED"}),
            expected_status="RESERVED",
        )


def test_episode_transition_rejects_execution_snapshot_mutation(tmp_path: Path) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    store.create_task(make_task())
    reserved = make_episode()
    store.write_episode(reserved)
    forged = reserved.model_copy(
        update={"status": "RUNNING", "execution_mode": "FRESH_EVALUATION"}
    )

    with pytest.raises(EvolutionConflictError, match="execution_mode"):
        store.transition_episode(forged, expected_status="RESERVED")

    assert store.load_episode("evo_test", "v001") == reserved


@pytest.mark.parametrize("terminal_status", ["COMPLETED", "FAILED"])
def test_terminal_episode_records_are_permanently_immutable(
    tmp_path: Path, terminal_status: str
) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    store.create_task(make_task())
    reserved = make_episode()
    store.write_episode(reserved)
    source = reserved
    expected_status = "RESERVED"
    if terminal_status == "COMPLETED":
        source = store.transition_episode(
            reserved.model_copy(
                update={"status": "RUNNING", "started_at": "2026-09-04T08:00:00Z"}
            ),
            expected_status="RESERVED",
        )
        expected_status = "RUNNING"
    terminal = store.transition_episode(
        source.model_copy(update={"status": terminal_status}),  # type: ignore[arg-type]
        expected_status=expected_status,  # type: ignore[arg-type]
    )

    with pytest.raises(EvolutionConflictError, match="terminal"):
        store.transition_episode(
            terminal.model_copy(update={"error": "changed"}),
            expected_status=terminal_status,  # type: ignore[arg-type]
        )


def test_concurrent_episode_terminal_transitions_allow_exactly_one_winner(
    tmp_path: Path,
) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    store.create_task(make_task())
    reserved = make_episode()
    store.write_episode(reserved)
    running = store.transition_episode(
        reserved.model_copy(
            update={"status": "RUNNING", "started_at": "2026-09-04T08:00:00Z"}
        ),
        expected_status="RESERVED",
    )
    barrier = Barrier(2)

    def finish(status: str) -> EpisodeRecord | Exception:
        candidate = running.model_copy(update={"status": status})
        barrier.wait()
        try:
            return store.transition_episode(  # type: ignore[arg-type]
                candidate,
                expected_status="RUNNING",
            )
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(finish, ["COMPLETED", "FAILED"]))

    assert sum(isinstance(result, EpisodeRecord) for result in results) == 1
    assert sum(isinstance(result, EvolutionConflictError) for result in results) == 1
    assert store.load_episode("evo_test", "v001").status in {"COMPLETED", "FAILED"}


def test_task_transaction_composes_record_and_manifest_without_reentrant_lock(
    tmp_path: Path,
) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    store.create_task(make_task())
    episode = make_episode()

    with store.transaction("evo_test") as transaction:
        task = transaction.load_task()
        transaction.write_episode(episode)
        saved = transaction.save_task(
            task.model_copy(
                update={
                    "status": "RUNNING",
                    "current_version": "v001",
                    "episode_ids": [episode.episode_id],
                }
            ),
            expected_revision=task.revision,
        )

    assert saved.revision == 1
    assert store.load_episode("evo_test", "v001") == episode


def test_task_transaction_cannot_be_reused_after_lock_release(tmp_path: Path) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    store.create_task(make_task())

    with store.transaction("evo_test") as transaction:
        assert transaction.load_task().evolution_id == "evo_test"

    with pytest.raises(EvolutionLockError, match="no longer active"):
        transaction.load_task()


def test_running_provenance_cannot_change_during_terminal_transition(
    tmp_path: Path,
) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    store.create_task(make_task())
    reserved = make_episode()
    store.write_episode(reserved)
    running = store.transition_episode(
        reserved.model_copy(
            update={
                "status": "RUNNING",
                "runtime_session_id": "session_one",
                "event_log_path": ".photomatagent/sessions/session_one/events.jsonl",
                "started_at": "2026-09-04T08:00:00Z",
            }
        ),
        expected_status="RESERVED",
    )

    for field, replacement in (
        ("runtime_session_id", "session_two"),
        ("event_log_path", "changed/events.jsonl"),
        ("started_at", "2026-09-04T09:00:00Z"),
    ):
        with pytest.raises(EvolutionConflictError, match=field):
            store.transition_episode(
                running.model_copy(
                    update={"status": "FAILED", field: replacement}
                ),
                expected_status="RUNNING",
            )


def test_episode_transition_enforces_status_specific_runtime_provenance(
    tmp_path: Path,
) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    store.create_task(make_task())
    reserved = make_episode()
    store.write_episode(reserved)

    with pytest.raises(EvolutionConflictError, match="started_at"):
        store.transition_episode(
            reserved.model_copy(update={"status": "RUNNING"}),
            expected_status="RESERVED",
        )
    with pytest.raises(EvolutionConflictError, match="runtime_session_id"):
        store.transition_episode(
            reserved.model_copy(
                update={
                    "status": "FAILED",
                    "runtime_session_id": "session_that_never_started",
                }
            ),
            expected_status="RESERVED",
        )


def test_store_loads_feedback_and_revision_with_managed_identity(tmp_path: Path) -> None:
    store = EvolutionStore(Workspace(tmp_path))
    store.create_task(make_task())
    feedback = make_feedback()
    revision = RevisionPlan(
        revision_id="rp_test",
        evolution_id="evo_test",
        source_version="v001",
        feedback_id=feedback.feedback_id,
    )
    store.write_feedback(feedback)
    store.write_revision(revision)

    loaded_feedback = store.load_feedback("evo_test", feedback.feedback_id)
    assert loaded_feedback.feedback_id == feedback.feedback_id
    assert "super-secret-value" not in loaded_feedback.raw_input
    assert store.load_revision("evo_test", revision.revision_id) == revision
