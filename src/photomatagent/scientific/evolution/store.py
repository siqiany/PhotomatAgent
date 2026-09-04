"""Workspace-contained, atomic persistence for scientific evolution records."""

from __future__ import annotations

import errno
import importlib
import json
import os
import re
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TypeVar

from pydantic import BaseModel, ValidationError

from photomatagent.redaction import redact_secrets
from photomatagent.scientific.evolution.models import (
    EpisodeRecord,
    EpisodeStatus,
    EvolutionTask,
    ExpertFeedbackRecord,
    FeedbackCompilation,
    RevisionPlan,
    StrategyVersion,
    utc_now,
    validate_managed_id,
)
from photomatagent.scientific.state import ScientificState
from photomatagent.workspace import Workspace

_ADVISORY_LOCK_API: Any = None
if os.name == "nt":
    _ADVISORY_LOCK_API = importlib.import_module("msvcrt")
elif os.name == "posix":
    _ADVISORY_LOCK_API = importlib.import_module("fcntl")

_STORE_PATH = ".photomatagent/evolutions"
_EPISODE_VERSION = re.compile(r"^v[0-9]{3}$")
_ModelT = TypeVar("_ModelT", bound=BaseModel)
_TASK_MUTABLE_FIELDS = (
    "status",
    "resume_status",
    "current_version",
    "last_completed_version",
    "accepted_version",
    "episode_ids",
    "feedback_ids",
    "compilation_ids",
    "revision_ids",
    "strategy_ids",
    "comparison_ids",
    "experience_ids",
)
_TASK_IMMUTABLE_FIELDS = (
    "goal",
    "target",
    "task_group_id",
    "input_sha256",
    "created_at",
)
_EPISODE_IMMUTABLE_FIELDS = (
    "evolution_id",
    "episode_id",
    "version",
    "parent_version",
    "applied_feedback_id",
    "revision_plan_id",
    "execution_mode",
    "strategy_id",
    "strategy_arm",
    "task_snapshot",
    "target_snapshot",
    "provider",
    "model",
    "tool_surface_fingerprint",
    "capability_fingerprint",
    "data_source_fingerprints",
    "created_at",
)
_EPISODE_TRANSITIONS: dict[EpisodeStatus, frozenset[EpisodeStatus]] = {
    "RESERVED": frozenset({"RUNNING", "FAILED"}),
    "RUNNING": frozenset({"COMPLETED", "FAILED"}),
    "COMPLETED": frozenset(),
    "FAILED": frozenset(),
}
_EPISODE_RUNNING_PROVENANCE_FIELDS = (
    "runtime_session_id",
    "event_log_path",
    "started_at",
)


class EvolutionStoreError(RuntimeError):
    """Base error for evolution persistence failures."""


class EvolutionAlreadyExistsError(EvolutionStoreError):
    """Raised when an immutable evolution record already exists."""


class EvolutionConflictError(EvolutionStoreError):
    """Raised when a task save uses a stale expected revision."""


class EvolutionLockError(EvolutionStoreError):
    """Raised when a task lock cannot be acquired before its deadline."""


class EvolutionCorruptRecordError(EvolutionStoreError):
    """Raised when a persisted evolution record cannot be decoded or validated."""

    def __init__(self, path: Path, detail: str) -> None:
        self.path = path
        self.detail = detail
        super().__init__(f"corrupt evolution record at {path}: {detail}")


class EvolutionUnsupportedSchemaError(EvolutionStoreError):
    """Raised when a persisted record declares an unsupported schema version."""

    def __init__(self, path: Path, schema_version: Any) -> None:
        self.path = path
        self.schema_version = schema_version
        super().__init__(
            f"unsupported evolution record at {path}: "
            f"schema_version={schema_version!r}"
        )


class EvolutionTransaction:
    """Operations performed while one evolution task lock is held."""

    def __init__(self, store: EvolutionStore, evolution_id: str) -> None:
        self.store = store
        self.evolution_id = evolution_id
        self._active = False

    def _require_active(self) -> None:
        if not self._active:
            raise EvolutionLockError("evolution transaction is no longer active")

    def load_task(self) -> EvolutionTask:
        self._require_active()
        return self.store.load_task(self.evolution_id)

    def save_task(
        self,
        task: EvolutionTask,
        *,
        expected_revision: int,
    ) -> EvolutionTask:
        self._require_active()
        return self.store._save_task_locked(task, expected_revision)

    def load_episode(self, version: str) -> EpisodeRecord:
        self._require_active()
        return self.store.load_episode(self.evolution_id, version)

    def write_episode(self, episode: EpisodeRecord) -> Path:
        self._require_active()
        if episode.evolution_id != self.evolution_id:
            raise EvolutionConflictError("episode belongs to a different transaction")
        self.store._validate_episode_version(episode.version)
        return self.store._write_record_locked(
            evolution_id=self.evolution_id,
            directory="episodes",
            filename=f"{episode.version}.json",
            record=episode,
            model_type=EpisodeRecord,
        )

    def transition_episode(
        self,
        episode: EpisodeRecord,
        *,
        expected_status: EpisodeStatus,
    ) -> EpisodeRecord:
        self._require_active()
        return self.store._transition_episode_locked(episode, expected_status)

    def load_feedback(self, feedback_id: str) -> ExpertFeedbackRecord:
        self._require_active()
        return self.store.load_feedback(self.evolution_id, feedback_id)

    def write_feedback(self, feedback: ExpertFeedbackRecord) -> Path:
        self._require_active()
        if feedback.evolution_id != self.evolution_id:
            raise EvolutionConflictError("feedback belongs to a different transaction")
        return self.store._write_record_locked(
            evolution_id=self.evolution_id,
            directory="feedback",
            filename=f"{feedback.feedback_id}.json",
            record=feedback,
            model_type=ExpertFeedbackRecord,
        )

    def load_compilation(self, compilation_id: str) -> FeedbackCompilation:
        self._require_active()
        return self.store.load_compilation(self.evolution_id, compilation_id)

    def write_compilation(self, compilation: FeedbackCompilation) -> Path:
        self._require_active()
        if compilation.evolution_id != self.evolution_id:
            raise EvolutionConflictError(
                "compilation belongs to a different transaction"
            )
        if compilation.compilation_id is None:
            raise ValueError("persisted compilation requires compilation_id")
        return self.store._write_record_locked(
            evolution_id=self.evolution_id,
            directory="compilations",
            filename=f"{compilation.compilation_id}.json",
            record=compilation,
            model_type=FeedbackCompilation,
        )

    def load_revision(self, revision_id: str) -> RevisionPlan:
        self._require_active()
        return self.store.load_revision(self.evolution_id, revision_id)

    def write_revision(self, revision: RevisionPlan) -> Path:
        self._require_active()
        if revision.evolution_id != self.evolution_id:
            raise EvolutionConflictError("revision belongs to a different transaction")
        return self.store._write_record_locked(
            evolution_id=self.evolution_id,
            directory="revisions",
            filename=f"{revision.revision_id}.json",
            record=revision,
            model_type=RevisionPlan,
        )


class EvolutionStore:
    """Persist evolution tasks and immutable records inside one workspace."""

    lock_timeout_seconds = 5.0
    lock_poll_seconds = 0.05

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.root = workspace.resolve(_STORE_PATH, must_exist=False)
        self.root.mkdir(parents=True, exist_ok=True)
        self.root = workspace.resolve(_STORE_PATH, must_exist=True)

    @contextmanager
    def transaction(self, evolution_id: str) -> Iterator[EvolutionTransaction]:
        """Hold the authoritative task lock across one logical mutation."""

        task_dir = self._task_dir(evolution_id)
        transaction = EvolutionTransaction(self, evolution_id)
        with self._task_lock(task_dir):
            transaction._active = True
            try:
                yield transaction
            finally:
                transaction._active = False

    def create_task(self, task: EvolutionTask) -> EvolutionTask:
        """Create a new revision-zero task without replacing an existing task."""
        validated, payload = self._prepare_model(task, EvolutionTask)
        if validated.revision != 0:
            raise ValueError("new evolution tasks must start at revision 0")
        task_dir = self._task_dir(validated.evolution_id, create=True)
        with self._task_lock(task_dir):
            path = self._task_path(validated.evolution_id)
            self._write_immutable_json(path, payload)
        return validated

    def load_task(self, evolution_id: str) -> EvolutionTask:
        """Load and validate the authoritative task record."""
        path = self._task_path(evolution_id)
        task = self._load_model(path, EvolutionTask, require_schema_version=True)
        if task.evolution_id != evolution_id:
            raise EvolutionCorruptRecordError(
                path,
                "task identity does not match its managed directory: "
                f"requested={evolution_id!r}, stored={task.evolution_id!r}",
            )
        return task

    def save_task(
        self, task: EvolutionTask, expected_revision: int
    ) -> EvolutionTask:
        """Save a task only if its stored revision matches the caller's view."""
        task_dir = self._task_dir(task.evolution_id)
        with self._task_lock(task_dir):
            return self._save_task_locked(task, expected_revision)

    def _save_task_locked(
        self,
        task: EvolutionTask,
        expected_revision: int,
    ) -> EvolutionTask:
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 0
        ):
            raise ValueError("expected_revision must be a non-negative integer")
        candidate, _ = self._prepare_model(task, EvolutionTask)
        if candidate.revision != expected_revision:
            raise EvolutionConflictError(
                f"stale evolution task write for {candidate.evolution_id}: "
                f"caller revision={candidate.revision}, "
                f"expected={expected_revision}"
            )
        current = self.load_task(candidate.evolution_id)
        if current.revision != expected_revision:
            raise EvolutionConflictError(
                f"stale evolution task write for {candidate.evolution_id}: "
                f"stored revision={current.revision}, "
                f"expected={expected_revision}"
            )
        for field in _TASK_IMMUTABLE_FIELDS:
            if getattr(candidate, field) != getattr(current, field):
                raise EvolutionConflictError(
                    f"immutable task field differs from stored record for "
                    f"{candidate.evolution_id}: {field}"
                )
        updated_data = current.model_dump(mode="python")
        for field in _TASK_MUTABLE_FIELDS:
            updated_data[field] = getattr(candidate, field)
        updated_data["revision"] = current.revision + 1
        updated_data["updated_at"] = utc_now()
        updated = EvolutionTask.model_validate(updated_data)
        updated, payload = self._prepare_model(updated, EvolutionTask)
        self._write_json_atomic(
            self._task_path(candidate.evolution_id),
            payload,
        )
        return updated

    def write_episode(self, episode: EpisodeRecord) -> Path:
        """Persist an immutable episode record under its monotonic version."""
        self._validate_episode_version(episode.version)
        return self._write_record(
            evolution_id=episode.evolution_id,
            directory="episodes",
            filename=f"{episode.version}.json",
            record=episode,
            model_type=EpisodeRecord,
        )

    def load_episode(self, evolution_id: str, version: str) -> EpisodeRecord:
        """Load one validated episode and bind it to its managed path identity."""
        self._validate_episode_version(version)
        path = self._managed_path(evolution_id, "episodes", f"{version}.json")
        episode = self._load_model(
            path,
            EpisodeRecord,
            require_schema_version=True,
        )
        if episode.evolution_id != evolution_id or episode.version != version:
            raise EvolutionCorruptRecordError(
                path,
                "episode identity does not match its managed path: "
                f"requested={evolution_id!r}/{version!r}, "
                f"stored={episode.evolution_id!r}/{episode.version!r}",
            )
        return episode

    def transition_episode(
        self,
        episode: EpisodeRecord,
        *,
        expected_status: EpisodeStatus,
    ) -> EpisodeRecord:
        """Atomically advance a nonterminal episode from its expected status."""
        task_dir = self._task_dir(episode.evolution_id)
        with self._task_lock(task_dir):
            return self._transition_episode_locked(episode, expected_status)

    def _transition_episode_locked(
        self,
        episode: EpisodeRecord,
        expected_status: EpisodeStatus,
    ) -> EpisodeRecord:
        candidate, payload = self._prepare_model(episode, EpisodeRecord)
        if expected_status not in _EPISODE_TRANSITIONS:
            raise ValueError(f"unsupported expected episode status: {expected_status!r}")
        current = self.load_episode(candidate.evolution_id, candidate.version)
        if current.status != expected_status:
            raise EvolutionConflictError(
                f"stale episode transition for {candidate.evolution_id}/"
                f"{candidate.version}: stored status={current.status}, "
                f"expected={expected_status}"
            )
        if not _EPISODE_TRANSITIONS[current.status]:
            raise EvolutionConflictError(
                f"terminal episode is immutable: {candidate.evolution_id}/"
                f"{candidate.version} status={current.status}"
            )
        if candidate.status not in _EPISODE_TRANSITIONS[current.status]:
            raise EvolutionConflictError(
                f"illegal episode transition for {candidate.evolution_id}/"
                f"{candidate.version}: {current.status} -> {candidate.status}"
            )
        for field in _EPISODE_IMMUTABLE_FIELDS:
            if getattr(candidate, field) != getattr(current, field):
                raise EvolutionConflictError(
                    "immutable episode execution snapshot differs from stored "
                    f"record for {candidate.evolution_id}/{candidate.version}: "
                    f"{field}"
                )
        if current.status == "RESERVED" and candidate.status == "RUNNING":
            if candidate.started_at is None:
                raise EvolutionConflictError(
                    "RUNNING episode transition requires started_at provenance"
                )
            for field in (
                "completed_at",
                "summary",
                "artifact",
                "error",
            ):
                if getattr(candidate, field) != getattr(current, field):
                    raise EvolutionConflictError(
                        f"RUNNING transition cannot set terminal field {field}"
                    )
        if current.status == "RESERVED" and candidate.status == "FAILED":
            for field in _EPISODE_RUNNING_PROVENANCE_FIELDS:
                if getattr(candidate, field) != getattr(current, field):
                    raise EvolutionConflictError(
                        "unstarted FAILED episode cannot add runtime provenance "
                        f"field {field}"
                    )
        if current.status == "RUNNING":
            for field in _EPISODE_RUNNING_PROVENANCE_FIELDS:
                if getattr(candidate, field) != getattr(current, field):
                    raise EvolutionConflictError(
                        "running episode provenance differs from stored record for "
                        f"{candidate.evolution_id}/{candidate.version}: {field}"
                    )
        self._write_json_atomic(
            self._managed_path(
                candidate.evolution_id,
                "episodes",
                f"{candidate.version}.json",
            ),
            payload,
        )
        return candidate

    def load_feedback(
        self,
        evolution_id: str,
        feedback_id: str,
    ) -> ExpertFeedbackRecord:
        """Load one feedback record bound to its managed identity."""

        self._validate_id(feedback_id)
        path = self._managed_path(evolution_id, "feedback", f"{feedback_id}.json")
        feedback = self._load_model(
            path,
            ExpertFeedbackRecord,
            require_schema_version=True,
        )
        if feedback.evolution_id != evolution_id or feedback.feedback_id != feedback_id:
            raise EvolutionCorruptRecordError(
                path,
                "feedback identity does not match its managed path",
            )
        return feedback

    def list_feedback(self, evolution_id: str) -> list[ExpertFeedbackRecord]:
        """Load every managed feedback record for duplicate-active checks."""

        directory = self._managed_path(evolution_id, "feedback")
        if not directory.is_dir():
            return []
        records: list[ExpertFeedbackRecord] = []
        for path in sorted(directory.glob("*.json")):
            try:
                self._validate_id(path.stem)
            except (TypeError, ValueError):
                continue
            records.append(self.load_feedback(evolution_id, path.stem))
        return records

    def load_compilation(
        self,
        evolution_id: str,
        compilation_id: str,
    ) -> FeedbackCompilation:
        """Load one immutable compilation bound to its managed identity."""

        self._validate_id(compilation_id)
        path = self._managed_path(
            evolution_id,
            "compilations",
            f"{compilation_id}.json",
        )
        compilation = self._load_model(
            path,
            FeedbackCompilation,
            require_schema_version=True,
        )
        if (
            compilation.evolution_id != evolution_id
            or compilation.compilation_id != compilation_id
        ):
            raise EvolutionCorruptRecordError(
                path,
                "compilation identity does not match its managed path",
            )
        return compilation

    def list_compilations(self, evolution_id: str) -> list[FeedbackCompilation]:
        """Load every immutable compilation attempt for one task."""

        directory = self._managed_path(evolution_id, "compilations")
        if not directory.is_dir():
            return []
        records: list[FeedbackCompilation] = []
        for path in sorted(directory.glob("*.json")):
            try:
                self._validate_id(path.stem)
            except (TypeError, ValueError):
                continue
            records.append(self.load_compilation(evolution_id, path.stem))
        return records

    def load_revision(self, evolution_id: str, revision_id: str) -> RevisionPlan:
        """Load one revision plan bound to its managed identity."""

        self._validate_id(revision_id)
        path = self._managed_path(evolution_id, "revisions", f"{revision_id}.json")
        revision = self._load_model(
            path,
            RevisionPlan,
            require_schema_version=True,
        )
        if revision.evolution_id != evolution_id or revision.revision_id != revision_id:
            raise EvolutionCorruptRecordError(
                path,
                "revision identity does not match its managed path",
            )
        return revision

    def list_revisions(self, evolution_id: str) -> list[RevisionPlan]:
        """Load every managed revision record for stable-operation checks."""

        directory = self._managed_path(evolution_id, "revisions")
        if not directory.is_dir():
            return []
        records: list[RevisionPlan] = []
        for path in sorted(directory.glob("*.json")):
            try:
                self._validate_id(path.stem)
            except (TypeError, ValueError):
                continue
            records.append(self.load_revision(evolution_id, path.stem))
        return records

    def _write_record_locked(
        self,
        *,
        evolution_id: str,
        directory: str,
        filename: str,
        record: _ModelT,
        model_type: type[_ModelT],
    ) -> Path:
        _, payload = self._prepare_model(record, model_type)
        self._require_task(evolution_id)
        path = self._managed_path(evolution_id, directory, filename)
        self._write_immutable_json(path, payload)
        return path

    def write_feedback(self, feedback: ExpertFeedbackRecord) -> Path:
        """Persist one immutable expert-feedback record."""
        self._validate_id(feedback.feedback_id)
        return self._write_record(
            evolution_id=feedback.evolution_id,
            directory="feedback",
            filename=f"{feedback.feedback_id}.json",
            record=feedback,
            model_type=ExpertFeedbackRecord,
        )

    def write_compilation(self, compilation: FeedbackCompilation) -> Path:
        """Persist one immutable feedback-compilation attempt."""

        if compilation.compilation_id is None:
            raise ValueError("persisted compilation requires compilation_id")
        self._validate_id(compilation.compilation_id)
        if compilation.evolution_id is None:
            raise ValueError("persisted compilation requires evolution_id")
        return self._write_record(
            evolution_id=compilation.evolution_id,
            directory="compilations",
            filename=f"{compilation.compilation_id}.json",
            record=compilation,
            model_type=FeedbackCompilation,
        )

    def write_revision(self, revision: RevisionPlan) -> Path:
        """Persist one immutable revision plan."""
        self._validate_id(revision.revision_id)
        return self._write_record(
            evolution_id=revision.evolution_id,
            directory="revisions",
            filename=f"{revision.revision_id}.json",
            record=revision,
            model_type=RevisionPlan,
        )

    def write_strategy(self, strategy: StrategyVersion) -> Path:
        """Persist one immutable strategy snapshot."""
        self._validate_id(strategy.strategy_id)
        return self._write_record(
            evolution_id=strategy.evolution_id,
            directory="strategies",
            filename=f"{strategy.strategy_id}.json",
            record=strategy,
            model_type=StrategyVersion,
        )

    def write_scientific_state(
        self,
        evolution_id: str,
        version: str,
        state: ScientificState,
    ) -> Path:
        """Persist the immutable scientific-state snapshot for one episode."""
        self._validate_episode_version(version)
        _, payload = self._prepare_model(state, ScientificState)
        task_dir = self._task_dir(evolution_id)
        with self._task_lock(task_dir):
            self._require_task(evolution_id)
            path = self._managed_path(
                evolution_id,
                "episodes",
                f"{version}.scientific.json",
            )
            self._write_immutable_json(path, payload)
        return path

    def load_scientific_state(
        self, evolution_id: str, version: str
    ) -> ScientificState:
        """Load and validate one persisted scientific-state snapshot."""
        self._validate_episode_version(version)
        path = self._managed_path(
            evolution_id,
            "episodes",
            f"{version}.scientific.json",
        )
        return self._load_model(path, ScientificState, require_schema_version=False)

    def list_tasks(self) -> list[EvolutionTask]:
        """Return tasks by ID; surface the first corrupt managed task in that order.

        Non-task files, unmanaged names, and directories without ``task.json`` are
        ignored. A present but unreadable or invalid task record is never skipped.
        """
        tasks: list[EvolutionTask] = []
        for entry in sorted(self.root.iterdir(), key=lambda path: path.name):
            try:
                validate_managed_id(entry.name)
            except (TypeError, ValueError):
                continue
            if not entry.is_dir():
                continue
            task_path = self._task_path(entry.name)
            if task_path.is_file():
                tasks.append(self.load_task(entry.name))
        return tasks

    def _write_record(
        self,
        *,
        evolution_id: str,
        directory: str,
        filename: str,
        record: _ModelT,
        model_type: type[_ModelT],
    ) -> Path:
        _, payload = self._prepare_model(record, model_type)
        task_dir = self._task_dir(evolution_id)
        with self._task_lock(task_dir):
            self._require_task(evolution_id)
            path = self._managed_path(evolution_id, directory, filename)
            self._write_immutable_json(path, payload)
        return path

    def _require_task(self, evolution_id: str) -> None:
        if not self._task_path(evolution_id).is_file():
            raise FileNotFoundError(f"evolution task does not exist: {evolution_id}")

    def _task_path(self, evolution_id: str) -> Path:
        return self._managed_path(evolution_id, "task.json")

    def _task_dir(self, evolution_id: str, *, create: bool = False) -> Path:
        self._validate_id(evolution_id)
        directory = self.workspace.resolve(
            f"{_STORE_PATH}/{evolution_id}", must_exist=False
        )
        if create:
            directory.mkdir(parents=False, exist_ok=True)
        if not directory.is_dir():
            raise FileNotFoundError(f"evolution task does not exist: {evolution_id}")
        return self.workspace.resolve(
            f"{_STORE_PATH}/{evolution_id}", must_exist=True
        )

    def _managed_path(self, evolution_id: str, *parts: str) -> Path:
        self._validate_id(evolution_id)
        relative = "/".join((_STORE_PATH, evolution_id, *parts))
        return self.workspace.resolve(relative, must_exist=False)

    @staticmethod
    def _validate_id(value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("managed IDs must be strings")
        return validate_managed_id(value)

    @staticmethod
    def _validate_episode_version(version: str) -> str:
        if not isinstance(version, str) or _EPISODE_VERSION.fullmatch(version) is None:
            raise ValueError("episode versions must use the form vNNN")
        return version

    @contextmanager
    def _task_lock(self, task_dir: Path) -> Iterator[None]:
        lock_path = self.workspace.resolve(
            self.workspace.relative(task_dir / ".lock"), must_exist=False
        )
        deadline = time.monotonic() + self.lock_timeout_seconds
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        acquired = False
        try:
            self._initialize_lock_file(descriptor)
            while not acquired:
                acquired = self._try_advisory_lock(descriptor)
                if acquired:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise EvolutionLockError(
                        f"timed out acquiring evolution task lock: {task_dir.name}"
                    )
                time.sleep(min(self.lock_poll_seconds, remaining))
            yield
        finally:
            try:
                if acquired:
                    self._release_advisory_lock(descriptor)
            finally:
                os.close(descriptor)

    @staticmethod
    def _initialize_lock_file(descriptor: int) -> None:
        """Ensure the persistent lock inode has a byte for Windows locking."""

        if os.fstat(descriptor).st_size == 0:
            os.lseek(descriptor, 0, os.SEEK_SET)
            written = os.write(descriptor, b"\0")
            if written != 1:
                raise OSError("evolution lock file initialization made no progress")
            os.fsync(descriptor)

    @staticmethod
    def _try_advisory_lock(descriptor: int) -> bool:
        if os.name == "nt":
            os.lseek(descriptor, 0, os.SEEK_SET)
            try:
                _ADVISORY_LOCK_API.locking(
                    descriptor,
                    _ADVISORY_LOCK_API.LK_NBLCK,
                    1,
                )
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    return False
                raise
            return True
        if os.name == "posix":
            try:
                _ADVISORY_LOCK_API.flock(
                    descriptor,
                    _ADVISORY_LOCK_API.LOCK_EX | _ADVISORY_LOCK_API.LOCK_NB,
                )
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EAGAIN}:
                    return False
                raise
            return True
        raise EvolutionLockError(
            f"advisory evolution locking is unsupported on os.name={os.name!r}"
        )

    @staticmethod
    def _release_advisory_lock(descriptor: int) -> None:
        if os.name == "nt":
            os.lseek(descriptor, 0, os.SEEK_SET)
            _ADVISORY_LOCK_API.locking(
                descriptor,
                _ADVISORY_LOCK_API.LK_UNLCK,
                1,
            )
            return
        if os.name == "posix":
            _ADVISORY_LOCK_API.flock(descriptor, _ADVISORY_LOCK_API.LOCK_UN)
            return
        raise EvolutionLockError(
            f"advisory evolution locking is unsupported on os.name={os.name!r}"
        )

    def _write_immutable_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path = self.workspace.resolve(self.workspace.relative(path), must_exist=False)
        temporary_path = self._write_temporary_json(path, payload)
        linked = False
        try:
            try:
                os.link(temporary_path, path)
                linked = True
            except FileExistsError as exc:
                raise EvolutionAlreadyExistsError(
                    f"immutable evolution record already exists: {path.name}"
                ) from exc
            self._fsync_directory(path.parent)
        finally:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            if linked:
                self._fsync_directory(path.parent)

    def _write_json_atomic(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path = self.workspace.resolve(self.workspace.relative(path), must_exist=False)
        temporary_path = self._write_temporary_json(path, payload)
        try:
            os.replace(temporary_path, path)
            self._fsync_directory(path.parent)
        finally:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _write_temporary_json(path: Path, payload: Any) -> Path:
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                descriptor = -1
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            raise
        return temporary_path

    @staticmethod
    def _prepare_model(
        record: _ModelT,
        model_type: type[_ModelT],
    ) -> tuple[_ModelT, dict[str, Any]]:
        untrusted = record.model_dump(mode="python", warnings=False)
        declared_fields = set(model_type.model_fields)
        untrusted.update(
            {
                key: value
                for key, value in vars(record).items()
                if key not in declared_fields
            }
        )
        validated = model_type.model_validate(untrusted)
        redacted = redact_secrets(validated.model_dump(mode="json"))
        persisted = model_type.model_validate(redacted)
        return persisted, persisted.model_dump(mode="json")

    @staticmethod
    def _load_model(
        path: Path,
        model_type: type[_ModelT],
        *,
        require_schema_version: bool,
    ) -> _ModelT:
        try:
            text = path.read_text(encoding="utf-8")
            payload = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EvolutionCorruptRecordError(path, str(exc)) from exc
        if (
            require_schema_version
            and isinstance(payload, dict)
            and "schema_version" in payload
            and payload["schema_version"] != 1
        ):
            raise EvolutionUnsupportedSchemaError(path, payload["schema_version"])
        try:
            return model_type.model_validate(payload)
        except (TypeError, ValueError, ValidationError) as exc:
            raise EvolutionCorruptRecordError(path, str(exc)) from exc

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


__all__ = [
    "EvolutionAlreadyExistsError",
    "EvolutionConflictError",
    "EvolutionCorruptRecordError",
    "EvolutionLockError",
    "EvolutionStore",
    "EvolutionStoreError",
    "EvolutionUnsupportedSchemaError",
]
