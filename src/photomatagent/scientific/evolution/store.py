"""Workspace-contained, atomic persistence for scientific evolution records."""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TypeVar

from pydantic import BaseModel, ValidationError

from photomatagent.redaction import redact_secrets
from photomatagent.scientific.evolution.models import (
    EpisodeRecord,
    EvolutionTask,
    ExpertFeedbackRecord,
    RevisionPlan,
    StrategyVersion,
    utc_now,
    validate_managed_id,
)
from photomatagent.scientific.state import ScientificState
from photomatagent.workspace import Workspace

_STORE_PATH = ".photomatagent/evolutions"
_EPISODE_VERSION = re.compile(r"^v[0-9]{3}$")
_ModelT = TypeVar("_ModelT", bound=BaseModel)
_TASK_MUTABLE_FIELDS = (
    "status",
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


class EvolutionStore:
    """Persist evolution tasks and immutable records inside one workspace."""

    lock_timeout_seconds = 5.0
    lock_poll_seconds = 0.05

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.root = workspace.resolve(_STORE_PATH, must_exist=False)
        self.root.mkdir(parents=True, exist_ok=True)
        self.root = workspace.resolve(_STORE_PATH, must_exist=True)

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
        return self._load_model(path, EvolutionTask, require_schema_version=True)

    def save_task(
        self, task: EvolutionTask, expected_revision: int
    ) -> EvolutionTask:
        """Save a task only if its stored revision matches the caller's view."""
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 0
        ):
            raise ValueError("expected_revision must be a non-negative integer")
        candidate, _ = self._prepare_model(task, EvolutionTask)
        task_dir = self._task_dir(candidate.evolution_id)
        with self._task_lock(task_dir):
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
        descriptor: int | None = None
        owner_token = uuid.uuid4().hex
        while descriptor is None:
            try:
                descriptor = os.open(
                    lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise EvolutionLockError(
                        f"timed out acquiring evolution task lock: {task_dir.name}"
                    ) from exc
                time.sleep(min(self.lock_poll_seconds, remaining))
        owner_stat = os.fstat(descriptor)
        initialized = False
        try:
            os.write(descriptor, owner_token.encode("ascii"))
            os.fsync(descriptor)
            initialized = True
            yield
        finally:
            try:
                os.close(descriptor)
            finally:
                self._release_owned_lock(
                    lock_path,
                    owner_stat,
                    owner_token if initialized else None,
                )

    @staticmethod
    def _release_owned_lock(
        lock_path: Path,
        owner_stat: os.stat_result,
        owner_token: str | None,
    ) -> None:
        try:
            current_stat = lock_path.stat(follow_symlinks=False)
        except FileNotFoundError:
            return
        if (
            current_stat.st_dev != owner_stat.st_dev
            or current_stat.st_ino != owner_stat.st_ino
        ):
            return
        if owner_token is not None:
            try:
                current_token = lock_path.read_text(encoding="ascii")
            except (FileNotFoundError, OSError, UnicodeError):
                return
            if current_token != owner_token:
                return
        try:
            verified_stat = lock_path.stat(follow_symlinks=False)
            if (
                verified_stat.st_dev == owner_stat.st_dev
                and verified_stat.st_ino == owner_stat.st_ino
            ):
                lock_path.unlink()
        except FileNotFoundError:
            pass

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
