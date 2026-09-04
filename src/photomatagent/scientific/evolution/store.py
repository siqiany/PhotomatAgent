"""Workspace-contained, atomic persistence for scientific evolution records."""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel

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
_EPISODE_VERSION = re.compile(r"^v\d{3}$")


class EvolutionStoreError(RuntimeError):
    """Base error for evolution persistence failures."""


class EvolutionAlreadyExistsError(EvolutionStoreError):
    """Raised when an immutable evolution record already exists."""


class EvolutionConflictError(EvolutionStoreError):
    """Raised when a task save uses a stale expected revision."""


class EvolutionLockError(EvolutionStoreError):
    """Raised when a task lock cannot be acquired before its deadline."""


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
        if task.revision != 0:
            raise ValueError("new evolution tasks must start at revision 0")
        task_dir = self._task_dir(task.evolution_id, create=True)
        with self._task_lock(task_dir):
            path = self._task_path(task.evolution_id)
            self._write_immutable_json(path, task.model_dump(mode="json"))
        return task

    def load_task(self, evolution_id: str) -> EvolutionTask:
        """Load and validate the authoritative task record."""
        path = self._task_path(evolution_id)
        return EvolutionTask.model_validate_json(path.read_text(encoding="utf-8"))

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
        task_dir = self._task_dir(task.evolution_id)
        with self._task_lock(task_dir):
            current = self.load_task(task.evolution_id)
            if current.revision != expected_revision:
                raise EvolutionConflictError(
                    f"stale evolution task write for {task.evolution_id}: "
                    f"stored revision={current.revision}, "
                    f"expected={expected_revision}"
                )
            updated = task.model_copy(
                update={
                    "revision": current.revision + 1,
                    "updated_at": utc_now(),
                }
            )
            self._write_json_atomic(
                self._task_path(task.evolution_id),
                updated.model_dump(mode="json"),
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
        )

    def write_feedback(self, feedback: ExpertFeedbackRecord) -> Path:
        """Persist one immutable expert-feedback record."""
        self._validate_id(feedback.feedback_id)
        return self._write_record(
            evolution_id=feedback.evolution_id,
            directory="feedback",
            filename=f"{feedback.feedback_id}.json",
            record=feedback,
        )

    def write_revision(self, revision: RevisionPlan) -> Path:
        """Persist one immutable revision plan."""
        self._validate_id(revision.revision_id)
        return self._write_record(
            evolution_id=revision.evolution_id,
            directory="revisions",
            filename=f"{revision.revision_id}.json",
            record=revision,
        )

    def write_strategy(self, strategy: StrategyVersion) -> Path:
        """Persist one immutable strategy snapshot."""
        self._validate_id(strategy.strategy_id)
        return self._write_record(
            evolution_id=strategy.evolution_id,
            directory="strategies",
            filename=f"{strategy.strategy_id}.json",
            record=strategy,
        )

    def write_scientific_state(
        self,
        evolution_id: str,
        version: str,
        state: ScientificState,
    ) -> Path:
        """Persist the immutable scientific-state snapshot for one episode."""
        self._validate_episode_version(version)
        task_dir = self._task_dir(evolution_id)
        with self._task_lock(task_dir):
            self._require_task(evolution_id)
            path = self._managed_path(
                evolution_id,
                "episodes",
                f"{version}.scientific.json",
            )
            self._write_immutable_json(path, state.model_dump(mode="json"))
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
        return ScientificState.model_validate_json(path.read_text(encoding="utf-8"))

    def list_tasks(self) -> list[EvolutionTask]:
        """Return all managed tasks ordered by evolution ID."""
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
        record: BaseModel,
    ) -> Path:
        task_dir = self._task_dir(evolution_id)
        with self._task_lock(task_dir):
            self._require_task(evolution_id)
            path = self._managed_path(evolution_id, directory, filename)
            self._write_immutable_json(path, record.model_dump(mode="json"))
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
        try:
            yield
        finally:
            try:
                os.close(descriptor)
            finally:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass

    def _write_immutable_json(self, path: Path, payload: Any) -> None:
        if path.exists() or path.is_symlink():
            raise EvolutionAlreadyExistsError(
                f"immutable evolution record already exists: {path.name}"
            )
        self._write_json_atomic(path, payload)

    def _write_json_atomic(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path = self.workspace.resolve(self.workspace.relative(path), must_exist=False)
        redacted = redact_secrets(payload)
        serialized = json.dumps(
            redacted,
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
            os.replace(temporary_path, path)
            self._fsync_directory(path.parent)
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            raise

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
    "EvolutionLockError",
    "EvolutionStore",
    "EvolutionStoreError",
]
