"""Workspace-contained, atomic persistence for scientific evolution records."""

from __future__ import annotations

import errno
import importlib
import json
import os
import re
import stat
import tempfile
import time
import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, TypeVar

import numpy as np
from pydantic import BaseModel, ValidationError

from photomatagent.redaction import redact_secrets
from photomatagent.runtime.events import (
    EvolutionComparisonCompleted,
    EvolutionEpisodeCompleted,
    EvolutionEpisodeFailed,
    EvolutionEpisodeStarted,
    EvolutionIterationStarted,
    EvolutionTaskAccepted,
    EvolutionTaskCreated,
    EvolutionTaskReopened,
    EvolutionTaskStopped,
    ExperienceStateChanged,
    ExpertFeedbackCompiled,
    ExpertFeedbackRecorded,
    RevisionPlanConfirmed,
    RuntimeEvent,
    parse_event,
)
from photomatagent.scientific.evolution.models import (
    ComparisonReport,
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
from photomatagent.scientific.evolution.experience import (
    ExperienceRecord,
    StrategyObservation,
    canonical_record_sha256,
    task_context_from_episode,
)
from photomatagent.scientific.evolution.events import bounded_summary
from photomatagent.scientific.evolution.strategy import (
    BayesianLinearStrategySelector,
    StrategyPosteriorSnapshot,
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
    "accepted_resume_status",
    "episode_ids",
    "evaluation_episode_ids",
    "current_evaluation_version",
    "feedback_ids",
    "compilation_ids",
    "revision_ids",
    "strategy_ids",
    "comparison_ids",
    "experience_ids",
    "event_outbox",
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
    "owner_token",
    "execution_mode",
    "strategy_id",
    "strategy_arm",
    "strategy_sha256",
    "strategy_cutoff_at",
    "evaluation_workspace_path",
    "evaluation_workspace_device",
    "evaluation_workspace_inode",
    "evaluation_workspace_fingerprint",
    "previous_owner_sha256",
    "owner_reclaimed_at",
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


@dataclass(slots=True)
class EvaluationLease:
    evolution_id: str
    descriptor: int
    workspace_root: Path
    active: bool = True


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

    def load_evaluation_episode(self, version: str) -> EpisodeRecord:
        self._require_active()
        return self.store.load_evaluation_episode(self.evolution_id, version)

    def write_evaluation_episode(self, episode: EpisodeRecord) -> Path:
        self._require_active()
        return self.store._write_record_locked(
            evolution_id=self.evolution_id,
            directory="evaluations",
            filename=f"{episode.version}.json",
            record=episode,
            model_type=EpisodeRecord,
        )

    def transition_evaluation_episode(
        self,
        episode: EpisodeRecord,
        *,
        expected_status: EpisodeStatus,
    ) -> EpisodeRecord:
        self._require_active()
        return self.store._transition_evaluation_episode_locked(
            episode,
            expected_status,
        )

    def reclaim_evaluation_owner(
        self,
        episode: EpisodeRecord,
        *,
        previous_owner_token: str,
    ) -> EpisodeRecord:
        self._require_active()
        current = self.load_evaluation_episode(episode.version)
        if current.status != "RESERVED" or current.owner_token != previous_owner_token:
            raise EvolutionConflictError("evaluation owner reclaim lost its race")
        if (
            episode.owner_token == previous_owner_token
            or episode.previous_owner_sha256
            != hashlib.sha256(previous_owner_token.encode("utf-8")).hexdigest()
            or episode.owner_reclaimed_at is None
        ):
            raise EvolutionConflictError("evaluation owner reclaim audit is invalid")
        for field in _EPISODE_IMMUTABLE_FIELDS:
            if field in {"owner_token", "previous_owner_sha256", "owner_reclaimed_at"}:
                continue
            if getattr(episode, field) != getattr(current, field):
                raise EvolutionConflictError(f"evaluation reclaim changed {field}")
        candidate, payload = self.store._prepare_model(episode, EpisodeRecord)
        self.store._write_json_atomic(
            self.store._managed_path(
                candidate.evolution_id, "evaluations", f"{candidate.version}.json"
            ),
            payload,
        )
        return candidate

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

    def load_strategy(self, strategy_id: str) -> StrategyVersion:
        self._require_active()
        return self.store.load_strategy(self.evolution_id, strategy_id)

    def write_strategy(self, strategy: StrategyVersion) -> Path:
        self._require_active()
        if strategy.evolution_id != self.evolution_id:
            raise EvolutionConflictError("strategy belongs to a different transaction")
        return self.store._write_record_locked(
            evolution_id=self.evolution_id,
            directory="strategies",
            filename=f"{strategy.strategy_id}.json",
            record=strategy,
            model_type=StrategyVersion,
        )

    def load_comparison(self, comparison_id: str) -> ComparisonReport:
        self._require_active()
        return self.store.load_comparison(self.evolution_id, comparison_id)

    def write_comparison(self, comparison: ComparisonReport) -> Path:
        self._require_active()
        if comparison.evolution_id != self.evolution_id:
            raise EvolutionConflictError(
                "comparison belongs to a different transaction"
            )
        return self.store._write_record_locked(
            evolution_id=self.evolution_id,
            directory="comparisons",
            filename=f"{comparison.comparison_id}.json",
            record=comparison,
            model_type=ComparisonReport,
        )

    def load_experience(self, experience_id: str) -> ExperienceRecord:
        self._require_active()
        return self.store.load_experience(self.evolution_id, experience_id)

    def write_experience(self, experience: ExperienceRecord) -> Path:
        self._require_active()
        if experience.evolution_id != self.evolution_id:
            raise EvolutionConflictError(
                "experience belongs to a different transaction"
            )
        self.store._validate_experience_lineage(experience)
        return self.store._write_record_locked(
            evolution_id=self.evolution_id,
            directory="experience",
            filename=f"{experience.experience_id}.json",
            record=experience,
            model_type=ExperienceRecord,
        )

    def load_strategy_observation(
        self, observation_id: str
    ) -> StrategyObservation:
        self._require_active()
        return self.store.load_strategy_observation(
            self.evolution_id, observation_id
        )

    def write_strategy_observation(
        self, observation: StrategyObservation
    ) -> Path:
        self._require_active()
        if observation.evolution_id != self.evolution_id:
            raise EvolutionConflictError(
                "strategy observation belongs to a different transaction"
            )
        return self.store._write_strategy_observation_locked(observation)

    def load_strategy_posterior(
        self, posterior_id: str
    ) -> StrategyPosteriorSnapshot:
        self._require_active()
        return self.store.load_strategy_posterior(self.evolution_id, posterior_id)

    def write_strategy_posterior(
        self, posterior: StrategyPosteriorSnapshot
    ) -> Path:
        self._require_active()
        self.store._validate_strategy_posterior_provenance(posterior)
        return self.store._write_record_locked(
            evolution_id=self.evolution_id,
            directory="strategy_posteriors",
            filename=f"{posterior.posterior_id}.json",
            record=posterior,
            model_type=StrategyPosteriorSnapshot,
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
        self._active_evaluation_leases: dict[int, EvaluationLease] = {}

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

    @contextmanager
    def evaluation_lease(self, evolution_id: str) -> Iterator[EvaluationLease]:
        """Hold the cross-process fresh-evaluation execution lease."""

        task_dir = self._task_dir(evolution_id)
        path = self.workspace.resolve(
            self.workspace.relative(task_dir / ".evaluation.lease"),
            must_exist=False,
        )
        lease_flags = os.O_CREAT | os.O_RDWR | int(getattr(os, "O_NOFOLLOW", 0))
        descriptor = os.open(path, lease_flags, 0o600)
        lease = EvaluationLease(
            evolution_id=evolution_id,
            descriptor=descriptor,
            workspace_root=self.workspace.root,
        )
        try:
            if os.name == "posix":
                try:
                    _ADVISORY_LOCK_API.flock(
                        descriptor,
                        _ADVISORY_LOCK_API.LOCK_EX | _ADVISORY_LOCK_API.LOCK_NB,
                    )
                except OSError as exc:
                    raise EvolutionLockError(
                        "fresh evaluation is leased by another process"
                    ) from exc
            elif os.name == "nt":  # pragma: no cover - Windows CI only
                try:
                    _ADVISORY_LOCK_API.locking(
                        descriptor, _ADVISORY_LOCK_API.LK_NBLCK, 1
                    )
                except OSError as exc:
                    raise EvolutionLockError(
                        "fresh evaluation is leased by another process"
                    ) from exc
            self._active_evaluation_leases[id(lease)] = lease
            yield lease
        finally:
            self._active_evaluation_leases.pop(id(lease), None)
            lease.active = False
            try:
                if os.name == "posix":
                    _ADVISORY_LOCK_API.flock(descriptor, _ADVISORY_LOCK_API.LOCK_UN)
                elif os.name == "nt":  # pragma: no cover
                    _ADVISORY_LOCK_API.locking(descriptor, _ADVISORY_LOCK_API.LK_UNLCK, 1)
            finally:
                os.close(descriptor)

    def has_active_evaluation_lease(
        self,
        lease: EvaluationLease,
        evolution_id: str,
    ) -> bool:
        """Verify that this store issued and still holds the exact lease object."""

        if (
            not lease.active
            or lease.evolution_id != evolution_id
            or lease.workspace_root != self.workspace.root
            or self._active_evaluation_leases.get(id(lease)) is not lease
        ):
            return False
        try:
            descriptor_stat = os.fstat(lease.descriptor)
            lease_path = self._task_dir(evolution_id) / ".evaluation.lease"
            path_stat = os.lstat(lease_path)
        except OSError:
            return False
        return (
            not stat.S_ISLNK(path_stat.st_mode)
            and descriptor_stat.st_dev == path_stat.st_dev
            and descriptor_stat.st_ino == path_stat.st_ino
        )

    def create_task(self, task: EvolutionTask) -> EvolutionTask:
        """Create a new revision-zero task without replacing an existing task."""
        created_event = EvolutionTaskCreated(
            evolution_id=task.evolution_id,
            goal_summary=task.goal[:240],
        )
        task.event_outbox = [
            {"sequence": 1, **created_event.model_dump(mode="json")}
        ]
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

    def append_events(
        self,
        evolution_id: str,
        events: Iterator[RuntimeEvent] | tuple[RuntimeEvent, ...] | list[RuntimeEvent],
        *,
        idempotency_scope: str | None = None,
    ) -> None:
        """Append lifecycle events to the evolution journal, once per exact event."""

        pending = list(events)
        if not pending:
            return
        task_dir = self._task_dir(evolution_id)
        with self._task_lock(task_dir):
            durable = self._sequenced_outbox_entries(evolution_id)
            self._append_sequenced_events_locked(evolution_id, durable)
            journal = self.read_event_journal(evolution_id)
            existing = {
                self._event_payload_identity(
                    envelope["event"],
                    ignore_transient=idempotency_scope is not None,
                )
                for _source, envelope in journal
            }
            next_sequence = max(
                [
                    *(entry[0] for entry in durable),
                    *(int(envelope["sequence"]) for _source, envelope in journal),
                    0,
                ]
            ) + 1
            sequenced: list[tuple[int, dict[str, Any]]] = []
            for event in pending:
                payload = event.model_dump(mode="json")
                identity = self._event_payload_identity(
                    payload,
                    ignore_transient=idempotency_scope is not None,
                )
                if identity in existing:
                    continue
                sequenced.append((next_sequence, payload))
                existing.add(identity)
                next_sequence += 1
            self._append_sequenced_events_locked(evolution_id, sequenced)

    def read_event_journal(self, evolution_id: str) -> list[tuple[bytes, dict[str, Any]]]:
        """Return exact source lines and decoded journal envelopes."""

        path = self._managed_path(evolution_id, "events.jsonl")
        if not path.exists():
            return []
        if path.is_symlink() or not path.is_file():
            raise EvolutionCorruptRecordError(path, "event journal is not a regular file")
        records: list[tuple[bytes, dict[str, Any]]] = []
        for line_number, raw in enumerate(
            path.read_bytes().splitlines(keepends=True), start=1
        ):
            try:
                decoded = json.loads(raw)
                if not isinstance(decoded, dict) or not isinstance(decoded.get("event"), dict):
                    raise TypeError("journal envelope must contain an event object")
                decoded.setdefault("sequence", line_number)
                if not isinstance(decoded["sequence"], int) or decoded["sequence"] < 1:
                    raise TypeError("journal envelope sequence must be a positive integer")
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise EvolutionCorruptRecordError(
                    path, f"invalid event journal line {line_number}: {exc}"
                ) from exc
            records.append((raw, decoded))
        return records

    def flush_event_outbox(self, evolution_id: str) -> None:
        """Idempotently materialize every durable event intent into the journal."""

        task_dir = self._task_dir(evolution_id)
        with self._task_lock(task_dir):
            self._append_sequenced_events_locked(
                evolution_id,
                self._sequenced_outbox_entries(evolution_id),
            )

    def _all_outbox_entries(
        self,
        evolution_id: str,
    ) -> list[tuple[str, int, dict[str, Any]]]:
        task = self.load_task(evolution_id)
        located = [
            ("task", index, entry)
            for index, entry in enumerate(task.event_outbox)
        ]
        for index in range(1, len(task.episode_ids) + 1):
            episode = self.load_episode(evolution_id, f"v{index:03d}")
            located.extend(
                (f"main:{episode.version}", item_index, entry)
                for item_index, entry in enumerate(episode.event_outbox)
            )
        for index in range(1, len(task.evaluation_episode_ids) + 1):
            episode = self.load_evaluation_episode(evolution_id, f"v{index:03d}")
            located.extend(
                (f"evaluation:{episode.version}", item_index, entry)
                for item_index, entry in enumerate(episode.event_outbox)
            )
        return located

    @staticmethod
    def _outbox_payload(entry: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in entry.items() if key != "sequence"}

    def _sequenced_outbox_entries(
        self,
        evolution_id: str,
    ) -> list[tuple[int, dict[str, Any]]]:
        located = self._all_outbox_entries(evolution_id)
        sequenced: list[tuple[int, dict[str, Any]]] = []
        legacy: list[tuple[str, int, dict[str, Any]]] = []
        used: set[int] = set()
        for source, index, entry in located:
            sequence = entry.get("sequence")
            payload = self._outbox_payload(entry)
            parse_event(payload)
            if isinstance(sequence, int) and not isinstance(sequence, bool) and sequence > 0:
                if sequence in used:
                    raise EvolutionConflictError(
                        f"duplicate evolution event sequence {sequence}"
                    )
                used.add(sequence)
                sequenced.append((sequence, payload))
            else:
                legacy.append((source, index, payload))
        legacy.sort(
            key=lambda item: (
                str(item[2].get("timestamp", "")),
                item[0],
                item[1],
            )
        )
        next_sequence = 1
        for _source, _index, payload in legacy:
            while next_sequence in used:
                next_sequence += 1
            used.add(next_sequence)
            sequenced.append((next_sequence, payload))
            next_sequence += 1
        return sorted(sequenced, key=lambda item: item[0])

    def _next_outbox_sequence(self, evolution_id: str) -> int:
        outbox = self._sequenced_outbox_entries(evolution_id)
        journal = self.read_event_journal(evolution_id)
        return max(
            [
                *(sequence for sequence, _payload in outbox),
                *(int(envelope["sequence"]) for _source, envelope in journal),
                0,
            ]
        ) + 1

    @staticmethod
    def _event_payload_identity(
        payload: dict[str, Any],
        *,
        ignore_transient: bool,
    ) -> str:
        stable = dict(payload)
        if ignore_transient:
            stable.pop("timestamp", None)
            stable.pop("session_id", None)
            stable.pop("run_id", None)
        return hashlib.sha256(
            json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _append_sequenced_events_locked(
        self,
        evolution_id: str,
        entries: list[tuple[int, dict[str, Any]]],
    ) -> None:
        if not entries:
            return
        path = self._managed_path(evolution_id, "events.jsonl")
        existing_ids: set[str] = set()
        legacy_identities: set[str] = set()
        if path.exists():
            if path.is_symlink() or not path.is_file():
                raise EvolutionCorruptRecordError(
                    path, "event journal is not a regular file"
                )
            for line_number, line in enumerate(path.read_bytes().splitlines(), start=1):
                try:
                    envelope = json.loads(line)
                    payload = envelope["event"]
                    existing_ids.add(str(envelope["event_id"]))
                    if "sequence" not in envelope:
                        legacy_identities.add(
                            self._event_payload_identity(
                                payload,
                                ignore_transient=True,
                            )
                        )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise EvolutionCorruptRecordError(
                        path, f"invalid event journal line {line_number}: {exc}"
                    ) from exc
        additions: list[bytes] = []
        for sequence, payload in sorted(entries, key=lambda item: item[0]):
            stable = dict(payload)
            stable.pop("timestamp", None)
            stable.pop("session_id", None)
            stable.pop("run_id", None)
            identity = json.dumps(
                {"sequence": sequence, "event": stable},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            event_id = hashlib.sha256(identity).hexdigest()
            if event_id in existing_ids or self._event_payload_identity(
                payload,
                ignore_transient=True,
            ) in legacy_identities:
                continue
            envelope = {
                "sequence": sequence,
                "event_id": event_id,
                "event": payload,
            }
            additions.append(
                json.dumps(
                    envelope,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
            existing_ids.add(event_id)
        if not additions:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        flags |= int(getattr(os, "O_NOFOLLOW", 0))
        descriptor = os.open(path, flags, 0o600)
        try:
            with os.fdopen(descriptor, "ab", closefd=True) as stream:
                for addition in additions:
                    stream.write(addition)
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise

    @staticmethod
    def event_scope(record: object) -> str:
        """Stable mutation identity used to suppress crash/retry event duplicates."""

        if isinstance(record, EvolutionTask):
            return f"task:{record.revision}:{record.status}"
        if isinstance(record, EpisodeRecord):
            return f"episode:{record.execution_mode}:{record.episode_id}"
        for field in (
            "feedback_id",
            "compilation_id",
            "revision_id",
            "comparison_id",
            "experience_id",
            "strategy_id",
        ):
            value = getattr(record, field, None)
            if isinstance(value, str):
                return f"{field}:{value}"
        raise ValueError("event-bearing record has no stable journal identity")

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
        transition_events = self._task_transition_events(current, candidate)
        updated_data["event_outbox"] = self._merge_outbox(
            candidate.evolution_id,
            current.event_outbox,
            transition_events,
        )
        updated_data["revision"] = current.revision + 1
        updated_data["updated_at"] = utc_now()
        updated = EvolutionTask.model_validate(updated_data)
        updated, payload = self._prepare_model(updated, EvolutionTask)
        self._write_json_atomic(
            self._task_path(candidate.evolution_id),
            payload,
        )
        return updated

    def _merge_outbox(
        self,
        evolution_id: str,
        existing: list[dict[str, Any]],
        events: list[RuntimeEvent],
    ) -> list[dict[str, Any]]:
        merged = list(existing)
        identities = {
            self._outbox_identity(item)
            for item in merged
        }
        next_sequence = self._next_outbox_sequence(evolution_id)
        for event in events:
            payload = event.model_dump(mode="json")
            identity = self._outbox_identity(payload)
            if identity not in identities:
                merged.append({"sequence": next_sequence, **payload})
                identities.add(identity)
                next_sequence += 1
        return merged

    @staticmethod
    def _outbox_identity(payload: dict[str, Any]) -> str:
        stable = EvolutionStore._outbox_payload(payload)
        stable.pop("timestamp", None)
        stable.pop("session_id", None)
        stable.pop("run_id", None)
        return hashlib.sha256(
            json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _episode_event_kwargs(self, episode: EpisodeRecord) -> dict[str, Any]:
        return {
            "evolution_id": episode.evolution_id,
            "episode_version": episode.version,
            "episode_id": episode.episode_id,
            "execution_mode": episode.execution_mode,
        }

    def _task_transition_events(
        self, current: EvolutionTask, candidate: EvolutionTask
    ) -> list[RuntimeEvent]:
        events: list[RuntimeEvent] = []
        evolution_id = current.evolution_id
        if candidate.current_version != current.current_version and candidate.current_version:
            episode = self.load_episode(evolution_id, candidate.current_version)
            kwargs = self._episode_event_kwargs(episode)
            if current.current_version is not None:
                events.append(EvolutionIterationStarted(**kwargs))
        if (
            candidate.current_evaluation_version != current.current_evaluation_version
            and candidate.current_evaluation_version
        ):
            episode = self.load_evaluation_episode(
                evolution_id, candidate.current_evaluation_version
            )
            kwargs = self._episode_event_kwargs(episode)
            events.append(EvolutionIterationStarted(**kwargs))
        for feedback_id in candidate.feedback_ids[len(current.feedback_ids):]:
            feedback = self.load_feedback(evolution_id, feedback_id)
            episode = self.load_episode(evolution_id, feedback.episode_version)
            events.append(
                ExpertFeedbackRecorded(
                    **self._episode_event_kwargs(episode),
                    feedback_id=feedback.feedback_id,
                    result_sha256=feedback.result_sha256,
                    scores={
                        "scientific_correctness": feedback.scores.scientific_correctness,
                        "evidence_sufficiency": feedback.scores.evidence_sufficiency,
                        "novelty": feedback.scores.novelty,
                        "actionability": feedback.scores.actionability,
                        "overall": feedback.scores.overall,
                    },
                )
            )
        for compilation_id in candidate.compilation_ids[len(current.compilation_ids):]:
            compilation = self.load_compilation(evolution_id, compilation_id)
            if compilation.episode_version is None:
                raise EvolutionConflictError("linked compilation has no episode version")
            episode = self.load_episode(evolution_id, compilation.episode_version)
            events.append(
                ExpertFeedbackCompiled(
                    **self._episode_event_kwargs(episode),
                    compilation_id=compilation.compilation_id,
                )
            )
        for revision_id in candidate.revision_ids[len(current.revision_ids):]:
            revision = self.load_revision(evolution_id, revision_id)
            episode = self.load_episode(evolution_id, revision.source_version)
            events.append(
                RevisionPlanConfirmed(
                    **self._episode_event_kwargs(episode),
                    revision_id=revision.revision_id,
                )
            )
        for comparison_id in candidate.comparison_ids[len(current.comparison_ids):]:
            comparison = self.load_comparison(evolution_id, comparison_id)
            episode = self.load_episode(evolution_id, comparison.current_version)
            events.append(
                EvolutionComparisonCompleted(
                    **self._episode_event_kwargs(episode),
                    comparison_id=comparison.comparison_id,
                )
            )
        for experience_id in candidate.experience_ids[len(current.experience_ids):]:
            events.append(
                ExperienceStateChanged(
                    evolution_id=evolution_id,
                    experience_id=experience_id,
                )
            )
        if candidate.status == "ACCEPTED" and current.status != "ACCEPTED":
            if candidate.accepted_version is None:
                raise EvolutionConflictError("accepted task has no accepted version")
            accepted_episode = self.load_episode(
                evolution_id, candidate.accepted_version
            )
            events.append(
                EvolutionTaskAccepted(
                    **self._episode_event_kwargs(accepted_episode),
                    task_revision=current.revision + 1,
                )
            )
        if candidate.status == "STOPPED" and current.status != "STOPPED":
            events.append(
                EvolutionTaskStopped(
                    evolution_id=evolution_id,
                    task_revision=current.revision + 1,
                )
            )
        if current.status in {"ACCEPTED", "STOPPED", "BLOCKED", "BUDGET_EXHAUSTED"} and candidate.status not in {"ACCEPTED", "STOPPED", "BLOCKED", "BUDGET_EXHAUSTED"}:
            events.append(
                EvolutionTaskReopened(
                    evolution_id=evolution_id,
                    task_revision=current.revision + 1,
                )
            )
        return events

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

    def load_evaluation_episode(
        self,
        evolution_id: str,
        version: str,
    ) -> EpisodeRecord:
        self._validate_episode_version(version)
        path = self._managed_path(evolution_id, "evaluations", f"{version}.json")
        episode = self._load_model(path, EpisodeRecord, require_schema_version=True)
        if (
            episode.evolution_id != evolution_id
            or episode.version != version
            or episode.execution_mode != "FRESH_EVALUATION"
        ):
            raise EvolutionCorruptRecordError(
                path,
                "evaluation episode identity does not match its managed path",
            )
        return episode

    def transition_evaluation_episode(
        self,
        episode: EpisodeRecord,
        *,
        expected_status: EpisodeStatus,
    ) -> EpisodeRecord:
        task_dir = self._task_dir(episode.evolution_id)
        with self._task_lock(task_dir):
            return self._transition_evaluation_episode_locked(episode, expected_status)

    def _transition_evaluation_episode_locked(
        self,
        episode: EpisodeRecord,
        expected_status: EpisodeStatus,
    ) -> EpisodeRecord:
        if expected_status not in _EPISODE_TRANSITIONS:
            raise ValueError(f"unsupported expected episode status: {expected_status!r}")
        current = self.load_evaluation_episode(episode.evolution_id, episode.version)
        owner_recovery = (
            current.status == "RUNNING"
            and episode.status == "FAILED"
            and current.owner_token is not None
            and episode.owner_token == current.owner_token
            and episode.previous_owner_sha256
            == hashlib.sha256(current.owner_token.encode("utf-8")).hexdigest()
            and episode.owner_reclaimed_at is not None
        )
        for field in _EPISODE_IMMUTABLE_FIELDS:
            if owner_recovery and field in {
                "previous_owner_sha256",
                "owner_reclaimed_at",
            }:
                continue
            if getattr(episode, field) != getattr(current, field):
                raise EvolutionConflictError(
                    f"immutable evaluation field differs: {field}"
                )
        episode = episode.model_copy(
            update={
                "event_outbox": self._merge_outbox(
                    current.evolution_id,
                    current.event_outbox,
                    self._episode_transition_events(current, episode),
                )
            }
        )
        candidate, payload = self._prepare_model(episode, EpisodeRecord)
        if current.status != expected_status:
            raise EvolutionConflictError(
                f"stale evaluation transition: stored={current.status}, expected={expected_status}"
            )
        if candidate.status not in _EPISODE_TRANSITIONS[current.status]:
            raise EvolutionConflictError(
                f"illegal evaluation transition: {current.status} -> {candidate.status}"
            )
        if current.status == "RESERVED" and candidate.status == "RUNNING":
            if candidate.started_at is None:
                raise EvolutionConflictError(
                    "RUNNING evaluation transition requires started_at provenance"
                )
            for field in ("completed_at", "summary", "artifact", "error"):
                if getattr(candidate, field) != getattr(current, field):
                    raise EvolutionConflictError(
                        f"RUNNING evaluation transition cannot set terminal field {field}"
                    )
        if current.status == "RESERVED" and candidate.status == "FAILED":
            for field in _EPISODE_RUNNING_PROVENANCE_FIELDS:
                if getattr(candidate, field) != getattr(current, field):
                    raise EvolutionConflictError(
                        "unstarted FAILED evaluation cannot add runtime provenance "
                        f"field {field}"
                    )
        if current.status == "RUNNING":
            for field in _EPISODE_RUNNING_PROVENANCE_FIELDS:
                if getattr(candidate, field) != getattr(current, field):
                    raise EvolutionConflictError(
                        f"running evaluation provenance differs: {field}"
                    )
        self._write_json_atomic(
            self._managed_path(
                candidate.evolution_id,
                "evaluations",
                f"{candidate.version}.json",
            ),
            payload,
        )
        return candidate

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
        if expected_status not in _EPISODE_TRANSITIONS:
            raise ValueError(f"unsupported expected episode status: {expected_status!r}")
        current = self.load_episode(episode.evolution_id, episode.version)
        for field in _EPISODE_IMMUTABLE_FIELDS:
            if getattr(episode, field) != getattr(current, field):
                raise EvolutionConflictError(
                    "immutable episode execution snapshot differs from stored "
                    f"record for {episode.evolution_id}/{episode.version}: {field}"
                )
        episode = episode.model_copy(
            update={
                "event_outbox": self._merge_outbox(
                    current.evolution_id,
                    current.event_outbox,
                    self._episode_transition_events(current, episode),
                )
            }
        )
        candidate, payload = self._prepare_model(episode, EpisodeRecord)
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

    def _episode_transition_events(
        self,
        current: EpisodeRecord,
        candidate: EpisodeRecord,
    ) -> list[RuntimeEvent]:
        kwargs = self._episode_event_kwargs(candidate)
        if current.status == candidate.status:
            return []
        if candidate.status == "RUNNING":
            return [EvolutionEpisodeStarted(**kwargs)]
        if candidate.status == "COMPLETED":
            return [EvolutionEpisodeCompleted(**kwargs)]
        if candidate.status == "FAILED":
            return [
                EvolutionEpisodeFailed(
                    **kwargs,
                    error_summary=bounded_summary(candidate.error or ""),
                )
            ]
        return []

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

    def load_strategy(self, evolution_id: str, strategy_id: str) -> StrategyVersion:
        """Load one strategy snapshot bound to its managed identity."""

        self._validate_id(strategy_id)
        path = self._managed_path(evolution_id, "strategies", f"{strategy_id}.json")
        strategy = self._load_model(
            path,
            StrategyVersion,
            require_schema_version=True,
        )
        if strategy.evolution_id != evolution_id or strategy.strategy_id != strategy_id:
            raise EvolutionCorruptRecordError(
                path,
                "strategy identity does not match its managed path",
            )
        return strategy

    def list_strategies(self, evolution_id: str) -> list[StrategyVersion]:
        """Load every immutable strategy snapshot for stable-operation checks."""

        directory = self._managed_path(evolution_id, "strategies")
        if not directory.is_dir():
            return []
        records: list[StrategyVersion] = []
        for path in sorted(directory.glob("*.json")):
            try:
                self._validate_id(path.stem)
            except (TypeError, ValueError):
                continue
            records.append(self.load_strategy(evolution_id, path.stem))
        return records

    def write_comparison(self, comparison: ComparisonReport) -> Path:
        """Persist one immutable adjacent-episode comparison snapshot."""

        self._validate_id(comparison.comparison_id)
        return self._write_record(
            evolution_id=comparison.evolution_id,
            directory="comparisons",
            filename=f"{comparison.comparison_id}.json",
            record=comparison,
            model_type=ComparisonReport,
        )

    def load_comparison(
        self,
        evolution_id: str,
        comparison_id: str,
    ) -> ComparisonReport:
        """Load one comparison bound to its managed task and filename."""

        self._validate_id(comparison_id)
        path = self._managed_path(
            evolution_id,
            "comparisons",
            f"{comparison_id}.json",
        )
        comparison = self._load_model(
            path,
            ComparisonReport,
            require_schema_version=True,
        )
        if (
            comparison.evolution_id != evolution_id
            or comparison.comparison_id != comparison_id
        ):
            raise EvolutionCorruptRecordError(
                path,
                "comparison identity does not match its managed path",
            )
        return comparison

    def list_comparisons(self, evolution_id: str) -> list[ComparisonReport]:
        """Load every immutable comparison for one evolution task."""

        return self._list_managed_records(
            evolution_id,
            "comparisons",
            self.load_comparison,
        )

    def write_experience(
        self,
        experience: ExperienceRecord,
    ) -> Path:
        """Persist one immutable experience maturity snapshot."""

        self._validate_id(experience.experience_id)
        self._validate_experience_lineage(experience)
        return self._write_record(
            evolution_id=experience.evolution_id,
            directory="experience",
            filename=f"{experience.experience_id}.json",
            record=experience,
            model_type=ExperienceRecord,
        )

    def _validate_experience_lineage(self, experience: ExperienceRecord) -> None:
        if experience.maturity == "OBSERVATION":
            return
        if experience.base_experience_id is None:
            raise EvolutionConflictError("promoted experience requires a base record")
        try:
            base = self.load_experience(
                experience.evolution_id,
                experience.base_experience_id,
            )
        except FileNotFoundError as exc:
            raise EvolutionConflictError(
                "promoted experience base record is unavailable"
            ) from exc
        if (
            base.evolution_id != experience.evolution_id
            or base.maturity != experience.previous_maturity
            or not set(base.observations).issubset(set(experience.observations))
        ):
            raise EvolutionConflictError(
                "promoted experience does not extend its exact legal base"
            )

    def load_experience(
        self,
        evolution_id: str,
        experience_id: str,
    ) -> ExperienceRecord:
        """Load one experience bound to its immutable managed filename."""

        self._validate_id(experience_id)
        path = self._managed_path(
            evolution_id,
            "experience",
            f"{experience_id}.json",
        )
        experience = self._load_model(
            path,
            ExperienceRecord,
            require_schema_version=True,
        )
        if (
            experience.evolution_id != evolution_id
            or experience.experience_id != experience_id
        ):
            raise EvolutionCorruptRecordError(
                path,
                "experience identity does not match its managed path",
            )
        return experience

    def list_experiences(self, evolution_id: str) -> list[ExperienceRecord]:
        """Load every immutable experience snapshot for one task."""

        return self._list_managed_records(
            evolution_id,
            "experience",
            self.load_experience,
        )

    def write_strategy_observation(
        self,
        observation: StrategyObservation,
    ) -> Path:
        """Persist one reviewed observation after rechecking every source link."""

        self._validate_id(observation.observation_id)
        task_dir = self._task_dir(observation.evolution_id)
        with self._task_lock(task_dir):
            return self._write_strategy_observation_locked(observation)

    def _write_strategy_observation_locked(
        self,
        observation: StrategyObservation,
    ) -> Path:
        candidate, payload = self._prepare_model(observation, StrategyObservation)
        self._validate_strategy_observation_provenance(candidate)
        path = self._managed_path(
            candidate.evolution_id,
            "strategy_observations",
            f"{candidate.observation_id}.json",
        )
        if path.exists():
            stored = self.load_strategy_observation(
                candidate.evolution_id, candidate.observation_id
            )
            if stored == candidate:
                return path
            raise EvolutionAlreadyExistsError(
                f"immutable evolution record already exists: {path}"
            )
        for stored in self.list_strategy_observations(candidate.evolution_id):
            if stored.comparison_id == candidate.comparison_id:
                if stored == candidate:
                    return self._managed_path(
                        stored.evolution_id,
                        "strategy_observations",
                        f"{stored.observation_id}.json",
                    )
                raise EvolutionConflictError(
                    f"comparison {candidate.comparison_id} already has a conflicting "
                    "strategy observation"
                )
        self._write_immutable_json(path, payload)
        return path

    def _validate_strategy_observation_provenance(
        self,
        observation: StrategyObservation,
    ) -> None:
        task = self.load_task(observation.evolution_id)
        if task.task_group_id != observation.task_group_id:
            raise EvolutionConflictError(
                "strategy observation task group does not match task manifest"
            )
        required = (
            (observation.comparison_id, task.comparison_ids, "comparison"),
            (observation.experience_id, task.experience_ids, "experience"),
            (observation.strategy_id, task.strategy_ids, "strategy"),
        )
        for record_id, manifest_ids, label in required:
            if record_id not in manifest_ids:
                raise EvolutionConflictError(
                    f"strategy observation {label} is not referenced by task manifest"
                )
        comparison = self.load_comparison(
            observation.evolution_id, observation.comparison_id
        )
        experience = self.load_experience(
            observation.evolution_id, observation.experience_id
        )
        strategy = self.load_strategy(
            observation.evolution_id, observation.strategy_id
        )
        episode = self.load_episode(
            observation.evolution_id, observation.current_version
        )
        previous_episode = self.load_episode(
            observation.evolution_id, observation.previous_version
        )
        if (
            comparison.phase != "POST_FEEDBACK"
            or comparison.reward is None
            or comparison.expert_utility_delta is None
            or "expert_utility_delta" not in comparison.components_used
        ):
            raise EvolutionConflictError(
                "strategy observations require a reviewed POST_FEEDBACK comparison "
                "whose reward includes expert utility"
            )
        if canonical_record_sha256(comparison) != observation.comparison_sha256:
            raise EvolutionConflictError("strategy observation comparison hash mismatch")
        if canonical_record_sha256(experience) != observation.experience_sha256:
            raise EvolutionConflictError("strategy observation experience hash mismatch")
        if canonical_record_sha256(strategy) != observation.strategy_record_sha256:
            raise EvolutionConflictError(
                "strategy observation strategy record hash mismatch"
            )
        matching_evidence = [
            item
            for item in experience.observations
            if item.comparison_id == comparison.comparison_id
        ]
        if (
            len(matching_evidence) != 1
            or matching_evidence[0].task_group_id != task.task_group_id
            or matching_evidence[0].reward != comparison.reward
        ):
            raise EvolutionConflictError(
                "strategy observation is not linked to matching Task-11 experience evidence"
            )
        expected_values = (
            (
                comparison.previous_version,
                observation.previous_version,
                "previous version",
            ),
            (
                comparison.current_version,
                observation.current_version,
                "current version",
            ),
            (
                comparison.current_feedback_id,
                observation.current_feedback_id,
                "feedback ID",
            ),
            (
                comparison.current_feedback_sha256,
                observation.current_feedback_sha256,
                "feedback hash",
            ),
            (
                comparison.current_compilation_id,
                observation.current_compilation_id,
                "compilation ID",
            ),
            (
                comparison.current_compilation_sha256,
                observation.current_compilation_sha256,
                "compilation hash",
            ),
            (strategy.arm, observation.strategy_arm, "strategy arm"),
            (strategy.strategy_sha256, observation.strategy_sha256, "strategy hash"),
            (strategy.cutoff_at, observation.strategy_cutoff_at, "strategy cutoff"),
            (
                episode.execution_mode,
                observation.source_execution_mode,
                "execution mode",
            ),
            (episode.strategy_id, observation.strategy_id, "episode strategy ID"),
            (episode.strategy_arm, observation.strategy_arm, "episode strategy arm"),
            (
                episode.strategy_sha256,
                observation.strategy_sha256,
                "episode strategy hash",
            ),
            (comparison.reward, observation.reward, "reward"),
            (comparison.created_at, observation.created_at, "created at"),
        )
        for authoritative, recorded, label in expected_values:
            if authoritative != recorded:
                raise EvolutionConflictError(
                    f"strategy observation {label} does not match provenance"
                )
        if episode.execution_mode == "FRESH_EVALUATION":
            raise EvolutionConflictError(
                "fresh evaluations cannot train the strategy selector"
            )
        if episode.parent_version != comparison.previous_version:
            raise EvolutionConflictError(
                "strategy observation current episode does not follow comparison parent"
            )
        expected_context = task_context_from_episode(task.target, previous_episode)
        if observation.context != expected_context:
            raise EvolutionConflictError(
                "strategy observation task context does not match task target and "
                "previous Episode critical gaps"
            )

    def load_strategy_observation(
        self,
        evolution_id: str,
        observation_id: str,
    ) -> StrategyObservation:
        self._validate_id(observation_id)
        path = self._managed_path(
            evolution_id,
            "strategy_observations",
            f"{observation_id}.json",
        )
        observation = self._load_model(
            path, StrategyObservation, require_schema_version=True
        )
        if (
            observation.evolution_id != evolution_id
            or observation.observation_id != observation_id
        ):
            raise EvolutionCorruptRecordError(
                path, "strategy observation identity does not match its managed path"
            )
        self._validate_strategy_observation_provenance(observation)
        return observation

    def list_strategy_observations(
        self,
        evolution_id: str,
    ) -> list[StrategyObservation]:
        return self._list_managed_records(
            evolution_id,
            "strategy_observations",
            self.load_strategy_observation,
        )

    def list_all_strategy_observations(self) -> list[StrategyObservation]:
        observations: list[StrategyObservation] = []
        for task in self.list_tasks():
            observations.extend(self.list_strategy_observations(task.evolution_id))
        return sorted(observations, key=lambda item: item.observation_sha256)

    def write_strategy_posterior(
        self,
        evolution_id: str,
        posterior: StrategyPosteriorSnapshot,
    ) -> Path:
        """Persist one immutable posterior under a managed evolution task."""

        self._validate_id(posterior.posterior_id)
        self._validate_strategy_posterior_provenance(posterior)
        return self._write_record(
            evolution_id=evolution_id,
            directory="strategy_posteriors",
            filename=f"{posterior.posterior_id}.json",
            record=posterior,
            model_type=StrategyPosteriorSnapshot,
        )

    def load_strategy_posterior(
        self,
        evolution_id: str,
        posterior_id: str,
    ) -> StrategyPosteriorSnapshot:
        self._validate_id(posterior_id)
        path = self._managed_path(
            evolution_id,
            "strategy_posteriors",
            f"{posterior_id}.json",
        )
        posterior = self._load_model(
            path, StrategyPosteriorSnapshot, require_schema_version=True
        )
        if posterior.posterior_id != posterior_id:
            raise EvolutionCorruptRecordError(
                path, "strategy posterior identity does not match its managed path"
            )
        self._validate_strategy_posterior_provenance(posterior)
        return posterior

    def list_strategy_posteriors(
        self,
        evolution_id: str,
    ) -> list[StrategyPosteriorSnapshot]:
        return self._list_managed_records(
            evolution_id,
            "strategy_posteriors",
            self.load_strategy_posterior,
        )

    def _validate_strategy_posterior_provenance(
        self,
        posterior: StrategyPosteriorSnapshot,
    ) -> None:
        authoritative = self.list_all_strategy_observations()
        expected_hashes = tuple(
            sorted(
                item.observation_sha256
                for item in authoritative
                if item.created_at <= posterior.training_cutoff_at
            )
        )
        if posterior.training_observation_hashes != expected_hashes:
            raise EvolutionConflictError(
                "strategy posterior is not a cutoff-complete authoritative snapshot"
            )
        by_hash = {item.observation_sha256: item for item in authoritative}
        try:
            training = [
                by_hash[value] for value in posterior.training_observation_hashes
            ]
        except KeyError as exc:
            raise EvolutionConflictError(
                "strategy posterior references a non-authoritative observation hash"
            ) from exc
        recomputed_selector = BayesianLinearStrategySelector(
            seed=0,
            prior_precision=posterior.prior_precision,
            noise_variance=posterior.noise_variance,
        ).fit(training)
        recomputed = recomputed_selector.posterior
        if recomputed is None:
            raise EvolutionConflictError(
                "strategy posterior does not meet the authoritative learning gate"
            )
        counts_match = (
            posterior.observation_count == recomputed.observation_count
            and posterior.effective_training_rows
            == recomputed.effective_training_rows
            and posterior.distinct_task_groups == recomputed.distinct_task_groups
            and posterior.training_cutoff_at == recomputed.training_cutoff_at
        )
        arrays_match = np.allclose(
            posterior.mean,
            recomputed.mean,
            rtol=0.0,
            atol=1e-12,
        ) and np.allclose(
            posterior.covariance,
            recomputed.covariance,
            rtol=0.0,
            atol=1e-12,
        )
        if not counts_match or not arrays_match:
            raise EvolutionConflictError(
                "strategy posterior does not match authoritative observations"
            )

    def _list_managed_records(
        self,
        evolution_id: str,
        directory_name: str,
        loader: Any,
    ) -> list[Any]:
        directory = self._managed_path(evolution_id, directory_name)
        if not directory.is_dir():
            return []
        records: list[Any] = []
        for path in sorted(directory.glob("*.json")):
            try:
                self._validate_id(path.stem)
            except (TypeError, ValueError):
                continue
            records.append(loader(evolution_id, path.stem))
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

    def write_evaluation_scientific_state(
        self,
        evolution_id: str,
        version: str,
        state: ScientificState,
    ) -> Path:
        self._validate_episode_version(version)
        _, payload = self._prepare_model(state, ScientificState)
        task_dir = self._task_dir(evolution_id)
        with self._task_lock(task_dir):
            path = self._managed_path(
                evolution_id,
                "evaluations",
                f"{version}.scientific.json",
            )
            self._write_immutable_json(path, payload)
        return path

    def load_evaluation_scientific_state(
        self,
        evolution_id: str,
        version: str,
    ) -> ScientificState:
        self._validate_episode_version(version)
        return self._load_model(
            self._managed_path(
                evolution_id,
                "evaluations",
                f"{version}.scientific.json",
            ),
            ScientificState,
            require_schema_version=False,
        )

    def write_export(
        self,
        output: Path | str,
        payload: Any,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Atomically write a workspace-contained export with explicit overwrite."""

        path = self.workspace.resolve(str(output), must_exist=False)
        if path == self.workspace.root or path.is_dir():
            raise ValueError("export output must name a file")
        if overwrite:
            self._write_json_atomic(path, redact_secrets(payload))
        else:
            self._write_immutable_json(path, redact_secrets(payload))
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
