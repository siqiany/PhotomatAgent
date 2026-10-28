"""Crash durable storage for quality-diversity trial state."""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator

from photomatagent.scientific.evolution.qd.models import (
    ArchiveEntry, ArtifactDigest, ProposalBatch, ProposalRecord, TrialRecord, ValidationRecord,
    WorkflowGenome, canonical_genome_sha256,
)
from photomatagent.workspace import Workspace


class QDStoreError(RuntimeError):
    """Base class for QD persistence errors."""


class QDStoreConflictError(QDStoreError):
    """An operation conflicts with an immutable record or stale revision."""


class QDStoreCorruptError(QDStoreError):
    """Persisted bytes or their content digest are invalid."""


class QDStorePathError(QDStoreError):
    """A QD path escapes its workspace or traverses a symlink."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()


def _seal_revision(value: dict[str, Any]) -> dict[str, Any]:
    sealed = dict(value)
    sealed["content_sha256"] = hashlib.sha256(_json_bytes(value)).hexdigest()
    return sealed


class QDStore:
    def __init__(self, workspace: Workspace, *, trial_factory: Callable[[str, str], TrialRecord]) -> None:
        self.workspace = workspace
        self.trial_factory = trial_factory
        self.root = workspace.root / ".photomatagent" / "qd"
        self._safe_directory(self.root)

    def reserve_trial(self, event_id: str, genome_id: str) -> TrialRecord:
        record = self.trial_factory(event_id, genome_id)
        self._validate_trial_id(record.trial_id)
        if record.event_id != event_id or record.genome_id != genome_id or record.status != "RESERVED":
            raise QDStoreConflictError("trial factory returned mismatched IDs or non-RESERVED state")
        with self._locked():
            trials_dir = self.root / "trials"
            if trials_dir.exists():
                self._safe_path(trials_dir)
                for prior_manifest in trials_dir.glob("*/manifest.json"):
                    prior_dir = prior_manifest.parent
                    prior = self._load_trial_unlocked(prior_dir)
                    if prior.event_id == event_id and prior.genome_id == genome_id:
                        expected = record.model_dump(mode="json")
                        expected["trial_id"] = prior.trial_id
                        existing = prior.model_dump(mode="json")
                        expected.pop("status")
                        existing.pop("status")
                        if expected != existing:
                            raise QDStoreConflictError("event/genome already reserved with different content")
                        return prior
            directory = self.root / "trials" / record.trial_id
            self._safe_directory(directory / "revisions")
            manifest = directory / "manifest.json"
            if manifest.exists():
                current = self._load_trial_unlocked(directory)
                if current.model_dump(mode="json") != record.model_dump(mode="json"):
                    raise QDStoreConflictError("trial ID already reserved with different content")
                return current
            revision = _seal_revision({"revision": 0, "trial": record.model_dump(mode="json"), "validation": None})
            digest = self._write_immutable(directory / "revisions" / "000000.json", revision)
            self._write_atomic(manifest, {"revision": 0, "sha256": digest})
            return record

    def commit_validation(self, record: ValidationRecord) -> None:
        with self._locked():
            directory = self.root / "trials" / record.trial_id
            current = self._load_trial_unlocked(directory)
            if record.trial_id != current.trial_id or record.contract_sha256 != current.contract_sha256:
                raise QDStoreConflictError("validation trial or contract does not match reserved trial")
            manifest = self._read_json(directory / "manifest.json")
            current_rev = int(manifest["revision"])
            if current.status == "VALIDATED":
                existing = self._read_revision(directory, current_rev).get("validation")
                if existing == record.model_dump(mode="json"):
                    return
                raise QDStoreConflictError("trial already has a different validation")
            if current.status != "COLLECTED":
                raise QDStoreConflictError("trial cannot be validated from its current state")
            validation_path = directory / "validations" / f"{record.validation_id}.json"
            self._safe_directory(validation_path.parent)
            self._write_immutable(validation_path, record.model_dump(mode="json"))
            validated = current.model_copy(update={"status": "VALIDATED"})
            self._append_trial_revision(directory, manifest, current_rev, validated, record)

    def start_trial(self, trial_id: str) -> TrialRecord:
        """Persist RESERVED -> RUNNING before an executor can be invoked."""
        return self._transition_status(trial_id, "RUNNING")

    @contextmanager
    def trial_execution_lease(self, trial_id: str) -> Iterator[None]:
        """Hold a process-scoped lease so a concurrent coordinator cannot interrupt a live run."""
        self._validate_trial_id(trial_id)
        directory = self.root / "trials" / trial_id
        with self._locked():
            self._load_trial_unlocked(directory)
        lock_path = directory / ".execution.lock"
        self._safe_path(lock_path)
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise QDStoreConflictError("trial has an active execution owner") from exc
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def commit_trial_outcome(self, record: TrialRecord) -> TrialRecord:
        """Persist collected outputs or a terminal failure under append-only CAS."""
        self._validate_trial_id(record.trial_id)
        if record.status not in ("COLLECTED", "FAILED", "CANCELLED", "INTERRUPTED"):
            raise QDStoreConflictError("outcome status is not terminal or collected")
        with self._locked():
            directory = self.root / "trials" / record.trial_id
            current = self._load_trial_unlocked(directory)
            if current.status == record.status and current.model_dump(mode="json") == record.model_dump(mode="json"):
                return current
            if current.status != "RUNNING":
                raise QDStoreConflictError("trial outcome requires RUNNING state")
            if self._trial_execution_identity(current) != self._trial_execution_identity(record):
                raise QDStoreConflictError("trial outcome changed reserved identity")
            if not self._valid_trace_ref(record):
                raise QDStorePathError("runtime trace reference does not match session identity")
            if record.status == "COLLECTED":
                self._verify_output_refs(record)
            manifest = self._read_json(directory / "manifest.json")
            revision = int(manifest["revision"])
            self._append_trial_revision(directory, manifest, revision, record, None)
            return record

    def load_trial(self, trial_id: str) -> TrialRecord:
        self._validate_trial_id(trial_id)
        with self._locked():
            return self._load_trial_unlocked(self.root / "trials" / trial_id)

    def trials_for_event(self, event_id: str) -> tuple[TrialRecord, ...]:
        """Load persisted trials for one event in stable reservation order."""
        self._validate_trial_id(event_id)
        with self._locked():
            trials_dir = self.root / "trials"
            if not trials_dir.exists():
                return ()
            self._safe_path(trials_dir)
            rows = [self._load_trial_unlocked(path.parent)
                    for path in trials_dir.glob("*/manifest.json")]
        return tuple(sorted((row for row in rows if row.event_id == event_id),
                            key=lambda row: row.trial_id))

    def load_archive_entries(self, archive_key: str) -> tuple[ArchiveEntry, ...]:
        self._validate_trial_id(archive_key)
        with self._locked():
            return tuple(self._load_archive_unlocked(archive_key).values())

    def archive_revision(self, archive_key: str) -> int:
        self._validate_trial_id(archive_key)
        with self._locked():
            path = self.root / "archives" / archive_key / "manifest.json"
            return int(self._read_json(path)["revision"]) if path.exists() else 0

    def load_validation(self, trial_id: str, validation_id: str) -> ValidationRecord:
        self._validate_trial_id(trial_id)
        self._validate_trial_id(validation_id)
        with self._locked():
            directory = self.root / "trials" / trial_id
            trial = self._load_trial_unlocked(directory)
            raw = self._read_json(directory / "validations" / f"{validation_id}.json")
            try:
                validation = ValidationRecord.model_validate(raw)
            except ValueError as exc:
                raise QDStoreCorruptError("invalid persisted validation") from exc
            if validation.trial_id != trial_id or trial.status != "VALIDATED":
                raise QDStoreCorruptError("validation is not attached to a validated trial")
            return validation

    def load_current_validation(self, trial_id: str) -> ValidationRecord:
        self._validate_trial_id(trial_id)
        with self._locked():
            directory = self.root / "trials" / trial_id
            trial = self._load_trial_unlocked(directory)
            manifest = self._read_json(directory / "manifest.json")
            revision = self._read_revision(directory, int(manifest["revision"]))
            raw = revision.get("validation")
            if trial.status != "VALIDATED" or raw is None:
                raise QDStoreConflictError("trial has no current validation")
            try:
                validation = ValidationRecord.model_validate(raw)
            except ValueError as exc:
                raise QDStoreCorruptError("invalid current validation") from exc
            if validation.trial_id != trial_id:
                raise QDStoreCorruptError("current validation trial mismatch")
            return validation

    def save_genome(self, genome: WorkflowGenome) -> None:
        if genome.content_sha256 != canonical_genome_sha256(genome):
            raise QDStoreConflictError("genome digest is not canonical")
        self._validate_trial_id(genome.genome_id)
        with self._locked():
            path = self.root / "genomes" / f"{genome.genome_id}.json"
            self._safe_directory(path.parent)
            self._write_immutable(path, genome.model_dump(mode="json"))

    def load_genome(self, genome_id: str) -> WorkflowGenome:
        self._validate_trial_id(genome_id)
        with self._locked():
            raw = self._read_json(self.root / "genomes" / f"{genome_id}.json")
            try:
                genome = WorkflowGenome.model_validate(raw)
            except ValueError as exc:
                raise QDStoreCorruptError("invalid persisted genome") from exc
            if genome.genome_id != genome_id or genome.content_sha256 != canonical_genome_sha256(genome):
                raise QDStoreCorruptError("persisted genome identity or digest mismatch")
            return genome

    def save_proposal(self, proposal: ProposalRecord) -> None:
        self._validate_trial_id(proposal.proposal_id)
        with self._locked():
            path = self.root / "proposals" / f"{proposal.proposal_id}.json"
            self._safe_directory(path.parent)
            self._write_immutable(path, proposal.model_dump(mode="json"))

    def save_proposal_batch(
        self, event_id: str, batch_index: int, batch: ProposalBatch,
        genomes_by_id: Mapping[str, WorkflowGenome], parent_ids: tuple[str, ...],
        elapsed_seconds: float,
    ) -> None:
        self._validate_trial_id(event_id)
        if (isinstance(batch_index, bool) or batch_index < 0 or not math.isfinite(elapsed_seconds)
                or elapsed_seconds < 0 or len(set(parent_ids)) != len(parent_ids)):
            raise QDStoreConflictError("invalid proposal batch metadata")
        selected: dict[str, WorkflowGenome] = {}
        for proposal in batch.proposals:
            genome = genomes_by_id.get(proposal.genome_id)
            if (genome is None or genome.genome_id != proposal.genome_id
                    or genome.content_sha256 != canonical_genome_sha256(genome)):
                raise QDStoreConflictError("proposal batch genome is missing or invalid")
            selected[genome.genome_id] = genome
        value = {
            "event_id": event_id, "batch_index": batch_index,
            "batch": batch.model_dump(mode="json"),
            "parent_ids": list(parent_ids), "elapsed_seconds": elapsed_seconds,
            "genomes": {key: genome.model_dump(mode="json") for key, genome in sorted(selected.items())},
        }
        with self._locked():
            path = self.root / "proposals" / "batches" / event_id / f"{batch_index:06d}.json"
            self._safe_directory(path.parent)
            self._write_immutable(path, value)

    def load_proposal_batch(
        self, event_id: str, batch_index: int,
    ) -> tuple[ProposalBatch, Mapping[str, WorkflowGenome], tuple[str, ...], float] | None:
        self._validate_trial_id(event_id)
        if isinstance(batch_index, bool) or batch_index < 0:
            raise QDStorePathError("proposal batch index is invalid")
        with self._locked():
            path = self.root / "proposals" / "batches" / event_id / f"{batch_index:06d}.json"
            if not path.exists():
                return None
            value = self._read_json(path)
            try:
                if value.get("event_id") != event_id or value.get("batch_index") != batch_index:
                    raise ValueError("proposal batch identity mismatch")
                batch = ProposalBatch.model_validate(value["batch"])
                genomes = {key: WorkflowGenome.model_validate(raw) for key, raw in value["genomes"].items()}
                parents = tuple(value["parent_ids"])
                elapsed = value["elapsed_seconds"]
                if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                        or not math.isfinite(elapsed) or elapsed < 0
                        or any(key != genome.genome_id or genome.content_sha256 != canonical_genome_sha256(genome)
                               for key, genome in genomes.items())
                        or any(proposal.genome_id not in genomes for proposal in batch.proposals)):
                    raise ValueError("proposal batch references are invalid")
                return batch, MappingProxyType(genomes), parents, float(elapsed)
            except (KeyError, TypeError, ValueError) as exc:
                raise QDStoreCorruptError(f"invalid proposal batch: {exc}") from exc

    def commit_archive_entry(self, entry: ArchiveEntry, *, expected_revision: int) -> int:
        """Append one cell update to the archive index using whole-index CAS."""
        self._validate_trial_id(entry.archive_key)
        self._validate_trial_id(entry.cell_key)
        with self._locked():
            archive_dir = self.root / "archives" / entry.archive_key
            self._safe_directory(archive_dir / "revisions")
            manifest_path = archive_dir / "manifest.json"
            current_manifest = self._read_json(manifest_path) if manifest_path.exists() else None
            current_revision = int(current_manifest["revision"]) if current_manifest else 0
            if current_revision != expected_revision:
                raise QDStoreConflictError("stale archive index revision")
            current_entries = self._load_archive_unlocked(entry.archive_key)
            previous = current_entries.get(entry.cell_key)
            expected_entry_revision = previous.revision + 1 if previous else 0
            if entry.revision != expected_entry_revision:
                raise QDStoreConflictError("invalid archive cell revision")
            if previous is None:
                if entry.previous_entry_ref is not None or len(entry.trial_ids) != 1:
                    raise QDStoreConflictError("new archive cell cannot reference a prior entry")
            else:
                previous_digest = hashlib.sha256(_json_bytes(previous.model_dump(mode="json"))).hexdigest()
                if entry.previous_entry_ref != previous_digest:
                    raise QDStoreConflictError("archive history reference mismatch")
                same_lineage = entry.trial_ids == previous.trial_ids and entry.validation_ids == previous.validation_ids
                appended_once = (entry.trial_ids[:-1] == previous.trial_ids
                                 and entry.validation_ids[:-1] == previous.validation_ids
                                 and len(entry.trial_ids) == len(previous.trial_ids) + 1)
                if not (same_lineage or appended_once):
                    raise QDStoreConflictError("archive update dropped or rewrote entry lineage")
            trial_id, validation_id = entry.trial_ids[-1], entry.validation_ids[-1]
            trial = self._load_trial_unlocked(self.root / "trials" / trial_id)
            validation_raw = self._read_json(self.root / "trials" / trial_id / "validations" / f"{validation_id}.json")
            try:
                validation = ValidationRecord.model_validate(validation_raw)
            except ValueError as exc:
                raise QDStoreCorruptError("archive references an invalid validation") from exc
            genome_raw = self._read_json(self.root / "genomes" / f"{entry.genome_id}.json")
            try:
                genome = WorkflowGenome.model_validate(genome_raw)
            except ValueError as exc:
                raise QDStoreCorruptError("archive references an invalid genome") from exc
            bundle = hashlib.sha256(json.dumps(
                [ref.model_dump(mode="json") for ref in sorted(trial.output_refs, key=lambda ref: ref.path)],
                sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
            ).encode()).hexdigest()
            if (trial.status != "VALIDATED" or trial.genome_id != entry.genome_id
                    or validation.trial_id != trial_id or validation.genome_id != entry.genome_id
                    or validation.qualification != "PASS"
                    or validation.quality is None or not math.isfinite(validation.quality)
                    or validation.quality != entry.quality
                    or validation.genome_sha256 != genome.content_sha256
                    or validation.contract_sha256 != trial.contract_sha256
                    or validation.bundle_sha256 != bundle
                    or validation.seed_set != trial.seed_set
                    or validation.resource_profile != trial.resource_profile
                    or dict(validation.cost) != dict(trial.cost)
                    or dict(entry.cost) != dict(trial.cost)):
                raise QDStoreConflictError("archive entry is not backed by a qualified validated trial")
            current_entries[entry.cell_key] = entry
            new_revision = current_revision + 1
            value = _seal_revision({
                "revision": new_revision,
                "entries": {key: item.model_dump(mode="json") for key, item in sorted(current_entries.items())},
            })
            digest = self._write_immutable(archive_dir / "revisions" / f"{new_revision:06d}.json", value)
            if current_manifest != (self._read_json(manifest_path) if manifest_path.exists() else None):
                raise QDStoreConflictError("stale archive index revision")
            self._write_atomic(manifest_path, {"revision": new_revision, "sha256": digest})
            return new_revision

    def _load_archive_unlocked(self, archive_key: str) -> dict[str, ArchiveEntry]:
        archive_dir = self.root / "archives" / archive_key
        manifest_path = archive_dir / "manifest.json"
        if not manifest_path.exists():
            return {}
        self._safe_path(manifest_path)
        manifest = self._read_json(manifest_path)
        try:
            revision, digest = manifest["revision"], manifest["sha256"]
            if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1 or not isinstance(digest, str):
                raise ValueError("invalid archive manifest")
            revision_path = archive_dir / "revisions" / f"{revision:06d}.json"
            if hashlib.sha256(revision_path.read_bytes()).hexdigest() != digest:
                raise ValueError("archive revision digest mismatch")
            value = self._read_revision(archive_dir, revision)
            if (isinstance(value.get("revision"), bool) or value.get("revision") != revision
                    or not isinstance(value.get("entries"), dict)):
                raise ValueError("invalid archive revision")
            entries = {key: ArchiveEntry.model_validate(raw) for key, raw in value["entries"].items()}
            if any(key != item.cell_key or item.archive_key != archive_key for key, item in entries.items()):
                raise ValueError("archive entry key mismatch")
            return entries
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise QDStoreCorruptError(f"invalid archive index: {exc}") from exc

    def _transition_status(self, trial_id: str, status: str) -> TrialRecord:
        self._validate_trial_id(trial_id)
        with self._locked():
            directory = self.root / "trials" / trial_id
            current = self._load_trial_unlocked(directory)
            if (current.status, status) not in (("RESERVED", "RUNNING"),):
                raise QDStoreConflictError("invalid trial status transition")
            manifest = self._read_json(directory / "manifest.json")
            revision = int(manifest["revision"])
            updated = current.model_copy(update={"status": status})
            self._append_trial_revision(directory, manifest, revision, updated, None)
            return updated

    def _append_trial_revision(self, directory: Path, manifest: dict[str, Any], revision: int,
                               trial: TrialRecord, validation: ValidationRecord | None) -> None:
        if self._read_json(directory / "manifest.json") != manifest:
            raise QDStoreConflictError("stale trial manifest revision")
        if validation is not None:
            path = directory / "validations" / f"{validation.validation_id}.json"
            self._safe_directory(path.parent)
            self._write_immutable(path, validation.model_dump(mode="json"))
        value = _seal_revision({"revision": revision + 1, "trial": trial.model_dump(mode="json"),
                                "validation": validation.model_dump(mode="json") if validation else None})
        digest = self._write_immutable(directory / "revisions" / f"{revision + 1:06d}.json", value)
        if self._read_json(directory / "manifest.json") != manifest:
            raise QDStoreConflictError("stale trial manifest revision")
        self._write_atomic(directory / "manifest.json", {"revision": revision + 1, "sha256": digest})

    @staticmethod
    def _trial_execution_identity(record: TrialRecord) -> dict[str, Any]:
        value = record.model_dump(mode="json")
        for field in ("status", "runtime_trace_ref", "cost", "output_refs"):
            value.pop(field)
        return value

    def _verify_output_refs(self, record: TrialRecord) -> None:
        for ref in record.output_refs:
            if not ref.path.startswith(record.attempt_dir + "/"):
                raise QDStorePathError("output artifact is outside the attempt directory")
            path = self.workspace.root
            for part in Path(ref.path).parts:
                path /= part
                if path.is_symlink():
                    raise QDStorePathError("symlink in output artifact path")
            self._safe_path(path)
            try:
                content = path.read_bytes()
            except OSError as exc:
                raise QDStoreCorruptError("collected output artifact is missing") from exc
            if len(content) != ref.size_bytes or hashlib.sha256(content).hexdigest() != ref.sha256:
                raise QDStoreCorruptError("collected output artifact digest mismatch")

    @staticmethod
    def _valid_trace_ref(record: TrialRecord) -> bool:
        return record.runtime_trace_ref in (
            "", f".photomatagent/sessions/{record.session_id}/events.jsonl",
        )

    @staticmethod
    def _allowed_transition(old: TrialRecord, new: TrialRecord) -> bool:
        transitions = {"RESERVED": {"RUNNING"}, "RUNNING": {"COLLECTED", "FAILED", "CANCELLED", "INTERRUPTED"},
                       "COLLECTED": {"VALIDATED"}}
        return new.status in transitions.get(old.status, set())

    @staticmethod
    def _validate_trial_id(trial_id: str) -> None:
        if (not isinstance(trial_id, str) or not trial_id or trial_id in (".", "..")
                or "/" in trial_id or "\\" in trial_id or Path(trial_id).is_absolute()
                or (len(trial_id) >= 2 and trial_id[1] == ":")):
            raise QDStorePathError("trial ID must be one safe path component")

    def _load_trial_unlocked(self, directory: Path) -> TrialRecord:
        self._safe_path(directory)
        if not (directory / "manifest.json").exists():
            raise QDStoreConflictError(f"trial is not reserved: {directory.name}")
        manifest = self._read_json(directory / "manifest.json")
        try:
            revision = manifest["revision"]
            digest = manifest["sha256"]
            if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0 or not isinstance(digest, str):
                raise ValueError("invalid manifest revision or digest")
        except (KeyError, TypeError, ValueError) as exc:
            raise QDStoreCorruptError(f"invalid trial manifest: {exc}") from exc
        revision_path = directory / "revisions" / f"{revision:06d}.json"
        value = self._read_revision(directory, revision)
        if hashlib.sha256(revision_path.read_bytes()).hexdigest() != digest:
            raise QDStoreCorruptError("trial revision digest mismatch")
        try:
            base_revision = value["revision"]
            if isinstance(base_revision, bool) or not isinstance(base_revision, int) or base_revision != revision:
                raise ValueError("manifest-selected base revision number mismatch")
            current_trial = TrialRecord.model_validate(value["trial"])
        except (KeyError, TypeError, ValueError) as exc:
            raise QDStoreCorruptError(f"invalid manifest-selected trial revision: {exc}") from exc
        orphan_path = directory / "revisions" / f"{revision + 1:06d}.json"
        if orphan_path.exists():
            orphan = self._read_revision(directory, revision + 1)
            try:
                orphan_trial = TrialRecord.model_validate(orphan["trial"])
                raw_validation = orphan.get("validation")
                validation = ValidationRecord.model_validate(raw_validation) if raw_validation is not None else None
                old_trial = current_trial
                orphan_revision = orphan["revision"]
                valid = (not isinstance(orphan_revision, bool) and isinstance(orphan_revision, int)
                         and orphan_revision == revision + 1
                         and self._allowed_transition(old_trial, orphan_trial)
                         and self._valid_trace_ref(orphan_trial)
                         and self._trial_execution_identity(orphan_trial) == self._trial_execution_identity(old_trial))
                if orphan_trial.status == "VALIDATED":
                    persisted_validation = self._read_json(directory / "validations" / f"{validation.validation_id}.json") if validation else None
                    valid = (valid and old_trial.status == "COLLECTED" and validation is not None
                             and validation.trial_id == old_trial.trial_id
                             and validation.contract_sha256 == old_trial.contract_sha256
                             and persisted_validation == validation.model_dump(mode="json")
                             and self._trial_identity(orphan_trial) == self._trial_identity(old_trial))
                elif validation is not None:
                    valid = False
                elif orphan_trial.status == "COLLECTED":
                    self._verify_output_refs(orphan_trial)
                if not valid:
                    raise ValueError("orphan revision is not a valid trial transition")
                orphan_digest = hashlib.sha256(orphan_path.read_bytes()).hexdigest()
                manifest_path = directory / "manifest.json"
                if self._read_json(manifest_path) != manifest:
                    raise QDStoreConflictError("stale trial manifest while recovering orphan revision")
                self._write_atomic(manifest_path, {"revision": revision + 1, "sha256": orphan_digest})
                return orphan_trial
            except (KeyError, TypeError, ValueError) as exc:
                raise QDStoreCorruptError(f"invalid orphan trial revision: {exc}") from exc
        return current_trial

    def _read_revision(self, directory: Path, revision: int) -> dict[str, Any]:
        value = self._read_json(directory / "revisions" / f"{revision:06d}.json")
        try:
            digest = value.pop("content_sha256")
            if not isinstance(digest, str) or hashlib.sha256(_json_bytes(value)).hexdigest() != digest:
                raise ValueError("revision content digest mismatch")
            return value
        except (KeyError, TypeError, ValueError) as exc:
            raise QDStoreCorruptError(f"invalid trial revision envelope: {exc}") from exc

    @staticmethod
    def _trial_identity(record: TrialRecord) -> dict[str, Any]:
        value = record.model_dump(mode="json")
        value.pop("status")
        return value

    def _read_json(self, path: Path) -> dict[str, Any]:
        self._safe_path(path)
        try:
            if path.stat().st_size > 1_000_000:
                raise QDStoreCorruptError("QD JSON exceeds size limit")
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("expected object")
            return value
        except QDStoreError:
            raise
        except (OSError, ValueError, TypeError) as exc:
            raise QDStoreCorruptError(f"cannot read {path.name}: {exc}") from exc

    @contextmanager
    def _locked(self) -> Iterator[None]:
        lock = self.root / ".lock"
        self._safe_path(lock)
        fd = os.open(lock, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _safe_directory(self, path: Path) -> None:
        relative = path.relative_to(self.workspace.root)
        current = self.workspace.root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise QDStorePathError(f"symlink in QD path: {current}")
            current.mkdir(exist_ok=True)
            if not current.is_dir():
                raise QDStorePathError(f"not a directory: {current}")

    def _safe_path(self, path: Path) -> None:
        try:
            relative = path.relative_to(self.workspace.root)
        except ValueError as exc:
            raise QDStorePathError("QD path outside workspace") from exc
        current = self.workspace.root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise QDStorePathError(f"symlink in QD path: {current}")

    def _write_immutable(self, path: Path, value: Any) -> str:
        payload = _json_bytes(value)
        if len(payload) > 1_000_000:
            raise QDStoreError("QD JSON exceeds size limit")
        self._safe_directory(path.parent)
        digest = hashlib.sha256(payload).hexdigest()
        if path.exists():
            if path.read_bytes() != payload:
                raise QDStoreConflictError(f"immutable record conflict: {path.name}")
            return digest
        self._write_atomic(path, value, exclusive=True)
        return digest

    def _write_atomic(self, path: Path, value: Any, *, exclusive: bool = False) -> None:
        self._safe_directory(path.parent)
        data = _json_bytes(value)
        fd, name = tempfile.mkstemp(prefix=".qd-", dir=path.parent)
        temp = Path(name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            self._safe_path(path)
            if exclusive and path.exists():
                if path.read_bytes() != data:
                    raise QDStoreConflictError(f"immutable record conflict: {path.name}")
                return
            os.replace(temp, path)
            dfd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        finally:
            temp.unlink(missing_ok=True)


__all__ = ["QDStore", "QDStoreError", "QDStoreConflictError", "QDStoreCorruptError", "QDStorePathError"]
