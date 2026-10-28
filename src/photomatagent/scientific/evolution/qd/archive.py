"""Deterministic comparison for qualified, protocol-compatible validations."""
from __future__ import annotations

import math
import hashlib
import json
import re
import time
from collections.abc import Mapping
from statistics import median
from types import MappingProxyType

from photomatagent.scientific.evolution.qd.models import (
    ArchiveDecision, ArchiveEntry, EvaluationContract, ProposalRecord, TrialRecord, ValidationRecord, WorkflowGenome,
)
from photomatagent.scientific.evolution.qd.store import QDStore, QDStoreConflictError
from photomatagent.scientific.evolution.qd.validation import CHECKER_VERSION, canonical_contract_sha256


def compare_validations(new: ValidationRecord, old: ValidationRecord,
                        contract: EvaluationContract) -> ArchiveDecision:
    expected = canonical_contract_sha256(contract)
    if (new.contract_sha256 != old.contract_sha256 or new.contract_sha256 != expected
            or new.suite_sha256 != old.suite_sha256
            or new.suite_sha256 != contract.evaluation_suite_sha256
            or new.checker_version != old.checker_version
            or new.seed_set != old.seed_set or new.seed_set != contract.seed_set
            or new.resource_profile != old.resource_profile):
        return ArchiveDecision(action="INCOMPARABLE", reason="PROTOCOL_MISMATCH")
    if new.qualification != "PASS" or new.quality is None or not math.isfinite(new.quality):
        return ArchiveDecision(action="REJECT", reason="NEW_NOT_QUALIFIED")
    if old.qualification != "PASS" or old.quality is None or not math.isfinite(old.quality):
        return ArchiveDecision(action="REPLACE", reason="OLD_NOT_QUALIFIED")
    delta = new.quality - old.quality
    if abs(delta) > contract.quality_tolerance:
        better = delta < 0 if contract.quality_direction == "minimize" else delta > 0
        return ArchiveDecision(action="REPLACE" if better else "KEEP", reason="QUALITY")
    if new.runtime_samples_seconds and old.runtime_samples_seconds:
        new_runtime, old_runtime = median(new.runtime_samples_seconds), median(old.runtime_samples_seconds)
        if abs(new_runtime - old_runtime) >= 0.05 * old_runtime and new_runtime != old_runtime:
            return ArchiveDecision(action="REPLACE" if new_runtime < old_runtime else "KEEP", reason="RUNTIME")
    if new.direct_dependency_count != old.direct_dependency_count:
        return ArchiveDecision(action="REPLACE" if new.direct_dependency_count < old.direct_dependency_count else "KEEP", reason="DEPENDENCIES")
    return ArchiveDecision(action="REPLACE" if new.genome_id < old.genome_id else "KEEP", reason="GENOME_IDENTITY")


class QDArchive:
    """Store-backed archive scoped to one contract and trusted method taxonomy."""

    def __init__(self, store: QDStore, contract: EvaluationContract,
                 genomes_by_id: Mapping[str, WorkflowGenome] | None = None, *,
                 method_class_by_binding: Mapping[str, str], resource_profile: str = "default") -> None:
        self.store = store
        self.contract = contract
        self._provided_genomes = MappingProxyType(dict(genomes_by_id or {}))
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", resource_profile):
            raise ValueError("RESOURCE_PROFILE_INVALID")
        self.resource_profile = resource_profile
        if any(not re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", value)
               for value in method_class_by_binding.values()):
            raise ValueError("METHOD_CLASS_INVALID")
        self.method_class_by_binding = MappingProxyType(dict(method_class_by_binding))

    def _archive_key(self, contract: EvaluationContract, resource_profile: str) -> str:
        payload = {"task_family": contract.task_family,
                   "contract_sha256": canonical_contract_sha256(contract),
                   "suite_sha256": contract.evaluation_suite_sha256,
                   "checker_version": CHECKER_VERSION, "resource_profile": resource_profile}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()
        return "a_" + digest[:32]

    def entries(self) -> tuple[ArchiveEntry, ...]:
        return self.store.load_archive_entries(self._archive_key(self.contract, self.resource_profile))

    def genomes_by_id(self) -> Mapping[str, WorkflowGenome]:
        genomes = dict(self._provided_genomes)
        for entry in self.entries():
            if entry.genome_id not in genomes:
                try:
                    genomes[entry.genome_id] = self.store.load_genome(entry.genome_id)
                except (OSError, ValueError):
                    continue
        return MappingProxyType(genomes)

    def mark_used(self, entries: tuple[ArchiveEntry, ...]) -> None:
        """Persist parent-use counters with the same archive CAS/history contract."""
        for selected in entries:
            for _ in range(3):
                current_entries = self.store.load_archive_entries(selected.archive_key)
                current = next((item for item in current_entries if item.cell_key == selected.cell_key), None)
                if current is None:
                    break
                updated = current.model_copy(update={
                    "revision": current.revision + 1,
                    "previous_entry_ref": hashlib.sha256((json.dumps(
                        current.model_dump(mode="json"), sort_keys=True, separators=(",", ":"),
                        ensure_ascii=False, allow_nan=False,
                    ) + "\n").encode()).hexdigest(),
                    "use_count": current.use_count + 1,
                    "last_used_at_ns": max(time.time_ns(), current.last_used_at_ns + 1),
                })
                try:
                    self.store.commit_archive_entry(
                        updated, expected_revision=self.store.archive_revision(selected.archive_key),
                    )
                    break
                except QDStoreConflictError:
                    continue

    def stage_genome(self, genome: WorkflowGenome, proposal: ProposalRecord) -> None:
        if not isinstance(proposal, ProposalRecord) or proposal.genome_id != genome.genome_id:
            raise ValueError("PROPOSAL_GENOME_MISMATCH")
        self.store.save_genome(genome)
        self.store.save_proposal(proposal)

    def consider(self, *, trial: TrialRecord, validation: ValidationRecord,
                 genome: WorkflowGenome, contract: EvaluationContract) -> ArchiveDecision:
        bundle = hashlib.sha256(json.dumps(
            [ref.model_dump(mode="json") for ref in sorted(trial.output_refs, key=lambda ref: ref.path)],
            sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        ).encode()).hexdigest()
        if (trial.status != "VALIDATED" or validation.trial_id != trial.trial_id
                or genome.genome_id != trial.genome_id or contract != self.contract
                or validation.resource_profile != trial.resource_profile
                or validation.resource_profile != self.resource_profile
                or validation.genome_id != genome.genome_id
                or validation.genome_sha256 != genome.content_sha256
                or validation.contract_sha256 != canonical_contract_sha256(contract)
                or validation.suite_sha256 != contract.evaluation_suite_sha256
                or validation.checker_version != CHECKER_VERSION
                or validation.seed_set != trial.seed_set or validation.seed_set != contract.seed_set
                or validation.bundle_sha256 != bundle
                or dict(validation.cost) != dict(trial.cost)):
            return ArchiveDecision(action="REJECT", reason="TRIAL_VALIDATION_MISMATCH")
        if validation.qualification != "PASS" or validation.quality is None or not math.isfinite(validation.quality):
            return ArchiveDecision(action="REJECT", reason="NEW_NOT_QUALIFIED")
        methods = {self.method_class_by_binding.get(step.binding_id) for step in genome.steps}
        methods.discard(None)
        if len(methods) != 1 or any(step.binding_id not in self.method_class_by_binding for step in genome.steps):
            return ArchiveDecision(action="REJECT", reason="DESCRIPTOR_METHOD_UNKNOWN")
        runtime_values = validation.runtime_samples_seconds
        if not runtime_values:
            return ArchiveDecision(action="REJECT", reason="DESCRIPTOR_RUNTIME_UNKNOWN")
        runtime = median(runtime_values)
        cost_bin = "fast" if runtime <= 1.0 else "standard" if runtime <= 10.0 else "expensive"
        resource_profile = validation.resource_profile
        archive_key = self._archive_key(contract, resource_profile)
        method_class = next(iter(methods))
        cell_key = f"{method_class}_{cost_bin}"
        for _ in range(3):
            entries = {entry.cell_key: entry for entry in self.store.load_archive_entries(archive_key)}
            old_entry = entries.get(cell_key)
            if old_entry is None:
                decision = ArchiveDecision(action="INSERT", reason="EMPTY_CELL")
            else:
                old_validation = self.store.load_validation(old_entry.trial_ids[-1], old_entry.validation_ids[-1])
                decision = compare_validations(validation, old_validation, contract)
                decision = decision.model_copy(update={"old_entry": old_entry})
                if decision.action not in ("REPLACE",):
                    return decision
            old = old_entry
            prior_hash = (hashlib.sha256((json.dumps(old.model_dump(mode="json"), sort_keys=True,
                                   separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()).hexdigest()
                          if old else None)
            updated = ArchiveEntry(
                archive_key=archive_key, cell_key=cell_key, genome_id=genome.genome_id,
                trial_ids=((*old.trial_ids, trial.trial_id) if old else (trial.trial_id,)),
                validation_ids=((*old.validation_ids, validation.validation_id) if old else (validation.validation_id,)),
                quality=validation.quality, cost=trial.cost, revision=(old.revision + 1 if old else 0),
                previous_entry_ref=prior_hash, use_count=(old.use_count if old else 0),
                last_used_at_ns=(old.last_used_at_ns if old else 0),
            )
            decision = decision.model_copy(update={"new_entry": updated})
            try:
                self.store.commit_archive_entry(updated, expected_revision=self.store.archive_revision(archive_key))
            except QDStoreConflictError:
                continue
            return decision
        return ArchiveDecision(action="REJECT", reason="ARCHIVE_CONFLICT")

    def best(self, *, contract: EvaluationContract) -> ArchiveEntry | None:
        if contract != self.contract:
            return None
        entries = self.store.load_archive_entries(self._archive_key(contract, self.resource_profile))
        qualified = [entry for entry in entries if math.isfinite(entry.quality)]
        if not qualified:
            return None
        return min(qualified, key=lambda item: (item.quality, item.cell_key)) if contract.quality_direction == "minimize" else max(qualified, key=lambda item: (item.quality, item.cell_key))


__all__ = ["QDArchive", "compare_validations"]
