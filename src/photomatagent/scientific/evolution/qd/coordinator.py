"""Bounded QD event orchestration over trusted host services.

The coordinator never executes model-requested tools. It reserves durable
budget and trial state before invoking the proposal engine or TrialExecutor.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping
from typing import Protocol

from photomatagent.scientific.evolution.qd.budget import (
    BudgetAmounts, BudgetConflictError, BudgetLedger,
)
from photomatagent.scientific.evolution.qd.config import EvolutionBudget
from photomatagent.scientific.evolution.qd.models import (
    ArchiveDecision, ArchiveEntry, EvaluationContract, EventOutcome, ProposalBatch,
    ProposalRecord, ProposalUsage, RevisionIssue, TrialOutcome, TrialRecord, ValidationRecord,
    WorkflowGenome, canonical_genome_sha256,
)
from photomatagent.scientific.evolution.qd.selection import select_parent_entries
from photomatagent.scientific.evolution.qd.store import QDStore, QDStoreConflictError
from photomatagent.scientific.evolution.qd.validation import CHECKER_VERSION, canonical_contract_sha256


class ArchiveService(Protocol):
    """Host-owned, durable archive and immutable Genome staging surface."""

    def entries(self) -> tuple[ArchiveEntry, ...]: ...
    def genomes_by_id(self) -> Mapping[str, WorkflowGenome]: ...
    def mark_used(self, entries: tuple[ArchiveEntry, ...]) -> None: ...
    def stage_genome(self, genome: WorkflowGenome, proposal: ProposalRecord) -> None: ...
    def consider(
        self, *, trial: TrialRecord, validation: ValidationRecord,
        genome: WorkflowGenome, contract: EvaluationContract,
    ) -> ArchiveDecision: ...
    def best(self, *, contract: EvaluationContract) -> ArchiveEntry | None: ...


class ProposalSource(Protocol):
    genomes_by_id: Mapping[str, WorkflowGenome]

    async def propose(
        self, *, parents: tuple[WorkflowGenome, ...], issues: tuple[RevisionIssue, ...],
        contract: EvaluationContract, limit: int, token_budget: int,
    ) -> ProposalBatch: ...


class TrialRunner(Protocol):
    async def execute(
        self, *, trial: TrialRecord, genome: WorkflowGenome,
        contract: EvaluationContract, token_budget: int, model_call_budget: int,
    ) -> TrialOutcome: ...


class TrialValidator(Protocol):
    def validate(self, *, trial: TrialRecord, contract: EvaluationContract) -> ValidationRecord: ...


class EvolutionCoordinator:
    """Run one finite event; a persistent RUNNING trial always blocks replay."""

    def __init__(
        self, *, store: QDStore, ledger: BudgetLedger,
        proposal_engine: ProposalSource, executor: TrialRunner,
        validator: TrialValidator | None, archive: ArchiveService,
        contract: EvaluationContract, baseline: WorkflowGenome,
        issues: tuple[RevisionIssue, ...], budget: EvolutionBudget,
        seed: int = 0, is_solved: Callable[[ValidationRecord], bool] | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> None:
        if baseline.task_family != contract.task_family or baseline.content_sha256 != canonical_genome_sha256(baseline):
            raise ValueError("BASELINE_GENOME_INVALID")
        self.store = store
        self.ledger = ledger
        self.proposal_engine = proposal_engine
        self.executor = executor
        self.validator = validator
        self.archive = archive
        self.contract = contract
        self.baseline = baseline
        self.issues = issues
        self.budget = budget
        self.seed = seed
        self.is_solved = is_solved or (lambda _: False)
        self.cancel_event = cancel_event
        self.cap = BudgetAmounts(
            tokens=budget.max_model_tokens, wall_seconds=budget.max_wall_seconds,
            trials=budget.max_trials, installs=budget.max_installs,
            download_bytes=budget.max_download_bytes,
            model_calls=budget.max_model_calls, proposals=budget.max_proposals,
        )

    async def run_event(self, event_id: str) -> EventOutcome:
        if not event_id or len(event_id) > 100 or not all(c.isalnum() or c in "_.:-" for c in event_id):
            raise ValueError("EVENT_ID_INVALID")
        scope = f"qd_{event_id}_"
        unresolved = {issue.issue_id for issue in self.issues if issue.status == "open"}
        event_issues = self.issues

        def result(status: str) -> EventOutcome:
            return self._result(status, unresolved, scope)

        if self._cancelled():
            return result("CANCELLED")
        recovered = self._recover_validated_trials(event_id, unresolved)
        if recovered == "BLOCKED":
            return result("BLOCKED")
        if recovered == "SOLVED":
            return result("SOLVED")
        if not self.archive.entries():
            baseline_proposal = ProposalRecord(
                proposal_id=_operation_id("baseline", event_id, self.baseline.content_sha256),
                parent_ids=(), operator="host_baseline", feedback_refs=(),
                change_summary="Frozen host baseline workflow", genome_id=self.baseline.genome_id,
                model_provenance={"source": "host"},
            )
            try:
                self.archive.stage_genome(self.baseline, baseline_proposal)
            except Exception:
                return result("BLOCKED")
            state, validation, accepted = await self._trial(event_id, self.baseline, scope)
            if state is not None:
                return result(state)
            self._resolve_issues(unresolved, validation)
            if validation is not None and accepted and self._solved(validation, unresolved):
                return result("SOLVED")
            if not accepted:
                diagnostic_codes = tuple(dict.fromkeys(
                    outcome.reason_code for outcome in (validation.outcomes if validation else ())
                    if outcome.required and outcome.verdict != "PASS"
                )) or (("BASELINE_TRIAL_FAILED",) if validation is None else ("BASELINE_NOT_QUALIFIED",))
                diagnostic = "Prior baseline evidence: " + ", ".join(diagnostic_codes) + ". Diagnose and address it."
                event_issues = tuple(
                    issue.model_copy(update={"requested_actions": (*issue.requested_actions, diagnostic)})
                    if issue.issue_id in unresolved else issue
                    for issue in self.issues
                )
            if self.ledger.snapshot(scope=scope).trials >= self.cap.trials:
                return result("BUDGET_EXHAUSTED")

        stalled = 0
        batch_index = 0
        seen_content = {genome.content_sha256 for genome in self.archive.genomes_by_id().values()}
        seen_content.add(self.baseline.content_sha256)
        while True:
            if self._cancelled():
                return result("CANCELLED")
            snapshot = self.ledger.snapshot(scope=scope)
            if stalled >= self.budget.patience:
                return result("STALLED")
            actionable_issues = tuple(
                issue for issue in event_issues
                if issue.issue_id in unresolved and issue.requested_actions and issue.step_ids
            )
            if not actionable_issues:
                return result("STALLED")
            batch, batch_genomes, selected_parent_ids, batch_state = await self._proposal_batch(
                event_id, batch_index, scope, actionable_issues,
            )
            if batch_state is not None:
                return result(batch_state)
            if self._cancelled():
                return result("CANCELLED")
            if not batch.proposals:
                return result("STALLED")
            improved = False
            for proposal in batch.proposals:
                if self._cancelled():
                    return result("CANCELLED")
                genome = batch_genomes.get(proposal.genome_id)
                if (genome is None or genome.genome_id != proposal.genome_id
                        or genome.content_sha256 != canonical_genome_sha256(genome)
                        or genome.task_family != self.contract.task_family
                        or not proposal.parent_ids
                        or not set(proposal.parent_ids).issubset(set(selected_parent_ids))):
                    return result("BLOCKED")
                if genome.content_sha256 in seen_content:
                    continue
                seen_content.add(genome.content_sha256)
                state, validation, accepted = await self._trial(event_id, genome, scope)
                if state is not None:
                    return result(state)
                improved = improved or accepted
                self._resolve_issues(unresolved, validation)
                if validation is not None and accepted and self._solved(validation, unresolved):
                    return result("SOLVED")
            stalled = 0 if improved else stalled + 1
            batch_index += 1

    async def _proposal_batch(
        self, event_id: str, batch_index: int, scope: str,
        issues: tuple[RevisionIssue, ...],
    ) -> tuple[ProposalBatch, Mapping[str, WorkflowGenome], tuple[str, ...], str | None]:
        operation_id = _operation_id("model", event_id, str(batch_index))
        reservation = self.ledger.reservation(operation_id)
        try:
            stored = self.store.load_proposal_batch(event_id, batch_index)
        except Exception:
            return ProposalBatch(proposals=(), diagnostics=("PROPOSAL_BATCH_CORRUPT",),
                                 usage=ProposalUsage(model_calls=0, model_tokens=0)), {}, (), "BLOCKED"
        if stored is not None:
            if reservation is None:
                return stored[0], stored[1], stored[2], "BLOCKED"
            batch, genomes, parent_ids, elapsed = stored
            if elapsed > reservation.wall_seconds:
                return batch, genomes, parent_ids, "BUDGET_EXHAUSTED"
            try:
                attempted = min(reservation.proposals, len(batch.proposals) + sum(
                    diagnostic == "PROPOSAL_DUPLICATE_CONTENT" for diagnostic in batch.diagnostics
                ))
                self.ledger.settle(operation_id, BudgetAmounts(
                    tokens=batch.usage.model_tokens, wall_seconds=elapsed,
                    model_calls=batch.usage.model_calls, proposals=attempted,
                ))
            except (BudgetConflictError, ValueError):
                return batch, genomes, parent_ids, "BLOCKED"
            return batch, genomes, parent_ids, None

        snapshot = self.ledger.snapshot(scope=scope)
        if reservation is None and (snapshot.trials >= self.cap.trials
                                    or snapshot.model_calls >= self.cap.model_calls
                                    or snapshot.proposals >= self.cap.proposals):
            return ProposalBatch(proposals=(), diagnostics=(), usage=ProposalUsage(model_calls=0, model_tokens=0)), {}, (), "BUDGET_EXHAUSTED"
        if reservation is None:
            remaining = self._remaining(snapshot)
            if remaining.tokens <= 0 or remaining.wall_seconds <= 0:
                return ProposalBatch(proposals=(), diagnostics=(), usage=ProposalUsage(model_calls=0, model_tokens=0)), {}, (), "BUDGET_EXHAUSTED"
            limit = min(self.budget.offspring_per_batch, remaining.proposals, 2)
            reservation = BudgetAmounts(tokens=remaining.tokens, wall_seconds=remaining.wall_seconds,
                                        model_calls=1, proposals=limit)
            try:
                self.ledger.reserve_with_cap(operation_id, reservation, self.cap, scope=scope)
            except BudgetConflictError:
                return ProposalBatch(proposals=(), diagnostics=(), usage=ProposalUsage(model_calls=0, model_tokens=0)), {}, (), "BUDGET_EXHAUSTED"
        else:
            limit = min(reservation.proposals, 2)
        if limit <= 0:
            return ProposalBatch(proposals=(), diagnostics=(), usage=ProposalUsage(model_calls=0, model_tokens=0)), {}, (), "BUDGET_EXHAUSTED"
        all_entries = self.archive.entries()
        genomes_by_id = self.archive.genomes_by_id()
        selected_entries = select_parent_entries(
            all_entries, genomes_by_id, self.seed + batch_index, 2,
            task_family=self.contract.task_family,
        )
        parents = tuple(genomes_by_id[entry.genome_id] for entry in selected_entries)
        if not parents and self.baseline.task_family == self.contract.task_family:
            # A failed or UNKNOWN baseline is not an elite, but its diagnostics
            # must still be able to seed a bounded repair proposal.
            parents = (self.baseline,)
        if not parents:
            return ProposalBatch(proposals=(), diagnostics=(), usage=ProposalUsage(model_calls=0, model_tokens=0)), {}, (), "BLOCKED"
        try:
            if not self.ledger.begin(operation_id):
                return ProposalBatch(proposals=(), diagnostics=(), usage=ProposalUsage(model_calls=0, model_tokens=0)), {}, (), "BLOCKED"
            self.archive.mark_used(selected_entries)
        except Exception:
            return ProposalBatch(proposals=(), diagnostics=(), usage=ProposalUsage(model_calls=0, model_tokens=0)), {}, (), "BLOCKED"
        started = time.monotonic()
        try:
            set_context = getattr(self.proposal_engine, "set_batch_context", None)
            if callable(set_context):
                set_context(seed=self.seed, batch_index=batch_index)
            batch = await asyncio.wait_for(
                self.proposal_engine.propose(parents=parents, issues=issues,
                                             contract=self.contract, limit=limit,
                                             token_budget=reservation.tokens),
                timeout=reservation.wall_seconds,
            )
        except asyncio.TimeoutError:
            return ProposalBatch(proposals=(), diagnostics=("PROPOSAL_TIMEOUT",),
                                 usage=ProposalUsage(model_calls=0, model_tokens=0)), {}, (), "BUDGET_EXHAUSTED"
        except Exception:
            return ProposalBatch(proposals=(), diagnostics=("PROPOSAL_PROVIDER_UNAVAILABLE",),
                                 usage=ProposalUsage(model_calls=0, model_tokens=0)), {}, (), "BLOCKED"
        elapsed = max(0.0, time.monotonic() - started)
        no_call_for_token_budget = (
            batch.usage.model_calls == 0 and not batch.proposals
            and "PROPOSAL_TOKEN_BUDGET_EXHAUSTED" in batch.diagnostics
        )
        host_random_batch = (
            batch.usage.model_calls == 0
            and batch.usage.model_tokens == 0
            and bool(batch.diagnostics)
            and all(diagnostic.startswith("RANDOM_") for diagnostic in batch.diagnostics)
            and all(proposal.model_provenance.get("source") == "host_random" for proposal in batch.proposals)
        )
        if (len(batch.proposals) > limit
                or (batch.usage.model_calls != 1 and not no_call_for_token_budget and not host_random_batch)
                or batch.usage.model_tokens > reservation.tokens):
            return batch, {}, tuple(parent.genome_id for parent in parents), "BLOCKED"
        candidate_genomes: dict[str, WorkflowGenome] = {}
        allowed_parents = {parent.genome_id for parent in parents}
        for proposal in batch.proposals:
            genome = self.proposal_engine.genomes_by_id.get(proposal.genome_id)
            if (genome is None or genome.genome_id != proposal.genome_id
                    or genome.content_sha256 != canonical_genome_sha256(genome)
                    or genome.task_family != self.contract.task_family or not proposal.parent_ids
                    or not set(proposal.parent_ids).issubset(allowed_parents)):
                return batch, {}, tuple(parent.genome_id for parent in parents), "BLOCKED"
            candidate_genomes[genome.genome_id] = genome
            try:
                self.archive.stage_genome(genome, proposal)
            except Exception:
                return batch, {}, tuple(parent.genome_id for parent in parents), "BLOCKED"
        parent_ids = tuple(parent.genome_id for parent in parents)
        try:
            self.store.save_proposal_batch(event_id, batch_index, batch, candidate_genomes,
                                           parent_ids, elapsed)
            attempted = min(limit, len(batch.proposals) + sum(
                diagnostic == "PROPOSAL_DUPLICATE_CONTENT" for diagnostic in batch.diagnostics
            ))
            self.ledger.settle(operation_id, BudgetAmounts(
                tokens=batch.usage.model_tokens, wall_seconds=elapsed,
                model_calls=batch.usage.model_calls, proposals=attempted,
            ))
        except (QDStoreConflictError, BudgetConflictError, ValueError):
            return batch, candidate_genomes, parent_ids, "BLOCKED"
        return batch, candidate_genomes, parent_ids, None

    async def _trial(self, event_id: str, genome: WorkflowGenome,
                     scope: str) -> tuple[str | None, ValidationRecord | None, bool]:
        operation_id = _operation_id("trial", event_id, genome.content_sha256)
        reservation = self.ledger.reservation(operation_id)
        if reservation is None:
            remaining = self._remaining(self.ledger.snapshot(scope=scope))
            if (remaining.trials <= 0 or remaining.tokens <= 0 or remaining.wall_seconds <= 0
                    or remaining.model_calls <= 0):
                return "BUDGET_EXHAUSTED", None, False
            reservation = BudgetAmounts(
                tokens=remaining.tokens, wall_seconds=remaining.wall_seconds,
                trials=1, model_calls=remaining.model_calls,
            )
            try:
                self.ledger.reserve_with_cap(operation_id, reservation, self.cap, scope=scope)
            except BudgetConflictError:
                return "BUDGET_EXHAUSTED", None, False
        try:
            trial = self.store.reserve_trial(event_id, genome.genome_id)
        except Exception:
            return "BLOCKED", None, False
        try:
            with self.store.trial_execution_lease(trial.trial_id):
                return await self._run_trial_leased(
                    trial, genome, operation_id, reservation,
                )
        except QDStoreConflictError:
            return "BLOCKED", None, False

    async def _run_trial_leased(
        self, trial: TrialRecord, genome: WorkflowGenome,
        operation_id: str, reservation: BudgetAmounts,
    ) -> tuple[str | None, ValidationRecord | None, bool]:
        if trial.status == "RUNNING":
            try:
                self.store.commit_trial_outcome(trial.model_copy(update={"status": "INTERRUPTED"}))
            except Exception:
                return "BLOCKED", None, False
            return "BLOCKED", None, False
        if trial.status == "VALIDATED":
            try:
                validation = self.store.load_current_validation(trial.trial_id)
            except Exception:
                return "BLOCKED", None, False
            return self._after_validation(trial, validation, genome, operation_id, reservation)
        if trial.status == "INTERRUPTED":
            return "BLOCKED", None, False
        if trial.status in ("FAILED", "CANCELLED"):
            if not self._settle_trial(operation_id, trial, reservation):
                return "BLOCKED", None, False
            return None, None, False
        if trial.status == "COLLECTED":
            if self.validator is None:
                return "BLOCKED", None, False
            try:
                validation = self.validator.validate(trial=trial, contract=self.contract)
                self.store.commit_validation(validation)
            except Exception:
                return "BLOCKED", None, False
            return self._after_validation(trial, validation, genome, operation_id, reservation)
        try:
            running = self.store.start_trial(trial.trial_id)
        except Exception:
            return "BLOCKED", None, False
        try:
            outcome = await asyncio.wait_for(
                self.executor.execute(trial=running, genome=genome, contract=self.contract,
                                      token_budget=reservation.tokens,
                                      model_call_budget=reservation.model_calls),
                timeout=reservation.wall_seconds,
            )
        except Exception:
            try:
                self.store.commit_trial_outcome(running.model_copy(update={"status": "INTERRUPTED"}))
            except Exception:
                pass
            return "BLOCKED", None, False
        if outcome.trial.trial_id != running.trial_id or outcome.trial.genome_id != genome.genome_id:
            return "BLOCKED", None, False
        if outcome.validation is not None:
            if outcome.trial.status not in ("COLLECTED", "VALIDATED"):
                return "BLOCKED", None, False
            collected = outcome.trial.model_copy(update={"status": "COLLECTED"})
        else:
            if outcome.trial.status not in ("FAILED", "CANCELLED", "INTERRUPTED"):
                return "BLOCKED", None, False
            collected = outcome.trial
        try:
            persisted = self.store.commit_trial_outcome(collected)
            if outcome.validation is not None:
                if not self._validation_identity(persisted, outcome.validation, genome):
                    return "BLOCKED", None, False
                self.store.commit_validation(outcome.validation)
        except Exception:
            return "BLOCKED", None, False
        if outcome.validation is None:
            if persisted.status == "INTERRUPTED":
                return "BLOCKED", None, False
            if not self._settle_trial(operation_id, persisted, reservation):
                return "BLOCKED", None, False
            return None, None, False
        return self._after_validation(persisted, outcome.validation, genome, operation_id, reservation)

    def _after_validation(
        self, trial: TrialRecord, validation: ValidationRecord, genome: WorkflowGenome,
        operation_id: str, reservation: BudgetAmounts,
    ) -> tuple[str | None, ValidationRecord | None, bool]:
        if not self._validation_identity(trial, validation, genome):
            return "BLOCKED", None, False
        if not self._settle_trial(operation_id, trial, reservation):
            return "BLOCKED", None, False
        if validation.qualification != "PASS" or validation.quality is None or not math.isfinite(validation.quality):
            return None, validation, False
        try:
            validated_trial = self.store.load_trial(trial.trial_id)
            if validated_trial.status != "VALIDATED":
                return "BLOCKED", validation, False
            decision = self.archive.consider(trial=validated_trial, validation=validation,
                                             genome=genome, contract=self.contract)
        except Exception:
            return "BLOCKED", validation, False
        return None, validation, decision.action in ("INSERT", "REPLACE")

    def _validation_identity(self, trial: TrialRecord, validation: ValidationRecord,
                             genome: WorkflowGenome) -> bool:
        refs = [ref.model_dump(mode="json") for ref in sorted(trial.output_refs, key=lambda ref: ref.path)]
        bundle = hashlib.sha256(json.dumps(refs, sort_keys=True, separators=(",", ":"),
                                           ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
        return (
            validation.trial_id == trial.trial_id
            and validation.genome_id == genome.genome_id
            and validation.genome_sha256 == genome.content_sha256
            and validation.bundle_sha256 == bundle
            and validation.contract_sha256 == canonical_contract_sha256(self.contract)
            and validation.suite_sha256 == self.contract.evaluation_suite_sha256
            and validation.checker_version == CHECKER_VERSION
            and validation.seed_set == trial.seed_set == self.contract.seed_set
            and validation.resource_profile == trial.resource_profile
            and dict(validation.cost) == dict(trial.cost)
        )

    def _settle_trial(self, operation_id: str, trial: TrialRecord, reservation: BudgetAmounts) -> bool:
        try:
            tokens = trial.cost.get("model_tokens")
            wall = trial.cost.get("runtime_seconds")
            if (tokens is None or wall is None or not math.isfinite(tokens) or not math.isfinite(wall)
                    or tokens < 0 or wall < 0 or not float(tokens).is_integer()
                    or tokens > reservation.tokens or wall > reservation.wall_seconds):
                return False
            calls = trial.cost.get("model_calls", 0)
            if not math.isfinite(calls) or calls < 0 or not float(calls).is_integer():
                return False
            self.ledger.settle(operation_id, BudgetAmounts(
                tokens=int(tokens), wall_seconds=wall, trials=1, model_calls=int(calls),
            ))
            return True
        except (BudgetConflictError, ValueError, TypeError):
            return False

    def _remaining(self, snapshot: BudgetAmounts) -> BudgetAmounts:
        return BudgetAmounts(
            tokens=max(0, self.cap.tokens - snapshot.tokens),
            wall_seconds=max(0.0, self.cap.wall_seconds - snapshot.wall_seconds),
            trials=max(0, self.cap.trials - snapshot.trials),
            installs=max(0, self.cap.installs - snapshot.installs),
            download_bytes=max(0, self.cap.download_bytes - snapshot.download_bytes),
            model_calls=max(0, self.cap.model_calls - snapshot.model_calls),
            proposals=max(0, self.cap.proposals - snapshot.proposals),
        )

    def _resolve_issues(self, unresolved: set[str], validation: ValidationRecord | None) -> None:
        if validation is None or validation.qualification != "PASS":
            return
        passed = {outcome.check_id for outcome in validation.outcomes if outcome.verdict == "PASS"}
        for issue in self.issues:
            if issue.acceptance_check_ids and set(issue.acceptance_check_ids).issubset(passed):
                unresolved.discard(issue.issue_id)

    def _recover_validated_trials(self, event_id: str, unresolved: set[str]) -> str | None:
        """Rebuild event progress from identity-checked durable trial evidence."""
        if not hasattr(self.store, "trials_for_event"):
            return None
        try:
            archived_trial_ids = {
                trial_id for entry in self.archive.entries() for trial_id in entry.trial_ids
            }
            for trial in self.store.trials_for_event(event_id):
                if trial.status != "VALIDATED" or trial.contract_sha256 != canonical_contract_sha256(self.contract):
                    continue
                genome = self.store.load_genome(trial.genome_id)
                validation = self.store.load_current_validation(trial.trial_id)
                if (genome.content_sha256 != canonical_genome_sha256(genome)
                        or genome.task_family != self.contract.task_family
                        or not self._validation_identity(trial, validation, genome)):
                    return "BLOCKED"
                operation_id = _operation_id("trial", event_id, genome.content_sha256)
                reservation = self.ledger.reservation(operation_id)
                if reservation is None or not self._settle_trial(operation_id, trial, reservation):
                    return "BLOCKED"
                self._resolve_issues(unresolved, validation)
                if validation.qualification == "PASS" and trial.trial_id not in archived_trial_ids:
                    decision = self.archive.consider(
                        trial=trial, validation=validation, genome=genome, contract=self.contract,
                    )
                    if decision.action not in ("INSERT", "REPLACE", "KEEP"):
                        return "BLOCKED"
                if self._solved(validation, unresolved):
                    return "SOLVED"
        except Exception:
            return "BLOCKED"
        return None

    def _solved(self, validation: ValidationRecord, unresolved: set[str]) -> bool:
        return validation.qualification == "PASS" and not unresolved and self.is_solved(validation)

    def _cancelled(self) -> bool:
        return self.cancel_event is not None and self.cancel_event.is_set()

    def _result(self, status: str, unresolved: set[str], scope: str) -> EventOutcome:
        best = self.archive.best(contract=self.contract)
        snapshot = self.ledger.snapshot(scope=scope)
        return EventOutcome(
            status=status,
            best_trial_id=best.trial_ids[-1] if best is not None and best.trial_ids else None,
            unresolved_issues=tuple(sorted(unresolved)),
            cost={key: float(value) for key, value in snapshot.to_dict().items()},
        )


def _operation_id(kind: str, event_id: str, item: str) -> str:
    digest = hashlib.sha256(f"{kind}\0{event_id}\0{item}".encode("utf-8")).hexdigest()
    return f"qd_{event_id}_{kind}_{digest[:40]}"
