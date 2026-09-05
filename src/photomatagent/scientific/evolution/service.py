"""Transactional application service for the scientific evolution lifecycle."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar, cast

from photomatagent.runtime.events import (
    EvolutionComparisonCompleted,
    EvolutionEpisodeCompleted,
    EvolutionEpisodeStarted,
    EvolutionIterationStarted,
    EvolutionTaskAccepted,
    EvolutionTaskCreated,
    EvolutionTaskStopped,
    ExperienceStateChanged,
    ExpertFeedbackCompiled,
    ExpertFeedbackRecorded,
    RevisionPlanConfirmed,
    RuntimeEvent,
)
from photomatagent.errors import ToolExecutionError
from photomatagent.redaction import redact_secrets
from photomatagent.scientific.evolution.comparison import (
    compare_episodes,
    evaluate_machine_acceptance,
)
from photomatagent.scientific.evolution.events import bounded_summary
from photomatagent.scientific.evolution.experience import create_experience
from photomatagent.scientific.evolution.models import (
    AcceptanceResult,
    ArtifactRef,
    ComparisonReport,
    EpisodeRecord,
    EpisodeVersion,
    EvolutionResumeStatus,
    EvolutionStatus,
    EvolutionTask,
    ExecutionMode,
    ExpertFeedbackDraft,
    ExpertFeedbackRecord,
    FeedbackCompilation,
    RevisionPlan,
    Sha256,
    StrategyArm,
    StrategyVersion,
    TargetSnapshot,
    new_evolution_id,
    utc_now,
)
from photomatagent.scientific.evolution.rubric import assess_hard_caps
from photomatagent.scientific.evolution.store import (
    EvolutionStore,
    EvolutionTransaction,
)
from photomatagent.scientific.loop.target import TargetSpec
from photomatagent.scientific.state import ScientificState

EventSink = Callable[[RuntimeEvent], Awaitable[None] | None]
_EntityT = TypeVar("_EntityT", covariant=True)
_EPISODE_COMPLETION_OUTPUT_FIELDS = frozenset(
    {
        "status",
        "scientific_state_path",
        "completed_at",
        "summary",
        "artifact",
        "cost",
        "acceptance_results",
        "error",
    }
)
_EPISODE_COMPLETION_IDENTITY_FIELDS = tuple(
    field
    for field in EpisodeRecord.model_fields
    if field not in _EPISODE_COMPLETION_OUTPUT_FIELDS
)
_USER_CANCELLATION_ERROR = "Cancelled by user via evolution CLI."


@dataclass(frozen=True, slots=True)
class MutationResult(Generic[_EntityT]):
    """One mutation's persisted entity and caller-owned lifecycle events."""

    entity: _EntityT
    events: tuple[RuntimeEvent, ...] = ()


@dataclass(frozen=True, slots=True)
class IterationContext:
    """Exact immutable checkpoint required to reserve a revised episode."""

    task: EvolutionTask
    source_episode: EpisodeRecord
    revision: RevisionPlan
    strategy: StrategyVersion
    previous_scientific_state: ScientificState


@dataclass(frozen=True, slots=True)
class IterationClaim:
    """One atomically verified and owner-bound revised episode claim."""

    context: IterationContext
    episode: EpisodeRecord
    owner_token: str
    events: tuple[RuntimeEvent, ...] = ()


ALLOWED_TRANSITIONS: dict[EvolutionStatus, frozenset[EvolutionStatus]] = {
    "CREATED": frozenset({"RUNNING", "STOPPED"}),
    "RUNNING": frozenset(
        {"AWAITING_EXPERT_FEEDBACK", "BLOCKED", "BUDGET_EXHAUSTED"}
    ),
    "AWAITING_EXPERT_FEEDBACK": frozenset(
        {"FEEDBACK_RECORDED", "ACCEPTED", "STOPPED"}
    ),
    "FEEDBACK_RECORDED": frozenset({"REVISION_READY", "STOPPED"}),
    "REVISION_READY": frozenset({"RUNNING", "STOPPED"}),
    "ACCEPTED": frozenset({"AWAITING_EXPERT_FEEDBACK"}),
    "STOPPED": frozenset({"AWAITING_EXPERT_FEEDBACK"}),
    "BUDGET_EXHAUSTED": frozenset({"AWAITING_EXPERT_FEEDBACK", "STOPPED"}),
    "BLOCKED": frozenset({"AWAITING_EXPERT_FEEDBACK", "STOPPED"}),
}


class EvolutionServiceError(RuntimeError):
    """Base error for lifecycle orchestration failures."""


class InvalidEvolutionTransition(EvolutionServiceError):
    """Raised when an operation is not legal in authoritative persisted state."""


class EvolutionOperationConflict(EvolutionServiceError):
    """Raised when a stable operation identity names different durable content."""


class ArtifactMismatchError(EvolutionServiceError):
    """Raised when a declared result artifact is absent or does not match bytes."""


class EvolutionService:
    """Coordinate durable lifecycle records without running models or tools."""

    def __init__(
        self,
        store: EvolutionStore,
        *,
        event_sink: EventSink | None = None,
    ) -> None:
        self.store = store
        self.event_sink = event_sink

    def get(self, evolution_id: str) -> EvolutionTask:
        return self.store.load_task(evolution_id)

    def compare(
        self,
        evolution_id: str,
        previous_version: EpisodeVersion,
        current_version: EpisodeVersion,
    ) -> MutationResult[ComparisonReport]:
        """Build and durably link one canonical adjacent-episode comparison.

        Every input is reloaded under the task lock.  Primary artifacts are
        hashed again, and declared scientific-state paths must name the
        canonical immutable snapshots.  The operation writes immutable records
        before atomically linking them into the task manifest, so a retry can
        reconcile a crash at either boundary without creating a second
        observation.
        """

        with self.store.transaction(evolution_id) as transaction:
            task = transaction.load_task()
            previous = transaction.load_episode(previous_version)
            current = transaction.load_episode(current_version)
            self._validate_comparison_episodes(task, previous, current)
            self._verify_artifact(previous.artifact)
            self._verify_artifact(current.artifact)
            previous_state = self._comparison_state(previous)
            current_state = self._comparison_state(current)
            if current.revision_plan_id is None:
                raise EvolutionOperationConflict(
                    "revised episode does not name its applied revision plan"
                )
            if current.revision_plan_id not in task.revision_ids:
                raise EvolutionOperationConflict(
                    "applied revision plan is not linked by the task manifest"
                )
            plan = transaction.load_revision(current.revision_plan_id)
            if current.applied_feedback_id != plan.feedback_id:
                raise EvolutionOperationConflict(
                    "revised episode feedback does not match its revision plan"
                )
            feedback_records = self.store.list_feedback(evolution_id)
            previous_feedback = self._active_feedback(
                feedback_records,
                previous.version,
            )
            current_feedback = self._active_feedback(
                feedback_records,
                current.version,
            )
            if previous_feedback is None:
                raise EvolutionOperationConflict(
                    "adjacent comparison requires the feedback that produced "
                    "the revision"
                )
            if previous_feedback.feedback_id not in task.feedback_ids:
                raise EvolutionOperationConflict(
                    "previous feedback is not linked by the task manifest"
                )
            self._require_feedback_artifact(previous_feedback, previous)
            if current_feedback is not None:
                if current_feedback.feedback_id not in task.feedback_ids:
                    raise EvolutionOperationConflict(
                        "current feedback is not linked by the task manifest"
                    )
                self._require_feedback_artifact(current_feedback, current)

            previous_compilation = self._comparison_compilation(
                task,
                previous_feedback,
            )
            if previous_compilation is None:
                raise EvolutionOperationConflict(
                    "comparison source feedback has no canonical available compilation"
                )
            from photomatagent.scientific.evolution.revision import (
                build_revision_plan,
            )

            canonical_plan = build_revision_plan(
                feedback=previous_feedback,
                compilation=previous_compilation,
                target=previous.target_snapshot,
                previous_summary=previous.summary,
            ).model_copy(
                update={
                    "confirmed": True,
                    "confirmed_at": plan.confirmed_at,
                }
            )
            if plan != canonical_plan:
                raise EvolutionOperationConflict(
                    "comparison revision plan does not match canonical persisted inputs"
                )
            if current.strategy_id is None or current.strategy_id not in task.strategy_ids:
                raise EvolutionOperationConflict(
                    "revised episode strategy is not linked by the task manifest"
                )
            from photomatagent.scientific.evolution.strategy import FixedStrategySelector

            strategy = transaction.load_strategy(current.strategy_id)
            canonical_strategy = FixedStrategySelector().select(task, plan)
            if (
                strategy != canonical_strategy
                or current.strategy_id != strategy.strategy_id
                or current.strategy_arm != strategy.arm
                or strategy.parameters.get("revision_id") != plan.revision_id
            ):
                raise EvolutionOperationConflict(
                    "revised episode strategy does not match its canonical plan chain"
                )
            canonical_acceptance = evaluate_machine_acceptance(
                plan=plan,
                episode=current,
                state=current_state,
            )
            if list(current.acceptance_results) != canonical_acceptance:
                raise EvolutionOperationConflict(
                    "revised episode acceptance results do not match the "
                    "deterministic evaluator"
                )
            current_compilation = (
                self._comparison_compilation(task, current_feedback)
                if current_feedback is not None
                else None
            )
            if current_feedback is not None and current_compilation is None:
                raise InvalidEvolutionTransition(
                    "current feedback is recorded but has no exact AVAILABLE "
                    "compilation; compile the current feedback before requesting "
                    "a POST_FEEDBACK comparison"
                )
            machine_results: dict[str, bool | str | AcceptanceResult] = {}
            for result in current.acceptance_results:
                machine_results[result.acceptance_id] = result
                if result.detail:
                    machine_results[result.detail] = result
            report = compare_episodes(
                previous=previous,
                current=current,
                previous_plan=plan,
                previous_feedback=previous_feedback,
                current_feedback=current_feedback,
                current_compilation=current_compilation,
                previous_items=previous_compilation.items,
                machine_results=machine_results,
                previous_state=previous_state,
                current_state=current_state,
            )
            try:
                persisted_report = transaction.load_comparison(report.comparison_id)
            except FileNotFoundError:
                transaction.write_comparison(report)
                persisted_report = transaction.load_comparison(report.comparison_id)
            else:
                if persisted_report != report:
                    raise EvolutionOperationConflict(
                        "comparison ID names different canonical content"
                    )
            persisted_experience = None
            if report.phase == "POST_FEEDBACK":
                unsafe = self._unsafe_experience_observation(
                    previous_feedback,
                    current_feedback,
                    previous_compilation,
                    current_compilation,
                )
                experience = create_experience(
                    report,
                    task_group_id=task.task_group_id,
                    safety_or_fabrication_failure=unsafe,
                )
                try:
                    persisted_experience = transaction.load_experience(
                        experience.experience_id
                    )
                except FileNotFoundError:
                    transaction.write_experience(experience)
                    persisted_experience = transaction.load_experience(
                        experience.experience_id
                    )
                else:
                    if persisted_experience != experience:
                        raise EvolutionOperationConflict(
                            "experience ID names different canonical content"
                        )

            comparison_ids = self._append_once(
                task.comparison_ids,
                persisted_report.comparison_id,
            )
            experience_ids = (
                self._append_once(task.experience_ids, persisted_experience.experience_id)
                if persisted_experience is not None
                else task.experience_ids
            )
            if (
                comparison_ids != task.comparison_ids
                or experience_ids != task.experience_ids
            ):
                transaction.save_task(
                    task.model_copy(
                        update={
                            "comparison_ids": comparison_ids,
                            "experience_ids": experience_ids,
                        }
                    ),
                    expected_revision=task.revision,
                )

        events: tuple[RuntimeEvent, ...] = (
            EvolutionComparisonCompleted(
                evolution_id=evolution_id,
                episode_version=current.version,
            ),
        )
        if persisted_experience is not None:
            events = (*events, ExperienceStateChanged(evolution_id=evolution_id))
        return MutationResult(
            persisted_report,
            events,
        )

    def reconcile(self, evolution_id: str) -> MutationResult[EvolutionTask]:
        """Repair a task manifest left behind a terminal episode write.

        Episode transitions and task-manifest updates are separate atomic file
        replacements.  If a process dies between them, the terminal episode is
        authoritative, but the task can remain ``RUNNING``.  Reconciliation is
        deliberately narrow: it only repairs that exact split state while the
        task transaction lock is held.
        """

        with self.store.transaction(evolution_id) as transaction:
            task = transaction.load_task()
            if task.status != "RUNNING":
                return MutationResult(task)
            if task.current_version is None:
                raise InvalidEvolutionTransition(
                    "RUNNING task has no current episode to reconcile"
                )
            episode = transaction.load_episode(task.current_version)
            expected_episode_id = self._episode_id(
                evolution_id,
                task.current_version,
            )
            if (
                episode.episode_id != expected_episode_id
                or not task.episode_ids
                or task.episode_ids[-1] != episode.episode_id
            ):
                raise EvolutionOperationConflict(
                    "current episode identity does not match the task manifest"
                )
            if episode.status == "COMPLETED":
                self.validate_transition(task.status, "AWAITING_EXPERT_FEEDBACK")
                saved = transaction.save_task(
                    task.model_copy(
                        update={
                            "status": "AWAITING_EXPERT_FEEDBACK",
                            "resume_status": None,
                            "last_completed_version": episode.version,
                        }
                    ),
                    expected_revision=task.revision,
                )
                return MutationResult(
                    saved,
                    (
                        EvolutionEpisodeCompleted(
                            evolution_id=evolution_id,
                            episode_version=episode.version,
                        ),
                    ),
                )
            if episode.status == "FAILED":
                self.validate_transition(task.status, "BLOCKED")
                checkpoint: EvolutionResumeStatus = (
                    "CREATED"
                    if task.last_completed_version is None
                    else "REVISION_READY"
                )
                saved = transaction.save_task(
                    task.model_copy(
                        update={
                            "status": "BLOCKED",
                            "resume_status": checkpoint,
                        }
                    ),
                    expected_revision=task.revision,
                )
                return MutationResult(saved)
            return MutationResult(task)

    def create_task(
        self,
        *,
        goal: str,
        target: TargetSpec,
        task_group_id: str | None = None,
        input_sha256: str | None = None,
        evolution_id: str | None = None,
    ) -> MutationResult[EvolutionTask]:
        resolved_id = evolution_id or new_evolution_id()
        task = EvolutionTask(
            evolution_id=resolved_id,
            goal=goal,
            target=target,
            task_group_id=task_group_id or resolved_id,
            input_sha256=input_sha256 or self._input_hash(goal, target),
        )
        created = self.store.create_task(task)
        event = EvolutionTaskCreated(
            evolution_id=created.evolution_id,
            goal_summary=bounded_summary(created.goal),
        )
        return MutationResult(created, (event,))

    def reserve_episode(
        self,
        evolution_id: str,
        *,
        mode: ExecutionMode,
        provider: str | None = None,
        model: str | None = None,
        tool_surface_fingerprint: Sha256 | None = None,
        capability_fingerprint: Sha256 | None = None,
        data_source_fingerprints: dict[str, Sha256] | None = None,
        owner_token: str | None = None,
    ) -> MutationResult[EpisodeRecord]:
        with self.store.transaction(evolution_id) as transaction:
            task = transaction.load_task()
            return self._reserve_episode_locked(
                transaction,
                task,
                mode=mode,
                provider=provider,
                model=model,
                tool_surface_fingerprint=tool_surface_fingerprint,
                capability_fingerprint=capability_fingerprint,
                data_source_fingerprints=data_source_fingerprints or {},
                owner_token=owner_token,
            )

    def _reserve_episode_locked(
        self,
        transaction: EvolutionTransaction,
        task: EvolutionTask,
        *,
        mode: ExecutionMode,
        provider: str | None,
        model: str | None,
        tool_surface_fingerprint: Sha256 | None,
        capability_fingerprint: Sha256 | None,
        data_source_fingerprints: dict[str, Sha256],
        owner_token: str | None,
    ) -> MutationResult[EpisodeRecord]:
        evolution_id = task.evolution_id
        initial = task.last_completed_version is None
        required_status: EvolutionStatus = "CREATED" if initial else "REVISION_READY"
        strategy_id, strategy_arm = self._reservation_strategy(
            transaction,
            task,
        )
        if task.status == "RUNNING" and task.current_version is not None:
            existing = transaction.load_episode(task.current_version)
            self._validate_reservation(
                existing,
                task,
                mode,
                provider,
                model,
                tool_surface_fingerprint,
                capability_fingerprint,
                data_source_fingerprints,
                strategy_id,
                strategy_arm,
                owner_token,
            )
            return MutationResult(existing, self._reservation_events(existing, initial))
        if task.status != required_status:
            raise InvalidEvolutionTransition(
                f"cannot reserve an episode while task is {task.status}; "
                f"required {required_status}"
            )
        if initial and mode != "NORMAL":
            raise InvalidEvolutionTransition("an initial or retry episode must use NORMAL")
        if not initial and mode == "NORMAL":
            raise InvalidEvolutionTransition(
                "a revised episode requires an explicit evidence/evaluation mode"
            )
        self.validate_transition(task.status, "RUNNING")
        version = self._next_version(task.current_version)
        episode = EpisodeRecord(
            evolution_id=evolution_id,
            episode_id=self._episode_id(evolution_id, version),
            version=version,
            parent_version=task.last_completed_version,
            applied_feedback_id=(task.feedback_ids[-1] if task.feedback_ids else None),
            revision_plan_id=(task.revision_ids[-1] if task.revision_ids else None),
            owner_token=owner_token,
            execution_mode=mode,
            strategy_id=strategy_id,
            strategy_arm=strategy_arm,
            task_snapshot=task.model_dump(mode="json"),
            target_snapshot=TargetSnapshot.model_validate(task.target),
            provider=provider,
            model=model,
            tool_surface_fingerprint=tool_surface_fingerprint,
            capability_fingerprint=capability_fingerprint,
            data_source_fingerprints=data_source_fingerprints,
        )
        try:
            existing = transaction.load_episode(version)
        except FileNotFoundError:
            transaction.write_episode(episode)
            persisted = episode
        else:
            self._validate_reservation(
                existing,
                task,
                mode,
                provider,
                model,
                tool_surface_fingerprint,
                capability_fingerprint,
                data_source_fingerprints,
                strategy_id,
                strategy_arm,
                owner_token,
            )
            persisted = existing
        updated = task.model_copy(
            update={
                "status": "RUNNING",
                "resume_status": None,
                "current_version": version,
                "episode_ids": self._append_once(
                    task.episode_ids, persisted.episode_id
                ),
            }
        )
        transaction.save_task(updated, expected_revision=task.revision)
        return MutationResult(
            persisted,
            self._reservation_events(persisted, initial),
        )

    def mark_episode_running(
        self,
        evolution_id: str,
        version: EpisodeVersion,
        *,
        owner_token: str | None = None,
        runtime_session_id: str | None = None,
        event_log_path: str | None = None,
    ) -> MutationResult[EpisodeRecord]:
        with self.store.transaction(evolution_id) as transaction:
            task, episode = self._current_episode(transaction, version)
            self._require_episode_owner(episode, owner_token)
            if task.status != "RUNNING":
                raise InvalidEvolutionTransition("task must be RUNNING")
            if episode.status == "RUNNING":
                if (
                    episode.runtime_session_id != runtime_session_id
                    or episode.event_log_path != event_log_path
                ):
                    raise EvolutionOperationConflict(
                        "running episode provenance differs from retry"
                    )
                saved = episode
            elif episode.status == "RESERVED":
                running = episode.model_copy(
                    update={
                        "status": "RUNNING",
                        "runtime_session_id": runtime_session_id,
                        "event_log_path": event_log_path,
                        "started_at": utc_now(),
                    }
                )
                saved = transaction.transition_episode(
                    running,
                    expected_status="RESERVED",
                )
            else:
                raise InvalidEvolutionTransition("only a reserved episode can start")
        event = EvolutionEpisodeStarted(
            evolution_id=evolution_id,
            episode_version=version,
        )
        return MutationResult(saved, (event,))

    def complete_episode(
        self,
        evolution_id: str,
        version: EpisodeVersion,
        *,
        result: EpisodeRecord,
        owner_token: str | None = None,
    ) -> MutationResult[EpisodeRecord]:
        with self.store.transaction(evolution_id) as transaction:
            task, episode = self._current_episode(transaction, version)
            self._require_episode_owner(episode, owner_token)
            artifact = self._verify_artifact(result.artifact)
            completed = self._validated_completion(episode, result, artifact)
            if episode.status == "COMPLETED":
                saved_episode = episode
            elif episode.status == "RUNNING":
                if task.status != "RUNNING":
                    raise InvalidEvolutionTransition("task must be RUNNING")
                saved_episode = transaction.transition_episode(
                    completed,
                    expected_status="RUNNING",
                )
            else:
                raise InvalidEvolutionTransition("only a running episode can complete")
            if task.status == "RUNNING":
                self.validate_transition(task.status, "AWAITING_EXPERT_FEEDBACK")
                updated = task.model_copy(
                    update={
                        "status": "AWAITING_EXPERT_FEEDBACK",
                        "resume_status": None,
                        "last_completed_version": version,
                    }
                )
                transaction.save_task(updated, expected_revision=task.revision)
            elif not (
                task.status == "AWAITING_EXPERT_FEEDBACK"
                and task.last_completed_version == version
            ):
                raise InvalidEvolutionTransition(
                    "completed episode cannot reconcile with task state"
                )
        event = EvolutionEpisodeCompleted(
            evolution_id=evolution_id,
            episode_version=version,
        )
        return MutationResult(saved_episode, (event,))

    def fail_episode(
        self,
        evolution_id: str,
        version: EpisodeVersion,
        error: str,
        *,
        owner_token: str | None = None,
    ) -> MutationResult[EpisodeRecord]:
        with self.store.transaction(evolution_id) as transaction:
            task, episode = self._current_episode(transaction, version)
            self._require_episode_owner(episode, owner_token)
            if episode.status == "FAILED":
                if episode.error != error:
                    raise EvolutionOperationConflict("failed episode error differs from retry")
                saved_episode = episode
            elif episode.status in {"RESERVED", "RUNNING"} and task.status == "RUNNING":
                failed = episode.model_copy(
                    update={
                        "status": "FAILED",
                        "completed_at": utc_now(),
                        "error": error,
                    }
                )
                saved_episode = transaction.transition_episode(
                    failed,
                    expected_status=episode.status,
                )
            else:
                raise InvalidEvolutionTransition("only the active episode may fail")
            if task.status == "RUNNING":
                self.validate_transition(task.status, "BLOCKED")
                checkpoint: EvolutionResumeStatus = (
                    "CREATED" if task.last_completed_version is None else "REVISION_READY"
                )
                updated = task.model_copy(
                    update={"status": "BLOCKED", "resume_status": checkpoint}
                )
                transaction.save_task(updated, expected_revision=task.revision)
            elif task.status != "BLOCKED":
                raise InvalidEvolutionTransition(
                    "failed episode cannot reconcile with task state"
                )
        return MutationResult(saved_episode)

    def cancel(
        self,
        evolution_id: str,
        *,
        owner_token: str | None = None,
    ) -> MutationResult[EvolutionTask]:
        """Safely reconcile or cancel the task's active episode.

        The episode transition precedes the task-manifest update, matching the
        existing crash-recovery protocol. Repeating cancellation after either
        terminal state is therefore a harmless read of the reconciled task.
        """

        events: tuple[RuntimeEvent, ...] = ()
        with self.store.transaction(evolution_id) as transaction:
            task = transaction.load_task()
            if task.current_version is None:
                raise InvalidEvolutionTransition(
                    "cancellation requires a current episode"
                )
            episode = transaction.load_episode(task.current_version)
            self._require_episode_owner(episode, owner_token)
            expected_episode_id = self._episode_id(
                evolution_id,
                task.current_version,
            )
            if (
                episode.episode_id != expected_episode_id
                or not task.episode_ids
                or task.episode_ids[-1] != episode.episode_id
            ):
                raise EvolutionOperationConflict(
                    "current episode identity does not match the task manifest"
                )

            if task.status != "RUNNING":
                if (
                    (task.status == "BLOCKED" and episode.status == "FAILED")
                    or (
                        task.status == "AWAITING_EXPERT_FEEDBACK"
                        and episode.status == "COMPLETED"
                    )
                ):
                    return MutationResult(task)
                raise InvalidEvolutionTransition(
                    f"cancel requires a RUNNING task; task is {task.status}"
                )

            if episode.status == "COMPLETED":
                self.validate_transition(task.status, "AWAITING_EXPERT_FEEDBACK")
                updated = task.model_copy(
                    update={
                        "status": "AWAITING_EXPERT_FEEDBACK",
                        "resume_status": None,
                        "last_completed_version": episode.version,
                    }
                )
                events = (
                    EvolutionEpisodeCompleted(
                        evolution_id=evolution_id,
                        episode_version=episode.version,
                    ),
                )
            else:
                if episode.status in {"RESERVED", "RUNNING"}:
                    failed = episode.model_copy(
                        update={
                            "status": "FAILED",
                            "completed_at": utc_now(),
                            "error": _USER_CANCELLATION_ERROR,
                        }
                    )
                    transaction.transition_episode(
                        failed,
                        expected_status=episode.status,
                    )
                elif episode.status != "FAILED":
                    raise InvalidEvolutionTransition(
                        f"cannot cancel episode in {episode.status} state"
                    )
                self.validate_transition(task.status, "BLOCKED")
                checkpoint: EvolutionResumeStatus = (
                    "CREATED"
                    if task.last_completed_version is None
                    else "REVISION_READY"
                )
                updated = task.model_copy(
                    update={"status": "BLOCKED", "resume_status": checkpoint}
                )

            saved = transaction.save_task(
                updated,
                expected_revision=task.revision,
            )
        return MutationResult(saved, events)

    def attach_feedback(
        self,
        evolution_id: str,
        version: EpisodeVersion,
        *,
        feedback_id: str,
        draft: ExpertFeedbackDraft,
        result_sha256: Sha256,
        raw_input: str | None = None,
        hard_cap_override_reason: str | None = None,
    ) -> MutationResult[ExpertFeedbackRecord]:
        with self.store.transaction(evolution_id) as transaction:
            task, episode = self._current_episode(transaction, version)
            if task.status not in {
                "AWAITING_EXPERT_FEEDBACK",
                "FEEDBACK_RECORDED",
            }:
                raise InvalidEvolutionTransition(
                    "feedback requires AWAITING_EXPERT_FEEDBACK"
                )
            if episode.status != "COMPLETED":
                raise InvalidEvolutionTransition("feedback requires a completed episode")
            artifact = self._verify_artifact(episode.artifact)
            if artifact.sha256 != result_sha256:
                raise ArtifactMismatchError(
                    "feedback hash does not match the persisted primary artifact"
                )
            feedback = self._build_feedback(
                feedback_id=feedback_id,
                evolution_id=evolution_id,
                version=version,
                draft=draft,
                result_sha256=result_sha256,
                raw_input=raw_input,
                hard_cap_override_reason=hard_cap_override_reason,
            )
            all_feedback = self.store.list_feedback(evolution_id)
            active = self._active_feedback(all_feedback, version)
            if active is not None and active.feedback_id != feedback_id:
                raise EvolutionOperationConflict(
                    f"episode {version} already has active feedback {active.feedback_id}"
                )
            existing = next(
                (item for item in all_feedback if item.feedback_id == feedback_id),
                None,
            )
            if existing is not None:
                self._require_matching_feedback(existing, feedback)
                persisted = existing
            else:
                if task.status == "FEEDBACK_RECORDED":
                    raise InvalidEvolutionTransition(
                        "recorded feedback retry requires its existing immutable record"
                    )
                transaction.write_feedback(feedback)
                persisted = transaction.load_feedback(feedback.feedback_id)
            if task.status == "AWAITING_EXPERT_FEEDBACK":
                self.validate_transition(task.status, "FEEDBACK_RECORDED")
                updated = task.model_copy(
                    update={
                        "status": "FEEDBACK_RECORDED",
                        "feedback_ids": self._append_once(
                            task.feedback_ids, persisted.feedback_id
                        ),
                    }
                )
                transaction.save_task(updated, expected_revision=task.revision)
            elif not (
                task.status == "FEEDBACK_RECORDED"
                and persisted.feedback_id in task.feedback_ids
            ):
                raise InvalidEvolutionTransition(
                    "feedback cannot reconcile with current task state"
                )
        event = ExpertFeedbackRecorded(
            evolution_id=evolution_id,
            episode_version=version,
            feedback_id=persisted.feedback_id,
            result_sha256=result_sha256,
            scores={
                "scientific_correctness": persisted.scores.scientific_correctness,
                "evidence_sufficiency": persisted.scores.evidence_sufficiency,
                "novelty": persisted.scores.novelty,
                "actionability": persisted.scores.actionability,
                "overall": persisted.scores.overall,
            },
        )
        return MutationResult(persisted, (event,))

    def compilation_context(
        self,
        evolution_id: str,
        version: EpisodeVersion | None = None,
    ) -> tuple[EvolutionTask, EpisodeRecord, ExpertFeedbackRecord]:
        """Resolve and verify the one active review and its exact result."""

        with self.store.transaction(evolution_id) as transaction:
            task = transaction.load_task()
            if task.status != "FEEDBACK_RECORDED":
                raise InvalidEvolutionTransition(
                    "feedback compilation requires FEEDBACK_RECORDED"
                )
            selected = version or task.last_completed_version
            if selected is None or selected != task.last_completed_version:
                raise InvalidEvolutionTransition(
                    "compilation version must be the latest completed episode"
                )
            episode = transaction.load_episode(selected)
            if episode.status != "COMPLETED":
                raise InvalidEvolutionTransition(
                    "feedback compilation requires a completed episode"
                )
            artifact = self._verify_artifact(episode.artifact)
            feedback = self._active_feedback(
                self.store.list_feedback(evolution_id),
                selected,
            )
            if (
                feedback is None
                or not task.feedback_ids
                or task.feedback_ids[-1] != feedback.feedback_id
            ):
                raise EvolutionOperationConflict(
                    "task manifest does not name the active feedback"
                )
            if feedback.result_sha256 != artifact.sha256:
                raise ArtifactMismatchError(
                    "active feedback hash does not match the persisted result"
                )
            return task, episode, feedback

    def available_compilation(
        self,
        evolution_id: str,
        feedback_id: str,
    ) -> FeedbackCompilation | None:
        """Return a durable successful compilation for this feedback, if any."""

        task = self.store.load_task(evolution_id)
        by_id = {
            item.compilation_id: item
            for item in self.store.list_compilations(evolution_id)
        }
        for compilation_id in task.compilation_ids:
            item = by_id.get(compilation_id)
            if item is not None and item.feedback_id == feedback_id and item.status == "AVAILABLE":
                return item
        # Recover an immutable write that completed immediately before a task
        # manifest update was interrupted.
        return next(
            (
                item
                for item in by_id.values()
                if item.feedback_id == feedback_id and item.status == "AVAILABLE"
            ),
            None,
        )

    def save_compilation(
        self,
        evolution_id: str,
        compilation: FeedbackCompilation,
    ) -> MutationResult[FeedbackCompilation]:
        """Persist and link one compilation without rewriting source feedback."""

        if compilation.status == "PENDING" or compilation.compilation_id is None:
            raise InvalidEvolutionTransition(
                "only a completed compilation attempt can be persisted"
            )
        with self.store.transaction(evolution_id) as transaction:
            task = transaction.load_task()
            if task.status != "FEEDBACK_RECORDED":
                raise InvalidEvolutionTransition(
                    "feedback compilation requires FEEDBACK_RECORDED"
                )
            if task.last_completed_version is None:
                raise InvalidEvolutionTransition(
                    "feedback compilation requires a completed episode"
                )
            episode = transaction.load_episode(task.last_completed_version)
            artifact = self._verify_artifact(episode.artifact)
            feedback = self._active_feedback(
                self.store.list_feedback(evolution_id),
                task.last_completed_version,
            )
            if (
                feedback is None
                or not task.feedback_ids
                or task.feedback_ids[-1] != feedback.feedback_id
                or compilation.evolution_id != evolution_id
                or compilation.feedback_id != feedback.feedback_id
                or compilation.episode_version != episode.version
            ):
                raise EvolutionOperationConflict(
                    "compilation does not match the active feedback"
                )
            if feedback.result_sha256 != artifact.sha256:
                raise ArtifactMismatchError(
                    "active feedback hash does not match the persisted result"
                )

            successful = next(
                (
                    item
                    for item in self.store.list_compilations(evolution_id)
                    if item.feedback_id == feedback.feedback_id
                    and item.status == "AVAILABLE"
                    and item.compilation_id != compilation.compilation_id
                ),
                None,
            )
            if successful is not None:
                persisted = successful
            else:
                try:
                    existing = transaction.load_compilation(
                        compilation.compilation_id
                    )
                except FileNotFoundError:
                    transaction.write_compilation(compilation)
                    persisted = transaction.load_compilation(
                        compilation.compilation_id
                    )
                else:
                    if existing != compilation:
                        raise EvolutionOperationConflict(
                            f"compilation ID {compilation.compilation_id} "
                            "names different content"
                        )
                    persisted = existing

            persisted_id = persisted.compilation_id
            if persisted_id is None or persisted.episode_version is None:
                raise EvolutionOperationConflict(
                    "persisted compilation is missing authoritative identity"
                )
            if persisted_id not in task.compilation_ids:
                updated = task.model_copy(
                    update={
                        "compilation_ids": self._append_once(
                            task.compilation_ids,
                            persisted_id,
                        )
                    }
                )
                transaction.save_task(updated, expected_revision=task.revision)
        event = ExpertFeedbackCompiled(
            evolution_id=evolution_id,
            episode_version=persisted.episode_version,
        )
        return MutationResult(persisted, (event,))

    def confirm_revision(
        self,
        evolution_id: str,
        plan: RevisionPlan,
        *,
        strategy: StrategyVersion | None = None,
    ) -> MutationResult[RevisionPlan]:
        with self.store.transaction(evolution_id) as transaction:
            task = transaction.load_task()
            self._validate_revision(task, evolution_id, plan)
            episode, feedback, compilation = self._revision_context(
                transaction,
                task,
                evolution_id,
            )
            from photomatagent.scientific.evolution.revision import (
                build_revision_plan,
            )
            from photomatagent.scientific.evolution.strategy import (
                FixedStrategySelector,
            )

            canonical_plan = build_revision_plan(
                feedback=feedback,
                compilation=compilation,
                target=episode.target_snapshot,
                previous_summary=episode.summary,
            )
            self._require_canonical_plan(plan, canonical_plan)
            if canonical_plan.has_blocking_ambiguity:
                raise InvalidEvolutionTransition(
                    "revision plan has an unresolved blocking ambiguity"
                )
            canonical_strategy = FixedStrategySelector().select(task, canonical_plan)
            if strategy is not None and strategy != canonical_strategy:
                raise EvolutionOperationConflict(
                    "submitted strategy does not match canonical fixed strategy"
                )
            strategy = canonical_strategy
            confirmed = canonical_plan.model_copy(
                update={"confirmed": True, "confirmed_at": utc_now()}
            )
            competing = next(
                (
                    item
                    for item in self.store.list_revisions(evolution_id)
                    if item.source_version == plan.source_version
                    and item.feedback_id == plan.feedback_id
                    and item.revision_id != plan.revision_id
                ),
                None,
            )
            if competing is not None:
                raise EvolutionOperationConflict(
                    "active feedback already has a different durable revision "
                    f"{competing.revision_id}"
                )
            competing_strategy = next(
                (
                    item
                    for item in self.store.list_strategies(evolution_id)
                    if item.parameters.get("revision_id") == plan.revision_id
                    and item.strategy_id != strategy.strategy_id
                ),
                None,
            )
            if competing_strategy is not None:
                raise EvolutionOperationConflict(
                    "revision already has a different durable strategy "
                    f"{competing_strategy.strategy_id}"
                )

            try:
                existing_revision = transaction.load_revision(plan.revision_id)
            except FileNotFoundError:
                existing_revision = None
            else:
                assert existing_revision is not None
                self._require_matching_revision(existing_revision, confirmed)
            try:
                existing_strategy = transaction.load_strategy(strategy.strategy_id)
            except FileNotFoundError:
                existing_strategy = None
            else:
                assert existing_strategy is not None
                self._require_matching_strategy(existing_strategy, strategy)

            if existing_revision is None:
                transaction.write_revision(confirmed)
                persisted = confirmed
            else:
                persisted = existing_revision
            if existing_strategy is None:
                transaction.write_strategy(strategy)
                persisted_strategy = strategy
            else:
                persisted_strategy = existing_strategy
            if task.status == "FEEDBACK_RECORDED":
                self.validate_transition(task.status, "REVISION_READY")
                updated = task.model_copy(
                    update={
                        "status": "REVISION_READY",
                        "revision_ids": self._append_once(
                            task.revision_ids, persisted.revision_id
                        ),
                        "strategy_ids": self._append_once(
                            task.strategy_ids, persisted_strategy.strategy_id
                        ),
                    }
                )
                transaction.save_task(updated, expected_revision=task.revision)
            elif not (
                task.status == "REVISION_READY"
                and persisted.revision_id in task.revision_ids
                and persisted_strategy.strategy_id in task.strategy_ids
            ):
                raise InvalidEvolutionTransition(
                    "revision cannot reconcile with current task state"
                )
        event = RevisionPlanConfirmed(
            evolution_id=evolution_id,
            episode_version=persisted.source_version,
        )
        return MutationResult(persisted, (event,))

    def iteration_context(self, evolution_id: str) -> IterationContext:
        """Load and validate the exact confirmed state used by ``iterate``."""

        with self.store.transaction(evolution_id) as transaction:
            task = transaction.load_task()
            return self._iteration_context_locked(transaction, task)

    def claim_iteration(
        self,
        evolution_id: str,
        *,
        owner_token: str,
        mode: ExecutionMode,
        provider: str | None = None,
        model: str | None = None,
        tool_surface_fingerprint: Sha256 | None = None,
        capability_fingerprint: Sha256 | None = None,
        data_source_fingerprints: dict[str, Sha256] | None = None,
    ) -> IterationClaim:
        """Atomically verify the exact revision checkpoint and claim its episode."""

        with self.store.transaction(evolution_id) as transaction:
            task = transaction.load_task()
            checkpoint_task = task
            if task.status == "RUNNING" and task.current_version is not None:
                existing = transaction.load_episode(task.current_version)
                if existing.owner_token != owner_token:
                    raise EvolutionOperationConflict(
                        "active episode is owned by another iteration invocation"
                    )
                checkpoint_task = EvolutionTask.model_validate(existing.task_snapshot)
            context = self._iteration_context_locked(transaction, checkpoint_task)
            reserved = self._reserve_episode_locked(
                transaction,
                task,
                mode=mode,
                provider=provider,
                model=model,
                tool_surface_fingerprint=tool_surface_fingerprint,
                capability_fingerprint=capability_fingerprint,
                data_source_fingerprints=data_source_fingerprints or {},
                owner_token=owner_token,
            )
            return IterationClaim(
                context=context,
                episode=reserved.entity,
                owner_token=owner_token,
                events=reserved.events,
            )

    def _iteration_context_locked(
        self,
        transaction: EvolutionTransaction,
        task: EvolutionTask,
    ) -> IterationContext:
        evolution_id = task.evolution_id
        if task.status != "REVISION_READY":
            raise InvalidEvolutionTransition("iterate requires REVISION_READY")
        if (
            task.last_completed_version is None
            or not task.feedback_ids
            or not task.revision_ids
            or not task.strategy_ids
        ):
            raise EvolutionOperationConflict(
                "REVISION_READY task is missing its exact iteration checkpoint"
            )
        source, feedback, compilation = self._revision_context(
            transaction, task, evolution_id
        )
        from photomatagent.scientific.evolution.revision import build_revision_plan
        from photomatagent.scientific.evolution.strategy import FixedStrategySelector

        revision = transaction.load_revision(task.revision_ids[-1])
        canonical_plan = build_revision_plan(
            feedback=feedback,
            compilation=compilation,
            target=task.target,
            previous_summary=source.summary,
        ).model_copy(
            update={
                "confirmed": True,
                "confirmed_at": revision.confirmed_at,
            }
        )
        if revision != canonical_plan:
            raise EvolutionOperationConflict(
                "persisted revision plan does not match canonical persisted inputs"
            )
        canonical_strategy = FixedStrategySelector().select(task, canonical_plan)
        strategy = transaction.load_strategy(task.strategy_ids[-1])
        if strategy != canonical_strategy:
            raise EvolutionOperationConflict(
                "persisted strategy does not match canonical persisted inputs"
            )
        if revision.has_blocking_ambiguity:
            raise EvolutionOperationConflict(
                "active revision has a blocking ambiguity"
            )
        if source.scientific_state_path is None:
            raise InvalidEvolutionTransition(
                "source episode has no persisted scientific-state snapshot"
            )
        state_path = self.store.workspace.resolve(
            source.scientific_state_path,
            must_exist=True,
        )
        if not state_path.is_file():
            raise InvalidEvolutionTransition(
                "source scientific-state snapshot is not a regular file"
            )
        previous = self.store.load_scientific_state(evolution_id, source.version)
        return IterationContext(
            task=task,
            source_episode=source,
            revision=revision,
            strategy=strategy,
            previous_scientific_state=previous,
        )

    def accept(
        self,
        evolution_id: str,
        version: EpisodeVersion,
    ) -> MutationResult[EvolutionTask]:
        with self.store.transaction(evolution_id) as transaction:
            task = transaction.load_task()
            if task.status != "AWAITING_EXPERT_FEEDBACK":
                raise InvalidEvolutionTransition(
                    "accept requires AWAITING_EXPERT_FEEDBACK"
                )
            episode = transaction.load_episode(version)
            if episode.status != "COMPLETED":
                raise InvalidEvolutionTransition("accepted version must be completed")
            self._verify_artifact(episode.artifact)
            self.validate_transition(task.status, "ACCEPTED")
            saved = transaction.save_task(
                task.model_copy(
                    update={
                        "status": "ACCEPTED",
                        "resume_status": None,
                        "accepted_version": version,
                    }
                ),
                expected_revision=task.revision,
            )
        event = EvolutionTaskAccepted(
            evolution_id=evolution_id,
            episode_version=version,
        )
        return MutationResult(saved, (event,))

    def stop(self, evolution_id: str) -> MutationResult[EvolutionTask]:
        with self.store.transaction(evolution_id) as transaction:
            task = transaction.load_task()
            self.validate_transition(task.status, "STOPPED")
            if task.status in {"BLOCKED", "BUDGET_EXHAUSTED"}:
                checkpoint = task.resume_status
            else:
                checkpoint = cast(EvolutionResumeStatus, task.status)
            if checkpoint is None:
                raise InvalidEvolutionTransition("paused task has no resume checkpoint")
            saved = transaction.save_task(
                task.model_copy(
                    update={"status": "STOPPED", "resume_status": checkpoint}
                ),
                expected_revision=task.revision,
            )
        return MutationResult(
            saved,
            (EvolutionTaskStopped(evolution_id=evolution_id),),
        )

    def reopen(self, evolution_id: str) -> MutationResult[EvolutionTask]:
        with self.store.transaction(evolution_id) as transaction:
            task = transaction.load_task()
            if task.status == "ACCEPTED":
                checkpoint: EvolutionStatus = "AWAITING_EXPERT_FEEDBACK"
            elif task.status in {"STOPPED", "BLOCKED", "BUDGET_EXHAUSTED"}:
                if task.resume_status is None:
                    raise InvalidEvolutionTransition("paused task has no resume checkpoint")
                checkpoint = task.resume_status
            else:
                raise InvalidEvolutionTransition(
                    f"cannot reopen an active task from {task.status}"
                )
            saved = transaction.save_task(
                task.model_copy(
                    update={"status": checkpoint, "resume_status": None}
                ),
                expected_revision=task.revision,
            )
        return MutationResult(saved)

    async def publish(
        self,
        result: MutationResult[object] | Iterable[RuntimeEvent],
    ) -> None:
        """Explicitly publish one result's events; never drive an event loop."""

        if self.event_sink is None:
            return
        events = result.events if isinstance(result, MutationResult) else result
        for event in events:
            pending = self.event_sink(event)
            if inspect.isawaitable(pending):
                await pending

    @staticmethod
    def validate_transition(source: str, target: str) -> None:
        if source not in ALLOWED_TRANSITIONS:
            raise InvalidEvolutionTransition(f"unknown evolution status: {source}")
        typed_source = cast(EvolutionStatus, source)
        if target not in ALLOWED_TRANSITIONS[typed_source]:
            raise InvalidEvolutionTransition(
                f"invalid evolution transition: {source} -> {target}"
            )

    def _current_episode(
        self,
        transaction: EvolutionTransaction,
        version: EpisodeVersion,
    ) -> tuple[EvolutionTask, EpisodeRecord]:
        task = transaction.load_task()
        if task.current_version != version:
            raise InvalidEvolutionTransition(
                f"episode version {version} is not current version {task.current_version}"
            )
        return task, transaction.load_episode(version)

    def _validate_comparison_episodes(
        self,
        task: EvolutionTask,
        previous: EpisodeRecord,
        current: EpisodeRecord,
    ) -> None:
        """Reject forged or stale records before computing scientific deltas."""

        if (
            previous.evolution_id != task.evolution_id
            or current.evolution_id != task.evolution_id
        ):
            raise EvolutionOperationConflict(
                "comparison episodes do not belong to the requested task"
            )
        previous_index = int(previous.version[1:])
        current_index = int(current.version[1:])
        if current_index <= previous_index:
            raise InvalidEvolutionTransition(
                "comparison versions must increase monotonically"
            )
        if (
            task.current_version != current.version
            or task.last_completed_version != current.version
        ):
            raise InvalidEvolutionTransition(
                "comparison current episode must be the task current completed version"
            )
        if current.parent_version != previous.version:
            raise InvalidEvolutionTransition(
                "compare requires adjacent parent/child episodes"
            )
        if previous.status != "COMPLETED" or current.status != "COMPLETED":
            raise InvalidEvolutionTransition(
                "compare requires two completed episodes"
            )
        for missing_index in range(previous_index + 1, current_index):
            missing_version = cast(EpisodeVersion, f"v{missing_index:03d}")
            missing = self.store.load_episode(task.evolution_id, missing_version)
            if missing.status != "FAILED":
                raise InvalidEvolutionTransition(
                    "numeric comparison gaps must be durable FAILED episodes"
                )
            expected_id = self._episode_id(task.evolution_id, missing_version)
            if (
                missing_index > len(task.episode_ids)
                or task.episode_ids[missing_index - 1] != expected_id
                or missing.episode_id != expected_id
                or missing.parent_version != previous.version
            ):
                raise EvolutionOperationConflict(
                    f"failed episode {missing_version} is not a canonical retry "
                    "between the compared parent/child episodes"
                )
        for episode, index in ((previous, previous_index), (current, current_index)):
            if index < 1 or index > len(task.episode_ids):
                raise EvolutionOperationConflict(
                    f"episode {episode.version} is not linked by the task manifest"
                )
            expected_id = self._episode_id(task.evolution_id, episode.version)
            if (
                task.episode_ids[index - 1] != expected_id
                or episode.episode_id != expected_id
            ):
                raise EvolutionOperationConflict(
                    f"episode {episode.version} identity is not canonical"
                )
            snapshot = episode.task_snapshot
            expected_snapshot = {
                "evolution_id": task.evolution_id,
                "goal": task.goal,
                "target": task.target.model_dump(mode="json"),
                "task_group_id": task.task_group_id,
                "input_sha256": task.input_sha256,
            }
            for field, expected in expected_snapshot.items():
                if snapshot.get(field) != expected:
                    raise EvolutionOperationConflict(
                        f"episode {episode.version} has noncanonical task "
                        f"snapshot field {field}"
                    )
            if episode.target_snapshot != TargetSnapshot.model_validate(task.target):
                raise EvolutionOperationConflict(
                    f"episode {episode.version} target snapshot is not canonical"
                )
            artifact = episode.artifact
            canonical_artifact = (
                f"user_output/{task.evolution_id}/{episode.version}/result.md"
            )
            if artifact is None or artifact.path != canonical_artifact:
                raise ArtifactMismatchError(
                    f"episode {episode.version} does not name its canonical "
                    "primary result"
                )

    def _comparison_state(self, episode: EpisodeRecord) -> ScientificState | None:
        if episode.scientific_state_path is None:
            return None
        expected = (
            f".photomatagent/evolutions/{episode.evolution_id}/episodes/"
            f"{episode.version}.scientific.json"
        )
        if episode.scientific_state_path != expected:
            raise EvolutionOperationConflict(
                f"episode {episode.version} scientific-state path is not canonical"
            )
        try:
            path = self.store.workspace.resolve(expected, must_exist=True)
        except (OSError, ValueError, ToolExecutionError) as exc:
            raise EvolutionOperationConflict(
                f"episode {episode.version} scientific-state snapshot is unavailable"
            ) from exc
        if not path.is_file():
            raise EvolutionOperationConflict(
                f"episode {episode.version} scientific-state snapshot is not a file"
            )
        return self.store.load_scientific_state(
            episode.evolution_id,
            episode.version,
        )

    def _comparison_compilation(
        self,
        task: EvolutionTask,
        feedback: ExpertFeedbackRecord,
    ) -> FeedbackCompilation | None:
        matches = [
            item
            for item in self.store.list_compilations(task.evolution_id)
            if item.compilation_id in task.compilation_ids
            and item.feedback_id == feedback.feedback_id
            and item.episode_version == feedback.episode_version
            and item.status == "AVAILABLE"
        ]
        if len(matches) > 1:
            raise EvolutionOperationConflict(
                f"feedback {feedback.feedback_id} has multiple available compilations"
            )
        return matches[0] if matches else None

    @staticmethod
    def _require_feedback_artifact(
        feedback: ExpertFeedbackRecord,
        episode: EpisodeRecord,
    ) -> None:
        if (
            episode.artifact is None
            or feedback.result_sha256 != episode.artifact.sha256
        ):
            raise ArtifactMismatchError(
                f"feedback {feedback.feedback_id} is not bound to its episode artifact"
            )

    @staticmethod
    def _unsafe_experience_observation(
        previous_feedback: ExpertFeedbackRecord,
        current_feedback: ExpertFeedbackRecord | None,
        previous_compilation: FeedbackCompilation | None,
        current_compilation: FeedbackCompilation | None,
    ) -> bool:
        feedback_records = [previous_feedback]
        if current_feedback is not None:
            feedback_records.append(current_feedback)
        if any(item.flags.fabricated_source for item in feedback_records):
            return True
        compilations = [
            item
            for item in (previous_compilation, current_compilation)
            if item is not None
        ]
        return any(
            delta.category == "SAFETY" and delta.status != "POSITIVE_SIGNAL"
            for compilation in compilations
            for delta in compilation.items
        )

    @staticmethod
    def _require_episode_owner(
        episode: EpisodeRecord,
        owner_token: str | None,
    ) -> None:
        if episode.owner_token is not None and episode.owner_token != owner_token:
            raise EvolutionOperationConflict(
                "episode transition owner token does not match the active invocation"
            )

    def _verify_artifact(self, artifact: ArtifactRef | None) -> ArtifactRef:
        if artifact is None:
            raise ArtifactMismatchError("a completed episode requires an artifact")
        if Path(artifact.path).is_absolute():
            raise ArtifactMismatchError("artifact path must be workspace-relative")
        try:
            path = self.store.workspace.resolve(artifact.path, must_exist=True)
        except (OSError, ValueError, ToolExecutionError) as exc:
            raise ArtifactMismatchError(f"artifact is unavailable: {artifact.path}") from exc
        if not path.is_file():
            raise ArtifactMismatchError("artifact is not a regular file")
        digest = hashlib.sha256()
        size = 0
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    size += len(chunk)
                    digest.update(chunk)
        except OSError as exc:
            raise ArtifactMismatchError(f"artifact cannot be read: {artifact.path}") from exc
        if size != artifact.size_bytes or digest.hexdigest() != artifact.sha256:
            raise ArtifactMismatchError("artifact bytes do not match declared size and SHA-256")
        return artifact

    @staticmethod
    def _validated_completion(
        existing: EpisodeRecord,
        result: EpisodeRecord,
        artifact: ArtifactRef,
    ) -> EpisodeRecord:
        """Normalize a completion and compare all recovery-authoritative fields."""

        completed_at = result.completed_at
        if completed_at is None:
            completed_at = existing.completed_at or utc_now()
        candidate_payload = cast(
            dict[str, object],
            redact_secrets(
                result.model_copy(
                    update={
                        "status": "COMPLETED",
                        "artifact": artifact,
                        "completed_at": completed_at,
                        "error": None,
                    }
                ).model_dump(mode="json")
            ),
        )
        existing_payload = existing.model_dump(mode="json")
        for field in _EPISODE_COMPLETION_IDENTITY_FIELDS:
            if candidate_payload[field] != existing_payload[field]:
                raise EvolutionOperationConflict(
                    f"episode completion changed authoritative field {field}"
                )
        candidate = EpisodeRecord.model_validate(candidate_payload)
        if existing.status != "COMPLETED":
            return candidate
        candidate_payload = candidate.model_dump(mode="json")
        differing = next(
            (
                field
                for field in EpisodeRecord.model_fields
                if existing_payload[field] != candidate_payload[field]
            ),
            None,
        )
        if differing is not None:
            raise EvolutionOperationConflict(
                "completed episode retry differs from stored record at field "
                f"{differing}"
            )
        return candidate

    @staticmethod
    def _validate_reservation(
        episode: EpisodeRecord,
        task: EvolutionTask,
        mode: ExecutionMode,
        provider: str | None,
        model: str | None,
        tool_surface_fingerprint: Sha256 | None,
        capability_fingerprint: Sha256 | None,
        data_source_fingerprints: dict[str, Sha256],
        strategy_id: str | None,
        strategy_arm: StrategyArm,
        owner_token: str | None,
    ) -> None:
        expected_version = (
            task.current_version
            if task.status == "RUNNING" and task.current_version is not None
            else EvolutionService._next_version(task.current_version)
        )
        expected_id = EvolutionService._episode_id(task.evolution_id, expected_version)
        if (
            episode.status != "RESERVED"
            or episode.episode_id != expected_id
            or episode.version != expected_version
            or episode.execution_mode != mode
            or episode.provider != provider
            or episode.model != model
            or episode.tool_surface_fingerprint != tool_surface_fingerprint
            or episode.capability_fingerprint != capability_fingerprint
            or episode.data_source_fingerprints != data_source_fingerprints
            or episode.strategy_id != strategy_id
            or episode.strategy_arm != strategy_arm
            or episode.owner_token != owner_token
            or episode.parent_version != task.last_completed_version
            or episode.applied_feedback_id
            != (task.feedback_ids[-1] if task.feedback_ids else None)
            or episode.revision_plan_id
            != (task.revision_ids[-1] if task.revision_ids else None)
            or (
                task.status != "RUNNING"
                and episode.task_snapshot != task.model_dump(mode="json")
            )
            or episode.target_snapshot
            != TargetSnapshot.model_validate(task.target)
        ):
            raise EvolutionOperationConflict(
                f"mismatched durable reservation for {expected_version}"
            )

    @staticmethod
    def _build_feedback(
        *,
        feedback_id: str,
        evolution_id: str,
        version: EpisodeVersion,
        draft: ExpertFeedbackDraft,
        result_sha256: Sha256,
        raw_input: str | None,
        hard_cap_override_reason: str | None,
    ) -> ExpertFeedbackRecord:
        assessment = assess_hard_caps(draft.scores, draft.flags)
        return ExpertFeedbackRecord(
            feedback_id=feedback_id,
            evolution_id=evolution_id,
            episode_version=version,
            result_sha256=result_sha256,
            rubric_version="expert-review-v1",
            raw_input=raw_input or draft.model_dump_json(),
            scores=draft.scores,
            flags=draft.flags,
            fatal_issue=draft.fatal_issue,
            comments=draft.comments,
            priority_corrections=draft.priority_corrections,
            preserved_strengths=draft.preserved_strengths,
            recommended_actions=draft.recommended_actions,
            resolved_issue_ids=draft.resolved_issue_ids,
            suggested_scores=(assessment.suggested_scores if assessment.reasons else None),
            hard_cap_reasons=assessment.reasons,
            hard_cap_override_reason=hard_cap_override_reason,
        )

    @staticmethod
    def _active_feedback(
        feedback: list[ExpertFeedbackRecord],
        version: EpisodeVersion,
    ) -> ExpertFeedbackRecord | None:
        superseded = {
            item.supersedes_feedback_id
            for item in feedback
            if item.supersedes_feedback_id is not None
        }
        active = [
            item
            for item in feedback
            if item.episode_version == version and item.feedback_id not in superseded
        ]
        if len(active) > 1:
            raise EvolutionOperationConflict(
                f"episode {version} has multiple active feedback records"
            )
        return active[0] if active else None

    @staticmethod
    def _require_matching_feedback(
        existing: ExpertFeedbackRecord,
        candidate: ExpertFeedbackRecord,
    ) -> None:
        ignored = {"confirmed_at"}
        existing_payload = existing.model_dump(mode="json", exclude=ignored)
        candidate_payload = redact_secrets(
            candidate.model_dump(mode="json", exclude=ignored)
        )
        if existing_payload != candidate_payload:
            raise EvolutionOperationConflict(
                f"feedback ID {candidate.feedback_id} names different content"
            )

    @staticmethod
    def _validate_revision(
        task: EvolutionTask,
        evolution_id: str,
        plan: RevisionPlan,
    ) -> None:
        if task.status not in {"FEEDBACK_RECORDED", "REVISION_READY"}:
            raise InvalidEvolutionTransition(
                "revision confirmation requires FEEDBACK_RECORDED"
            )
        if not plan.confirmed:
            raise InvalidEvolutionTransition("revision plan is not confirmed")
        if (
            plan.evolution_id != evolution_id
            or plan.source_version != task.last_completed_version
            or not task.feedback_ids
            or plan.feedback_id != task.feedback_ids[-1]
        ):
            raise InvalidEvolutionTransition(
                "revision plan does not match active task version and feedback"
            )

    def _revision_context(
        self,
        transaction: EvolutionTransaction,
        task: EvolutionTask,
        evolution_id: str,
    ) -> tuple[EpisodeRecord, ExpertFeedbackRecord, FeedbackCompilation]:
        if task.last_completed_version is None or not task.feedback_ids:
            raise InvalidEvolutionTransition(
                "revision confirmation requires a source episode and active feedback"
            )
        episode = transaction.load_episode(task.last_completed_version)
        if episode.status != "COMPLETED":
            raise InvalidEvolutionTransition(
                "revision confirmation requires a completed source episode"
            )
        artifact = self._verify_artifact(episode.artifact)
        feedback_records = [
            transaction.load_feedback(feedback_id)
            for feedback_id in task.feedback_ids
        ]
        feedback = feedback_records[-1]
        active_feedback = self._active_feedback(feedback_records, episode.version)
        if active_feedback is None or active_feedback != feedback:
            raise EvolutionOperationConflict(
                "latest feedback is not the exact active source feedback"
            )
        if (
            feedback.evolution_id != evolution_id
            or feedback.episode_version != episode.version
            or feedback.result_sha256 != artifact.sha256
        ):
            raise EvolutionOperationConflict(
                "active feedback does not match the source episode"
            )
        available: list[FeedbackCompilation] = []
        for compilation_id in task.compilation_ids:
            compilation = transaction.load_compilation(compilation_id)
            if compilation.status == "AVAILABLE":
                available.append(compilation)
        matching = [
            item
            for item in available
            if item.feedback_id == feedback.feedback_id
            and item.episode_version == episode.version
        ]
        if len(matching) != 1:
            raise InvalidEvolutionTransition(
                "revision confirmation requires exactly one active AVAILABLE compilation"
            )
        return episode, feedback, matching[0]

    @staticmethod
    def _require_canonical_plan(
        submitted: RevisionPlan,
        canonical: RevisionPlan,
    ) -> None:
        service_owned = {"confirmed", "confirmed_at"}
        if submitted.model_dump(mode="json", exclude=service_owned) != canonical.model_dump(
            mode="json",
            exclude=service_owned,
        ):
            raise EvolutionOperationConflict(
                "submitted revision plan does not match canonical persisted inputs"
            )

    @staticmethod
    def _reservation_strategy(
        transaction: EvolutionTransaction,
        task: EvolutionTask,
    ) -> tuple[str | None, StrategyArm]:
        if task.last_completed_version is None:
            return None, "STATIC"
        if not task.revision_ids or not task.strategy_ids:
            raise InvalidEvolutionTransition(
                "revised reservation requires a confirmed revision and strategy"
            )
        revision = transaction.load_revision(task.revision_ids[-1])
        strategy = transaction.load_strategy(task.strategy_ids[-1])
        from photomatagent.scientific.evolution.strategy import FixedStrategySelector

        canonical = FixedStrategySelector().select(task, revision)
        if (
            not revision.confirmed
            or revision.source_version != task.last_completed_version
            or strategy != canonical
        ):
            raise EvolutionOperationConflict(
                "active revision and strategy do not match the task checkpoint"
            )
        return strategy.strategy_id, strategy.arm

    @staticmethod
    def _require_matching_revision(existing: RevisionPlan, candidate: RevisionPlan) -> None:
        ignored = {"confirmed_at", "created_at"}
        existing_payload = existing.model_dump(mode="json", exclude=ignored)
        candidate_payload = redact_secrets(
            candidate.model_dump(mode="json", exclude=ignored)
        )
        if existing_payload != candidate_payload:
            raise EvolutionOperationConflict(
                f"revision ID {candidate.revision_id} names different content"
            )

    @staticmethod
    def _require_matching_strategy(
        existing: StrategyVersion,
        candidate: StrategyVersion,
    ) -> None:
        if existing != candidate:
            raise EvolutionOperationConflict(
                f"strategy ID {candidate.strategy_id} names different content"
            )

    @staticmethod
    def _append_once(values: list[str], value: str) -> list[str]:
        return values if value in values else [*values, value]

    @staticmethod
    def _reservation_events(
        episode: EpisodeRecord,
        initial: bool,
    ) -> tuple[RuntimeEvent, ...]:
        if initial:
            return ()
        return (
            EvolutionIterationStarted(
                evolution_id=episode.evolution_id,
                episode_version=episode.version,
            ),
        )

    @staticmethod
    def _next_version(current: EpisodeVersion | None) -> EpisodeVersion:
        value = 1 if current is None else int(current[1:]) + 1
        if value > 999:
            raise InvalidEvolutionTransition("episode version space is exhausted")
        return cast(EpisodeVersion, f"v{value:03d}")

    @staticmethod
    def _episode_id(evolution_id: str, version: EpisodeVersion) -> str:
        digest = hashlib.sha256(f"{evolution_id}:{version}".encode()).hexdigest()[:10]
        return f"ep_{digest}"

    @staticmethod
    def _input_hash(goal: str, target: TargetSpec) -> str:
        payload = json.dumps(
            {"goal": goal, "target": target.model_dump(mode="json")},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


__all__ = [
    "ALLOWED_TRANSITIONS",
    "ArtifactMismatchError",
    "EventSink",
    "EvolutionOperationConflict",
    "EvolutionService",
    "EvolutionServiceError",
    "InvalidEvolutionTransition",
    "IterationContext",
    "MutationResult",
]
