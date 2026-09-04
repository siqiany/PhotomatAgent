"""Application service enforcing the persistent evolution lifecycle."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Awaitable, Callable, Iterable
from typing import cast

from photomatagent.runtime.events import (
    EvolutionEpisodeCompleted,
    EvolutionEpisodeStarted,
    EvolutionIterationStarted,
    EvolutionTaskAccepted,
    EvolutionTaskCreated,
    EvolutionTaskStopped,
    ExpertFeedbackRecorded,
    RevisionPlanConfirmed,
    RuntimeEvent,
)
from photomatagent.scientific.evolution.events import bounded_summary
from photomatagent.scientific.evolution.models import (
    EpisodeRecord,
    EpisodeVersion,
    EvolutionStatus,
    EvolutionTask,
    ExecutionMode,
    ExpertFeedbackDraft,
    ExpertFeedbackRecord,
    RevisionPlan,
    Sha256,
    TargetSnapshot,
    new_episode_id,
    new_evolution_id,
    new_feedback_id,
    utc_now,
)
from photomatagent.scientific.evolution.rubric import assess_hard_caps
from photomatagent.scientific.evolution.store import EvolutionStore
from photomatagent.scientific.loop import TargetSpec

EventSink = Callable[[RuntimeEvent], Awaitable[None] | None]

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
    """Raised when an operation is not legal in the authoritative state."""


class ArtifactMismatchError(EvolutionServiceError):
    """Raised when feedback is not bound to the persisted result artifact."""


class EvolutionService:
    """Coordinate lifecycle records without executing models or scientific tools."""

    def __init__(
        self,
        store: EvolutionStore,
        *,
        event_sink: EventSink | None = None,
    ) -> None:
        self.store = store
        self.event_sink = event_sink
        self._pending_events: list[RuntimeEvent] = []

    def get(self, evolution_id: str) -> EvolutionTask:
        """Return the authoritative persisted task manifest."""

        return self.store.load_task(evolution_id)

    def create_task(
        self,
        *,
        goal: str,
        target: TargetSpec,
        task_group_id: str | None = None,
        input_sha256: str | None = None,
        evolution_id: str | None = None,
    ) -> EvolutionTask:
        """Persist a new task before any episode is reserved or executed."""

        resolved_id = evolution_id or new_evolution_id()
        task = EvolutionTask(
            evolution_id=resolved_id,
            goal=goal,
            target=target,
            task_group_id=task_group_id or resolved_id,
            input_sha256=input_sha256 or self._input_hash(goal, target),
        )
        created = self.store.create_task(task)
        self._queue(
            EvolutionTaskCreated(
                evolution_id=created.evolution_id,
                goal_summary=bounded_summary(created.goal),
            )
        )
        return created

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
    ) -> EpisodeRecord:
        """Persist the next monotonic episode, then advance the task to RUNNING."""

        task = self.get(evolution_id)
        first = task.current_version is None
        required_status: EvolutionStatus = "CREATED" if first else "REVISION_READY"
        if task.status != required_status:
            raise InvalidEvolutionTransition(
                f"cannot reserve an episode while task is {task.status}; "
                f"required {required_status}"
            )
        if first and mode != "NORMAL":
            raise InvalidEvolutionTransition("the first episode must use NORMAL mode")
        if not first and mode == "NORMAL":
            raise InvalidEvolutionTransition(
                "a revised episode must use an explicit evidence or evaluation mode"
            )
        self.validate_transition(task.status, "RUNNING")
        version = self._next_version(task.current_version)
        episode = EpisodeRecord(
            evolution_id=evolution_id,
            episode_id=new_episode_id(),
            version=version,
            parent_version=task.last_completed_version,
            applied_feedback_id=(task.feedback_ids[-1] if task.feedback_ids else None),
            revision_plan_id=(task.revision_ids[-1] if task.revision_ids else None),
            execution_mode=mode,
            task_snapshot=task.model_dump(mode="json"),
            target_snapshot=TargetSnapshot.model_validate(task.target),
            provider=provider,
            model=model,
            tool_surface_fingerprint=tool_surface_fingerprint,
            capability_fingerprint=capability_fingerprint,
            data_source_fingerprints=data_source_fingerprints or {},
        )
        self.store.write_episode(episode)
        updated = task.model_copy(
            update={
                "status": "RUNNING",
                "current_version": version,
                "episode_ids": [*task.episode_ids, episode.episode_id],
            }
        )
        self.store.save_task(updated, expected_revision=task.revision)
        if not first:
            self._queue(
                EvolutionIterationStarted(
                    evolution_id=evolution_id,
                    episode_version=version,
                )
            )
        return episode

    def mark_episode_running(
        self,
        evolution_id: str,
        version: EpisodeVersion,
        *,
        runtime_session_id: str | None = None,
        event_log_path: str | None = None,
    ) -> EpisodeRecord:
        """Start exactly the currently reserved episode."""

        task, episode = self._current_episode(evolution_id, version)
        if task.status != "RUNNING" or episode.status != "RESERVED":
            raise InvalidEvolutionTransition(
                "episode can start only from task RUNNING / episode RESERVED"
            )
        running = episode.model_copy(
            update={
                "status": "RUNNING",
                "runtime_session_id": runtime_session_id,
                "event_log_path": event_log_path,
                "started_at": utc_now(),
            }
        )
        saved = self.store.transition_episode(running, expected_status="RESERVED")
        self._queue(
            EvolutionEpisodeStarted(
                evolution_id=evolution_id,
                episode_version=version,
            )
        )
        return saved

    def complete_episode(
        self,
        evolution_id: str,
        version: EpisodeVersion,
        *,
        result: EpisodeRecord,
    ) -> EpisodeRecord:
        """Persist a reviewable terminal result before exposing it for feedback."""

        task, episode = self._current_episode(evolution_id, version)
        if task.status != "RUNNING" or episode.status != "RUNNING":
            raise InvalidEvolutionTransition(
                "episode can complete only from task RUNNING / episode RUNNING"
            )
        self._validate_result_identity(result, episode)
        if result.artifact is None:
            raise InvalidEvolutionTransition(
                "a completed episode requires a program-selected primary artifact"
            )
        self.validate_transition(task.status, "AWAITING_EXPERT_FEEDBACK")
        completed = result.model_copy(
            update={
                "status": "COMPLETED",
                "completed_at": result.completed_at or utc_now(),
                "error": None,
            }
        )
        saved_episode = self.store.transition_episode(
            completed,
            expected_status="RUNNING",
        )
        updated = task.model_copy(
            update={
                "status": "AWAITING_EXPERT_FEEDBACK",
                "last_completed_version": version,
            }
        )
        self.store.save_task(updated, expected_revision=task.revision)
        self._queue(
            EvolutionEpisodeCompleted(
                evolution_id=evolution_id,
                episode_version=version,
            )
        )
        return saved_episode

    def fail_episode(
        self,
        evolution_id: str,
        version: EpisodeVersion,
        error: str,
    ) -> EpisodeRecord:
        """Persist a failed attempt while retaining the last completed result."""

        task, episode = self._current_episode(evolution_id, version)
        if task.status != "RUNNING" or episode.status not in {"RESERVED", "RUNNING"}:
            raise InvalidEvolutionTransition(
                "only the active reserved or running episode may fail"
            )
        self.validate_transition(task.status, "BLOCKED")
        failed = episode.model_copy(
            update={
                "status": "FAILED",
                "completed_at": utc_now(),
                "error": error,
            }
        )
        saved_episode = self.store.transition_episode(
            failed,
            expected_status=episode.status,
        )
        updated = task.model_copy(update={"status": "BLOCKED"})
        self.store.save_task(updated, expected_revision=task.revision)
        return saved_episode

    def attach_feedback(
        self,
        evolution_id: str,
        version: EpisodeVersion,
        *,
        draft: ExpertFeedbackDraft,
        result_sha256: Sha256,
        raw_input: str | None = None,
        hard_cap_override_reason: str | None = None,
    ) -> ExpertFeedbackRecord:
        """Bind one immutable active expert review to the exact result hash."""

        task, episode = self._current_episode(evolution_id, version)
        if task.status != "AWAITING_EXPERT_FEEDBACK" or episode.status != "COMPLETED":
            raise InvalidEvolutionTransition(
                "feedback requires an awaiting task and a completed episode"
            )
        if episode.artifact is None or episode.artifact.sha256 != result_sha256:
            raise ArtifactMismatchError(
                "expert feedback hash does not match the persisted primary artifact"
            )
        self.validate_transition(task.status, "FEEDBACK_RECORDED")
        assessment = assess_hard_caps(draft.scores, draft.flags)
        feedback = ExpertFeedbackRecord(
            feedback_id=new_feedback_id(),
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
            suggested_scores=(assessment.suggested_scores if assessment.reasons else None),
            hard_cap_reasons=assessment.reasons,
            hard_cap_override_reason=hard_cap_override_reason,
        )
        self.store.write_feedback(feedback)
        updated = task.model_copy(
            update={
                "status": "FEEDBACK_RECORDED",
                "feedback_ids": [*task.feedback_ids, feedback.feedback_id],
            }
        )
        self.store.save_task(updated, expected_revision=task.revision)
        self._queue(
            ExpertFeedbackRecorded(
                evolution_id=evolution_id,
                episode_version=version,
                feedback_id=feedback.feedback_id,
                result_sha256=result_sha256,
                scores={
                    "scientific_correctness": feedback.scores.scientific_correctness,
                    "evidence_sufficiency": feedback.scores.evidence_sufficiency,
                    "novelty": feedback.scores.novelty,
                    "actionability": feedback.scores.actionability,
                    "overall": feedback.scores.overall,
                },
            )
        )
        return feedback

    def confirm_revision(
        self,
        evolution_id: str,
        plan: RevisionPlan,
    ) -> RevisionPlan:
        """Persist a human-confirmed revision contract before iteration."""

        task = self.get(evolution_id)
        if task.status != "FEEDBACK_RECORDED":
            raise InvalidEvolutionTransition(
                "revision confirmation requires FEEDBACK_RECORDED"
            )
        if not plan.confirmed:
            raise InvalidEvolutionTransition("revision plan is not confirmed")
        if plan.has_blocking_ambiguity:
            raise InvalidEvolutionTransition(
                "revision plan has an unresolved blocking ambiguity"
            )
        if (
            plan.evolution_id != evolution_id
            or plan.source_version != task.last_completed_version
            or not task.feedback_ids
            or plan.feedback_id != task.feedback_ids[-1]
        ):
            raise InvalidEvolutionTransition(
                "revision plan does not match the active task version and feedback"
            )
        self.validate_transition(task.status, "REVISION_READY")
        confirmed = plan.model_copy(update={"confirmed_at": plan.confirmed_at or utc_now()})
        self.store.write_revision(confirmed)
        updated = task.model_copy(
            update={
                "status": "REVISION_READY",
                "revision_ids": [*task.revision_ids, confirmed.revision_id],
            }
        )
        self.store.save_task(updated, expected_revision=task.revision)
        self._queue(
            RevisionPlanConfirmed(
                evolution_id=evolution_id,
                episode_version=confirmed.source_version,
            )
        )
        return confirmed

    def accept(self, evolution_id: str, version: EpisodeVersion) -> EvolutionTask:
        """Select a completed result without changing its scientific verdict."""

        task = self.get(evolution_id)
        if task.status != "AWAITING_EXPERT_FEEDBACK":
            raise InvalidEvolutionTransition(
                "accept requires AWAITING_EXPERT_FEEDBACK"
            )
        episode = self.store.load_episode(evolution_id, version)
        if episode.status != "COMPLETED" or version != task.last_completed_version:
            raise InvalidEvolutionTransition(
                "accepted version must be the latest completed episode"
            )
        self.validate_transition(task.status, "ACCEPTED")
        updated = task.model_copy(
            update={"status": "ACCEPTED", "accepted_version": version}
        )
        saved = self.store.save_task(updated, expected_revision=task.revision)
        self._queue(
            EvolutionTaskAccepted(
                evolution_id=evolution_id,
                episode_version=version,
            )
        )
        return saved

    def stop(self, evolution_id: str) -> EvolutionTask:
        """Explicitly stop a task from a state allowed by the transition table."""

        task = self.get(evolution_id)
        self.validate_transition(task.status, "STOPPED")
        updated = task.model_copy(update={"status": "STOPPED"})
        saved = self.store.save_task(updated, expected_revision=task.revision)
        self._queue(EvolutionTaskStopped(evolution_id=evolution_id))
        return saved

    def reopen(self, evolution_id: str) -> EvolutionTask:
        """Return an explicitly closed or paused task to human review."""

        task = self.get(evolution_id)
        if task.status not in {"ACCEPTED", "STOPPED", "BUDGET_EXHAUSTED", "BLOCKED"}:
            raise InvalidEvolutionTransition(
                f"cannot reopen an active task from {task.status}"
            )
        self.validate_transition(task.status, "AWAITING_EXPERT_FEEDBACK")
        updated = task.model_copy(update={"status": "AWAITING_EXPERT_FEEDBACK"})
        return self.store.save_task(updated, expected_revision=task.revision)

    def drain_events(self) -> list[RuntimeEvent]:
        """Return generated events so the caller can persist them explicitly."""

        events = self._pending_events
        self._pending_events = []
        return events

    async def publish_events(self, events: Iterable[RuntimeEvent]) -> None:
        """Send returned events through the configured sink without a hidden loop."""

        if self.event_sink is None:
            return
        for event in events:
            pending = self.event_sink(event)
            if inspect.isawaitable(pending):
                await pending

    @staticmethod
    def validate_transition(source: str, target: str) -> None:
        """Validate one lifecycle edge against the single explicit table."""

        if source not in ALLOWED_TRANSITIONS:
            raise InvalidEvolutionTransition(f"unknown evolution status: {source}")
        typed_source = cast(EvolutionStatus, source)
        if target not in ALLOWED_TRANSITIONS[typed_source]:
            raise InvalidEvolutionTransition(
                f"invalid evolution transition: {source} -> {target}"
            )

    def _current_episode(
        self,
        evolution_id: str,
        version: EpisodeVersion,
    ) -> tuple[EvolutionTask, EpisodeRecord]:
        task = self.get(evolution_id)
        if task.current_version != version:
            raise InvalidEvolutionTransition(
                f"episode version {version} is not current version {task.current_version}"
            )
        return task, self.store.load_episode(evolution_id, version)

    @staticmethod
    def _validate_result_identity(result: EpisodeRecord, episode: EpisodeRecord) -> None:
        if result.evolution_id != episode.evolution_id or result.version != episode.version:
            raise InvalidEvolutionTransition(
                "episode result identity does not match the active episode version"
            )

    @staticmethod
    def _next_version(current: EpisodeVersion | None) -> EpisodeVersion:
        value = 1 if current is None else int(current[1:]) + 1
        if value > 999:
            raise InvalidEvolutionTransition("episode version space is exhausted")
        return cast(EpisodeVersion, f"v{value:03d}")

    @staticmethod
    def _input_hash(goal: str, target: TargetSpec) -> str:
        payload = json.dumps(
            {"goal": goal, "target": target.model_dump(mode="json")},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _queue(self, event: RuntimeEvent) -> None:
        self._pending_events.append(event)


__all__ = [
    "ALLOWED_TRANSITIONS",
    "ArtifactMismatchError",
    "EventSink",
    "EvolutionService",
    "EvolutionServiceError",
    "InvalidEvolutionTransition",
]
