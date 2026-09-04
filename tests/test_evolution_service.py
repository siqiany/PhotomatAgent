from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock

import pytest

from photomatagent.runtime.events import RuntimeEvent
from photomatagent.scientific.evolution.models import (
    ArtifactRef,
    EpisodeRecord,
    ExpertFeedbackDraft,
    RevisionPlan,
    RubricScores,
)
from photomatagent.scientific.evolution.service import (
    ALLOWED_TRANSITIONS,
    ArtifactMismatchError,
    EvolutionService,
    InvalidEvolutionTransition,
)
from photomatagent.scientific.evolution.store import (
    EvolutionConflictError,
    EvolutionStore,
)
from photomatagent.scientific.loop import ScientificLoopSummary, TargetSpec
from photomatagent.workspace import Workspace


def make_service(tmp_path: Path) -> EvolutionService:
    return EvolutionService(EvolutionStore(Workspace(tmp_path)))


def test_service_contract_is_exported_from_evolution_package() -> None:
    from photomatagent.scientific.evolution import (
        ArtifactMismatchError as ExportedArtifactMismatchError,
        EvolutionService as ExportedEvolutionService,
        InvalidEvolutionTransition as ExportedInvalidEvolutionTransition,
    )

    assert ExportedEvolutionService is EvolutionService
    assert ExportedArtifactMismatchError is ArtifactMismatchError
    assert ExportedInvalidEvolutionTransition is InvalidEvolutionTransition


def completed_result(episode: EpisodeRecord, *, digest: str = "b" * 64) -> EpisodeRecord:
    return episode.model_copy(
        update={
            "summary": ScientificLoopSummary(
                status="INCONCLUSIVE",
                rounds=1,
                candidate_count=1,
                best_candidate_id="candidate_1",
                best_score=0.5,
                final_evaluation=None,
            ),
            "artifact": ArtifactRef(
                path=f"user_output/{episode.evolution_id}/{episode.version}/result.md",
                size_bytes=10,
                sha256=digest,
            ),
        }
    )


def feedback_draft() -> ExpertFeedbackDraft:
    return ExpertFeedbackDraft(
        scores=RubricScores(
            scientific_correctness=3,
            evidence_sufficiency=3,
            novelty=3,
            actionability=3,
            overall=3,
        ),
        comments="Needs stronger primary evidence.",
    )


def complete_first(service: EvolutionService) -> tuple[str, EpisodeRecord]:
    task = service.create_task(
        goal="find a stable infrared absorber",
        target=TargetSpec(goal="find a stable infrared absorber"),
    )
    episode = service.reserve_episode(task.evolution_id, mode="NORMAL")
    running = service.mark_episode_running(task.evolution_id, episode.version)
    completed = service.complete_episode(
        task.evolution_id,
        running.version,
        result=completed_result(running),
    )
    return task.evolution_id, completed


def prepare_revision(service: EvolutionService) -> tuple[str, EpisodeRecord]:
    evolution_id, completed = complete_first(service)
    feedback = service.attach_feedback(
        evolution_id,
        completed.version,
        draft=feedback_draft(),
        result_sha256=completed.artifact.sha256,  # type: ignore[union-attr]
        raw_input='{"scores": "confirmed by expert"}',
    )
    service.confirm_revision(
        evolution_id,
        RevisionPlan(
            revision_id="rp_test",
            evolution_id=evolution_id,
            source_version=completed.version,
            feedback_id=feedback.feedback_id,
            confirmed=True,
        ),
    )
    return evolution_id, completed


def test_lifecycle_requires_feedback_before_next_episode(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    evolution_id, _ = complete_first(service)

    assert service.get(evolution_id).status == "AWAITING_EXPERT_FEEDBACK"
    with pytest.raises(InvalidEvolutionTransition):
        service.reserve_episode(evolution_id, mode="CARRY_VERIFIED_EVIDENCE")


def test_failed_next_episode_never_overwrites_last_good_result(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    evolution_id, first = prepare_revision(service)
    second = service.reserve_episode(
        evolution_id,
        mode="CARRY_VERIFIED_EVIDENCE",
    )
    service.fail_episode(evolution_id, second.version, "provider failed")

    task = service.get(evolution_id)
    assert task.current_version == "v002"
    assert task.last_completed_version == first.version == "v001"
    assert task.status == "BLOCKED"


@pytest.mark.parametrize(
    ("inner_status", "expected_status"),
    [
        ("STALLED", "AWAITING_EXPERT_FEEDBACK"),
        ("INCONCLUSIVE", "AWAITING_EXPERT_FEEDBACK"),
        ("BUDGET_EXHAUSTED", "AWAITING_EXPERT_FEEDBACK"),
    ],
)
def test_reviewable_inner_loop_results_do_not_exhaust_evolution_budget(
    tmp_path: Path, inner_status: str, expected_status: str
) -> None:
    service = make_service(tmp_path)
    task = service.create_task(goal="goal", target=TargetSpec(goal="goal"))
    episode = service.reserve_episode(task.evolution_id, mode="NORMAL")
    running = service.mark_episode_running(task.evolution_id, episode.version)
    result = completed_result(running)
    result.summary.status = inner_status  # type: ignore[assignment,union-attr]

    service.complete_episode(task.evolution_id, running.version, result=result)

    assert service.get(task.evolution_id).status == expected_status


def test_feedback_rejects_wrong_version_hash_and_duplicate_active_review(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)
    evolution_id, completed = complete_first(service)
    digest = completed.artifact.sha256  # type: ignore[union-attr]

    with pytest.raises(InvalidEvolutionTransition, match="version"):
        service.attach_feedback(
            evolution_id,
            "v002",
            draft=feedback_draft(),
            result_sha256=digest,
            raw_input="review",
        )
    with pytest.raises(ArtifactMismatchError):
        service.attach_feedback(
            evolution_id,
            completed.version,
            draft=feedback_draft(),
            result_sha256="0" * 64,
            raw_input="review",
        )

    service.attach_feedback(
        evolution_id,
        completed.version,
        draft=feedback_draft(),
        result_sha256=digest,
        raw_input="review",
    )
    with pytest.raises(InvalidEvolutionTransition):
        service.attach_feedback(
            evolution_id,
            completed.version,
            draft=feedback_draft(),
            result_sha256=digest,
            raw_input="second review",
        )


def test_unconfirmed_or_mismatched_revision_is_never_persisted(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    evolution_id, completed = complete_first(service)
    feedback = service.attach_feedback(
        evolution_id,
        completed.version,
        draft=feedback_draft(),
        result_sha256=completed.artifact.sha256,  # type: ignore[union-attr]
        raw_input="review",
    )

    for plan in (
        RevisionPlan(
            revision_id="rp_unconfirmed",
            evolution_id=evolution_id,
            source_version=completed.version,
            feedback_id=feedback.feedback_id,
        ),
        RevisionPlan(
            revision_id="rp_wrong_feedback",
            evolution_id=evolution_id,
            source_version=completed.version,
            feedback_id="fb_wrong",
            confirmed=True,
        ),
    ):
        with pytest.raises(InvalidEvolutionTransition):
            service.confirm_revision(evolution_id, plan)

    assert service.get(evolution_id).revision_ids == []
    assert not (service.store.root / evolution_id / "revisions").exists()


def test_revision_with_blocking_ambiguity_cannot_become_ready(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    evolution_id, completed = complete_first(service)
    feedback = service.attach_feedback(
        evolution_id,
        completed.version,
        draft=feedback_draft(),
        result_sha256=completed.artifact.sha256,  # type: ignore[union-attr]
        raw_input="review",
    )
    plan = RevisionPlan(
        revision_id="rp_ambiguous",
        evolution_id=evolution_id,
        source_version=completed.version,
        feedback_id=feedback.feedback_id,
        confirmed=True,
        unresolved_ambiguities=["Which measurement protocol applies?"],
        has_blocking_ambiguity=True,
    )

    with pytest.raises(InvalidEvolutionTransition, match="ambiguity"):
        service.confirm_revision(evolution_id, plan)

    assert service.get(evolution_id).status == "FEEDBACK_RECORDED"
    assert service.get(evolution_id).revision_ids == []


def test_accepted_task_cannot_iterate_until_explicit_reopen_and_history_survives(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)
    evolution_id, completed = complete_first(service)
    accepted = service.accept(evolution_id, completed.version)

    with pytest.raises(InvalidEvolutionTransition):
        service.reserve_episode(evolution_id, mode="CARRY_VERIFIED_EVIDENCE")

    reopened = service.reopen(evolution_id)
    assert reopened.status == "AWAITING_EXPERT_FEEDBACK"
    assert reopened.accepted_version == accepted.accepted_version == "v001"
    assert reopened.episode_ids == accepted.episode_ids
    assert reopened.last_completed_version == "v001"


def test_every_disallowed_transition_is_rejected_by_the_transition_guard() -> None:
    statuses = set(ALLOWED_TRANSITIONS)
    for source, allowed in ALLOWED_TRANSITIONS.items():
        for target in statuses - allowed:
            with pytest.raises(InvalidEvolutionTransition):
                EvolutionService.validate_transition(source, target)


def test_mutations_validate_episode_state_and_target_version(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    task = service.create_task(goal="goal", target=TargetSpec(goal="goal"))
    reserved = service.reserve_episode(task.evolution_id, mode="NORMAL")

    with pytest.raises(InvalidEvolutionTransition, match="version"):
        service.mark_episode_running(task.evolution_id, "v002")
    with pytest.raises(InvalidEvolutionTransition, match="RUNNING"):
        service.complete_episode(
            task.evolution_id,
            reserved.version,
            result=completed_result(reserved),
        )
    with pytest.raises(InvalidEvolutionTransition):
        service.stop(task.evolution_id)
    with pytest.raises(InvalidEvolutionTransition):
        service.reopen(task.evolution_id)


def test_async_event_sink_is_only_run_when_caller_explicitly_publishes(
    tmp_path: Path,
) -> None:
    received: list[RuntimeEvent] = []

    async def sink(event: RuntimeEvent) -> None:
        received.append(event)

    service = EvolutionService(EvolutionStore(Workspace(tmp_path)), event_sink=sink)
    task = service.create_task(goal="x" * 300, target=TargetSpec(goal="goal"))

    assert received == []
    events = service.drain_events()
    assert events[0].kind == "evolution_task_created"
    assert len(events[0].goal_summary) == 240  # type: ignore[attr-defined]

    import asyncio

    asyncio.run(service.publish_events(events))
    assert received == events
    assert service.get(task.evolution_id).status == "CREATED"


def test_concurrent_service_mutations_surface_optimistic_revision_conflict(
    tmp_path: Path,
) -> None:
    barrier = Barrier(2)
    count_lock = Lock()

    class BarrierStore(EvolutionStore):
        load_count = 0

        def load_task(self, evolution_id: str):  # type: ignore[no-untyped-def]
            task = super().load_task(evolution_id)
            with count_lock:
                self.load_count += 1
                wait = self.load_count <= 2
            if wait:
                barrier.wait()
            return task

    store = BarrierStore(Workspace(tmp_path))
    service = EvolutionService(store)
    task = service.create_task(goal="goal", target=TargetSpec(goal="goal"))

    def stop() -> object:
        try:
            return service.stop(task.evolution_id)
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: stop(), range(2)))

    assert sum(not isinstance(result, Exception) for result in results) == 1
    assert sum(isinstance(result, EvolutionConflictError) for result in results) == 1
