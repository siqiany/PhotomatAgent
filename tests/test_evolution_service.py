from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
from pathlib import Path
from threading import Barrier

import pytest

from photomatagent.runtime.events import RuntimeEvent
from photomatagent.scientific.evolution.models import (
    ArtifactRef,
    EpisodeRecord,
    ExpertFeedbackDraft,
    FeedbackCompilation,
    FeedbackDelta,
    RevisionPlan,
    RubricScores,
    StrategyVersion,
)
from photomatagent.scientific.evolution.revision import build_revision_plan
from photomatagent.scientific.evolution.service import (
    ALLOWED_TRANSITIONS,
    ArtifactMismatchError,
    EvolutionOperationConflict,
    EvolutionService,
    InvalidEvolutionTransition,
    MutationResult,
)
from photomatagent.scientific.evolution.store import (
    EvolutionConflictError,
    EvolutionStore,
)
from photomatagent.scientific.evolution.strategy import FixedStrategySelector
from photomatagent.scientific.loop import ScientificLoopSummary, TargetSpec
from photomatagent.workspace import Workspace


def make_service(tmp_path: Path) -> EvolutionService:
    return EvolutionService(EvolutionStore(Workspace(tmp_path)))


def mutated(result: MutationResult[object]):
    return result.entity


def test_service_contract_is_exported_from_evolution_package() -> None:
    from photomatagent.scientific.evolution import (
        ArtifactMismatchError as ExportedArtifactMismatchError,
        EvolutionService as ExportedEvolutionService,
        EvolutionOperationConflict as ExportedEvolutionOperationConflict,
        InvalidEvolutionTransition as ExportedInvalidEvolutionTransition,
        MutationResult as ExportedMutationResult,
    )

    assert ExportedEvolutionService is EvolutionService
    assert ExportedEvolutionOperationConflict is EvolutionOperationConflict
    assert ExportedArtifactMismatchError is ArtifactMismatchError
    assert ExportedInvalidEvolutionTransition is InvalidEvolutionTransition
    assert ExportedMutationResult is MutationResult


def completed_result(
    service: EvolutionService,
    episode: EpisodeRecord,
    *,
    content: bytes = b"result data",
    declared_digest: str | None = None,
) -> EpisodeRecord:
    relative = f"user_output/{episode.evolution_id}/{episode.version}/result.md"
    path = service.store.workspace.resolve(relative, must_exist=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
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
                path=relative,
                size_bytes=len(content),
                sha256=declared_digest or hashlib.sha256(content).hexdigest(),
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
    task = mutated(service.create_task(
        goal="find a stable infrared absorber",
        target=TargetSpec(goal="find a stable infrared absorber"),
    ))
    assert isinstance(task, object)
    episode = mutated(service.reserve_episode(task.evolution_id, mode="NORMAL"))
    running = mutated(service.mark_episode_running(task.evolution_id, episode.version))
    completed = mutated(service.complete_episode(
        task.evolution_id,
        running.version,
        result=completed_result(service, running),
    ))
    return task.evolution_id, completed


def prepare_revision(service: EvolutionService) -> tuple[str, EpisodeRecord]:
    evolution_id, completed = complete_first(service)
    feedback = mutated(service.attach_feedback(
        evolution_id,
        completed.version,
        feedback_id="fb_test",
        draft=feedback_draft(),
        result_sha256=completed.artifact.sha256,  # type: ignore[union-attr]
        raw_input='{"scores": "confirmed by expert"}',
    ))
    compilation = save_available_compilation(
        service, evolution_id, feedback.feedback_id, completed.version
    )
    plan, strategy = canonical_confirmation(service, evolution_id, compilation)
    mutated(service.confirm_revision(evolution_id, plan, strategy=strategy))
    return evolution_id, completed


def save_available_compilation(
    service: EvolutionService,
    evolution_id: str,
    feedback_id: str,
    version: str,
) -> FeedbackCompilation:
    return service.save_compilation(
        evolution_id,
        FeedbackCompilation(
            compilation_id=f"comp_{feedback_id}",
            evolution_id=evolution_id,
            feedback_id=feedback_id,
            episode_version=version,
            status="AVAILABLE",
            provider="fake",
            model="fake",
        ),
    ).entity


def canonical_confirmation(
    service: EvolutionService,
    evolution_id: str,
    compilation: FeedbackCompilation,
) -> tuple[RevisionPlan, StrategyVersion]:
    task, episode, feedback = service.compilation_context(
        evolution_id,
        compilation.episode_version,
    )
    plan = build_revision_plan(
        feedback=feedback,
        compilation=compilation,
        target=episode.target_snapshot,
        previous_summary=episode.summary,
    )
    return (
        plan.model_copy(update={"confirmed": True}),
        FixedStrategySelector().select(task, plan),
    )


def compiled_revision_context(
    service: EvolutionService,
    *,
    critical_without_contract: bool = False,
) -> tuple[str, EpisodeRecord, FeedbackCompilation]:
    evolution_id, completed = complete_first(service)
    feedback = mutated(service.attach_feedback(
        evolution_id,
        completed.version,
        feedback_id="fb_canonical_context",
        draft=feedback_draft(),
        result_sha256=completed.artifact.sha256,  # type: ignore[union-attr]
        raw_input="review",
    ))
    compilation = FeedbackCompilation(
        compilation_id="comp_canonical_context",
        evolution_id=evolution_id,
        feedback_id=feedback.feedback_id,
        episode_version=completed.version,
        status="AVAILABLE",
        items=(
            FeedbackDelta(
                item_id="item_001",
                category="EVIDENCE_SUFFICIENCY",
                status="CORRECTION",
                severity="CRITICAL" if critical_without_contract else "HIGH",
                responsible_module="retrieval",
                problem="Evidence is insufficient",
                requested_actions=(
                    () if critical_without_contract else ("Read full text",)
                ),
                acceptance_test=(
                    None if critical_without_contract else "Full-text evidence is bound"
                ),
                confidence=1.0,
                source_span="source",
            ),
        ),
        provider="fake",
        model="fake",
    )
    return evolution_id, completed, service.save_compilation(
        evolution_id,
        compilation,
    ).entity


def test_confirm_revision_rebuilds_canonical_plan_before_any_write(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)
    evolution_id, _completed, compilation = compiled_revision_context(service)
    plan, strategy = canonical_confirmation(service, evolution_id, compilation)
    forged = plan.model_copy(update={"contract_changes": ["forged change"]})

    with pytest.raises(EvolutionOperationConflict, match="canonical"):
        service.confirm_revision(evolution_id, forged, strategy=strategy)

    assert service.get(evolution_id).status == "FEEDBACK_RECORDED"
    assert service.store.list_revisions(evolution_id) == []
    assert service.store.list_strategies(evolution_id) == []


def test_confirm_revision_cannot_hide_canonical_critical_ambiguity(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)
    evolution_id, _completed, compilation = compiled_revision_context(
        service,
        critical_without_contract=True,
    )
    canonical, strategy = canonical_confirmation(service, evolution_id, compilation)
    assert canonical.has_blocking_ambiguity is True
    forged = canonical.model_copy(
        update={"has_blocking_ambiguity": False, "unresolved_ambiguities": []}
    )

    with pytest.raises(EvolutionOperationConflict, match="canonical"):
        service.confirm_revision(evolution_id, forged, strategy=strategy)

    assert service.get(evolution_id).status == "FEEDBACK_RECORDED"
    assert service.store.list_revisions(evolution_id) == []
    assert service.store.list_strategies(evolution_id) == []


@pytest.mark.parametrize("forgery", ["strategy_id", "strategy_sha256", "parameters"])
def test_confirm_revision_rejects_forged_strategy_before_any_write(
    forgery: str,
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)
    evolution_id, _completed, compilation = compiled_revision_context(service)
    plan, canonical = canonical_confirmation(service, evolution_id, compilation)
    updates = {
        "strategy_id": {"strategy_id": "strategy_forged"},
        "strategy_sha256": {"strategy_sha256": "f" * 64},
        "parameters": {
            "parameters": {
                "selector": "fixed-v1",
                "revision_id": plan.revision_id,
                "forged": True,
            }
        },
    }
    forged = canonical.model_copy(update=updates[forgery])

    with pytest.raises(EvolutionOperationConflict, match="canonical"):
        service.confirm_revision(evolution_id, plan, strategy=forged)

    assert service.get(evolution_id).status == "FEEDBACK_RECORDED"
    assert service.store.list_revisions(evolution_id) == []
    assert service.store.list_strategies(evolution_id) == []


@pytest.mark.parametrize(
    ("forged_field", "forged_value"),
    [("strategy_id", "strategy_forged"), ("strategy_arm", "DIVERSITY_FIRST")],
)
def test_revised_reservation_binds_latest_confirmed_strategy_and_checks_retry(
    forged_field: str,
    forged_value: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = make_service(tmp_path)
    evolution_id, _completed, compilation = compiled_revision_context(service)
    plan, strategy = canonical_confirmation(service, evolution_id, compilation)
    service.confirm_revision(evolution_id, plan, strategy=strategy)

    reserved = service.reserve_episode(
        evolution_id,
        mode="CARRY_VERIFIED_EVIDENCE",
    ).entity
    assert reserved.strategy_id == strategy.strategy_id
    assert reserved.strategy_arm == strategy.arm

    original_load = service.store.load_episode

    def load_with_forged_strategy(selected_id, version):  # type: ignore[no-untyped-def]
        episode = original_load(selected_id, version)
        if version == reserved.version:
            return episode.model_copy(update={forged_field: forged_value})
        return episode

    monkeypatch.setattr(service.store, "load_episode", load_with_forged_strategy)
    with pytest.raises(EvolutionOperationConflict, match="reservation"):
        service.reserve_episode(
            evolution_id,
            mode="CARRY_VERIFIED_EVIDENCE",
        )


def test_confirm_revision_atomically_persists_exact_plan_and_strategy(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)
    evolution_id, completed = complete_first(service)
    feedback = mutated(service.attach_feedback(
        evolution_id,
        completed.version,
        feedback_id="fb_atomic_revision",
        draft=feedback_draft(),
        result_sha256=completed.artifact.sha256,  # type: ignore[union-attr]
        raw_input="review",
    ))
    compilation = save_available_compilation(
        service, evolution_id, feedback.feedback_id, completed.version
    )
    plan, strategy = canonical_confirmation(service, evolution_id, compilation)

    result = service.confirm_revision(evolution_id, plan, strategy=strategy)
    repeated = service.confirm_revision(evolution_id, plan, strategy=strategy)

    assert result.entity == repeated.entity
    task = service.get(evolution_id)
    assert task.status == "REVISION_READY"
    assert task.revision_ids == [plan.revision_id]
    assert task.strategy_ids == [strategy.strategy_id]
    assert service.store.load_revision(evolution_id, plan.revision_id) == result.entity
    assert service.store.load_strategy(evolution_id, strategy.strategy_id) == strategy


def test_confirm_revision_rejects_strategy_mismatch_without_writing_either_record(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)
    evolution_id, completed = complete_first(service)
    feedback = mutated(service.attach_feedback(
        evolution_id,
        completed.version,
        feedback_id="fb_strategy_mismatch",
        draft=feedback_draft(),
        result_sha256=completed.artifact.sha256,  # type: ignore[union-attr]
    ))
    compilation = save_available_compilation(
        service, evolution_id, feedback.feedback_id, completed.version
    )
    plan, strategy = canonical_confirmation(service, evolution_id, compilation)
    mismatch = strategy.model_copy(update={"reason": "wrong"})

    with pytest.raises(EvolutionOperationConflict, match="strategy"):
        service.confirm_revision(evolution_id, plan, strategy=mismatch)

    task = service.get(evolution_id)
    assert task.status == "FEEDBACK_RECORDED"
    assert task.revision_ids == []
    assert task.strategy_ids == []
    assert service.store.list_revisions(evolution_id) == []
    assert service.store.list_strategies(evolution_id) == []


def test_confirm_revision_requires_available_active_compilation(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    evolution_id, completed = complete_first(service)
    feedback = mutated(service.attach_feedback(
        evolution_id,
        completed.version,
        feedback_id="fb_without_compilation",
        draft=feedback_draft(),
        result_sha256=completed.artifact.sha256,  # type: ignore[union-attr]
    ))
    plan = RevisionPlan(
        revision_id="rp_without_compilation",
        evolution_id=evolution_id,
        source_version=completed.version,
        feedback_id=feedback.feedback_id,
        confirmed=True,
    )

    with pytest.raises(InvalidEvolutionTransition, match="AVAILABLE compilation"):
        service.confirm_revision(evolution_id, plan)

    assert service.get(evolution_id).status == "FEEDBACK_RECORDED"


def test_confirm_revision_recovers_plan_write_before_strategy_write_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = make_service(tmp_path)
    evolution_id, completed = complete_first(service)
    feedback = mutated(service.attach_feedback(
        evolution_id,
        completed.version,
        feedback_id="fb_strategy_crash",
        draft=feedback_draft(),
        result_sha256=completed.artifact.sha256,  # type: ignore[union-attr]
    ))
    compilation = save_available_compilation(
        service, evolution_id, feedback.feedback_id, completed.version
    )
    plan, _strategy = canonical_confirmation(service, evolution_id, compilation)
    original = service.store._write_record_locked
    failed = False

    def fail_first_strategy_write(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal failed
        if kwargs["directory"] == "strategies" and not failed:
            failed = True
            raise OSError("strategy write interrupted")
        return original(**kwargs)

    monkeypatch.setattr(service.store, "_write_record_locked", fail_first_strategy_write)

    with pytest.raises(OSError, match="strategy write interrupted"):
        service.confirm_revision(evolution_id, plan)
    assert len(service.store.list_revisions(evolution_id)) == 1
    assert service.store.list_strategies(evolution_id) == []
    assert service.get(evolution_id).status == "FEEDBACK_RECORDED"

    recovered = service.confirm_revision(evolution_id, plan)

    task = service.get(evolution_id)
    assert recovered.entity.revision_id == plan.revision_id
    assert task.status == "REVISION_READY"
    assert task.revision_ids == [plan.revision_id]
    assert len(task.strategy_ids) == 1


def test_lifecycle_requires_feedback_before_next_episode(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    evolution_id, _ = complete_first(service)

    assert service.get(evolution_id).status == "AWAITING_EXPERT_FEEDBACK"
    with pytest.raises(InvalidEvolutionTransition):
        service.reserve_episode(evolution_id, mode="CARRY_VERIFIED_EVIDENCE")


def test_failed_next_episode_never_overwrites_last_good_result(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    evolution_id, first = prepare_revision(service)
    second = mutated(service.reserve_episode(
        evolution_id,
        mode="CARRY_VERIFIED_EVIDENCE",
    ))
    mutated(service.fail_episode(evolution_id, second.version, "provider failed"))

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
    task = mutated(service.create_task(goal="goal", target=TargetSpec(goal="goal")))
    episode = mutated(service.reserve_episode(task.evolution_id, mode="NORMAL"))
    running = mutated(service.mark_episode_running(task.evolution_id, episode.version))
    result = completed_result(service, running)
    result.summary.status = inner_status  # type: ignore[assignment,union-attr]

    mutated(service.complete_episode(task.evolution_id, running.version, result=result))

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
            feedback_id="fb_wrong_version",
            draft=feedback_draft(),
            result_sha256=digest,
            raw_input="review",
        )
    with pytest.raises(ArtifactMismatchError):
        service.attach_feedback(
            evolution_id,
            completed.version,
            feedback_id="fb_wrong_hash",
            draft=feedback_draft(),
            result_sha256="0" * 64,
            raw_input="review",
        )

    mutated(service.attach_feedback(
        evolution_id,
        completed.version,
        feedback_id="fb_active",
        draft=feedback_draft(),
        result_sha256=digest,
        raw_input="review",
    ))
    with pytest.raises(EvolutionOperationConflict):
        service.attach_feedback(
            evolution_id,
            completed.version,
            feedback_id="fb_second",
            draft=feedback_draft(),
            result_sha256=digest,
            raw_input="second review",
        )


def test_unconfirmed_or_mismatched_revision_is_never_persisted(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    evolution_id, completed = complete_first(service)
    feedback = mutated(service.attach_feedback(
        evolution_id,
        completed.version,
        feedback_id="fb_test",
        draft=feedback_draft(),
        result_sha256=completed.artifact.sha256,  # type: ignore[union-attr]
        raw_input="review",
    ))

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
    feedback = mutated(service.attach_feedback(
        evolution_id,
        completed.version,
        feedback_id="fb_test",
        draft=feedback_draft(),
        result_sha256=completed.artifact.sha256,  # type: ignore[union-attr]
        raw_input="review",
    ))
    compilation = FeedbackCompilation(
        compilation_id="comp_ambiguous",
        evolution_id=evolution_id,
        feedback_id=feedback.feedback_id,
        episode_version=completed.version,
        status="AVAILABLE",
        items=(
            FeedbackDelta(
                item_id="item_ambiguous",
                category="OTHER",
                status="CORRECTION",
                severity="CRITICAL",
                responsible_module="method",
                problem="Which measurement protocol applies?",
                confidence=1.0,
                source_span="source",
            ),
        ),
        provider="fake",
        model="fake",
    )
    service.save_compilation(evolution_id, compilation)
    plan, strategy = canonical_confirmation(service, evolution_id, compilation)

    with pytest.raises(InvalidEvolutionTransition, match="ambiguity"):
        service.confirm_revision(evolution_id, plan, strategy=strategy)

    assert service.get(evolution_id).status == "FEEDBACK_RECORDED"
    assert service.get(evolution_id).revision_ids == []


def test_accepted_task_cannot_iterate_until_explicit_reopen_and_history_survives(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)
    evolution_id, completed = complete_first(service)
    accepted = mutated(service.accept(evolution_id, completed.version))

    with pytest.raises(InvalidEvolutionTransition):
        service.reserve_episode(evolution_id, mode="CARRY_VERIFIED_EVIDENCE")

    reopened = mutated(service.reopen(evolution_id))
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
    task = mutated(service.create_task(goal="goal", target=TargetSpec(goal="goal")))
    reserved = mutated(service.reserve_episode(task.evolution_id, mode="NORMAL"))

    with pytest.raises(InvalidEvolutionTransition, match="version"):
        service.mark_episode_running(task.evolution_id, "v002")
    with pytest.raises(InvalidEvolutionTransition, match="running"):
        service.complete_episode(
            task.evolution_id,
            reserved.version,
            result=completed_result(service, reserved),
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
    outcome = service.create_task(goal="x" * 300, target=TargetSpec(goal="goal"))
    task = outcome.entity

    assert received == []
    events = outcome.events
    assert events[0].kind == "evolution_task_created"
    assert len(events[0].goal_summary) == 240  # type: ignore[attr-defined]

    import asyncio

    asyncio.run(service.publish(events))
    assert received == list(events)
    assert service.get(task.evolution_id).status == "CREATED"


def test_concurrent_stop_and_reserve_are_serialized_as_one_logical_mutation(
    tmp_path: Path,
) -> None:
    barrier = Barrier(2)

    class BarrierStore(EvolutionStore):
        @contextmanager
        def transaction(self, evolution_id: str):  # type: ignore[no-untyped-def]
            barrier.wait()
            with super().transaction(evolution_id) as transaction:
                yield transaction

    store = BarrierStore(Workspace(tmp_path))
    service = EvolutionService(store)
    task = mutated(service.create_task(goal="goal", target=TargetSpec(goal="goal")))

    def mutate(operation: str) -> object:
        try:
            if operation == "stop":
                return service.stop(task.evolution_id)
            return service.reserve_episode(task.evolution_id, mode="NORMAL")
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(mutate, ["stop", "reserve"]))

    assert sum(not isinstance(result, Exception) for result in results) == 1
    assert sum(isinstance(result, InvalidEvolutionTransition) for result in results) == 1


@pytest.mark.parametrize(
    "checkpoint",
    ["CREATED", "FEEDBACK_RECORDED", "REVISION_READY"],
)
def test_stop_and_reopen_restore_exact_checkpoint(
    tmp_path: Path, checkpoint: str
) -> None:
    service = make_service(tmp_path)
    if checkpoint == "CREATED":
        task = mutated(service.create_task(goal="goal", target=TargetSpec(goal="goal")))
    else:
        evolution_id, completed = complete_first(service)
        feedback = mutated(service.attach_feedback(
            evolution_id,
            completed.version,
            feedback_id="fb_checkpoint",
            draft=feedback_draft(),
            result_sha256=completed.artifact.sha256,  # type: ignore[union-attr]
            raw_input="review",
        ))
        task = service.get(evolution_id)
        if checkpoint == "REVISION_READY":
            compilation = save_available_compilation(
                service,
                evolution_id,
                feedback.feedback_id,
                completed.version,
            )
            plan, strategy = canonical_confirmation(service, evolution_id, compilation)
            mutated(service.confirm_revision(evolution_id, plan, strategy=strategy))
            task = service.get(evolution_id)

    stopped = mutated(service.stop(task.evolution_id))
    reopened = mutated(service.reopen(task.evolution_id))

    assert stopped.resume_status == checkpoint
    assert reopened.status == checkpoint
    assert reopened.resume_status is None


def test_failed_episode_reopens_to_retry_checkpoint(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    task = mutated(service.create_task(goal="goal", target=TargetSpec(goal="goal")))
    first = mutated(service.reserve_episode(task.evolution_id, mode="NORMAL"))
    failed = mutated(service.fail_episode(task.evolution_id, first.version, "failed"))

    blocked = service.get(task.evolution_id)
    assert failed.status == "FAILED"
    assert blocked.resume_status == "CREATED"
    reopened = mutated(service.reopen(task.evolution_id))
    retry = mutated(service.reserve_episode(task.evolution_id, mode="NORMAL"))
    assert reopened.status == "CREATED"
    assert retry.version == "v002"


def test_completion_rejects_missing_or_forged_artifact(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    task = mutated(service.create_task(goal="goal", target=TargetSpec(goal="goal")))
    episode = mutated(service.reserve_episode(task.evolution_id, mode="NORMAL"))
    running = mutated(service.mark_episode_running(task.evolution_id, episode.version))
    missing = running.model_copy(
        update={
            "artifact": ArtifactRef(
                path="user_output/missing/result.md",
                size_bytes=1,
                sha256="a" * 64,
            )
        }
    )

    with pytest.raises(ArtifactMismatchError):
        service.complete_episode(task.evolution_id, running.version, result=missing)
    forged = completed_result(service, running, declared_digest="a" * 64)
    with pytest.raises(ArtifactMismatchError):
        service.complete_episode(task.evolution_id, running.version, result=forged)


def test_feedback_and_accept_reverify_artifact_bytes(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    evolution_id, completed = complete_first(service)
    path = service.store.workspace.resolve(completed.artifact.path)  # type: ignore[union-attr]
    path.write_text("tampered", encoding="utf-8")

    with pytest.raises(ArtifactMismatchError):
        service.attach_feedback(
            evolution_id,
            completed.version,
            feedback_id="fb_tampered",
            draft=feedback_draft(),
            result_sha256=completed.artifact.sha256,  # type: ignore[union-attr]
            raw_input="review",
        )
    with pytest.raises(ArtifactMismatchError):
        service.accept(evolution_id, completed.version)


def test_accept_can_select_an_older_completed_episode(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    evolution_id, first = prepare_revision(service)
    second = mutated(service.reserve_episode(
        evolution_id, mode="CARRY_VERIFIED_EVIDENCE"
    ))
    running = mutated(service.mark_episode_running(evolution_id, second.version))
    mutated(service.complete_episode(
        evolution_id,
        second.version,
        result=completed_result(service, running, content=b"second result"),
    ))

    accepted = mutated(service.accept(evolution_id, first.version))

    assert accepted.accepted_version == "v001"


def test_terminal_transition_preserves_started_runtime_provenance(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    task = mutated(service.create_task(goal="goal", target=TargetSpec(goal="goal")))
    episode = mutated(service.reserve_episode(task.evolution_id, mode="NORMAL"))
    running = mutated(service.mark_episode_running(
        task.evolution_id,
        episode.version,
        runtime_session_id="session_one",
        event_log_path=".photomatagent/sessions/session_one/events.jsonl",
    ))
    forged = completed_result(service, running).model_copy(
        update={"runtime_session_id": "session_two"}
    )

    with pytest.raises(EvolutionOperationConflict, match="runtime_session_id"):
        service.complete_episode(task.evolution_id, running.version, result=forged)


def test_reservation_reconciles_matching_record_after_manifest_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make_service(tmp_path)
    task = mutated(service.create_task(goal="goal", target=TargetSpec(goal="goal")))
    original = service.store._save_task_locked
    calls = 0

    def fail_once(candidate, expected_revision):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("manifest crash")
        return original(candidate, expected_revision)

    monkeypatch.setattr(service.store, "_save_task_locked", fail_once)
    with pytest.raises(OSError, match="manifest crash"):
        service.reserve_episode(task.evolution_id, mode="NORMAL")

    retry = service.reserve_episode(task.evolution_id, mode="NORMAL")
    assert retry.entity.version == "v001"
    assert len(service.get(task.evolution_id).episode_ids) == 1


def test_feedback_retry_requires_same_stable_id_and_reconciles_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make_service(tmp_path)
    evolution_id, completed = complete_first(service)
    original = service.store._save_task_locked
    calls = 0

    def fail_once(candidate, expected_revision):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("manifest crash")
        return original(candidate, expected_revision)

    monkeypatch.setattr(service.store, "_save_task_locked", fail_once)
    kwargs = {
        "draft": feedback_draft(),
        "result_sha256": completed.artifact.sha256,  # type: ignore[union-attr]
        "raw_input": "review",
    }
    with pytest.raises(OSError, match="manifest crash"):
        service.attach_feedback(
            evolution_id, completed.version, feedback_id="fb_stable", **kwargs
        )

    with pytest.raises(EvolutionOperationConflict):
        service.attach_feedback(
            evolution_id, completed.version, feedback_id="fb_different", **kwargs
        )
    retry = service.attach_feedback(
        evolution_id, completed.version, feedback_id="fb_stable", **kwargs
    )
    assert retry.entity.feedback_id == "fb_stable"
    assert service.get(evolution_id).feedback_ids == ["fb_stable"]


def test_feedback_retry_compares_the_redacted_persisted_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make_service(tmp_path)
    evolution_id, completed = complete_first(service)
    original = service.store._save_task_locked
    calls = 0

    def fail_once(candidate, expected_revision):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("manifest crash")
        return original(candidate, expected_revision)

    monkeypatch.setattr(service.store, "_save_task_locked", fail_once)
    kwargs = {
        "feedback_id": "fb_secret_retry",
        "draft": feedback_draft(),
        "result_sha256": completed.artifact.sha256,  # type: ignore[union-attr]
        "raw_input": "Authorization: Bearer super-secret-value",
    }
    with pytest.raises(OSError):
        service.attach_feedback(evolution_id, completed.version, **kwargs)

    retry = service.attach_feedback(evolution_id, completed.version, **kwargs)
    assert "super-secret-value" not in retry.entity.raw_input


def test_completion_retry_reconciles_terminal_record_after_manifest_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make_service(tmp_path)
    task = mutated(service.create_task(goal="goal", target=TargetSpec(goal="goal")))
    episode = mutated(service.reserve_episode(task.evolution_id, mode="NORMAL"))
    running = mutated(service.mark_episode_running(task.evolution_id, episode.version))
    result = completed_result(service, running)
    original = service.store._save_task_locked
    calls = 0

    def fail_once(candidate, expected_revision):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("manifest crash")
        return original(candidate, expected_revision)

    monkeypatch.setattr(service.store, "_save_task_locked", fail_once)
    with pytest.raises(OSError, match="manifest crash"):
        service.complete_episode(task.evolution_id, running.version, result=result)

    retry = service.complete_episode(task.evolution_id, running.version, result=result)
    assert retry.entity.status == "COMPLETED"
    assert service.get(task.evolution_id).status == "AWAITING_EXPERT_FEEDBACK"


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("episode_id", "ep_forged"),
        ("capability_fingerprint", "f" * 64),
    ],
)
def test_completion_recovery_rejects_forged_frozen_episode_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
) -> None:
    service = make_service(tmp_path)
    task = mutated(service.create_task(goal="goal", target=TargetSpec(goal="goal")))
    episode = mutated(service.reserve_episode(task.evolution_id, mode="NORMAL"))
    running = mutated(service.mark_episode_running(task.evolution_id, episode.version))
    result = completed_result(service, running)
    original = service.store._save_task_locked
    calls = 0

    def fail_once(candidate, expected_revision):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("manifest crash")
        return original(candidate, expected_revision)

    monkeypatch.setattr(service.store, "_save_task_locked", fail_once)
    with pytest.raises(OSError, match="manifest crash"):
        service.complete_episode(task.evolution_id, running.version, result=result)

    forged = result.model_copy(update={field: replacement})
    with pytest.raises(EvolutionOperationConflict, match=field):
        service.complete_episode(task.evolution_id, running.version, result=forged)

    assert service.get(task.evolution_id).status == "RUNNING"


def test_failure_retry_reconciles_terminal_record_after_manifest_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make_service(tmp_path)
    task = mutated(service.create_task(goal="goal", target=TargetSpec(goal="goal")))
    episode = mutated(service.reserve_episode(task.evolution_id, mode="NORMAL"))
    original = service.store._save_task_locked
    calls = 0

    def fail_once(candidate, expected_revision):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("manifest crash")
        return original(candidate, expected_revision)

    monkeypatch.setattr(service.store, "_save_task_locked", fail_once)
    with pytest.raises(OSError, match="manifest crash"):
        service.fail_episode(task.evolution_id, episode.version, "provider failed")

    retry = service.fail_episode(task.evolution_id, episode.version, "provider failed")
    assert retry.entity.status == "FAILED"
    assert service.get(task.evolution_id).status == "BLOCKED"


def test_restart_reconciles_completed_split_once_and_feedback_can_continue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make_service(tmp_path)
    task = mutated(service.create_task(goal="goal", target=TargetSpec(goal="goal")))
    episode = mutated(service.reserve_episode(task.evolution_id, mode="NORMAL"))
    running = mutated(service.mark_episode_running(task.evolution_id, episode.version))
    result = completed_result(service, running)

    def crash_after_episode_write(candidate, expected_revision):  # type: ignore[no-untyped-def]
        raise OSError("process died before task manifest write")

    monkeypatch.setattr(service.store, "_save_task_locked", crash_after_episode_write)
    with pytest.raises(OSError, match="process died"):
        service.complete_episode(task.evolution_id, running.version, result=result)
    assert service.store.load_episode(task.evolution_id, running.version).status == "COMPLETED"
    assert service.get(task.evolution_id).status == "RUNNING"

    restarted_store = EvolutionStore(Workspace(tmp_path))
    restarted = EvolutionService(restarted_store)
    repaired = restarted.reconcile(task.evolution_id)

    assert repaired.entity.status == "AWAITING_EXPERT_FEEDBACK"
    assert repaired.entity.last_completed_version == "v001"
    assert [event.kind for event in repaired.events] == ["evolution_episode_completed"]
    revision = repaired.entity.revision

    repeated = restarted.reconcile(task.evolution_id)
    assert repeated.entity.revision == revision
    assert repeated.events == ()

    completed = restarted_store.load_episode(task.evolution_id, "v001")
    feedback = restarted.attach_feedback(
        task.evolution_id,
        "v001",
        feedback_id="fb_after_restart",
        draft=feedback_draft(),
        result_sha256=completed.artifact.sha256,  # type: ignore[union-attr]
        raw_input="post-restart review",
    )
    assert feedback.entity.feedback_id == "fb_after_restart"
    assert restarted.get(task.evolution_id).status == "FEEDBACK_RECORDED"


def test_restart_reconciles_failed_split_preserving_last_good_then_reopens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make_service(tmp_path)
    evolution_id, first = prepare_revision(service)
    second = mutated(service.reserve_episode(
        evolution_id,
        mode="CARRY_VERIFIED_EVIDENCE",
    ))
    running = mutated(service.mark_episode_running(evolution_id, second.version))

    def crash_after_episode_write(candidate, expected_revision):  # type: ignore[no-untyped-def]
        raise OSError("process died before task manifest write")

    monkeypatch.setattr(service.store, "_save_task_locked", crash_after_episode_write)
    with pytest.raises(OSError, match="process died"):
        service.fail_episode(evolution_id, running.version, "provider failed")
    assert service.store.load_episode(evolution_id, running.version).status == "FAILED"
    assert service.get(evolution_id).status == "RUNNING"

    restarted = EvolutionService(EvolutionStore(Workspace(tmp_path)))
    repaired = restarted.reconcile(evolution_id)

    assert repaired.entity.status == "BLOCKED"
    assert repaired.entity.resume_status == "REVISION_READY"
    assert repaired.entity.last_completed_version == first.version == "v001"
    revision = repaired.entity.revision
    repeated = restarted.reconcile(evolution_id)
    assert repeated.entity.revision == revision
    assert repeated.events == ()

    reopened = mutated(restarted.reopen(evolution_id))
    assert reopened.status == "REVISION_READY"
    retry = mutated(restarted.reserve_episode(
        evolution_id,
        mode="CARRY_VERIFIED_EVIDENCE",
    ))
    assert retry.status == "RESERVED"
    assert retry.version == "v003"
    assert retry.parent_version == "v001"


def test_reconcile_rejects_terminal_episode_not_owned_by_task_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make_service(tmp_path)
    task = mutated(service.create_task(goal="goal", target=TargetSpec(goal="goal")))
    episode = mutated(service.reserve_episode(task.evolution_id, mode="NORMAL"))
    running = mutated(service.mark_episode_running(task.evolution_id, episode.version))
    result = completed_result(service, running)

    def crash_after_episode_write(candidate, expected_revision):  # type: ignore[no-untyped-def]
        raise OSError("process died before task manifest write")

    monkeypatch.setattr(service.store, "_save_task_locked", crash_after_episode_write)
    with pytest.raises(OSError, match="process died"):
        service.complete_episode(task.evolution_id, running.version, result=result)

    restarted_store = EvolutionStore(Workspace(tmp_path))
    stale = restarted_store.load_task(task.evolution_id)
    restarted_store.save_task(
        stale.model_copy(update={"episode_ids": []}),
        expected_revision=stale.revision,
    )
    restarted = EvolutionService(restarted_store)

    with pytest.raises(EvolutionOperationConflict, match="current episode identity"):
        restarted.reconcile(task.evolution_id)
    assert restarted.get(task.evolution_id).status == "RUNNING"


def test_revision_retry_reconciles_matching_stable_id_after_manifest_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make_service(tmp_path)
    evolution_id, completed = complete_first(service)
    feedback = mutated(service.attach_feedback(
        evolution_id,
        completed.version,
        feedback_id="fb_revision",
        draft=feedback_draft(),
        result_sha256=completed.artifact.sha256,  # type: ignore[union-attr]
        raw_input="review",
    ))
    compilation = save_available_compilation(
        service,
        evolution_id,
        feedback.feedback_id,
        completed.version,
    )
    plan, strategy = canonical_confirmation(service, evolution_id, compilation)
    original = service.store._save_task_locked
    calls = 0

    def fail_once(candidate, expected_revision):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("manifest crash")
        return original(candidate, expected_revision)

    monkeypatch.setattr(service.store, "_save_task_locked", fail_once)
    with pytest.raises(OSError, match="manifest crash"):
        service.confirm_revision(evolution_id, plan, strategy=strategy)

    with pytest.raises(EvolutionOperationConflict):
        service.confirm_revision(
            evolution_id,
            plan.model_copy(update={"revision_id": "rp_different"}),
            strategy=strategy,
        )
    retry = service.confirm_revision(evolution_id, plan, strategy=strategy)
    assert retry.entity.revision_id == plan.revision_id
    assert service.get(evolution_id).revision_ids == [plan.revision_id]


def test_mismatched_orphan_reservation_is_a_typed_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make_service(tmp_path)
    task = mutated(service.create_task(goal="goal", target=TargetSpec(goal="goal")))

    monkeypatch.setattr(
        service.store,
        "_save_task_locked",
        lambda candidate, expected_revision: (_ for _ in ()).throw(
            OSError("manifest crash")
        ),
    )
    with pytest.raises(OSError):
        service.reserve_episode(task.evolution_id, mode="NORMAL", provider="provider_a")

    with pytest.raises(EvolutionOperationConflict):
        service.reserve_episode(task.evolution_id, mode="NORMAL", provider="provider_b")


def test_orphan_reservation_rejects_changed_execution_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make_service(tmp_path)
    task = mutated(service.create_task(goal="goal", target=TargetSpec(goal="goal")))
    original = service.store._save_task_locked

    monkeypatch.setattr(
        service.store,
        "_save_task_locked",
        lambda candidate, expected_revision: (_ for _ in ()).throw(
            OSError("manifest crash")
        ),
    )
    with pytest.raises(OSError):
        service.reserve_episode(
            task.evolution_id,
            mode="NORMAL",
            capability_fingerprint="a" * 64,
        )

    monkeypatch.setattr(service.store, "_save_task_locked", original)
    with pytest.raises(EvolutionOperationConflict):
        service.reserve_episode(
            task.evolution_id,
            mode="NORMAL",
            capability_fingerprint="b" * 64,
        )


def test_each_mutation_returns_only_its_own_events(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    created = service.create_task(goal="goal", target=TargetSpec(goal="goal"))
    stopped = service.stop(created.entity.evolution_id)

    assert [event.kind for event in created.events] == ["evolution_task_created"]
    assert [event.kind for event in stopped.events] == ["evolution_task_stopped"]
