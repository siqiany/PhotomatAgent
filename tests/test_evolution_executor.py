from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from photomatagent.errors import ProviderError
from photomatagent.logging.event_logger import EventLogger
from photomatagent.models.fake import FakeModelProvider, FakeResponse, scripted_tool_call
from photomatagent.models.types import (
    AssistantMessage,
    ModelRequest,
    ModelStreamEvent,
    ModelUsage,
)
from photomatagent.runtime.budget import BudgetState
from photomatagent.runtime.events import RuntimeEvent
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import AllowAllPolicy
from photomatagent.scientific.evolution.executor import ScientificEpisodeExecutor
from photomatagent.scientific.evolution.artifacts import (
    EpisodeResultAlreadyExistsError,
)
from photomatagent.scientific.evolution.models import (
    ArtifactRef,
    ExpertFeedbackDraft,
    FeedbackCompilation,
    RevisionPlan,
    RubricScores,
)
from photomatagent.scientific.evolution.revision import build_revision_plan
from photomatagent.scientific.evolution.service import EvolutionService
from photomatagent.scientific.evolution.store import EvolutionStore
from photomatagent.scientific.evolution.strategy import FixedStrategySelector
from photomatagent.scientific.loop import (
    ScientificLoopConfig,
    ScientificLoopSummary,
    TargetSpec,
)
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.registry import ToolRegistry
from photomatagent.tools.write import WriteTool
from photomatagent.workspace import Workspace


def _service(tmp_path: Path) -> EvolutionService:
    return EvolutionService(EvolutionStore(Workspace(tmp_path)))


def _reserved(service: EvolutionService):  # type: ignore[no-untyped-def]
    task = service.create_task(
        goal="Produce a reviewable report",
        target=TargetSpec(goal="Produce a reviewable report"),
        evolution_id="evo_test",
    ).entity
    episode = service.reserve_episode(task.evolution_id, mode="NORMAL").entity
    return task, episode


def _reserved_revision(service: EvolutionService):  # type: ignore[no-untyped-def]
    task, first = _reserved(service)
    running = service.mark_episode_running(task.evolution_id, first.version).entity
    content = b"first result"
    relative = "user_output/evo_test/v001/result.md"
    path = service.store.workspace.resolve(relative, must_exist=False)
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    completed = service.complete_episode(
        task.evolution_id,
        first.version,
        result=running.model_copy(
            update={
                "summary": ScientificLoopSummary(
                    status="INCONCLUSIVE",
                    rounds=1,
                    candidate_count=0,
                    best_candidate_id=None,
                    best_score=0.0,
                    final_evaluation=None,
                ),
                "artifact": ArtifactRef(
                    path=relative,
                    size_bytes=len(content),
                    sha256=hashlib.sha256(content).hexdigest(),
                ),
            }
        ),
    ).entity
    feedback = service.attach_feedback(
        task.evolution_id,
        first.version,
        feedback_id="fb_test",
        draft=ExpertFeedbackDraft(
            scores=RubricScores(
                scientific_correctness=3,
                evidence_sufficiency=3,
                novelty=3,
                actionability=3,
                overall=3,
            )
        ),
        result_sha256=completed.artifact.sha256,  # type: ignore[union-attr]
        raw_input="review",
    ).entity
    compilation = service.save_compilation(
        task.evolution_id,
        FeedbackCompilation(
            compilation_id="comp_test",
            evolution_id=task.evolution_id,
            feedback_id=feedback.feedback_id,
            episode_version=first.version,
            status="AVAILABLE",
            provider="fake",
            model="fake",
        ),
    ).entity
    plan = build_revision_plan(
        feedback=feedback,
        compilation=compilation,
        target=completed.target_snapshot,
        previous_summary=completed.summary,
    ).model_copy(update={"confirmed": True})
    strategy = FixedStrategySelector().select(service.get(task.evolution_id), plan)
    persisted_plan = service.confirm_revision(
        task.evolution_id, plan, strategy=strategy
    ).entity
    second = service.reserve_episode(
        task.evolution_id,
        mode="CARRY_VERIFIED_EVIDENCE",
    ).entity
    return service.get(task.evolution_id), second, persisted_plan


def _runtime(
    workspace: Workspace,
    model,  # type: ignore[no-untyped-def]
    *,
    session_id: str = "session_test",
    event_sinks=None,  # type: ignore[no-untyped-def]
) -> AgentRuntime:
    scientific = ScientificState()
    tools = ToolRegistry()
    tools.register(WriteTool(workspace))
    return AgentRuntime(
        model=model,
        tools=tools,
        workspace=workspace,
        scientific_state=scientific,
        permission_policy=AllowAllPolicy(),
        budget=BudgetState(max_iterations=10),
        event_sinks=event_sinks,
        session_id=session_id,
    )


def test_agent_runtime_exposes_read_only_session_id(tmp_path: Path) -> None:
    runtime = _runtime(
        Workspace(tmp_path),
        FakeModelProvider([FakeResponse(text="unused")]),
        session_id="session_public",
    )

    assert runtime.session_id == "session_public"
    with pytest.raises(AttributeError):
        runtime.session_id = "changed"  # type: ignore[misc]


@pytest.mark.asyncio
async def test_executor_runs_write_only_through_agent_runtime_and_completes_episode(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    task, episode = _reserved(service)
    logger = EventLogger(
        tmp_path / ".photomatagent/sessions", session_id="session_test"
    )
    model = FakeModelProvider(
        [
            scripted_tool_call(
                "write",
                {
                    "path": "user_output/evo_test/v001/result.md",
                    "content": "runtime-created report",
                },
                tool_call_id="call_result",
            ),
            FakeResponse(
                text="Report created.",
                usage=ModelUsage(input_tokens=7, output_tokens=3),
            ),
        ]
    )
    runtime = _runtime(service.store.workspace, model, event_sinks=[logger.log])
    observed: list[RuntimeEvent] = []
    executor = ScientificEpisodeExecutor(service.store, event_logger=logger)

    result = await executor.execute(
        task=task,
        episode=episode,
        runtime=runtime,
        config=ScientificLoopConfig(max_rounds=1),
        on_event=observed.append,
    )

    stored = service.store.load_episode(task.evolution_id, episode.version)
    assert result.runtime_session_id == "session_test"
    assert result.episode == stored
    assert stored.status == "COMPLETED"
    assert result.artifact.path == "user_output/evo_test/v001/result.md"
    assert service.store.workspace.resolve(result.artifact.path).read_text() == (
        "runtime-created report"
    )
    assert result.scientific_summary.status == "BUDGET_EXHAUSTED"
    assert result.scientific_state_path.endswith("v001.scientific.json")
    assert service.store.load_scientific_state("evo_test", "v001") == (
        runtime.scientific_state
    )
    assert result.cost.input_tokens == 7
    assert result.cost.output_tokens == 3
    assert result.cost.tool_calls == 1
    assert "tool_completed" in [event.kind for event in observed]
    assert "evolution_episode_started" in [event.kind for event in observed]
    assert "evolution_episode_completed" in [event.kind for event in observed]
    logged_kinds = [event.kind for event in logger.read_events()]
    assert logged_kinds.count("tool_completed") == 1
    assert logged_kinds.count("scientific_loop_completed") == 1


@pytest.mark.asyncio
async def test_executor_falls_back_to_final_assistant_text(tmp_path: Path) -> None:
    service = _service(tmp_path)
    task, episode = _reserved(service)
    runtime = _runtime(
        service.store.workspace,
        FakeModelProvider([FakeResponse(text="final report")]),
    )

    result = await ScientificEpisodeExecutor(service.store).execute(
        task=task,
        episode=episode,
        runtime=runtime,
        config=ScientificLoopConfig(max_rounds=1),
    )

    assert service.store.workspace.resolve(result.artifact.path).read_text() == (
        "final report\n"
    )


class FailingProvider:
    provider = "broken"
    model = "broken"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        raise ProviderError("broken", "boom")
        yield  # pragma: no cover


@pytest.mark.asyncio
async def test_failed_execution_marks_episode_failed_and_never_completes(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    task, episode = _reserved(service)
    runtime = _runtime(service.store.workspace, FailingProvider())
    observed: list[RuntimeEvent] = []

    with pytest.raises(ProviderError, match="boom"):
        await ScientificEpisodeExecutor(service.store).execute(
            task=task,
            episode=episode,
            runtime=runtime,
            config=ScientificLoopConfig(max_rounds=1),
            on_event=observed.append,
        )

    stored = service.store.load_episode(task.evolution_id, episode.version)
    assert stored.status == "FAILED"
    assert stored.artifact is None
    assert service.get(task.evolution_id).status == "BLOCKED"
    assert "evolution_episode_completed" not in [event.kind for event in observed]


@pytest.mark.asyncio
async def test_executor_rejects_runtime_from_another_workspace(tmp_path: Path) -> None:
    service = _service(tmp_path)
    task, episode = _reserved(service)
    other_root = tmp_path / "other"
    other_root.mkdir()
    runtime = _runtime(
        Workspace(other_root), FakeModelProvider([FakeResponse(text="result")])
    )

    with pytest.raises(ValueError, match="workspace"):
        await ScientificEpisodeExecutor(service.store).execute(
            task=task,
            episode=episode,
            runtime=runtime,
            config=ScientificLoopConfig(max_rounds=1),
        )

    assert service.store.load_episode("evo_test", "v001").status == "RESERVED"


@pytest.mark.asyncio
async def test_existing_result_collision_fails_episode_without_overwriting(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    task, episode = _reserved(service)
    path = service.store.workspace.resolve(
        "user_output/evo_test/v001/result.md", must_exist=False
    )
    path.parent.mkdir(parents=True)
    path.write_text("existing", encoding="utf-8")
    runtime = _runtime(
        service.store.workspace,
        FakeModelProvider([FakeResponse(text="replacement")]),
    )

    with pytest.raises(EpisodeResultAlreadyExistsError, match="already exists"):
        await ScientificEpisodeExecutor(service.store).execute(
            task=task,
            episode=episode,
            runtime=runtime,
            config=ScientificLoopConfig(max_rounds=1),
        )

    assert path.read_text(encoding="utf-8") == "existing"
    assert service.store.load_episode("evo_test", "v001").status == "FAILED"


@pytest.mark.asyncio
async def test_invalid_revision_fails_before_episode_is_marked_running(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    task, episode = _reserved(service)
    runtime = _runtime(
        service.store.workspace,
        FakeModelProvider([FakeResponse(text="unused")]),
    )
    invalid_revision = RevisionPlan(
        revision_id="rp_wrong",
        evolution_id="evo_test",
        source_version="v001",
        feedback_id="fb_wrong",
        confirmed=True,
    )

    with pytest.raises(ValueError, match="revision"):
        await ScientificEpisodeExecutor(service.store).execute(
            task=task,
            episode=episode,
            runtime=runtime,
            config=ScientificLoopConfig(max_rounds=1),
            revision=invalid_revision,
        )

    assert service.store.load_episode("evo_test", "v001").status == "RESERVED"


@pytest.mark.asyncio
async def test_revised_executor_rejects_strategy_tampering_before_runtime(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    task, episode, revision = _reserved_revision(service)
    assert episode.strategy_id is not None
    strategy_path = service.store.workspace.resolve(
        f".photomatagent/evolutions/{task.evolution_id}/strategies/"
        f"{episode.strategy_id}.json",
        must_exist=True,
    )
    payload = json.loads(strategy_path.read_text(encoding="utf-8"))
    payload["arm"] = "DIVERSITY_FIRST"
    strategy_path.write_text(json.dumps(payload), encoding="utf-8")
    runtime = _runtime(
        service.store.workspace,
        FakeModelProvider([FakeResponse(text="must not execute")]),
        session_id="session_tampered_strategy",
    )

    with pytest.raises(ValueError, match="strategy"):
        await ScientificEpisodeExecutor(service.store).execute(
            task=task,
            episode=episode,
            runtime=runtime,
            config=ScientificLoopConfig(max_rounds=1),
            revision=revision,
        )

    assert service.store.load_episode(task.evolution_id, episode.version).status == (
        "RESERVED"
    )
    assert runtime.conversation_state.messages == []

@pytest.mark.asyncio
async def test_forged_revision_with_persisted_id_is_rejected_before_running(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    task, episode, persisted = _reserved_revision(service)
    forged = persisted.model_copy(update={"contract_changes": ["Forged instruction"]})
    runtime = _runtime(
        service.store.workspace,
        FakeModelProvider([FakeResponse(text="unused")]),
    )

    with pytest.raises(ValueError, match="persisted RevisionPlan"):
        await ScientificEpisodeExecutor(service.store).execute(
            task=task,
            episode=episode,
            runtime=runtime,
            config=ScientificLoopConfig(max_rounds=1),
            revision=forged,
        )

    assert service.store.load_episode("evo_test", "v002").status == "RESERVED"


@pytest.mark.asyncio
@pytest.mark.parametrize("dirty", ["conversation", "budget"])
async def test_executor_rejects_reused_runtime_before_episode_attribution(
    tmp_path: Path,
    dirty: str,
) -> None:
    service = _service(tmp_path)
    task, episode = _reserved(service)
    runtime = _runtime(
        service.store.workspace,
        FakeModelProvider([FakeResponse(text="new result")]),
    )
    if dirty == "conversation":
        runtime.conversation_state.add(AssistantMessage(text="old answer"))
    else:
        runtime.budget.record_iteration()

    with pytest.raises(ValueError, match="fresh"):
        await ScientificEpisodeExecutor(service.store).execute(
            task=task,
            episode=episode,
            runtime=runtime,
            config=ScientificLoopConfig(max_rounds=1),
        )

    assert service.store.load_episode("evo_test", "v001").status == "RESERVED"
    assert not (
        service.store.workspace.root / "user_output/evo_test/v001/result.md"
    ).exists()


@pytest.mark.asyncio
async def test_executor_rejects_session_used_by_another_episode(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    first_task = service.create_task(
        goal="first",
        target=TargetSpec(goal="first"),
        evolution_id="evo_first",
    ).entity
    first = service.reserve_episode(first_task.evolution_id, mode="NORMAL").entity
    service.mark_episode_running(
        first_task.evolution_id,
        first.version,
        runtime_session_id="session_shared",
    )
    service.fail_episode(first_task.evolution_id, first.version, "finished failure")
    second_task = service.create_task(
        goal="second",
        target=TargetSpec(goal="second"),
        evolution_id="evo_second",
    ).entity
    second = service.reserve_episode(second_task.evolution_id, mode="NORMAL").entity
    runtime = _runtime(
        service.store.workspace,
        FakeModelProvider([FakeResponse(text="unused")]),
        session_id="session_shared",
    )

    with pytest.raises(ValueError, match="already attributed"):
        await ScientificEpisodeExecutor(service.store).execute(
            task=second_task,
            episode=second,
            runtime=runtime,
            config=ScientificLoopConfig(max_rounds=1),
        )

    assert service.store.load_episode("evo_second", "v001").status == "RESERVED"


@pytest.mark.asyncio
async def test_callback_failure_after_durable_completion_does_not_mark_failed(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    task, episode = _reserved(service)
    runtime = _runtime(
        service.store.workspace,
        FakeModelProvider([FakeResponse(text="complete result")]),
    )

    def fail_on_completed(event):  # type: ignore[no-untyped-def]
        if event.kind == "evolution_episode_completed":
            raise RuntimeError("renderer failed")

    with pytest.raises(RuntimeError, match="renderer failed"):
        await ScientificEpisodeExecutor(service.store).execute(
            task=task,
            episode=episode,
            runtime=runtime,
            config=ScientificLoopConfig(max_rounds=1),
            on_event=fail_on_completed,
        )

    assert service.store.load_episode("evo_test", "v001").status == "COMPLETED"


@pytest.mark.asyncio
async def test_completion_manifest_crash_does_not_reclassify_durable_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    task, episode = _reserved(service)
    runtime = _runtime(
        service.store.workspace,
        FakeModelProvider([FakeResponse(text="complete result")]),
    )
    executor = ScientificEpisodeExecutor(service.store)
    original = service.store._save_task_locked

    def fail_manifest_once(candidate, expected_revision):  # type: ignore[no-untyped-def]
        monkeypatch.setattr(service.store, "_save_task_locked", original)
        raise OSError("completion manifest crash")

    monkeypatch.setattr(service.store, "_save_task_locked", fail_manifest_once)

    with pytest.raises(OSError, match="completion manifest crash"):
        await executor.execute(
            task=task,
            episode=episode,
            runtime=runtime,
            config=ScientificLoopConfig(max_rounds=1),
        )

    assert service.store.load_episode("evo_test", "v001").status == "COMPLETED"
    assert service.get("evo_test").status == "AWAITING_EXPERT_FEEDBACK"


@pytest.mark.asyncio
async def test_failure_manifest_crash_is_reconciled_to_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    task, episode = _reserved(service)
    runtime = _runtime(service.store.workspace, FailingProvider())
    executor = ScientificEpisodeExecutor(service.store)
    original = service.store._save_task_locked

    def fail_manifest_once(candidate, expected_revision):  # type: ignore[no-untyped-def]
        monkeypatch.setattr(service.store, "_save_task_locked", original)
        raise OSError("failure manifest crash")

    monkeypatch.setattr(service.store, "_save_task_locked", fail_manifest_once)

    with pytest.raises(ProviderError, match="boom"):
        await executor.execute(
            task=task,
            episode=episode,
            runtime=runtime,
            config=ScientificLoopConfig(max_rounds=1),
        )

    assert service.store.load_episode("evo_test", "v001").status == "FAILED"
    assert service.get("evo_test").status == "BLOCKED"


@pytest.mark.asyncio
async def test_restarted_executor_reconciles_durable_split_before_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    task, episode = _reserved(service)
    running = service.mark_episode_running(task.evolution_id, episode.version).entity
    content = b"completed before process death"
    relative = "user_output/evo_test/v001/result.md"
    path = service.store.workspace.resolve(relative, must_exist=False)
    path.parent.mkdir(parents=True)
    path.write_bytes(content)

    def crash_after_episode_write(candidate, expected_revision):  # type: ignore[no-untyped-def]
        raise OSError("process died before task manifest write")

    monkeypatch.setattr(service.store, "_save_task_locked", crash_after_episode_write)
    with pytest.raises(OSError, match="process died"):
        service.complete_episode(
            task.evolution_id,
            episode.version,
            result=running.model_copy(
                update={
                    "summary": ScientificLoopSummary(
                        status="INCONCLUSIVE",
                        rounds=1,
                        candidate_count=0,
                        best_candidate_id=None,
                        best_score=0.0,
                        final_evaluation=None,
                    ),
                    "artifact": ArtifactRef(
                        path=relative,
                        size_bytes=len(content),
                        sha256=hashlib.sha256(content).hexdigest(),
                    ),
                }
            ),
        )

    restarted_store = EvolutionStore(Workspace(tmp_path))
    restarted_executor = ScientificEpisodeExecutor(restarted_store)
    runtime = _runtime(
        restarted_store.workspace,
        FakeModelProvider([FakeResponse(text="must not execute")]),
    )

    with pytest.raises(ValueError, match="exact persisted RESERVED episode"):
        await restarted_executor.execute(
            task=task,
            episode=episode,
            runtime=runtime,
            config=ScientificLoopConfig(max_rounds=1),
        )

    repaired = restarted_store.load_task(task.evolution_id)
    assert repaired.status == "AWAITING_EXPERT_FEEDBACK"
    assert repaired.last_completed_version == "v001"


@pytest.mark.asyncio
async def test_executor_logs_every_yielded_event_once_without_runtime_logger_sink(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    task, episode = _reserved(service)
    logger = EventLogger(
        tmp_path / ".photomatagent/sessions", session_id="session_test"
    )
    runtime = _runtime(
        service.store.workspace,
        FakeModelProvider(
            [
                scripted_tool_call(
                    "write",
                    {
                        "path": "user_output/evo_test/v001/result.md",
                        "content": "result",
                    },
                    tool_call_id="call_result",
                ),
                FakeResponse(text="done"),
            ]
        ),
    )

    await ScientificEpisodeExecutor(service.store, event_logger=logger).execute(
        task=task,
        episode=episode,
        runtime=runtime,
        config=ScientificLoopConfig(max_rounds=1),
    )

    kinds = [event.kind for event in logger.read_events()]
    for expected in (
        "scientific_loop_started",
        "loop_started",
        "model_request_started",
        "tool_call_completed",
        "tool_started",
        "tool_completed",
        "loop_completed",
        "scientific_loop_completed",
    ):
        assert expected in kinds
    assert kinds.count("tool_call_completed") == 1
    assert kinds.count("tool_completed") == 1
    assert kinds.count("scientific_loop_started") == 1
    assert kinds.count("scientific_loop_completed") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("via_symlink", [False, True])
async def test_outside_event_logger_is_rejected_before_running(
    tmp_path: Path,
    via_symlink: bool,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    service = _service(workspace_root)
    task, episode = _reserved(service)
    if via_symlink:
        sessions = workspace_root / ".photomatagent/sessions"
        sessions.symlink_to(outside, target_is_directory=True)
        logger = EventLogger(sessions, session_id="session_test")
    else:
        logger = EventLogger(outside, session_id="session_test")
    runtime = _runtime(
        service.store.workspace,
        FakeModelProvider([FakeResponse(text="unused")]),
    )

    with pytest.raises(ValueError, match="event log"):
        await ScientificEpisodeExecutor(service.store, event_logger=logger).execute(
            task=task,
            episode=episode,
            runtime=runtime,
            config=ScientificLoopConfig(max_rounds=1),
        )

    assert service.store.load_episode("evo_test", "v001").status == "RESERVED"
