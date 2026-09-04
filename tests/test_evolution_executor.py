from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from photomatagent.errors import ProviderError
from photomatagent.logging.event_logger import EventLogger
from photomatagent.models.fake import FakeModelProvider, FakeResponse, scripted_tool_call
from photomatagent.models.types import ModelRequest, ModelStreamEvent, ModelUsage
from photomatagent.runtime.budget import BudgetState
from photomatagent.runtime.events import RuntimeEvent
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import AllowAllPolicy
from photomatagent.scientific.evolution.executor import ScientificEpisodeExecutor
from photomatagent.scientific.evolution.artifacts import (
    EpisodeResultAlreadyExistsError,
)
from photomatagent.scientific.evolution.models import RevisionPlan
from photomatagent.scientific.evolution.service import EvolutionService
from photomatagent.scientific.evolution.store import EvolutionStore
from photomatagent.scientific.loop import ScientificLoopConfig, TargetSpec
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
    assert "scientific_loop_completed" in [event.kind for event in logger.read_events()]


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
    assert service.get("evo_test").status == "RUNNING"
