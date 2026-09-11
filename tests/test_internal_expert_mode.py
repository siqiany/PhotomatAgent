from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest
from types import SimpleNamespace

from photomatagent.observability.trace import load_trace
from photomatagent.scientific.evolution.importer import (
    HistoricalSessionImporter,
    preview_historical_session,
)
from photomatagent.scientific.evolution.models import ExecutionMode
from photomatagent.scientific.evolution.service import EvolutionService
from photomatagent.scientific.evolution.service import EvolutionOperationConflict
from photomatagent.scientific.evolution.store import EvolutionStore
from photomatagent.scientific.loop import TargetSpec
from photomatagent.scientific.state import ScientificState
from photomatagent.sessions.store import save_session_snapshot
from photomatagent.runtime.state import ConversationState
from photomatagent.models.types import AssistantMessage
from photomatagent.models.fake import FakeModelProvider
from photomatagent.models.fake import FakeResponse
from photomatagent.tools.registry import ToolRegistry
from photomatagent.workspace import Workspace
from photomatagent.cli.chat import build_runtime
from rich.console import Console
from photomatagent.cli.commands import ChatCommandRouter
from photomatagent.cli.expert import _resume_linked


class _Prompt:
    def __init__(self, answers: list[str]) -> None:
        self.answers = iter(answers)
        self.prompts: list[str] = []

    async def prompt_async(self, message: str) -> str:
        self.prompts.append(message)
        return next(self.answers)


def test_import_historical_session_materializes_provenance_bound_v001(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    session_dir = tmp_path / ".photomatagent" / "sessions" / "session-abc"
    session_dir.mkdir(parents=True)
    events = [
        {
            "kind": "loop_started",
            "session_id": "session-abc",
            "goal": "find a stable infrared absorber",
        },
        {
            "kind": "text_delta",
            "session_id": "session-abc",
            "iteration": 1,
            "text": "Historical final answer",
        },
    ]
    (session_dir / "events.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in events), encoding="utf-8"
    )
    original_state = ScientificState(
        goal="find a stable infrared absorber", open_questions=["verify stability"]
    )
    save_session_snapshot(
        session_dir,
        conversation=ConversationState(
            messages=[AssistantMessage(content="Historical final answer")]
        ),
        scientific=original_state,
    )
    target = TargetSpec(
        goal="find a stable infrared absorber",
        constraints=[{"property": "bandgap", "operator": "between", "value": [0.1, 0.5]}],
    )
    preview = preview_historical_session(workspace, session_dir)
    service = EvolutionService(EvolutionStore(workspace))
    task = HistoricalSessionImporter(workspace, service).import_session(
        session_dir, target=target
    )

    assert ExecutionMode.__args__[-1] == "IMPORTED_SESSION"
    assert task.current_version == "v001"
    episode = service.store.load_episode(task.evolution_id, "v001")
    assert episode.execution_mode == "IMPORTED_SESSION"
    assert episode.runtime_session_id == "session-abc"
    assert episode.summary is None
    assert episode.artifact is not None
    assert episode.artifact.sha256 == hashlib.sha256(
        b"Historical final answer\n"
    ).hexdigest()
    assert service.store.load_scientific_state(task.evolution_id, "v001").model_dump() == original_state.model_dump()
    assert preview.session_id == "session-abc"
    assert HistoricalSessionImporter(workspace, service).find_linked_task("session-abc") == task
    canonical = workspace.resolve(f"user_output/{task.evolution_id}/v001/result.md", must_exist=True)
    canonical.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(EvolutionOperationConflict):
        HistoricalSessionImporter(workspace, service).import_session(session_dir, target=target)


def test_import_retries_after_reservation_interruption(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = Workspace(tmp_path)
    session_dir = tmp_path / ".photomatagent" / "sessions" / "session-retry"
    session_dir.mkdir(parents=True)
    (session_dir / "events.jsonl").write_text(
        json.dumps({"kind": "loop_started", "session_id": "session-retry", "goal": "goal"})
        + "\n"
        + json.dumps({"kind": "text_delta", "session_id": "session-retry", "iteration": 1, "text": "answer"})
        + "\n", encoding="utf-8"
    )
    target = TargetSpec(goal="goal", constraints=[{"property": "x", "operator": "ge", "value": 1}])
    service = EvolutionService(EvolutionStore(workspace))
    importer = HistoricalSessionImporter(workspace, service)
    original = service._reserve_imported_episode
    calls = 0

    def interrupt(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "_reserve_imported_episode", interrupt)
    with pytest.raises(RuntimeError, match="simulated"):
        importer.import_session(session_dir, target=target)
    task = importer.import_session(session_dir, target=target)
    assert task.current_version == "v001"
    assert service.store.load_episode(task.evolution_id, "v001").status == "COMPLETED"


def test_runtime_logger_sessions_follow_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "photomatagent.cli.chat.create_default_registry",
        lambda *args, **kwargs: ToolRegistry(),
    )
    monkeypatch.setattr(
        "photomatagent.cli.chat.create_provider",
        lambda *args, **kwargs: FakeModelProvider(),
    )
    runtime, logger = build_runtime(workspace_root=tmp_path, provider="fake")
    assert runtime.workspace.root == tmp_path.resolve()
    assert logger is not None
    assert logger.session_dir.parent == tmp_path.resolve() / ".photomatagent" / "sessions"


@pytest.mark.asyncio
async def test_linked_awaiting_feedback_continues_compile_then_iterate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task = SimpleNamespace(
        status="AWAITING_EXPERT_FEEDBACK", evolution_id="evo-linked",
        last_completed_version="v001", current_version="v001",
    )
    prompt = _Prompt(["y", "y"])
    states = iter([SimpleNamespace(status="REVISION_READY")])
    service = SimpleNamespace(get=lambda _id: next(states))
    feedback_calls: list[str] = []
    async def fake_feedback(**kwargs: object) -> object:
        feedback_calls.append("feedback")
        return object()
    monkeypatch.setattr("photomatagent.cli.expert.run_feedback_command", fake_feedback)
    compile_calls: list[str] = []
    async def fake_compile(**kwargs: object) -> object:
        compile_calls.append("compile")
        return object()
    iterated: list[str] = []
    async def iterate(evolution_id: str) -> None:
        iterated.append(evolution_id)
    await _resume_linked(task=task, session=prompt, output=Console(record=True),
                         workspace=tmp_path, compile_runner=fake_compile,
                         iterate_callback=iterate, service=service)  # type: ignore[arg-type]
    assert feedback_calls == ["feedback"]
    assert compile_calls == ["compile"]
    assert iterated == ["evo-linked"]


def test_import_cleans_stale_partial_temporary_artifact(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    session_dir = tmp_path / ".photomatagent" / "sessions" / "session-temp"
    session_dir.mkdir(parents=True)
    (session_dir / "events.jsonl").write_text(
        json.dumps({"kind": "loop_started", "session_id": "session-temp", "goal": "goal"})
        + "\n" + json.dumps({"kind": "text_delta", "session_id": "session-temp", "iteration": 1, "text": "answer"}) + "\n",
        encoding="utf-8",
    )
    target = TargetSpec(goal="goal", constraints=[{"property": "x", "operator": "ge", "value": 1}])
    service = EvolutionService(EvolutionStore(workspace))
    evolution_id = "evo_import_" + hashlib.sha256(b"session-temp").hexdigest()[:32]
    result_dir = tmp_path / "user_output" / evolution_id / "v001"
    result_dir.mkdir(parents=True)
    (result_dir / ".result.md.importing-stale").write_bytes(b"partial")
    task = HistoricalSessionImporter(workspace, service).import_session(session_dir, target=target)
    assert task.current_version == "v001"
    assert list(result_dir.glob(".result.md.importing-*"))


@pytest.mark.asyncio
async def test_expert_route_imports_scores_and_keeps_chat_state_clean(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    session_dir = tmp_path / ".photomatagent" / "sessions" / "session-e2e"
    session_dir.mkdir(parents=True)
    (session_dir / "events.jsonl").write_text(
        json.dumps({"kind": "loop_started", "session_id": "session-e2e", "goal": "goal"})
        + "\n" + json.dumps({"kind": "text_delta", "session_id": "session-e2e", "iteration": 1, "text": "answer"}) + "\n",
        encoding="utf-8",
    )
    generated_target = json.dumps({
        "goal": "goal",
        "constraints": [{
            "property": "x", "operator": "ge", "value": 1,
            "unit": "", "severity": "HARD", "weight": 1.0,
            "description": "explicit requirement", "basis": "EXPLICIT_GOAL",
            "rationale": "goal says so", "confidence": 1.0,
            "requires_confirmation": False,
        }],
        "objectives": [], "operating_conditions": {}, "warnings": [],
    })
    answers = ["y", "y", "y", "1", "1", "1", "1", "1", "", "", "", "", "", "", "/submit", "", "y", "n"]
    prompt = _Prompt(answers)
    class _Runtime:
        def __init__(self) -> None:
            self.workspace = workspace
            self.conversation_state = ConversationState()
            self.session_id = None
            self.run_calls = 0
            self.model_provider = FakeModelProvider([FakeResponse(text=generated_target)])

        def run(self, _prompt: str) -> None:
            self.run_calls += 1
            raise AssertionError("expert input must not call AgentRuntime.run")

    runtime = _Runtime()
    console = Console(record=True)
    router = ChatCommandRouter(console, runtime, workspace, sessions_dir=session_dir.parent, prompt_session=prompt)
    await router.execute("/expert session-e2e")
    service = EvolutionService(EvolutionStore(workspace))
    task = service.get("evo_import_" + hashlib.sha256(b"session-e2e").hexdigest()[:32])
    feedback = service.store.list_feedback(task.evolution_id)
    assert task.current_version == "v001"
    assert task.status == "FEEDBACK_RECORDED"
    assert len(feedback) == 1
    assert feedback[0].result_sha256 == hashlib.sha256(b"answer\n").hexdigest()
    assert runtime.conversation_state.messages == []
    assert runtime.run_calls == 0
    assert all(message.startswith("[EXPERT MODE |") for message in prompt.prompts)
