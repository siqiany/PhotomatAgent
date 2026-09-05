from __future__ import annotations

import hashlib
from collections import deque
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

import photomatagent.cli.commands as commands_module
import photomatagent.cli.evolve as evolve_module
from photomatagent.cli.chat import run_interactive_chat
from photomatagent.cli.commands import ChatCommandRouter
from photomatagent.models.fake import FakeModelProvider, FakeResponse
from photomatagent.models.types import UserMessage
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import AllowAllPolicy
from photomatagent.scientific.evolution.models import (
    ArtifactRef,
    ExpertFeedbackDraft,
    FeedbackCompilation,
    FeedbackDelta,
    RubricScores,
)
from photomatagent.scientific.evolution.service import EvolutionService
from photomatagent.scientific.evolution.store import EvolutionStore
from photomatagent.scientific.loop import ScientificLoopSummary, TargetSpec
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.registry import ToolRegistry
from photomatagent.workspace import Workspace


class _ScriptedPrompt:
    def __init__(self, *answers: str | BaseException) -> None:
        self.answers = deque(answers)

    async def prompt_async(self, message: str) -> str:
        del message
        answer = self.answers.popleft()
        if isinstance(answer, BaseException):
            raise answer
        return answer


def _runtime(tmp_path: Path) -> AgentRuntime:
    scientific = ScientificState()
    return AgentRuntime(
        model=FakeModelProvider([FakeResponse(text="ordinary response")]),
        tools=ToolRegistry(),
        workspace=Workspace(tmp_path),
        scientific_state=scientific,
        permission_policy=AllowAllPolicy(),
    )


def _console() -> tuple[Console, StringIO]:
    stream = StringIO()
    return Console(file=stream, force_terminal=False, width=120), stream


def _completed_task(tmp_path: Path) -> str:
    service = EvolutionService(EvolutionStore(Workspace(tmp_path)))
    task = service.create_task(
        goal="Produce a reviewable report",
        target=TargetSpec(goal="Produce a reviewable report"),
        evolution_id="evo_chat_test",
    ).entity
    reserved = service.reserve_episode(task.evolution_id, mode="NORMAL").entity
    running = service.mark_episode_running(task.evolution_id, reserved.version).entity
    content = b"reviewable result\n"
    relative = "user_output/evo_chat_test/v001/result.md"
    path = service.store.workspace.resolve(relative, must_exist=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    service.complete_episode(
        task.evolution_id,
        running.version,
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
    return task.evolution_id


@pytest.mark.asyncio
async def test_evolve_slash_command_never_calls_chat_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(tmp_path)
    console, _ = _console()
    router = ChatCommandRouter(console, runtime, runtime.workspace)
    routed: list[list[str]] = []

    async def forbidden_run(*args: object, **kwargs: object):
        del args, kwargs
        raise AssertionError("/evolve must not enter the current chat runtime")
        yield  # pragma: no cover

    async def capture_cli(args: list[str]) -> None:
        routed.append(args)

    monkeypatch.setattr(runtime, "run", forbidden_run)
    monkeypatch.setattr(router, "_run_cli", capture_cli)

    await router.execute("/evolve status evo_test")

    assert routed == [["evolve", "status", "evo_test"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/evolve-now status evo_test", "/EVOLVE status evo_test"])
async def test_only_exact_evolve_first_token_enters_evolution_route(
    command: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(tmp_path)
    console, stream = _console()
    router = ChatCommandRouter(console, runtime, runtime.workspace)
    routed: list[list[str]] = []

    async def capture_cli(args: list[str]) -> None:
        routed.append(args)

    monkeypatch.setattr(router, "_run_cli", capture_cli)

    await router.execute(command)

    assert routed == []
    assert "未知命令" in stream.getvalue()


@pytest.mark.asyncio
async def test_normal_expert_sentence_remains_normal_goal_and_does_not_write_store(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    console, _ = _console()

    await run_interactive_chat(
        console,
        runtime,
        _ScriptedPrompt("专家说证据不足，需要反馈", "/exit"),
    )

    user_messages = [
        message.content
        for message in runtime.conversation_state.messages
        if isinstance(message, UserMessage)
    ]
    assert user_messages == ["专家说证据不足，需要反馈"]
    assert EvolutionStore(runtime.workspace).list_tasks() == []


@pytest.mark.asyncio
async def test_interactive_chat_passes_its_prompt_session_to_router(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(tmp_path)
    console, _ = _console()
    prompt = _ScriptedPrompt("/exit")
    observed: list[object] = []

    class _CapturingRouter:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args
            observed.append(kwargs["prompt_session"])

    monkeypatch.setattr(commands_module, "ChatCommandRouter", _CapturingRouter)

    await run_interactive_chat(console, runtime, prompt)

    assert observed == [prompt]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("subcommand", "handler_name"),
    [("feedback", "run_feedback_command"), ("compile", "run_compile_command")],
)
async def test_interactive_evolve_flows_reuse_current_prompt_session(
    subcommand: str,
    handler_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(tmp_path)
    console, _ = _console()
    prompt = _ScriptedPrompt()
    observed: list[object] = []

    async def capture_handler(**kwargs: object) -> None:
        observed.append(kwargs["session"])

    monkeypatch.setattr(evolve_module, handler_name, capture_handler, raising=False)
    router = ChatCommandRouter(
        console,
        runtime,
        runtime.workspace,
        prompt_session=prompt,
    )

    async def forbidden_cli(args: list[str]) -> None:
        raise AssertionError(f"interactive flow entered background CLI: {args}")

    monkeypatch.setattr(router, "_run_cli", forbidden_cli)

    await router.execute(f"/evolve {subcommand} evo_test")

    assert observed == [prompt]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_signal", ["/cancel", KeyboardInterrupt()])
async def test_feedback_cancel_does_not_persist_partial_review(
    cancel_signal: str | BaseException,
    tmp_path: Path,
) -> None:
    evolution_id = _completed_task(tmp_path)
    runtime = _runtime(tmp_path)
    console, _ = _console()
    router = ChatCommandRouter(
        console,
        runtime,
        runtime.workspace,
        prompt_session=_ScriptedPrompt(cancel_signal),
    )

    await router.execute(f"/evolve feedback {evolution_id}")

    task = EvolutionService(EvolutionStore(runtime.workspace)).get(evolution_id)
    assert task.status == "AWAITING_EXPERT_FEEDBACK"
    assert task.feedback_ids == []
    assert task.compilation_ids == []
    assert task.revision_ids == []


@pytest.mark.asyncio
async def test_compile_ctrl_c_does_not_persist_partial_compilation_or_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evolution_id = _completed_task(tmp_path)
    service = EvolutionService(EvolutionStore(Workspace(tmp_path)))
    episode = service.store.load_episode(evolution_id, "v001")
    service.attach_feedback(
        evolution_id,
        episode.version,
        feedback_id="fb_chat_compile",
        draft=ExpertFeedbackDraft(
            scores=RubricScores(
                scientific_correctness=3,
                evidence_sufficiency=3,
                novelty=3,
                actionability=3,
                overall=3,
            )
        ),
        result_sha256=episode.artifact.sha256,  # type: ignore[union-attr]
        raw_input="review",
    )

    class _InterruptingCompiler:
        async def compile(self, **kwargs: object) -> None:
            del kwargs
            raise KeyboardInterrupt

    monkeypatch.setattr(
        evolve_module,
        "_build_feedback_compiler",
        lambda config: _InterruptingCompiler(),
    )
    runtime = _runtime(tmp_path)
    console, _ = _console()
    router = ChatCommandRouter(
        console,
        runtime,
        runtime.workspace,
        prompt_session=_ScriptedPrompt(),
    )

    await router.execute(f"/evolve compile {evolution_id} --provider fake")

    task = service.get(evolution_id)
    assert task.status == "FEEDBACK_RECORDED"
    assert task.compilation_ids == []
    assert task.revision_ids == []
    assert task.strategy_ids == []


@pytest.mark.asyncio
async def test_compile_cancel_keeps_available_compilation_but_no_revision_or_strategy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evolution_id = _completed_task(tmp_path)
    service = EvolutionService(EvolutionStore(Workspace(tmp_path)))
    episode = service.store.load_episode(evolution_id, "v001")
    feedback = service.attach_feedback(
        evolution_id,
        episode.version,
        feedback_id="fb_chat_cancel",
        draft=ExpertFeedbackDraft(
            scores=RubricScores(
                scientific_correctness=3,
                evidence_sufficiency=3,
                novelty=3,
                actionability=3,
                overall=3,
            )
        ),
        result_sha256=episode.artifact.sha256,  # type: ignore[union-attr]
        raw_input="review",
    ).entity

    class _AvailableCompiler:
        async def compile(self, **kwargs: object) -> FeedbackCompilation:
            del kwargs
            return FeedbackCompilation(
                compilation_id="comp_chat_cancel",
                evolution_id=evolution_id,
                feedback_id=feedback.feedback_id,
                episode_version=episode.version,
                status="AVAILABLE",
                items=(
                    FeedbackDelta(
                        item_id="item_chat_cancel",
                        category="EVIDENCE_SUFFICIENCY",
                        status="CORRECTION",
                        severity="HIGH",
                        responsible_module="evidence",
                        problem="Evidence is insufficient",
                        requested_actions=("Add validated evidence",),
                        acceptance_test="Validated evidence is present",
                        confidence=0.9,
                        source_span="review",
                    ),
                ),
                provider="fake",
                model="fake",
            )

    monkeypatch.setattr(
        evolve_module,
        "_build_feedback_compiler",
        lambda config: _AvailableCompiler(),
    )
    runtime = _runtime(tmp_path)
    console, _ = _console()
    router = ChatCommandRouter(
        console,
        runtime,
        runtime.workspace,
        prompt_session=_ScriptedPrompt("/cancel"),
    )

    await router.execute(f"/evolve compile {evolution_id} --provider fake")

    task = service.get(evolution_id)
    assert task.status == "FEEDBACK_RECORDED"
    assert task.compilation_ids == ["comp_chat_cancel"]
    assert task.revision_ids == []
    assert task.strategy_ids == []
