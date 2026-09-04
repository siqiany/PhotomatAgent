from __future__ import annotations

import hashlib
import shlex
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

import photomatagent.cli.chat as chat_module
import photomatagent.cli.evolve as evolve_module
import photomatagent.cli.loop as loop_module
from photomatagent.cli.app import app
from photomatagent.errors import ProviderError
from photomatagent.logging.event_logger import EventLogger
from photomatagent.models.fake import FakeModelProvider, FakeResponse
from photomatagent.models.types import ModelRequest, ModelStreamEvent
from photomatagent.runtime.budget import BudgetState
from photomatagent.runtime.events import RuntimeEvent
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import AllowAllPolicy
from photomatagent.scientific.evolution.models import (
    ArtifactRef,
    ExpertFeedbackDraft,
    FeedbackCompilation,
    FeedbackDelta,
    RevisionPlan,
    RubricScores,
)
from photomatagent.scientific.evolution.service import EvolutionService
from photomatagent.scientific.evolution.store import EvolutionStore
from photomatagent.scientific.loop import (
    JudgeIssue,
    JudgeReport,
    ScientificLoopSummary,
    TargetSpec,
)
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.registry import ToolRegistry
from photomatagent.workspace import Workspace


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


class _ScriptedPrompt:
    def __init__(self, *answers: str) -> None:
        self.answers = deque(answers)

    async def prompt_async(self, message: str) -> str:
        del message
        return self.answers.popleft()


def _target_json() -> str:
    return (
        '{"goal":"design material","constraints":['
        '{"property":"band_gap","operator":"le","value":0.2,"unit":"eV"}'
        "]}"
    )


def _completed_task(tmp_path: Path):  # type: ignore[no-untyped-def]
    service = EvolutionService(EvolutionStore(Workspace(tmp_path)))
    task = service.create_task(
        goal="Produce a reviewable report",
        target=TargetSpec(goal="Produce a reviewable report"),
        evolution_id="evo_cli_test",
    ).entity
    reserved = service.reserve_episode(task.evolution_id, mode="NORMAL").entity
    running = service.mark_episode_running(task.evolution_id, reserved.version).entity
    content = b"reviewable result\n"
    relative = "user_output/evo_cli_test/v001/result.md"
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
    return service.get(task.evolution_id)


def _runtime(workspace: Path, *, session_id: str = "session_cli_test") -> AgentRuntime:
    scientific = ScientificState()
    return AgentRuntime(
        model=FakeModelProvider([FakeResponse(text="final reviewable report")]),
        tools=ToolRegistry(),
        workspace=Workspace(workspace),
        scientific_state=scientific,
        permission_policy=AllowAllPolicy(),
        budget=BudgetState(max_iterations=10),
        session_id=session_id,
    )


def _failed_initial_task(tmp_path: Path, *, evolution_id: str = "evo_retry_test"):
    service = EvolutionService(EvolutionStore(Workspace(tmp_path)))
    target = TargetSpec.model_validate_json(_target_json())
    task = service.create_task(
        goal=target.goal,
        target=target,
        evolution_id=evolution_id,
    ).entity
    reserved = service.reserve_episode(
        task.evolution_id,
        mode="NORMAL",
        provider="fake",
        model="fake",
    ).entity
    service.mark_episode_running(
        task.evolution_id,
        reserved.version,
        runtime_session_id="session_failed_attempt",
    )
    service.fail_episode(task.evolution_id, reserved.version, "provider failed")
    return service.get(task.evolution_id)


def test_evolve_help_registers_feedback_and_labels_future_execution_commands(
    cli_runner: CliRunner,
) -> None:
    result = cli_runner.invoke(app, ["evolve", "--help"])

    assert result.exit_code == 0
    assert "start" in result.stdout
    assert "list" in result.stdout
    assert "status" in result.stdout
    assert "history" in result.stdout
    assert "feedback" in result.stdout
    assert "iterate" in result.stdout
    feedback_help = cli_runner.invoke(app, ["evolve", "feedback", "--help"])
    assert feedback_help.exit_code == 0
    assert "--file" in feedback_help.output
    assert "--version" in feedback_help.output


def test_feedback_file_import_is_confirmed_without_constructing_runtime(
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _completed_task(tmp_path)
    review = tmp_path / "review.json"
    review.write_text(
        '{"scores":{"scientific_correctness":4,"evidence_sufficiency":3,'
        '"novelty":2,"actionability":3,"overall":3},'
        '"comments":"Authorization: Bearer cli-secret"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        evolve_module,
        "make_prompt_session",
        lambda: _ScriptedPrompt("y"),
    )
    monkeypatch.setattr(
        chat_module,
        "build_runtime",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError(f"feedback constructed runtime: {kwargs}")
        ),
    )

    result = cli_runner.invoke(
        app,
        [
            "evolve",
            "feedback",
            task.evolution_id,
            "--version",
            "v001",
            "--file",
            str(review),
            "--workspace",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    stored = EvolutionService(EvolutionStore(Workspace(tmp_path))).get(task.evolution_id)
    assert stored.status == "FEEDBACK_RECORDED"
    assert len(stored.feedback_ids) == 1
    assert "cli-secret" not in result.output
    feedback = EvolutionStore(Workspace(tmp_path)).load_feedback(
        task.evolution_id, stored.feedback_ids[0]
    )
    assert "cli-secret" not in feedback.raw_input


@pytest.mark.parametrize(
    ("confirmation", "expected_status", "expected_revision_count"),
    [("y", "REVISION_READY", 1), ("yes", "FEEDBACK_RECORDED", 0)],
)
def test_compile_previews_bounded_plan_and_requires_exact_confirmation(
    confirmation: str,
    expected_status: str,
    expected_revision_count: int,
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _completed_task(tmp_path)
    service = EvolutionService(EvolutionStore(Workspace(tmp_path)))
    episode = service.store.load_episode(task.evolution_id, "v001")
    service.attach_feedback(
        task.evolution_id,
        episode.version,
        feedback_id="fb_compile_preview",
        draft=ExpertFeedbackDraft(
            scores=RubricScores(
                scientific_correctness=2,
                evidence_sufficiency=3,
                novelty=3,
                actionability=3,
                overall=3,
            )
        ),
        result_sha256=episode.artifact.sha256,  # type: ignore[union-attr]
        raw_input="Authorization: Bearer never-render-this-secret",
    )

    class Compiler:
        async def compile(self, **kwargs):  # type: ignore[no-untyped-def]
            feedback = kwargs["feedback"]
            return FeedbackCompilation(
                compilation_id="comp_compile_preview",
                evolution_id=task.evolution_id,
                feedback_id=feedback.feedback_id,
                episode_version=episode.version,
                status="AVAILABLE",
                items=(
                    FeedbackDelta(
                        item_id="item_001",
                        category="SCIENTIFIC_CORRECTNESS",
                        status="CORRECTION",
                        severity="HIGH",
                        responsible_module="scientific_checker",
                        problem="Do not repeat unsupported conclusion",
                        requested_actions=("Re-evaluate the conclusion",),
                        acceptance_test="Conclusion passes deterministic check",
                        preserve=("Keep verified evidence",),
                        confidence=0.9,
                        source_span="never-render-this-secret",
                    ),
                ),
                provider="fake",
                model="fake",
            )

    monkeypatch.setattr(evolve_module, "_build_feedback_compiler", lambda config: Compiler())
    monkeypatch.setattr(
        evolve_module,
        "make_prompt_session",
        lambda: _ScriptedPrompt(confirmation),
    )

    result = cli_runner.invoke(
        app,
        [
            "evolve",
            "compile",
            task.evolution_id,
            "--provider",
            "fake",
            "--workspace",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Revision plan" in result.output
    assert "Strategy" in result.output
    assert "Keep verified evidence" in result.output
    assert "Do not repeat unsupported conclusion" in result.output
    assert "Conclusion passes deterministic check" in result.output
    assert "never-render-this-secret" not in result.output
    stored = service.get(task.evolution_id)
    assert stored.status == expected_status
    assert len(stored.revision_ids) == expected_revision_count
    assert len(stored.strategy_ids) == expected_revision_count


def test_feedback_requires_exact_y_and_rejects_files_outside_workspace(
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task = _completed_task(workspace)
    outside = tmp_path / "outside-review.json"
    outside.write_text(
        '{"scores":{"scientific_correctness":3,"evidence_sufficiency":3,'
        '"novelty":3,"actionability":3,"overall":3}}',
        encoding="utf-8",
    )

    rejected = cli_runner.invoke(
        app,
        [
            "evolve",
            "feedback",
            task.evolution_id,
            "--file",
            str(outside),
            "--workspace",
            str(workspace),
        ],
    )
    assert rejected.exit_code == 2
    assert "outside workspace" in rejected.output

    inside = workspace / "review.json"
    inside.write_text(outside.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(
        evolve_module,
        "make_prompt_session",
        lambda: _ScriptedPrompt("yes"),
    )
    unconfirmed = cli_runner.invoke(
        app,
        [
            "evolve",
            "feedback",
            task.evolution_id,
            "--file",
            str(inside),
            "--workspace",
            str(workspace),
        ],
    )

    assert unconfirmed.exit_code == 0, unconfirmed.output
    service = EvolutionService(EvolutionStore(Workspace(workspace)))
    assert service.get(task.evolution_id).feedback_ids == []
    assert service.store.list_feedback(task.evolution_id) == []


def test_invalid_feedback_import_does_not_echo_rejected_secret_values(
    cli_runner: CliRunner,
    tmp_path: Path,
) -> None:
    task = _completed_task(tmp_path)
    review = tmp_path / "invalid-review.json"
    review.write_text(
        '{"scores":{"scientific_correctness":3,"evidence_sufficiency":3,'
        '"novelty":3,"actionability":3,"overall":3},'
        '"password":"raw-feedback-secret"}',
        encoding="utf-8",
    )

    result = cli_runner.invoke(
        app,
        [
            "evolve",
            "feedback",
            task.evolution_id,
            "--file",
            str(review),
            "--workspace",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 2
    assert "raw-feedback-secret" not in result.output
    assert "strict ExpertFeedbackDraft" in result.output
    service = EvolutionService(EvolutionStore(Workspace(tmp_path)))
    assert service.get(task.evolution_id).feedback_ids == []


def test_start_requires_machine_verifiable_target_without_persisting_task(
    cli_runner: CliRunner,
    tmp_path: Path,
) -> None:
    result = cli_runner.invoke(
        app,
        ["evolve", "start", "--goal", "design material", "--workspace", str(tmp_path)],
    )

    assert result.exit_code != 0
    assert "target" in result.stdout.lower()
    assert EvolutionStore(Workspace(tmp_path)).list_tasks() == []


def test_start_rejects_structured_target_without_verifiable_constraints(
    cli_runner: CliRunner,
    tmp_path: Path,
) -> None:
    result = cli_runner.invoke(
        app,
        [
            "evolve",
            "start",
            "--target-json",
            '{"goal":"design material"}',
            "--provider",
            "fake",
            "--workspace",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 2
    assert "constraint" in result.stdout.lower()
    assert EvolutionStore(Workspace(tmp_path)).list_tasks() == []


def test_status_prints_exact_state_fields_and_next_feedback_command(
    cli_runner: CliRunner,
    tmp_path: Path,
) -> None:
    task = _completed_task(tmp_path)

    result = cli_runner.invoke(
        app,
        ["evolve", "status", task.evolution_id, "--workspace", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert task.evolution_id in result.stdout
    assert "AWAITING_EXPERT_FEEDBACK" in result.stdout
    assert "v001" in result.stdout
    assert "Feedback records" in result.stdout
    assert "Revision plans" in result.stdout
    assert (
        f"photomatagent evolve feedback {task.evolution_id} --version v001"
        in result.stdout
    )


def test_list_and_history_are_read_only_and_do_not_construct_provider(
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _completed_task(tmp_path)

    def unexpected_runtime(**kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError(f"read-only command constructed runtime: {kwargs}")

    monkeypatch.setattr(chat_module, "build_runtime", unexpected_runtime)

    listed = cli_runner.invoke(app, ["evolve", "list", "--workspace", str(tmp_path)])
    status = cli_runner.invoke(
        app,
        ["evolve", "status", task.evolution_id, "--workspace", str(tmp_path)],
    )
    history = cli_runner.invoke(
        app,
        ["evolve", "history", task.evolution_id, "--workspace", str(tmp_path)],
    )

    assert listed.exit_code == 0
    assert task.evolution_id in listed.stdout
    assert "AWAITING_EXPERT_FEEDBACK" in listed.stdout
    assert "v001" in listed.stdout
    assert "0 / 0" in listed.stdout
    assert "evolve feedback" in listed.stdout
    assert status.exit_code == 0
    assert "AWAITING_EXPERT_FEEDBACK" in status.stdout
    assert history.exit_code == 0
    assert task.evolution_id in history.stdout
    assert "COMPLETED" in history.stdout
    assert "v001" in history.stdout
    assert "user_output/evo_cli_test/v001/result.md" in history.stdout


def test_start_reserves_before_runtime_failure_and_leaves_recoverable_task(
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_runtime(**kwargs):  # type: ignore[no-untyped-def]
        store = EvolutionStore(Workspace(tmp_path))
        tasks = store.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].status == "RUNNING"
        assert tasks[0].current_version == "v001"
        assert store.load_episode(tasks[0].evolution_id, "v001").status == "RESERVED"
        raise RuntimeError("provider construction failed")

    monkeypatch.setattr(chat_module, "build_runtime", broken_runtime)

    result = cli_runner.invoke(
        app,
        [
            "evolve",
            "start",
            "--goal",
            "design material",
            "--target-json",
            _target_json(),
            "--provider",
            "fake",
            "--workspace",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    store = EvolutionStore(Workspace(tmp_path))
    task = store.list_tasks()[0]
    assert task.status == "BLOCKED"
    assert task.resume_status == "CREATED"
    assert task.current_version == "v001"
    episode = store.load_episode(task.evolution_id, "v001")
    assert episode.status == "FAILED"
    assert episode.error == "RuntimeError: provider construction failed"
    assert task.evolution_id in result.stdout
    assert f"photomatagent evolve start --resume {task.evolution_id}" in result.stdout


def test_start_executes_first_episode_and_prints_resume_coordinates(
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def build_test_runtime(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return _runtime(tmp_path), None

    monkeypatch.setattr(chat_module, "build_runtime", build_test_runtime)

    result = cli_runner.invoke(
        app,
        [
            "evolve",
            "start",
            "--target-json",
            _target_json(),
            "--provider",
            "fake",
            "--approval",
            "deny",
            "--max-rounds",
            "1",
            "--patience",
            "1",
            "--min-confidence",
            "0.7",
            "--no-log-events",
            "--workspace",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert calls == [
        {
                "provider": "fake",
                "model": "fake",
                "workspace_root": tmp_path,
            "approval": "deny",
            "max_iterations": 10000,
            "session_dir": tmp_path / ".photomatagent/sessions",
            "log_events": False,
        }
    ]
    store = EvolutionStore(Workspace(tmp_path))
    task = store.list_tasks()[0]
    episode = store.load_episode(task.evolution_id, "v001")
    assert task.status == "AWAITING_EXPERT_FEEDBACK"
    assert episode.status == "COMPLETED"
    assert episode.runtime_session_id == "session_cli_test"
    assert episode.artifact is not None
    assert task.evolution_id in result.stdout
    assert "v001" in result.stdout
    assert "session_cli_test" in result.stdout
    assert episode.artifact.path in result.stdout
    assert (
        f"photomatagent evolve feedback {task.evolution_id} --version v001"
        in result.stdout
    )


def test_start_accepts_workspace_contained_target_file(
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_file = tmp_path / "target.json"
    target_file.write_text(_target_json(), encoding="utf-8")
    monkeypatch.setattr(chat_module, "build_runtime", lambda **kwargs: (_runtime(tmp_path), None))

    result = cli_runner.invoke(
        app,
        [
            "evolve",
            "start",
            "--target-file",
            str(target_file),
            "--provider",
            "fake",
            "--max-rounds",
            "1",
            "--workspace",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.stdout


@pytest.mark.parametrize("interrupt_stage", ["runtime", "judge", "execution"])
def test_start_interrupt_marks_episode_failed_and_preserves_exit_130(
    interrupt_stage: str,
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupting_runtime(**kwargs):  # type: ignore[no-untyped-def]
        raise KeyboardInterrupt("password=interrupt-secret")

    def working_runtime(**kwargs):  # type: ignore[no-untyped-def]
        return _runtime(tmp_path), None

    monkeypatch.setattr(
        chat_module,
        "build_runtime",
        interrupting_runtime if interrupt_stage == "runtime" else working_runtime,
    )
    if interrupt_stage == "judge":
        monkeypatch.setattr(
            loop_module,
            "_build_judge",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                KeyboardInterrupt("password=interrupt-secret")
            ),
        )
    if interrupt_stage == "execution":
        async def interrupting_execution(**kwargs):  # type: ignore[no-untyped-def]
            raise KeyboardInterrupt("password=interrupt-secret")

        monkeypatch.setattr(
            evolve_module,
            "_execute_initial_episode",
            interrupting_execution,
        )

    result = cli_runner.invoke(
        app,
        [
            "evolve",
            "start",
            "--target-json",
            _target_json(),
            "--provider",
            "fake",
            "--workspace",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 130
    store = EvolutionStore(Workspace(tmp_path))
    task = store.list_tasks()[0]
    episode = store.load_episode(task.evolution_id, "v001")
    assert task.status == "BLOCKED"
    assert task.resume_status == "CREATED"
    assert episode.status == "FAILED"
    assert "interrupt-secret" not in (episode.error or "")
    assert "interrupt-secret" not in result.output
    assert "[REDACTED]" in (episode.error or "")


def test_execution_failure_redacts_terminal_store_log_and_delivered_events(
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bearer_secret = "bearer-secret-value"
    password_secret = "plain-password-value"
    observed: list[RuntimeEvent] = []

    class SecretFailingProvider:
        provider = "broken"
        model = "broken-model"

        async def stream(
            self,
            request: ModelRequest,
        ) -> AsyncIterator[ModelStreamEvent]:
            raise ProviderError(
                "broken",
                f"Authorization: Bearer {bearer_secret} password={password_secret}",
            )
            yield  # pragma: no cover

    def build_failing_runtime(**kwargs):  # type: ignore[no-untyped-def]
        logger = EventLogger(
            tmp_path / ".photomatagent/sessions",
            session_id="session_secret_failure",
        )
        scientific = ScientificState()
        runtime = AgentRuntime(
            model=SecretFailingProvider(),
            tools=ToolRegistry(),
            workspace=Workspace(tmp_path),
            scientific_state=scientific,
            permission_policy=AllowAllPolicy(),
            budget=BudgetState(max_iterations=10),
            event_sinks=[logger.log],
            session_id=logger.session_id,
        )
        return runtime, logger

    monkeypatch.setattr(chat_module, "build_runtime", build_failing_runtime)
    monkeypatch.setattr(loop_module, "_render_event", lambda console, event: observed.append(event))

    result = cli_runner.invoke(
        app,
        [
            "evolve",
            "start",
            "--target-json",
            _target_json(),
            "--provider",
            "fake",
            "--workspace",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    store = EvolutionStore(Workspace(tmp_path))
    task = store.list_tasks()[0]
    episode = store.load_episode(task.evolution_id, "v001")
    event_text = "\n".join(
        str(getattr(event, "error", "")) for event in observed
    )
    persisted_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / ".photomatagent").rglob("*.json*")
    )
    for secret in (bearer_secret, password_secret):
        assert secret not in result.output
        assert secret not in (episode.error or "")
        assert secret not in event_text
        assert secret not in persisted_text
    assert "[REDACTED]" in (episode.error or "")
    assert "[REDACTED]" in event_text
    assert "[REDACTED]" in persisted_text


def test_history_redacts_error_from_preexisting_episode_record(
    cli_runner: CliRunner,
    tmp_path: Path,
) -> None:
    secret = "legacy-password-value"
    service = EvolutionService(EvolutionStore(Workspace(tmp_path)))
    target = TargetSpec.model_validate_json(_target_json())
    task = service.create_task(
        goal=target.goal,
        target=target,
        evolution_id="evo_legacy_error",
    ).entity
    episode = service.reserve_episode(task.evolution_id, mode="NORMAL").entity
    service.fail_episode(
        task.evolution_id,
        episode.version,
        f"RuntimeError: password={secret}",
    )

    result = cli_runner.invoke(
        app,
        ["evolve", "history", task.evolution_id, "--workspace", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert secret not in result.output
    assert "password=[REDACTED]" in result.output


def test_start_resume_reopens_failed_initial_task_and_uses_fresh_runtime(
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _failed_initial_task(tmp_path)
    runtime = _runtime(tmp_path, session_id="session_retry_attempt")
    monkeypatch.setattr(chat_module, "build_runtime", lambda **kwargs: (runtime, None))

    result = cli_runner.invoke(
        app,
        [
            "evolve",
            "start",
            "--resume",
            original.evolution_id,
            "--provider",
            "fake",
            "--max-rounds",
            "1",
            "--workspace",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    store = EvolutionStore(Workspace(tmp_path))
    task = store.load_task(original.evolution_id)
    assert task.status == "AWAITING_EXPERT_FEEDBACK"
    assert task.current_version == "v002"
    assert task.last_completed_version == "v002"
    assert store.load_episode(task.evolution_id, "v001").status == "FAILED"
    retry = store.load_episode(task.evolution_id, "v002")
    assert retry.status == "COMPLETED"
    assert retry.runtime_session_id == "session_retry_attempt"
    assert retry.runtime_session_id != "session_failed_attempt"
    assert retry.target_snapshot.to_target_spec() == original.target
    assert runtime.conversation_state.messages[0].content == original.goal


def test_start_resume_executes_created_task_without_previous_attempt(
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = EvolutionService(EvolutionStore(Workspace(tmp_path)))
    target = TargetSpec.model_validate_json(_target_json())
    created = service.create_task(
        goal=target.goal,
        target=target,
        evolution_id="evo_created_retry",
    ).entity
    monkeypatch.setattr(chat_module, "build_runtime", lambda **kwargs: (_runtime(tmp_path), None))

    result = cli_runner.invoke(
        app,
        [
            "evolve",
            "start",
            "--resume",
            created.evolution_id,
            "--provider",
            "fake",
            "--max-rounds",
            "1",
            "--workspace",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    task = service.get(created.evolution_id)
    assert task.current_version == "v001"
    assert service.store.load_episode(created.evolution_id, "v001").status == "COMPLETED"


def test_start_resume_is_mutually_exclusive_with_creation_arguments(
    cli_runner: CliRunner,
    tmp_path: Path,
) -> None:
    task = _failed_initial_task(tmp_path)

    result = cli_runner.invoke(
        app,
        [
            "evolve",
            "start",
            "--resume",
            task.evolution_id,
            "--goal",
            "different goal",
            "--target-json",
            _target_json(),
            "--workspace",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 2
    assert "mutually exclusive" in result.output
    assert EvolutionStore(Workspace(tmp_path)).load_task(task.evolution_id) == task


@pytest.mark.parametrize(
    ("status", "resume_status", "expected_action"),
    [
        ("CREATED", None, ["start", "--resume"]),
        ("RUNNING", None, ["cancel"]),
        ("AWAITING_EXPERT_FEEDBACK", None, ["feedback"]),
        ("FEEDBACK_RECORDED", None, ["compile"]),
        ("REVISION_READY", None, ["iterate"]),
        ("ACCEPTED", None, ["reopen"]),
        ("STOPPED", "CREATED", ["reopen"]),
        ("BUDGET_EXHAUSTED", "CREATED", ["reopen"]),
        ("BLOCKED", "REVISION_READY", ["reopen"]),
    ],
)
def test_every_status_has_workspace_qualified_non_self_loop_next_command(
    status: str,
    resume_status: str | None,
    expected_action: list[str],
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace with spaces"
    workspace.mkdir()
    service = EvolutionService(EvolutionStore(Workspace(workspace)))
    target = TargetSpec.model_validate_json(_target_json())
    task = service.create_task(
        goal=target.goal,
        target=target,
        evolution_id="evo_mapping_test",
    ).entity.model_copy(
        update={"status": status, "resume_status": resume_status}
    )

    command = evolve_module._next_command(task, workspace)
    arguments = shlex.split(command)

    assert arguments[:2] == ["photomatagent", "evolve"]
    assert all(part in arguments for part in expected_action)
    assert arguments[-2:] == ["--workspace", str(workspace.resolve())]
    assert arguments[2] != "status"


def test_blocked_initial_retry_next_command_is_executable_start_resume(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace with spaces"
    workspace.mkdir()
    task = _failed_initial_task(workspace)

    command = evolve_module._next_command(task, workspace)

    assert shlex.split(command) == [
        "photomatagent",
        "evolve",
        "start",
        "--resume",
        task.evolution_id,
        "--workspace",
        str(workspace.resolve()),
    ]


def test_start_redacts_every_judge_summary_string_before_rendering(
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secrets = {
        "error": "judge-error-secret",
        "issue": "judge-issue-secret",
        "recommendation": "judge-recommendation-secret",
        "rationale": "judge-rationale-secret",
    }
    original_execute = evolve_module._execute_initial_episode
    original_render = loop_module._render_summary
    rendered: list[ScientificLoopSummary] = []

    async def execute_with_secret_summary(**kwargs):  # type: ignore[no-untyped-def]
        result = await original_execute(**kwargs)
        report = JudgeReport(
            status="AVAILABLE",
            scientific_quality=0.5,
            issues=[
                JudgeIssue(
                    severity="HIGH",
                    description=f"password={secrets['issue']}",
                )
            ],
            recommendations=[
                f"Authorization: Bearer {secrets['recommendation']}"
            ],
            rationale=f"api_key={secrets['rationale']}",
            error=f"password={secrets['error']}",
        )
        return replace(
            result,
            scientific_summary=result.scientific_summary.model_copy(
                update={"judge_report": report}
            ),
        )

    def capture_render(console, summary):  # type: ignore[no-untyped-def]
        rendered.append(summary)
        original_render(console, summary)

    monkeypatch.setattr(chat_module, "build_runtime", lambda **kwargs: (_runtime(tmp_path), None))
    monkeypatch.setattr(evolve_module, "_execute_initial_episode", execute_with_secret_summary)
    monkeypatch.setattr(loop_module, "_render_summary", capture_render)

    result = cli_runner.invoke(
        app,
        [
            "evolve",
            "start",
            "--target-json",
            _target_json(),
            "--provider",
            "fake",
            "--max-rounds",
            "1",
            "--workspace",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(rendered) == 1
    redacted_json = rendered[0].model_dump_json()
    for secret in secrets.values():
        assert secret not in result.output
        assert secret not in redacted_json
    assert "[REDACTED]" in redacted_json


@pytest.mark.parametrize("episode_status", ["RESERVED", "RUNNING"])
def test_cancel_fails_active_episode_and_is_idempotent(
    episode_status: str,
    cli_runner: CliRunner,
    tmp_path: Path,
) -> None:
    service = EvolutionService(EvolutionStore(Workspace(tmp_path)))
    target = TargetSpec.model_validate_json(_target_json())
    task = service.create_task(goal=target.goal, target=target).entity
    episode = service.reserve_episode(task.evolution_id, mode="NORMAL").entity
    if episode_status == "RUNNING":
        service.mark_episode_running(
            task.evolution_id,
            episode.version,
            runtime_session_id="session_to_cancel",
        )

    arguments = [
        "evolve",
        "cancel",
        task.evolution_id,
        "--workspace",
        str(tmp_path),
    ]
    first = cli_runner.invoke(app, arguments)
    second = cli_runner.invoke(app, arguments)

    assert first.exit_code == second.exit_code == 0
    cancelled = service.get(task.evolution_id)
    failed = service.store.load_episode(task.evolution_id, episode.version)
    assert cancelled.status == "BLOCKED"
    assert cancelled.resume_status == "CREATED"
    assert cancelled.last_completed_version is None
    assert failed.status == "FAILED"
    assert failed.error == "Cancelled by user via evolution CLI."
    assert "BLOCKED" in second.output


def test_cancel_preserves_last_good_revision(
    cli_runner: CliRunner,
    tmp_path: Path,
) -> None:
    service = EvolutionService(EvolutionStore(Workspace(tmp_path)))
    task = _completed_task(tmp_path)
    completed = service.store.load_episode(task.evolution_id, "v001")
    feedback = service.attach_feedback(
        task.evolution_id,
        completed.version,
        feedback_id="fb_cancel_test",
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
    ).entity
    service.save_compilation(
        task.evolution_id,
        FeedbackCompilation(
            compilation_id="comp_cancel_test",
            evolution_id=task.evolution_id,
            feedback_id=feedback.feedback_id,
            episode_version=completed.version,
            status="AVAILABLE",
            provider="fake",
            model="fake",
        ),
    )
    service.confirm_revision(
        task.evolution_id,
        RevisionPlan(
            revision_id="rp_cancel_test",
            evolution_id=task.evolution_id,
            source_version=completed.version,
            feedback_id=feedback.feedback_id,
            confirmed=True,
        ),
    )
    second = service.reserve_episode(
        task.evolution_id,
        mode="CARRY_VERIFIED_EVIDENCE",
    ).entity
    service.mark_episode_running(task.evolution_id, second.version)

    result = cli_runner.invoke(
        app,
        ["evolve", "cancel", task.evolution_id, "--workspace", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    cancelled = service.get(task.evolution_id)
    assert cancelled.status == "BLOCKED"
    assert cancelled.resume_status == "REVISION_READY"
    assert cancelled.current_version == "v002"
    assert cancelled.last_completed_version == "v001"
    assert service.store.load_episode(task.evolution_id, "v001").status == "COMPLETED"
    assert service.store.load_episode(task.evolution_id, "v002").status == "FAILED"


def test_cancel_reconciles_completed_episode_manifest_split(
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = EvolutionService(EvolutionStore(Workspace(tmp_path)))
    target = TargetSpec.model_validate_json(_target_json())
    task = service.create_task(goal=target.goal, target=target).entity
    reserved = service.reserve_episode(task.evolution_id, mode="NORMAL").entity
    running = service.mark_episode_running(task.evolution_id, reserved.version).entity
    relative = f"user_output/{task.evolution_id}/v001/result.md"
    artifact_path = service.store.workspace.resolve(relative, must_exist=False)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    content = b"completed result"
    artifact_path.write_bytes(content)
    completed = running.model_copy(
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
    )

    original_save = service.store._save_task_locked
    monkeypatch.setattr(
        service.store,
        "_save_task_locked",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("manifest split")),
    )
    with pytest.raises(OSError, match="manifest split"):
        service.complete_episode(task.evolution_id, running.version, result=completed)
    monkeypatch.setattr(service.store, "_save_task_locked", original_save)

    result = cli_runner.invoke(
        app,
        ["evolve", "cancel", task.evolution_id, "--workspace", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    reconciled = service.get(task.evolution_id)
    assert reconciled.status == "AWAITING_EXPERT_FEEDBACK"
    assert reconciled.last_completed_version == "v001"
    assert service.store.load_episode(task.evolution_id, "v001").status == "COMPLETED"


def test_status_renders_bracketed_workspace_next_command_literally(
    cli_runner: CliRunner,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace [bold] with spaces"
    workspace.mkdir()
    task = _completed_task(workspace)
    expected = (
        f"photomatagent evolve feedback {task.evolution_id} --version v001 "
        f"--workspace {shlex.quote(str(workspace.resolve()))}"
    )

    result = cli_runner.invoke(
        app,
        ["evolve", "status", task.evolution_id, "--workspace", str(workspace)],
    )

    assert result.exit_code == 0, result.output
    assert f"Next command: {expected}" in result.output
