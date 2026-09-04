from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from typer.testing import CliRunner

import photomatagent.cli.chat as chat_module
from photomatagent.cli.app import app
from photomatagent.models.fake import FakeModelProvider, FakeResponse
from photomatagent.runtime.budget import BudgetState
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import AllowAllPolicy
from photomatagent.scientific.evolution.models import ArtifactRef
from photomatagent.scientific.evolution.service import EvolutionService
from photomatagent.scientific.evolution.store import EvolutionStore
from photomatagent.scientific.loop import ScientificLoopSummary, TargetSpec
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.registry import ToolRegistry
from photomatagent.workspace import Workspace


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


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


def test_evolve_help_is_registered_and_labels_future_execution_commands(
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
    assert "planned" in result.stdout.lower()

    unavailable = cli_runner.invoke(app, ["evolve", "feedback", "evo_cli_test"])
    assert unavailable.exit_code != 0
    assert "No such command" in unavailable.output


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
    assert f"photomatagent evolve reopen {task.evolution_id}" in result.stdout


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
