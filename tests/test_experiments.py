from __future__ import annotations

from pathlib import Path

import pytest

from photomatagent.experiments.compare import compare_summaries
from photomatagent.experiments.evaluator import evaluate_expectations
from photomatagent.experiments.loader import load_experiment_config
from photomatagent.experiments.models import (
    ConfigurationSnapshot,
    Expectations,
    ExperimentConfig,
    ExperimentSummary,
    ExperimentTask,
)
from photomatagent.experiments.runner import run_experiment
from photomatagent.experiments.storage import load_experiment_summary, save_experiment
from photomatagent.observability.analyzer import SessionSummary


def _session_summary(**overrides) -> SessionSummary:
    values = {
        "session_id": "s",
        "path": Path("s"),
        "iterations": 2,
        "model_calls": 2,
        "tool_calls": 2,
        "unique_tools": 2,
        "tools_used": ["glob", "read"],
    }
    values.update(overrides)
    return SessionSummary(**values)


def test_all_deterministic_expectation_types() -> None:
    evaluation = evaluate_expectations(
        Expectations(
            answer_contains=["loop"],
            answer_not_contains=["secret"],
            tools_used=["read"],
            tools_not_used=["bash"],
            max_tool_calls=3,
            max_iterations=2,
        ),
        answer="The Loop is in runtime/loop.py",
        summary=_session_summary(),
    )
    assert evaluation.status == "PASS"
    assert len(evaluation.checks) == 6

    failed = evaluate_expectations(
        Expectations(tools_used=["grep"], max_iterations=1),
        answer="",
        summary=_session_summary(),
    )
    assert failed.status == "FAIL"
    assert [check.passed for check in failed.checks] == [False, False]

    assert (
        evaluate_expectations(None, answer="anything", summary=_session_summary()).status
        == "UNEVALUATED"
    )


def test_experiment_comparison_calculates_b_minus_a() -> None:
    snapshot = ConfigurationSnapshot(
        provider="fake",
        model="fake",
        system_prompt={},
        stop_policy={},
        context_builder={},
    )
    base = dict(
        name="x",
        configuration=snapshot,
        tasks_total=1,
        tasks_completed=1,
        expectations_passed=1,
        expectations_failed=0,
        tasks_unevaluated=0,
        average_model_calls=2,
        average_tool_failures=0,
        tool_failure_rate=0,
        repeated_tool_calls=0,
        input_tokens=10,
        output_tokens=5,
    )
    a = ExperimentSummary(
        experiment_id="a",
        average_iterations=4,
        average_tool_calls=7,
        average_repeated_tool_calls=1,
        duration_seconds=10,
        expectation_pass_rate=0.5,
        **base,
    )
    b = ExperimentSummary(
        experiment_id="b",
        average_iterations=3,
        average_tool_calls=5,
        average_repeated_tool_calls=0,
        duration_seconds=8,
        expectation_pass_rate=1.0,
        **base,
    )
    rows = {row.metric: row for row in compare_summaries(a, b)}
    assert rows["Avg iterations"].delta == -1
    assert rows["Avg tool calls"].delta == -2
    assert rows["Expectation pass rate"].delta == 0.5


def test_experiment_comparison_rejects_different_models() -> None:
    snapshot_a = ConfigurationSnapshot(
        provider="fake",
        model="a",
        system_prompt={},
        stop_policy={},
        context_builder={},
    )
    snapshot_b = snapshot_a.model_copy(update={"model": "b"})
    base = dict(
        name="x",
        tasks_total=1,
        tasks_completed=1,
        expectations_passed=1,
        expectations_failed=0,
        tasks_unevaluated=0,
        average_iterations=1,
        average_model_calls=1,
        average_tool_calls=0,
        average_tool_failures=0,
        tool_failure_rate=0,
        repeated_tool_calls=0,
        average_repeated_tool_calls=0,
        duration_seconds=0,
    )
    summary_a = ExperimentSummary(
        experiment_id="a", configuration=snapshot_a, **base
    )
    summary_b = ExperimentSummary(
        experiment_id="b", configuration=snapshot_b, **base
    )

    with pytest.raises(ValueError, match="mismatched: model"):
        compare_summaries(summary_a, summary_b)


@pytest.mark.asyncio
async def test_fake_experiment_runs_sequentially_and_persists(tmp_path: Path) -> None:
    config = ExperimentConfig(
        name="offline-baseline",
        tasks=[
            ExperimentTask(
                id="one",
                prompt="investigate material InAs",
                expect=Expectations(
                    answer_contains=["InAs"],
                    tools_used=["mock.run_calculation"],
                    max_tool_calls=3,
                    max_iterations=4,
                ),
            ),
            ExperimentTask(id="two", prompt="investigate material GaAs"),
        ],
    )
    result = await run_experiment(
        config,
        provider="fake",
        model="fake",
        workspace_root=tmp_path,
        sessions_dir=tmp_path / "sessions",
    )
    assert result.summary.tasks_total == 2
    assert result.summary.tasks_completed == 2
    assert result.summary.expectations_passed == 1
    assert result.summary.tasks_unevaluated == 1
    assert result.runs[0].session_id != result.runs[1].session_id

    saved = save_experiment(result, tmp_path / "experiments")
    restored = load_experiment_summary(saved)
    assert restored.experiment_id == result.experiment_id
    assert (saved / "config.json").is_file()
    assert (saved / "runs.json").is_file()


def test_json_experiment_config_loader(tmp_path: Path) -> None:
    config = tmp_path / "experiment.json"
    config.write_text(
        '{"name":"x","tasks":[{"id":"t","prompt":"hello"}]}',
        encoding="utf-8",
    )
    loaded = load_experiment_config(config)
    assert loaded.name == "x"
    assert loaded.tasks[0].expect is None
