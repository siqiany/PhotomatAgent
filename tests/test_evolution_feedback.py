from __future__ import annotations

import hashlib
import io
import json
from collections.abc import AsyncIterator
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError
from rich.console import Console
from typer.testing import CliRunner

import photomatagent.cli.evolve as evolve_module
import photomatagent.scientific.evolution.feedback as feedback_module
from photomatagent.cli.app import app
from photomatagent.models.fake import FakeModelProvider, FakeResponse
from photomatagent.models.types import ModelRequest, ModelStreamEvent, SystemMessage, UserMessage
from photomatagent.scientific.evolution.feedback import FeedbackCompiler
from photomatagent.scientific.evolution.models import (
    ArtifactRef,
    EpisodeRecord,
    EvolutionTask,
    ExpertFeedbackDraft,
    ExpertFeedbackRecord,
    RubricScores,
)
from photomatagent.scientific.evolution.service import (
    EvolutionOperationConflict,
    EvolutionService,
)
from photomatagent.scientific.evolution.store import EvolutionStore
from photomatagent.scientific.loop import TargetSpec
from photomatagent.workspace import Workspace


def test_clean_feedback_import_does_not_load_runtime_or_scientific_execution_graph() -> None:
    forbidden = [
        "photomatagent.scientific.loop.judge",
        "photomatagent.runtime.loop",
        "photomatagent.runtime.state",
        "photomatagent.scientific.state",
        "photomatagent.tools.registry",
        "photomatagent.scientific.capabilities.registry",
    ]
    script = (
        "import json, sys\n"
        "import photomatagent.scientific.evolution.feedback\n"
        f"names = {forbidden!r}\n"
        "print(json.dumps([name for name in names if name in sys.modules]))\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def _available_response(*, status: str = "QUERY", source_span: str = "正文信息呢？") -> str:
    return json.dumps(
        {
            "status": "AVAILABLE",
            "items": [
                {
                    "category": "EVIDENCE_SUFFICIENCY",
                    "status": status,
                    "severity": "HIGH",
                    "responsible_module": "retrieval_planner",
                    "problem": "摘要是否足够",
                    "requested_actions": ["读取正文"],
                    "acceptance_test": "核心结论绑定全文证据",
                    "preserve": [],
                    "confidence": 0.9,
                    "source_span": source_span,
                }
            ],
            "warnings": [],
        },
        ensure_ascii=False,
    )


def _context() -> tuple[EvolutionTask, EpisodeRecord, ExpertFeedbackRecord]:
    task = EvolutionTask(
        evolution_id="evo_compiler_unit",
        goal="Improve the report",
        target=TargetSpec(goal="Improve the report"),
        task_group_id="group_compiler_unit",
        input_sha256="0" * 64,
    )
    episode = EpisodeRecord(
        evolution_id=task.evolution_id,
        episode_id="ep_compiler",
        version="v001",
        status="COMPLETED",
        task_snapshot=task.model_dump(mode="json"),
        target_snapshot=task.target,
        artifact=ArtifactRef(
            path="user_output/result.md",
            size_bytes=6,
            sha256="a" * 64,
        ),
    )
    feedback = ExpertFeedbackRecord(
        feedback_id="fb_compiler",
        evolution_id=task.evolution_id,
        episode_version="v001",
        result_sha256="a" * 64,
        rubric_version="expert-review-v1",
        raw_input="目前检索没有本地文献，完全依靠 arXiv 摘要是否足够，正文信息呢？",
        comments="正文信息呢？",
        scores=RubricScores(
            scientific_correctness=3,
            evidence_sufficiency=2,
            novelty=3,
            actionability=3,
            overall=3,
        ),
    )
    return task, episode, feedback


@pytest.mark.asyncio
async def test_compiler_has_no_tools_preserves_query_and_bounds_result_text() -> None:
    task, episode, feedback = _context()
    model = FakeModelProvider([FakeResponse(text=_available_response())])

    result = await FeedbackCompiler(model).compile(
        task=task,
        episode=episode,
        feedback=feedback,
        result_text="x" * 12_500,
    )

    assert result.status == "AVAILABLE"
    assert result.items[0].status == "QUERY"
    assert result.provider == "fake"
    assert result.model == "fake"
    assert len(model.requests) == 1
    request = model.requests[0]
    assert request.tools == []
    assert isinstance(request.messages[0], SystemMessage)
    assert "do not grade" in request.messages[0].content.lower()
    assert "deterministic hard constraints" in request.messages[0].content.lower()
    assert isinstance(request.messages[1], UserMessage)
    payload = json.loads(request.messages[1].content)
    assert len(payload["result_text"]) == 12_000
    assert payload["result_text_truncated"] is True


@pytest.mark.asyncio
async def test_invalid_json_degrades_without_mutating_raw_feedback() -> None:
    task, episode, feedback = _context()
    raw_before = feedback.raw_input
    model = FakeModelProvider([FakeResponse(text="not json password=output-secret")])

    result = await FeedbackCompiler(model).compile(
        task=task,
        episode=episode,
        feedback=feedback,
        result_text="report text",
    )

    assert result.status == "UNAVAILABLE"
    assert "JSON/schema" in (result.error or "")
    assert "output-secret" not in (result.error or "")
    assert feedback.raw_input == raw_before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_text", "expected_error"),
    [
        ("", "no completed text"),
        (
            json.dumps(
                {
                    "status": "AVAILABLE",
                    "items": [],
                    "warnings": [],
                    "hard_constraint_verdict": "PASS",
                }
            ),
            "JSON/schema",
        ),
    ],
)
async def test_empty_or_schema_invalid_output_degrades_to_unavailable(
    response_text: str,
    expected_error: str,
) -> None:
    task, episode, feedback = _context()
    model = FakeModelProvider([FakeResponse(text=response_text)])

    result = await FeedbackCompiler(model).compile(
        task=task,
        episode=episode,
        feedback=feedback,
        result_text="report text",
    )

    assert result.status == "UNAVAILABLE"
    assert expected_error in (result.error or "")
    assert result.items == ()


@pytest.mark.asyncio
async def test_compiler_redacts_secret_bearing_structured_output() -> None:
    task, episode, feedback = _context()
    response = _available_response(
        source_span="Authorization: Bearer structured-output-secret"
    )

    result = await FeedbackCompiler(
        FakeModelProvider([FakeResponse(text=response)])
    ).compile(
        task=task,
        episode=episode,
        feedback=feedback,
        result_text="report text",
    )

    assert result.status == "AVAILABLE"
    assert "structured-output-secret" not in result.model_dump_json()
    assert "[REDACTED]" in result.items[0].source_span


@pytest.mark.asyncio
async def test_compilation_is_deeply_immutable_and_json_round_trips() -> None:
    task, episode, feedback = _context()
    result = await FeedbackCompiler(
        FakeModelProvider([FakeResponse(text=_available_response())])
    ).compile(
        task=task,
        episode=episode,
        feedback=feedback,
        result_text="report text",
    )

    with pytest.raises(ValidationError):
        result.items[0].problem = "rewritten"
    with pytest.raises(TypeError):
        result.items[0].requested_actions[0] = "rewritten"
    with pytest.raises(TypeError):
        result.items[0] = result.items[0]
    with pytest.raises(ValidationError):
        result.items = ()
    with pytest.raises(ValidationError):
        result.warnings = ("rewritten",)

    restored = type(result).model_validate_json(result.model_dump_json())
    assert restored == result
    assert isinstance(restored.items, tuple)
    assert isinstance(restored.items[0].requested_actions, tuple)
    assert isinstance(restored.items[0].preserve, tuple)
    assert isinstance(restored.warnings, tuple)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("responsible_module", "x" * 4_001),
        ("problem", "x" * 4_001),
        ("requested_actions", ["x" * 4_001]),
        ("acceptance_test", "x" * 4_001),
        ("preserve", ["x" * 4_001]),
        ("source_span", "x" * 4_001),
    ],
)
async def test_every_model_controlled_item_string_is_bounded(
    field: str,
    value: object,
) -> None:
    task, episode, feedback = _context()
    payload = json.loads(_available_response())
    payload["items"][0][field] = value

    result = await FeedbackCompiler(
        FakeModelProvider([FakeResponse(text=json.dumps(payload))])
    ).compile(
        task=task,
        episode=episode,
        feedback=feedback,
        result_text="report text",
    )

    assert result.status == "UNAVAILABLE"
    assert "JSON/schema" in (result.error or "")


@pytest.mark.asyncio
async def test_model_controlled_warning_strings_are_bounded() -> None:
    task, episode, feedback = _context()
    payload = json.loads(_available_response())
    payload["warnings"] = ["x" * 4_001]

    result = await FeedbackCompiler(
        FakeModelProvider([FakeResponse(text=json.dumps(payload))])
    ).compile(
        task=task,
        episode=episode,
        feedback=feedback,
        result_text="report text",
    )

    assert result.status == "UNAVAILABLE"


@pytest.mark.asyncio
async def test_raw_provider_response_is_rejected_before_json_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task, episode, feedback = _context()
    response = _available_response() + (" " * 64_000)
    monkeypatch.setattr(
        feedback_module,
        "_extract_json_object",
        lambda text: (_ for _ in ()).throw(
            AssertionError(f"oversized response was parsed: {len(text)}")
        ),
    )

    result = await FeedbackCompiler(
        FakeModelProvider([FakeResponse(text=response)])
    ).compile(
        task=task,
        episode=episode,
        feedback=feedback,
        result_text="report text",
    )

    assert result.status == "UNAVAILABLE"
    assert "response exceeded" in (result.error or "")


def test_cli_compilation_rendering_is_bounded() -> None:
    task, episode, feedback = _context()
    item = json.loads(_available_response())["items"][0]
    item["problem"] = "x" * 4_000
    payload = {
        "compilation_id": "comp_render",
        "evolution_id": task.evolution_id,
        "feedback_id": feedback.feedback_id,
        "episode_version": episode.version,
        "status": "AVAILABLE",
        "items": [
            {**item, "problem": f"item-{index}-" + ("x" * 3_980)}
            for index in range(100)
        ],
        "warnings": [f"warning-{index}-" + ("x" * 3_900) for index in range(100)],
        "provider": "fake",
        "model": "fake",
    }
    compilation = evolve_module.FeedbackCompilation.model_validate(payload)
    output = Console(
        file=io.StringIO(),
        record=True,
        force_terminal=False,
        width=200,
    )

    evolve_module._render_compilation(output, compilation)
    rendered = output.export_text()

    assert len(rendered) < 30_000
    assert "item-0" in rendered
    assert "item-99" not in rendered
    assert "warning-0" in rendered
    assert "warning-99" not in rendered


@pytest.mark.asyncio
async def test_provider_failure_records_only_bounded_redacted_provenance() -> None:
    task, episode, feedback = _context()

    class BrokenProvider:
        provider = "provider password=provider-secret"
        model = "model Authorization: Bearer model-secret"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            raise RuntimeError("password=error-secret " + "x" * 2_000)
            yield  # pragma: no cover

    result = await FeedbackCompiler(BrokenProvider()).compile(
        task=task,
        episode=episode,
        feedback=feedback,
        result_text="report text",
    )

    serialized = result.model_dump_json()
    assert result.status == "UNAVAILABLE"
    assert "provider failed" in (result.error or "")
    assert len(result.error or "") <= 1_000
    assert "provider-secret" not in serialized
    assert "model-secret" not in serialized
    assert "error-secret" not in serialized
    assert "[REDACTED]" in serialized


def _recorded_feedback(
    tmp_path: Path,
) -> tuple[EvolutionService, EvolutionTask, EpisodeRecord, ExpertFeedbackRecord]:
    service = EvolutionService(EvolutionStore(Workspace(tmp_path)))
    task = service.create_task(
        goal="Produce a report",
        target=TargetSpec(goal="Produce a report"),
        evolution_id="evo_compile_store",
    ).entity
    reserved = service.reserve_episode(task.evolution_id, mode="NORMAL").entity
    running = service.mark_episode_running(task.evolution_id, reserved.version).entity
    content = b"reviewable result\n"
    relative = "user_output/evo_compile_store/v001/result.md"
    result_path = service.store.workspace.resolve(relative, must_exist=False)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_bytes(content)
    completed = service.complete_episode(
        task.evolution_id,
        running.version,
        result=running.model_copy(
            update={
                "artifact": ArtifactRef(
                    path=relative,
                    size_bytes=len(content),
                    sha256=hashlib.sha256(content).hexdigest(),
                )
            }
        ),
    ).entity
    attached = service.attach_feedback(
        task.evolution_id,
        completed.version,
        feedback_id="fb_compile_store",
        draft=ExpertFeedbackDraft(
            scores=RubricScores(
                scientific_correctness=3,
                evidence_sufficiency=2,
                novelty=3,
                actionability=3,
                overall=3,
            ),
            comments="摘要够吗？",
        ),
        result_sha256=completed.artifact.sha256,
        raw_input="摘要够吗？",
    ).entity
    return service, service.get(task.evolution_id), completed, attached


@pytest.mark.asyncio
async def test_save_compilation_is_separate_transactional_and_idempotent(
    tmp_path: Path,
) -> None:
    service, task, episode, feedback = _recorded_feedback(tmp_path)
    feedback_path = (
        service.store.root
        / task.evolution_id
        / "feedback"
        / f"{feedback.feedback_id}.json"
    )
    feedback_bytes = feedback_path.read_bytes()
    compilation = await FeedbackCompiler(
        FakeModelProvider([FakeResponse(text=_available_response())])
    ).compile(task=task, episode=episode, feedback=feedback, result_text="reviewable result")

    first = service.save_compilation(task.evolution_id, compilation)
    revision_after_first = service.get(task.evolution_id).revision
    second = service.save_compilation(task.evolution_id, compilation)
    stored_task = service.get(task.evolution_id)

    assert first.entity == second.entity
    assert stored_task.status == "FEEDBACK_RECORDED"
    assert stored_task.compilation_ids == [compilation.compilation_id]
    assert stored_task.revision == revision_after_first
    assert feedback_path.read_bytes() == feedback_bytes
    stored_feedback = service.store.load_feedback(
        task.evolution_id, feedback.feedback_id
    )
    assert stored_feedback.compilation_id is None
    assert service.store.load_compilation(
        task.evolution_id, compilation.compilation_id
    ) == compilation


@pytest.mark.asyncio
async def test_save_compilation_rejects_wrong_feedback_without_writing(tmp_path: Path) -> None:
    service, task, episode, feedback = _recorded_feedback(tmp_path)
    compilation = await FeedbackCompiler(
        FakeModelProvider([FakeResponse(text=_available_response())])
    ).compile(task=task, episode=episode, feedback=feedback, result_text="reviewable result")
    wrong = compilation.model_copy(update={"feedback_id": "fb_other"})

    with pytest.raises(EvolutionOperationConflict, match="active feedback"):
        service.save_compilation(task.evolution_id, wrong)

    assert service.get(task.evolution_id).compilation_ids == []
    assert service.store.list_compilations(task.evolution_id) == []


def test_evolve_compile_retries_same_feedback_without_constructing_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, task, _episode, feedback = _recorded_feedback(tmp_path)
    providers = iter(
        [
            FakeModelProvider([FakeResponse(text="not json")]),
            FakeModelProvider([FakeResponse(text=_available_response())]),
        ]
    )
    monkeypatch.setattr(
        evolve_module,
        "_build_feedback_compiler",
        lambda *_: FeedbackCompiler(next(providers)),
    )
    monkeypatch.setattr(
        "photomatagent.cli.chat.build_runtime",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError(f"compile constructed runtime: {kwargs}")
        ),
    )
    runner = CliRunner()
    args = [
        "evolve",
        "compile",
        task.evolution_id,
        "--version",
        "v001",
        "--provider",
        "fake",
        "--workspace",
        str(tmp_path),
    ]

    unavailable = runner.invoke(app, args)
    available = runner.invoke(app, args)
    stored = service.get(task.evolution_id)
    compilations = [
        service.store.load_compilation(task.evolution_id, compilation_id)
        for compilation_id in stored.compilation_ids
    ]

    assert unavailable.exit_code == 1, unavailable.output
    assert "UNAVAILABLE" in unavailable.output
    assert available.exit_code == 0, available.output
    assert "AVAILABLE" in available.output
    assert stored.status == "FEEDBACK_RECORDED"
    assert stored.feedback_ids == [feedback.feedback_id]
    assert len(stored.compilation_ids) == 2
    assert [item.status for item in compilations] == ["UNAVAILABLE", "AVAILABLE"]
