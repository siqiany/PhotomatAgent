from __future__ import annotations

import hashlib
from collections import deque
from pathlib import Path

import pytest
from pydantic import ValidationError
from rich.console import Console

from photomatagent.cli.evolve import (
    FeedbackEntryCancelled,
    collect_expert_feedback,
    load_feedback_file,
    run_feedback_flow,
)
from photomatagent.scientific.evolution.models import ArtifactRef
from photomatagent.scientific.evolution.service import (
    EvolutionService,
    InvalidEvolutionTransition,
)
from photomatagent.scientific.evolution.store import EvolutionStore
from photomatagent.scientific.loop import ScientificLoopSummary, TargetSpec
from photomatagent.workspace import Workspace


class ScriptedPrompt:
    def __init__(self, *answers: str) -> None:
        self.answers = deque(answers)
        self.prompt_history: list[str] = []

    async def prompt_async(self, message: str) -> str:
        self.prompt_history.append(message)
        return self.answers.popleft()


def _completed_service(tmp_path: Path) -> tuple[EvolutionService, str, str]:
    service = EvolutionService(EvolutionStore(Workspace(tmp_path)))
    task = service.create_task(
        goal="review this result",
        target=TargetSpec(goal="review this result"),
        evolution_id="evo_feedback_entry",
    ).entity
    reserved = service.reserve_episode(task.evolution_id, mode="NORMAL").entity
    running = service.mark_episode_running(task.evolution_id, reserved.version).entity
    content = b"reviewable result\n"
    relative = "user_output/evo_feedback_entry/v001/result.md"
    artifact_path = service.store.workspace.resolve(relative, must_exist=False)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(content)
    completed = service.complete_episode(
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
    ).entity
    assert completed.artifact is not None
    return service, task.evolution_id, completed.artifact.sha256


@pytest.mark.asyncio
async def test_feedback_form_collects_every_field_with_distinct_chinese_rubric(
    tmp_path: Path,
) -> None:
    session = ScriptedPrompt(
        "3.0",
        "4",
        "2",
        "3",
        "2",
        "2",
        "y",
        "n",
        "yes",
        "",
        "n",
        "y",
        "证据不足",
        "需要补充原始数据",
        "/submit",
    )
    console = Console(file=(tmp_path / "console.txt").open("w", encoding="utf-8"))

    draft = await collect_expert_feedback(
        session=session,  # type: ignore[arg-type]
        console=console,
        evolution_id="evo_test",
        version="v001",
    )

    assert draft.scores.model_dump() == {
        "scientific_correctness": 4,
        "evidence_sufficiency": 2,
        "novelty": 3,
        "actionability": 2,
        "overall": 2,
    }
    assert draft.flags.model_dump() == {
        "fabricated_source": True,
        "conclusion_changing_error": False,
        "abstract_only_core_evidence": True,
        "unsupported_novelty": False,
        "process_parameters_only": False,
    }
    assert draft.fatal_issue is True
    assert draft.comments == "证据不足\n需要补充原始数据"
    assert all("EXPERT FEEDBACK | evo_test | v001" in prompt for prompt in session.prompt_history)
    rendered = (tmp_path / "console.txt").read_text(encoding="utf-8")
    assert "expert-review-v1" in rendered
    assert "科学正确性" in rendered
    assert "多源可审计证据链" in rendered
    assert "请输入 1–5 的整数" in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answers",
    [
        ("/cancel",),
        ("3", "3", "3", "3", "3", "/cancel"),
        ("3", "3", "3", "3", "3", "n", "n", "n", "n", "n", "/cancel"),
        ("3", "3", "3", "3", "3", "n", "n", "n", "n", "n", "n", "/cancel"),
    ],
)
async def test_cancel_is_honored_at_every_form_phase(
    answers: tuple[str, ...], tmp_path: Path
) -> None:
    session = ScriptedPrompt(*answers)
    console = Console(file=(tmp_path / "console.txt").open("w", encoding="utf-8"))

    with pytest.raises(FeedbackEntryCancelled):
        await collect_expert_feedback(
            session=session,  # type: ignore[arg-type]
            console=console,
            evolution_id="evo_test",
            version="v001",
        )


def test_feedback_file_uses_strict_draft_schema(tmp_path: Path) -> None:
    valid = tmp_path / "valid.json"
    valid.write_text(
        '{"scores":{"scientific_correctness":4,"evidence_sufficiency":3,'
        '"novelty":2,"actionability":3,"overall":3},"comments":"review"}',
        encoding="utf-8",
    )
    assert load_feedback_file(valid).scores.scientific_correctness == 4

    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        valid.read_text(encoding="utf-8")[:-1] + ',"unexpected":true}',
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_feedback_file(invalid)


@pytest.mark.asyncio
async def test_feedback_flow_requires_override_then_confirmation_before_write(
    tmp_path: Path,
) -> None:
    service, evolution_id, digest = _completed_service(tmp_path)
    session = ScriptedPrompt(
        "5", "5", "5", "5", "5",
        "y", "n", "n", "n", "n", "n",
        "review", "/submit",
        "专家复核了原始来源，判定标记来自自动误报",
        "y",
    )
    console = Console(file=(tmp_path / "console.txt").open("w", encoding="utf-8"))

    record = await run_feedback_flow(
        session=session,  # type: ignore[arg-type]
        console=console,
        service=service,
        evolution_id=evolution_id,
        version="v001",
    )

    assert record is not None
    assert record.result_sha256 == digest
    assert record.suggested_scores is not None
    assert record.suggested_scores.evidence_sufficiency == 1
    assert record.hard_cap_override_reason == "专家复核了原始来源，判定标记来自自动误报"
    assert service.get(evolution_id).status == "FEEDBACK_RECORDED"
    assert len(service.get(evolution_id).feedback_ids) == 1
    output = (tmp_path / "console.txt").read_text(encoding="utf-8")
    assert evolution_id in output
    assert digest in output
    assert "Comments length" in output


@pytest.mark.asyncio
async def test_feedback_flow_cancel_at_confirmation_writes_nothing(tmp_path: Path) -> None:
    service, evolution_id, _ = _completed_service(tmp_path)
    session = ScriptedPrompt(
        "3", "3", "3", "3", "3",
        "n", "n", "n", "n", "n", "n",
        "review", "/submit", "/cancel",
    )
    console = Console(file=(tmp_path / "console.txt").open("w", encoding="utf-8"))

    result = await run_feedback_flow(
        session=session,  # type: ignore[arg-type]
        console=console,
        service=service,
        evolution_id=evolution_id,
        version="v001",
    )

    assert result is None
    assert service.get(evolution_id).feedback_ids == []
    assert service.store.list_feedback(evolution_id) == []


@pytest.mark.asyncio
async def test_raw_feedback_is_persisted_only_in_feedback_record_not_event_or_output(
    tmp_path: Path,
) -> None:
    service, evolution_id, _ = _completed_service(tmp_path)
    events: list[object] = []
    service.event_sink = events.append  # type: ignore[assignment]
    marker = "RAW-EXPERT-PROSE-MUST-STAY-ISOLATED"
    session = ScriptedPrompt(
        "3", "3", "3", "3", "3",
        "n", "n", "n", "n", "n", "n",
        marker, "/submit", "y",
    )
    output_path = tmp_path / "console.txt"
    console = Console(file=output_path.open("w", encoding="utf-8"))

    record = await run_feedback_flow(
        session=session,  # type: ignore[arg-type]
        console=console,
        service=service,
        evolution_id=evolution_id,
        version="v001",
    )

    assert record is not None
    assert marker in record.raw_input
    assert marker not in output_path.read_text(encoding="utf-8")
    assert all(marker not in repr(event) for event in events)


def test_service_invalid_state_never_leaves_orphan_feedback(tmp_path: Path) -> None:
    service, evolution_id, digest = _completed_service(tmp_path)
    service.accept(evolution_id, "v001")
    review_file = tmp_path / "review.json"
    review_file.write_text(
        '{"scores":{"scientific_correctness":3,"evidence_sufficiency":3,'
        '"novelty":3,"actionability":3,"overall":3}}',
        encoding="utf-8",
    )

    with pytest.raises(InvalidEvolutionTransition):
        service.attach_feedback(
            evolution_id,
            "v001",
            feedback_id="fb_invalid_state",
            draft=load_feedback_file(review_file),
            result_sha256=digest,
        )

    assert service.store.list_feedback(evolution_id) == []
