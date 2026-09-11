from __future__ import annotations

import json
from pathlib import Path

import pytest

from photomatagent.models.fake import FakeModelProvider, FakeResponse
from photomatagent.scientific.evolution.targeting import (
    ConfirmedTargetStore,
    TargetSpecCompiler,
)
from photomatagent.scientific.state import ScientificState
from photomatagent.workspace import Workspace


@pytest.mark.asyncio
async def test_target_compiler_is_tool_free_and_does_not_receive_result() -> None:
    response = {
        "goal": "设计中红外光窗材料",
        "constraints": [
            {
                "property": "mid_ir_transmittance",
                "operator": "ge",
                "value": 0.8,
                "unit": "fraction",
                "severity": "SOFT",
                "weight": 1.0,
                "description": "中红外透过率目标",
                "basis": "INFERRED_CONTEXT",
                "rationale": "任务要求面向中红外光窗",
                "confidence": 0.7,
                "requires_confirmation": True,
            }
        ],
        "objectives": ["给出完整工艺路线"],
        "operating_conditions": {},
        "warnings": ["用户没有给出明确透过率阈值"],
    }
    model = FakeModelProvider([FakeResponse(text=json.dumps(response, ensure_ascii=False))])
    draft = await TargetSpecCompiler(model).compile(
        goal="设计中红外光窗材料",
        scientific_state=ScientificState(
            goal="设计中红外光窗材料",
            open_questions=["需要验证中红外透过率"],
        ),
        correction=None,
    )

    assert draft.target.constraints[0].severity == "SOFT"
    assert model.requests[0].tools == []
    request_text = "\n".join(message.content for message in model.requests[0].messages)
    assert "SECRET_RESULT_SENTINEL" not in request_text
    assert "需要验证中红外透过率" in request_text


def test_confirmed_target_store_reloads_only_matching_history_context(tmp_path: Path) -> None:
    store = ConfirmedTargetStore(Workspace(tmp_path))
    state = ScientificState(goal="goal", open_questions=["verify stability"])
    target = {
        "goal": "goal",
        "constraints": [
            {"property": "stability", "operator": "ge", "value": 1}
        ],
    }
    record = store.save(
        session_id="session/history:1",
        goal="goal",
        scientific_state=state,
        target=target,
        provider="fake",
        model="fake",
    )

    assert store.load("session/history:1", goal="goal", scientific_state=state) == record
    assert store.load("session/history:1", goal="changed", scientific_state=state) is None
    assert len(list((tmp_path / ".photomatagent" / "evolution-targets").glob("*.json"))) == 1


def test_confirmed_target_store_treats_corrupt_or_late_changed_context_as_miss(
    tmp_path: Path,
) -> None:
    store = ConfirmedTargetStore(Workspace(tmp_path))
    state = ScientificState(goal="goal", open_questions=["x" * 17_000, "old"])
    target = {
        "goal": "goal",
        "constraints": [{"property": "x", "operator": "ge", "value": 1}],
    }
    store.save(
        session_id="session-long",
        goal="goal",
        scientific_state=state,
        target=target,
        provider="fake",
        model="fake",
    )
    changed = ScientificState(goal="goal", open_questions=["x" * 17_000, "new"])
    assert store.load("session-long", goal="goal", scientific_state=changed) is None

    cache_file = next((tmp_path / ".photomatagent" / "evolution-targets").glob("*.json"))
    cache_file.write_text("{broken", encoding="utf-8")
    assert store.load("session-long", goal="goal", scientific_state=state) is None
