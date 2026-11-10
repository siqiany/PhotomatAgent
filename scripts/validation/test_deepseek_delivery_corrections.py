"""Regression checks for scoped scientific delivery and wire-format recovery."""
from __future__ import annotations

import pytest

from photomatagent.models.fake import FakeModelProvider, FakeResponse, scripted_tool_call
from photomatagent.models.openai import OpenAIProvider, OpenAIStreamMapper
from photomatagent.runtime.budget import BudgetState
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import AutoApproveHandler
from photomatagent.scientific.capabilities.photodetector import CheckTargetsTool
from photomatagent.scientific.capabilities.photodetector.delivery import DetectorCheckReceipt
from photomatagent.scientific.process import ProcessSample
from photomatagent.scientific.process_delivery import review_architecture_claim
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.factory import create_default_registry
from photomatagent.tools.registry import ToolRegistry
from photomatagent.tools.surface import ToolSurfaceConfig
from photomatagent.workspace import Workspace


def test_structure_uncertainty_keeps_hypotheses_and_references_separate():
    sample = {"sample_id": "film", "label": "film", "role": "input"}
    assert ProcessSample.model_validate(sample | {"architecture_status": "UNKNOWN"}).architecture is None
    with pytest.raises(ValueError, match="unknown architecture"):
        ProcessSample.model_validate(sample | {"architecture_status": "UNKNOWN", "architecture": "single phase"})
    with pytest.raises(ValueError, match="sample-bound evidence"):
        ProcessSample.model_validate(sample | {"architecture_status": "OBSERVED", "architecture": "single phase"})
    assert ProcessSample.model_validate(sample | {
        "architecture_status": "HYPOTHESIS", "architecture": "radial gradient",
    }).architecture == "radial gradient"
    assert review_architecture_claim("If the spatial distribution is unconfirmed, classify it as homogeneous.")
    assert review_architecture_claim("空间分布未确认，可仅按均匀模型做参照。") is None
    assert review_architecture_claim("> 原报告称梯度未证实，应按均匀合金评估。") is None


async def test_unverified_architecture_draft_is_withheld(tmp_path):
    workspace = Workspace(tmp_path)
    state = ScientificState()
    wrong = "若径向组分剖面未证实梯度，应按均匀合金评估。"
    right = "径向分布尚未确认，结构身份保持未知。均匀合金只作对照。"
    runner = AgentRuntime(model=FakeModelProvider([FakeResponse(text=wrong), FakeResponse(text=right)]),
                          tools=create_default_registry(state, workspace), workspace=workspace, scientific_state=state)
    events = [event async for event in runner.run("请审查空间组成梯度。")]
    assert [event.text for event in events if event.kind == "text_delta"] == [right]
    assert wrong not in str(runner.conversation_state.messages)
    assert any(getattr(event, "finish_reason", "") == "delivery_rejected" for event in events)


async def test_actual_checker_conclusion_guards_the_final_prose(tmp_path):
    workspace = Workspace(tmp_path)
    state = ScientificState()
    values = {"spectral_min_um": 1, "spectral_max_um": 5, "target_responsivity_a_w": 0.3,
              "mode": "at_wavelength", "wavelength_um": 2, "eqe_percent": 12, "eqe_relation": "minimum"}
    wrong = "故 EQE≥12% 与 2 μm 的响应度下限仍互斥。"
    right = "两个下限相容。固定 EQE=12% 时可以达不到响应度要求，实际器件尚未验证。"
    tool = CheckTargetsTool()
    check = (await tool.execute(values)).data
    assert check["mutually_consistent"] is True
    review = DetectorCheckReceipt.model_validate(check).review
    assert not review("EQE 下限与响应度下限并不冲突。")
    assert not review("EQE 下限和固定值不足不是目标冲突。")
    assert not review("若假定上限为 12%，EQE 下限可能冲突。")
    assert not review("1.1 μm 的 EQE 与响应度下限有冲突。")
    assert not review("原报告写两个 EQE 下限冲突。")
    assert not review("**修正**：原报告把 EQE 下限与 2 μm 的响应度下限称为冲突。")
    assert not review("若把 EQE 下限误读为固定工况，则认为两个下限冲突。")
    assert not review("2 μm 的联合 EQE 下限高于 12% 基准。")
    assert not review("流速下限与停留时间要求有冲突。")
    assert review("联合下限：2 μm 12%、4 μm 10%。")
    assert "CHECKER_UNSCOPED_JOINT_BOUNDS" in review("| 计算值 | EQE 联合下界（12%/15%/18%） |")
    assert not review("固定 EQE 情景（12%/15%/18%）。")
    assert review("**联合下限**：**2 μm 处 EQE ≥ 12%**、4 μm 处 **≥ 10%**。")
    table = (
        "| 测点 | 固定 EQE 情景 | 联合 EQE 下限 |\n"
        "|---|---|---|\n"
        "| 4 μm | 12% | 12% |\n"
        "| 2 μm | 12% | 12%（回执 18.6%） |"
    )
    assert review(table)
    assert not review(table.replace("12%（回执 18.6%）", "18.6%"))
    assert not review(table.replace("联合 EQE 下限", "联合 EQE 下限（R ≥ 0.2 A/W）"))
    target_rows = (
        "| 波长 | R 目标 | EQE 硬下限（联合） |\n| --- | --- | --- |\n"
        "| 2 μm | ≥0.3 A/W | 12% |\n| 2 μm | ≥0.2 A/W | 12% |"
    )
    assert DetectorCheckReceipt.model_validate(check).review(target_rows, explicit_target=True)
    assert not DetectorCheckReceipt.model_validate(check).review(
        target_rows.replace("≥0.3 A/W | 12%", "≥0.3 A/W | 18.6%"), explicit_target=True,
    )
    responses = [
        scripted_tool_call("tool_describe", {"name": tool.name}),
        scripted_tool_call("tool_call", {"name": tool.name, "arguments": values}),
        FakeResponse(text=wrong), FakeResponse(text=right),
    ]
    runner = AgentRuntime(model=FakeModelProvider(responses), tools=create_default_registry(state, workspace),
                          workspace=workspace, scientific_state=state, approval_handler=AutoApproveHandler())
    events = [event async for event in runner.run("Check target compatibility.")]
    delivered = [event.text for event in events if event.kind == "text_delta"]
    assert len(delivered) == 1 and delivered[0].startswith(right)
    assert "光电 Checker 回执" in delivered[0]
    assert any(getattr(event, "finish_reason", "") == "delivery_rejected" for event in events)


async def test_exhausted_review_preserves_usable_proposal_and_checker_facts(tmp_path):
    workspace = Workspace(tmp_path)
    state = ScientificState()
    values = {"spectral_min_um": 1, "spectral_max_um": 5, "target_responsivity_a_w": 0.3,
              "mode": "at_wavelength", "wavelength_um": 2, "eqe_percent": 12, "eqe_relation": "minimum"}
    tool = CheckTargetsTool()
    wrong = "EQE≥12% 与 2 μm 的响应度下限仍互斥。"
    valid = "候选方案：保留同批薄膜作对照，先测噪声与膜厚，再评估接触工艺。"
    runner = AgentRuntime(
        model=FakeModelProvider([
            scripted_tool_call("tool_describe", {"name": tool.name}),
            scripted_tool_call("tool_call", {"name": tool.name, "arguments": values}),
            FakeResponse(text=valid + "\n\n" + wrong),
        ]),
        tools=create_default_registry(state, workspace), workspace=workspace, scientific_state=state,
        approval_handler=AutoApproveHandler(), budget=BudgetState(max_iterations=3),
    )
    events = [event async for event in runner.run("Review the process and two targets.")]
    delivered = next(event.text for event in events if event.kind == "text_delta" and event.research_delivery)
    assert valid in delivered and wrong not in delivered
    assert "光电 Checker 回执" in delivered and "PARTIAL" in delivered
    assert state.research_delivery and state.research_delivery.status == "PARTIAL"


async def test_registration_stall_gets_one_tool_free_proposal_synthesis(tmp_path):
    workspace = Workspace(tmp_path)
    state = ScientificState()
    answer = "条件方案：先用同批基片比较湿膜回缩，再用测得的黏度决定走刀范围。"
    model = FakeModelProvider([
        scripted_tool_call("process.register_plan", {}),
        FakeResponse(text=answer),
    ])
    runner = AgentRuntime(
        model=model, tools=create_default_registry(state, workspace), workspace=workspace, scientific_state=state,
        approval_handler=AutoApproveHandler(), budget=BudgetState(max_iterations=4),
        tool_surface_config=ToolSurfaceConfig(mode="eager"), max_no_progress_iterations=1,
    )
    events = [event async for event in runner.run("Compare two film routes.")]
    assert [event.text for event in events if event.kind == "text_delta"] == [answer]
    assert model.requests[-1].tools == []
    assert any(getattr(event, "reason", "") == "final_response" for event in events)


async def test_registered_draft_does_not_replace_candidate_synthesis(tmp_path):
    workspace = Workspace(tmp_path)
    state = ScientificState()
    answer = "Compare edge and center measurements on the same-batch specimen."
    draft = {"decision_question": "Initial trial", "sample": "film",
             "steps": [{"operation": "inspect", "purpose": "Review the sample"}]}
    model = FakeModelProvider([
        scripted_tool_call("process.register_plan", draft),
        scripted_tool_call("process.register_plan", {}),
        FakeResponse(text=answer),
    ])
    runner = AgentRuntime(
        model=model, tools=create_default_registry(state, workspace), workspace=workspace, scientific_state=state,
        approval_handler=AutoApproveHandler(), budget=BudgetState(max_iterations=5),
        tool_surface_config=ToolSurfaceConfig(mode="eager"), max_no_progress_iterations=1,
    )
    events = [event async for event in runner.run("Design a conditional film trial.")]
    assert len(state.process_plans) == 1 and state.process_plans[0].proposal is None
    assert model.requests[-1].tools == []
    assert any(event.kind == "text_delta" and answer in event.text for event in events)
    assert any(getattr(event, "reason", "") == "research_partial" for event in events)


def test_split_wire_markers_are_not_text_or_native_calls():
    mapper = OpenAIStreamMapper("deepseek-v4-flash")
    events = []
    for text in ("Progress <", "｜", "｜DSM", "L｜｜ calls>malicious arguments"):
        events += mapper.feed({"type": "response.output_text.delta", "delta": text})
    events += mapper.feed({"type": "response.completed", "response": {"output": []}})
    assert "".join(event.text for event in events if hasattr(event, "text")) == "Progress "
    assert events[-1].response.finish_reason == "invalid_tool_markup"
    assert events[-1].response.tool_calls == []


async def test_marker_only_response_retries_within_existing_runtime_budget(tmp_path):
    class Responses:
        requests = []

        async def create(self, **kwargs):
            self.requests.append(kwargs)
            output = "<｜｜DSML｜｜ calls>bogus" if len(self.requests) == 1 else "Use a matched coating control."

            async def stream():
                yield {"type": "response.output_text.delta", "delta": output}
                yield {"type": "response.completed", "response": {"output": []}}

            return stream()

    class Client:
        responses = Responses()

    client = Client()
    runner = AgentRuntime(model=OpenAIProvider("deepseek-v4-flash", client=client), tools=ToolRegistry(),
                          workspace=Workspace(tmp_path), budget=BudgetState(max_iterations=4))
    events = [event async for event in runner.run("Suggest a coating validation.")]
    assert [event.text for event in events if event.kind == "text_delta"] == ["Use a matched coating control."]
    assert runner.budget.model_calls == 2 and runner.budget.tool_calls == 0
    assert "markup in answer text was discarded" in str(client.responses.requests[-1]["input"])
    assert any(getattr(event, "reason", "") == "final_response" for event in events)
