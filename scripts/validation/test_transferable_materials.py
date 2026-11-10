"""Engineering acceptance of transferable process contracts."""
from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from photomatagent.models.fake import FakeModelProvider, FakeResponse, scripted_tool_call
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import AutoApproveHandler, default_permission_policy
from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.process import ProcessParameter, ProcessPlan
from photomatagent.scientific.state import EvidenceAttestation, ScientificState
from photomatagent.sessions.store import load_session_snapshot, save_session_snapshot
from photomatagent.tools.factory import create_default_registry
from photomatagent.tools.process_plan import RegisterProcessPlanTool
from photomatagent.tools.scientific_state_inspect import ScientificStateInspectTool
from photomatagent.workspace import Workspace


VALIDATION = {"control": "Matched baseline", "procedure": "Compare on matched specimens",
              "metric": "Across-specimen variation", "criterion": "Check reproducibility and damage together",
              "next_action": "Characterize control specimens"}
PROPOSAL = {"kind": "CANDIDATE", "baseline": "Baseline route", "change": "Alter only the stated treatment",
            "rationale": "Explore whether changing the interface changes the result",
            "variables": ["Treatment intensity"], "validation": VALIDATION}


def quantity(name, amount, unit, **extra):
    return {"name": name, "basis": "PROPOSED", "unit": unit, "value": {"kind": "quantity", "amount": amount}, **extra}


def outcome(result_id, **extra):
    return {"result_id": result_id, "metric": result_id, "kind": "QUALITATIVE",
            "basis": "Directional hypothesis only", **extra}


def link(source, target, relation="DEPENDS_ON"):
    return {"source": {"kind": source[0], "id": source[1]}, "target": {"kind": target[0], "id": target[1]},
            "relation": relation, "explanation": "Explicit scope of the connection"}


async def test_custom_operation_accepts_typed_values_without_a_method_card(tmp_path):
    workspace = Workspace(tmp_path)
    state = ScientificState()
    payload = {
        "decision_question": "Can alternating gas pulses preserve a uniform film?", "sample": "aqueous film",
        "steps": [{"operation": "pulsed gas spread with vacuum relaxation", "operation_tags": ["coating"],
                   "purpose": "Explore a different wetting front",
                   "parameters": [
                       quantity("dose", [1, 2], "mL", intent="EXPLORATION", measurement_basis="per exposed 4 cm²"),
                       {"name": "carrier gas", "basis": "PROPOSED", "value": {"kind": "category", "label": "inert gas"}},
                       {"name": "pressure", "basis": "PROPOSED", "unit": "kPa", "value": {
                           "kind": "timeline", "time_unit": "s", "points": [{"time": 0, "value": 98}, {"time": 2, "value": 95}],
                       }},
                       {"name": "endpoint", "basis": "PROPOSED", "value": {
                           "kind": "endpoint", "observable": "no visible wetting front", "operator": "observed",
                       }},
                       {"name": "hold time", "basis": "PROPOSED", "value": {
                           "kind": "variable", "symbol": "t_hold", "resolution": "Measure platform response",
                       }},
                   ], "required_inputs": ["hold time"]}],
        "proposal": PROPOSAL,
    }
    plan_id = ProcessPlan.model_validate(payload).id
    call = lambda name, args: scripted_tool_call("tool_call", {"name": name, "arguments": args})
    provider = FakeModelProvider([
        scripted_tool_call("tool_describe", {"name": "process.register_plan"}),
        call("process.register_plan", payload),
        scripted_tool_call("tool_describe", {"name": "decision.record"}),
        call("decision.record", {"question": "Should the custom route be tested?", "process_plan_ids": [plan_id],
                                 "research_stop": {"reason": "VALIDATION_REQUIRED", "rationale": "A wetting test resolves the gap."}}),
    ])
    runtime = AgentRuntime(model=provider, tools=create_default_registry(state, workspace),
                           workspace=workspace, scientific_state=state, permission_policy=default_permission_policy(),
                           approval_handler=AutoApproveHandler())
    events = [event async for event in runtime.run("Explore a custom coating operation.")]
    assert not [event for event in events if event.kind == "tool_failed"]
    answer = next(event.text for event in events if event.kind == "text_delta" and event.research_delivery)
    for detail in ("pulsed gas spread", "per exposed 4 cm²", "inert gas", "no visible wetting front", "t_hold", "UNKNOWN"):
        assert detail in answer
    assert state.research_delivery.status == "READY"
    assert state.process_plans[0].proposal_readiness()["missing_inputs"] == ["hold time"]
    save_session_snapshot(tmp_path / "session", conversation=runtime.conversation_state, scientific=state)
    restored = load_session_snapshot(tmp_path / "session").scientific
    assert restored.process_plans[0].id == plan_id


@pytest.mark.parametrize("parameter", [
    {"name": "variable", "basis": "PROPOSED", "intent": "SETPOINT",
     "value": {"kind": "variable", "symbol": "t", "resolution": "measure"}},
    {"name": "pressure", "basis": "PROPOSED", "unit": "kPa",
     "value": {"kind": "timeline", "time_unit": "s", "points": [{"time": 2, "value": 1}, {"time": 1, "value": 0}]}},
    {"name": "rate", "basis": "PROPOSED", "value": {"kind": "quantity", "amount": 2}},
    {"name": "dose", "basis": "PROPOSED", "unit": "mL", "bounds": [1, 2],
     "value": {"kind": "quantity", "amount": [1, 2]}},
    quantity("temperature", [320, 380], "K", intent="SETPOINT"),
    quantity("dose", True, "mL"),
])
def test_process_parameters_reject_deterministic_shape_errors(parameter):
    with pytest.raises(ValidationError):
        ProcessParameter.model_validate(parameter)


async def test_spin_to_blade_marks_only_dependent_results(tmp_path):
    state = ScientificState()
    tool = RegisterProcessPlanTool(state, Workspace(tmp_path))
    spin = {
        "decision_question": "Which coating fixture?", "sample": "same ink",
        "steps": [{"step_id": "ink", "operation": "storage test", "purpose": "Retain ink control"},
                  {"step_id": "coating", "operation": "spin coating", "purpose": "Coat glass", "equipment": "spinner",
                   "parameters": [quantity("rotation speed", 800, "rpm")]}],
        "outcomes": [outcome("uniformity"), outcome("ink_stability"), outcome("untracked")],
        "links": [link(("result", "uniformity"), ("step", "coating")),
                  link(("result", "ink_stability"), ("step", "ink"))],
    }
    original = await tool.execute(spin)
    state.add_process_plan(original.state_updates[0])
    evidence = ScientificEvidence(subject="ink", property="storage note", value="same batch")
    state.add_evidence(evidence)
    blade = deepcopy(spin)
    blade.update(previous_plan_id=original.data["plan_id"], revision_basis="Spinner unavailable")
    blade["steps"][1] = {"step_id": "coating", "operation": "blade coating", "purpose": "Coat same glass",
                         "equipment": "adjustable doctor blade", "parameters": [quantity("gap", 80, "um")],
                         "required_inputs": ["gap", "traverse speed"]}
    revised = await tool.execute(blade)
    assert not revised.is_error and revised.data["missing_inputs"] == ["traverse speed"]
    assert revised.data["revision_impact"]["results_to_review"] == ["uniformity"]
    assert revised.data["revision_impact"]["unknown_impact"] == ["untracked"]
    assert state.evidence == [evidence]
    state.add_process_plan(revised.state_updates[0])
    detail = await ScientificStateInspectTool(state).execute({
        "section": "process_plans", "process_plan_id": original.data["plan_id"],
    })
    assert detail.data["revisions"][0]["results_to_review"] == ["uniformity"]
    assert detail.data["outcomes"][1]["result_id"] == "ink_stability"


async def test_electrodeposition_uses_its_own_basis_and_marks_envelope(tmp_path):
    tool = RegisterProcessPlanTool(ScientificState(), Workspace(tmp_path))
    process = {
        "decision_question": "Uniform copper coating with a constant-current cell", "sample": "copper coupon",
        "steps": [{"operation": "galvanostatic electrodeposition", "equipment": "three-electrode cell",
                   "purpose": "Compare charge efficiency and thickness map",
                   "parameters": [quantity("charge density", [12, 15], "C/cm2", measurement_basis="exposed geometric area")]}],
        "outcomes": [{"result_id": "thickness", "metric": "thickness", "kind": "PREDICTION", "relation": "RANGE",
                      "value": [4, 5], "unit": "um", "range_basis": "SCENARIO_ENVELOPE",
                      "conditions": {"exposed_area_cm2": 2, "bath": "approved copper bath"},
                      "basis": "Illustrative efficiency scenarios", "limitations": "Efficiency and edge profile unmeasured"}],
        "proposal": PROPOSAL,
    }
    result = await tool.execute(process)
    assert not result.is_error and result.data["missing_inputs"] == []
    assert result.state_updates[0].outcomes[0].range_basis == "SCENARIO_ENVELOPE"
    assert result.data["scientific_validation"] == "UNKNOWN"
    for changes in ({"range_basis": "STATISTICAL_INTERVAL"}, {"kind": "MEASURED"},
                    {"relation": "EXACT"}, {"relation": "MIN"}):
        altered = process | {"outcomes": [process["outcomes"][0] | changes]}
        assert (await tool.execute(altered)).is_error


async def test_core_shell_cannot_inherit_uniform_reference_coverage(tmp_path):
    tool = RegisterProcessPlanTool(ScientificState(), Workspace(tmp_path))
    process = {
        "decision_question": "Use radial-shell candidate or homogeneous baseline?", "sample": "radial shell",
        "samples": [{"sample_id": "dot", "label": "candidate", "role": "input", "architecture": "radial-gradient"}],
        "steps": [{"step_id": "compare", "operation": "heterostructure comparison", "purpose": "Scope model",
                   "input_samples": ["dot"]}],
        "outcomes": [outcome("gap", sample_id="dot")],
        "links": [link(("result", "gap"), ("method", "band-bowing"), "MECHANISM_HYPOTHESIS")],
        "proposal": PROPOSAL | {"unknowns": [
            {"cause": "Possible interdiffusion", "impact": "Shell loses contrast", "resolution": "Map composition",
             "basis": "MECHANISM", "affects": [{"kind": "sample", "id": "dot"}],
             "decision_effect": "Retain homogeneous control", "mitigation_cost": "Extra profiling"},
        ]},
    }
    result = await tool.execute(process)
    assert not result.is_error and result.data["proposal_delivery"] == "READY"
    assert result.data["method_domain_gaps"][0]["method_id"] == "band-bowing"
    assert result.data["scientific_validation"] == "UNKNOWN"
    alternative = process | {"links": [link(("result", "gap"), ("method", "invented"))]}
    assert (await tool.execute(alternative)).output == "PROCESS_METHOD_REFERENCE_UNAVAILABLE"


async def test_known_conflict_preserves_two_lower_bounds_and_condition_scope():
    def target(key, relation, value, **conditions):
        return {"result_id": key, "metric": "EQE", "kind": "REQUIREMENT", "relation": relation,
                "value": value, "unit": "%", "conditions": conditions, "basis": "Target or derived requirement"}

    plan = {
        "decision_question": "Are scoped requirements compatible?", "sample": "detector",
        "steps": [{"operation": "constraint comparison", "purpose": "Check intervals"}],
        "outcomes": [target("user_min", "MIN", 15, wavelength=2.7), target("derived_min", "MIN", 22.96, wavelength=2.7),
                     target("different_scope", "MAX", 20, wavelength=4.3)],
    }
    tool = RegisterProcessPlanTool(ScientificState())
    assert (await tool.execute(plan)).data["outcome_conflicts"] == []
    plan["outcomes"].append(target("same_scope_max", "MAX", 20, wavelength=2.7))
    result = await tool.execute(plan)
    assert result.data["outcome_conflicts"] == [["derived_min", "same_scope_max"]]
    assert result.data["execution_readiness"] == "BLOCKED"


async def test_measured_claim_preserves_scope_and_lower_bound_identity():
    state = ScientificState()
    authority = object()
    state._bind_runtime_authority(authority)
    evidence = ScientificEvidence(subject="coupon", property="thickness", value=4, unit="um", source_type="experiment",
                                  conditions={"sample_label": "coupon", "area_cm2": 2}, method="profilometer")
    state.add_evidence(evidence)
    state._attest_evidence(EvidenceAttestation(
        evidence_id=evidence.id, authority="observation", origin="trusted_builtin",
        tool_name="fixture", tool_call_id="test",
    ), capability=authority)
    outcome_data = {"result_id": "thickness", "metric": "thickness", "kind": "MEASURED", "relation": "EXACT",
                    "value": 4, "unit": "um", "sample_id": "coupon", "evidence_id": evidence.id,
                    "conditions": {"area_cm2": 2}, "basis": "Single measured point"}
    process = {
        "decision_question": "How thick was the tested coupon?", "sample": "coupon",
        "samples": [{"sample_id": "coupon", "label": "coupon", "role": "input"}],
        "steps": [{"step_id": "measure", "operation": "profilometry", "purpose": "Record thickness"}],
        "outcomes": [outcome_data],
    }
    tool = RegisterProcessPlanTool(state)
    good = await tool.execute(process)
    assert not good.is_error and good.data["scientific_validation"] == "UNKNOWN"
    for changes in ({"conditions": {"area_cm2": 3}}, {"relation": "MIN"}, {"value": 5}):
        bad = await tool.execute(process | {"outcomes": [outcome_data | changes]})
        assert bad.output == "PROCESS_OUTCOME_EVIDENCE_NOT_BOUND"


async def test_simple_consultation_needs_no_process_record(tmp_path):
    workspace = Workspace(tmp_path)
    state = ScientificState()
    provider = FakeModelProvider([FakeResponse(text="Spin coats small coupons quickly. Blade coating favors larger areas.")])
    runtime = AgentRuntime(model=provider, tools=create_default_registry(state, workspace),
                           workspace=workspace, scientific_state=state)
    events = [event async for event in runtime.run("Compare two coating routes briefly.")]
    assert events[-2].reason == "final_response"
    assert not state.process_plans and not state.research_deliveries
