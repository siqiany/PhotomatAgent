"""Runtime acceptance for process proposals, search bounds and recovery."""
from __future__ import annotations

import asyncio

import pytest

from photomatagent.errors import ProviderError
from photomatagent.models.fake import FakeModelProvider, FakeResponse, scripted_tool_call
from photomatagent.models.types import ModelTextDelta
from photomatagent.runtime.budget import BudgetState
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import AutoApproveHandler, default_permission_policy
from photomatagent.scientific.capabilities.literature.federated import FederatedLiterature
from photomatagent.scientific.capabilities.literature.source_fetch import LiteratureFetchSourceTool
from photomatagent.scientific.capabilities.optics import meep_thinfilm
from photomatagent.scientific.evidence import Evidence
from photomatagent.scientific.loop.candidate import CandidateState
from photomatagent.scientific.loop.controller import ScientificLoopController
from photomatagent.scientific.loop.target import ConstraintSpec, TargetSpec
from photomatagent.scientific.process import ProcessPlan
from photomatagent.scientific.state import ScientificState
from photomatagent.sessions.store import load_session_snapshot, save_session_snapshot
from photomatagent.tools.base import ToolResult
from photomatagent.tools.factory import create_default_registry
from photomatagent.tools.process_plan import RegisterProcessPlanTool
from photomatagent.tools.scientific_state_inspect import ScientificStateInspectTool
from photomatagent.workspace import Workspace


GOAL = "设计一个新薄膜处理路线"
PROPOSAL = {
    "decision_question": "Can intermittent dry annealing protect the film?",
    "sample": "PbS film",
    "steps": [{"operation": "anneal", "purpose": "Test interrupted heating", "parameters": [
        {"name": "temperature", "unit": "K", "bounds": [300, 320], "basis": "PROPOSED"},
    ]}],
    "proposal": {
        "kind": "CONDITIONAL", "baseline": "Untreated film from the same batch",
        "change": "Pulse the dry anneal with a full cool-down between pulses",
        "rationale": "Hypothesis: intermittent heating limits ligand loss",
        "variables": ["Anneal temperature range is a design variable", "Pulse duration"],
        "unknowns": [{"cause": "Furnace and thermal budget unknown", "impact": "Pulse timing may drift",
                      "resolution": "Measure fixture response before coating", "blocks_execution": True}],
        "branches": [{"condition": "If the furnace can repeat short pulses", "approach": "Compare pulsed and continuous anneal",
                      "information_action": "Log fixture temperature on an uncoated substrate"},
                     {"condition": "If it cannot repeat short pulses", "approach": "Use a long-pulse control first",
                      "information_action": "Measure the minimum repeatable pulse interval"}],
        "validation": {"control": "Untreated and equal-dose continuous controls",
                       "procedure": "Use matched films with independently logged thermal traces",
                       "metric": "Crack fraction and sheet resistance",
                       "criterion": "Support only if crack fraction falls without higher sheet resistance",
                       "next_action": "Measure three matched specimens before comparing routes"},
    },
}
PLAN_ID = ProcessPlan.model_validate(PROPOSAL).id
DECISION = {
    "question": "Which film route merits testing?",
    "process_plan_ids": [PLAN_ID],
    "research_stop": {"reason": "VALIDATION_REQUIRED", "rationale": "A same-batch control can discriminate the routes."},
}


def call(name, args):
    return scripted_tool_call("tool_call", {"name": name, "arguments": args})


def runtime(tmp_path, responses, *, state=None, budget=None):
    state = state or ScientificState()
    workspace = Workspace(tmp_path)
    return AgentRuntime(
        model=FakeModelProvider(responses), tools=create_default_registry(state, workspace),
        workspace=workspace, scientific_state=state, budget=budget,
        permission_policy=default_permission_policy(), approval_handler=AutoApproveHandler(),
    )


def kind(events, name):
    return [event for event in events if event.kind == name]


async def test_thin_film_candidate_without_direct_recipe_delivers_and_restores(tmp_path):
    runner = runtime(tmp_path, [
        scripted_tool_call("tool_describe", {"name": "process.register_plan"}),
        call("process.register_plan", PROPOSAL),
        scripted_tool_call("tool_describe", {"name": "decision.record"}),
        call("decision.record", DECISION),
    ])
    events = [event async for event in runner.run(GOAL)]
    assert kind(events, "loop_completed")[-1].reason == "research_delivery"
    answer = "".join(item.text for item in kind(events, "text_delta"))
    for fragment in ("Pulse the dry anneal", "Furnace and thermal budget unknown", "Crack fraction", "UNKNOWN",
                     "BLOCKED", "VALIDATION_REQUIRED", "Use a long-pulse control first"):
        assert fragment in answer
    state = runner.scientific_state
    assert not state.evidence and state.research_delivery.status == "READY"
    assert state.process_plans[0].steps[0].parameters[0].basis == "PROPOSED"
    save_session_snapshot(tmp_path / "session", conversation=runner.conversation_state, scientific=state)
    restored = load_session_snapshot(tmp_path / "session").scientific
    assert restored.research_delivery.stop_reason == "VALIDATION_REQUIRED"
    assert restored.process_plans[0].proposal.unknowns[0].blocks_execution

    again = runtime(tmp_path, [scripted_tool_call("method_search", {"query": "thin film"})], state=restored)
    resumed = [event async for event in again.run(GOAL)]
    assert any("RESEARCH_SEARCH_CLOSED" in str(event.model_dump()) for event in resumed)
    assert not any(event.tool_name == "method_search" for event in kind(resumed, "tool_started"))

    restored.add_evidence(Evidence(type="observation", confidence=0.8, source="fixture",
                                   content="The furnace cannot repeat the pulses."))
    reopened = runtime(tmp_path, [scripted_tool_call("method_search", {"query": "thin film"}), FakeResponse(text="Reopened.")],
                       state=restored)
    resumed = [event async for event in reopened.run(GOAL)]
    assert any(event.tool_name == "method_search" for event in kind(resumed, "tool_started"))
    assert restored.research_deliveries[0].superseded and restored.research_deliveries[0].decision_id
    assert restored.process_plans[0].id == PLAN_ID
    plans = await ScientificStateInspectTool(restored).execute({"section": "process_plans"})
    assert plans.data["items"][0]["delivery_superseded"] is True


async def test_repeated_queries_close_on_source_identity_and_render_partial(tmp_path, monkeypatch):
    count = 0

    async def search(self, *args):
        nonlocal count
        count += 1
        return {"hits": [{"doi": "10.1234/film", "title": f"same DOI with display change {count}"}],
                "source_status": {"crossref": "OK"}}

    monkeypatch.setattr(FederatedLiterature, "search", search)
    runner = runtime(tmp_path, [
        scripted_tool_call("tool_describe", {"name": "literature.search"}),
        *[call("literature.search", {"query": f"film query {index}"}) for index in range(5)],
        FakeResponse(text="Compare both treatments on same-batch specimens and measure dry thickness."),
    ])
    events = [event async for event in runner.run(GOAL)]
    assert count == 3
    assert kind(events, "loop_completed")[-1].reason == "research_partial"
    delivery = runner.scientific_state.research_delivery
    assert delivery.stop_reason == "NO_NEW_SOURCES" and len(delivery.source_keys) == 1
    handoff = next(event for event in kind(events, "text_delta") if event.research_delivery == "PARTIAL")
    assert "10.1234/film" in handoff.text
    assert any("same-batch specimens" in event.text for event in kind(events, "text_delta"))


async def test_source_limits_allow_alternatives_then_deliver_partial(tmp_path, monkeypatch):
    async def search(self, query, sources, *args):
        return {"hits": [], "source_status": {sources[0]: "NOT_CONFIGURED" if sources[0] == "crossref" else "HTTP_429"}}

    async def fetch(self, arguments):
        return ToolResult(output="SUPPLEMENT_FORMAT_UNSUPPORTED", is_error=True)

    monkeypatch.setattr(FederatedLiterature, "search", search)
    monkeypatch.setattr(LiteratureFetchSourceTool, "execute", fetch)
    runner = runtime(tmp_path, [
        scripted_tool_call("tool_describe", {"name": "literature.search"}),
        call("literature.search", {"query": "film recipe", "sources": ["crossref"]}),
        call("literature.search", {"query": "film recipe", "sources": ["semantic_scholar"]}),
        scripted_tool_call("tool_describe", {"name": "literature.fetch_source"}),
        call("literature.fetch_source", {"url": "https://arxiv.org/pdf/2401.01234v1"}),
        FakeResponse(text="A same-batch validation is still needed."),
    ])
    events = [event async for event in runner.run(GOAL)]
    delivery = runner.scientific_state.research_delivery
    assert delivery.stop_reason == "SOURCE_LIMITATIONS" and delivery.attempts == 3
    assert set(delivery.source_status.values()) == {"NOT_CONFIGURED", "HTTP_429", "SUPPLEMENT_FORMAT_UNSUPPORTED"}
    assert kind(events, "loop_completed")[-1].reason == "research_partial"


async def test_distinct_sources_still_respect_finite_search_limit(tmp_path, monkeypatch):
    async def search(self, query, *args):
        return {"hits": [{"doi": f"10.1234/{query}"}], "source_status": {"crossref": "OK"}}

    monkeypatch.setattr(FederatedLiterature, "search", search)
    runner = runtime(tmp_path, [
        scripted_tool_call("tool_describe", {"name": "literature.search"}),
        *[call("literature.search", {"query": f"film-{index}"}) for index in range(10)],
    ])
    events = [event async for event in runner.run(GOAL)]
    delivery = runner.scientific_state.research_delivery
    assert delivery.attempts == len(delivery.source_keys) == 8
    assert delivery.stop_reason == "SEARCH_LIMIT"
    assert kind(events, "loop_completed")[-1].reason == "research_partial"


@pytest.mark.parametrize("with_proposal", [True, False])
async def test_existing_iteration_budget_distinguishes_ready_from_partial(tmp_path, with_proposal):
    responses = [
        scripted_tool_call("tool_describe", {"name": "process.register_plan"}),
        call("process.register_plan", PROPOSAL),
    ] if with_proposal else [
        scripted_tool_call("method_search", {"query": "film"}),
        scripted_tool_call("method_search", {"query": "anneal"}),
    ]
    answer = "用同批未处理试片验证退火后裂纹比例。"
    runner = runtime(tmp_path, [*responses, FakeResponse(text=answer)],
                     budget=BudgetState(max_iterations=3))
    events = [event async for event in runner.run(GOAL)]
    delivery = runner.scientific_state.research_delivery
    assert delivery.stop_reason == "RUNTIME_BUDGET"
    assert delivery.status == ("READY" if with_proposal else "PARTIAL")
    assert kind(events, "loop_completed")[-1].reason == ("research_delivery" if with_proposal else "research_partial")
    handoff = next(event for event in kind(events, "text_delta") if event.research_delivery)
    assert answer in handoff.text and not runner.model_provider.requests[-1].tools


async def test_last_research_call_keeps_candidate_and_checker_in_partial_delivery(tmp_path):
    values = {"spectral_min_um": 1, "spectral_max_um": 5, "target_responsivity_a_w": 0.3,
              "mode": "at_wavelength", "wavelength_um": 2, "eqe_percent": 12, "eqe_relation": "minimum"}
    answer = "候选方案：用同批对照片比较两种处理，测干膜厚度和均匀性。"
    runner = runtime(tmp_path, [
        scripted_tool_call("tool_describe", {"name": "photodetector.check_targets"}),
        call("photodetector.check_targets", values),
        scripted_tool_call("method_search", {"query": "thin film"}),
        FakeResponse(text=answer),
    ], budget=BudgetState(max_iterations=4))
    events = [event async for event in runner.run(GOAL)]
    handoff = next(event for event in kind(events, "text_delta") if event.research_delivery == "PARTIAL")
    assert answer in handoff.text and "光电 Checker 回执" in handoff.text
    assert "RUNTIME_BUDGET" in handoff.text
    assert not runner.model_provider.requests[-1].tools
    assert runner.budget.model_calls == 4


async def test_research_provider_timeout_leaves_bounded_synthesis(tmp_path):
    answer = "候选方案：同批试片采用两种处理，测量干膜厚度并保留对照。"
    runner = runtime(tmp_path, [
        scripted_tool_call("method_search", {"query": "thin film"}),
        FakeResponse(text=answer),
    ], budget=BudgetState(max_iterations=3))
    model = runner.model_provider
    original_stream = model.stream
    calls = 0

    async def stream(request):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ProviderError("fake", "timeout", code="NETWORK_TIMEOUT")
        async for event in original_stream(request):
            yield event

    model.stream = stream
    events = [event async for event in runner.run(GOAL)]
    handoff = next(event for event in kind(events, "text_delta") if event.research_delivery == "PARTIAL")
    assert answer in handoff.text and "PROVIDER_TIMEOUT" in handoff.text
    assert not model.requests[-1].tools
    assert runner.budget.model_calls == 3


async def test_truncated_registration_path_recovers_as_prose(tmp_path):
    class Truncated(FakeResponse):
        def to_response(self):
            return super().to_response().model_copy(update={"finish_reason": "max_tokens"})

    answer = "候选方案：记录同批空白与处理样的质量差，验证过程效率。"
    runner = runtime(tmp_path, [
        scripted_tool_call("tool_describe", {"name": "process.register_plan"}),
        call("process.register_plan", {}),
        Truncated(),
        FakeResponse(text=answer),
    ], budget=BudgetState(max_iterations=5))
    events = [event async for event in runner.run(GOAL)]
    handoff = next(event for event in kind(events, "text_delta") if event.research_delivery == "PARTIAL")
    assert answer in handoff.text and "OUTPUT_TRUNCATED" in handoff.text
    assert not runner.model_provider.requests[-1].tools


async def test_wall_bounded_calculation_keeps_last_call_for_answer(tmp_path):
    answer = "同批对照验证预计质量变化，实际结果待测。"
    runner = runtime(tmp_path, [
        scripted_tool_call("tool_describe", {"name": "calculator"}),
        call("calculator", {"expression": "2 + 2"}),
        FakeResponse(text=answer),
    ], budget=BudgetState(max_iterations=3))
    events = [event async for event in runner.run(GOAL, wall_seconds=10)]
    handoff = next(event for event in kind(events, "text_delta") if event.research_delivery == "PARTIAL")
    assert answer in handoff.text and "RUNTIME_BUDGET" in handoff.text
    assert not runner.model_provider.requests[-1].tools


@pytest.mark.parametrize("synthesis_stalls", [False, True])
async def test_turn_deadline_reserves_synthesis_and_retains_completed_text(tmp_path, synthesis_stalls):
    answer = "候选方案：比较同批对照和处理组，测量膜厚分布。"
    runner = runtime(tmp_path, [
        scripted_tool_call("method_search", {"query": "thin film"}),
        FakeResponse(text=answer),
    ], budget=BudgetState(max_iterations=8))
    model = runner.model_provider
    original_stream = model.stream
    calls = 0

    async def stream(request):
        nonlocal calls
        calls += 1
        if calls == 2:
            await asyncio.sleep(10)
        if calls == 3:
            assert not request.tools
            if synthesis_stalls:
                yield ModelTextDelta(text=answer + "\n")
                await asyncio.sleep(10)
        async for event in original_stream(request):
            yield event

    model.stream = stream
    clock = asyncio.get_running_loop().time
    started = clock()
    events = [event async for event in runner.run(GOAL, wall_seconds=1)]
    assert clock() - started < 1
    handoff = next(event for event in kind(events, "text_delta") if event.research_delivery == "PARTIAL")
    assert answer in handoff.text and "WALL_BUDGET" in handoff.text
    assert calls == runner.budget.model_calls == 3


async def test_hard_conflict_delivers_relaxation_without_claiming_feasibility(tmp_path):
    conflict = {"question": "Can annealing be used?",
                "research_stop": {"reason": "HARD_CONFLICT", "rationale": "Requested heat exceeds fixed substrate limit",
                                  "constraint_change": "Allow another substrate or a lower target temperature"}}
    runner = runtime(tmp_path, [
        scripted_tool_call("tool_describe", {"name": "decision.record"}),
        call("decision.record", conflict),
    ])
    events = [event async for event in runner.run(GOAL)]
    answer = "".join(item.text for item in kind(events, "text_delta"))
    assert "CONFLICT" in answer and "Allow another substrate" in answer and "UNKNOWN" in answer
    assert not runner.scientific_state.process_plans


async def test_outer_loop_keeps_scientific_verdict_independent_of_delivery(tmp_path):
    runner = runtime(tmp_path, [
        scripted_tool_call("tool_describe", {"name": "process.register_plan"}),
        call("process.register_plan", PROPOSAL),
        scripted_tool_call("tool_describe", {"name": "decision.record"}),
        call("decision.record", DECISION),
    ])
    target = TargetSpec(goal=GOAL, constraints=[ConstraintSpec(property="sheet_resistance", operator="le", value=1)])
    controller = ScientificLoopController(runtime=runner, target=target,
                                          candidate_extractor=lambda *_: CandidateState(label="Pulsed anneal"))
    events = [event async for event in controller.run()]
    assert controller.summary.rounds == 1 and controller.summary.status == "INCONCLUSIVE"
    assert controller.summary.delivery_status == "READY"
    assert controller.summary.final_evaluation.verdict == "INCONCLUSIVE"
    assert controller.summary.final_evaluation.critical_evidence_gaps
    assert kind(events, "scientific_loop_completed")[-1].research_stop_reason == "VALIDATION_REQUIRED"


async def test_proposed_parameters_cannot_claim_evidence(tmp_path):
    altered = PROPOSAL | {"steps": [PROPOSAL["steps"][0] | {"parameters": [
        PROPOSAL["steps"][0]["parameters"][0] | {"evidence_id": "forged"},
    ]}]}
    result = await RegisterProcessPlanTool(ScientificState()).execute(altered)
    assert result.is_error and not result.state_updates


async def test_optics_runtime_still_requires_concrete_thickness(tmp_path, monkeypatch):
    calls = []

    def solve(**kwargs):
        calls.append(kwargs)
        return {"absorptance": 0.0, "reflectance": 0.1, "transmittance": 0.9,
                "energy_conservation_residual": 0.0}

    monkeypatch.setattr(meep_thinfilm, "run_meep_thinfilm", solve)
    runner = runtime(tmp_path, [
        scripted_tool_call("tool_describe", {"name": meep_thinfilm.MeepThinFilmTool.name}),
        call(meep_thinfilm.MeepThinFilmTool.name, {"wavelength_um": 1.5, "refractive_index": 2.0}),
        call(meep_thinfilm.MeepThinFilmTool.name, {"wavelength_um": 1.5, "refractive_index": 2.0, "thickness_um": 0.12}),
        FakeResponse(text="Completed the stated simulated case."),
    ])
    # The solver is optional. Register the real adapter when its backend is replaced.
    if not any(tool.name == meep_thinfilm.MeepThinFilmTool.name for tool in runner._tools.list_tools()):
        runner._tools.register(meep_thinfilm.MeepThinFilmTool())
    events = [event async for event in runner.run(GOAL)]
    assert len(calls) == 1 and calls[0]["thickness_um"] == 0.12
    assert any("thickness_um" in str(event.model_dump()) for event in kind(events, "tool_failed"))
