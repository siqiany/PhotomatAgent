from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from photomatagent.models.fake import FakeModelProvider, FakeResponse, scripted_tool_call
from photomatagent.models.types import ToolCall
from photomatagent.runtime.budget import BudgetState
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import AllowAllPolicy
from photomatagent.scientific.capabilities.contracts import (
    ScientificEvidence,
    ScientificToolResult,
)
from photomatagent.scientific.loop.controller import ScientificLoopController
from photomatagent.scientific.loop.target import (
    ConstraintSpec,
    TargetSpec,
    canonical_lwir_detector_target,
)
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.base import Tool
from photomatagent.tools.exposure import ToolExposure
from photomatagent.tools.factory import create_default_registry
from photomatagent.workspace import Workspace


class PropertyReportTool(Tool):
    """TEST-ONLY tool: emits structured ScientificEvidence like a real capability.

    Lets deterministic controller tests inject DF-quality evidence without any
    external solver.
    """

    name = "test.report_property"
    namespace = "test"
    description = "TEST-ONLY: report a scientific property value with fidelity."
    exposure = ToolExposure.DIRECT
    input_schema = {
        "type": "object",
        "properties": {
            "subject": {"type": "string"},
            "property": {"type": "string"},
            "value": {},
            "unit": {"type": "string"},
            "fidelity": {"type": "string"},
        },
        "required": ["subject", "property", "value"],
    }

    async def execute(self, arguments: dict) -> ScientificToolResult:
        evidence = ScientificEvidence(
            subject=str(arguments["subject"]),
            property=str(arguments["property"]),
            value=arguments["value"],
            unit=str(arguments.get("unit", "")),
            source="test-only provider",
            source_type="dft_calculation",
            fidelity=str(arguments.get("fidelity", "dft")),
            summary=f"{arguments['property']}={arguments['value']}",
        )
        return ScientificToolResult(
            output=f"reported {arguments['property']}={arguments['value']}",
            data={"property": evidence.property, "value": evidence.value},
            evidence=[evidence],
        )


def _target() -> TargetSpec:
    return canonical_lwir_detector_target()


def propose_with_property(
    subject: str, property: str, value, *, unit: str = "", fidelity: str = "dft"
) -> FakeResponse:
    """One maker turn that both names the candidate and reports a property."""
    return FakeResponse(
        tool_calls=[
            ToolCall(
                name="test.report_property",
                arguments={
                    "subject": subject,
                    "property": "candidate_formula",
                    "value": subject,
                },
            ),
            ToolCall(
                name="test.report_property",
                arguments={
                    "subject": subject,
                    "property": property,
                    "value": value,
                    "unit": unit,
                    "fidelity": fidelity,
                },
            ),
        ]
    )


def build_controller(
    responses_per_round: list[list[FakeResponse]],
    *,
    max_rounds: int = 6,
    target: TargetSpec | None = None,
    events: list | None = None,
    tmp_path=None,
) -> tuple[ScientificLoopController, FakeModelProvider]:
    script: list[FakeResponse] = []
    for round_responses in responses_per_round:
        script.extend(round_responses)
        # one final-response per round closes the maker turn
        script.append(FakeResponse(text="proposal complete"))
    model = FakeModelProvider(script)

    workspace = Workspace(tmp_path or ".")
    scientific = ScientificState()
    registry = create_default_registry(scientific, workspace)
    registry.register(PropertyReportTool())
    runtime = AgentRuntime(
        model=model,
        tools=registry,
        workspace=workspace,
        scientific_state=scientific,
        permission_policy=AllowAllPolicy(),
        budget=BudgetState(max_iterations=20),
    )
    controller = ScientificLoopController(
        target=target or _target(),
        runtime=runtime,
        config=__import__(
            "photomatagent.scientific.loop.controller", fromlist=["ScientificLoopConfig"]
        ).ScientificLoopConfig(max_rounds=max_rounds),
        event_sinks=[],
    )
    if events is not None:
        events.clear()
        controller.event_sinks.append(lambda e: events.append(e))
    return controller, model


async def collect(controller: ScientificLoopController) -> list:
    events = []
    async for event in controller.run():
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_controller_success_trajectory(tmp_path):
    """Two rounds: band gap evidence, then responsivity evidence -> SUCCESS."""
    controller, _ = build_controller(
        [
            [
                propose_with_property("HgTe", "band_gap", 0.10, unit="eV", fidelity="dft")
            ],
            [
                propose_with_property("HgTe", "responsivity", 2.0, unit="A/W", fidelity="experimental")
            ],
        ],
        tmp_path=tmp_path,
    )
    events = await collect(controller)
    kinds = [e.kind for e in events]
    assert "candidate_proposed" in kinds
    assert "candidate_evaluated" in kinds
    assert "scientific_feedback_generated" in kinds
    assert "scientific_loop_decision_made" in kinds
    assert "scientific_loop_completed" in kinds
    assert controller.summary is not None
    assert controller.summary.status == "SUCCESS"
    assert controller.summary.candidate_count == 2
    assert controller.summary.best_candidate_id is not None
    final_eval = controller.summary.final_evaluation
    assert final_eval is not None and final_eval.verdict == "PASS"
    assert controller.summary.unresolved_violations == []
    completed = next(e for e in events if e.kind == "scientific_loop_completed")
    assert completed.status == "SUCCESS"


@pytest.mark.asyncio
async def test_controller_missing_responsivity_is_inconclusive_at_budget(tmp_path):
    """A passing band gap alone must never produce scientific success."""
    controller, _ = build_controller(
        [
            [
                propose_with_property("HgTe", "band_gap", 0.10, unit="eV", fidelity="dft")
            ]
        ],
        max_rounds=3,
        tmp_path=tmp_path,
    )
    events = await collect(controller)
    assert controller.summary is not None
    assert controller.summary.status in {"BUDGET_EXHAUSTED", "STALLED"}
    assert "responsivity" in controller.summary.unresolved_evidence_gaps
    assert controller.summary.final_evaluation.verdict != "PASS"


@pytest.mark.asyncio
async def test_controller_stalls_on_repeated_candidate(tmp_path):
    """Three rounds of the identical proposal must terminate as STALLED."""
    controller, _ = build_controller(
        [
            [
                propose_with_property("HgTe", "band_gap", 0.10, unit="eV", fidelity="dft")
            ],
            [FakeResponse(text="keep working")],
            [FakeResponse(text="keep working")],
        ],
        max_rounds=6,
        tmp_path=tmp_path,
    )
    events = await collect(controller)
    # Round 1 reports band gap; rounds 2-3 add nothing -> repeated fingerprint.
    assert controller.summary is not None
    assert controller.summary.status == "STALLED"
    assert "scientific_loop_stalled" in [e.kind for e in events]
    assert controller.summary.rounds == 4  # 1 improvement + 3 no-progress


@pytest.mark.asyncio
async def test_controller_feedback_changes_next_round_instruction(tmp_path):
    """The second-round user message must contain the round-1 feedback."""
    controller, model = build_controller(
        [
            [
                propose_with_property("HgTe", "band_gap", 0.10, unit="eV", fidelity="empirical")
            ],
            [
                propose_with_property("HgTe", "responsivity", 2.0, unit="A/W", fidelity="experimental")
            ],
        ],
        tmp_path=tmp_path,
    )
    await collect(controller)
    # Run 1 consumed requests[0] (tool calls) and requests[1] (final answer).
    # Run 2's first request must embed the round-1 feedback instruction.
    assert len(model.requests) >= 3
    joined = " ".join(
        m.content
        for m in model.requests[2].messages
        if getattr(m, "content", "")
    )
    assert "Scientific feedback from round 1" in joined
    assert "Do not claim completion until" in joined


@pytest.mark.asyncio
async def test_controller_events_are_jsonl_parseable(tmp_path, tmp_path2=None):
    """Every emitted event must round-trip through parse_event (schema v1)."""
    from photomatagent.runtime.events import parse_event

    collected: list = []
    controller, _ = build_controller(
        [
            [
                propose_with_property("HgTe", "band_gap", 0.10, unit="eV", fidelity="dft")
            ],
            [
                propose_with_property("HgTe", "responsivity", 2.0, unit="A/W", fidelity="experimental")
            ],
        ],
        events=collected,
        tmp_path=tmp_path,
    )
    await collect(controller)
    loop_events = [e for e in collected if e.kind.startswith("scientific_") or e.kind in {"candidate_proposed", "candidate_evaluated"}]
    assert loop_events
    for event in loop_events:
        parsed = parse_event(event.model_dump(mode="json"))
        assert parsed.kind == event.kind


@pytest.mark.asyncio
async def test_controller_default_runtime_pathway_uses_mock_calculation(tmp_path):
    """The maker can run the existing mock tool; a job-free trajectory still terminates."""
    script = [
        scripted_tool_call(
            "tool_call",
            {
                "name": "mock.run_calculation",
                "arguments": {
                    "material": "InAs",
                    "calculation_type": "band_structure",
                },
            },
        ),
        FakeResponse(text="mock result obtained"),
    ]
    model = FakeModelProvider(script)
    workspace = Workspace(tmp_path)
    scientific = ScientificState()
    registry = create_default_registry(scientific, workspace)
    runtime = AgentRuntime(
        model=model,
        tools=registry,
        workspace=workspace,
        scientific_state=scientific,
        permission_policy=AllowAllPolicy(),
        budget=BudgetState(max_iterations=20),
    )
    controller = ScientificLoopController(
        target=_target(),
        runtime=runtime,
        config=__import__(
            "photomatagent.scientific.loop.controller", fromlist=["ScientificLoopConfig"]
        ).ScientificLoopConfig(max_rounds=2),
        event_sinks=[],
    )
    await collect(controller)
    assert controller.summary is not None
    # Mock evidence is 0.31 eV empirical -> hard band-gap violation, no success.
    assert controller.summary.status in {"BUDGET_EXHAUSTED", "INCONCLUSIVE"}
    assert "band_gap" in [v.property for v in controller.summary.unresolved_violations]