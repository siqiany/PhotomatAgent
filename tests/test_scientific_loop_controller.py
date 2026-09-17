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
from photomatagent.scientific.discovery.models import HypothesisOrigin, HypothesisProposal
from photomatagent.scientific.discovery.registration import build_hypothesis
from photomatagent.scientific.loop.candidate import candidate_from_formula
from photomatagent.scientific.loop.controller import ScientificLoopController
from photomatagent.scientific.loop.evaluation import (
    EvidenceEvaluationPolicy,
    ScientificEvaluator,
)
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
    candidate_extractor=None,
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
    effective_target = target or _target()
    controller = ScientificLoopController(
        target=effective_target,
        runtime=runtime,
        evaluator=ScientificEvaluator(
            effective_target,
            policy=EvidenceEvaluationPolicy(allow_synthetic_evidence=True),
        ),
        config=__import__(
            "photomatagent.scientific.loop.controller", fromlist=["ScientificLoopConfig"]
        ).ScientificLoopConfig(max_rounds=max_rounds),
        candidate_extractor=candidate_extractor,
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


def discovery_hypothesis(request_id: str, formula: str):
    return build_hypothesis(
        HypothesisProposal(
            request_id=request_id,
            formula=formula,
            statement=f"investigate {formula}",
            design_operation="other",
            validation_questions=["Does it satisfy the target?"],
        ),
        HypothesisOrigin(
            tool_name="generation.register_hypothesis",
            tool_call_id=f"call-{request_id}",
            session_id="session",
            run_id="run",
            provider="fake",
            model="fake",
        ),
    )


def registration_call(request_id: str, formula: str, statement: str) -> FakeResponse:
    return scripted_tool_call(
        "tool_call",
        {
            "name": "generation.register_hypothesis",
            "arguments": {
                "request_id": request_id,
                "formula": formula,
                "statement": statement,
                "design_operation": "other",
                "validation_questions": ["Does it satisfy the target?"],
            },
        },
        tool_call_id=f"call-{request_id}",
    )


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
    # Re-evaluating the same stable composition updates its projection rather
    # than consuming another candidate-budget slot.
    assert controller.summary.candidate_count == 1
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


@pytest.mark.asyncio
async def test_controller_queues_two_registered_candidates_across_rounds(tmp_path):
    controller, _ = build_controller([[], []], max_rounds=2, tmp_path=tmp_path)
    first = discovery_hypothesis("first", "NaBiS2")
    second = discovery_hypothesis("second", "HgTe")
    controller.runtime.scientific_state.material_hypotheses.extend([first, second])

    await collect(controller)

    assert [report.candidate_id for report in controller.state.evaluations] == [
        first.candidate_id,
        second.candidate_id,
    ]
    assert {item.candidate_id for item in controller.state.candidates} == {
        first.candidate_id,
        second.candidate_id,
    }
    assert controller.state.pending_candidate_ids == []
    assert controller.state.active_candidate_id == second.candidate_id


@pytest.mark.asyncio
async def test_controller_keeps_active_candidate_when_new_applicable_evidence_arrives(
    tmp_path,
):
    controller, _ = build_controller(
        [
            [],
            [propose_with_property("NaBiS2", "band_gap", 0.10, unit="eV")],
            [],
        ],
        max_rounds=3,
        tmp_path=tmp_path,
    )
    first = discovery_hypothesis("first", "NaBiS2")
    second = discovery_hypothesis("second", "HgTe")
    controller.runtime.scientific_state.material_hypotheses.extend([first, second])

    await collect(controller)

    assert [report.candidate_id for report in controller.state.evaluations] == [
        first.candidate_id,
        first.candidate_id,
        second.candidate_id,
    ]


@pytest.mark.asyncio
async def test_controller_preserves_pending_candidates_when_budget_is_reached(tmp_path):
    controller, _ = build_controller([[]], max_rounds=3, tmp_path=tmp_path)
    controller.config.max_candidates = 1
    first = discovery_hypothesis("first", "NaBiS2")
    second = discovery_hypothesis("second", "HgTe")
    controller.runtime.scientific_state.material_hypotheses.extend([first, second])

    await collect(controller)

    assert controller.summary is not None
    assert controller.summary.status == "BUDGET_EXHAUSTED"
    assert controller.state.pending_candidate_ids == [second.candidate_id]


@pytest.mark.asyncio
async def test_controller_custom_candidate_extractor_remains_supported(tmp_path):
    custom = candidate_from_formula("PbTe")
    controller, _ = build_controller(
        [[]],
        max_rounds=1,
        tmp_path=tmp_path,
        candidate_extractor=lambda scientific, iteration: custom,
    )

    await collect(controller)

    assert controller.state.evaluations[0].candidate_id == custom.candidate_id


@pytest.mark.asyncio
async def test_controller_registers_two_candidates_in_first_maker_round(tmp_path):
    controller, _ = build_controller(
        [
            [
                registration_call("first", "NaBiS2", "first mechanism"),
                registration_call("second", "HgTe", "second mechanism"),
            ],
            [],
        ],
        max_rounds=2,
        tmp_path=tmp_path,
    )
    round_one: dict[str, object] = {}

    def capture_round_one(event) -> None:
        if event.kind == "candidate_evaluated" and event.round == 1:
            round_one.update(
                candidates=[item.candidate_id for item in controller.state.candidates],
                active=controller.state.active_candidate_id,
                pending=list(controller.state.pending_candidate_ids),
            )

    controller.event_sinks.append(capture_round_one)

    await collect(controller)

    registered = controller.runtime.scientific_state.material_hypotheses
    expected = [record.candidate_id for record in registered]
    assert len(registered) == 2
    assert round_one == {
        "candidates": expected,
        "active": expected[0],
        "pending": [expected[1]],
    }
    assert [report.candidate_id for report in controller.state.evaluations] == expected
    assert len(controller.state.evaluations) == controller.state.round == 2


@pytest.mark.asyncio
async def test_controller_merges_two_registered_reasons_into_one_budget_slot(tmp_path):
    controller, _ = build_controller(
        [
            [
                registration_call("first", "Na0.75Ag0.25BiS2", "alloy sites"),
                registration_call("second", "Na3AgBi4S8", "order sites"),
            ]
        ],
        max_rounds=1,
        tmp_path=tmp_path,
    )
    controller.config.max_candidates = 1

    await collect(controller)

    registered = controller.runtime.scientific_state.material_hypotheses
    assert len(registered) == 2
    assert len(controller.state.candidates) == 1
    assert controller.state.candidates[0].representation["hypothesis_ids"] == [
        record.id for record in registered
    ]
    assert controller.state.pending_candidate_ids == []
    assert len(controller.state.evaluations) == 1


@pytest.mark.asyncio
async def test_projection_diagnostic_is_visible_in_summary_and_event(tmp_path):
    controller, _ = build_controller([[]], max_rounds=1, tmp_path=tmp_path)
    valid = discovery_hypothesis("valid", "NaBiS2")
    invalid = ScientificEvidence(
        subject="bad formula",
        property="candidate_formula",
        value="bad formula",
        source="legacy import",
        source_type="model",
    )
    controller.runtime.scientific_state.material_hypotheses.append(valid)
    controller.runtime.scientific_state.evidence.append(invalid)

    events = await collect(controller)

    assert controller.summary is not None
    assert len(controller.state.candidates) == 1
    expected = f"INVALID_LEGACY_CANDIDATE:{invalid.id}"
    assert any(item.startswith(expected) for item in controller.summary.projection_diagnostics)
    completed = next(event for event in events if event.kind == "scientific_loop_completed")
    assert completed.projection_diagnostics == controller.summary.projection_diagnostics


@pytest.mark.asyncio
async def test_custom_extractor_programming_error_propagates(tmp_path):
    def broken_extractor(scientific, iteration):
        raise RuntimeError("programming defect")

    controller, _ = build_controller(
        [[]],
        max_rounds=1,
        tmp_path=tmp_path,
        candidate_extractor=broken_extractor,
    )

    with pytest.raises(RuntimeError, match="programming defect"):
        await collect(controller)
