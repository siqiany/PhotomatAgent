from __future__ import annotations

import json

import pytest

from photomatagent.logging.event_logger import EventLogger
from photomatagent.models.fake import FakeModelProvider, FakeResponse, scripted_tool_call
from photomatagent.runtime.events import (
    HypothesisRegistered,
    HypothesisRegistrationRejected,
    parse_event,
)
from photomatagent.runtime.permissions import AskPolicy, DenyAllPolicy, DenyHandler
from photomatagent.scientific.discovery.models import HypothesisRegistration
from photomatagent.scientific.capabilities.generation.hypotheses import (
    RegisterHypothesisTool,
)
from photomatagent.scientific.evidence import Evidence
from photomatagent.tools.base import Tool, ToolResult
from photomatagent.tools.exposure import ToolExposure
from photomatagent.workspace import Workspace

from conftest import collect, make_runtime


def _arguments(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "request_id": "event-r1",
        "formula": "Na0.75Ag0.25BiS2",
        "statement": "Explore isovalent substitution",
        "design_operation": "isovalent_substitution",
        "validation_questions": ["Is phase separation preferred?"],
    }
    payload.update(changes)
    return payload


def _bridge(payload: dict[str, object], call_id: str) -> FakeResponse:
    return scripted_tool_call(
        "tool_call",
        {"name": "generation.register_hypothesis", "arguments": payload},
        tool_call_id=call_id,
    )


class _ExplodingRegistrationTool(Tool):
    name = "generation.register_hypothesis"
    exposure = ToolExposure.DEFERRED
    input_schema = RegisterHypothesisTool.input_schema

    async def execute(self, arguments: dict[str, object]) -> ToolResult:
        raise RuntimeError("PRIVATE EXECUTION BODY")


class _RejectedRegistrationTool(_ExplodingRegistrationTool):
    async def execute(self, arguments: dict[str, object]) -> ToolResult:
        return ToolResult(
            output="PRIVATE TOOL REJECTION BODY",
            is_error=True,
            data={"error_type": "PRIVATE ARBITRARY REASON"},
        )


class _InvalidUpdateRegistrationTool(_ExplodingRegistrationTool):
    async def execute(self, arguments: dict[str, object]) -> ToolResult:
        proposal = _arguments(parent_hypothesis_ids=["hyp_missing"])
        return ToolResult(
            output="accepted too early",
            state_updates=[HypothesisRegistration.model_validate({"proposal": proposal})],
        )


def _replace_registration_tool(runtime, tool: Tool) -> None:
    runtime._tools._tools["generation.register_hypothesis"] = tool


def test_hypothesis_events_round_trip_through_discriminated_union():
    registered = HypothesisRegistered(
        hypothesis_id="hyp_123",
        candidate_id="cand_123",
        request_id="request-1",
    )
    rejected = HypothesisRegistrationRejected(
        request_id="request-2",
        reason_code="TOOL_REJECTED",
    )

    assert parse_event(registered.model_dump(mode="json")) == registered
    assert parse_event(rejected.model_dump(mode="json")) == rejected


@pytest.mark.asyncio
async def test_runtime_emits_one_registration_event_for_idempotent_retry(tmp_path):
    model = FakeModelProvider(
        [
            _bridge(_arguments(), "call-1"),
            _bridge(_arguments(), "call-2"),
            FakeResponse(text="done"),
        ]
    )
    runtime = make_runtime(model, workspace=Workspace(tmp_path))

    events = await collect(runtime, "register twice")

    registered = [event for event in events if event.kind == "hypothesis_registered"]
    assert len(registered) == 1
    record = runtime.scientific_state.material_hypotheses[0]
    assert registered[0].hypothesis_id == record.id
    assert registered[0].candidate_id == record.candidate_id
    assert registered[0].request_id == "event-r1"
    assert not any(
        event.kind == "hypothesis_registration_rejected" for event in events
    )


@pytest.mark.asyncio
async def test_runtime_rejection_event_follows_failed_registration(tmp_path):
    invalid = _arguments(parent_hypothesis_ids=["hyp_missing"])
    model = FakeModelProvider(
        [_bridge(invalid, "call-invalid"), FakeResponse(text="done")]
    )
    runtime = make_runtime(model, workspace=Workspace(tmp_path))

    events = await collect(runtime, "reject invalid")

    rejected = [
        event for event in events if event.kind == "hypothesis_registration_rejected"
    ]
    assert len(rejected) == 1
    assert rejected[0].request_id == "event-r1"
    assert rejected[0].reason_code == "TOOL_REJECTED"
    assert runtime.scientific_state.material_hypotheses == []
    assert not any(event.kind == "hypothesis_registered" for event in events)


@pytest.mark.asyncio
async def test_runtime_schema_rejection_emits_typed_registration_event(tmp_path):
    invalid = {**_arguments(), "origin": {"provider": "forged"}}
    model = FakeModelProvider(
        [_bridge(invalid, "call-invalid-schema"), FakeResponse(text="done")]
    )
    runtime = make_runtime(model, workspace=Workspace(tmp_path))

    events = await collect(runtime, "reject forged provenance")

    rejected = [
        event for event in events if event.kind == "hypothesis_registration_rejected"
    ]
    assert len(rejected) == 1
    assert rejected[0].request_id == "event-r1"
    assert rejected[0].reason_code == "SCHEMA_VALIDATION_FAILED"
    assert runtime.scientific_state.material_hypotheses == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "reason_code"),
    [
        ("protocol_direct", "DEFERRED_TOOL_REQUIRES_BRIDGE"),
        ("permission_deny", "PERMISSION_DENIED"),
        ("approval_reject", "APPROVAL_REJECTED"),
        ("execution_exception", "TOOL_EXECUTION_FAILED"),
        ("tool_result_reject", "TOOL_REJECTED"),
        ("state_update_invalid", "STATE_UPDATE_VALIDATION_FAILED"),
    ],
)
async def test_every_registration_termination_emits_one_typed_rejection(
    tmp_path, case: str, reason_code: str
):
    payload = _arguments()
    response = (
        scripted_tool_call(
            "generation.register_hypothesis",
            payload,
            tool_call_id="direct-registration",
        )
        if case == "protocol_direct"
        else _bridge(payload, f"{case}-call")
    )
    kwargs: dict[str, object] = {}
    if case == "permission_deny":
        kwargs["permission_policy"] = DenyAllPolicy()
    elif case == "approval_reject":
        kwargs["permission_policy"] = AskPolicy()
        kwargs["approval_handler"] = DenyHandler()
    runtime = make_runtime(
        FakeModelProvider([response, FakeResponse(text="done")]),
        workspace=Workspace(tmp_path),
        **kwargs,
    )
    if case == "execution_exception":
        _replace_registration_tool(runtime, _ExplodingRegistrationTool())
    elif case == "tool_result_reject":
        _replace_registration_tool(runtime, _RejectedRegistrationTool())
    elif case == "state_update_invalid":
        _replace_registration_tool(runtime, _InvalidUpdateRegistrationTool())

    events = await collect(runtime, f"reject {case}")

    rejected = [
        event for event in events if event.kind == "hypothesis_registration_rejected"
    ]
    assert [(event.request_id, event.reason_code) for event in rejected] == [
        ("event-r1", reason_code)
    ]
    assert runtime.scientific_state.material_hypotheses == []
    assert not any(event.kind == "hypothesis_registered" for event in events)


@pytest.mark.asyncio
async def test_hypothesis_event_log_omits_full_basis_text(tmp_path):
    secret_basis = "FULL PRIVATE BASIS " + "source passage " * 40
    evidence_body = "FULL PRIVATE EVIDENCE " + "article body " * 40
    payload = _arguments(
        basis=[
            {
                "evidence_id": "basis-1",
                "relation": "supports",
                "anchor": secret_basis,
            }
        ]
    )
    logger = EventLogger(tmp_path, session_id="hypothesis-events")
    model = FakeModelProvider(
        [_bridge(payload, "call-with-basis"), FakeResponse(text="done")]
    )
    runtime = make_runtime(
        model,
        workspace=Workspace(tmp_path),
        event_sinks=[logger.log],
    )
    runtime.scientific_state.add_evidence(
        Evidence(
            id="basis-1",
            type="literature",
            source="paper",
            content=evidence_body,
            confidence=0.5,
        )
    )

    await collect(runtime, "register sourced hypothesis")

    raw_log = logger.events_path.read_text(encoding="utf-8")
    assert secret_basis not in raw_log
    assert evidence_body not in raw_log
    payloads = [json.loads(line) for line in raw_log.splitlines()]
    parsed = [parse_event(item) for item in payloads]
    registered = next(
        item for item in payloads if item["kind"] == "hypothesis_registered"
    )
    assert set(registered) >= {"hypothesis_id", "candidate_id", "request_id"}
    assert any(event.kind == "tool_call_arguments_delta" for event in parsed)
    assert any(event.kind == "tool_call_completed" for event in parsed)
    assert any(event.kind == "tool_requested" for event in parsed)
