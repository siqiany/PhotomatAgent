from __future__ import annotations

import json

import pytest

from photomatagent.models.fake import FakeModelProvider, FakeResponse, scripted_tool_call
from photomatagent.runtime.observation import ObservationPolicy, ObservationPolicyConfig
from photomatagent.runtime.permissions import DenyAllPolicy
from photomatagent.scientific.discovery.models import HypothesisRegistration
from photomatagent.scientific.capabilities.generation.hypotheses import (
    RegisterHypothesisTool,
)
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.base import Tool, ToolResult
from photomatagent.tools.exposure import ToolExposure
from photomatagent.tools.factory import create_default_registry
from photomatagent.workspace import Workspace

from conftest import collect, make_runtime


def arguments(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "request_id": "runtime-r1",
        "formula": "Na0.75Ag0.25BiS2",
        "statement": "Explore isovalent substitution",
        "design_operation": "isovalent_substitution",
        "validation_questions": ["Is phase separation preferred?"],
    }
    payload.update(changes)
    return payload


def bridge_call(payload: dict[str, object], *, call_id: str) -> FakeResponse:
    return scripted_tool_call(
        "tool_call",
        {"name": "generation.register_hypothesis", "arguments": payload},
        tool_call_id=call_id,
    )


@pytest.mark.asyncio
async def test_runtime_registers_with_true_call_context_through_bridge(tmp_path) -> None:
    model = FakeModelProvider(
        [bridge_call(arguments(), call_id="provider-call-7"), FakeResponse(text="done")]
    )
    runtime = make_runtime(model, workspace=Workspace(tmp_path))

    events = await collect(runtime, "register it")

    record = runtime.scientific_state.material_hypotheses[0]
    assert record.origin.tool_name == "generation.register_hypothesis"
    assert record.origin.tool_call_id == "provider-call-7"
    assert record.origin.session_id == runtime.session_id
    assert record.origin.run_id == events[0].run_id
    assert record.origin.provider == "fake"
    assert record.origin.model == "fake"
    assert record.proposal.request_id == "runtime-r1"
    kinds = [event.kind for event in events]
    assert kinds.index("tool_completed") < kinds.index("scientific_state_updated")


@pytest.mark.asyncio
async def test_runtime_retry_is_idempotent_and_changed_payload_is_rejected(tmp_path) -> None:
    model = FakeModelProvider(
        [
            bridge_call(arguments(), call_id="call-1"),
            bridge_call(arguments(), call_id="call-2"),
            bridge_call(arguments(statement="changed"), call_id="call-3"),
            FakeResponse(text="done"),
        ]
    )
    runtime = make_runtime(model, workspace=Workspace(tmp_path))

    events = await collect(runtime, "register and retry")

    assert len(runtime.scientific_state.material_hypotheses) == 1
    assert len([event for event in events if event.kind == "tool_completed"]) == 2
    failed = [event for event in events if event.kind == "tool_failed"]
    assert len(failed) == 1
    assert "request_id" in failed[0].error


@pytest.mark.asyncio
async def test_runtime_rejects_unknown_references_without_partial_state(tmp_path) -> None:
    payload = arguments(
        parent_hypothesis_ids=["hyp_missing"],
        basis=[{"evidence_id": "ev_missing", "relation": "supports"}],
    )
    model = FakeModelProvider(
        [bridge_call(payload, call_id="call-invalid"), FakeResponse(text="done")]
    )
    runtime = make_runtime(model, workspace=Workspace(tmp_path))

    events = await collect(runtime, "register invalid")

    assert runtime.scientific_state.material_hypotheses == []
    assert any(event.kind == "tool_failed" for event in events)
    assert not any(event.kind == "tool_completed" for event in events)


@pytest.mark.asyncio
async def test_denied_and_invalid_schema_never_append_hypothesis(tmp_path) -> None:
    denied_model = FakeModelProvider(
        [bridge_call(arguments(), call_id="denied"), FakeResponse(text="done")]
    )
    denied_runtime = make_runtime(
        denied_model,
        workspace=Workspace(tmp_path),
        permission_policy=DenyAllPolicy(),
    )
    await collect(denied_runtime, "deny")
    assert denied_runtime.scientific_state.material_hypotheses == []

    invalid_model = FakeModelProvider(
        [
            bridge_call({**arguments(), "origin": {"provider": "forged"}}, call_id="bad"),
            FakeResponse(text="done"),
        ]
    )
    invalid_runtime = make_runtime(invalid_model, workspace=Workspace(tmp_path))
    await collect(invalid_runtime, "invalid")
    assert invalid_runtime.scientific_state.material_hypotheses == []


@pytest.mark.asyncio
async def test_hidden_registration_tool_cannot_append_hypothesis(tmp_path) -> None:
    class HiddenRegistrationTool(RegisterHypothesisTool):
        exposure = ToolExposure.HIDDEN

    model = FakeModelProvider(
        [
            scripted_tool_call(
                "generation.register_hypothesis", arguments(), tool_call_id="hidden"
            ),
            FakeResponse(text="done"),
        ]
    )
    runtime = make_runtime(model, workspace=Workspace(tmp_path))
    runtime._tools._tools["generation.register_hypothesis"] = HiddenRegistrationTool(
        runtime.scientific_state
    )

    events = await collect(runtime, "hidden")

    assert runtime.scientific_state.material_hypotheses == []
    assert any(event.kind == "tool_failed" for event in events)


class TwoUpdatesTool(Tool):
    name = "test.two_updates"
    exposure = ToolExposure.DIRECT
    input_schema = {"type": "object", "properties": {}}

    async def execute(self, arguments: dict[str, object]) -> ToolResult:
        good = HypothesisRegistration.model_validate({"proposal": arguments_template()})
        return ToolResult(
            output="must not be reported successful",
            state_updates=[good, object()],
        )


def arguments_template() -> dict[str, object]:
    return arguments(request_id="atomic")


@pytest.mark.asyncio
async def test_runtime_validates_all_updates_before_atomic_application(tmp_path) -> None:
    model = FakeModelProvider(
        [scripted_tool_call("test.two_updates", {}), FakeResponse(text="done")]
    )
    runtime = make_runtime(model, workspace=Workspace(tmp_path))
    runtime._tools._tools["test.two_updates"] = TwoUpdatesTool()

    events = await collect(runtime, "atomic")

    assert runtime.scientific_state.material_hypotheses == []
    assert any(event.kind == "tool_failed" for event in events)
    assert not any(event.kind == "tool_completed" for event in events)


@pytest.mark.asyncio
async def test_registration_failure_observation_is_bounded_and_redacted(tmp_path) -> None:
    secret = "sk-" + "a" * 40
    model = FakeModelProvider(
        [
            bridge_call(arguments(request_id="x" * 129, statement=secret * 30), call_id="bad"),
            FakeResponse(text="done"),
        ]
    )
    runtime = make_runtime(model, workspace=Workspace(tmp_path))
    runtime._observation = ObservationPolicy(
        ObservationPolicyConfig(default_max_chars=256)
    )

    events = await collect(runtime, "invalid long secret")

    failed = next(event for event in events if event.kind == "tool_failed")
    assert failed.delivered_chars <= 256
    assert secret not in failed.error


def test_default_registry_passes_its_live_state_and_rejects_duplicate_name(tmp_path) -> None:
    class MCPRegistrationShadow(RegisterHypothesisTool):
        source = "mcp"

    state = ScientificState()
    registry = create_default_registry(state, Workspace(tmp_path))
    tool = registry.get("generation.register_hypothesis")

    assert tool.state is state
    with pytest.raises(ValueError, match="already registered"):
        registry.register(MCPRegistrationShadow(state))
