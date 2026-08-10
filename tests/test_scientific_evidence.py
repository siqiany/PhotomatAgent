"""ScientificEvidence / ScientificToolResult contract and runtime wiring."""

from __future__ import annotations

import pytest

from photomatagent.models.fake import FakeModelProvider, FakeResponse, scripted_tool_call
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import AllowAllPolicy
from photomatagent.scientific.capabilities.contracts import (
    ScientificEvidence,
    ScientificToolResult,
)
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.base import Tool
from photomatagent.tools.bridges import ToolCallBridge
from photomatagent.tools.exposure import ToolExposure
from photomatagent.tools.registry import ToolRegistry
from photomatagent.workspace import Workspace


def test_scientific_evidence_carries_provenance():
    evidence = ScientificEvidence(
        subject="HgTe",
        property="band_gap",
        value=0.0,
        unit="eV",
        source="Materials Project",
        source_type="database",
        method="mp-api",
        summary="gap 0.0 eV",
        limitations="DFT-derived",
        provenance={"material_id": "mp-1990"},
    )
    assert evidence.id.startswith("sev_")
    assert evidence.source_type == "database"
    assert evidence.provenance["material_id"] == "mp-1990"


def test_scientific_tool_result_auto_appends_evidence_to_state_updates():
    evidence = ScientificEvidence(
        subject="x", property="y", value=1, unit="eV", source="test"
    )
    result = ScientificToolResult(output="ok", evidence=[evidence])
    assert any(item is evidence for item in result.state_updates)


async def _run_with_tool(tool: Tool, goal: str) -> AgentRuntime:
    state = ScientificState()
    registry = ToolRegistry()
    registry.register(ToolCallBridge())
    registry.register(tool)
    runtime = AgentRuntime(
        model=FakeModelProvider(
            [
                scripted_tool_call(
                    "tool_call",
                    {"name": tool.name, "arguments": {}},
                ),
                FakeResponse(text="done"),
            ]
        ),
        tools=registry,
        workspace=Workspace("."),
        scientific_state=state,
        permission_policy=AllowAllPolicy(),
    )
    events = [event async for event in runtime.run(goal)]
    assert any(event.kind == "scientific_trace_meta" for event in events)
    runtime._collected_events = events
    return runtime


class _EvidenceTool(Tool):
    name = "scientific.test_evidence"
    exposure = ToolExposure.DEFERRED
    namespace = "testpack"

    async def execute(self, arguments):
        evidence = ScientificEvidence(
            subject="GaAs",
            property="density",
            value=5.32,
            unit="g/cm3",
            source="pymatgen",
            method="computed",
            summary="density 5.32 g/cm3",
        )
        return ScientificToolResult(output="5.32 g/cm3", evidence=[evidence])


@pytest.mark.asyncio
async def test_evidence_lands_in_scientific_state_and_trace():
    runtime = await _run_with_tool(_EvidenceTool(), "analyze density")
    assert len(runtime.scientific_state.evidence) == 1
    stored = runtime.scientific_state.evidence[0]
    assert stored.property == "density"
    assert stored.source == "pymatgen"
    # Innovation trace fields were populated.
    meta = [
        event
        for event in _collected(runtime)
        if event.kind == "scientific_trace_meta"
    ]
    assert meta and meta[0].evidence_created == 1
    assert meta[0].evidence_sources == ["pymatgen"]
    assert "scientific.test_evidence" in meta[0].scientific_tools_used


def _collected(runtime):
    return getattr(runtime, "_collected_events", [])
