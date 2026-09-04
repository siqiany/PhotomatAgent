"""Task 0: characterize the real VASP tool-surface baseline.

These tests intentionally distinguish:
- registry membership (all registered VASP tools, including legacy families)
- model visibility (progressive manifest / eager definitions)

The registry helper disables MCP auto-connect so no online server is required.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from photomatagent.models.fake import FakeModelProvider, FakeResponse, scripted_tool_call
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import AllowAllPolicy
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.bridges import ToolDescribeTool
from photomatagent.tools.factory import create_default_registry
from photomatagent.tools.registry import ToolRegistry
from photomatagent.tools.surface import ToolCatalog, ToolSurfaceConfig, ToolSurfacePlanner
from photomatagent.workspace import Workspace

PUBLIC = {
    "vasp.capabilities",
    "vasp.plan",
    "vasp.prepare",
    "vasp.preflight",
    "vasp.submit",
    "vasp.status",
    "vasp.wait",
    "vasp.resume",
    "vasp.collect",
    "vasp.report",
}

LEGACY_PERIODIC_NAMES = {
    "vasp.inspect_result",
    "vasp.run_workflow",
}
LEGACY_MOLECULAR_NAMES = {
    "vasp_molecule.analyze_esp",
    "vasp_molecule.analyze_orbitals",
    "vasp_molecule.binding_energy",
    "vasp_molecule.capabilities",
    "vasp_molecule.collect",
    "vasp_molecule.preflight",
    "vasp_molecule.prepare",
    "vasp_molecule.resume_workflow",
    "vasp_molecule.status",
    "vasp_molecule.submit",
}
LEGACY_STUDY_NAMES = {
    "vasp_study.collect",
    "vasp_study.execute",
    "vasp_study.plan",
    "vasp_study.report",
    "vasp_study.resume",
    "vasp_study.status",
}

# Names drawn from the three families currently registered by VaspCapabilityPack.
# The exact set may evolve, but the descriptive baseline should observe all three.
LEGACY_VASP_SAMPLE = (
    {"vasp.prepare", "vasp.submit"}
    | LEGACY_PERIODIC_NAMES
    | LEGACY_MOLECULAR_NAMES
    | LEGACY_STUDY_NAMES
)


def is_vasp_family_name(name: str) -> bool:
    """True for the VASP families this plan is consolidating."""
    return name.startswith("vasp") or name.startswith("scnet_science.vasp_")


def build_registry(workspace: Path | str | None = None) -> tuple[ToolRegistry, set[str]]:
    """Build the default registry with MCP auto-connect disabled.

    Returns (registry, all registered VASP-family tool names).  The returned
    set is intentionally registry membership, not model visibility.
    """
    os.environ["PHOTOMATAGENT_MCP_AUTO_CONNECT"] = "0"
    root = Path(workspace) if workspace else Path.cwd()
    registry = create_default_registry(ScientificState(), Workspace(root))
    names = {tool.name for tool in registry.list_tools() if is_vasp_family_name(tool.name)}
    return registry, names


def assert_unique_vasp_surface(names: set[str]) -> None:
    visible = {name for name in names if is_vasp_family_name(name)}
    assert visible == PUBLIC


def vasp_names_from_progressive_manifest(manifest_text: str) -> set[str]:
    """Extract VASP family names from a progressive capability manifest."""
    pattern = re.compile(
        r"(?:vasp(?:_molecule|_study)?\.[A-Za-z0-9_.]+|scnet_science\.vasp_[A-Za-z0-9_]+)"
    )
    return set(pattern.findall(manifest_text))


# ---------------------------------------------------------------------------
# Descriptive baseline
# ---------------------------------------------------------------------------


def test_registry_public_vasp_names_equal_documented_surface(tmp_path):
    """After consolidation, only the unified-name surface is registered."""
    _registry, names = build_registry(tmp_path)

    assert_unique_vasp_surface(names)


# ---------------------------------------------------------------------------
# Offline MCP characterization
# ---------------------------------------------------------------------------


def test_registry_helper_builds_with_mcp_auto_connect_disabled(monkeypatch, tmp_path):
    """Use a fake MCP handle and assert auto-connect never starts a server."""
    from photomatagent.mcp.config import MCPServerConfig
    from photomatagent.mcp.manager import MCPServerManager, MCPServerState

    monkeypatch.setenv("PHOTOMATAGENT_MCP_AUTO_CONNECT", "0")
    config = MCPServerConfig(
        name="fake-vasp-mcp",
        namespace="scnet_science",
        command="definitely-not-started",
    )
    manager = MCPServerManager([config], auto_connect=False)
    handle = manager.handles["fake-vasp-mcp"]

    tools = manager.register_tools()

    assert [tool.name for tool in tools] == ["scnet_science.status"]
    assert handle.state is MCPServerState.UNCONFIGURED
    assert handle.remote_tools == []


# ---------------------------------------------------------------------------
# Strict target tests: these fail until Task 12 registers only PUBLIC tools.
# ---------------------------------------------------------------------------


def test_progressive_surface_exposes_only_unified_vasp_names(tmp_path):
    registry, _ = build_registry(tmp_path)
    plan = ToolSurfacePlanner(
        registry, ToolSurfaceConfig(mode="progressive")
    ).plan()
    names = {item.name for item in plan.definitions}
    names.update(vasp_names_from_progressive_manifest(plan.manifest.text))

    assert_unique_vasp_surface(names)


def test_eager_surface_exposes_only_unified_vasp_names(tmp_path):
    registry, _ = build_registry(tmp_path)
    plan = ToolSurfacePlanner(registry, ToolSurfaceConfig(mode="eager")).plan()
    names = {item.name for item in plan.definitions}

    assert_unique_vasp_surface(names)


def test_capability_search_returns_only_unified_vasp_names(tmp_path):
    registry, _ = build_registry(tmp_path)
    catalog = ToolCatalog(registry)
    queries = ("vasp", "VASP molecule", "VASP study", "submit VASP")

    for query in queries:
        matches = catalog.search(query, limit=20)
        found = {match.entry.name for match in matches if is_vasp_family_name(match.entry.name)}
        assert_unique_vasp_surface(found)


@pytest.mark.asyncio
async def test_describe_rejects_legacy_vasp_names(tmp_path):
    registry, _ = build_registry(tmp_path)
    tool = ToolDescribeTool(ToolCatalog(registry))

    result = await tool.execute({"name": "vasp_molecule.capabilities"})

    assert result.is_error
    assert result.data.get("error") == "not_deferred_or_unavailable"


@pytest.mark.asyncio
async def test_bridged_call_rejects_legacy_vasp_name(tmp_path):
    model = FakeModelProvider(
        [
            scripted_tool_call(
                "tool_call",
                {
                    "name": "vasp_molecule.capabilities",
                    "arguments": {},
                },
            ),
            FakeResponse(text="done"),
        ]
    )
    registry, _ = build_registry(tmp_path)
    runtime = AgentRuntime(
        model=model,
        tools=registry,
        workspace=Workspace(tmp_path),
        permission_policy=AllowAllPolicy(),
    )

    events = [event async for event in runtime.run("use vasp molecule")]

    completed = [
        event for event in events if event.kind == "tool_completed"
    ]
    assert not any(
        event.tool_name == "vasp_molecule.capabilities"
        for event in completed
    )
    assert any(
        event.kind == "tool_failed"
        and "vasp_molecule.capabilities" in event.error
        and "unknown tool" in event.error
        for event in events
    )


@pytest.mark.asyncio
async def test_guessed_direct_call_rejects_legacy_vasp_name(tmp_path):
    model = FakeModelProvider(
        [
            scripted_tool_call("vasp_molecule.capabilities", {}),
            FakeResponse(text="done"),
        ]
    )
    registry, _ = build_registry(tmp_path)
    runtime = AgentRuntime(
        model=model,
        tools=registry,
        workspace=Workspace(tmp_path),
        permission_policy=AllowAllPolicy(),
        tool_surface_config=ToolSurfaceConfig(mode="eager"),
    )

    events = [event async for event in runtime.run("use vasp molecule")]

    assert any(
        event.kind == "tool_failed" and event.tool_name == "vasp_molecule.capabilities"
        for event in events
    )
    assert not any(
        event.kind == "tool_completed" and event.tool_name == "vasp_molecule.capabilities"
        for event in events
    )
