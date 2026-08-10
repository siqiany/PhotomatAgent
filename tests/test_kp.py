"""kdotpy adapter tests (probe + subprocess run, offline-friendly)."""

from __future__ import annotations

import asyncio
import json

import pytest

from photomatagent.scientific.capabilities.base import CapabilityStatus
from photomatagent.scientific.capabilities.kp import (
    KdotpyCapabilitiesTool,
    KdotpyRunTool,
)
from photomatagent.scientific.capabilities.quantum_dot.provider import (
    probe_kdotpy,
)
from photomatagent.workspace import Workspace


def test_probe_reports_available_with_isolated_venv():
    workspace = Workspace(".")
    probe = probe_kdotpy(workspace.root)
    if probe.status is CapabilityStatus.AVAILABLE:
        assert probe.version
    else:
        # Either state is fine offline; the important part is a clear
        # diagnostic rather than a crash.
        assert probe.detail


@pytest.mark.asyncio
async def test_run_tool_requires_args_or_config():
    tool = KdotpyRunTool(Workspace("."))
    result = await tool.execute({})
    assert result.is_error
    assert "requires" in result.output


@pytest.mark.asyncio
async def test_run_tool_typed_error_when_unavailable(monkeypatch):
    from photomatagent.scientific.capabilities import kp as kp_module

    def fake_probe(workspace_root=None):  # noqa: ANN001
        from photomatagent.scientific.capabilities.base import ProbeResult

        return ProbeResult(
            status=CapabilityStatus.MISSING_DEPENDENCY,
            detail="kdotpy not installed anywhere",
        )

    monkeypatch.setattr(kp_module, "probe_kdotpy", fake_probe)
    tool = KdotpyRunTool(Workspace("."))
    result = await tool.execute({"args": ["version"]})
    assert result.is_error
    payload = json.loads(result.output)
    assert payload["error_type"] == "external_solver_unavailable"


@pytest.mark.asyncio
async def test_capabilities_tool_reports_scope():
    tool = KdotpyCapabilitiesTool(Workspace("."))
    result = await tool.execute({})
    assert not result.is_error
    payload = json.loads(result.output)
    assert "0D" in payload["scope"] or "0d" in payload["scope"]
    assert "quantum-dot" in payload["warning"]


@pytest.mark.asyncio
async def test_real_kdotpy_version_run_when_available():
    workspace = Workspace(".")
    probe = probe_kdotpy(workspace.root)
    if probe.status is not CapabilityStatus.AVAILABLE:
        pytest.skip("kdotpy venv not present in this checkout")
    tool = KdotpyRunTool(workspace)
    result = await tool.execute({"args": ["version"]})
    assert result.is_error is False
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["exit_code"] == 0
    assert any(char.isdigit() for char in payload["stdout_tail"])
    assert result.evidence[0].source_type == "kp_calculation"
    assert result.evidence[0].fidelity == "kp"
