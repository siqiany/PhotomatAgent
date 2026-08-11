"""MCP gateway manager tests (official SDK transport, mocked for offline runs).

Offline tests patch the ``mcp`` SDK transports so no process or network is
required. Live tests spawn a real FastMCP stdio server through the official
SDK and are gated by ``PHOTOMATAGENT_RUN_LIVE_SCIENCE=1`` (they need a normal
Linux environment; the Codex sandbox blocks the SDK's process plumbing).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from photomatagent.mcp.config import MCPServerConfig, load_mcp_servers
from photomatagent.mcp.manager import (
    MCPRemoteTool,
    MCPServerHandle,
    MCPServerManager,
    MCPServerState,
    MCPServerStatusTool,
    resolve_command,
)
from photomatagent.scientific.capabilities.registry import build_scientific_tools
from photomatagent.tools.exposure import ToolExposure
from photomatagent.workspace import Workspace


# ---------------------------------------------------------------------------
# Configuration parsing
# ---------------------------------------------------------------------------


def test_load_servers_new_style_dict(tmp_path):
    (tmp_path / ".photomatagent").mkdir()
    (tmp_path / ".photomatagent" / "mcp.json").write_text(
        json.dumps(
            {
                "servers": {
                    "materials-project": {
                        "enabled": True,
                        "transport": "stdio",
                        "namespace": "materials_mcp",
                        "command": "mpmcp",
                        "env": {"MP_API_KEY": "${MATERIALS_API_KEY}"},
                        "trust_level": "local_trusted",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    servers = load_mcp_servers(tmp_path)
    assert len(servers) == 1
    server = servers[0]
    assert server.name == "materials-project"
    assert server.effective_namespace == "materials_mcp"
    assert server.transport == "stdio"
    assert server.trust_level == "local_trusted"


def test_load_servers_legacy_list_style(tmp_path):
    (tmp_path / "mcp.json").write_text(
        json.dumps(
            {
                "servers": [
                    {"name": "legacy", "command": "some-cmd", "args": ["--x"]}
                ]
            }
        ),
        encoding="utf-8",
    )
    servers = load_mcp_servers(tmp_path)
    assert [s.name for s in servers] == ["legacy"]
    assert servers[0].args == ["--x"]


def test_load_servers_missing_file_yields_none(tmp_path):
    assert load_mcp_servers(tmp_path) == []


def test_env_reference_expansion(monkeypatch):
    monkeypatch.setenv("MATERIALS_API_KEY", "secret-abc")
    config = MCPServerConfig(
        name="m", command="mpmcp", env={"MP_API_KEY": "${MATERIALS_API_KEY}"}
    )
    env, missing = config.resolved_env()
    assert env["MP_API_KEY"] == "secret-abc"
    assert missing == []


def test_env_reference_missing_var_reported(monkeypatch):
    monkeypatch.delenv("MATERIALS_API_KEY", raising=False)
    config = MCPServerConfig(
        name="m", command="mpmcp", env={"MP_API_KEY": "${MATERIALS_API_KEY}"}
    )
    env, missing = config.resolved_env()
    assert env["MP_API_KEY"] == ""
    assert missing == ["MATERIALS_API_KEY"]


def test_env_reference_falls_back_to_workspace_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("MATERIALS_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        "MATERIALS_API_KEY=secret-from-dotenv\n", encoding="utf-8"
    )
    config = MCPServerConfig(
        name="m",
        workspace=str(tmp_path),
        command="mpmcp",
        env={"MP_API_KEY": "${MATERIALS_API_KEY}"},
    )
    env, missing = config.resolved_env()
    assert env["MP_API_KEY"] == "secret-from-dotenv"
    assert missing == []


def test_process_env_wins_over_workspace_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("MATERIALS_API_KEY", "secret-from-env")
    (tmp_path / ".env").write_text(
        "MATERIALS_API_KEY=secret-from-dotenv\n", encoding="utf-8"
    )
    config = MCPServerConfig(
        name="m",
        workspace=str(tmp_path),
        command="mpmcp",
        env={"MP_API_KEY": "${MATERIALS_API_KEY}"},
    )
    env, _ = config.resolved_env()
    assert env["MP_API_KEY"] == "secret-from-env"


def test_transport_aliases_normalized_by_parser(tmp_path):
    (tmp_path / ".photomatagent").mkdir()
    (tmp_path / ".photomatagent" / "mcp.json").write_text(
        json.dumps(
            {
                "servers": {
                    "m": {
                        "transport": "streamable-http",
                        "url": "http://localhost:8000/mcp",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    server = load_mcp_servers(tmp_path)[0]
    assert server.transport == "http"


def test_config_defaults_and_namespace():
    config = MCPServerConfig(name="m")
    assert config.exposure is ToolExposure.DEFERRED
    assert config.effective_namespace == "m"
    assert config.transport == "stdio"
    assert config.trust_level == "local_isolated"


def test_command_resolution_uses_venv_bin(tmp_path, monkeypatch):
    bin_dir = Path(sys.executable).parent
    fake = bin_dir / "definitely-not-a-real-tool-xyz"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o755)
    try:
        monkeypatch.setenv("PATH", str(tmp_path))
        assert resolve_command("definitely-not-a-real-tool-xyz") == str(fake)
    finally:
        fake.unlink()


def test_command_resolution_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert resolve_command("no-such-executable-xyz") is None


# ---------------------------------------------------------------------------
# Fake official-SDK plumbing (offline)
# ---------------------------------------------------------------------------


class _FakeTransport:
    def __init__(self) -> None:
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> tuple[Any, Any]:
        self.entered = True
        return object(), object()

    async def __aexit__(self, *exc_info: Any) -> None:
        self.exited = True


class _FakeSession:
    def __init__(
        self,
        *,
        tools: list[dict[str, Any]] | None = None,
        call_result: str = "{}",
        call_error: bool = False,
        fail_initialize: Exception | None = None,
        ping_error: Exception | None = None,
    ) -> None:
        self.tools = tools or [
            {
                "name": "search",
                "description": "Search materials.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]
        self.call_result = call_result
        self.call_error = call_error
        self.fail_initialize = fail_initialize
        self.ping_error = ping_error
        self.exited = False
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def initialize(self) -> None:
        if self.fail_initialize is not None:
            raise self.fail_initialize

    async def list_tools(self) -> Any:
        return SimpleNamespace(tools=[SimpleNamespace(**tool) for tool in self.tools])

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, arguments))
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.call_result)],
            isError=self.call_error,
        )

    async def send_ping(self) -> None:
        if self.ping_error is not None:
            raise self.ping_error

    async def __aexit__(self, *exc_info: Any) -> None:
        self.exited = True


def _patch_sdk(monkeypatch, session: _FakeSession) -> _FakeTransport:
    transport = _FakeTransport()

    def fake_stdio_client(params):  # noqa: ANN001
        return transport

    class FakeClientSession:
        def __init__(self, read, write, **kwargs) -> None:  # noqa: ANN001
            pass

        async def __aenter__(self) -> _FakeSession:
            # Mirror the official SDK: __aenter__ only starts the receive
            # loop; the manager sends initialize() explicitly afterwards.
            return session

        async def __aexit__(self, *exc_info: Any) -> None:
            session.exited = True

    monkeypatch.setattr("mcp.client.stdio.stdio_client", fake_stdio_client)
    monkeypatch.setattr("mcp.ClientSession", FakeClientSession)
    return transport


def _config(**overrides: Any) -> MCPServerConfig:
    base: dict[str, Any] = {
        "name": "fake-server",
        "command": sys.executable,
    }
    base.update(overrides)
    return MCPServerConfig(**base)


# ---------------------------------------------------------------------------
# Handle lifecycle (offline, mocked SDK)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_start_lists_tools(monkeypatch):
    session = _FakeSession()
    _patch_sdk(monkeypatch, session)
    handle = MCPServerHandle(_config())
    state = await handle.start()
    assert state is MCPServerState.READY
    assert [t.name for t in handle.remote_tools] == ["search"]
    assert handle.remote_tools[0].required_parameters == ["query"]


@pytest.mark.asyncio
async def test_handle_invoke_returns_text_and_evidence(monkeypatch):
    payload = json.dumps(
        {"results": [{"material_id": "mp-1", "band_gap": 0.35}]}
    )
    session = _FakeSession(call_result=payload)
    _patch_sdk(monkeypatch, session)
    handle = MCPServerHandle(_config())
    await handle.start()
    text, is_error, data = await handle.invoke("search", {"query": "InAs"})
    assert not is_error
    assert "mp-1" in text
    assert session.calls == [("search", {"query": "InAs"})]


@pytest.mark.asyncio
async def test_handle_healthcheck_and_restart(monkeypatch):
    session = _FakeSession()
    _patch_sdk(monkeypatch, session)
    handle = MCPServerHandle(_config())
    await handle.start()
    report = await handle.healthcheck()
    assert report["state"] == "READY"
    assert report["latency_ms"] is not None
    state = await handle.restart()
    assert state is MCPServerState.READY


@pytest.mark.asyncio
async def test_handle_disabled_state():
    handle = MCPServerHandle(_config(enabled=False))
    state = await handle.start()
    assert state is MCPServerState.DISABLED


@pytest.mark.asyncio
async def test_handle_missing_command_is_diagnosed(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    handle = MCPServerHandle(_config(command="definitely-missing-cmd-xyz"))
    state = await handle.start()
    assert state is MCPServerState.MISSING_DEPENDENCY
    assert "not found" in handle.detail


@pytest.mark.asyncio
async def test_handle_start_failure_is_typed_not_raised(monkeypatch):
    session = _FakeSession(fail_initialize=RuntimeError("boom"))
    transport = _patch_sdk(monkeypatch, session)
    handle = MCPServerHandle(_config())
    state = await handle.start()
    assert state is MCPServerState.START_FAILED
    assert "boom" in handle.last_error
    assert session.exited
    assert transport.exited


@pytest.mark.asyncio
async def test_invoke_timeout_marks_unhealthy(monkeypatch):
    class SlowSession(_FakeSession):
        async def call_tool(self, name, arguments):  # noqa: ANN001
            await asyncio.sleep(5)

    session = SlowSession()
    _patch_sdk(monkeypatch, session)
    handle = MCPServerHandle(_config(timeout_seconds=0.2))
    await handle.start()
    text, is_error, data = await handle.invoke("search", {"query": "x"})
    assert is_error
    assert "timed out" in text
    assert handle.state is MCPServerState.UNHEALTHY


@pytest.mark.asyncio
async def test_close_and_restart_after_failure(monkeypatch):
    session = _FakeSession()
    _patch_sdk(monkeypatch, session)
    handle = MCPServerHandle(_config())
    await handle.start()
    await handle.close()
    assert handle.state is MCPServerState.STOPPED
    assert await handle.restart() is MCPServerState.READY


# ---------------------------------------------------------------------------
# Manager registration surface
# ---------------------------------------------------------------------------


def test_register_tools_auto_connect_registers_adapters(monkeypatch):
    session = _FakeSession()
    _patch_sdk(monkeypatch, session)
    manager = MCPServerManager(
        [_config(namespace="materials_mcp")], auto_connect=True
    )
    tools = manager.register_tools()
    names = [tool.name for tool in tools]
    assert "materials_mcp.status" in names
    assert "materials_mcp.search" in names
    adapter = next(tool for tool in tools if tool.name == "materials_mcp.search")
    assert isinstance(adapter, MCPRemoteTool)
    assert adapter.exposure is ToolExposure.DEFERRED
    assert adapter.tags == ("mcp", "scientific", "fake-server", "materials_mcp")


def test_register_tools_without_auto_connect_only_stubs(monkeypatch):
    _patch_sdk(monkeypatch, _FakeSession())
    manager = MCPServerManager([_config()], auto_connect=False)
    tools = manager.register_tools()
    assert [tool.name for tool in tools] == ["fake_server.status"]
    assert isinstance(tools[0], MCPServerStatusTool)


def test_register_tools_failed_server_still_registers_stub(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    manager = MCPServerManager(
        [_config(command="definitely-missing-cmd-xyz")], auto_connect=True
    )
    asyncio.run(manager.handles["fake-server"].start())
    tools = manager.register_tools()
    assert [tool.name for tool in tools] == ["fake_server.status"]


@pytest.mark.asyncio
async def test_status_stub_invocation_reports_typed_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    manager = MCPServerManager(
        [_config(command="definitely-missing-cmd-xyz")], auto_connect=True
    )
    await manager.handles["fake-server"].start()
    tools = manager.register_tools()
    stub = tools[0]
    result = await stub.execute({})
    assert result.is_error is False
    payload = json.loads(result.output)
    assert payload["state"] == "MISSING_DEPENDENCY"
    assert "not found" in payload["detail"]


@pytest.mark.asyncio
async def test_remote_tool_invocation_via_registry(monkeypatch):
    session = _FakeSession(call_result=json.dumps({"ok": True}))
    _patch_sdk(monkeypatch, session)
    manager = MCPServerManager(
        [_config(namespace="materials_mcp")], auto_connect=True
    )
    # register_tools() skips auto-connect inside a running loop, so start the
    # handle explicitly first (mirrors the CLI path where no loop is running).
    await manager.handles["fake-server"].start()
    tools = manager.register_tools()
    adapter = next(tool for tool in tools if tool.name == "materials_mcp.search")
    result = await adapter.execute({"query": "HgTe"})
    assert not result.is_error
    assert json.loads(result.output)["ok"] is True
    # evidence flows into ScientificToolResult.state_updates
    assert result.evidence == []


def test_build_scientific_tools_registers_mcp_from_workspace(
    monkeypatch, tmp_path
):
    session = _FakeSession()
    _patch_sdk(monkeypatch, session)
    monkeypatch.setenv("PHOTOMATAGENT_MCP_AUTO_CONNECT", "1")
    (tmp_path / ".photomatagent").mkdir()
    (tmp_path / ".photomatagent" / "mcp.json").write_text(
        json.dumps(
            {
                "servers": {
                    "fake": {
                        "command": sys.executable,
                        "namespace": "materials_mcp",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    tools = build_scientific_tools(workspace=Workspace(tmp_path))
    names = {tool.name for tool in tools}
    assert "materials_mcp.status" in names
    assert "materials_mcp.search" in names


def test_evidence_extraction_from_bandgap_json(monkeypatch):
    session = _FakeSession(
        call_result=json.dumps(
            {
                "results": [
                    {"material_id": "mp-20305", "band_gap": 0.0},
                    {"material_id": "mp-1", "band_gap": 0.354},
                ]
            }
        )
    )
    _patch_sdk(monkeypatch, session)
    manager = MCPServerManager([_config()], auto_connect=True)
    asyncio.run(manager.handles["fake-server"].start())
    tools = manager.register_tools()
    adapter = next(tool for tool in tools if tool.name == "fake_server.search")
    result = asyncio.run(adapter.execute({"query": "InAs"}))
    assert len(result.evidence) == 2
    first = result.evidence[0]
    assert first.source_type == "database"
    assert first.property == "band_gap"
    assert first.provenance["remote_tool"] == "search"


# ---------------------------------------------------------------------------
# Live test against a real FastMCP stdio server (official SDK end to end)
# ---------------------------------------------------------------------------


LIVE_FAKE_SERVER = '''
from fastmcp import FastMCP

mcp = FastMCP("fake-live")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@mcp.tool()
def band_gap(material_id: str) -> dict:
    """Return a fixed band gap for a material id (test fixture)."""
    return {"material_id": material_id, "band_gap": 0.354}


mcp.run(transport="stdio")
'''


@pytest.mark.skipif(
    os.environ.get("PHOTOMATAGENT_RUN_LIVE_SCIENCE") != "1",
    reason="live MCP test; set PHOTOMATAGENT_RUN_LIVE_SCIENCE=1 on a normal "
    "Linux host (the Codex sandbox blocks SDK stdio plumbing)",
)
@pytest.mark.asyncio
async def test_live_fastmcp_server_roundtrip(tmp_path):
    server_path = tmp_path / "live_server.py"
    server_path.write_text(LIVE_FAKE_SERVER, encoding="utf-8")
    config = MCPServerConfig(
        name="live-fake",
        command=sys.executable,
        args=[str(server_path)],
        namespace="live",
        startup_timeout_seconds=30,
        timeout_seconds=30,
    )
    manager = MCPServerManager([config], auto_connect=True)
    state = await manager.handles["live-fake"].start()
    if state is not MCPServerState.READY:
        pytest.skip(
            "official SDK stdio cannot run in this environment "
            f"({state.value}: {manager.handles['live-fake'].detail or manager.handles['live-fake'].last_error}); "
            "run on a normal Linux host"
        )
    tools = manager.register_tools()
    names = [tool.name for tool in tools]
    assert "live.status" in names
    assert "live.add" in names
    adapter = next(tool for tool in tools if tool.name == "live.add")
    result = await adapter.execute({"a": 2, "b": 3})
    assert not result.is_error
    assert "5" in result.output
    gap = next(tool for tool in tools if tool.name == "live.band_gap")
    gap_result = await gap.execute({"material_id": "mp-1"})
    assert gap_result.evidence
    assert gap_result.evidence[0].value == 0.354
