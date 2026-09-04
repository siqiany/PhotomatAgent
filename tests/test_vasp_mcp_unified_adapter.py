"""Task 13: SCNet MCP adapters converge to the unified VASP service."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from photomatagent.mcp.config import MCPServerConfig
from photomatagent.mcp.manager import MCPServerManager, RemoteToolSpec
from photomatagent.workspace import Workspace

UNIFIED_BUILTIN = {
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


def test_unified_factory_graph_is_the_shared_typed_composition_root(
    tmp_path,
) -> None:
    """The production pack and MCP adapter must share one named graph."""
    from photomatagent.scientific.applications.vasp.application import (
        VaspApplication,
    )
    from photomatagent.scientific.applications.vasp.tools import VaspCapabilityPack
    from photomatagent.scientific.applications.vasp.unified.factory import (
        UnifiedVaspGraph,
        build_unified_vasp_graph,
    )
    from photomatagent.scientific.remote.fake import FakeSCNetBackend

    application = VaspApplication(FakeSCNetBackend(), workspace=tmp_path)
    graph = build_unified_vasp_graph(application=application, workspace=tmp_path)
    pack = VaspCapabilityPack(application=application, workspace=tmp_path)

    assert isinstance(graph, UnifiedVaspGraph)
    assert graph.service is graph.study.child_service
    assert pack.unified_graph is pack.unified_graph
    assert pack.unified_graph.service is pack.unified_graph.study.child_service


@pytest.mark.asyncio
async def test_server_vasp_capabilities_uses_the_injected_unified_graph(
    tmp_path,
) -> None:
    """An unconfigured graph remains a typed unified service response."""
    from photomatagent.mcp_servers.scnet import server
    from photomatagent.scientific.applications.vasp.unified.factory import (
        build_unified_vasp_graph,
    )

    server._set_unified_vasp_graph_for_test(
        build_unified_vasp_graph(application=None, workspace=tmp_path)
    )
    try:
        payload = await server.vasp_capabilities()
    finally:
        server._set_unified_vasp_graph_for_test(None)

    assert payload["is_error"] is False
    assert payload["workflow_kinds"] == ["periodic", "molecular", "study"]


@pytest.mark.asyncio
async def test_server_lazily_builds_one_unified_graph_per_process_owner(
    monkeypatch, tmp_path
) -> None:
    """Repeated aliases reuse one server-owned graph rather than rebuilding it."""
    from photomatagent.mcp_servers.scnet import server
    from photomatagent.scientific.applications.vasp.unified import factory

    calls = 0

    class FakeService:
        def capabilities(self, workflow_kind: str | None) -> dict[str, Any]:
            return {"ok": True, "workflow_kinds": [workflow_kind]}

    def build_graph(**kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return SimpleNamespace(service=FakeService())

    monkeypatch.setattr(factory, "build_unified_vasp_graph", build_graph)
    server._set_unified_vasp_graph_for_test(None)
    try:
        await server.vasp_capabilities("periodic")
        await server.vasp_capabilities("molecular")
    finally:
        server._set_unified_vasp_graph_for_test(None)

    assert calls == 1


def test_server_advertises_only_the_unified_transport_aliases() -> None:
    from photomatagent.mcp_servers.scnet import server

    names = {
        component.name
        for component in server.mcp._local_provider._components.values()
        if getattr(component, "name", "").startswith("vasp")
    }
    submit = next(
        component
        for component in server.mcp._local_provider._components.values()
        if getattr(component, "name", "") == "vasp_submit"
    )

    assert names == {
        "vasp_capabilities", "vasp_plan", "vasp_prepare", "vasp_preflight",
        "vasp_submit", "vasp_status", "vasp_wait", "vasp_resume",
        "vasp_collect", "vasp_report",
    }
    assert set(submit.parameters["properties"]) == {"workflow_id", "stage"}
    forbidden = {
        "input_dir", "workflow_dir", "result_dir", "job_id", "job_name",
        "partition", "nodes", "tasks_per_node", "memory", "walltime",
        "approval_id", "approved", "fingerprint", "force_new_attempt",
    }
    assert not (forbidden & set(submit.parameters["properties"]))


@pytest.mark.asyncio
async def test_server_aliases_delegate_once_and_preserve_service_payload() -> None:
    from photomatagent.mcp_servers.scnet import server

    class FakeService:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def capabilities(self, workflow_kind: str | None) -> dict[str, Any]:
            self.calls.append("capabilities")
            return {"ok": True, "workflow_kinds": [workflow_kind]}

        def plan(self, request: Any) -> dict[str, Any]:
            self.calls.append("plan")
            return {"ok": True, "workflow_kind": request.workflow_kind.value}

        async def prepare(self, workflow_id: str) -> dict[str, Any]:
            return self._result("prepare", workflow_id)

        async def preflight(self, workflow_id: str) -> dict[str, Any]:
            return self._result("preflight", workflow_id)

        async def submit(self, workflow_id: str, stage: str | None) -> dict[str, Any]:
            return self._result("submit", workflow_id, stage=stage)

        async def status(self, workflow_id: str) -> dict[str, Any]:
            return self._result("status", workflow_id)

        async def wait(self, workflow_id: str) -> dict[str, Any]:
            return self._result("wait", workflow_id)

        async def resume(self, workflow_id: str) -> dict[str, Any]:
            return self._result("resume", workflow_id)

        async def collect(self, workflow_id: str) -> dict[str, Any]:
            return self._result("collect", workflow_id)

        async def report(self, workflow_id: str, request: Any) -> dict[str, Any]:
            return self._result("report", workflow_id, report_kind=request.kind.value)

        def _result(self, name: str, workflow_id: str, **extra: Any) -> dict[str, Any]:
            self.calls.append(name)
            return {
                "ok": True,
                "workflow_id": workflow_id,
                "state": "PREFLIGHTED",
                "evidence": [{"source": "offline"}],
                "evidence_gaps": ["none"],
                "pending_decision": {"kind": "approval", "state": "PENDING"},
                "artifacts": ["artifact.json"],
                "provenance": {"safe": "yes", "token": "redacted"},
                **extra,
            }

    service = FakeService()
    server._set_unified_vasp_graph_for_test(SimpleNamespace(service=service))
    try:
        planned = await server.vasp_plan(
            "periodic",
            {"kind": "periodic", "structure_path": "in.POSCAR", "profile": "standard_semiconductor"},
        )
        payloads = [
            await server.vasp_capabilities("periodic"),
            await server.vasp_prepare("vasp_0123456789abcdef"),
            await server.vasp_preflight("vasp_0123456789abcdef"),
            await server.vasp_submit("vasp_0123456789abcdef", "relax"),
            await server.vasp_status("vasp_0123456789abcdef"),
            await server.vasp_wait("vasp_0123456789abcdef"),
            await server.vasp_resume("vasp_0123456789abcdef"),
            await server.vasp_collect("vasp_0123456789abcdef"),
            await server.vasp_report("vasp_0123456789abcdef", {"kind": "summary"}),
        ]
    finally:
        server._set_unified_vasp_graph_for_test(None)

    assert planned["workflow_kind"] == "periodic"
    assert service.calls == [
        "plan", "capabilities", "prepare", "preflight", "submit", "status",
        "wait", "resume", "collect", "report",
    ]
    assert all(payload["is_error"] is False for payload in payloads)
    assert payloads[-1]["evidence"] == [{"source": "offline"}]
    assert payloads[-1]["pending_decision"]["state"] == "PENDING"
    assert "token" not in payloads[-1]["provenance"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("workflow_kind", "scientific_spec"),
    [
        ("periodic", {"kind": "molecular"}),
        ("molecular", {"kind": "molecular", "workflow": {}}),
        ("study", {"kind": "study", "request": {"systems": "not-a-list"}}),
    ],
)
async def test_server_plan_rejects_malformed_nested_unified_specs(
    workflow_kind: str, scientific_spec: dict[str, Any]
) -> None:
    """Transport input uses the same discriminated Pydantic model as tools."""
    from photomatagent.mcp_servers.scnet import server

    class ServiceMustNotRun:
        def plan(self, request: Any) -> dict[str, Any]:
            raise AssertionError(f"invalid request reached service: {request!r}")

    server._set_unified_vasp_graph_for_test(SimpleNamespace(service=ServiceMustNotRun()))
    try:
        payload = await server.vasp_plan(workflow_kind, scientific_spec)
    finally:
        server._set_unified_vasp_graph_for_test(None)

    assert payload["is_error"] is True
    assert payload["error_type"] == "ValidationError"


@pytest.mark.asyncio
async def test_server_vasp_errors_redact_paths_values_and_long_exception_text() -> None:
    """MCP errors expose locations/types, never submitted values or secrets."""
    from photomatagent.mcp_servers.scnet import server

    class ExplodingService:
        async def prepare(self, workflow_id: str) -> dict[str, Any]:
            raise RuntimeError(
                "/home/user/PRIVATE_TOKEN/run private_key=TOPSECRET " + "x" * 20_000
            )

    class PlanMustNotRun:
        def plan(self, request: Any) -> dict[str, Any]:
            raise AssertionError("invalid input reached service")

    class TypedErrorService:
        async def status(self, workflow_id: str) -> dict[str, Any]:
            return {
                "ok": False,
                "errors": ["/tmp/PRIVATE_TOKEN/result token=TOPSECRET"],
            }

    server._set_unified_vasp_graph_for_test(SimpleNamespace(service=ExplodingService()))
    try:
        service_error = await server.vasp_prepare("workflow")
    finally:
        server._set_unified_vasp_graph_for_test(None)

    server._set_unified_vasp_graph_for_test(SimpleNamespace(service=PlanMustNotRun()))
    try:
        validation_error = await server.vasp_plan(
            "periodic",
            {
                "kind": "periodic",
                "structure_path": {"private_key": "/workspace/PRIVATE_TOKEN/TOPSECRET"},
                "profile": "TOPSECRET",
            },
        )
    finally:
        server._set_unified_vasp_graph_for_test(None)

    server._set_unified_vasp_graph_for_test(SimpleNamespace(service=TypedErrorService()))
    try:
        typed_error = await server.vasp_status("workflow")
    finally:
        server._set_unified_vasp_graph_for_test(None)

    for payload in (service_error, validation_error, typed_error):
        rendered = json.dumps(payload)
        assert "PRIVATE_TOKEN" not in rendered
        assert "TOPSECRET" not in rendered
        assert "/home/user" not in rendered
        assert "/workspace" not in rendered
        if "message" in payload:
            assert len(payload["message"]) <= 512
            assert payload["message"] == payload["output"]
    assert validation_error["error_type"] == "ValidationError"


def test_server_error_keeps_safe_field_location_without_inline_input_value() -> None:
    """Inline Pydantic rendering retains the field but never its value."""
    from photomatagent.mcp_servers.scnet import server

    payload = server._error(
        "scientific_spec.periodic.profile Input should be a valid string "
        "[type=string_type, input_value='TOPSECRET', input_type=str]",
        "ValidationError",
    )

    assert "scientific_spec.periodic.profile" in payload["message"]
    assert "TOPSECRET" not in payload["message"]


@pytest.mark.asyncio
async def test_non_vasp_scnet_tools_remain_available_without_legacy_vasp_helper(
    monkeypatch,
) -> None:
    """NAMD validation and partition discovery must not need a legacy VASP API."""
    from photomatagent.mcp_servers.scnet import server

    class FakeNamd:
        def validate_inputs(self, trajectory_dir: str) -> list[str]:
            assert trajectory_dir == "trajectory"
            return []

    class FakeBackend:
        async def available_partitions(self) -> list[str]:
            return ["debug", "normal"]

    monkeypatch.setattr(server, "_namd_application", lambda: FakeNamd())
    monkeypatch.setattr(server, "_partition_backend", lambda: FakeBackend())

    assert await server.namd_validate_inputs("trajectory") == {
        "trajectory_dir": "trajectory",
        "problems": [],
        "valid": True,
    }
    assert (await server.scnet_partitions())["partitions"] == ["debug", "normal"]


@pytest.mark.asyncio
async def test_doctor_keeps_vasp_diagnostics_after_legacy_helper_removal(
    monkeypatch,
) -> None:
    """The diagnostic path uses configured VASP only for read-only probing."""
    from photomatagent.mcp_servers.scnet import server

    monkeypatch.setattr(server, "_unified_vasp_service", lambda: object())
    monkeypatch.setattr(server, "_namd_application", lambda: None)
    monkeypatch.setattr(server, "_magus_application", lambda: None)
    monkeypatch.setattr(
        "photomatagent.scientific.applications.vasp.application.default_vasp_application",
        lambda: None,
    )

    report = await server.build_doctor_report()

    assert report["vasp"]["error"] == "no backend configured"


def test_manager_detail_does_not_advertise_removed_duplicate_vasp_override(
    monkeypatch, tmp_path
) -> None:
    """Duplicate filtering is unconditional even in its diagnostic detail."""
    session = _FakeSession()
    _patch_sdk(monkeypatch, session)
    config = MCPServerConfig(
        name="scnet",
        command=sys.executable,
        namespace="scnet_science",
        timeout_seconds=1,
        startup_timeout_seconds=1,
    )
    manager = MCPServerManager([config], auto_connect=True)
    try:
        manager.register_tools(builtin_tool_names=UNIFIED_BUILTIN)
        assert "PHOTOMATAGENT_MCP_INCLUDE_DUPLICATE_MOLECULAR" not in manager.handles["scnet"].detail
    finally:
        manager.close_all()


def test_manager_filters_vasp_only_for_the_complete_unified_surface() -> None:
    """A partial local VASP pack must not hide a remote lifecycle adapter."""
    config = MCPServerConfig(name="scnet", namespace="scnet_science")
    manager = MCPServerManager([config], auto_connect=False)

    assert not manager._is_duplicate_molecular_adapter(
        manager.handles["scnet"],
        RemoteToolSpec(name="vasp_submit"),
        UNIFIED_BUILTIN - {"vasp.wait"},
    )


class _FakeTransport:
    async def __aenter__(self):
        return object(), object()

    async def __aexit__(self, *exc_info: Any):
        pass


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def initialize(self) -> None:
        pass

    async def list_tools(self) -> Any:
        tools = [
            {"name": "status", "description": "status", "inputSchema": {"type": "object", "properties": {}}},
            {"name": "vasp_capabilities", "description": "vasp caps", "inputSchema": {"type": "object", "properties": {}}},
            {"name": "vasp_molecule_prepare", "description": "molecular prepare", "inputSchema": {"type": "object", "properties": {}}},
            {"name": "vasp_study_plan", "description": "study plan", "inputSchema": {"type": "object", "properties": {}}},
            {"name": "namd_run", "description": "NAMD", "inputSchema": {"type": "object", "properties": {}}},
            {"name": "magus_search", "description": "MAGUS", "inputSchema": {"type": "object", "properties": {}}},
        ]
        return SimpleNamespace(tools=[SimpleNamespace(**tool) for tool in tools])

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, arguments))
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps({"ok": True}))],
            isError=False,
        )

    async def __aexit__(self, *exc_info: Any) -> None:
        pass


def _patch_sdk(monkeypatch, session: _FakeSession) -> None:
    transport = _FakeTransport()

    def fake_stdio_client(params):  # noqa: ANN001
        return transport

    class FakeClientSession:
        def __init__(self, read, write, **kwargs) -> None:  # noqa: ANN001
            pass

        async def __aenter__(self) -> _FakeSession:
            return session

        async def __aexit__(self, *exc_info: Any) -> None:
            pass

    monkeypatch.setattr("mcp.client.stdio.stdio_client", fake_stdio_client)
    monkeypatch.setattr("mcp.ClientSession", FakeClientSession)


def test_all_vasp_scnet_adapters_skipped_when_unified_pack_exists(monkeypatch, tmp_path):
    session = _FakeSession()
    _patch_sdk(monkeypatch, session)
    config = MCPServerConfig(
        name="scnet",
        command=sys.executable,
        namespace="scnet_science",
        timeout_seconds=1,
        startup_timeout_seconds=1,
    )
    manager = MCPServerManager([config], auto_connect=True)
    tools = manager.register_tools(builtin_tool_names=UNIFIED_BUILTIN)
    names = {tool.name for tool in tools}
    assert "scnet_science.status" in names
    assert "scnet_science.namd_run" in names
    assert "scnet_science.magus_search" in names
    assert "scnet_science.vasp_capabilities" not in names
    assert "scnet_science.vasp_molecule_prepare" not in names
    assert "scnet_science.vasp_study_plan" not in names
    manager.close_all()


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE", "Yes"])
def test_no_env_setting_reenables_duplicate_vasp_adapters(monkeypatch, tmp_path, value):
    session = _FakeSession()
    _patch_sdk(monkeypatch, session)
    monkeypatch.setenv("PHOTOMATAGENT_MCP_INCLUDE_DUPLICATE_MOLECULAR", value)
    config = MCPServerConfig(
        name="scnet",
        command=sys.executable,
        namespace="scnet_science",
        timeout_seconds=1,
        startup_timeout_seconds=1,
    )
    manager = MCPServerManager([config], auto_connect=True)
    tools = manager.register_tools(builtin_tool_names=UNIFIED_BUILTIN)
    names = {tool.name for tool in tools}
    assert "scnet_science.vasp_capabilities" not in names
    assert "scnet_science.vasp_molecule_prepare" not in names
    manager.close_all()
