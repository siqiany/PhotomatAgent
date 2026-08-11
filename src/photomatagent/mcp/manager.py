"""Production MCP scientific gateway manager.

Lifecycle: ``discover -> start -> connect -> list_tools -> invoke ->
healthcheck -> disconnect -> restart``. A failing server never breaks agent
startup: at registry build time it is represented by a ``<namespace>.status``
stub tool that reports START_FAILED / UNHEALTHY / MISSING_DEPENDENCY details
instead of raising.

Transports use the official ``mcp`` SDK exactly as documented upstream:
``stdio_client`` / ``streamable_http_client`` context managers wrapped in a
``ClientSession`` with ``initialize`` + ``list_tools`` + ``call_tool`` +
``send_ping``. Secrets from ``${VAR}`` env references are expanded at spawn
time and never appear in tool output, status reports, or traces.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import timedelta
from typing import Any

from photomatagent.mcp.config import MCPServerConfig, load_mcp_servers
from photomatagent.scientific.capabilities.contracts import (
    ScientificEvidence,
    ScientificToolResult,
)
from photomatagent.tools.base import Tool
from photomatagent.tools.exposure import ToolExposure

MAX_MCP_RESULT_CHARS = 16000
MAX_EVIDENCE_ITEMS = 5


class MCPServerState(str, Enum):
    """Lifecycle state of one MCP server connection."""

    UNCONFIGURED = "UNCONFIGURED"
    DISABLED = "DISABLED"
    MISSING_DEPENDENCY = "MISSING_DEPENDENCY"
    START_FAILED = "START_FAILED"
    UNHEALTHY = "UNHEALTHY"
    READY = "READY"
    STOPPED = "STOPPED"


@dataclass(frozen=True)
class RemoteToolSpec:
    """One tool advertised by a remote MCP server."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)

    @property
    def required_parameters(self) -> list[str]:
        required = self.input_schema.get("required", [])
        return [str(item) for item in required] if isinstance(required, list) else []


@dataclass(frozen=True)
class MCPStatusRow:
    """CLI-friendly status snapshot for one server."""

    name: str
    transport: str
    state: MCPServerState
    namespace: str
    tools: int
    latency_ms: float | None = None
    error: str = ""
    detail: str = ""


class _MissingCommand(RuntimeError):
    pass


def resolve_command(command: str) -> str | None:
    """Resolve an executable name to an absolute path.

    Tries PATH first, then the active interpreter's own bin directory so
    ``uv run``-style venv entry points work without activation.
    """
    if os.path.sep in command or (os.path.altsep and os.path.altsep in command):
        return command if os.path.exists(command) else None
    found = shutil.which(command)
    if found:
        return found
    candidate = Path(sys.executable).parent / command
    return str(candidate) if candidate.is_file() else None


class MCPServerHandle:
    """One live (or attempted) MCP server connection.

    The handle owns the transport context manager and the ``ClientSession``.
    ``invoke`` reconnects lazily if the server dropped, so an agent turn that
    outlives a server restart still works.
    """

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self.state = MCPServerState.UNCONFIGURED
        self.detail = ""
        self.last_error = ""
        self.latency_ms: float | None = None
        self.remote_tools: list[RemoteToolSpec] = []
        self.started_at: str = ""
        self._session: Any = None
        self._transport_cm: Any = None
        self._lock = asyncio.Lock()

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> MCPServerState:
        """Spawn/connect, initialize, and list remote tools."""
        async with self._lock:
            if self.state is MCPServerState.READY and self._session is not None:
                return self.state
            await self._close_locked()
            if not self.config.enabled:
                self.state = MCPServerState.DISABLED
                self.detail = "server is disabled in configuration"
                return self.state
            read, write, transport = await self._open_transport()
            if read is None:
                return self.state
            self._transport_cm = transport
            try:
                from mcp import ClientSession

                # Official SDK usage: __aenter__ only starts the receive
                # loop; the initialize handshake must be sent explicitly
                # (mcp>=1.24 no longer auto-initializes). We keep the context
                # open for the handle's lifetime and close it in
                # ``_close_locked``.
                session = await asyncio.wait_for(
                    ClientSession(
                        read,
                        write,
                        read_timeout_seconds=timedelta(
                            seconds=self.config.timeout_seconds
                        ),
                    ).__aenter__(),
                    timeout=self.config.startup_timeout_seconds,
                )
                self._session = session
                await asyncio.wait_for(
                    session.initialize(),
                    timeout=self.config.startup_timeout_seconds,
                )
                response = await asyncio.wait_for(
                    session.list_tools(), timeout=self.config.timeout_seconds
                )
            except asyncio.TimeoutError:
                self.state = MCPServerState.START_FAILED
                self.detail = "timed out during initialize/list_tools"
                self.last_error = "startup timeout"
                await self._close_locked()
                return self.state
            except Exception as exc:
                self.state = MCPServerState.START_FAILED
                self.detail = f"initialize failed: {type(exc).__name__}"
                self.last_error = str(exc)[:500]
                await self._close_locked()
                return self.state
            self.remote_tools = [
                RemoteToolSpec(
                    name=str(tool.name),
                    description=str(tool.description or ""),
                    input_schema=dict(tool.inputSchema or {}),
                )
                for tool in getattr(response, "tools", [])
            ]
            self.state = MCPServerState.READY
            self.started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self.last_error = ""
            return self.state

    async def _open_transport(self) -> tuple[Any, Any, Any]:
        """Return (read, write, transport_cm) or (None, None, None) on failure."""
        resolved_env, missing = self.config.resolved_env()
        if missing:
            self.detail = f"unresolved env references: {', '.join(missing)}"
        try:
            if self.config.transport == "http":
                url = self.config.resolved_url()
                if not url:
                    self.state = MCPServerState.START_FAILED
                    self.detail = "http transport requires a url"
                    self.last_error = "missing url"
                    return None, None, None
                from mcp.client.streamable_http import streamable_http_client

                transport = streamable_http_client(url)
                http_streams: Any = await asyncio.wait_for(
                    transport.__aenter__(),
                    timeout=self.config.startup_timeout_seconds,
                )
                read, write = http_streams[0], http_streams[1]
                return read, write, transport
            else:
                command = resolve_command(self.config.resolved_command() or "")
                if command is None:
                    self.state = MCPServerState.MISSING_DEPENDENCY
                    self.detail = (
                        f"command {self.config.command!r} not found on PATH or in "
                        "the active venv bin directory"
                    )
                    self.last_error = "executable not found"
                    return None, None, None
                from mcp import StdioServerParameters
                from mcp.client.stdio import stdio_client

                params = StdioServerParameters(
                    command=command,
                    args=self.config.resolved_args(),
                    env=resolved_env,
                )
                transport = stdio_client(params)
                stdio_streams: Any = await asyncio.wait_for(
                    transport.__aenter__(),
                    timeout=self.config.startup_timeout_seconds,
                )
                read, write = stdio_streams[0], stdio_streams[1]
                return read, write, transport
        except asyncio.TimeoutError:
            self.state = MCPServerState.START_FAILED
            self.detail = "timed out while opening transport"
            self.last_error = "transport timeout"
            return None, None, None
        except _MissingCommand:
            self.state = MCPServerState.MISSING_DEPENDENCY
            self.detail = f"command {self.config.command!r} is not available"
            self.last_error = "executable not found"
            return None, None, None
        except Exception as exc:
            self.state = MCPServerState.START_FAILED
            self.detail = f"transport failed: {type(exc).__name__}"
            self.last_error = str(exc)[:500]
            return None, None, None

    async def invoke(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> tuple[str, bool, dict[str, Any]]:
        """Invoke a remote tool; returns (text, is_error, structured)."""
        if self.state is not MCPServerState.READY or self._session is None:
            await self.start()
        if self.state is not MCPServerState.READY or self._session is None:
            return (
                self._failure_text(),
                True,
                {"state": self.state.value, "error": self.last_error},
            )
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(tool_name, arguments),
                timeout=self.config.timeout_seconds,
            )
            latency = (time.perf_counter() - started) * 1000.0
            self.latency_ms = round(latency, 1)
            self.state = MCPServerState.READY
            return self._format_result(result)
        except asyncio.TimeoutError:
            self.state = MCPServerState.UNHEALTHY
            self.last_error = f"{tool_name} timed out after {self.config.timeout_seconds}s"
            return (
                f"mcp:{self.config.name} tool {tool_name} timed out "
                f"(timeout={self.config.timeout_seconds}s)",
                True,
                {"state": self.state.value, "error": self.last_error},
            )
        except Exception as exc:
            self.state = MCPServerState.UNHEALTHY
            self.last_error = str(exc)[:500]
            return (
                f"mcp:{self.config.name} tool {tool_name} failed: "
                f"{type(exc).__name__}: {self.last_error}",
                True,
                {"state": self.state.value, "error": self.last_error},
            )

    def _failure_text(self) -> str:
        return (
            f"mcp server {self.config.name} is not available: "
            f"state={self.state.value}; detail={self.detail or self.last_error or 'no detail'}"
        )

    def _format_result(
        self, result: Any
    ) -> tuple[str, bool, dict[str, Any]]:
        if isinstance(result, dict):
            content = result.get("content") or []
            is_error = bool(result.get("isError", False))
            pieces: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("text") is not None:
                    pieces.append(str(item["text"]))
            text = "\n".join(pieces) or json.dumps(result, ensure_ascii=False)
            if len(text) > MAX_MCP_RESULT_CHARS:
                text = text[:MAX_MCP_RESULT_CHARS] + "\n...[truncated]"
            return text, is_error, {}
        content_items = getattr(result, "content", None) or []
        out_pieces: list[str] = []
        for item in content_items:
            item_text = getattr(item, "text", None)
            if item_text is not None:
                out_pieces.append(str(item_text))
            elif isinstance(item, dict) and item.get("text") is not None:
                out_pieces.append(str(item["text"]))
        out_text = "\n".join(out_pieces)
        if not out_text:
            structured = getattr(result, "structured_content", None)
            out_text = json.dumps(structured or {}, ensure_ascii=False)
        if len(out_text) > MAX_MCP_RESULT_CHARS:
            out_text = out_text[:MAX_MCP_RESULT_CHARS] + "\n...[truncated]"
        is_error = bool(
            getattr(result, "isError", False)
            or getattr(result, "is_error", False)
        )
        return out_text, is_error, {}

    async def healthcheck(self) -> dict[str, Any]:
        """Ping the server and return a bounded status snapshot."""
        if self.state is not MCPServerState.READY or self._session is None:
            await self.start()
        if self.state is not MCPServerState.READY or self._session is None:
            return {
                "server": self.config.name,
                "state": self.state.value,
                "detail": self.detail or self.last_error,
                "tools": len(self.remote_tools),
                "latency_ms": None,
            }
        started = time.perf_counter()
        try:
            await asyncio.wait_for(
                self._session.send_ping(), timeout=self.config.timeout_seconds
            )
            self.latency_ms = round((time.perf_counter() - started) * 1000.0, 1)
            self.state = MCPServerState.READY
            return {
                "server": self.config.name,
                "state": MCPServerState.READY.value,
                "detail": "ping ok",
                "tools": len(self.remote_tools),
                "latency_ms": self.latency_ms,
            }
        except Exception as exc:
            self.state = MCPServerState.UNHEALTHY
            self.last_error = str(exc)[:500]
            return {
                "server": self.config.name,
                "state": MCPServerState.UNHEALTHY.value,
                "detail": f"ping failed: {type(exc).__name__}",
                "tools": len(self.remote_tools),
                "latency_ms": None,
            }

    async def close(self) -> None:
        async with self._lock:
            await self._close_locked()

    async def _close_locked(self) -> None:
        session, self._session = self._session, None
        transport, self._transport_cm = self._transport_cm, None
        if session is not None:
            try:
                await session.__aexit__(None, None, None)
            except Exception:
                pass
        if transport is not None:
            try:
                await transport.__aexit__(None, None, None)
            except Exception:
                pass
        if self.state not in {
            MCPServerState.START_FAILED,
            MCPServerState.MISSING_DEPENDENCY,
            MCPServerState.DISABLED,
        }:
            self.state = MCPServerState.STOPPED

    async def restart(self) -> MCPServerState:
        await self.close()
        return await self.start()

    def status_row(self) -> MCPStatusRow:
        return MCPStatusRow(
            name=self.config.name,
            transport=self.config.transport,
            state=self.state,
            namespace=self.config.effective_namespace,
            tools=len(self.remote_tools) if self.state is MCPServerState.READY else 0,
            latency_ms=self.latency_ms,
            error=self.last_error,
            detail=self.detail,
        )


def _auto_connect_enabled() -> bool:
    value = os.environ.get("PHOTOMATAGENT_MCP_AUTO_CONNECT", "1").strip().lower()
    return value in {"1", "true", "yes", "on"}


class MCPServerManager:
    """Discover, connect, and register MCP servers as deferred tools."""

    def __init__(
        self,
        configs: list[MCPServerConfig] | None = None,
        *,
        workspace: Path | str | None = None,
        auto_connect: bool | None = None,
    ) -> None:
        self.workspace = Path(workspace or Path.cwd()).expanduser().resolve()
        self.configs = list(configs if configs is not None else load_mcp_servers(self.workspace))
        self.handles: dict[str, MCPServerHandle] = {
            cfg.name: MCPServerHandle(cfg) for cfg in self.configs
        }
        self.auto_connect = (
            _auto_connect_enabled() if auto_connect is None else auto_connect
        )

    # -- discovery / status --------------------------------------------------

    def discovered(self) -> list[MCPServerConfig]:
        return list(self.configs)

    def status_rows(self) -> list[MCPStatusRow]:
        return [handle.status_row() for handle in self.handles.values()]

    def live_status(self) -> list[MCPStatusRow]:
        """Best-effort live probe (used by ``mcp status`` / ``mcp doctor``)."""
        for handle in self.handles.values():
            self._run_async(handle.start())
            self._run_async(handle.healthcheck())
        return self.status_rows()

    def doctor(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for handle in self.handles.values():
            cfg = handle.config
            resolved_env, missing = cfg.resolved_env()
            command = resolve_command(cfg.resolved_command() or "")
            report = {
                "server": cfg.name,
                "transport": cfg.transport,
                "enabled": cfg.enabled,
                "namespace": cfg.effective_namespace,
                "command": command if command is not None else cfg.command,
                "url": cfg.resolved_url() if cfg.transport == "http" else None,
                "trust_level": cfg.trust_level,
                "env_keys": sorted(resolved_env),
                "missing_env_refs": missing,
                "timeout_seconds": cfg.timeout_seconds,
                "startup_timeout_seconds": cfg.startup_timeout_seconds,
                "tool_exposure": cfg.exposure.value,
            }
            self._run_async(handle.start())
            report["state"] = handle.state.value
            report["detail"] = handle.detail or handle.last_error
            report["tools"] = len(handle.remote_tools)
            report["latency_ms"] = handle.latency_ms
            rows.append(report)
        return rows

    # -- registration ---------------------------------------------------------

    def register_tools(self) -> list[Tool]:
        """Return status stubs plus (when reachable) remote tool adapters.

        Never raises: unreadable servers degrade to a status stub whose
        invocation reports the typed failure instead of crashing the agent.
        """
        tools: list[Tool] = []
        for handle in self.handles.values():
            tools.append(MCPServerStatusTool(handle))
            if not handle.config.enabled:
                handle.state = MCPServerState.DISABLED
                continue
            if self.auto_connect:
                self._start_handle_sync(handle)
            if handle.state is MCPServerState.READY:
                for spec in handle.remote_tools:
                    tools.append(MCPRemoteTool(handle, spec))
        return tools

    @staticmethod
    def _start_handle_sync(handle: MCPServerHandle) -> None:
        try:
            asyncio.get_running_loop()
            return  # inside an event loop: lazy reconnect happens on invoke
        except RuntimeError:
            pass
        try:
            asyncio.run(handle.start())
        except Exception as exc:
            handle.state = MCPServerState.START_FAILED
            handle.last_error = str(exc)[:500]

    @staticmethod
    def _run_async(coro: Any) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                asyncio.run(coro)
            except Exception:
                pass
            return
        loop = asyncio.get_running_loop()
        if loop.is_running():
            import threading

            result: dict[str, Any] = {}

            def worker() -> None:
                result["value"] = asyncio.run(coro)

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            thread.join(timeout=90)
        else:
            loop.run_until_complete(coro)


class MCPRemoteTool(Tool):
    """Adapter exposing one remote MCP tool as a local deferred Tool."""

    def __init__(self, handle: MCPServerHandle, spec: RemoteToolSpec) -> None:
        self._handle = handle
        self._spec = spec
        self.name = f"{handle.config.effective_namespace}.{spec.name}"
        self.description = _tool_description(handle.config, spec)
        self.short_description = (spec.description or spec.name)[:200]
        self.input_schema = spec.input_schema or {"type": "object", "properties": {}}
        self.namespace = handle.config.effective_namespace
        self.exposure = handle.config.exposure
        self.source = f"mcp:{handle.config.name}"
        self.tags = ("mcp", "scientific", handle.config.name, handle.config.effective_namespace)

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        text, is_error, structured = await self._handle.invoke(
            self._spec.name, arguments or {}
        )
        evidence = _evidence_from_result(
            self._handle, self._spec.name, text, is_error
        )
        data = {
            "server": self._handle.config.name,
            "remote_tool": self._spec.name,
            "state": self._handle.state.value,
            **structured,
        }
        return ScientificToolResult(
            output=text,
            is_error=is_error,
            data=data,
            evidence=evidence,
        )


class MCPServerStatusTool(Tool):
    """Health/status report for one MCP server (always registered)."""

    def __init__(self, handle: MCPServerHandle) -> None:
        self._handle = handle
        ns = handle.config.effective_namespace
        self.name = f"{ns}.status"
        self.description = (
            f"Report the connection state of MCP server {handle.config.name!r} "
            "(READY / UNCONFIGURED / MISSING_DEPENDENCY / START_FAILED / "
            "UNHEALTHY / DISABLED), its advertised tools, and last latency. "
            "Never contains secrets."
        )
        self.short_description = f"MCP server {handle.config.name} status/health."
        self.namespace = ns
        self.exposure = ToolExposure.DEFERRED
        self.source = f"mcp:{handle.config.name}"
        self.tags = ("mcp", "status", "healthcheck", handle.config.name)
        self.input_schema = {"type": "object", "properties": {}}

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        handle = self._handle
        row = handle.status_row()
        payload = {
            "server": row.name,
            "transport": row.transport,
            "state": row.state.value,
            "namespace": row.namespace,
            "tools": row.tools,
            "latency_ms": row.latency_ms,
            "detail": row.detail or row.error,
            "hint": (
                "run `photomatagent mcp status` or `photomatagent mcp doctor` "
                "for a live probe"
                if row.state
                in {
                    MCPServerState.UNCONFIGURED,
                    MCPServerState.MISSING_DEPENDENCY,
                    MCPServerState.START_FAILED,
                    MCPServerState.UNHEALTHY,
                }
                else ""
            ),
        }
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False),
            data=payload,
        )


def _tool_description(config: MCPServerConfig, spec: RemoteToolSpec) -> str:
    base = (
        f"Remote MCP tool {spec.name!r} exposed by server {config.name!r} "
        f"(transport={config.transport}, trust={config.trust_level}). "
        f"Result is produced by an external scientific service; treat as "
        f"evidence from source_type=database or external solver, not as a "
        f"local calculation."
    )
    if spec.description:
        base = f"{base} Server description: {spec.description}"
    return base


def _evidence_from_result(
    handle: MCPServerHandle,
    tool_name: str,
    text: str,
    is_error: bool,
) -> list[ScientificEvidence]:
    """Best-effort evidence extraction from structured MCP text results.

    Only JSON payloads with recognizable ``band_gap``/``material_id`` fields
    are converted; everything else is left as opaque text so no hallucinated
    evidence is fabricated.
    """
    if is_error:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    items = parsed.get("results", parsed) if isinstance(parsed, dict) else parsed
    if not isinstance(items, list):
        items = [items] if isinstance(items, dict) else []
    evidence: list[ScientificEvidence] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        gap = item.get("band_gap")
        if gap is None:
            continue
        try:
            value = float(gap)
        except (TypeError, ValueError):
            continue
        material_id = str(item.get("material_id", item.get("id", "")))
        evidence.append(
            ScientificEvidence(
                subject=material_id or tool_name,
                property="band_gap",
                value=value,
                unit="eV",
                source=f"MCP server {handle.config.name} ({tool_name})",
                source_type="database",
                method=f"mcp:{handle.config.name}:{tool_name}",
                summary=f"MCP database band gap for {material_id or 'material'} is {value:.3f} eV",
                limitations=(
                    "value returned by external MCP service; trust level "
                    f"{handle.config.trust_level}"
                ),
                provenance={
                    "server": handle.config.name,
                    "remote_tool": tool_name,
                    "material_id": material_id,
                },
            )
        )
        if len(evidence) >= MAX_EVIDENCE_ITEMS:
            break
    return evidence
