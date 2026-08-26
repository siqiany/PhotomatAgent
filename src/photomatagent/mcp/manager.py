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
import atexit
import json
import os
import shutil
import sys
import threading
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
        # The event loop that owns the open transport/session. All lifecycle
        # calls on this handle (start/invoke/healthcheck/close/restart) must
        # run on this loop while it is alive; cross-loop calls are dispatched
        # back onto it so an anyio cancel scope is always exited in the same
        # task that entered it ("Attempted to exit cancel scope in a
        # different task" must never surface).
        self._owner_loop: asyncio.AbstractEventLoop | None = None
        # The manager's single lifecycle loop (set once the manager starts
        # one). When present, long-lived sessions are always created there,
        # never on a caller's throwaway loop.
        self._lifecycle_loop: asyncio.AbstractEventLoop | None = None

    def _owner_dispatch(self) -> asyncio.AbstractEventLoop | None:
        """Return the live owner loop when this call runs on another loop."""
        owner = self._owner_loop
        if owner is None or owner.is_closed():
            return None
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None
        if current is owner:
            return None
        return owner

    def _preferred_loop(self) -> asyncio.AbstractEventLoop | None:
        """Return the manager lifecycle loop when this call runs elsewhere."""
        loop = self._lifecycle_loop
        if loop is None or loop.is_closed():
            return None
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            return loop
        if current is loop:
            return None
        return loop

    async def _run_on_loop(
        self, loop: asyncio.AbstractEventLoop, coro: Any
    ) -> Any:
        """Await ``coro`` on ``loop`` from a different loop."""
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return await asyncio.wrap_future(future)

    async def _run_on_owner(self, coro: Any) -> Any:
        """Await ``coro`` on the owner loop from a different loop."""
        owner = self._owner_dispatch()
        assert owner is not None
        return await self._run_on_loop(owner, coro)

    def _reset_stale_locked(self) -> None:
        """Drop objects whose owner loop is dead (never touch anyio again)."""
        self._session = None
        self._transport_cm = None
        self._owner_loop = None
        if self.state in {
            MCPServerState.READY,
            MCPServerState.UNHEALTHY,
            MCPServerState.START_FAILED,
        }:
            self.state = MCPServerState.STOPPED
        self.detail = (
            "previous connection was bound to a closed event loop; "
            "reconnecting on demand"
        )

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> MCPServerState:
        """Spawn/connect, initialize, and list remote tools."""
        if (
            self._owner_dispatch() is not None
            and (self._session is not None or self._transport_cm is not None)
        ):
            return await self._run_on_owner(self.start())
        preferred = self._preferred_loop()
        if preferred is not None:
            return await self._run_on_loop(preferred, self.start())
        async with self._lock:
            if (
                self.state is MCPServerState.READY
                and self._session is not None
                and (
                    self._owner_loop is None
                    or not self._owner_loop.is_closed()
                )
            ):
                return self.state
            if (
                self._owner_loop is not None
                and self._owner_loop.is_closed()
            ):
                self._reset_stale_locked()
            await self._close_locked()
            if not self.config.enabled:
                self.state = MCPServerState.DISABLED
                self.detail = "server is disabled in configuration"
                return self.state
            try:
                read, write, transport = await self._open_transport()
            except Exception as exc:
                self.state = MCPServerState.START_FAILED
                self.detail = f"transport failed: {type(exc).__name__}"
                self.last_error = str(exc)[:500]
                return self.state
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
            self._owner_loop = asyncio.get_running_loop()
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
        if (
            self._owner_dispatch() is not None
            and self._session is not None
        ):
            return await self._run_on_owner(
                self.invoke(tool_name, arguments)
            )
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
        if (
            self._owner_dispatch() is not None
            and self._session is not None
        ):
            return await self._run_on_owner(self.healthcheck())
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
        if (
            self._owner_dispatch() is not None
            and (self._session is not None or self._transport_cm is not None)
        ):
            await self._run_on_owner(self.close())
            return
        async with self._lock:
            if (
                self._owner_loop is not None
                and self._owner_loop.is_closed()
            ):
                self._reset_stale_locked()
            await self._close_locked()

    async def _close_locked(self) -> None:
        session, self._session = self._session, None
        transport, self._transport_cm = self._transport_cm, None
        self._owner_loop = None
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
        # Single, explicit lifecycle owner for sync contexts: one persistent
        # event loop + daemon thread. Long-lived MCP sessions are never opened
        # inside a throwaway ``asyncio.run`` loop that dies before the
        # session is closed (that produced "Attempted to exit cancel scope in
        # a different task" at interpreter shutdown).
        self._lifecycle_loop: asyncio.AbstractEventLoop | None = None
        self._lifecycle_thread: threading.Thread | None = None
        self._lifecycle_started = False
        atexit.register(self.close_all)

    # -- lifecycle owner -----------------------------------------------------

    def _ensure_lifecycle_loop(self) -> asyncio.AbstractEventLoop:
        if self._lifecycle_loop is not None and not self._lifecycle_loop.is_closed():
            return self._lifecycle_loop
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=loop.run_forever,
            name=f"mcp-lifecycle-{id(self)}",
            daemon=True,
        )
        thread.start()
        self._lifecycle_loop = loop
        self._lifecycle_thread = thread
        self._lifecycle_started = True
        for handle in self.handles.values():
            handle._lifecycle_loop = loop
        return loop

    def _run_sync(self, coro: Any, timeout: float = 120.0) -> Any:
        """Run ``coro`` on the single lifecycle loop from a sync context."""
        loop = self._ensure_lifecycle_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=timeout)

    def close_all(self, timeout: float = 10.0) -> None:
        """Close every handle on its own loop; stop the lifecycle loop.

        Idempotent and safe to call from atexit: handles whose transport was
        opened on the lifecycle loop are closed there (same task/scope), and
        handles owned by an ambient loop are dispatched back to that loop
        when it is still alive.
        """
        handles = list(self.handles.values())
        for handle in handles:
            owner = handle._owner_loop
            if owner is None:
                continue
            if owner.is_closed():
                handle._reset_stale_locked()
                continue
            try:
                future = asyncio.run_coroutine_threadsafe(handle.close(), owner)
                future.result(timeout=timeout)
            except Exception:
                # Best-effort shutdown: never mask the caller's real error,
                # and never let a teardown hiccup crash interpreter exit.
                handle._reset_stale_locked()
        loop = self._lifecycle_loop
        if loop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(loop.stop)
                if self._lifecycle_thread is not None:
                    self._lifecycle_thread.join(timeout=timeout)
            finally:
                try:
                    loop.close()
                except Exception:
                    pass
                self._lifecycle_loop = None
                self._lifecycle_thread = None

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

    def register_tools(
        self, builtin_tool_names: set[str] | None = None
    ) -> list[Tool]:
        """Return status stubs plus (when reachable) remote tool adapters.

        Never raises: unreadable servers degrade to a status stub whose
        invocation reports the typed failure instead of crashing the agent.
        ``builtin_tool_names`` lets the gateway skip MCP adapters that merely
        duplicate tools the local packs already register (the SCNet MCP
        ``vasp_molecule_*`` / ``vasp_study_*`` families duplicate the built-in
        ``vasp_molecule.*`` / ``vasp_study.*`` packs); set
        ``PHOTOMATAGENT_MCP_INCLUDE_DUPLICATE_MOLECULAR=1`` to re-enable the
        MCP copies.
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
                skipped = 0
                for spec in handle.remote_tools:
                    if self._is_duplicate_molecular_adapter(
                        handle, spec, builtin_tool_names
                    ):
                        skipped += 1
                        continue
                    tools.append(MCPRemoteTool(handle, spec))
                if skipped:
                    handle.detail = (
                        f"{handle.detail}; skipped {skipped} duplicate "
                        "vasp_molecule_* MCP adapters (built-in pack is "
                        "authoritative; set "
                        "PHOTOMATAGENT_MCP_INCLUDE_DUPLICATE_MOLECULAR=1 "
                        "to re-enable)"
                    ).lstrip("; ")
        return tools

    @staticmethod
    def _is_duplicate_molecular_adapter(
        handle: MCPServerHandle,
        spec: RemoteToolSpec,
        builtin_tool_names: set[str] | None,
    ) -> bool:
        """True when an SCNet MCP tool duplicates a built-in local pack tool."""
        if not builtin_tool_names:
            return False
        if handle.config.effective_namespace != "scnet_science":
            return False
        prefixes = ("vasp_molecule_", "vasp_study_")
        prefix = next(
            (
                candidate
                for candidate in prefixes
                if spec.name.startswith(candidate)
            ),
            None,
        )
        if prefix is None:
            return False
        override = (
            os.environ.get(
                "PHOTOMATAGENT_MCP_INCLUDE_DUPLICATE_MOLECULAR", ""
            )
            .strip()
            .lower()
        )
        if override in {"1", "true", "yes", "on"}:
            return False
        local_name = prefix.rstrip("_") + "." + spec.name[len(prefix):]
        return local_name in builtin_tool_names

    def _start_handle_sync(self, handle: MCPServerHandle) -> None:
        try:
            asyncio.get_running_loop()
            return  # inside an event loop: lazy reconnect happens on invoke
        except RuntimeError:
            pass
        try:
            self._run_sync(handle.start())
        except Exception as exc:
            handle.state = MCPServerState.START_FAILED
            handle.last_error = str(exc)[:500]

    def _run_async(self, coro: Any) -> None:
        """Best-effort async execution from a sync context (single loop)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                self._run_sync(coro)
            except Exception:
                pass
            return
        # Inside an ambient loop: still route through the lifecycle owner so
        # sessions are never created on a per-call throwaway loop.
        try:
            self._run_sync(coro)
        except Exception:
            pass


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
