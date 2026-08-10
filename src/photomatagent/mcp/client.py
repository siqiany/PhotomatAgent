"""Thin MCP client boundary.

An MCP server exposes tools over JSON-RPC; ``MCPTool`` adapters wrap them so
they look exactly like local ``Tool`` instances to the Agent Loop. The loop
never knows whether a tool is local, MCP, or a scientific backend. MCP is
strictly optional: configuration happens in ``.photomatagent/mcp.json`` and
any connection failure is reported instead of breaking agent startup.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess  # noqa: F401  (documented fallback for stdio spawning)
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from photomatagent.tools.base import Tool, ToolResult
from photomatagent.tools.exposure import ToolExposure


@dataclass(frozen=True)
class MCPServerSpec:
    """Where/how to reach an MCP server (command+args or URL)."""

    name: str
    command: str | None = None
    args: list[str] | None = None
    url: str | None = None
    env: dict[str, str] | None = None


class MCPTool(Tool):
    """Adapter wrapping one MCP remote tool as a deferred local Tool."""

    def __init__(
        self,
        server: MCPServerSpec,
        remote_name: str,
        description: str,
        input_schema: dict[str, Any],
    ) -> None:
        self.server = server
        self.remote_name = remote_name
        self.name = f"materials_mcp.{remote_name}"
        self.description = description
        self.input_schema = input_schema or {"type": "object", "properties": {}}
        self.namespace = "materials_mcp"
        self.exposure = ToolExposure.DEFERRED
        self.source = f"mcp:{server.name}"
        self.tags = ("mcp", "materials", server.name)

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            result = await _call(self.server, self.remote_name, arguments)
        except Exception as exc:
            return ToolResult(
                output=f"materials_mcp.{self.remote_name} failed: {type(exc).__name__}: {exc}",
                is_error=True,
                data={"error": type(exc).__name__},
            )
        return ToolResult(output=result, data={"remote": self.remote_name})


async def connect(server: MCPServerSpec) -> list[Tool]:
    """Connect to one MCP server and return its tools as Tool adapters.

    Supports stdio servers (``command`` + ``args``) and simple HTTP/SSE URLs
    via the ``url`` field. Raises on connection or protocol failure so callers
    can decide whether the failure is fatal (it never is in PhotomatAgent).
    """
    remote_tools = await _request(server, "tools/list", {})
    entries = remote_tools.get("tools", []) if isinstance(remote_tools, dict) else []
    if not isinstance(entries, list):
        raise RuntimeError(f"MCP server {server.name} returned no tools list")
    adapters: list[Tool] = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("name"):
            continue
        schema = entry.get("inputSchema") or entry.get("input_schema") or {}
        adapters.append(
            MCPTool(
                server=server,
                remote_name=str(entry["name"]),
                description=str(entry.get("description", "")),
                input_schema=dict(schema) if isinstance(schema, dict) else {},
            )
        )
    return adapters


async def _call(
    server: MCPServerSpec, tool_name: str, arguments: dict[str, Any]
) -> str:
    response = await _request(
        server, "tools/call", {"name": tool_name, "arguments": arguments}
    )
    if not isinstance(response, dict):
        return json.dumps(response, ensure_ascii=False)
    content = response.get("content", [])
    texts = [
        str(item.get("text", ""))
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    result = "\n".join(texts) or json.dumps(response, ensure_ascii=False)
    return result[:16000]


async def _request(
    server: MCPServerSpec, method: str, params: dict[str, Any]
) -> dict[str, Any]:
    payload = {
        "jsonrpc": "2.0",
        "id": uuid4().hex,
        "method": method,
        "params": params,
    }
    if server.url:
        return await _request_http(server, payload)
    if not server.command:
        raise RuntimeError(f"MCP server {server.name} needs command or url")
    return await _request_stdio(server, payload)


async def _request_stdio(
    server: MCPServerSpec, payload: dict[str, Any]
) -> dict[str, Any]:
    env = dict(os.environ)
    if server.env:
        env.update(server.env)
    process = await asyncio.create_subprocess_exec(
        server.command,
        *(server.args or []),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    assert process.stdin is not None and process.stdout is not None
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    try:
        process.stdin.write(line.encode("utf-8"))
        await asyncio.wait_for(process.stdin.drain(), timeout=5)
        response_line = await asyncio.wait_for(process.stdout.readline(), timeout=15)
    except asyncio.TimeoutError as exc:
        process.kill()
        raise RuntimeError(f"MCP server {server.name} timed out") from exc
    finally:
        process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            process.kill()
    if not response_line:
        stderr = await process.stderr.read() if process.stderr else b""
        raise RuntimeError(
            f"MCP server {server.name} closed without response: "
            f"{stderr.decode(errors='replace')[:500]}"
        )
    try:
        response = json.loads(response_line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"MCP server {server.name} returned invalid JSON: {response_line[:200]!r}"
        ) from exc
    return _check_response(response)


async def _request_http(
    server: MCPServerSpec, payload: dict[str, Any]
) -> dict[str, Any]:
    import urllib.request

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        server.url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310
            raw = response.read().decode("utf-8")
    except Exception as exc:
        raise RuntimeError(f"MCP HTTP request failed: {exc}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"MCP server returned invalid JSON: {raw[:200]!r}") from exc
    return _check_response(parsed)


def _check_response(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise RuntimeError(f"MCP response is not an object: {response!r}")
    if "error" in response and response["error"]:
        raise RuntimeError(f"MCP error: {response['error']}")
    result = response.get("result")
    if result is None:
        raise RuntimeError("MCP response has no result")
    return result if isinstance(result, dict) else {"value": result}
