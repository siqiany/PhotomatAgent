"""Thin MCP client boundary.

Future design: an MCP server exposes tools; a ``MCPTool`` adapter wraps each
one so it looks exactly like a local ``Tool`` to the Agent Loop. The loop will
never know whether a tool is local, MCP, or a scientific backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from photomatagent.tools.base import Tool, ToolResult


@dataclass(frozen=True)
class MCPServerSpec:
    """Where/how to reach an MCP server (command+args or URL)."""

    name: str
    command: str | None = None
    args: list[str] | None = None
    url: str | None = None
    env: dict[str, str] | None = None


class MCPTool(Tool):
    """Adapter that will wrap an MCP server tool as a local Tool.

    TODO(mcp): implement when the MCP phase starts. The constructor should
    connect to the server, map ``inputSchema`` to ``input_schema``, and
    translate ``tools/call`` responses into ``ToolResult``.
    """

    def __init__(self, server: MCPServerSpec, remote_name: str, description: str) -> None:
        self.server = server
        self.remote_name = remote_name
        self.name = f"mcp.{server.name}.{remote_name}"
        self.description = description
        self.input_schema = {"type": "object", "properties": {}, "required": []}

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        raise NotImplementedError("MCP integration is a future phase; no server connected.")


async def connect(server: MCPServerSpec) -> list[Tool]:
    """Future: connect to an MCP server and return its tools as Tool objects.

    Raises NotImplementedError until the MCP phase. The signature is the
    integration contract the rest of the codebase is built against.
    """
    raise NotImplementedError(
        "MCP connect() is a future phase. The Tool interface is already the "
        "uniform seam local tools and MCP tools will share."
    )
