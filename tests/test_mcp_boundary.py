"""MCP client boundary tests against a minimal fake stdio server."""

from __future__ import annotations

import asyncio
import json
import sys

import pytest

from photomatagent.mcp.client import MCPServerSpec, MCPTool, connect


FAKE_SERVER = """
import json
import sys

TOOLS = {
    "tools/list": {
        "tools": [
            {
                "name": "get_summary",
                "description": "Fetch a materials summary.",
                "inputSchema": {"type": "object", "properties": {"material_id": {"type": "string"}}},
            }
        ]
    },
    "tools/call": {
        "content": [{"type": "text", "text": '{"material_id": "mp-1990", "band_gap": 0.0}'}]
    },
}

for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    response = {"jsonrpc": "2.0", "id": request.get("id")}
    if method in TOOLS:
        response["result"] = TOOLS[method]
    else:
        response["error"] = {"code": -32601, "message": f"unknown method {method}"}
    sys.stdout.write(json.dumps(response) + "\\n")
    sys.stdout.flush()
    if request.get("method") == "tools/call":
        break
"""


def _server_spec() -> MCPServerSpec:
    return MCPServerSpec(name="fake-materials", command=sys.executable, args=["-c", FAKE_SERVER])


def test_connect_lists_remote_tools():
    tools = asyncio.run(connect(_server_spec()))
    assert len(tools) == 1
    tool = tools[0]
    assert isinstance(tool, MCPTool)
    assert tool.name == "materials_mcp.get_summary"
    assert tool.namespace == "materials_mcp"
    assert tool.exposure.value == "deferred"
    assert "material_id" in tool.input_schema["properties"]


@pytest.mark.asyncio
async def test_mcp_tool_call_returns_text():
    tool = MCPTool(
        _server_spec(),
        remote_name="get_summary",
        description="Fetch a materials summary.",
        input_schema={"type": "object", "properties": {"material_id": {"type": "string"}}},
    )
    result = await tool.execute({"material_id": "mp-1990"})
    assert not result.is_error
    assert "mp-1990" in result.output


def test_connect_missing_command_raises():
    with pytest.raises(RuntimeError):
        asyncio.run(connect(MCPServerSpec(name="vasp")))


def test_connect_broken_server_raises_instead_of_silently_breaking():
    spec = MCPServerSpec(name="broken", command=sys.executable, args=["-c", "raise SystemExit(3)"])
    with pytest.raises(RuntimeError):
        asyncio.run(connect(spec))

