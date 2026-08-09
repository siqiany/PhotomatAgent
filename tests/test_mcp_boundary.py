from __future__ import annotations

import pytest

from photomatagent.mcp.client import MCPServerSpec, MCPTool, connect


def test_connect_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        import asyncio

        asyncio.run(connect(MCPServerSpec(name="vasp")))


@pytest.mark.asyncio
async def test_mcp_tool_execute_raises_not_implemented():
    tool = MCPTool(MCPServerSpec(name="vasp"), remote_name="run", description="x")
    assert tool.name == "mcp.vasp.run"
    with pytest.raises(NotImplementedError):
        await tool.execute({})
