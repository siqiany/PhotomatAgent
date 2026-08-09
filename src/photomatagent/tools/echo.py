"""echo tool: returns its input unchanged."""

from __future__ import annotations

from photomatagent.tools.base import Tool, ToolResult


class EchoTool(Tool):
    name = "echo"
    description = "Echo the given text back. Useful for smoke tests."
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def execute(self, arguments: dict) -> ToolResult:
        return ToolResult(output=arguments["text"], data={"echo": arguments["text"]})
