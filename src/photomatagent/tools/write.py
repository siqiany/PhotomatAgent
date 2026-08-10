"""Create a new UTF-8 file inside the workspace."""

from __future__ import annotations

from typing import Any

from photomatagent.errors import ToolError
from photomatagent.tools.base import Tool, ToolResult
from photomatagent.tools.exposure import ToolExposure
from photomatagent.workspace import Workspace


class WriteTool(Tool):
    name = "write"
    description = "Create a new UTF-8 file in the workspace. Refuses to overwrite existing files."
    exposure = ToolExposure.DIRECT
    tags = ("file", "write", "workspace")
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "summary": {"type": "string"},
        },
        "required": ["path", "content"],
    }

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            path = self.workspace.resolve(str(arguments["path"]), must_exist=False)
            if path.exists():
                raise ToolError(f"refusing to overwrite existing path: {arguments['path']}")
            path.parent.mkdir(parents=True, exist_ok=True)
            content = str(arguments["content"])
            path.write_text(content, encoding="utf-8")
            return ToolResult(
                output=f"created {self.workspace.relative(path)} ({len(content)} characters)",
                data={"path": self.workspace.relative(path), "characters": len(content)},
            )
        except (OSError, ToolError) as exc:
            return ToolResult(output=f"write failed: {exc}", is_error=True)
