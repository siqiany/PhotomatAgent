"""Perform one unambiguous exact-text replacement."""

from __future__ import annotations

from typing import Any

from photomatagent.errors import ToolError
from photomatagent.tools.base import Tool, ToolResult
from photomatagent.workspace import Workspace


class EditTool(Tool):
    name = "edit"
    description = "Replace exactly one occurrence of old_text in an existing workspace file."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_text": {"type": "string"},
            "new_text": {"type": "string"},
            "summary": {"type": "string"},
        },
        "required": ["path", "old_text", "new_text"],
    }

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            path = self.workspace.resolve(str(arguments["path"]))
            if not path.is_file():
                raise ToolError(f"not a file: {arguments['path']}")
            content = path.read_text(encoding="utf-8")
            old_text = str(arguments["old_text"])
            count = content.count(old_text)
            if count == 0:
                raise ToolError("old_text was not found")
            if count > 1:
                raise ToolError(f"old_text is ambiguous: found {count} occurrences")
            new_text = str(arguments["new_text"])
            path.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
            return ToolResult(
                output=f"edited {self.workspace.relative(path)}",
                data={"path": self.workspace.relative(path)},
            )
        except (OSError, UnicodeError, ToolError) as exc:
            return ToolResult(output=f"edit failed: {exc}", is_error=True)
