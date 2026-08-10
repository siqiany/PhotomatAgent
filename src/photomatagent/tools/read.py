"""Read bounded UTF-8 text from the workspace."""

from __future__ import annotations

from typing import Any

from photomatagent.errors import ToolError
from photomatagent.tools.base import Tool, ToolResult
from photomatagent.tools.exposure import ToolExposure
from photomatagent.runtime.sensitive import SensitiveAccessError, SensitivePathPolicy
from photomatagent.workspace import Workspace


class ReadTool(Tool):
    name = "read"
    description = "Read a UTF-8 text file inside the workspace, optionally by line range."
    exposure = ToolExposure.DIRECT
    tags = ("file", "inspect", "text")
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": "integer", "minimum": 1},
        },
        "required": ["path"],
    }

    def __init__(
        self,
        workspace: Workspace,
        *,
        max_chars: int = 50_000,
        sensitive_paths: SensitivePathPolicy | None = None,
    ) -> None:
        self.workspace = workspace
        self.max_chars = max_chars
        self.sensitive_paths = sensitive_paths or SensitivePathPolicy()

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            self.sensitive_paths.check_path(str(arguments["path"]))
            path = self.workspace.resolve(str(arguments["path"]))
            if not path.is_file():
                raise ToolError(f"not a file: {arguments['path']}")
            lines = path.read_text(encoding="utf-8").splitlines()
            start = int(arguments.get("start_line", 1))
            end = int(arguments.get("end_line", len(lines)))
            if end < start:
                raise ToolError("end_line must be greater than or equal to start_line")
            selected = lines[start - 1 : end]
            content = "\n".join(f"{number}: {line}" for number, line in enumerate(selected, start))
            original_chars = len(content)
            truncated = len(content) > self.max_chars
            if truncated:
                marker = (
                    f"\n... [truncated: read bounded from {original_chars} chars; "
                    "request start_line/end_line to continue]"
                )
                content = content[: self.max_chars] + marker
            return ToolResult(
                output=content,
                data={
                    "path": self.workspace.relative(path),
                    "start_line": start,
                    "end_line": min(end, len(lines)),
                    "truncated": truncated,
                    "original_chars": original_chars,
                    "delivered_chars": len(content),
                },
            )
        except (OSError, UnicodeError, ToolError, SensitiveAccessError) as exc:
            return ToolResult(output=f"read failed: {exc}", is_error=True)
