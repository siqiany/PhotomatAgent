"""Create a new UTF-8 file inside the workspace."""

from __future__ import annotations

from typing import Any

from photomatagent.errors import ToolError
from photomatagent.tools.base import Tool, ToolResult
from photomatagent.tools.exposure import ToolExposure
from photomatagent.workspace import Workspace

# A single ``content`` argument above this size starts consuming most of the
# model's per-response output budget. The default reserve is 8192 tokens and a
# truncated tool call is discarded entirely, so long files must be chunked.
_CHUNK_ADVISORY_CHARS = 3_000


class WriteTool(Tool):
    name = "write"
    description = (
        "Create a new UTF-8 file in the workspace. Refuses to overwrite existing files. "
        "One tool call may return at most the model output budget (8192 tokens by "
        "default), so keep a single `content` under about 3000 characters and build a "
        "longer file from several calls: end each chunk with a unique placeholder line "
        "such as `<!--CONTINUE-->`, then call `edit` repeatedly with `old_text` set to "
        "that placeholder and `new_text` set to the next section followed by the "
        "placeholder again, dropping the placeholder in the final edit. An oversized "
        "single write is cut off mid-argument by the endpoint and the whole call is "
        "discarded."
    )
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
            data: dict[str, Any] = {
                "path": self.workspace.relative(path),
                "characters": len(content),
            }
            output = f"created {self.workspace.relative(path)} ({len(content)} characters)"
            if len(content) > _CHUNK_ADVISORY_CHARS:
                # This write succeeded, but it used most of the model's output
                # budget. Say so now, because the file already exists and a
                # second oversized attempt cannot overwrite it: the next long
                # file should be built in chunks instead.
                data["chunk_advisory"] = True
                output += (
                    f"\nnote: {len(content)} characters is a large single write and a "
                    "longer one would be truncated by the endpoint. For the next large "
                    "file, write a short skeleton first and append sections with `edit` "
                    "using a placeholder marker, keeping each `content` under about 3000 "
                    "characters."
                )
            return ToolResult(output=output, data=data)
        except (OSError, ToolError) as exc:
            return ToolResult(output=f"write failed: {exc}", is_error=True)
