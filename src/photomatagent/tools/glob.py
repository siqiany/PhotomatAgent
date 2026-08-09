"""Find workspace files by glob pattern."""

from __future__ import annotations

from typing import Any

from photomatagent.errors import ToolError
from photomatagent.tools.base import Tool, ToolResult
from photomatagent.workspace import Workspace


class GlobTool(Tool):
    name = "glob"
    description = "List workspace files matching a glob such as src/**/*.py."
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
        },
        "required": ["pattern"],
    }

    def __init__(self, workspace: Workspace, *, default_limit: int = 200) -> None:
        self.workspace = workspace
        self.default_limit = default_limit

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            pattern = self.workspace.validate_glob(str(arguments["pattern"]))
            limit = min(int(arguments.get("limit", self.default_limit)), 1000)
            matches = sorted(
                self.workspace.relative(path)
                for path in self.workspace.root.glob(pattern)
                if path.is_file()
                and self.workspace.contains(path.resolve(strict=False))
            )
            shown = matches[:limit]
            output = "\n".join(shown) or "(no matches)"
            if len(matches) > limit:
                output += f"\n... [{len(matches) - limit} more files omitted]"
            return ToolResult(
                output=output,
                data={"matches": shown, "total": len(matches), "truncated": len(matches) > limit},
            )
        except (OSError, ToolError) as exc:
            return ToolResult(output=f"glob failed: {exc}", is_error=True)
