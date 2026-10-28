"""Find workspace files by glob pattern."""

from __future__ import annotations

import asyncio
from typing import Any

from photomatagent.errors import ToolError
from photomatagent.tools.base import Tool, ToolResult
from photomatagent.tools.exposure import ToolExposure
from photomatagent.runtime.sensitive import SensitiveAccessError, SensitivePathPolicy
from photomatagent.workspace import Workspace


class GlobTool(Tool):
    name = "glob"
    description = "List workspace files matching a glob such as src/**/*.py. Results are bounded; narrow the pattern when truncated."
    exposure = ToolExposure.DIRECT
    tags = ("file", "search", "workspace")
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
        },
        "required": ["pattern"],
    }

    def __init__(
        self,
        workspace: Workspace,
        *,
        default_limit: int = 200,
        sensitive_paths: SensitivePathPolicy | None = None,
    ) -> None:
        self.workspace = workspace
        self.default_limit = default_limit
        self.sensitive_paths = sensitive_paths or SensitivePathPolicy()

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            pattern = self.workspace.validate_glob(str(arguments["pattern"]))
            self.sensitive_paths.check_path(pattern)
            limit = min(int(arguments.get("limit", self.default_limit)), 1000)
            matches: list[str] = []
            truncated = False
            for scanned, path in enumerate(self.workspace.root.glob(pattern), start=1):
                if scanned % 128 == 0:
                    await asyncio.sleep(0)  # Let cancellation interrupt broad searches.
                if (path.is_file()
                        and self.workspace.contains(path.resolve(strict=False))
                        and not self.sensitive_paths.is_sensitive(self.workspace.relative(path))):
                    matches.append(self.workspace.relative(path))
                if len(matches) > limit or scanned >= 10_000:
                    truncated = True
                    break
            shown = sorted(matches[:limit])
            output = "\n".join(shown) or "(no matches)"
            if truncated:
                output += f"\n... [search stopped; at least {len(matches)} matches; narrow the pattern]"
            return ToolResult(
                output=output,
                data={"matches": shown, "total": len(matches), "total_is_exact": not truncated,
                      "truncated": truncated},
            )
        except (OSError, ToolError, SensitiveAccessError) as exc:
            return ToolResult(output=f"glob failed: {exc}", is_error=True)
