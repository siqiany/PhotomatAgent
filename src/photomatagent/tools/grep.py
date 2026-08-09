"""Bounded regular-expression search within workspace text files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from photomatagent.errors import ToolError
from photomatagent.tools.base import Tool, ToolResult
from photomatagent.workspace import Workspace


class GrepTool(Tool):
    name = "grep"
    description = "Search text with a regular expression inside the workspace."
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
            "glob": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
        },
        "required": ["pattern"],
    }

    def __init__(
        self,
        workspace: Workspace,
        *,
        default_limit: int = 100,
        max_chars: int = 50_000,
    ) -> None:
        self.workspace = workspace
        self.default_limit = default_limit
        self.max_chars = max_chars

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            regex = re.compile(str(arguments["pattern"]))
            base = self.workspace.resolve(str(arguments.get("path", ".")))
            glob_pattern = arguments.get("glob")
            if glob_pattern is not None:
                self.workspace.validate_glob(str(glob_pattern))
            limit = min(int(arguments.get("limit", self.default_limit)), 1000)
            matches: list[str] = []
            for path in self._files(base, str(glob_pattern) if glob_pattern else None):
                try:
                    lines = path.read_text(encoding="utf-8").splitlines()
                except (UnicodeError, OSError):
                    continue
                for line_number, line in enumerate(lines, 1):
                    if regex.search(line):
                        matches.append(f"{self.workspace.relative(path)}:{line_number}:{line}")
                        if len(matches) >= limit:
                            break
                if len(matches) >= limit:
                    break
            output = "\n".join(matches) or "(no matches)"
            truncated = len(matches) >= limit or len(output) > self.max_chars
            if len(output) > self.max_chars:
                output = output[: self.max_chars] + "\n... [truncated]"
            return ToolResult(output=output, data={"matches": matches, "truncated": truncated})
        except (re.error, OSError, ToolError) as exc:
            return ToolResult(output=f"grep failed: {exc}", is_error=True)

    def _files(self, base: Path, pattern: str | None) -> Iterable[Path]:
        if base.is_file():
            return [base]
        candidates = base.glob(pattern or "**/*")
        return sorted(
            path
            for path in candidates
            if path.is_file() and self.workspace.contains(path.resolve(strict=False))
        )
