"""Bounded regular-expression search within workspace text files."""

from __future__ import annotations

import re
import asyncio
from pathlib import Path
from typing import Any, Iterable

from photomatagent.errors import ToolError
from photomatagent.tools.base import Tool, ToolResult
from photomatagent.tools.exposure import ToolExposure
from photomatagent.runtime.sensitive import SensitiveAccessError, SensitivePathPolicy
from photomatagent.workspace import Workspace


class GrepTool(Tool):
    name = "grep"
    description = "Search text with a regular expression inside the workspace. Search is bounded; narrow the path/glob when truncated."
    exposure = ToolExposure.DIRECT
    tags = ("file", "search", "text")
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
        sensitive_paths: SensitivePathPolicy | None = None,
    ) -> None:
        self.workspace = workspace
        self.default_limit = default_limit
        self.max_chars = max_chars
        self.sensitive_paths = sensitive_paths or SensitivePathPolicy()

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            regex = re.compile(str(arguments["pattern"]))
            self.sensitive_paths.check_path(str(arguments.get("path", ".")))
            base = self.workspace.resolve(str(arguments.get("path", ".")))
            glob_pattern = arguments.get("glob")
            if glob_pattern is not None:
                self.workspace.validate_glob(str(glob_pattern))
            limit = min(int(arguments.get("limit", self.default_limit)), 1000)
            matches: list[str] = []
            scan_limit_reached = False
            for scanned, path in enumerate(self._files(base, str(glob_pattern) if glob_pattern else None), start=1):
                if scanned >= 10_000:
                    scan_limit_reached = True
                    break
                if scanned % 128 == 0:
                    await asyncio.sleep(0)
                if (not path.is_file()
                        or not self.workspace.contains(path.resolve(strict=False))
                        or self.sensitive_paths.is_sensitive(self.workspace.relative(path))):
                    continue
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
            original_chars = len(output)
            truncated = scan_limit_reached or len(matches) >= limit or len(output) > self.max_chars
            if scan_limit_reached:
                output += "\n... [candidate scan limit reached; results incomplete; narrow the path/glob]"
            if len(output) > self.max_chars:
                marker = (
                    f"\n... [grep output bounded from {original_chars} chars; "
                    "use a narrower pattern/path or lower limit]"
                )
                output = output[: max(0, self.max_chars - len(marker))] + marker
            return ToolResult(
                output=output,
                data={
                    "matches": matches,
                    "truncated": truncated,
                    "scan_limit_reached": scan_limit_reached,
                    "original_chars": original_chars,
                    "delivered_chars": len(output),
                },
            )
        except (re.error, OSError, ToolError, SensitiveAccessError) as exc:
            return ToolResult(output=f"grep failed: {exc}", is_error=True)

    def _files(self, base: Path, pattern: str | None) -> Iterable[Path]:
        if base.is_file():
            return [base]
        return base.glob(pattern or "**/*")
