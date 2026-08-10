"""Execute a shell command in the workspace with bounded output and time."""

from __future__ import annotations

import asyncio
import os
import signal
from typing import Any

from photomatagent.tools.base import Tool, ToolResult
from photomatagent.tools.exposure import ToolExposure
from photomatagent.runtime.sensitive import SensitiveAccessError, SensitivePathPolicy
from photomatagent.workspace import Workspace


class BashTool(Tool):
    name = "bash"
    description = (
        "Run a shell command with cwd fixed to the workspace. This is not an OS sandbox."
    )
    exposure = ToolExposure.DIRECT
    tags = ("shell", "process", "workspace")
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout_seconds": {"type": "number", "minimum": 0.1, "maximum": 120},
        },
        "required": ["command"],
    }

    def __init__(
        self,
        workspace: Workspace,
        *,
        default_timeout: float = 30.0,
        max_output_chars: int = 50_000,
        sensitive_paths: SensitivePathPolicy | None = None,
    ) -> None:
        self.workspace = workspace
        self.default_timeout = default_timeout
        self.max_output_chars = max_output_chars
        self.sensitive_paths = sensitive_paths or SensitivePathPolicy()

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        command = str(arguments["command"])
        try:
            self.sensitive_paths.check_tool_call("bash", arguments)
        except SensitiveAccessError as exc:
            return ToolResult(output=str(exc), is_error=True)
        timeout = min(float(arguments.get("timeout_seconds", self.default_timeout)), 120.0)
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=self.workspace.root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        timed_out = False
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout)
        except TimeoutError:
            timed_out = True
            await self._terminate(process)
            stdout_bytes, stderr_bytes = await process.communicate()
        except asyncio.CancelledError:
            await self._terminate(process)
            raise

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        combined = self._bounded(stdout, stderr)
        exit_code = process.returncode
        if timed_out:
            combined = f"command timed out after {timeout:g}s\n{combined}".rstrip()
        elif exit_code != 0:
            combined = f"command exited with code {exit_code}\n{combined}".rstrip()
        return ToolResult(
            output=combined or "(no output)",
            is_error=timed_out or exit_code != 0,
            data={
                "command": command,
                "exit_code": exit_code,
                "stdout": stdout[: self.max_output_chars],
                "stderr": stderr[: self.max_output_chars],
                "timed_out": timed_out,
            },
        )

    async def _terminate(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            await asyncio.wait_for(process.wait(), 1.0)
        except (ProcessLookupError, TimeoutError):
            if process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()

    def _bounded(self, stdout: str, stderr: str) -> str:
        sections: list[str] = []
        if stdout:
            sections.append(f"stdout:\n{stdout}")
        if stderr:
            sections.append(f"stderr:\n{stderr}")
        output = "\n".join(sections)
        if len(output) <= self.max_output_chars:
            return output
        marker = f"\n... [tool output bounded from {len(output)} chars; middle omitted] ...\n"
        available = max(0, self.max_output_chars - len(marker))
        head = available * 2 // 3
        tail = available - head
        return output[:head] + marker + output[-tail:]
