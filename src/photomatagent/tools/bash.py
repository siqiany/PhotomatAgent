"""Execute a shell command in the workspace with bounded output and time."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
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
        exit_code, stdout_bytes, stderr_bytes, timed_out = await self._run_command(
            command, timeout
        )

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        combined = self._bounded(stdout, stderr)
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

    async def _run_command(
        self, command: str, timeout: float
    ) -> tuple[int | None, bytes, bytes, bool]:
        """Run one command without threads or the asyncio child watcher.

        Exit status is observed by polling ``Popen.poll()`` from the event
        loop, and output is read with non-blocking readers on the pipe fds.
        This is robust in environments whose supervisors reap child processes
        before the asyncio watcher can observe them.
        """
        loop = asyncio.get_running_loop()
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=self.workspace.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        stdout_fd = process.stdout.fileno()
        stderr_fd = process.stderr.fileno()
        os.set_blocking(stdout_fd, False)
        os.set_blocking(stderr_fd, False)
        out_chunks: list[bytes] = []
        err_chunks: list[bytes] = []

        def on_read(fd: int, chunks: list[bytes]) -> None:
            try:
                chunk = os.read(fd, 65536)
            except BlockingIOError:
                return
            if chunk:
                chunks.append(chunk)
            else:
                try:
                    loop.remove_reader(fd)
                except (ValueError, RuntimeError):
                    pass

        loop.add_reader(stdout_fd, on_read, stdout_fd, out_chunks)
        loop.add_reader(stderr_fd, on_read, stderr_fd, err_chunks)
        timed_out = False
        deadline = loop.time() + timeout
        try:
            while True:
                if process.poll() is not None:
                    break
                if loop.time() >= deadline:
                    timed_out = True
                    self._kill_group(process, signal.SIGTERM)
                    grace = loop.time() + 2.0
                    while process.poll() is None and loop.time() < grace:
                        await asyncio.sleep(0.02)
                    if process.poll() is None:
                        self._kill_group(process, signal.SIGKILL)
                    break
                await asyncio.sleep(0.02)
        finally:
            for fd in (stdout_fd, stderr_fd):
                try:
                    loop.remove_reader(fd)
                except (ValueError, RuntimeError):
                    pass
            for fd, chunks in (
                (stdout_fd, out_chunks),
                (stderr_fd, err_chunks),
            ):
                while True:
                    try:
                        chunk = os.read(fd, 65536)
                    except BlockingIOError:
                        break
                    if not chunk:
                        break
                    chunks.append(chunk)
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
        return (
            process.returncode,
            b"".join(out_chunks),
            b"".join(err_chunks),
            timed_out,
        )

    def _kill_group(self, process: subprocess.Popen, sig: int) -> None:
        try:
            os.killpg(process.pid, sig)
        except (ProcessLookupError, PermissionError):
            try:
                process.kill()
            except (ProcessLookupError, PermissionError):
                pass

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
