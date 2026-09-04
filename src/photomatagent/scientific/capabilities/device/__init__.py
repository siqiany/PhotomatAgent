"""Semiconductor device capability pack backed by DEVSIM (namespace ``device``).

DEVSIM is DEFERRED and only invoked through its Python API. Script execution
is restricted: the script must live inside the workspace, imports are limited
to a safe allow-list, and execution requires the devsim dependency. This is
not a general-purpose Python execution tool.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
import multiprocessing
import queue
from pathlib import Path
from typing import Any

from photomatagent.scientific.capabilities.base import (
    CapabilityPack,
    CapabilityStatus,
    ProbeResult,
)
from photomatagent.scientific.capabilities.config import ScientificConfig
from photomatagent.scientific.capabilities.contracts import ScientificToolResult
from photomatagent.tools.base import Tool
from photomatagent.tools.exposure import ToolExposure
from photomatagent.workspace import Workspace

_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "round": round,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def _device_worker(script: str, source: str, results: Any) -> None:
    """Execute one restricted script in a process that can be terminated."""
    captured: list[str] = []
    try:
        import devsim

        def restricted_import(
            name: str,
            globals: object = None,
            locals: object = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> Any:
            if name != "devsim" or level:
                raise ImportError(f"only 'devsim' may be imported, got {name!r}")
            return devsim

        builtins = dict(_SAFE_BUILTINS)
        builtins["__import__"] = restricted_import
        namespace: dict[str, Any] = {
            "__builtins__": builtins,
            "devsim": devsim,
            "print": lambda *args: captured.append(" ".join(str(a) for a in args)),
        }
        exec(compile(source, script, "exec"), namespace)
        results.put({"ok": True, "stdout": captured})
    except BaseException as exc:
        results.put(
            {"ok": False, "error_type": type(exc).__name__, "error": str(exc)[:1000]}
        )


class DeviceProbe(CapabilityPack):
    name = "device"
    description = "Semiconductor device simulation via DEVSIM."

    def probe(self) -> ProbeResult:
        try:
            import devsim  # noqa: F401
        except Exception as exc:
            return ProbeResult(
                status=CapabilityStatus.MISSING_DEPENDENCY,
                detail=(
                    f"devsim not importable: {type(exc).__name__}: {exc} "
                    "(extra: photomatagent[device])"
                ),
            )
        return ProbeResult(
            status=CapabilityStatus.AVAILABLE,
            version=importlib.metadata.version("devsim"),
        )

    def tools(self) -> list[Tool]:
        return [
            DeviceCapabilitiesTool(),
            DeviceRunScriptTool(self._config, self._workspace),
            DeviceInspectResultTool(self._workspace),
        ]

    def __init__(self, config: ScientificConfig, workspace: Workspace) -> None:
        self._config = config
        self._workspace = workspace


class DeviceCapabilitiesTool(Tool):
    name = "device.devsim_capabilities"
    description = (
        "Describe DEVSIM device simulation capabilities (DC/AC device models, "
        "Poisson/drift-diffusion) and required inputs."
    )
    short_description = "DEVSIM device simulation capabilities and prerequisites."
    exposure = ToolExposure.DEFERRED
    namespace = "device"
    source = "devsim"
    tags = ("device", "simulation", "semiconductor device", "devsim", "photodetector")
    input_schema = {"type": "object", "properties": {}}

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        payload = {
            "capabilities": [
                "DC and AC device simulation via the DEVSIM Python API",
                "Poisson + drift-diffusion carrier transport in device geometries",
                "current-voltage characteristics, photocurrent with generation term",
                "dark current / detectivity-oriented device studies",
            ],
            "prerequisites": [
                "device geometry and mesh definition script (Python, devsim API)",
                "material parameters (band gap, mobilities, doping, lifetimes)",
                "boundary conditions and bias/illumination setup",
                "devsim package installed (photomatagent[device])",
            ],
            "note": (
                "device.run_script executes workspace scripts through the DEVSIM "
                "Python API only; it is not a general Python executor."
            ),
        }
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False),
            data=payload,
        )


class DeviceRunScriptTool(Tool):
    name = "device.run_script"
    description = (
        "Run a workspace DEVSIM Python script through the DEVSIM API; the script "
        "must import devsim and live inside the workspace. Returns captured "
        "stdout and any result JSON written by the script."
    )
    short_description = "Run a workspace DEVSIM script (restricted execution)."
    exposure = ToolExposure.DEFERRED
    namespace = "device"
    source = "devsim"
    tags = ("device", "devsim", "script", "simulation")
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Script path relative to workspace."},
            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 120},
        },
        "required": ["path"],
    }

    def __init__(self, config: ScientificConfig, workspace: Workspace) -> None:
        self._config = config
        self._workspace = workspace

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        try:
            import devsim  # noqa: F401
        except Exception:
            return ScientificToolResult(
                output=(
                    "missing prerequisite: the 'devsim' package is not importable in "
                    "this environment (install photomatagent[device]). No device "
                    "simulation was attempted."
                ),
                is_error=True,
                data={"error": "MISSING_DEPENDENCY"},
            )
        script = (self._workspace.root / str(arguments["path"])).resolve()
        if self._workspace.root.resolve() not in script.parents:
            return ScientificToolResult(
                output="device.run_script only accepts scripts inside the workspace",
                is_error=True,
                data={"error": "outside_workspace"},
            )
        if not script.is_file() or script.suffix != ".py":
            return ScientificToolResult(
                output=f"missing prerequisite: script not found: {arguments['path']}",
                is_error=True,
                data={"error": "MISSING_PREREQUISITE"},
            )
        source = script.read_text(encoding="utf-8")
        if "devsim" not in source:
            return ScientificToolResult(
                output=(
                    "device.run_script only runs DEVSIM scripts (script must import "
                    "devsim); arbitrary Python execution is not supported."
                ),
                is_error=True,
                data={"error": "not_devsim_script"},
            )
        timeout = float(arguments.get("timeout_seconds", 60))
        with ThreadPoolExecutor(max_workers=1) as executor:
            execution = executor.submit(
                self._execute_script,
                script,
                source,
                timeout,
            )
            while not execution.done():
                await asyncio.sleep(0.01)
            result = execution.result()
        if result.get("timeout"):
            return ScientificToolResult(
                output=f"device script timed out after {timeout:g}s and was terminated",
                is_error=True,
                data={"error": "timeout"},
            )
        if not result.get("ok"):
            return ScientificToolResult(
                output=(
                    f"device script failed: {result.get('error_type', 'Error')}: "
                    f"{result.get('error', '')}"
                ),
                is_error=True,
                data={"error": result.get("error_type", "Error")},
            )
        payload = {
            "script": str(script),
            "stdout": "\n".join(result.get("stdout", []))[-8000:],
        }
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False),
            data=payload,
        )

    def _execute_script(self, script: Path, source: str, timeout: float) -> dict[str, Any]:
        context = multiprocessing.get_context("spawn")
        results = context.Queue(maxsize=1)
        process = context.Process(
            target=_device_worker,
            args=(str(script), source, results),
            daemon=True,
        )
        process.start()
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join(2)
            if process.is_alive():
                process.kill()
                process.join()
            return {"ok": False, "timeout": True}
        try:
            return dict(results.get(timeout=1))
        except queue.Empty:
            return {
                "ok": False,
                "error_type": "WorkerExit",
                "error": f"worker exited with code {process.exitcode}",
            }


class DeviceInspectResultTool(Tool):
    name = "device.inspect_result"
    description = (
        "Summarize a DEVSIM result file (JSON or .npz) written by device.run_script."
    )
    short_description = "Inspect a DEVSIM result file (JSON/npz)."
    exposure = ToolExposure.DEFERRED
    namespace = "device"
    source = "builtin"
    tags = ("device", "devsim", "results", "inspect")
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "max_keys": {"type": "integer", "minimum": 1, "maximum": 50},
        },
        "required": ["path"],
    }

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        path = Path(str(arguments["path"]))
        if not path.is_absolute():
            candidate = self._workspace.root / path
            if candidate.is_file():
                path = candidate
        if not path.is_file():
            return ScientificToolResult(
                output=f"result file not found: {arguments['path']}",
                is_error=True,
                data={"error": "not_found"},
            )
        max_keys = int(arguments.get("max_keys", 20))
        try:
            if path.suffix == ".npz":
                import numpy as np

                with np.load(str(path), allow_pickle=False) as data:
                    summary = {
                        key: {
                            "shape": list(value.shape),
                            "dtype": str(value.dtype),
                            "min": round(float(value.min()), 6) if value.size else None,
                            "max": round(float(value.max()), 6) if value.size else None,
                        }
                        for key, value in list(data.items())[:max_keys]
                    }
                payload = {"file": path.name, "format": "npz", "keys": summary}
            else:
                import json as jsonlib

                raw = jsonlib.loads(path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    raw = {"value": raw}
                keys = list(raw)[:max_keys]
                payload = {"file": path.name, "format": "json", "keys": keys}
        except Exception as exc:
            return ScientificToolResult(
                output=f"device.inspect_result failed: {type(exc).__name__}: {exc}",
                is_error=True,
                data={"error": type(exc).__name__},
            )
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False),
            data=payload,
        )


def device_pack(config: ScientificConfig, workspace: Workspace) -> CapabilityPack:
    return DeviceProbe(config, workspace)
