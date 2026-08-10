"""kdotpy k.p external-solver adapter (namespace ``kp``).

Execution mode: ``subprocess`` against an isolated kdotpy venv (never the
main PhotoMatAgent environment). The adapter does NOT implement k.p
algorithms -- it runs the external solver with caller-supplied, valid
kdotpy arguments/config files and summarizes the output. Supported
geometries are whatever the installed kdotpy supports (bulk, 1d strip,
2d slab); 0D quantum-dot confinement is NOT supported and is explicitly
guarded.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from photomatagent.scientific.capabilities.base import (
    CapabilityPack,
    CapabilityStatus,
    ProbeResult,
)
from photomatagent.scientific.capabilities.contracts import (
    ScientificEvidence,
    ScientificToolResult,
)
from photomatagent.scientific.capabilities.quantum_dot.provider import (
    kdotpy_capability_note,
    probe_kdotpy,
)
from photomatagent.tools.base import Tool
from photomatagent.tools.exposure import ToolExposure
from photomatagent.workspace import Workspace

MAX_OUTPUT_CHARS = 12000


class KdotpyProbe(CapabilityPack):
    name = "kp"
    description = "kdotpy k.p external solver adapter (subprocess)."

    def __init__(self, workspace: Workspace | None = None) -> None:
        self._workspace = workspace

    def probe(self) -> ProbeResult:
        return probe_kdotpy()

    def tools(self) -> list[Tool]:
        return [
            KdotpyCapabilitiesTool(self._workspace),
            KdotpyRunTool(self._workspace),
        ]


def _find_kdotpy_python(workspace: Workspace | None) -> str | None:
    """Return the interpreter to use for kdotpy (venv first, then main env)."""
    env_python = os.environ.get("PHOTOMATAGENT_KDOTPY_PYTHON", "")
    if env_python and Path(env_python).is_file():
        return env_python
    if workspace is not None:
        candidate = workspace.root / ".venvs" / "kdotpy" / "bin" / "python"
        if candidate.is_file():
            return str(candidate)
    return None


class KdotpyRunTool(Tool):
    name = "kp.run_kdotpy"
    description = (
        "Run the external kdotpy k.p solver in its isolated venv with "
        "caller-supplied arguments or a config file path, and return a "
        "bounded summary (exit code, stdout/stderr tail, produced data "
        "files). Execution mode: subprocess. SUPPORTED: whatever the "
        "installed kdotpy supports (bulk 3D dispersions, 1d strip, 2d "
        "slab) given valid kdotpy arguments. NOT SUPPORTED: 0D quantum-dot "
        "confinement, colloidal QD levels, absorption spectra of dots. "
        "Arguments must be valid kdotpy syntax (e.g. ['6o', 'mater', '1', "
        "'InAs', ...]); the adapter does not invent material parameters."
    )
    short_description = "Run kdotpy k.p solver (subprocess, isolated venv)."
    exposure = ToolExposure.DEFERRED
    namespace = "kp"
    source = "kdotpy (external subprocess)"
    tags = ("k dot p", "kdotpy", "band structure", "subprocess", "external solver")
    input_schema = {
        "type": "object",
        "properties": {
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "kdotpy command-line arguments, e.g. ['bulk', '6o', ...].",
            },
            "config_path": {
                "type": "string",
                "description": "Optional kdotpy XML/config file to run.",
            },
            "timeout_seconds": {"type": "integer", "minimum": 5, "maximum": 600},
            "workdir": {"type": "string", "description": "Optional output dir."},
        },
        "anyOf": [
            {"required": ["args"]},
            {"required": ["config_path"]},
        ],
    }

    def __init__(self, workspace: Workspace | None = None) -> None:
        self._workspace = workspace

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        probe = probe_kdotpy(
            self._workspace.root if self._workspace is not None else None
        )
        if probe.status is not CapabilityStatus.AVAILABLE:
            return ScientificToolResult(
                output=json.dumps(
                    {
                        "error_type": "external_solver_unavailable",
                        "message": (
                            "kdotpy is not available; cannot run k.p calculation"
                        ),
                        "detail": probe.detail,
                    },
                    ensure_ascii=False,
                ),
                is_error=True,
                data={"error_type": "external_solver_unavailable"},
            )
        python = _find_kdotpy_python(self._workspace)
        if python is None:
            return ScientificToolResult(
                output=json.dumps(
                    {
                        "error_type": "external_solver_unavailable",
                        "message": "kdotpy interpreter not located",
                    },
                    ensure_ascii=False,
                ),
                is_error=True,
                data={"error_type": "external_solver_unavailable"},
            )
        args = [str(item) for item in arguments.get("args", [])] if arguments.get("args") else []
        config_path = arguments.get("config_path")
        if config_path:
            args = [str(config_path)]
        if not args:
            return ScientificToolResult(
                output="kp.run_kdotpy requires 'args' or 'config_path'",
                is_error=True,
                data={"error_type": "invalid_input"},
            )
        timeout = min(int(arguments.get("timeout_seconds", 180)), 600)
        workdir = arguments.get("workdir")
        if workdir:
            workdir_path = Path(workdir).expanduser()
            if not workdir_path.is_dir():
                workdir_path.mkdir(parents=True, exist_ok=True)
        else:
            workdir_path = Path(tempfile.mkdtemp(prefix="kdotpy_out_"))
        # kdotpy writes ~/.kdotpy; keep it writable.
        env = {**os.environ, "HOME": str(Path(tempfile.mkdtemp(prefix="kdotpy_home_"))), "MPLCONFIGDIR": str(Path(tempfile.mkdtemp(prefix="mpl_")))}
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                [python, "-m", "kdotpy", *args],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(workdir_path),
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return ScientificToolResult(
                output=json.dumps(
                    {
                        "error_type": "external_solver_timeout",
                        "message": f"kdotpy exceeded timeout {timeout}s",
                        "stdout_tail": (exc.stdout or "")[-2000:],
                    },
                    ensure_ascii=False,
                ),
                is_error=True,
                data={"error_type": "external_solver_timeout"},
            )
        except OSError as exc:
            return ScientificToolResult(
                output=f"failed to launch kdotpy: {exc}",
                is_error=True,
                data={"error_type": "external_solver_unavailable"},
            )
        latency = round((time.perf_counter() - started) * 1000.0, 1)
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        outputs = sorted(
            str(path.relative_to(workdir_path))
            for path in workdir_path.iterdir()
            if path.is_file()
        )[:20]
        payload = {
            "exit_code": completed.returncode,
            "ok": completed.returncode == 0,
            "latency_ms": latency,
            "interpreter": python,
            "workdir": str(workdir_path),
            "stdout_tail": stdout[-MAX_OUTPUT_CHARS:],
            "stderr_tail": stderr[-4000:],
            "output_files": outputs,
            "scope": kdotpy_capability_note(),
        }
        evidence = []
        if completed.returncode == 0:
            evidence.append(
                ScientificEvidence(
                    subject="kdotpy_run",
                    property="exit_code",
                    value=completed.returncode,
                    unit="",
                    source=f"kdotpy subprocess ({python})",
                    source_type="kp_calculation",
                    method="external kdotpy k.p solver",
                    fidelity="kp",
                    summary=f"kdotpy finished in {latency:.0f} ms with exit 0",
                    limitations=kdotpy_capability_note(),
                    provenance={"args": args, "workdir": str(workdir_path)},
                )
            )
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False, indent=2)[:20000],
            is_error=completed.returncode != 0,
            data=payload,
            evidence=evidence,
        )


class KdotpyCapabilitiesTool(Tool):
    name = "kp.capabilities"
    description = (
        "Report the kdotpy k.p solver capability status: reachability "
        "(isolated venv or main env), supported geometries (bulk, 1d/2d "
        "heterostructures) and explicit unsupported use cases (0D "
        "quantum-dot confinement). Call before planning any k.p work."
    )
    short_description = "kdotpy k.p solver capability status and scope."
    exposure = ToolExposure.DEFERRED
    namespace = "kp"
    source = "capability probe"
    tags = ("k dot p", "kdotpy", "band structure", "capability")
    input_schema = {"type": "object", "properties": {}}

    def __init__(self, workspace: Workspace | None = None) -> None:
        self._workspace = workspace

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        probe = probe_kdotpy(self._workspace.root if self._workspace else None)
        payload = {
            "kdotpy_available": probe.status is CapabilityStatus.AVAILABLE,
            "status": probe.status.value,
            "detail": probe.detail,
            "version": probe.version,
            "interpreter": _find_kdotpy_python(self._workspace),
            "scope": kdotpy_capability_note(),
            "warning": (
                "kdotpy is not a 0D quantum-dot confinement solver; do not use it for "
                "colloidal QD diameter-to-level calculations"
            ),
        }
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False, indent=2),
            data=payload,
        )


def kp_pack(workspace: Workspace | None = None) -> CapabilityPack:
    return KdotpyProbe(workspace)
