"""Deterministic Slurm helpers: state mapping, job-id parsing, script rendering.

Everything in this module is pure (no I/O) so offline tests can pin the
contracts: job id parsing, state normalization, and script generation.
"""

from __future__ import annotations

import re
import shlex
from typing import Any

from photomatagent.scientific.remote.models import (
    HPCJobState,
    ResourceRequest,
)

_SBATCH_JOB_ID = re.compile(r"Submitted batch job\s+(\d+)")
_SAFE_JOB_NAME = re.compile(r"[^A-Za-z0-9_-]")
_FORBIDDEN_IN_SCRIPT = ("\r", "\n", ";", "&", "|", "$", "`", "\\")


def parse_sbatch_job_id(output: str) -> str | None:
    """Extract the Slurm job id from ``sbatch`` output; None if absent."""
    match = _SBATCH_JOB_ID.search(output or "")
    return match.group(1) if match else None


def validate_job_id(job_id: str) -> None:
    """Strict validation: Slurm job ids are non-negative integers."""
    if not job_id or not job_id.isdigit():
        raise ValueError(f"Slurm job_id must be numeric, got {job_id!r}")


def slurm_state_to_hpc_state(state: str) -> HPCJobState:
    """Normalize a Slurm ``%T``/sacct state to the unified vocabulary."""
    normalized = (state or "").strip().upper().rstrip("+")
    if normalized in {"PENDING", "PD", "CONFIGURING", "RESIZING"}:
        return HPCJobState.PENDING
    if normalized in {"RUNNING", "R", "COMPLETING"}:
        return HPCJobState.RUNNING
    if normalized in {"COMPLETED", "CD"}:
        return HPCJobState.COMPLETED
    if normalized in {"FAILED", "F", "BOOT_FAIL", "DEADLINE", "PREEMPTED"}:
        return HPCJobState.FAILED
    if normalized in {"CANCELLED", "CA"}:
        return HPCJobState.CANCELLED
    if normalized in {"TIMEOUT", "TO"}:
        return HPCJobState.TIMEOUT
    if normalized in {"OUT_OF_MEMORY", "OOM"}:
        return HPCJobState.OUT_OF_MEMORY
    if normalized in {"NODE_FAIL", "NODE_FAILURE", "NF"}:
        return HPCJobState.NODE_FAIL
    return HPCJobState.UNKNOWN


def sanitize_job_name(job_name: str, max_len: int = 64) -> str:
    cleaned = _SAFE_JOB_NAME.sub("-", job_name or "")[:max_len]
    return cleaned or "job"


def _validate_token(value: str, what: str) -> str:
    """Reject control characters / shell metacharacters in script tokens."""
    if any(character in value for character in _FORBIDDEN_IN_SCRIPT):
        raise ValueError(f"unsafe {what}: {value!r}")
    if value != value.strip():
        raise ValueError(f"{what} must not have leading/trailing whitespace")
    return value


def render_slurm_script(
    *,
    job_name: str,
    resource: ResourceRequest,
    module_load: str = "",
    executable: str,
    executable_args: list[str] | None = None,
    env_vars: dict[str, str] | None = None,
    extra_lines: list[str] | None = None,
    preamble: str = "",
    launcher: str = "srun --mpi=pmi2",
) -> str:
    """Render a deterministic, validated Slurm submission script.

    Every interpolated token is validated: job name sanitized, module and
    executable must be plain tokens (no control characters or shell
    metacharacters), env var values are shell-quoted. Nothing is taken from
    the LLM without passing through these checks.
    """
    safe_name = sanitize_job_name(job_name)
    if module_load:
        _validate_token(module_load, "module name")
    _validate_token(executable, "executable")
    if launcher not in {"", "srun --mpi=pmi2"}:
        raise ValueError(f"unsupported launcher: {launcher!r}")
    args = [_validate_token(str(arg), "executable arg") for arg in (executable_args or [])]
    env = {key: value for key, value in (env_vars or {}).items()}
    for key in env:
        _validate_token(key, "env var name")
    lines = [
        "#!/bin/bash",
        f"#SBATCH -J {safe_name}",
        f"#SBATCH -p {_validate_token(resource.partition, 'partition')}",
        f"#SBATCH -N {resource.nodes}",
        f"#SBATCH --ntasks-per-node={resource.tasks_per_node}",
        f"#SBATCH -t {resource.walltime_minutes}",
        "#SBATCH -o %j.out",
        "#SBATCH -e %j.err",
        "set -euo pipefail",
        "ulimit -s unlimited",
        "module purge",
    ]
    if module_load:
        lines.append(f"module load {module_load}")
    for key, value in env.items():
        lines.append(f"export {key}={shlex.quote(value)}")
    if preamble:
        lines.append(preamble)
    for extra in extra_lines or []:
        lines.append(extra)
    command = shlex.join([executable, *args])
    lines.append(f"{launcher} {command}".strip())
    lines.append("")
    return "\n".join(lines)


def remote_status_command(job_id: str) -> str:
    """Remote shell command string querying squeue then sacct."""
    validate_job_id(job_id)
    return (
        "state=$(squeue -h -j {jid} -o %T); "
        "if [ -z \"$state\" ]; then "
        "sacct -n -X -j {jid} -o State | head -n 1; "
        "else printf '%s' \"$state\"; fi"
    ).format(jid=job_id)


def remote_jobs_by_name_command(job_name: str) -> str:
    """Query every Slurm job matching a name: squeue first, sacct fallback.

    Used by reconciliation: a unique per-request job name plus ``squeue``/
    ``sacct`` decides whether a timed-out sbatch actually created a job.
    Output lines are ``<jobid> <state>`` (both sources use that order).
    """
    safe = sanitize_job_name(job_name, max_len=64)
    return (
        "state=$(squeue -h --name={name} -o \"%i %T\" 2>/dev/null); "
        "if [ -z \"$state\" ]; then "
        "sacct -n -X --name={name} -o JobID,State 2>/dev/null | head -n 50; "
        "else printf '%s\\n' \"$state\"; fi"
    ).format(name=shlex.quote(safe))


def remote_cancel_command(job_id: str) -> str:
    validate_job_id(job_id)
    return f"scancel {job_id}"


def remote_mkdir_command(remote_directory: str) -> str:
    return f"mkdir -p {_remote_path_expression(remote_directory)}"


def remote_ls_command(remote_directory: str) -> str:
    return (
        f"cd {_remote_path_expression(remote_directory)} 2>/dev/null && "
        # Depth 3 so VASP evidence inside MAGUS calculator work dirs
        # (calcFold/VASP/OUTCAR etc.) is reachable; still bounded by the
        # collect-side file/size caps.
        "find . -maxdepth 3 -type f -printf '%p %s\\n' 2>/dev/null | sort"
    )


def remote_submit_command(remote_directory: str, script_name: str) -> str:
    return (
        f"cd {_remote_path_expression(remote_directory)} && "
        f"sbatch {shlex.quote(script_name)}"
    )


def remote_read_command(remote_directory: str, filename: str, max_bytes: int) -> str:
    return (
        f"tail -c {int(max_bytes)} "
        f"{_remote_path_expression(remote_directory + '/' + filename)}"
    )


def _remote_path_expression(path: str) -> str:
    """Quote a validated remote path while preserving home expansion."""
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return '"$HOME"/' + shlex.quote(path[2:])
    return shlex.quote(path)


def remote_artifact_sizes_command(remote_directory: str) -> str:
    return (
        f"cd {shlex.quote(remote_directory)} 2>/dev/null && "
        "for f in *; do [ -f \"$f\" ] && stat -c '%n %s' \"$f\"; done 2>/dev/null"
    )


def remote_copy_artifact_command(
    source_remote_directory: str,
    destination_remote_directory: str,
    filename: str,
) -> str:
    """Copy one validated artifact between remote job directories."""
    return (
        f"cp -f "
        f"{_remote_path_expression(source_remote_directory + '/' + filename)} "
        f"{_remote_path_expression(destination_remote_directory + '/')}"
    )
