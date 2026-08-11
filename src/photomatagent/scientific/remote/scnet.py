"""SCNetBackend: generic HPC execution over system OpenSSH + Slurm.

This is internal infrastructure -- the agent never receives a
``run_shell``-style tool. Application adapters (VASP, Hefei-NAMD, MAGUS)
implement narrow, validated tools on top of this backend.

Security contract (Sprint 3 section 12):
* no ``shell=True``; every SSH/SCP invocation uses argv lists
* remote paths are shell-quoted and character-validated
* job ids are strictly validated before any squeue/scancel call
* the private key path never appears in results, errors, or logs
* timeouts on every call; stdout/stderr are bounded
* HPC submission requires ``PHOTOMATAGENT_ALLOW_HPC_SUBMIT=1`` (via
  :class:`ResourcePolicy`) plus a deterministic resource policy check
"""

from __future__ import annotations

import asyncio
import re
import shlex
from pathlib import Path
from typing import Any

from photomatagent.scientific.remote.models import (
    HPCJobState,
    RemoteArtifactRef,
    RemoteExecutionResult,
    RemoteJobRef,
    RemoteJobSpec,
    RemoteServerConfig,
    ResourcePolicy,
    ResourceRequest,
)
from photomatagent.scientific.remote.scheduler import (
    parse_sbatch_job_id,
    remote_cancel_command,
    remote_ls_command,
    remote_mkdir_command,
    remote_read_command,
    remote_status_command,
    remote_submit_command,
    slurm_state_to_hpc_state,
    validate_job_id,
)

_REMOTE_PATH_OK = re.compile(r"^[A-Za-z0-9_./~-]+$")
_MAX_OUTPUT_CHARS = 20000


class RemotePathError(ValueError):
    """A remote path failed character/format validation."""


class RemoteSubmissionBlocked(RuntimeError):
    """Resource policy refused an HPC submission (deterministic)."""


def validate_remote_path(path: str, *, allow_tilde: bool = True) -> str:
    """Validate a remote path: absolute (or ~/), safe characters only."""
    if not path:
        raise RemotePathError("remote path must not be empty")
    if allow_tilde and path == "~":
        return path
    if allow_tilde and path.startswith("~/"):
        tail = path[2:]
    elif path.startswith("/"):
        tail = path[1:]
    else:
        raise RemotePathError(
            f"remote path must be absolute or start with ~/: {path!r}"
        )
    if not tail or not _REMOTE_PATH_OK.match(path):
        raise RemotePathError(f"remote path contains unsafe characters: {path!r}")
    if "//" in path:
        raise RemotePathError(f"remote path must not contain '//': {path!r}")
    return path


class SCNetBackend:
    """Generic SSH + Slurm execution backend for SCNet (system OpenSSH)."""

    name = "scnet"

    def __init__(
        self,
        config: RemoteServerConfig,
        *,
        policy: ResourcePolicy | None = None,
        max_output_chars: int = _MAX_OUTPUT_CHARS,
        ssh_executable: str = "ssh",
        scp_executable: str = "scp",
    ) -> None:
        self.config = config
        self.policy = policy or ResourcePolicy.from_environment()
        self.max_output_chars = max_output_chars
        self.ssh_executable = ssh_executable
        self.scp_executable = scp_executable

    # -- low-level SSH ------------------------------------------------------

    def _ssh_base(self) -> list[str]:
        args = [
            self.ssh_executable,
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={int(self.config.connect_timeout_seconds)}",
            "-p",
            str(self.config.port),
        ]
        if self.config.private_key_path:
            args.extend(["-i", self.config.private_key_path])
        args.append(self.config.destination)
        return args

    def _scp_base(self) -> list[str]:
        args = [
            self.scp_executable,
            "-P",
            str(self.config.port),
            "-o",
            "BatchMode=yes",
        ]
        if self.config.private_key_path:
            args.extend(["-i", self.config.private_key_path])
        return args

    def _redact(self, text: str) -> str:
        if self.config.private_key_path:
            text = text.replace(self.config.private_key_path, "[private-key]")
        return text

    async def _run_ssh(
        self, remote_command: str, *, timeout_seconds: float | None = None
    ) -> RemoteExecutionResult:
        """Run one remote command via ssh argv list (never shell=True)."""
        timeout = timeout_seconds or self.config.connect_timeout_seconds * 5
        command = " ".join(
            [*[shlex.quote(part) for part in self._ssh_base()], shlex.quote(remote_command)]
        )
        try:
            process = await asyncio.create_subprocess_exec(
                *self._ssh_base(),
                remote_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except TimeoutError:
                process.kill()
                await process.wait()
                return RemoteExecutionResult(
                    ok=False,
                    returncode=-1,
                    stdout="",
                    stderr="",
                    command=command,
                    error=f"ssh timed out after {timeout:.0f}s",
                )
            stdout_text = self._redact(stdout.decode("utf-8", errors="replace"))
            stderr_text = self._redact(stderr.decode("utf-8", errors="replace"))
            return RemoteExecutionResult(
                ok=process.returncode == 0,
                returncode=process.returncode or 0,
                stdout=stdout_text[: self.max_output_chars],
                stderr=stderr_text[: self.max_output_chars],
                command=command,
                error="" if process.returncode == 0 else stderr_text[:2000],
            )
        except (FileNotFoundError, PermissionError) as exc:
            return RemoteExecutionResult(
                ok=False,
                returncode=-1,
                stdout="",
                stderr="",
                command=command,
                error=self._redact(f"{type(exc).__name__}: {exc}"),
            )

    # -- capability surface (section 11) -----------------------------------

    async def check_connection(self) -> dict[str, str]:
        """Read-only probe: hostname + availability of sbatch/squeue."""
        result = await self._run_ssh(
            "printf 'host='; hostname; "
            "printf 'sbatch='; command -v sbatch || true; "
            "printf 'squeue='; command -v squeue || true; "
            "printf 'scancel='; command -v scancel || true; "
            "printf 'sacct='; command -v sacct || true",
            timeout_seconds=self.config.connect_timeout_seconds * 2,
        )
        if not result.ok:
            return {
                "connected": "false",
                "error": result.error or "ssh failed",
                "host": "",
                "sbatch": "",
                "squeue": "",
            }
        info: dict[str, str] = {"connected": "true"}
        for line in result.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                info[key] = value
        info["slurm_ready"] = (
            "true"
            if info.get("sbatch") and info.get("squeue") and info.get("sacct")
            else "false"
        )
        return info

    async def ensure_remote_directory(self, remote_directory: str) -> bool:
        validate_remote_path(remote_directory)
        result = await self._run_ssh(remote_mkdir_command(remote_directory))
        if not result.ok:
            raise RuntimeError(f"mkdir on SCNet failed: {result.error}")
        return True

    async def upload_files(
        self, local_paths: list[Path], remote_directory: str
    ) -> list[str]:
        """SCP files into the (existing) remote directory; returns basenames."""
        validate_remote_path(remote_directory)
        resolved = [Path(path).expanduser().resolve() for path in local_paths]
        missing = [str(path) for path in resolved if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"local upload files missing: {missing}")
        await self.ensure_remote_directory(remote_directory)
        destination = f"{self.config.destination}:{remote_directory}/"
        process = await asyncio.create_subprocess_exec(
            *self._scp_base(),
            *[str(path) for path in resolved],
            destination,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=self.config.connect_timeout_seconds * 10
        )
        stderr_text = self._redact(stderr.decode("utf-8", errors="replace"))
        if process.returncode != 0:
            raise RuntimeError(f"scp upload failed: {stderr_text[:2000]}")
        return [path.name for path in resolved]

    async def submit_script(self, spec: RemoteJobSpec) -> RemoteJobRef:
        """Upload a rendered script and submit; refuses without authorization."""
        violations = self.policy.violations(spec.resource)
        if violations:
            raise RemoteSubmissionBlocked("; ".join(violations))
        if not spec.executable:
            raise ValueError("RemoteJobSpec.executable is required")
        validate_remote_path(spec.remote_directory)
        await self.ensure_remote_directory(spec.remote_directory)
        result = await self._run_ssh(remote_submit_command(
            spec.remote_directory, spec.script_name
        ))
        if not result.ok:
            raise RuntimeError(f"sbatch failed: {result.error}")
        job_id = parse_sbatch_job_id(result.stdout)
        if not job_id:
            raise RuntimeError(
                f"could not parse Slurm job id from sbatch output: "
                f"{result.stdout[:500]}"
            )
        return RemoteJobRef(
            backend=self.name,
            application=spec.application,
            job_id=job_id,
            remote_directory=spec.remote_directory,
            state=HPCJobState.SUBMITTED,
            resource_request=spec.resource,
            stdout_ref=f"{spec.remote_directory}/{job_id}.out",
            stderr_ref=f"{spec.remote_directory}/{job_id}.err",
            provenance=spec.provenance,
        )

    async def job_status(self, job_id: str) -> HPCJobState:
        validate_job_id(job_id)
        result = await self._run_ssh(remote_status_command(job_id))
        if not result.ok:
            return HPCJobState.UNKNOWN
        state = result.stdout.strip().split()[0] if result.stdout.strip() else ""
        return slurm_state_to_hpc_state(state)

    async def cancel_job(self, job_id: str) -> RemoteExecutionResult:
        validate_job_id(job_id)
        return await self._run_ssh(remote_cancel_command(job_id))

    async def read_stdout(
        self, remote_directory: str, job_id: str, max_bytes: int = 20000
    ) -> str:
        validate_job_id(job_id)
        validate_remote_path(remote_directory)
        result = await self._run_ssh(
            remote_read_command(remote_directory, f"{job_id}.out", max_bytes)
        )
        return result.stdout if result.ok else ""

    async def read_stderr(
        self, remote_directory: str, job_id: str, max_bytes: int = 20000
    ) -> str:
        validate_job_id(job_id)
        validate_remote_path(remote_directory)
        result = await self._run_ssh(
            remote_read_command(
                remote_directory, f"{job_id}.err", max_bytes
            )
        )
        return result.stdout if result.ok else ""

    async def download_file(
        self, remote_directory: str, filename: str, local_directory: Path
    ) -> Path | None:
        """Download one file; returns the local path or None if absent."""
        validate_remote_path(remote_directory)
        if not _REMOTE_PATH_OK.match(filename):
            raise RemotePathError(f"unsafe remote filename: {filename!r}")
        local_directory = Path(local_directory).expanduser().resolve()
        local_directory.mkdir(parents=True, exist_ok=True)
        source = f"{self.config.destination}:{remote_directory}/{filename}"
        process = await asyncio.create_subprocess_exec(
            *self._scp_base(),
            source,
            str(local_directory),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(
            process.communicate(), timeout=self.config.connect_timeout_seconds * 10
        )
        local_path = local_directory / filename
        if process.returncode != 0 or not local_path.is_file():
            return None
        return local_path

    async def download_files(
        self, remote_directory: str, filenames: list[str], local_directory: Path
    ) -> list[Path]:
        downloaded: list[Path] = []
        for filename in filenames:
            path = await self.download_file(remote_directory, filename, local_directory)
            if path is not None:
                downloaded.append(path)
        return downloaded

    async def list_remote_artifacts(
        self, remote_directory: str
    ) -> list[RemoteArtifactRef]:
        validate_remote_path(remote_directory)
        result = await self._run_ssh(remote_ls_command(remote_directory))
        artifacts: list[RemoteArtifactRef] = []
        if not result.ok:
            return artifacts
        for line in result.stdout.splitlines():
            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                continue
            name, size = parts
            try:
                size_bytes = int(size)
            except ValueError:
                size_bytes = None
            artifacts.append(
                RemoteArtifactRef(
                    name=name,
                    remote_path=f"{remote_directory}/{name}",
                    size_bytes=size_bytes,
                )
            )
        return artifacts

    # -- diagnostics ---------------------------------------------------------

    async def doctor(self) -> dict[str, Any]:
        """Read-only diagnostics for ``scnet doctor`` (no secrets)."""
        connection = await self.check_connection()
        report: dict[str, Any] = {
            "backend": self.name,
            "server": self.config.public_dict(),
            "connection": connection,
            "submission_authorized": self.policy.allow_hpc_submit,
            "resource_policy": self.policy.model_dump(),
        }
        if connection.get("slurm_ready") == "true":
            partition = await self._run_ssh("squeue --version | head -n 1")
            home = await self._run_ssh("printf '%s' \"$HOME\"")
            report["slurm_version"] = partition.stdout.strip()[:500]
            report["remote_home"] = home.stdout.strip()[:500]
        return report
