"""FakeSCNetBackend: in-memory stand-in for offline tests (spec section 65).

Supports the full submission surface with scripted state progression,
downloads, failures, timeouts, OOM and cancellation -- no SSH, no Slurm,
no network. Application adapters are tested against this backend so the
default ``uv run pytest`` never touches SCNet.
"""

from __future__ import annotations

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
)
from photomatagent.scientific.remote.scnet import (
    RemoteSubmissionBlocked,
    validate_remote_path,
)


class FakeSCNetBackend:
    """Deterministic in-memory SCNet backend for tests and demos."""

    name = "fake-scnet"

    def __init__(
        self,
        *,
        policy: ResourcePolicy | None = None,
        scripted_states: list[HPCJobState] | None = None,
        remote_files: dict[str, dict[str, bytes]] | None = None,
        fail_submit_with: str = "",
    ) -> None:
        self.policy = policy or ResourcePolicy(allow_hpc_submit=True)
        self.scripted_states = list(scripted_states or [])
        self.remote_files: dict[str, dict[str, bytes]] = {
            key: dict(value) for key, value in (remote_files or {}).items()
        }
        self.fail_submit_with = fail_submit_with
        self._job_counter = 1000
        self._state_index: dict[str, int] = {}
        self.uploaded: list[str] = []
        self.submitted_scripts: dict[str, str] = {}
        self.cancelled: list[str] = []

    # -- helpers ------------------------------------------------------------

    def _next_job_id(self) -> str:
        self._job_counter += 1
        return str(self._job_counter)

    def _advance_state(self, job_id: str) -> HPCJobState:
        if not self.scripted_states:
            return HPCJobState.COMPLETED
        index = self._state_index.get(job_id, 0)
        state = self.scripted_states[min(index, len(self.scripted_states) - 1)]
        self._state_index[job_id] = index + 1
        return state

    # -- capability surface ------------------------------------------------

    async def check_connection(self) -> dict[str, str]:
        return {
            "connected": "true",
            "host": "fake-login",
            "sbatch": "/usr/bin/sbatch",
            "squeue": "/usr/bin/squeue",
            "scancel": "/usr/bin/scancel",
            "sacct": "/usr/bin/sacct",
            "slurm_ready": "true",
        }

    async def ensure_remote_directory(self, remote_directory: str) -> bool:
        validate_remote_path(remote_directory)
        self.remote_files.setdefault(remote_directory, {})
        return True

    async def upload_files(
        self, local_paths: list[Path], remote_directory: str
    ) -> list[str]:
        validate_remote_path(remote_directory)
        await self.ensure_remote_directory(remote_directory)
        names: list[str] = []
        for path in local_paths:
            local = Path(path).expanduser().resolve()
            if not local.is_file():
                raise FileNotFoundError(f"local upload file missing: {local}")
            self.remote_files[remote_directory][local.name] = local.read_bytes()
            names.append(local.name)
            self.uploaded.append(f"{remote_directory}/{local.name}")
        return names

    async def upload_tree(
        self, local_directory: Path, remote_directory: str
    ) -> list[str]:
        validate_remote_path(remote_directory)
        local = Path(local_directory).expanduser().resolve()
        if not local.is_dir():
            raise FileNotFoundError(f"local upload directory missing: {local}")
        await self.ensure_remote_directory(remote_directory)
        names: list[str] = []
        for path in sorted(local.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(local).as_posix()
            self.remote_files[remote_directory][relative] = path.read_bytes()
            names.append(relative)
            self.uploaded.append(f"{remote_directory}/{relative}")
        return names

    async def available_partitions(self) -> list[str]:
        return ["kshcnormal"]

    async def probe_module(
        self, module_name: str, executable: str
    ) -> dict[str, str]:
        return {
            "configured": "true" if module_name else "false",
            "available": "true" if module_name else "false",
            "executable": f"/fake/bin/{executable}" if module_name else "",
            "error": "" if module_name else "module name is not configured",
            "module_candidates": "hefei-namd/1.0" if not module_name else "",
        }

    async def submit_script(self, spec: RemoteJobSpec) -> RemoteJobRef:
        violations = self.policy.violations(spec.resource)
        if violations:
            raise RemoteSubmissionBlocked("; ".join(violations))
        if self.fail_submit_with:
            raise RuntimeError(self.fail_submit_with)
        await self.ensure_remote_directory(spec.remote_directory)
        job_id = self._next_job_id()
        self.submitted_scripts[job_id] = spec.script_name
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
        if not job_id.isdigit():
            raise ValueError(f"Slurm job_id must be numeric, got {job_id!r}")
        state = self._advance_state(job_id)
        if job_id in self.cancelled:
            return HPCJobState.CANCELLED
        return state

    async def cancel_job(self, job_id: str) -> RemoteExecutionResult:
        if not job_id.isdigit():
            raise ValueError(f"Slurm job_id must be numeric, got {job_id!r}")
        self.cancelled.append(job_id)
        return RemoteExecutionResult(ok=True, returncode=0, command=f"scancel {job_id}")

    async def read_stdout(
        self, remote_directory: str, job_id: str, max_bytes: int = 20000
    ) -> str:
        files = self.remote_files.get(remote_directory, {})
        content = files.get(f"{job_id}.out", b"")
        return content.decode("utf-8", errors="replace")[:max_bytes]

    async def read_stderr(
        self, remote_directory: str, job_id: str, max_bytes: int = 20000
    ) -> str:
        files = self.remote_files.get(remote_directory, {})
        content = files.get(f"{job_id}.err", b"")
        return content.decode("utf-8", errors="replace")[:max_bytes]

    async def download_file(
        self, remote_directory: str, filename: str, local_directory: Path
    ) -> Path | None:
        validate_remote_path(remote_directory)
        files = self.remote_files.get(remote_directory, {})
        if filename not in files:
            return None
        local = Path(local_directory).expanduser().resolve()
        local.mkdir(parents=True, exist_ok=True)
        target = local / filename
        target.write_bytes(files[filename])
        return target

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
        files = self.remote_files.get(remote_directory, {})
        return [
            RemoteArtifactRef(
                name=name,
                remote_path=f"{remote_directory}/{name}",
                size_bytes=len(content),
            )
            for name, content in sorted(files.items())
        ]

    async def doctor(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "server": RemoteServerConfig(host="fake-login").public_dict(),
            "connection": await self.check_connection(),
            "submission_authorized": self.policy.allow_hpc_submit,
            "resource_policy": self.policy.model_dump(),
            "note": "fake backend; no real SCNet contact",
        }

    # -- test helpers --------------------------------------------------------

    def add_remote_file(
        self, remote_directory: str, filename: str, content: bytes | str
    ) -> None:
        self.remote_files.setdefault(remote_directory, {})[filename] = (
            content.encode("utf-8") if isinstance(content, str) else content
        )

    def quoted_script(self, job_id: str) -> str:
        """Return the rendered slurm script text stored at submission time."""
        return self.submitted_scripts.get(job_id, "")
