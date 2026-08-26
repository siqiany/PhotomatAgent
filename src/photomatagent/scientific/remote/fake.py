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

from photomatagent.scientific.remote.scheduler import (
    remote_jobs_by_name_command,
    remote_read_command,
)
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
        submit_succeeds_but_times_out: bool = False,
        strict: bool = False,
    ) -> None:
        self.policy = policy or ResourcePolicy(allow_hpc_submit=True)
        self.scripted_states = list(scripted_states or [])
        self.remote_files: dict[str, dict[str, bytes]] = {
            key: dict(value) for key, value in (remote_files or {}).items()
        }
        self.fail_submit_with = fail_submit_with
        self.strict = strict
        # Models the real-world ambiguity: sbatch accepted the job on the
        # cluster but the client process died/timed out before parsing the id.
        self.submit_succeeds_but_times_out = submit_succeeds_but_times_out
        # Scripted ``_run_ssh`` replies (substring -> (stdout, ok)) for the
        # read-only probe commands used by application adapters (MAGUS
        # environment probe, pseudopotential checks).
        self.ssh_script: dict[str, tuple[str, bool]] = {}
        self._job_counter = 1000
        self._state_index: dict[str, int] = {}
        self.uploaded: list[str] = []
        self.submitted_scripts: dict[str, str] = {}
        self.submitted_job_names: dict[str, str] = {}
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

    async def _run_ssh(
        self, remote_command: str, *, timeout_seconds: float | None = None
    ) -> RemoteExecutionResult:
        """Scripted stand-in for SCNetBackend._run_ssh (offline tests)."""
        # Longest needles first so ``test -d ~/magus/bin`` wins over the
        # prefix ``test -d ~/magus``.
        for needle, (stdout, ok) in sorted(
            self.ssh_script.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if needle in remote_command:
                return RemoteExecutionResult(
                    ok=ok,
                    returncode=0 if ok else 1,
                    stdout=stdout if ok else "",
                    stderr="" if ok else "scripted failure",
                    command=remote_command,
                )
        # Conservative default: unmatched ``test`` checks fail (a missing
        # remote file/dir must never be reported as present).
        if remote_command.lstrip().startswith("test "):
            return RemoteExecutionResult(
                ok=False,
                returncode=1,
                stdout="",
                stderr="scripted test failure",
                command=remote_command,
            )
        return RemoteExecutionResult(
            ok=True, returncode=0, stdout="", command=remote_command
        )

    def add_ssh_script(self, needle: str, stdout: str = "", ok: bool = True) -> None:
        """Script one substring-triggered ``_run_ssh`` reply."""
        self.ssh_script[needle] = (stdout, ok)

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
        if self.strict:
            self._strict_verify_script(spec)
        job_id = self._next_job_id()
        self.submitted_scripts[job_id] = spec.script_name
        self.submitted_job_names[job_id] = spec.job_name
        # The job really exists on the (fake) cluster before the timeout.
        if self.submit_succeeds_but_times_out:
            raise TimeoutError("sbatch client timed out (job was accepted)")
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

    def _strict_verify_script(self, spec: RemoteJobSpec) -> None:
        """Reject the historical false-positive: a submit that would run with
        no ``run.slurm``, no ``vasp_std`` invocation, or no POTCAR strategy.

        Mirrors the real pre-sbatch gates so offline tests fail loudly when
        the caller forgot one of: the rendered ``run.slurm`` upload; the
        actual ``vasp_std`` launcher; or a resolvable POTCAR (local copy or
        remote PSP assembly declared inside the script).
        """
        files = self.remote_files.get(spec.remote_directory, {})
        if spec.script_name not in files:
            raise RuntimeError(
                f"strict backend: {spec.script_name} was not uploaded; "
                "refusing to submit"
            )
        script = files[spec.script_name].decode("utf-8", errors="replace")
        if "vasp_std" not in script:
            raise RuntimeError(
                "strict backend: run.slurm does not invoke vasp_std; "
                "refusing to submit"
            )
        potcar_mode = (spec.provenance or {}).get("potcar_mode", "none")
        symbols = (spec.provenance or {}).get("potcar_symbols") or []
        if potcar_mode == "local" and "POTCAR" not in files:
            raise RuntimeError(
                "strict backend: local POTCAR declared but not uploaded; "
                "refusing to submit"
            )
        if potcar_mode == "remote":
            remote_assembly = (
                "psp_base=" in script and "POTCAR" in script and bool(symbols)
            )
            if not remote_assembly:
                raise RuntimeError(
                    "strict backend: remote POTCAR assembly strategy missing "
                    "from run.slurm; refusing to submit"
                )

    async def jobs_by_name(self, job_name: str) -> list[tuple[str, str]]:
        """Scripted squeue/sacct lookup; falls back to submitted jobs."""
        # Scripted needles drive the squeue-then-sacct precedence from the
        # tests: e.g. only "squeue -h --name" -> running job, or squeue empty
        # + "sacct -n -X --name" -> completed job.
        command = remote_jobs_by_name_command(job_name)
        for needle, (stdout, ok) in sorted(
            self.ssh_script.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if needle not in command:
                continue
            if not needle.startswith(("squeue", "sacct")):
                continue
            if not ok:
                raise RuntimeError("scripted job-name query failure")
            jobs: list[tuple[str, str]] = []
            for line in stdout.splitlines():
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0].isdigit():
                    jobs.append((parts[0], parts[1]))
            return jobs
        return [
            (job_id, "RUNNING")
            for job_id, name in self.submitted_job_names.items()
            if name == job_name
        ]

    async def read_remote_text(
        self, remote_directory: str, filename: str, max_bytes: int
    ) -> str | None:
        """Read a remote text file; None when absent or scripted as missing."""
        validate_remote_path(remote_directory)
        command = remote_read_command(remote_directory, filename, max_bytes)
        for needle, (stdout, ok) in sorted(
            self.ssh_script.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if needle in command:
                if not ok:
                    return None
                return stdout
        files = self.remote_files.get(remote_directory, {})
        content = files.get(filename)
        if content is None:
            return None
        return content.decode("utf-8", errors="replace")[:max_bytes]

    async def copy_remote_artifact(
        self,
        source_remote_directory: str,
        destination_remote_directory: str,
        filename: str,
    ) -> bool:
        """In-memory mirror of the SCNet stream copy (allow-listed names)."""
        validate_remote_path(source_remote_directory)
        validate_remote_path(destination_remote_directory)
        from photomatagent.scientific.remote.scnet import _REMOTE_COPY_ALLOWLIST

        if filename not in _REMOTE_COPY_ALLOWLIST:
            raise ValueError(
                f"remote artifact copy refuses {filename!r}"
            )
        source = self.remote_files.get(source_remote_directory, {}).get(filename)
        if source is None:
            return False
        await self.ensure_remote_directory(destination_remote_directory)
        self.remote_files[destination_remote_directory][filename] = source
        return True

    async def verify_remote_inputs(
        self, remote_directory: str, names: list[str]
    ) -> list[str]:
        """Determine which required inputs are missing on the remote dir."""
        validate_remote_path(remote_directory)
        files = self.remote_files.get(remote_directory, {})
        return [name for name in names if name not in files]

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
        # Mirror real scp semantics: the remote basename lands inside the
        # target directory (relative structure is arranged by the caller).
        target = local / Path(filename).name
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
