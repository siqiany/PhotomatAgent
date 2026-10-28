"""Task-local development backend for hashed Python wheels and argv execution.

Processes run on the host. A venv controls dependencies, but is not an OS sandbox.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import sys
import sysconfig
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from photomatagent.capability_acquisition.discovery import bounded_https_download
from photomatagent.capability_acquisition.container_backend import (
    _collect_artifacts, _run_bounded_process, _safe_workspace_file, _sha256_file,
    _tree_sha256, _path_has_symlink,
)
from photomatagent.capability_acquisition.models import (
    EnvironmentRecord, ExecutionRecord, ExecutionRequest, ExecutionState, InstallPlan,
)


def local_python_digest() -> str:
    identity = f"{Path(sys.executable).resolve()}\0{sys.version_info[:3]}\0{sysconfig.get_platform()}"
    return "sha256:" + hashlib.sha256(identity.encode()).hexdigest()


class LocalDevBackend:
    """Install one plan in a task-owned venv and run bounded host processes."""

    def __init__(self, storage_root: Path | str, *, uv_executable: str = "uv") -> None:
        self.storage_root = Path(os.path.abspath(Path(storage_root).expanduser()))
        self.uv_executable = shutil.which(uv_executable)
        self._environments: dict[str, EnvironmentRecord] = {}
        self._records: dict[str, ExecutionRecord] = {}

    def restore_environment(self, environment_id: str) -> EnvironmentRecord:
        if not re.fullmatch(r"env_[0-9a-f]{24}", environment_id):
            raise ValueError("ENVIRONMENT_MANIFEST_INVALID")
        root = self.storage_root / environment_id
        manifest = root / "installed-manifest.json"
        venv = root / "python"
        if _path_has_symlink(root) or _path_has_symlink(manifest) or not venv.is_dir():
            raise ValueError("ENVIRONMENT_MANIFEST_INVALID")
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            plan_sha = data["plan_sha256"]
            digest = data["environment_sha256"]
            if (environment_id != f"env_{plan_sha[:24]}" or data["python_digest"] != local_python_digest()
                    or digest != _tree_sha256(venv) or data["backend"] != "local_dev"
                    or not isinstance(data["entrypoints"], dict)):
                raise ValueError("ENVIRONMENT_MANIFEST_INVALID")
            record = EnvironmentRecord(
                environment_id=environment_id, plan_sha256=plan_sha, status="READY",
                lock_ref=str((root / "requirements.txt").relative_to(self.storage_root)),
                installed_manifest_ref=str(manifest.relative_to(self.storage_root)),
                entrypoints=data["entrypoints"], backend="local_dev",
                created_at=datetime.fromisoformat(data["created_at"]),
                environment_sha256=digest,
            )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("ENVIRONMENT_MANIFEST_INVALID") from exc
        self._environments[environment_id] = record
        return record

    async def install(self, plan: InstallPlan) -> EnvironmentRecord:
        if plan.execution_backend != "local_dev" or plan.image_digest != local_python_digest():
            return self._failed(plan, "BACKEND_MISMATCH")
        if self.uv_executable is None:
            return self._failed(plan, "INSTALLER_UNAVAILABLE")
        if plan.estimated_bytes > 1_000_000_000 or _path_has_symlink(self.storage_root):
            return self._failed(plan, "INSTALL_SIZE_LIMIT_EXCEEDED")
        self.storage_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        environment_id = f"env_{plan.plan_sha256[:24]}"
        root = self.storage_root / environment_id
        if root.exists():
            try:
                restored = self.restore_environment(environment_id)
                if restored.plan_sha256 == plan.plan_sha256:
                    return restored
            except ValueError:
                pass
            return self._failed(plan, "ENVIRONMENT_MANIFEST_INVALID")
        staging = self.storage_root / f".{environment_id}.{uuid4().hex[:12]}.preparing"
        wheels_dir = staging / "wheels"
        venv = root / "python"
        lock = root / "requirements.txt"
        try:
            root.mkdir(mode=0o700)
        except FileExistsError:
            return self._failed(plan, "INSTALL_IN_PROGRESS")
        try:
            wheels_dir.mkdir(parents=True, mode=0o700)
            lines: list[str] = []
            for wheel in plan.wheels:
                await bounded_https_download(
                    wheel.url, wheels_dir / wheel.filename, expected_size=wheel.size,
                    expected_sha256=wheel.sha256, max_bytes=1_000_000_000,
                )
                lines.extend((f"{wheel.distribution}=={wheel.version} \\", f"    --hash=sha256:{wheel.sha256}"))
            lock.write_text("\n".join(lines) + "\n", encoding="utf-8")
            created = await _run_bounded_process(
                [sys.executable, "-m", "venv", "--copies", str(venv)],
                timeout_seconds=120, output_limit=64_000,
            )
            if created[0] != 0 or created[3] or created[4]:
                raise RuntimeError("ENVIRONMENT_CREATE_FAILED")
            installed = await _run_bounded_process(
                [self.uv_executable, "pip", "install", "--python", str(venv / "bin" / "python"),
                 "--no-index", "--find-links", str(wheels_dir), "--require-hashes", "-r", str(lock)],
                timeout_seconds=300, output_limit=64_000, cwd=staging,
                env=self._env(staging, venv),
            )
            if installed[0] != 0 or installed[3] or installed[4]:
                raise RuntimeError("OFFLINE_INSTALL_FAILED")
            probe = await _run_bounded_process(
                [str(venv / "bin" / "python"), "-c",
                 "import importlib.metadata as m,json,re,sys; allowed=set(json.loads(sys.argv[1])); print(json.dumps({e.name:e.value for d in m.distributions() if re.sub(r'[-_.]+','-',d.metadata.get('Name','')).lower() in allowed for e in d.entry_points if e.group == 'console_scripts'}))",
                 json.dumps([re.sub(r"[-_.]+", "-", wheel.distribution).lower() for wheel in plan.wheels])],
                timeout_seconds=30, output_limit=64_000, cwd=staging,
                env=self._env(staging, venv),
            )
            if probe[0] != 0 or probe[3] or probe[4]:
                raise RuntimeError("ENVIRONMENT_PROBE_FAILED")
            entrypoints = json.loads(probe[1].decode("utf-8"))
            for wheel_file in wheels_dir.iterdir():
                wheel_file.unlink()
            wheels_dir.rmdir()
            digest = _tree_sha256(venv)
            created_at = datetime.now(timezone.utc)
            manifest = root / "installed-manifest.json"
            manifest.write_text(json.dumps({
                "backend": "local_dev", "plan_sha256": plan.plan_sha256,
                "python_digest": local_python_digest(), "environment_sha256": digest,
                "entrypoints": entrypoints, "created_at": created_at.isoformat(),
            }, sort_keys=True), encoding="utf-8")
            shutil.rmtree(staging, ignore_errors=True)
            return self.restore_environment(environment_id)
        except asyncio.CancelledError:
            shutil.rmtree(staging, ignore_errors=True)
            shutil.rmtree(root, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(staging, ignore_errors=True)
            shutil.rmtree(root, ignore_errors=True)
            reason = str(exc) if str(exc).isupper() and " " not in str(exc) else type(exc).__name__
            return self._failed(plan, reason[:100])

    async def run(self, request: ExecutionRequest, *, output_directory: Path) -> ExecutionRecord:
        started = datetime.now(timezone.utc)
        environment = self._environments.get(request.environment_id)
        if environment is None:
            return self._failed_execution(request, started, "ENVIRONMENT_NOT_READY")
        root = self.storage_root / request.environment_id
        venv = root / "python"
        try:
            if _tree_sha256(venv) != environment.environment_sha256:
                raise ValueError("ENVIRONMENT_CHANGED")
            stage = output_directory.parent / f".{request.execution_id}-inputs"
            stage.mkdir(mode=0o700)
            try:
                if request.script_path.startswith("entrypoint:"):
                    name = request.script_path.removeprefix("entrypoint:")
                    value = environment.entrypoints.get(name)
                    digest = hashlib.sha256(
                        f"entrypoint:{name}:{value}:{environment.environment_sha256}".encode()
                    ).hexdigest()
                    if value is None or digest != request.script_digest:
                        raise ValueError("ENTRYPOINT_UNAVAILABLE")
                    command = [str(venv / "bin" / "python"), "-c",
                               "import importlib.metadata as m,sys; name=sys.argv[1]; sys.argv=sys.argv[1:]; sys.exit(m.entry_points(group='console_scripts')[name].load()())",
                               name, *request.argv]
                else:
                    script = _safe_workspace_file(Path(request.workspace_root).resolve(), request.script_path)
                    if script is None or _sha256_file(script) != request.script_digest:
                        raise ValueError("GENOME_MISMATCH")
                    staged_script = stage / "main.py"
                    shutil.copyfile(script, staged_script)
                    if _sha256_file(staged_script) != request.script_digest:
                        raise ValueError("GENOME_MISMATCH")
                    command = [str(venv / "bin" / "python"), str(staged_script), *request.argv]
                inputs = stage / "inputs"
                inputs.mkdir()
                for key, relative in request.input_refs.items():
                    source = _safe_workspace_file(Path(request.workspace_root).resolve(), relative)
                    if source is None:
                        raise ValueError("INPUT_BINDING_UNAUTHORIZED")
                    shutil.copyfile(source, inputs / key)
                    if request.input_sha256 and _sha256_file(inputs / key) != request.input_sha256.get(key):
                        raise ValueError("INPUT_BINDING_CHANGED")
                code, stdout, stderr, timed_out, overflow = await _run_bounded_process(
                    command, timeout_seconds=request.timeout_seconds,
                    output_limit=request.output_limit_bytes, cwd=output_directory,
                    env=self._env(output_directory, venv, input_dir=inputs),
                )
                artifacts = () if timed_out or overflow else _collect_artifacts(output_directory, request.artifact_limit_bytes)
                state: ExecutionState = "INTERRUPTED" if timed_out else "FAILED" if overflow or code != 0 else "COMPLETED"
                record = ExecutionRecord(
                    execution_id=request.execution_id, backend="local_dev", state=state,
                    exit_code=code, timed_out=timed_out,
                    stdout=stdout.decode("utf-8", errors="replace"),
                    stderr=stderr.decode("utf-8", errors="replace"), artifacts=artifacts,
                    environment_sha256=environment.environment_sha256,
                    script_sha256=request.script_digest, started_at=started,
                    ended_at=datetime.now(timezone.utc),
                    reason_code="EXECUTION_TIMEOUT" if timed_out else "OUTPUT_LIMIT_EXCEEDED" if overflow else None,
                )
                self._records[request.execution_id] = record
                return record
            finally:
                shutil.rmtree(stage, ignore_errors=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            reason = str(exc) if str(exc).isupper() and " " not in str(exc) else type(exc).__name__
            return self._failed_execution(request, started, reason[:100], environment.environment_sha256)

    async def reconcile(self, execution_id: str) -> ExecutionRecord | None:
        return self._records.get(execution_id)

    @staticmethod
    def _env(cwd: Path, venv: Path, *, input_dir: Path | None = None) -> dict[str, str]:
        environment = {
            "PATH": f"{venv / 'bin'}{os.pathsep}{os.defpath}", "HOME": str(cwd),
            "LANG": "C.UTF-8", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "UV_NO_CONFIG": "1", "UV_CACHE_DIR": str(cwd / ".uv-cache"),
        }
        if input_dir is not None:
            environment["PHOTOMAT_INPUT_DIR"] = str(input_dir)
        return environment

    @staticmethod
    def _failed(plan: InstallPlan, reason: str) -> EnvironmentRecord:
        return EnvironmentRecord(
            environment_id=f"env_failed_{plan.plan_sha256[:16]}", plan_sha256=plan.plan_sha256,
            status="FAILED", lock_ref="", installed_manifest_ref="", entrypoints={},
            backend="local_dev", created_at=datetime.now(timezone.utc),
            environment_sha256="0" * 64, reason_code=reason,
        )

    @staticmethod
    def _failed_execution(request: ExecutionRequest, started: datetime, reason: str,
                          environment_sha: str = "0" * 64) -> ExecutionRecord:
        return ExecutionRecord(
            execution_id=request.execution_id, backend="local_dev", state="FAILED", exit_code=None,
            stdout="", stderr="", artifacts=(), environment_sha256=environment_sha,
            script_sha256=request.script_digest, started_at=started,
            ended_at=datetime.now(timezone.utc), reason_code=reason,
        )
