"""Rootless Podman backend for offline package installation and script runs."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import signal
import shutil
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from photomatagent.capability_acquisition.discovery import bounded_https_download
from photomatagent.capability_acquisition.models import (
    ArtifactRecord,
    EnvironmentRecord,
    ExecutionRecord,
    ExecutionRequest,
    ExecutionState,
    InstallPlan,
)
from photomatagent.capability_acquisition.sandbox import SandboxBackend

_IMAGE_NAME = "python:3.12-slim"
_MAX_ENVIRONMENT_BYTES = 4 * 1024 * 1024 * 1024
_MAX_ARTIFACTS = 256


class PodmanSandboxBackend:
    """Podman is used only when already installed and running rootless.

    This backend never pulls images. Missing Podman or the pinned image
    produces a typed failure and cannot fall back to host Python or Bash.
    """

    def __init__(self, storage_root: Path | str, *, executable: str = "podman") -> None:
        self.storage_root = Path(os.path.abspath(Path(storage_root).expanduser()))
        self.executable = shutil.which(executable)
        self._environments: dict[str, Path] = {}
        self._environment_digests: dict[str, str] = {}
        self._environment_plans: dict[str, str] = {}
        self._environment_images: dict[str, str] = {}

    def restore_environment(self, environment_id: str) -> EnvironmentRecord:
        """Restore a persisted READY environment after verifying its immutable manifest."""
        if not re.fullmatch(r"env_[0-9a-f]{24}", environment_id):
            raise ValueError("ENVIRONMENT_MANIFEST_INVALID")
        root = self.storage_root / environment_id
        manifest_path = root / "installed-manifest.json"
        python_root = root / "python"
        lock_path = root / "requirements.txt"
        try:
            if any(_path_has_symlink(path) for path in (root, manifest_path, python_root, lock_path)):
                raise ValueError("ENVIRONMENT_MANIFEST_INVALID")
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            plan_sha = payload["plan_sha256"]
            environment_sha = payload["environment_sha256"]
            image = payload["image_ref"]
            created_at = datetime.fromisoformat(payload["created_at"])
            entrypoints = payload["entrypoints"]
            if (
                not isinstance(plan_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", plan_sha)
                or environment_id != f"env_{plan_sha[:24]}"
                or not isinstance(environment_sha, str)
                or not re.fullmatch(r"[0-9a-f]{64}", environment_sha)
                or environment_sha != _tree_sha256(python_root)
                or not isinstance(image, str)
                or re.fullmatch(rf"{re.escape(_IMAGE_NAME)}@sha256:[0-9a-f]{{64}}", image) is None
                or not isinstance(entrypoints, dict)
                or any(not isinstance(key, str) or not isinstance(value, str)
                       for key, value in entrypoints.items())
                or created_at.tzinfo is None
                or not lock_path.is_file()
                or not python_root.is_dir()
            ):
                raise ValueError("ENVIRONMENT_MANIFEST_INVALID")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("ENVIRONMENT_MANIFEST_INVALID") from exc
        self._environments[environment_id] = python_root
        self._environment_digests[environment_id] = environment_sha
        self._environment_plans[environment_id] = plan_sha
        self._environment_images[environment_id] = image
        return EnvironmentRecord(
            environment_id=environment_id, plan_sha256=plan_sha, status="READY",
            lock_ref=lock_path.relative_to(self.storage_root).as_posix(),
            installed_manifest_ref=manifest_path.relative_to(self.storage_root).as_posix(),
            entrypoints=entrypoints, backend="podman-rootless", created_at=created_at,
            environment_sha256=environment_sha,
        )

    async def install(self, plan: InstallPlan) -> EnvironmentRecord:
        if plan.execution_backend != "podman":
            return _failed_environment(plan, "BACKEND_MISMATCH")
        if self.executable is None:
            return _failed_environment(plan, "SANDBOX_UNAVAILABLE")
        if not await self._ready_image(plan.image_digest):
            return _failed_environment(plan, "SANDBOX_UNAVAILABLE")
        if plan.estimated_bytes > 1_000_000_000:
            return _failed_environment(plan, "INSTALL_SIZE_LIMIT_EXCEEDED")
        if _path_has_symlink(self.storage_root):
            return _failed_environment(plan, "INVALID_STORAGE_PATH")
        self.storage_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.storage_root, 0o700)
        environment_id = f"env_{plan.plan_sha256[:24]}"
        environment_root = self.storage_root / environment_id
        if environment_root.exists():
            if _path_has_symlink(environment_root):
                return _failed_environment(plan, "INVALID_STORAGE_PATH")
            try:
                manifest_path = environment_root / "installed-manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                existing_python = environment_root / "python"
                environment_sha = _tree_sha256(existing_python)
                if manifest.get("plan_sha256") != plan.plan_sha256 or manifest.get("environment_sha256") != environment_sha:
                    return _failed_environment(plan, "INSTALL_PLAN_CHANGED")
                if manifest.get("image_ref") != f"{_IMAGE_NAME}@{plan.image_digest}":
                    return _failed_environment(plan, "INSTALL_PLAN_CHANGED")
                entries = manifest.get("entrypoints")
                if not isinstance(entries, dict):
                    return _failed_environment(plan, "ENVIRONMENT_MANIFEST_INVALID")
                self._environments[environment_id] = existing_python
                self._environment_digests[environment_id] = environment_sha
                self._environment_plans[environment_id] = plan.plan_sha256
                self._environment_images[environment_id] = f"{_IMAGE_NAME}@{plan.image_digest}"
                return EnvironmentRecord(
                    environment_id=environment_id, plan_sha256=plan.plan_sha256,
                    status="READY", lock_ref=str((environment_root / "requirements.txt").relative_to(self.storage_root)),
                    installed_manifest_ref=str(manifest_path.relative_to(self.storage_root)),
                    entrypoints=entries, backend="podman-rootless",
                    created_at=datetime.now(timezone.utc), environment_sha256=environment_sha,
                )
            except (OSError, ValueError, TypeError):
                return _failed_environment(plan, "ENVIRONMENT_MANIFEST_INVALID")
        staging_root = self.storage_root / f".{environment_id}.{uuid4().hex[:12]}.preparing"
        wheels_dir = staging_root / "wheels"
        environment_dir = staging_root / "python"
        lock_path = staging_root / "requirements.txt"
        try:
            wheels_dir.mkdir(parents=True, mode=0o700)
            environment_dir.mkdir(mode=0o700)
            lines = []
            total_bytes = 0
            for wheel in plan.wheels:
                target = wheels_dir / wheel.filename
                await bounded_https_download(
                    wheel.url, target, expected_size=wheel.size,
                    expected_sha256=wheel.sha256, max_bytes=1_000_000_000,
                )
                total_bytes += wheel.size
                if total_bytes > plan.estimated_bytes:
                    raise ValueError("INSTALL_SIZE_MISMATCH")
                lines.append(f"{wheel.distribution}=={wheel.version} \\")
                lines.append(f"    --hash=sha256:{wheel.sha256}")
            lock_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            os.chmod(lock_path, 0o600)
            image = f"{_IMAGE_NAME}@{plan.image_digest}"
            base_mounts = [
                (environment_dir, "/opt/environment", "rw"),
                (wheels_dir, "/wheels", "ro"),
                (lock_path, "/tmp/requirements.txt", "ro"),
            ]
            create_env = await self._container_command(
                image, ["python", "-m", "venv", "--copies", "/opt/environment"],
                mounts=base_mounts, timeout_seconds=120, output_limit=64_000,
            )
            if create_env[0] != 0 or create_env[3] or create_env[4]:
                raise RuntimeError("ENVIRONMENT_CREATE_FAILED")
            install = await self._container_command(
                image,
                [
                    "/opt/environment/bin/python", "-m", "pip", "install",
                    "--no-index", "--find-links", "/wheels", "--require-hashes",
                    "--no-cache-dir", "-r", "/tmp/requirements.txt",
                ],
                mounts=base_mounts, timeout_seconds=300, output_limit=64_000,
            )
            if install[0] != 0 or install[3] or install[4]:
                raise RuntimeError("OFFLINE_INSTALL_FAILED")
            probe = await self._container_command(
                image,
                [
                    "/opt/environment/bin/python", "-c",
                    "import importlib.metadata as m,json; print(json.dumps({e.name:e.value for d in m.distributions() for e in d.entry_points if e.group == 'console_scripts'}))",
                ],
                mounts=[(environment_dir, "/opt/environment", "ro")],
                timeout_seconds=30, output_limit=64_000,
            )
            if probe[0] != 0 or probe[3] or probe[4]:
                raise RuntimeError("ENVIRONMENT_PROBE_FAILED")
            entrypoints = json.loads(probe[1].decode("utf-8"))
            environment_sha = _tree_sha256(environment_dir)
            manifest = {
                "plan_sha256": plan.plan_sha256,
                "environment_sha256": environment_sha,
                "entrypoints": entrypoints,
                "image_ref": image,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            manifest_path = staging_root / "installed-manifest.json"
            manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
            os.chmod(manifest_path, 0o400)
            for wheel_path in wheels_dir.iterdir():
                wheel_path.unlink()
            wheels_dir.rmdir()
            os.chmod(environment_dir, 0o555)
            os.replace(staging_root, environment_root)
            final_environment_dir = environment_root / "python"
            final_manifest_path = environment_root / "installed-manifest.json"
            self._environments[environment_id] = final_environment_dir
            self._environment_digests[environment_id] = environment_sha
            self._environment_plans[environment_id] = plan.plan_sha256
            self._environment_images[environment_id] = image
            return EnvironmentRecord(
                environment_id=environment_id, plan_sha256=plan.plan_sha256,
                status="READY", lock_ref=str((environment_root / "requirements.txt").relative_to(self.storage_root)),
                installed_manifest_ref=str(final_manifest_path.relative_to(self.storage_root)),
                entrypoints=entrypoints, backend="podman-rootless",
                created_at=datetime.now(timezone.utc), environment_sha256=environment_sha,
            )
        except asyncio.CancelledError:
            shutil.rmtree(staging_root, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(staging_root, ignore_errors=True)
            reason = str(exc) if str(exc).isupper() and " " not in str(exc) else type(exc).__name__
            return _failed_environment(plan, reason[:100])

    async def run(
        self, request: ExecutionRequest, *, output_directory: Path,
    ) -> ExecutionRecord:
        started = datetime.now(timezone.utc)
        environment = self._environments.get(request.environment_id)
        if self.executable is None or environment is None:
            return _execution_failure(request, started, "SANDBOX_UNAVAILABLE")
        expected_environment_sha = self._environment_digests.get(request.environment_id)
        if (
            expected_environment_sha is None or _path_has_symlink(environment)
            or _tree_sha256(environment) != expected_environment_sha
        ):
            return _execution_failure(request, started, "ENVIRONMENT_CHANGED")
        workspace = Path(request.workspace_root).resolve()
        script = _safe_workspace_file(workspace, request.script_path)
        if script is None or _sha256_file(script) != request.script_digest:
            return _execution_failure(request, started, "GENOME_MISMATCH")
        stage = output_directory.parent / f".{request.execution_id}-inputs"
        try:
            stage.mkdir(mode=0o700)
            staged_script = stage / "main.py"
            shutil.copyfile(script, staged_script)
            if _sha256_file(staged_script) != request.script_digest:
                raise ValueError("GENOME_MISMATCH")
            input_dir = stage / "inputs"
            input_dir.mkdir(mode=0o700)
            for key, relative in request.input_refs.items():
                source = _safe_workspace_file(workspace, relative)
                if source is None:
                    raise ValueError("INPUT_BINDING_UNAUTHORIZED")
                shutil.copyfile(source, input_dir / key)
                if request.input_sha256 and _sha256_file(input_dir / key) != request.input_sha256.get(key):
                    raise ValueError("INPUT_BINDING_CHANGED")
            image = self._image_for_environment(request.environment_id)
            name = f"photomat-{request.execution_id.removeprefix('exec_')[:20]}"
            command = [
                str(self.executable), "run", "--name", name,
                "--network=none", "--read-only", "--cap-drop=all",
                "--security-opt=no-new-privileges", "--pids-limit=128",
                "--memory=1g", "--cpus=2", "--userns=keep-id",
                "--user", f"{os.getuid()}:{os.getgid()}",
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=128m,mode=1777",
                "--env", "HOME=/tmp", "--env", "PYTHONNOUSERSITE=1",
                "--volume", f"{stage}:/input:ro,Z",
                "--volume", f"{environment}:/opt/environment:ro,Z",
                "--volume", f"{output_directory}:/output:rw,Z",
                "--workdir", "/output", image,
                "/opt/environment/bin/python", "/input/main.py", *request.argv,
            ]
            try:
                exit_code, stdout, stderr, timed_out, output_overflow = await _run_bounded_process(
                    command, timeout_seconds=request.timeout_seconds,
                    output_limit=request.output_limit_bytes,
                    env=_podman_environment(),
                )
            finally:
                await self._cleanup_container(name)
            artifacts = _collect_artifacts(output_directory, request.artifact_limit_bytes)
            if timed_out or output_overflow:
                artifacts = ()
            state: ExecutionState = "INTERRUPTED" if timed_out else "FAILED" if output_overflow or exit_code != 0 else "COMPLETED"
            return ExecutionRecord(
                execution_id=request.execution_id, backend="podman", state=state, exit_code=exit_code,
                timed_out=timed_out,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                artifacts=artifacts, environment_sha256=_environment_sha_from_ref(environment),
                script_sha256=request.script_digest, started_at=started,
                ended_at=datetime.now(timezone.utc),
                reason_code="EXECUTION_TIMEOUT" if timed_out else "OUTPUT_LIMIT_EXCEEDED" if output_overflow else None,
            )
        except Exception as exc:
            return _execution_failure(request, started, _reason_code(exc))
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    async def reconcile(self, execution_id: str) -> ExecutionRecord | None:
        return None

    async def _ready_image(self, digest: str) -> bool:
        if self.executable is None:
            return False
        info = await _run_bounded_process(
            [str(self.executable), "info", "--format", "json"],
            timeout_seconds=15, output_limit=64_000,
            env=_podman_environment(),
        )
        if info[0] != 0 or info[3] or info[4]:
            return False
        try:
            payload = json.loads(info[1])
            rootless = bool(payload.get("host", {}).get("security", {}).get("rootless"))
        except (ValueError, AttributeError, TypeError):
            return False
        if not rootless:
            return False
        image = f"{_IMAGE_NAME}@{digest}"
        result = await _run_bounded_process(
            [str(self.executable), "image", "exists", image],
            timeout_seconds=15, output_limit=8_000,
            env=_podman_environment(),
        )
        return result[0] == 0 and not result[3] and not result[4]

    async def _container_command(
        self,
        image: str,
        command: list[str],
        *,
        mounts: list[tuple[Path, str, str]],
        timeout_seconds: int,
        output_limit: int,
    ) -> tuple[int, bytes, bytes, bool, bool]:
        if self.executable is None:
            return 127, b"", b"", False, False
        name = f"photomat-install-{uuid4().hex[:20]}"
        arguments = [
            str(self.executable), "run", "--name", name, "--network=none", "--read-only",
            "--cap-drop=all", "--security-opt=no-new-privileges", "--pids-limit=128",
            "--memory=2g", "--cpus=2", "--userns=keep-id", "--user",
            f"{os.getuid()}:{os.getgid()}", "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=128m,mode=1777",
            "--env", "HOME=/tmp", "--env", "PIP_NO_CACHE_DIR=1",
        ]
        for source, target, permission in mounts:
            if any(character in str(source) for character in (":", ",", "\n", "\x00")):
                raise ValueError("INVALID_STORAGE_PATH")
            arguments.extend(["--volume", f"{source}:{target}:{permission},Z"])
        arguments.extend([image, *command])
        try:
            return await _run_bounded_process(arguments, timeout_seconds=timeout_seconds, output_limit=output_limit,
                                              env=_podman_environment())
        finally:
            await self._cleanup_container(name)

    async def _cleanup_container(self, name: str) -> None:
        if self.executable is None:
            return
        # Podman's default stop grace period is itself ten seconds, exceeding
        # our cleanup deadline for a live timed-out container. Stop immediately
        # before removal; this also works with older Podman without rm --time.
        await _run_bounded_process(
            [str(self.executable), "stop", "--time", "0", name],
            timeout_seconds=10, output_limit=8_000, env=_podman_environment(),
        )
        await _run_bounded_process(
            [str(self.executable), "rm", "--force", name], timeout_seconds=10, output_limit=8_000,
            env=_podman_environment(),
        )

    def _image_for_environment(self, environment_id: str) -> str:
        manifest = self.storage_root / environment_id / "installed-manifest.json"
        if _path_has_symlink(manifest):
            raise ValueError("ENVIRONMENT_MANIFEST_INVALID")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        expected_plan = self._environment_plans.get(environment_id)
        expected_image = self._environment_images.get(environment_id)
        if not expected_plan or not expected_image:
            raise ValueError("INSTALL_PLAN_CHANGED")
        if payload.get("plan_sha256") != expected_plan:
            raise ValueError("SANDBOX_IMAGE_BINDING_MISSING")
        if payload.get("image_ref") != expected_image:
            raise ValueError("INSTALL_PLAN_CHANGED")
        return expected_image


def _podman_environment() -> dict[str, str]:
    """Supply rootless host identity without forwarding provider credentials.

    These variables configure the Podman host process only. Container variables
    remain the explicit --env allowlist in the container command.
    """
    import pwd

    environment = {"PATH": os.defpath, "HOME": pwd.getpwuid(os.getuid()).pw_dir, "LANG": "C.UTF-8"}
    runtime = Path(f"/run/user/{os.getuid()}")
    if runtime.is_dir() and not runtime.is_symlink() and runtime.stat().st_uid == os.getuid():
        environment["XDG_RUNTIME_DIR"] = str(runtime)
    return environment


async def _run_bounded_process(
    arguments: list[str], *, timeout_seconds: float, output_limit: int,
    cwd: Path | None = None, env: dict[str, str] | None = None,
    stdin_data: bytes | None = None,
) -> tuple[int, bytes, bytes, bool, bool]:
    process = await asyncio.create_subprocess_exec(
        *arguments, stdin=asyncio.subprocess.PIPE if stdin_data is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env or {"PATH": os.defpath, "HOME": "/nonexistent", "LANG": "C.UTF-8"},
        start_new_session=(os.name == "posix"),
    )
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("SUBPROCESS_PIPE_UNAVAILABLE")

    async def read(stream, limit):
        output = bytearray()
        while chunk := await stream.read(32 * 1024):
            output.extend(chunk)
            if len(output) > limit:
                _kill_process_tree(process)
                return bytes(output[:limit]), True
        return bytes(output), False

    async def write_input() -> None:
        if process.stdin is None or stdin_data is None:
            return
        process.stdin.write(stdin_data)
        await process.stdin.drain()
        process.stdin.close()

    try:
        output_task = asyncio.create_task(read(process.stdout, output_limit))
        error_task = asyncio.create_task(read(process.stderr, output_limit))
        input_task = asyncio.create_task(write_input())
        (stdout, stdout_over), (stderr, stderr_over), _, _ = await asyncio.wait_for(
            asyncio.gather(output_task, error_task, process.wait(), input_task), timeout=timeout_seconds,
        )
        overflow = stdout_over or stderr_over
        # A parent may exit after starting a detached child that closes or
        # redirects both pipes. Do not leave that process behind on success.
        _kill_process_tree(process)
        return process.returncode or 0, stdout, stderr, False, overflow
    except TimeoutError:
        _kill_process_tree(process)
        await process.wait()
        await asyncio.gather(output_task, error_task, input_task, return_exceptions=True)
        return 124, b"", b"", True, False
    except asyncio.CancelledError:
        _kill_process_tree(process)
        await process.wait()
        await asyncio.gather(output_task, error_task, input_task, return_exceptions=True)
        raise
    except Exception:
        _kill_process_tree(process)
        await process.wait()
        await asyncio.gather(output_task, error_task, input_task, return_exceptions=True)
        raise


def _kill_process_tree(process: asyncio.subprocess.Process) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
            return
        except ProcessLookupError:
            return
    if process.returncode is None:
        process.kill()


def _failed_environment(plan: InstallPlan, reason: str) -> EnvironmentRecord:
    return EnvironmentRecord(
        environment_id=f"env_failed_{plan.plan_sha256[:16]}",
        plan_sha256=plan.plan_sha256, status="FAILED", lock_ref="",
        installed_manifest_ref="", entrypoints={}, backend="podman-rootless",
        created_at=datetime.now(timezone.utc), environment_sha256="0" * 64,
        reason_code=reason,
    )


def _execution_failure(request: ExecutionRequest, started: datetime, reason: str) -> ExecutionRecord:
    return ExecutionRecord(
        execution_id=request.execution_id, state="FAILED", exit_code=None,
        timed_out=reason == "EXECUTION_TIMEOUT", stdout="", stderr="",
        artifacts=(), environment_sha256="0" * 64, script_sha256=request.script_digest,
        started_at=started, ended_at=datetime.now(timezone.utc), reason_code=reason[:100],
    )


def _safe_workspace_file(root: Path, relative: str) -> Path | None:
    path = Path(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        return None
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            return None
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(root)
        return resolved if resolved.is_file() else None
    except (OSError, ValueError):
        return None


def _collect_artifacts(root: Path, output_limit: int) -> tuple[ArtifactRecord, ...]:
    artifacts: list[ArtifactRecord] = []
    total = 0
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("ARTIFACT_CHANGED")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("ARTIFACT_CHANGED")
        relative = path.relative_to(root).as_posix()
        total += info.st_size
        if len(artifacts) >= _MAX_ARTIFACTS or total > output_limit:
            raise ValueError("OUTPUT_LIMIT_EXCEEDED")
        artifacts.append(ArtifactRecord(
            path=relative, size=info.st_size, sha256=_sha256_file(path),
        ))
    return tuple(artifacts)


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    total = 0
    count = 0
    resolved_root = root.resolve(strict=True)
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        relative = path.relative_to(root).as_posix().encode()
        if stat.S_ISLNK(info.st_mode):
            try:
                target = path.resolve(strict=True)
                target_info = target.stat()
            except (OSError, RuntimeError) as exc:
                raise ValueError("ENVIRONMENT_MANIFEST_INVALID") from exc
            if (target != resolved_root and resolved_root not in target.parents
                    or not (stat.S_ISDIR(target_info.st_mode) or stat.S_ISREG(target_info.st_mode))):
                raise ValueError("ENVIRONMENT_MANIFEST_INVALID")
            target_name = os.readlink(path).encode("utf-8", errors="strict")
            count += 1
            total += len(target_name)
            if count > 100_000 or total > _MAX_ENVIRONMENT_BYTES:
                raise ValueError("ENVIRONMENT_SIZE_LIMIT_EXCEEDED")
            digest.update(b"L")
            digest.update(relative)
            digest.update(target_name)
            continue
        if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
            raise ValueError("ENVIRONMENT_MANIFEST_INVALID")
        if stat.S_ISDIR(info.st_mode):
            continue
        count += 1
        total += info.st_size
        if count > 100_000 or total > _MAX_ENVIRONMENT_BYTES:
            raise ValueError("ENVIRONMENT_SIZE_LIMIT_EXCEEDED")
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _environment_sha_from_ref(environment: Path) -> str:
    manifest = environment.parent / "installed-manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    value = payload.get("environment_sha256")
    return value if isinstance(value, str) and len(value) == 64 else "0" * 64


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_has_symlink(path: Path) -> bool:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _reason_code(exc: Exception) -> str:
    known = {
        "GENOME_MISMATCH", "INPUT_BINDING_UNAUTHORIZED", "ARTIFACT_CHANGED",
        "OUTPUT_LIMIT_EXCEEDED", "SANDBOX_UNAVAILABLE", "INSTALL_PLAN_CHANGED",
    }
    message = str(exc)
    return message if message in known else type(exc).__name__
