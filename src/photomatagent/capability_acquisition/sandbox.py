"""Trusted binding, lifecycle checks, and optional sandbox execution backend."""
from __future__ import annotations

import json
import hashlib
import asyncio
import os
import re
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from photomatagent.capability_acquisition.models import (
    MAX_SCRIPT_ARGV_TOKENS,
    EnvironmentRecord,
    ExecutionRecord,
    ExecutionRequest,
    InstallPlan,
    SandboxResult,
)
from photomatagent.workspace import Workspace


class SandboxBackend(Protocol):
    async def install(self, plan: InstallPlan) -> EnvironmentRecord: ...

    async def run(self, request: ExecutionRequest, *, output_directory: Path) -> ExecutionRecord: ...

    async def reconcile(self, execution_id: str) -> ExecutionRecord | None: ...


class SandboxService:
    """Expose only pre-bound scripts and READY environments to runtime tools."""

    def __init__(
        self,
        *,
        workspace: Path | str,
        trial_id: str,
        genome_sha256: str,
        scripts: dict[str, str],
        backend: SandboxBackend | None = None,
        input_bindings: dict[str, str] | None = None,
        artifact_limit_bytes: int = 64 * 1024 * 1024,
        dynamic_script_prefix: str | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.trial_id = trial_id
        self.genome_sha256 = genome_sha256
        self.scripts = dict(scripts)
        self.input_bindings = dict(input_bindings or {})
        self.backend = backend
        self.artifact_limit_bytes = artifact_limit_bytes
        self.dynamic_script_prefix = dynamic_script_prefix
        self._environments: dict[str, EnvironmentRecord] = {}
        self._requests: dict[str, ExecutionRequest] = {}
        self._output_directories: dict[str, Path] = {}
        self._records: dict[str, ExecutionRecord] = {}
        self._plans: dict[str, InstallPlan] = {}

    def environment(self, environment_id: str) -> EnvironmentRecord | None:
        return self._environments.get(environment_id)

    def register_environment(self, environment: EnvironmentRecord, *, plan: InstallPlan | None = None) -> None:
        if environment.status != "READY":
            raise ValueError("ENVIRONMENT_NOT_READY")
        if plan is not None:
            if plan.plan_sha256 != environment.plan_sha256:
                raise ValueError("INSTALL_PLAN_CHANGED")
            self._plans[environment.environment_id] = plan
        self._environments[environment.environment_id] = environment

    def execution_inputs(self, execution_id: str) -> tuple[dict[str, Path], dict[str, str]]:
        """Return hash-verified input snapshots bound to an actual execution.

        Consumers must check these bytes, not mutable caller paths observed before
        dependency installation or after execution. Backend staging verifies the
        same request digests before invoking the script.
        """
        request = self._requests.get(execution_id)
        record = self._records.get(execution_id)
        if (request is None or record is None or record.execution_id != request.execution_id
                or record.script_sha256 != request.script_digest
                or set(request.input_refs) != set(request.input_sha256)):
            raise ValueError("EXECUTION_INPUTS_UNAVAILABLE")
        root = self._provenance_root(execution_id)
        try:
            persisted = json.loads((root / "request.json").read_text(encoding="utf-8"))
            if (ExecutionRequest.model_validate(persisted["request"]) != request
                    or persisted["input_sha256"] != request.input_sha256):
                raise ValueError("EXECUTION_INPUT_BINDING_CHANGED")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("EXECUTION_INPUT_BINDING_CHANGED") from exc
        inputs: dict[str, Path] = {}
        digests: dict[str, str] = {}
        for key, expected in request.input_sha256.items():
            snapshot = _safe_output_file(root, f"input-{key}")
            if snapshot is None or _sha256_file(snapshot) != expected:
                raise ValueError("EXECUTION_INPUT_SNAPSHOT_CHANGED")
            inputs[key] = snapshot
            digests[key] = expected
        return inputs, digests

    def archive_artifacts(
        self, execution_id: str, *, destination: Workspace,
        declared_paths: frozenset[str],
    ) -> tuple[str, ...]:
        """Copy hash-verified execution artifacts to declared trial output paths."""
        record = self._records.get(execution_id)
        output_directory = self._output_directories.get(execution_id)
        if record is None or output_directory is None:
            raise ValueError("EXECUTION_ARTIFACTS_UNAVAILABLE")
        archived: list[str] = []
        for artifact in record.artifacts:
            relative = Path(artifact.path)
            normalized = relative.as_posix()
            if (relative.is_absolute() or not relative.parts or ".." in relative.parts
                    or "\\" in artifact.path or normalized not in declared_paths):
                raise ValueError("UNDECLARED_EXECUTION_ARTIFACT")
            source = _safe_output_file(output_directory, artifact.path)
            if (source is None or source.stat().st_size != artifact.size
                    or _sha256_file(source) != artifact.sha256):
                raise ValueError("EXECUTION_ARTIFACT_CHANGED")
            target = destination.root
            for part in relative.parts:
                target = target / part
                if target.is_symlink():
                    raise ValueError("QD_OUTPUT_PATH_UNSAFE")
            if not destination.contains(target.resolve(strict=False)) or target.exists():
                raise ValueError("QD_OUTPUT_PATH_UNSAFE")
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=".qd-artifact-", dir=target.parent)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(source.read_bytes())
                    stream.flush()
                    os.fsync(stream.fileno())
                if _sha256_file(Path(temporary)) != artifact.sha256:
                    raise ValueError("EXECUTION_ARTIFACT_CHANGED")
                os.replace(temporary, target)
            except BaseException:
                Path(temporary).unlink(missing_ok=True)
                raise
            archived.append(normalized)
        return tuple(archived)

    async def install(self, plan: InstallPlan) -> SandboxResult:
        if self.backend is None:
            return SandboxResult(code="SANDBOX_UNAVAILABLE", diagnostics=("BACKEND_NOT_CONFIGURED",))
        if plan.expires_at <= datetime.now(timezone.utc):
            return SandboxResult(code="INSTALL_PLAN_EXPIRED")
        try:
            environment = await asyncio.wait_for(self.backend.install(plan), timeout=900.0)
        except TimeoutError:
            return SandboxResult(code="INSTALL_TIMEOUT")
        except Exception as exc:
            return SandboxResult(code="INSTALL_FAILED", diagnostics=(type(exc).__name__,))
        if environment.plan_sha256 != plan.plan_sha256:
            return SandboxResult(code="INSTALL_PLAN_CHANGED", environment=environment)
        if environment.status != "READY":
            return SandboxResult(
                code=environment.reason_code or "INSTALL_FAILED", environment=environment,
                diagnostics=("ENVIRONMENT_NOT_READY",),
            )
        self.register_environment(environment)
        self._plans[environment.environment_id] = plan
        plan_root = self.workspace / ".photomatagent" / "capability-plans"
        if _contains_symlink(self.workspace, plan_root):
            return SandboxResult(code="INVALID_PROVENANCE_PATH")
        plan_root.mkdir(parents=True, exist_ok=True)
        plan_file = plan_root / f"{plan.plan_sha256}.json"
        if not plan_file.exists():
            plan_file.write_text(plan.model_dump_json(), encoding="utf-8")
        return SandboxResult(code="OK", environment=environment)

    async def run(
        self,
        *,
        script_path: str,
        environment_id: str,
        input_refs: dict[str, str] | None = None,
        argv: list[str] | tuple[str, ...] = (),
        timeout_seconds: int = 60,
        output_limit_bytes: int = 65_536,
    ) -> SandboxResult:
        if self.backend is None:
            return SandboxResult(code="SANDBOX_UNAVAILABLE", diagnostics=("BACKEND_NOT_CONFIGURED",))
        environment = self._environments.get(environment_id)
        if environment is None or environment.status != "READY":
            return SandboxResult(code="ENVIRONMENT_NOT_READY")
        if not 1 <= timeout_seconds <= 120 or not 1024 <= output_limit_bytes <= 1_048_576:
            return SandboxResult(code="INVALID_EXECUTION_LIMIT")
        if len(argv) > MAX_SCRIPT_ARGV_TOKENS or any(not isinstance(value, str) or len(value) > 1000 or "\x00" in value for value in argv):
            return SandboxResult(code="INVALID_EXECUTION_ARGUMENT")
        entrypoint_name = script_path.removeprefix("entrypoint:") if script_path.startswith("entrypoint:") else None
        if entrypoint_name is not None:
            if (environment.backend != "local_dev" or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", entrypoint_name)
                    or entrypoint_name not in environment.entrypoints):
                return SandboxResult(code="ENTRYPOINT_UNAVAILABLE")
            script_digest = hashlib.sha256(
                f"entrypoint:{entrypoint_name}:{environment.entrypoints[entrypoint_name]}:{environment.environment_sha256}".encode()
            ).hexdigest()
        else:
            script = self._bound_file(script_path)
            expected_digest = self.scripts.get(script_path)
            if script is None or (expected_digest is None and not self._dynamic_path(script_path)):
                return SandboxResult(code="INVALID_SCRIPT_PATH")
            script_digest = _sha256_file(script)
            if expected_digest is not None and script_digest != expected_digest:
                return SandboxResult(code="GENOME_MISMATCH")
        validated_inputs: dict[str, str] = {}
        input_digests: dict[str, str] = {}
        for key, path in (input_refs or {}).items():
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", key):
                return SandboxResult(code="INVALID_INPUT_BINDING")
            if self.input_bindings.get(key) != path and not self._dynamic_path(path):
                return SandboxResult(code="INPUT_BINDING_UNAUTHORIZED")
            input_file = self._bound_file(path)
            if input_file is None or not input_file.is_file():
                return SandboxResult(code="INVALID_INPUT_BINDING")
            if input_file.stat().st_size > self.artifact_limit_bytes:
                return SandboxResult(code="INPUT_LIMIT_EXCEEDED")
            validated_inputs[key] = path
            input_digests[key] = _sha256_file(input_file)
        execution_id = f"exec_{uuid4().hex}"
        output_root = self.workspace / "tmp" / "capability-executions"
        if _contains_symlink(self.workspace, output_root):
            return SandboxResult(code="INVALID_OUTPUT_PATH")
        output_root.mkdir(parents=True, exist_ok=True)
        output_directory = output_root / execution_id
        output_directory.mkdir()
        request = ExecutionRequest(
            execution_id=execution_id,
            trial_id=self.trial_id,
            genome_sha256=self.genome_sha256,
            environment_id=environment.environment_id,
            script_digest=script_digest,
            script_path=script_path,
            input_refs=validated_inputs,
            input_sha256=input_digests,
            argv=tuple(argv),
            timeout_seconds=timeout_seconds,
            output_limit_bytes=output_limit_bytes,
            artifact_limit_bytes=self.artifact_limit_bytes,
            workspace_root=str(self.workspace),
        )
        self._requests[execution_id] = request
        try:
            self._persist_request(request, environment)
        except (OSError, ValueError):
            return SandboxResult(code="EXECUTION_PROVENANCE_INVALID")
        self._output_directories[execution_id] = output_directory
        try:
            record = await asyncio.wait_for(
                self.backend.run(request, output_directory=output_directory),
                timeout=timeout_seconds + 30.0,
            )
        except TimeoutError:
            return SandboxResult(code="EXECUTION_TIMEOUT", diagnostics=("SANDBOX_BACKEND_TIMEOUT",))
        except Exception as exc:
            return SandboxResult(code="SANDBOX_FAILED", diagnostics=(type(exc).__name__,))
        if (
            record.execution_id != execution_id
            or record.script_sha256 != script_digest
            or record.environment_sha256 != environment.environment_sha256
        ):
            return SandboxResult(code="ARTIFACT_CHANGED", diagnostics=("EXECUTION_BINDING_MISMATCH",))
        if len(record.stdout.encode("utf-8")) > output_limit_bytes or len(record.stderr.encode("utf-8")) > output_limit_bytes:
            return SandboxResult(code="OUTPUT_LIMIT_EXCEEDED", record=record.model_copy(update={"stdout": "", "stderr": "", "artifacts": ()}))
        if sum(artifact.size for artifact in record.artifacts) > self.artifact_limit_bytes:
            return SandboxResult(code="OUTPUT_LIMIT_EXCEEDED", diagnostics=("ARTIFACT_QUOTA_EXCEEDED",))
        for artifact in record.artifacts:
            artifact_file = _safe_output_file(output_directory, artifact.path)
            if artifact_file is None or not artifact_file.is_file() or artifact_file.stat().st_size != artifact.size or _sha256_file(artifact_file) != artifact.sha256:
                return SandboxResult(code="ARTIFACT_CHANGED", diagnostics=("INVALID_OUTPUT_ARTIFACT",))
        self._records[execution_id] = record
        self._persist_record(record)
        if record.timed_out:
            code = "EXECUTION_TIMEOUT"
        elif record.reason_code == "OUTPUT_LIMIT_EXCEEDED":
            code = "OUTPUT_LIMIT_EXCEEDED"
        elif record.state != "COMPLETED" or record.exit_code != 0:
            code = "EXECUTION_FAILED"
        else:
            code = "OK"
        return SandboxResult(code=code, record=record)

    async def reconcile(self, execution_id: str) -> SandboxResult:
        if self.backend is None:
            return SandboxResult(code="SANDBOX_UNAVAILABLE")
        if execution_id not in self._requests:
            return SandboxResult(code="EXECUTION_NOT_FOUND")
        try:
            record = await asyncio.wait_for(self.backend.reconcile(execution_id), timeout=30.0)
        except TimeoutError:
            return SandboxResult(code="RECONCILIATION_FAILED", diagnostics=("RECONCILIATION_TIMEOUT",))
        except Exception as exc:
            return SandboxResult(code="RECONCILIATION_FAILED", diagnostics=(type(exc).__name__,))
        if record is None:
            return SandboxResult(code="EXECUTION_UNKNOWN")
        request = self._requests[execution_id]
        environment = self._environments.get(request.environment_id)
        if (
            record.execution_id != execution_id
            or record.script_sha256 != request.script_digest
            or environment is None
            or record.environment_sha256 != environment.environment_sha256
        ):
            return SandboxResult(code="ARTIFACT_CHANGED", diagnostics=("EXECUTION_BINDING_MISMATCH",))
        if (
            len(record.stdout.encode("utf-8")) > request.output_limit_bytes
            or len(record.stderr.encode("utf-8")) > request.output_limit_bytes
            or sum(artifact.size for artifact in record.artifacts) > self.artifact_limit_bytes
        ):
            return SandboxResult(code="OUTPUT_LIMIT_EXCEEDED")
        output_directory = self._output_directories[execution_id]
        for artifact in record.artifacts:
            artifact_file = _safe_output_file(output_directory, artifact.path)
            if (
                artifact_file is None or not artifact_file.is_file()
                or artifact_file.stat().st_size != artifact.size
                or _sha256_file(artifact_file) != artifact.sha256
            ):
                return SandboxResult(code="ARTIFACT_CHANGED", diagnostics=("INVALID_OUTPUT_ARTIFACT",))
        self._records[execution_id] = record
        self._persist_record(record)
        if record.timed_out:
            code = "EXECUTION_TIMEOUT"
        elif record.state != "COMPLETED" or record.exit_code != 0:
            code = "EXECUTION_FAILED"
        else:
            code = "OK"
        return SandboxResult(code=code, record=record)

    def _provenance_root(self, execution_id: str) -> Path:
        root = self.workspace / ".photomatagent" / "executions" / execution_id
        if _contains_symlink(self.workspace, root):
            raise ValueError("INVALID_PROVENANCE_PATH")
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _persist_request(self, request: ExecutionRequest, environment: EnvironmentRecord) -> None:
        """Freeze inputs and script at the authoritative execution boundary."""
        root = self._provenance_root(request.execution_id)
        script = self._bound_file(request.script_path)
        if script is not None:
            (root / "script.py").write_bytes(script.read_bytes())
        inputs: dict[str, str] = {}
        for key, relative in request.input_refs.items():
            source = self._bound_file(relative)
            if source is not None:
                snapshot = root / f"input-{key}"
                snapshot.write_bytes(source.read_bytes())
                inputs[key] = _sha256_file(snapshot)
                if inputs[key] != request.input_sha256.get(key):
                    raise ValueError("INPUT_BINDING_CHANGED")
        plan = self._plans.get(environment.environment_id)
        payload = {"request": request.model_dump(mode="json"),
                   "environment": environment.model_dump(mode="json"),
                   "plan": plan.model_dump(mode="json") if plan else None,
                   "input_sha256": inputs}
        (root / "request.json").write_text(json.dumps(payload), encoding="utf-8")

    def _persist_record(self, record: ExecutionRecord) -> None:
        root = self._provenance_root(record.execution_id)
        temporary = root / ".record.json.tmp"
        temporary.write_text(record.model_dump_json(), encoding="utf-8")
        os.replace(temporary, root / "record.json")

    def _bound_file(self, relative_path: str) -> Path | None:
        path = Path(relative_path)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            return None
        current = self.workspace
        for part in path.parts:
            current = current / part
            if current.is_symlink():
                return None
        try:
            resolved = current.resolve(strict=True)
            resolved.relative_to(self.workspace)
        except (OSError, ValueError):
            return None
        return resolved if resolved.is_file() else None

    def _dynamic_path(self, relative_path: str) -> bool:
        prefix = self.dynamic_script_prefix
        if prefix is None:
            return False
        path = Path(relative_path)
        return not path.is_absolute() and ".." not in path.parts and path.parts[:len(Path(prefix).parts)] == Path(prefix).parts


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contains_symlink(root: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _safe_output_file(output_root: Path, relative_path: str) -> Path | None:
    path = Path(relative_path)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        return None
    current = output_root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            return None
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(output_root.resolve())
    except (OSError, ValueError):
        return None
    try:
        mode = resolved.lstat().st_mode
    except OSError:
        return None
    return resolved if stat.S_ISREG(mode) else None
