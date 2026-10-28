"""Restore an exact published QD workflow for one isolated evaluation case."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from photomatagent.capability_acquisition.models import EnvironmentRecord
from photomatagent.capability_acquisition.sandbox import SandboxBackend, SandboxService
from photomatagent.capability_acquisition.tools import ScriptRunTool
from photomatagent.scientific.evolution.qd.models import WorkflowGenome
from photomatagent.skills.published import PublishedSkillSource
from photomatagent.tools.base import ToolResult
from photomatagent.workspace import Workspace


class FrozenScriptTool(ScriptRunTool):
    """The ordinary script tool, restricted to exact frozen workflow calls."""

    def __init__(self, sandbox: SandboxService, allowed: dict[str, dict[str, Any]]) -> None:
        super().__init__(sandbox)
        self.allowed = allowed
        self.succeeded: set[str] = set()
        self.result: dict[str, bytes | str] = {}

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        path = arguments.get("script_path")
        frozen = self.allowed.get(path) if isinstance(path, str) else None
        if frozen is None or set(arguments) - set(frozen):
            return ToolResult(output="EVALUATION_SCRIPT_UNAUTHORIZED", is_error=True)
        normalized = {**frozen, **arguments}
        if normalized != frozen:
            return ToolResult(output="EVALUATION_SCRIPT_UNAUTHORIZED", is_error=True)
        result = await super().execute(frozen)
        if not result.is_error and result.data.get("code") == "OK":
            assert isinstance(path, str)
            try:
                payload = json.loads(result.output)
                execution_id = payload["execution_id"]
                artifacts = payload["artifacts"]
                artifact = next(item for item in artifacts if item["path"] == "result.json")
                if (not isinstance(execution_id, str)
                        or re.fullmatch(r"exec_[0-9a-f]{32}", execution_id) is None
                        or not isinstance(artifact["sha256"], str)):
                    raise ValueError("EVALUATION_RESULT_ARTIFACT_INVALID")
                relative = f"tmp/capability-executions/{execution_id}/result.json"
                output = Workspace(self.sandbox.workspace).resolve(relative, must_exist=True)
                data = output.read_bytes()
                if (output.is_symlink() or len(data) != artifact["size"]
                        or hashlib.sha256(data).hexdigest() != artifact["sha256"]
                        or not isinstance(json.loads(data), (dict, list))):
                    raise ValueError("EVALUATION_RESULT_ARTIFACT_INVALID")
            except (OSError, KeyError, ValueError, TypeError, StopIteration):
                return ToolResult(output="EVALUATION_RESULT_ARTIFACT_INVALID", is_error=True)
            self.result["bytes"] = data
            self.result["sha256"] = artifact["sha256"]
            self.succeeded.add(path)
        return result


def frozen_script_tool(
    case_root: Path, *, bundle_digests: tuple[str, ...],
    backend: SandboxBackend, storage_root: Path,
) -> FrozenScriptTool:
    """Verify all bundle bytes and restore matching preinstalled environments.

    No installation or network work is done by evaluation. The host must supply
    the backend and its existing environment store explicitly.
    """
    skills_root = case_root / "frozen-skills"
    bundles = sorted(path for path in skills_root.iterdir() if path.is_dir() and not path.is_symlink())
    if not bundles or len(bundles) != len(bundle_digests):
        raise ValueError("EVALUATION_FROZEN_SKILL_MISMATCH")
    verifier = PublishedSkillSource(skills_root)
    scripts: dict[str, str] = {}
    allowed: dict[str, dict[str, Any]] = {}
    environments: dict[str, EnvironmentRecord] = {}
    verified: set[str] = set()
    for bundle in bundles:
        matches = [digest for digest in bundle_digests if _matches(verifier, bundle, digest)]
        if len(matches) != 1 or matches[0] in verified:
            raise ValueError("EVALUATION_FROZEN_SKILL_MISMATCH")
        verified.add(matches[0])
        genome = WorkflowGenome.model_validate_json((bundle / "workflow.json").read_text(encoding="utf-8"))
        provenance = json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))
        lock = (bundle / "requirements.lock").read_bytes()
        if (provenance.get("genome_sha256") != genome.content_sha256
                or provenance.get("environment_sha256") != genome.environment_ref
                or provenance.get("lock_sha256") != hashlib.sha256(lock).hexdigest()):
            raise ValueError("EVALUATION_FROZEN_PROVENANCE_MISMATCH")
        environment = _restore_environment(backend, storage_root, genome.environment_ref, lock)
        environments[environment.environment_id] = environment
        refs = {Path(ref.path).name: ref for ref in genome.script_refs}
        if len(refs) != 1 or len(genome.script_refs) != 1 or len(genome.steps) != 1:
            raise ValueError("EVALUATION_FROZEN_WORKFLOW_INVALID")
        for step in genome.steps:
            if step.kind != "script" or step.binding_id != "script_run":
                raise ValueError("EVALUATION_FROZEN_WORKFLOW_INVALID")
            source_path = step.parameters.get("script_path")
            if not isinstance(source_path, str) or Path(source_path).name not in refs:
                raise ValueError("EVALUATION_FROZEN_WORKFLOW_INVALID")
            ref = refs[Path(source_path).name]
            path = f"frozen-skills/{bundle.name}/scripts/{Path(source_path).name}"
            script = case_root / path
            if not script.is_file() or hashlib.sha256(script.read_bytes()).hexdigest() != ref.sha256:
                raise ValueError("EVALUATION_FROZEN_SCRIPT_MISMATCH")
            original_inputs = step.parameters.get("input_refs")
            if not isinstance(original_inputs, Mapping) or len(original_inputs) != 1:
                raise ValueError("EVALUATION_FROZEN_INPUT_MAPPING_INCOMPLETE")
            key = next(iter(original_inputs))
            if not isinstance(key, str) or not key:
                raise ValueError("EVALUATION_FROZEN_INPUT_MAPPING_INCOMPLETE")
            argv = step.parameters.get("argv", [])
            if not isinstance(argv, (list, tuple)) or any(not isinstance(item, str) for item in argv):
                raise ValueError("EVALUATION_FROZEN_WORKFLOW_INVALID")
            timeout = step.parameters.get("timeout_seconds", 60)
            if type(timeout) is not int or not 1 <= timeout <= 120 or path in allowed:
                raise ValueError("EVALUATION_FROZEN_WORKFLOW_INVALID")
            scripts[path] = ref.sha256
            allowed[path] = {
                "script_path": path, "environment_id": environment.environment_id,
                "input_refs": {key: "input/data.json"}, "argv": list(argv),
                "timeout_seconds": timeout,
            }
    if verified != set(bundle_digests):
        raise ValueError("EVALUATION_FROZEN_SKILL_MISMATCH")
    sandbox = SandboxService(
        workspace=case_root, trial_id="frozen-evaluation",
        genome_sha256=hashlib.sha256("\0".join(sorted(verified)).encode()).hexdigest(),
        scripts=scripts, input_bindings={key: "input/data.json" for call in allowed.values()
                                         for key in call["input_refs"]}, backend=backend,
    )
    for environment in environments.values():
        sandbox.register_environment(environment)
    return FrozenScriptTool(sandbox, allowed)


def _matches(source: PublishedSkillSource, bundle: Path, digest: str) -> bool:
    try:
        source._verify_bundle(bundle, digest)
        return True
    except (OSError, ValueError, TypeError):
        return False


def preflight_frozen_skill(
    bundle: Path, digest: str, *, backend: SandboxBackend, storage_root: Path,
) -> None:
    """Reject missing executable dependencies before reserving a Web run."""
    if not _matches(PublishedSkillSource(bundle.parent.parent), bundle, digest):
        raise ValueError("EVALUATION_FROZEN_SKILL_MISMATCH")
    genome = WorkflowGenome.model_validate_json((bundle / "workflow.json").read_text(encoding="utf-8"))
    provenance = json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))
    lock = (bundle / "requirements.lock").read_bytes()
    if (provenance.get("genome_sha256") != genome.content_sha256
            or provenance.get("environment_sha256") != genome.environment_ref
            or provenance.get("lock_sha256") != hashlib.sha256(lock).hexdigest()):
        raise ValueError("EVALUATION_FROZEN_PROVENANCE_MISMATCH")
    _restore_environment(backend, storage_root, genome.environment_ref, lock)
    refs = {Path(ref.path).name: ref for ref in genome.script_refs}
    if len(genome.steps) != 1 or len(refs) != 1 or len(genome.script_refs) != 1:
        raise ValueError("EVALUATION_FROZEN_WORKFLOW_INVALID")
    for name, ref in refs.items():
        script = bundle / "scripts" / name
        if not script.is_file() or hashlib.sha256(script.read_bytes()).hexdigest() != ref.sha256:
            raise ValueError("EVALUATION_FROZEN_SCRIPT_MISMATCH")
    for step in genome.steps:
        inputs = step.parameters.get("input_refs")
        source_path = step.parameters.get("script_path")
        argv = step.parameters.get("argv", [])
        timeout = step.parameters.get("timeout_seconds", 60)
        if (step.kind != "script" or step.binding_id != "script_run"
                or not isinstance(source_path, str) or Path(source_path).name not in refs
                or not isinstance(inputs, Mapping) or len(inputs) != 1
                or not all(isinstance(key, str) and key for key in inputs)
                or not isinstance(argv, (list, tuple))
                or any(not isinstance(item, str) for item in argv)
                or type(timeout) is not int or not 1 <= timeout <= 120):
            raise ValueError("EVALUATION_FROZEN_INPUT_MAPPING_INCOMPLETE")


def _restore_environment(
    backend: SandboxBackend, storage_root: Path, digest: str, lock: bytes,
) -> EnvironmentRecord:
    restore = getattr(backend, "restore_environment", None)
    if not callable(restore) or not storage_root.is_dir() or storage_root.is_symlink():
        raise ValueError("EVALUATION_ENVIRONMENT_UNAVAILABLE")
    for directory in sorted(storage_root.iterdir()):
        if not directory.is_dir() or directory.is_symlink() or not directory.name.startswith("env_"):
            continue
        try:
            environment = restore(directory.name)
            lock_path = storage_root / environment.lock_ref
            if (environment.status == "READY" and environment.environment_sha256 == digest
                    and lock_path.is_file() and not lock_path.is_symlink()
                    and lock_path.read_bytes() == lock):
                return environment
        except (OSError, ValueError, TypeError):
            continue
    raise ValueError("EVALUATION_ENVIRONMENT_UNAVAILABLE")
