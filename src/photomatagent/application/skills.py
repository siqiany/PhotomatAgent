"""Immutable local method candidates, deterministic checking and narrow replay.

Local trialability is deliberately independent of QD publication maturity.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from photomatagent.application.errors import ServiceError
from photomatagent.capability_acquisition.models import EnvironmentRecord, ExecutionRecord, ExecutionRequest, InstallPlan
from photomatagent.capability_acquisition.sandbox import SandboxBackend, SandboxService, _sha256_file, _safe_output_file, _contains_symlink
from photomatagent.workspace import Workspace
from photomatagent.errors import ToolExecutionError


class CheckerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    checker_id: Literal["json_numeric_sum", "json_numeric_bounds"] = "json_numeric_sum"
    input_key: str = Field(default="data", pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    output_path: str = Field(min_length=1, max_length=500)
    field: str = Field(min_length=1, max_length=100)
    minimum: float | None = Field(default=None, allow_inf_nan=False)
    maximum: float | None = Field(default=None, allow_inf_nan=False)
    units: str = Field(default="unspecified", max_length=100)
    tolerance: float = Field(default=1e-9, ge=0, le=0.001, allow_inf_nan=False)


class OrdinaryRunSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["ordinary_run"] = "ordinary_run"
    task_id: str
    run_id: str
    execution_id: str = Field(pattern=r"^exec_[0-9a-f]{32}$")


class QDTrialSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["qd_trial"] = "qd_trial"
    task_id: str
    trial_id: str
    execution_id: str = Field(pattern=r"^exec_[0-9a-f]{32}$")


class SkillCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[a-z][a-z0-9-]{1,60}$")
    description: str = Field(min_length=1, max_length=500)
    scope: str = Field(min_length=1, max_length=200)
    source: OrdinaryRunSource | QDTrialSource = Field(discriminator="kind")
    checker: CheckerSpec | None = None


class SkillReplay(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str
    scope: str = Field(min_length=1, max_length=200)
    input_refs: dict[str, str] | None = None
    argv: list[str] | None = None


def check_numeric_sum(spec: CheckerSpec, inputs: dict[str, Path], output: Path) -> dict[str, Any]:
    """Trusted relation checker: current input numeric array -> JSON sum field."""
    try:
        values = json.loads(inputs[spec.input_key].read_text())
        result = json.loads(output.read_text())[spec.field]
        if (not isinstance(values, list) or not values or len(values) > 100_000
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)
                or isinstance(result, bool) or not isinstance(result, (int, float)) or not math.isfinite(result)):
            return {"status": "UNKNOWN", "checker_id": spec.checker_id, "gap": "NUMERIC_CONTRACT_INVALID"}
        passed = math.isclose(result, math.fsum(values), rel_tol=0, abs_tol=spec.tolerance)
        return {"status": "PASS" if passed else "FAIL", "checker_id": spec.checker_id,
                "method": "numeric relation against current input", "units": "input units",
                "input_sha256": _sha256_file(inputs[spec.input_key]), "output_sha256": _sha256_file(output),
                "limitations": ["Checks numeric sum only; does not validate a material hypothesis."]}
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        return {"status": "UNKNOWN", "checker_id": spec.checker_id, "gap": "CHECKER_EVIDENCE_MISSING"}


def check_numeric_bounds(spec: CheckerSpec, inputs: dict[str, Path], output: Path) -> dict[str, Any]:
    try:
        value = json.loads(output.read_text())[spec.field]
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or (spec.minimum is None and spec.maximum is None)
                or (spec.minimum is not None and spec.maximum is not None and spec.minimum > spec.maximum)):
            raise ValueError("CHECKER_CONTRACT_INVALID")
        passed = (spec.minimum is None or value >= spec.minimum) and (spec.maximum is None or value <= spec.maximum)
        return {"status": "PASS" if passed else "FAIL", "checker_id": spec.checker_id,
                "method": "frozen numeric bounds", "units": spec.units, "output_sha256": _sha256_file(output),
                "limitations": ["Validates a numeric field against host-selected bounds only; scientific interpretation requires separate evidence."]}
    except (OSError, ValueError, TypeError, KeyError):
        return {"status": "UNKNOWN", "checker_id": spec.checker_id, "gap": "CHECKER_EVIDENCE_MISSING"}


CHECKERS: dict[str, Callable[[CheckerSpec, dict[str, Path], Path], dict[str, Any]]] = {
    "json_numeric_sum": check_numeric_sum,
    "json_numeric_bounds": check_numeric_bounds,
}


class LocalSkillService:
    def __init__(self, workspace: Path | str, *, backend_factory: Callable[[Path, str], SandboxBackend] | None = None) -> None:
        self.workspace = Workspace(workspace)
        self.root = self.workspace.resolve(".photomatagent/local-skills", must_exist=False)
        self.backend_factory = backend_factory or self._backend

    @staticmethod
    def _backend(root: Path, kind: str) -> SandboxBackend:
        if kind == "local_dev":
            from photomatagent.capability_acquisition.local_backend import LocalDevBackend
            return LocalDevBackend(root)
        from photomatagent.capability_acquisition.container_backend import PodmanSandboxBackend
        return PodmanSandboxBackend(root)

    def _directory(self, skill_id: str) -> Path:
        if not re.fullmatch(r"skill_[0-9a-f]{32}", skill_id):
            raise ServiceError("SKILL_NOT_FOUND", "技能不存在。", status_code=404)
        return self.workspace.resolve(f".photomatagent/local-skills/{skill_id}", must_exist=False)

    def read(self, skill_id: str) -> dict[str, Any]:
        try:
            directory = self._directory(skill_id)
            data: dict[str, Any] = json.loads((directory / "candidate.json").read_text())
            frozen = dict(data)
            digest = frozen.pop("contract_sha256")
            if hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest() != digest:
                raise ValueError("CONTRACT_CHANGED")
            for relative, expected in data["files"].items():
                if _sha256_file(self.workspace.resolve(f".photomatagent/local-skills/{skill_id}/{relative}")) != expected:
                    raise ValueError("SCRIPT_OR_INPUT_CHANGED")
            return data
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ServiceError("SKILL_REVALIDATION_REQUIRED", "技能合同或文件已改变，需要新建版本并重新回放。") from exc

    def _append_use(self, directory: Path, use: dict[str, Any]) -> None:
        use["use_sha256"] = hashlib.sha256(json.dumps(use, sort_keys=True).encode()).hexdigest()
        with (directory / "uses.jsonl").open("a") as stream:
            stream.write(json.dumps(use) + "\n")

    def uses(self, skill_id: str) -> list[dict[str, Any]]:
        path = self._directory(skill_id) / "uses.jsonl"
        records = []
        for line in path.read_text().splitlines() if path.exists() else []:
            try:
                data = json.loads(line)
                if not isinstance(data, dict) or "checker" not in data:
                    raise ValueError("USE_RECORD_INVALID")
            except (ValueError, TypeError):
                records.append({"code": "SKILL_USE_CHANGED", "checker": {"status": "UNKNOWN", "gap": "USE_RECORD_INVALID"}})
                continue
            digest = data.pop("use_sha256", None)
            if digest != hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest():
                data["code"] = "SKILL_USE_CHANGED"
                data["checker"] = {"status": "UNKNOWN", "gap": "USE_RECORD_CHANGED"}
            records.append(data)
        return records

    def list_skills(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        results = []
        for directory in sorted(self.root.iterdir()):
            if directory.is_symlink() or not directory.is_dir() or not directory.name.startswith("skill_"):
                continue
            try:
                data = self.read(directory.name)
                uses = self.uses(directory.name)
                passes = [u for u in uses if u["checker"].get("status") == "PASS" and u["code"] == "OK"
                          and u["contract_sha256"] == data["contract_sha256"] and u.get("execution")
                          and u["execution"]["script_sha256"] == data["files"]["script.py"]]
                status = "REUSED" if any(u["task_id"] != data["source"]["task_id"] for u in passes) else "TRIALABLE" if passes else "CANDIDATE"
                if (directory / "disabled").exists():
                    status = "DISABLED"
                results.append(data | {"status": status, "uses": uses})
            except ServiceError:
                results.append({"skill_id": directory.name, "status": "REVALIDATION_REQUIRED", "uses": self.uses(directory.name)})
        return results

    def create(self, payload: SkillCreate) -> dict[str, Any]:
        source = payload.source
        source_workspace = self.workspace.root
        if isinstance(source, QDTrialSource):
            from photomatagent.scientific.evolution.qd.store import QDStore, QDStoreError
            try:
                store = QDStore(self.workspace, trial_factory=lambda event, genome: (_ for _ in ()).throw(ValueError("READ_ONLY")))
                trial = store.load_trial(source.trial_id)
                if trial.status not in {"COLLECTED", "VALIDATED"}:
                    raise ValueError("QD_SOURCE_NOT_COLLECTED")
                source_workspace = self.workspace.resolve(trial.attempt_dir)
                genome = store.load_genome(trial.genome_id)
            except (ValueError, OSError, QDStoreError, ToolExecutionError) as exc:
                raise ServiceError("SKILL_QD_SOURCE_INVALID", "QD 来源缺少可信 trial 记录。") from exc
        root = Workspace(source_workspace).resolve(f".photomatagent/executions/{source.execution_id}", must_exist=False)
        try:
            provenance = json.loads((root / "request.json").read_text())
            record = ExecutionRecord.model_validate_json((root / "record.json").read_text())
            request = ExecutionRequest.model_validate(provenance["request"])
            raw_plan = dict(provenance["plan"] or {})
            raw_plan["expires_at"] = datetime.now(timezone.utc) + timedelta(days=1)
            source_plan = InstallPlan.model_validate(raw_plan)
            source_environment = EnvironmentRecord.model_validate(provenance["environment"])
            if (request.execution_id != record.execution_id or request.script_digest != record.script_sha256
                    or source_environment.environment_sha256 != record.environment_sha256
                    or source_environment.status != "READY" or source_environment.environment_id != request.environment_id
                    or source_environment.plan_sha256 != source_plan.plan_sha256):
                raise ValueError("SOURCE_PROVENANCE_BINDING_MISMATCH")
            expected_run = source.run_id if isinstance(source, OrdinaryRunSource) else source.trial_id
            if request.trial_id != expected_run or record.execution_id != source.execution_id or Path(request.workspace_root) != source_workspace:
                raise ValueError("SOURCE_BINDING_MISMATCH")
            if isinstance(source, QDTrialSource) and (request.genome_sha256 != genome.content_sha256 or record.environment_sha256 != trial.environment_sha256):
                raise ValueError("QD_SOURCE_BINDING_MISMATCH")
            if isinstance(source, OrdinaryRunSource) and not request.script_path.startswith(f"tmp/workbench/{source.task_id}/"):
                raise ValueError("SOURCE_TASK_MISMATCH")
            if record.state != "COMPLETED" or record.exit_code != 0 or record.timed_out:
                raise ValueError("SOURCE_EXECUTION_NOT_SUCCESSFUL")
            if request.input_sha256 != provenance["input_sha256"] or set(request.input_refs) != set(request.input_sha256):
                raise ValueError("SOURCE_INPUT_BINDING_MISMATCH")
            for key, digest in provenance["input_sha256"].items():
                if _sha256_file(root / f"input-{key}") != digest:
                    raise ValueError("SOURCE_INPUT_CHANGED")
            if _sha256_file(root / "script.py") != record.script_sha256:
                raise ValueError("SOURCE_SCRIPT_CHANGED")
            if provenance["plan"] is None:
                raise ValueError("FROZEN_DEPENDENCIES_MISSING")
        except (OSError, ValueError, KeyError) as exc:
            raise ServiceError("SKILL_SOURCE_INVALID", "缺少真实执行、冻结依赖或源绑定。") from exc
        skill_id = f"skill_{uuid4().hex}"
        directory = self._directory(skill_id)
        directory.mkdir(parents=True)
        files = {}
        for name in ["script.py", *[f"input-{key}" for key in request.input_refs]]:
            content = (root / name).read_bytes()
            (directory / name).write_bytes(content)
            files[name] = hashlib.sha256(content).hexdigest()
        skill_text = f"---\nname: {payload.name}-{skill_id[-8:]}\ndescription: {payload.description}\n---\nLocal method, scope: {payload.scope}. Execute with skill_run, skill_id={skill_id}.\nInputs: {', '.join(request.input_refs)}. Outputs: {', '.join(a.path for a in record.artifacts)}.\n"
        (directory / "SKILL.md").write_text(skill_text)
        files["SKILL.md"] = hashlib.sha256(skill_text.encode()).hexdigest()
        inputs = {key: directory / f"input-{key}" for key in request.input_refs}
        evidence = self._check(payload.checker, inputs, source_workspace / "tmp/capability-executions" / source.execution_id, record)
        data = {"skill_id": skill_id, "name": f"{payload.name}-{skill_id[-8:]}", "description": payload.description,
                "scope": payload.scope, "source": source.model_dump(), "checker": payload.checker.model_dump() if payload.checker else None,
                "source_evidence": evidence, "execution": record.model_copy(update={"stdout": "", "stderr": ""}).model_dump(mode="json"),
                "request": request.model_dump(mode="json"), "plan": provenance["plan"], "files": files,
                "input_sha256": provenance["input_sha256"], "environment_sha256": record.environment_sha256}
        data["contract_sha256"] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        (directory / "candidate.json").write_text(json.dumps(data))
        return data | {"status": "CANDIDATE", "uses": []}

    @staticmethod
    def _check(spec: CheckerSpec | None, inputs: dict[str, Path], output_root: Path, record: ExecutionRecord | None) -> dict[str, Any]:
        if spec is None or record is None or record.state != "COMPLETED" or record.exit_code != 0 or record.timed_out:
            return {"status": "UNKNOWN", "gap": "CHECKER_NOT_SELECTED_OR_EXECUTION_MISSING"}
        artifact = next((a for a in record.artifacts if a.path == spec.output_path), None)
        output = _safe_output_file(output_root, spec.output_path)
        if artifact is None or output is None or _sha256_file(output) != artifact.sha256:
            return {"status": "UNKNOWN", "gap": "CHECKER_ARTIFACT_MISSING_OR_CHANGED"}
        return CHECKERS[spec.checker_id](spec, inputs, output)

    def disable(self, skill_id: str) -> dict[str, str]:
        self.read(skill_id)
        (self._directory(skill_id) / "disabled").touch()
        return {"skill_id": skill_id, "status": "DISABLED"}

    def sources(self) -> list[dict[str, Any]]:
        root = self.workspace.resolve(".photomatagent/executions", must_exist=False)
        items: list[dict[str, Any]] = []
        if not root.exists():
            return items
        for directory in sorted(root.iterdir()):
            if directory.is_symlink() or not re.fullmatch(r"exec_[0-9a-f]{32}", directory.name):
                continue
            try:
                data = json.loads((directory / "request.json").read_text())
                record = ExecutionRecord.model_validate_json((directory / "record.json").read_text())
                request = ExecutionRequest.model_validate(data["request"])
                parts = Path(request.script_path).parts
                if len(parts) >= 4 and parts[:2] == ("tmp", "workbench"):
                    items.append({"execution_id": record.execution_id, "task_id": parts[2], "run_id": request.trial_id,
                                  "script_path": request.script_path, "argv": list(request.argv), "input_keys": list(request.input_refs),
                                  "artifacts": [a.path for a in record.artifacts], "state": record.state})
            except (OSError, ValueError, KeyError):
                continue
        return items

    async def replay(self, skill_id: str, payload: SkillReplay, *, model_requested: bool = False, run_id: str | None = None) -> dict[str, Any]:
        if not re.fullmatch(r"task_[0-9a-f]{32}", payload.task_id):
            raise ServiceError("INVALID_TASK_ID", "任务 ID 无效。")
        data = self.read(skill_id)
        directory = self._directory(skill_id)
        if (directory / "disabled").exists():
            raise ServiceError("SKILL_DISABLED", "该技能已停用。")
        if model_requested and not any(s["skill_id"] == skill_id and s["status"] in {"TRIALABLE", "REUSED"} for s in self.list_skills()):
            raise ServiceError("SKILL_NOT_TRIALABLE", "候选必须先通过独立回放检查。")
        use: dict[str, Any] = {"use_id": f"use_{uuid4().hex}", "skill_id": skill_id, "task_id": payload.task_id,
                               "run_id": run_id, "contract_sha256": data["contract_sha256"], "code": "UNKNOWN",
                               "checker": {"status": "UNKNOWN"}, "execution": None, "artifacts": []}
        try:
            if payload.scope != data["scope"]:
                raise ServiceError("SKILL_SCOPE_MISMATCH", "任务适用范围与技能合同不同。")
            if data["source_evidence"]["status"] != "PASS":
                raise ServiceError("SKILL_SOURCE_UNCHECKED", "候选缺少通过的独立源检查证据。")
            refs = payload.input_refs
            if refs is None:
                if payload.task_id != data["source"]["task_id"]:
                    raise ServiceError("SKILL_NEW_INPUT_REQUIRED", "跨任务使用必须提供当前任务输入。")
                refs = {key: self.workspace.relative(directory / f"input-{key}") for key in data["request"]["input_refs"]}
            if set(refs) != set(data["request"]["input_refs"]):
                raise ServiceError("SKILL_INPUT_CONTRACT_MISMATCH", "输入名称与冻结合同不符。")
            if any(Path(path).is_absolute() or ".." in Path(path).parts for path in refs.values()):
                raise ServiceError("SKILL_INPUT_PATH_INVALID", "输入必须为工作区内相对路径。")
            allowed_prefixes = (f"tmp/workbench/{payload.task_id}/", f"user_output/{payload.task_id}/",
                                f".photomatagent/workbench/uploads/{payload.task_id}/")
            if payload.input_refs is not None and any(not path.startswith(allowed_prefixes) for path in refs.values()):
                raise ServiceError("SKILL_INPUT_SCOPE_VIOLATION", "输入不属于当前任务。")
            if any(_contains_symlink(self.workspace.root, self.workspace.root / path) for path in refs.values()):
                raise ServiceError("SKILL_INPUT_PATH_INVALID", "输入路径不能经过符号链接。")
            for path in refs.values():
                self.workspace.resolve(path)
            raw_plan = dict(data["plan"])
            raw_plan["expires_at"] = datetime.now(timezone.utc) + timedelta(days=1)
            plan = InstallPlan.model_validate(raw_plan)
            backend = self.backend_factory(directory / "environment", plan.execution_backend)
            script_path = self.workspace.relative(directory / "script.py")
            sandbox = SandboxService(workspace=self.workspace.root, trial_id=use["use_id"],
                                     genome_sha256=data["contract_sha256"], scripts={script_path: data["files"]["script.py"]},
                                     input_bindings=refs, backend=backend)
            installed = await sandbox.install(plan)
            if installed.code != "OK" or installed.environment is None:
                raise ServiceError("SKILL_DEPENDENCIES_UNAVAILABLE", f"冻结依赖不可用：{installed.code}")
            if installed.environment.environment_sha256 != data["environment_sha256"] and plan.execution_backend == "podman":
                raise ServiceError("SKILL_ENVIRONMENT_CHANGED", "环境摘要与源合同不同。")
            result = await sandbox.run(script_path=script_path, environment_id=installed.environment.environment_id,
                                       input_refs=refs, argv=payload.argv if payload.argv is not None else data["request"]["argv"])
            use["code"] = result.code
            use["execution"] = result.record.model_dump(mode="json") if result.record else None
            if result.record:
                inputs, executed_digests = sandbox.execution_inputs(result.record.execution_id)
                use["input_sha256"] = executed_digests
                if payload.task_id != data["source"]["task_id"] and executed_digests == data["input_sha256"]:
                    raise ServiceError("SKILL_NEW_INPUT_REQUIRED", "跨任务验证需要不同的实际执行输入。")
                if {a.path for a in result.record.artifacts} != {a["path"] for a in data["execution"]["artifacts"]}:
                    raise ServiceError("SKILL_OUTPUT_CONTRACT_MISMATCH", "产物路径与冻结输出合同不符。")
                spec = CheckerSpec.model_validate(data["checker"])
                use["checker"] = self._check(spec, inputs, self.workspace.root / "tmp/capability-executions" / result.record.execution_id, result.record)
                destination = f"user_output/{payload.task_id}/skill-uses/{use['use_id']}"
                output_root = self.workspace.resolve(destination, must_exist=False)
                output_root.mkdir(parents=True, exist_ok=True)
                output_workspace = Workspace(output_root)
                paths = sandbox.archive_artifacts(result.record.execution_id, destination=output_workspace,
                                                  declared_paths=frozenset(a.path for a in result.record.artifacts))
                use["artifacts"] = [f"{destination}/{path}" for path in paths]
        except asyncio.CancelledError:
            use["code"] = "SKILL_REPLAY_CANCELLED"
            use["checker"] = {"status": "UNKNOWN", "gap": "EXECUTION_CANCELLED"}
            self._append_use(directory, use)
            raise
        except (ServiceError, ValueError, OSError, ToolExecutionError) as exc:
            use["code"] = exc.code if isinstance(exc, ServiceError) else "SKILL_INPUT_OR_ENVIRONMENT_UNAVAILABLE"
            use["checker"] = {"status": "UNKNOWN", "gap": use["code"]}
        self._append_use(directory, use)
        return use
