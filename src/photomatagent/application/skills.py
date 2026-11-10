"""Immutable local method candidates, deterministic checking and narrow replay.

Local trialability is deliberately independent of QD publication maturity.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from photomatagent.application.errors import ServiceError
from photomatagent.application.plugin_contracts import PluginInterface, validate_interface
from photomatagent.capability_acquisition.models import EnvironmentRecord, ExecutionRecord, ExecutionRequest, InstallPlan
from photomatagent.capability_acquisition.sandbox import SandboxBackend, SandboxService, _sha256_file, _safe_output_file, _contains_symlink
from photomatagent.workspace import Workspace
from photomatagent.errors import ToolExecutionError
from photomatagent.application.research_checkers import check_rc_network, check_report_tables
from photomatagent.application.optical_reference import check_fresnel_film
from photomatagent.persistence.atomic import atomic_write


class CheckerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    checker_id: Literal["json_numeric_sum", "json_numeric_bounds", "mwir_noise_budget", "rc_network", "report_tables", "fresnel_film"] = "json_numeric_sum"
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
    interface: PluginInterface | None = None
    parent_skill_id: str | None = Field(default=None, pattern=r"^skill_[0-9a-f]{32}$")


class SkillReplay(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str
    scope: str = Field(min_length=1, max_length=200)
    input_refs: dict[str, str] | None = None
    argv: list[str] | None = Field(default=None, max_length=32)


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
        value: Any = json.loads(output.read_text())
        for part in spec.field.split("."):
            if not isinstance(value, dict) or not part or part not in value:
                raise KeyError(part)
            value = value[part]
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


def check_mwir_noise_budget(spec: CheckerSpec, inputs: dict[str, Path], output: Path) -> dict[str, Any]:
    """Recompute a narrow MWIR area, NEP and D* contract from current input."""
    limitations = [
        "Checks unit conversion and internal noise-budget arithmetic only; it does not validate measured device performance or a material hypothesis.",
        "scientific_status remains UNKNOWN because one noise-budget record cannot establish a device goal.",
    ]
    try:
        source = json.loads(inputs[spec.input_key].read_text())
        result = json.loads(output.read_text())
        if not isinstance(source, dict) or not isinstance(result, dict):
            raise ValueError("MWIR_CONTRACT_INVALID")
        required = {
            "area_um2", "responsivity_a_w", "noise_asd_a_sqrt_hz", "frequency_hz",
            "wavelength_um", "temperature_k", "bias_v", "noise_basis", "measurement_kind",
        }
        if not required.issubset(source):
            raise ValueError("MWIR_CONTRACT_INVALID")
        numeric = {key: source[key] for key in required - {"noise_basis", "measurement_kind"}}
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in numeric.values()):
            raise ValueError("MWIR_CONTRACT_INVALID")
        if (numeric["area_um2"] <= 0 or numeric["responsivity_a_w"] <= 0
                or numeric["noise_asd_a_sqrt_hz"] <= 0 or numeric["frequency_hz"] <= 0
                or numeric["wavelength_um"] <= 0 or numeric["temperature_k"] <= 0):
            raise ValueError("MWIR_CONTRACT_INVALID")
        noise_basis = source["noise_basis"]
        measurement_kind = source["measurement_kind"]
        if noise_basis not in {"measured_total", "estimated_total", "shot_only"} or measurement_kind not in {
            "synthetic_validation", "experimental", "design_assumption",
        }:
            raise ValueError("MWIR_CONTRACT_INVALID")

        area_cm2 = numeric["area_um2"] * 1e-8
        nep = numeric["noise_asd_a_sqrt_hz"] / numeric["responsivity_a_w"]
        detectivity = math.sqrt(area_cm2) / nep
        if not all(math.isfinite(value) and value > 0 for value in (area_cm2, nep, detectivity)):
            raise ValueError("MWIR_RESULT_OUT_OF_RANGE")
        assessment = ("INSUFFICIENT_NOISE_EVIDENCE" if noise_basis == "shot_only" else
                      "MEASURED" if measurement_kind == "experimental" and noise_basis == "measured_total" else
                      "ESTIMATED")
        expected = {
            "area_cm2": area_cm2,
            "nep_w_sqrt_hz": nep,
            "detectivity_jones": detectivity,
        }
        for key, value in expected.items():
            actual = result.get(key)
            if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual):
                raise ValueError("MWIR_RESULT_INVALID")
        labels_match = (result.get("assessment") == assessment
                        and result.get("scientific_status") == "UNKNOWN")
        values_match = all(math.isclose(result[key], value, rel_tol=1e-9, abs_tol=1e-300)
                           for key, value in expected.items())
        return {
            "status": "PASS" if values_match and labels_match else "FAIL",
            "checker_id": spec.checker_id,
            "method": "recomputed MWIR area conversion, NEP and specific detectivity",
            "assessment": assessment,
            "scientific_status": "UNKNOWN",
            "input_sha256": _sha256_file(inputs[spec.input_key]),
            "output_sha256": _sha256_file(output),
            "limitations": limitations,
        }
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        return {"status": "UNKNOWN", "checker_id": spec.checker_id, "gap": "MWIR_CONTRACT_OR_EVIDENCE_INVALID",
                "scientific_status": "UNKNOWN", "limitations": limitations}


CHECKERS: dict[str, Callable[[CheckerSpec, dict[str, Path], Path], dict[str, Any]]] = {
    "json_numeric_sum": check_numeric_sum,
    "json_numeric_bounds": check_numeric_bounds,
    "mwir_noise_budget": check_mwir_noise_budget,
    "rc_network": check_rc_network,
    "report_tables": check_report_tables,
    "fresnel_film": check_fresnel_film,
}


def checker_digest(checker_id: str) -> str:
    """Bind the host reference and its module helpers, not a candidate's test code."""
    module_file = inspect.getsourcefile(CHECKERS[checker_id])
    if module_file is None:
        raise ServiceError("SKILL_CHECKER_UNAVAILABLE", "可信检查器源码不可用。")
    return _sha256_file(Path(module_file))


class LocalSkillService:
    def __init__(self, workspace: Path | str, *, backend_factory: Callable[[Path, str], SandboxBackend] | None = None) -> None:
        self.workspace = Workspace(workspace)
        self.root = self.workspace.resolve(".photomatagent/local-skills", must_exist=False)
        self._custom_backend_factory = backend_factory is not None
        self.backend_factory = backend_factory or self._backend

    def _restore_frozen_podman_environment(
        self, data: dict[str, Any], plan: InstallPlan, sandbox: SandboxService,
    ) -> EnvironmentRecord:
        """Bind replay to the exact persisted source environment, never reinstall it."""
        config_path = self.workspace.root / ".photomatagent" / "qd.json"
        if (not config_path.is_file() or config_path.is_symlink()
                or not self.workspace.contains(config_path.resolve())):
            raise ServiceError("SKILL_ENVIRONMENT_UNAVAILABLE", "源容器环境存储未配置。")
        try:
            from photomatagent.scientific.evolution.qd.config import QDConfig
            from photomatagent.capability_acquisition.container_backend import PodmanSandboxBackend
            from photomatagent.capability_acquisition.sandbox import load_persisted_install_plan

            config = QDConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
            if not config.enabled or config.sandbox != "podman":
                raise ValueError("PODMAN_STORAGE_NOT_CONFIGURED")
            storage = self.workspace.resolve(config.environment_storage, must_exist=True)
            if not storage.is_dir() or storage.is_symlink():
                raise ValueError("PODMAN_STORAGE_NOT_CONFIGURED")
            backend = PodmanSandboxBackend(storage)
            environment_id = data["request"]["environment_id"]
            try:
                environment = backend.restore_environment(environment_id)
            except ValueError as exc:
                raise ValueError("SKILL_ENVIRONMENT_CHANGED") from exc
            persisted_plan = load_persisted_install_plan(self.workspace.root, plan.plan_sha256)
            if (persisted_plan is None or persisted_plan.plan_sha256 != plan.plan_sha256
                    or environment.plan_sha256 != plan.plan_sha256
                    or environment.environment_sha256 != data["environment_sha256"]
                    or environment.environment_id != environment_id):
                raise ValueError("SKILL_ENVIRONMENT_CHANGED")
            sandbox.backend = backend
            sandbox.register_environment(environment, plan=plan)
            return environment
        except ServiceError:
            raise
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            code = "SKILL_ENVIRONMENT_CHANGED" if str(exc) == "SKILL_ENVIRONMENT_CHANGED" else "SKILL_ENVIRONMENT_UNAVAILABLE"
            raise ServiceError(code, "冻结源容器环境无法通过摘要与计划校验。") from exc

    @staticmethod
    def _backend(root: Path, kind: str) -> SandboxBackend:
        if kind == "local_dev":
            from photomatagent.capability_acquisition.local_backend import LocalDevBackend
            return LocalDevBackend(root)
        from photomatagent.capability_acquisition.container_backend import PodmanSandboxBackend
        return PodmanSandboxBackend(root)

    def _source_tree_backend(self, sandbox: SandboxService, environment: EnvironmentRecord) -> SandboxBackend | None:
        return sandbox.backend

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
            if data.get("checker_sha256") and checker_digest(data["checker"]["checker_id"]) != data["checker_sha256"]:
                raise ValueError("CHECKER_CHANGED")
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
                if data.get("reproduction") and not self._reproduction_qualified(data):
                    status = "CANDIDATE"
                if (directory / "disabled").exists():
                    status = "DISABLED"
                request = ExecutionRequest.model_validate(data["request"])
                results.append(data | {"status": status, "uses": uses, "input_keys": list(request.replay_input_keys)})
            except (ServiceError, ValueError):
                results.append({"skill_id": directory.name, "status": "REVALIDATION_REQUIRED", "uses": self.uses(directory.name)})
        return results

    def create(self, payload: SkillCreate, *, capture_key: str | None = None) -> dict[str, Any]:
        parent = self.read(payload.parent_skill_id) if payload.parent_skill_id else None
        if parent and parent.get("logical_name", parent["name"].rsplit("-", 1)[0]) != payload.name:
            raise ServiceError("SKILL_LINEAGE_MISMATCH", "新版本名称必须与父版本相同。")
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
        reproduction = None
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
            if isinstance(source, OrdinaryRunSource):
                from photomatagent.application.reproduction import ReproductionService

                reproduction = ReproductionService(
                    self.workspace, None, scratch=f"tmp/workbench/{source.task_id}",
                ).qualification_source(source.execution_id)
        except (OSError, ValueError, KeyError) as exc:
            raise ServiceError("SKILL_SOURCE_INVALID", "缺少真实执行、冻结依赖或源绑定。") from exc
        skill_id = "skill_" + hashlib.sha256(capture_key.encode()).hexdigest()[:32] if capture_key else f"skill_{uuid4().hex}"
        if capture_key and self._directory(skill_id).exists():
            return self.read(skill_id) | {"status": "CANDIDATE", "uses": self.uses(skill_id),
                                          "input_keys": list(request.replay_input_keys)}
        directory = self._directory(skill_id)
        directory.mkdir(parents=True)
        files = {}
        for name in ["script.py", *[f"input-{key}" for key in request.input_refs]]:
            content = (root / name).read_bytes()
            (directory / name).write_bytes(content)
            files[name] = hashlib.sha256(content).hexdigest()
        skill_text = f"---\nname: {payload.name}-{skill_id[-8:]}\ndescription: {payload.description}\n---\nLocal method, scope: {payload.scope}. Execute with skill_run, skill_id={skill_id}.\nInputs: {', '.join(request.replay_input_keys)}. Outputs: {', '.join(a.path for a in record.artifacts)}.\n"
        (directory / "SKILL.md").write_text(skill_text)
        files["SKILL.md"] = hashlib.sha256(skill_text.encode()).hexdigest()
        inputs = {key: directory / f"input-{key}" for key in request.input_refs}
        evidence = self._check(payload.checker, inputs, source_workspace / "tmp/capability-executions" / source.execution_id, record)
        data = {"skill_id": skill_id, "name": f"{payload.name}-{skill_id[-8:]}", "description": payload.description,
                "interface": payload.interface.model_dump(mode="json") if payload.interface else None,
                "logical_name": payload.name, "plugin_id": parent.get("plugin_id", parent["skill_id"]) if parent else skill_id,
                "version": skill_id, "parent_skill_id": payload.parent_skill_id,
                "parent_contract_sha256": parent["contract_sha256"] if parent else None,
                "scope": payload.scope, "source": source.model_dump(), "checker": payload.checker.model_dump() if payload.checker else None,
                "checker_sha256": checker_digest(payload.checker.checker_id) if payload.checker else None,
                "source_evidence": evidence, "execution": record.model_copy(update={"stdout": "", "stderr": ""}).model_dump(mode="json"),
                "request": request.model_dump(mode="json"), "plan": provenance["plan"], "files": files,
                "input_sha256": provenance["input_sha256"], "environment_sha256": record.environment_sha256}
        if reproduction:
            data["reproduction"] = reproduction
        data["contract_sha256"] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        (directory / "candidate.json").write_text(json.dumps(data))
        return data | {"status": "CANDIDATE", "uses": [], "input_keys": list(request.replay_input_keys)}

    def record_reproduction_qualification(self, skill_id: str, trial_id: str) -> None:
        path = self._directory(skill_id) / "qualification.json"
        if not path.exists():
            atomic_write(path, json.dumps({"trial_id": trial_id}).encode())

    def _reproduction_qualified(self, data: dict[str, Any]) -> bool:
        from photomatagent.scientific.evolution.qd.store import QDStore, QDStoreError

        try:
            path = self._directory(data["skill_id"]) / "qualification.json"
            trial_id = json.loads(path.read_bytes())["trial_id"]

            def read_only(event: str, genome: str) -> Any:
                raise ValueError("READ_ONLY")

            store = QDStore(self.workspace, trial_factory=read_only)
            trial = store.load_trial(trial_id)
            validation = store.load_validation(trial_id, "v_" + trial_id[2:])
            return (
                trial.status == "VALIDATED" and trial.genome_id == "g_" + data["contract_sha256"][:32]
                and validation.qualification == "PASS" and bool(validation.outcomes)
                and all(outcome.verdict == "PASS" for outcome in validation.outcomes)
                and bool(trial.output_refs)
                and all(_sha256_file(self.workspace.resolve(ref.path)) == ref.sha256 for ref in trial.output_refs)
            )
        except (OSError, ValueError, KeyError, TypeError, ToolExecutionError, QDStoreError):
            return False

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

    def enable(self, skill_id: str) -> dict[str, str]:
        """Restore an already checked immutable version, without inheriting parent tests."""
        data = self.read(skill_id)
        if data["source_evidence"].get("status") != "PASS" or not any(
            use.get("code") == "OK" and use.get("checker", {}).get("status") == "PASS"
            and use.get("contract_sha256") == data["contract_sha256"]
            and use.get("execution") and use["execution"].get("script_sha256") == data["files"]["script.py"]
            for use in self.uses(skill_id)
        ):
            raise ServiceError("SKILL_REPLAY_REQUIRED", "此版本需要自身的成功回放，不能继承父版本的验证。")
        (self._directory(skill_id) / "disabled").unlink(missing_ok=True)
        item = next(item for item in self.list_skills() if item["skill_id"] == skill_id)
        return {"skill_id": skill_id, "status": item["status"]}

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
                                  "script_path": request.script_path, "argv": list(request.argv),
                                  "input_keys": list(request.replay_input_keys),
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
            source_request = ExecutionRequest.model_validate(data["request"])
            layout = source_request.source_tree
            static_refs = {key: self.workspace.relative(directory / f"input-{key}") for key in layout.inputs} if layout else {}
            input_keys = set(source_request.replay_input_keys)
            if refs is None:
                if payload.task_id != data["source"]["task_id"]:
                    raise ServiceError("SKILL_NEW_INPUT_REQUIRED", "跨任务使用必须提供当前任务输入。")
                refs = {key: self.workspace.relative(directory / f"input-{key}") for key in input_keys}
            if set(refs) != input_keys:
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
            if data.get("interface"):
                interface = PluginInterface.model_validate(data["interface"])
                argv = payload.argv if payload.argv is not None else data["request"]["argv"]
                if argv not in interface.argv_choices:
                    raise ServiceError("SKILL_CONFIGURATION_REJECTED", "配置不属于冻结的 argv 白名单。")
                submitted = {key: self.workspace.resolve(path).read_bytes() for key, path in refs.items()}
                use["presented_input_sha256"] = {key: hashlib.sha256(raw).hexdigest() for key, raw in submitted.items()}
                documents = {key: json.loads(raw) if key != "report" else raw.decode() for key, raw in submitted.items()}
                try:
                    validate_interface(data["interface"], documents)
                except Exception as exc:
                    raise ServiceError("SKILL_SCHEMA_REJECTED", "输入不满足冻结接口。") from exc
            raw_plan = dict(data["plan"])
            raw_plan["expires_at"] = datetime.now(timezone.utc) + timedelta(days=1)
            plan = InstallPlan.model_validate(raw_plan)
            restore_source_environment = plan.execution_backend == "podman" and not self._custom_backend_factory
            backend = None if restore_source_environment else self.backend_factory(directory / "environment", plan.execution_backend)
            script_path = self.workspace.relative(directory / "script.py")
            bindings = {**refs, **static_refs}
            sandbox = SandboxService(workspace=self.workspace.root, trial_id=use["use_id"],
                                     genome_sha256=data["contract_sha256"], scripts={script_path: data["files"]["script.py"]},
                                     input_bindings=bindings, backend=backend)
            if restore_source_environment:
                installed_environment = self._restore_frozen_podman_environment(data, plan, sandbox)
            else:
                installed = await sandbox.install(plan)
                if installed.code != "OK" or installed.environment is None:
                    raise ServiceError("SKILL_DEPENDENCIES_UNAVAILABLE", f"冻结依赖不可用：{installed.code}")
                installed_environment = installed.environment
            if installed_environment.environment_sha256 != data["environment_sha256"] and plan.execution_backend == "podman":
                raise ServiceError("SKILL_ENVIRONMENT_CHANGED", "环境摘要与源合同不同。")
            if layout:
                sandbox.backend = self._source_tree_backend(sandbox, installed_environment)
            hashes = {key: _sha256_file(self.workspace.resolve(path)) for key, path in refs.items()} if layout else None
            if hashes is not None:
                hashes.update({key: data["files"][f"input-{key}"] for key in static_refs})
            result = await sandbox.run(script_path=script_path, environment_id=installed_environment.environment_id,
                                       input_refs=bindings, argv=payload.argv if payload.argv is not None else data["request"]["argv"],
                                       source_tree=layout, expected_input_sha256=hashes)
            use["code"] = result.code
            use["execution"] = result.record.model_dump(mode="json") if result.record else None
            if result.record and result.code == "OK":
                inputs, executed_digests = sandbox.execution_inputs(result.record.execution_id)
                use["input_sha256"] = executed_digests
                if payload.task_id != data["source"]["task_id"] and executed_digests == data["input_sha256"]:
                    raise ServiceError("SKILL_NEW_INPUT_REQUIRED", "跨任务验证需要不同的实际执行输入。")
                if {a.path for a in result.record.artifacts} != {a["path"] for a in data["execution"]["artifacts"]}:
                    raise ServiceError("SKILL_OUTPUT_CONTRACT_MISMATCH", "产物路径与冻结输出合同不符。")
                spec = CheckerSpec.model_validate(data["checker"])
                if data.get("interface"):
                    try:
                        validate_interface(data["interface"], documents, json.loads((self.workspace.root / "tmp/capability-executions" / result.record.execution_id / spec.output_path).read_text()))
                    except Exception as exc:
                        raise ServiceError("SKILL_SCHEMA_REJECTED", "输出不满足冻结接口。") from exc
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
