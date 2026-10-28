"""Durable evaluation requests and host-owned execution for the Web workbench."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast
from photomatagent.capability_acquisition.sandbox import SandboxBackend

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from photomatagent.application.errors import ServiceError
from photomatagent.application.models import BudgetSpec, RunRef, RunRequest
from photomatagent.application.runner import RunManager
from photomatagent.application.store import (
    ActiveRunConflict,
    IdempotencyConflict,
    WorkbenchStore,
)
from photomatagent.scientific.evolution.qd.checkers import EvaluationCheckerSpec, check_reference
from photomatagent.scientific.evolution.qd.config import EvolutionBudget
from photomatagent.scientific.evolution.qd.evaluation import (
    EvaluationCase,
    EvaluationCondition,
    EvaluationReport,
    FrozenEvaluationManifestV2,
    FrozenSkillRef,
    dataset_identity_sha256,
)
from photomatagent.scientific.evolution.qd.models import ArtifactDigest
from photomatagent.workspace import Workspace

_SUPPORTED_CONDITIONS: tuple[EvaluationCondition, ...] = (
    "STATIC", "FEEDBACK_CHAIN", "RANDOM_CANDIDATE", "DIRECTED_MUTATION",
)
_UNSUPPORTED_CONDITIONS: dict[str, str] = {
    "FULL_QD_SKILLS": "需要配置真实 Podman 后端，以及摘要校验通过的冻结技能和已安装依赖环境。",
}
_MAX_CASES = 50
_MAX_SKILLS = 20


class EvaluationCaseSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9_.:-]+$")
    task_group_id: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9_.:-]+$")
    signal_family: str = Field(min_length=1, max_length=200)
    source_id: str = Field(min_length=1, max_length=200)
    input_artifact_id: str = Field(min_length=1, max_length=200)
    reference_artifact_id: str = Field(min_length=1, max_length=200)


class EvaluationSkillSelection(BaseModel):
    """An exact published QD version; local trial skills remain a separate source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_type: Literal["published_qd"]
    skill_id: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9_.:-]+$")
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EvaluationStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1, max_length=200)
    cases: tuple[EvaluationCaseSelection, ...] = Field(min_length=2, max_length=_MAX_CASES)
    conditions: tuple[EvaluationCondition, ...] = Field(min_length=1, max_length=5)
    checker: EvaluationCheckerSpec
    budget: EvolutionBudget = Field(default_factory=EvolutionBudget)
    seed: StrictInt = 0
    development_groups: tuple[str, ...] = ()
    frozen_qd_skills: tuple[EvaluationSkillSelection, ...] = Field(default=(), max_length=_MAX_SKILLS)

    @field_validator("conditions")
    @classmethod
    def condition_subset_is_unique(cls, values: tuple[EvaluationCondition, ...]) -> tuple[EvaluationCondition, ...]:
        if len(values) != len(set(values)):
            raise ValueError("EVALUATION_CONDITIONS_INVALID")
        return values

    @model_validator(mode="after")
    def finite_positive_budget(self) -> EvaluationStartRequest:
        if len({item.case_id for item in self.cases}) != len(self.cases):
            raise ValueError("EVALUATION_CASE_DUPLICATE")
        if len({(item.skill_id, item.bundle_sha256) for item in self.frozen_qd_skills}) != len(self.frozen_qd_skills):
            raise ValueError("EVALUATION_SKILL_DUPLICATE")
        if min(
            self.budget.max_trials,
            self.budget.max_model_calls,
            self.budget.max_model_tokens,
        ) < 1 or self.budget.max_wall_seconds <= 0:
            raise ValueError("EVALUATION_BUDGET_INVALID")
        if (self.budget.max_trials > 100 or self.budget.max_model_calls > 500
                or self.budget.max_model_tokens > 2_000_000
                or self.budget.max_wall_seconds > 7200
                or self.budget.max_proposals > 500):
            raise ValueError("EVALUATION_BUDGET_LIMIT_EXCEEDED")
        return self


class EvaluationStartResponse(BaseModel):
    evaluation_id: str
    task_id: str
    run_id: str
    status: str


class EvaluationStatusResponse(BaseModel):
    evaluation_id: str
    task_id: str
    run_id: str
    status: str
    manifest_sha256: str
    created_at: str
    updated_at: str
    report: dict[str, object] | None = None
    error_code: str | None = None


class EvaluationCapabilitiesResponse(BaseModel):
    conditions: list[dict[str, object]]
    checker_ids: list[str]
    frozen_skill_sources: list[str]
    frozen_skill_versions: list[dict[str, str]] = Field(default_factory=list)


class EvaluationService:
    """Reserves evaluations as ordinary durable host operations owned by RunManager."""

    def __init__(
        self,
        *,
        store: WorkbenchStore,
        manager: RunManager,
        workspace_root: Path | str,
        runner_factory: Callable[..., Any] | None = None,
        model_id_factory: Callable[[], str] | None = None,
        frozen_backend: SandboxBackend | None = None,
        environment_storage: Path | None = None,
    ) -> None:
        self.store = store
        self.manager = manager
        self.workspace = Workspace(Path(workspace_root).resolve())
        self.runner_factory = runner_factory
        self.model_id_factory = model_id_factory
        if frozen_backend is None and environment_storage is None:
            frozen_backend, environment_storage = _configured_frozen_backend(self.workspace)
        self.frozen_backend = frozen_backend
        self.environment_storage = environment_storage
        manager.register_host_operation("evaluation", self.execute)

    def capabilities(self) -> dict[str, object]:
        versions: list[dict[str, str]] = []
        try:
            from photomatagent.skills.published import PublishedSkillSource
            source = PublishedSkillSource(
                self.workspace.root / ".photomatagent" / "qd" / "published-skills"
            )
            versions = [
                {
                    "source_type": "published_qd",
                    "skill_id": skill_id,
                    "bundle_sha256": str(record["bundle_sha256"]),
                }
                for skill_id, record, _bundle, _files in source._active_bundles()
            ]
        except (OSError, ValueError, TypeError, KeyError):
            versions = []
        full_available = False
        if _production_frozen_backend(self.frozen_backend) and self.environment_storage is not None:
            assert self.frozen_backend is not None
            from photomatagent.scientific.evolution.qd.frozen_executor import preflight_frozen_skill
            for item in versions:
                try:
                    path = self.workspace.resolve(
                        f".photomatagent/qd/published-skills/{item['skill_id']}/{item['bundle_sha256']}",
                        must_exist=True,
                    )
                    preflight_frozen_skill(
                        path, item["bundle_sha256"], backend=self.frozen_backend,
                        storage_root=self.environment_storage,
                    )
                except (OSError, ValueError, TypeError):
                    continue
                full_available = True
                break
        return {
            "conditions": [
                {"id": condition, "available": True, "reason": None}
                for condition in _SUPPORTED_CONDITIONS
            ] + [
                {"id": "FULL_QD_SKILLS", "available": full_available,
                 "reason": None if full_available else _UNSUPPORTED_CONDITIONS["FULL_QD_SKILLS"]}
            ],
            "checker_ids": ["json_exact_v1", "json_numeric_tolerance_v1"],
            "frozen_skill_sources": ["published_qd"],
            "frozen_skill_versions": versions,
        }

    async def start(self, payload: EvaluationStartRequest, request_key: str) -> EvaluationStartResponse:
        try:
            task = self.store.task(payload.task_id)
        except KeyError as error:
            raise ServiceError("NOT_FOUND", "科研任务不存在。", status_code=404) from error
        del task
        full_available = _production_frozen_backend(self.frozen_backend) and self.environment_storage is not None
        unsupported = sorted(set(payload.conditions) & set(_UNSUPPORTED_CONDITIONS)) if not full_available else []
        if unsupported:
            reason = "; ".join(_UNSUPPORTED_CONDITIONS[item] for item in unsupported)
            raise ServiceError(
                "EVALUATION_CONDITION_UNSUPPORTED", reason, status_code=409,
                next_action="取消该条件后运行目前可用的比较；技能执行接入后再选择完整技能条件。",
            )
        if payload.frozen_qd_skills and "FULL_QD_SKILLS" not in payload.conditions:
            raise ServiceError(
                "EVALUATION_FROZEN_SKILLS_UNSUPPORTED",
                "评测运行时尚未执行已发布 QD 技能；不能将选定技能版本作为静态条件的输入。",
                status_code=409,
                next_action="当前请使用无冻结技能的可用条件；冻结技能运行接入后再选择具体版本。",
            )
        if "FULL_QD_SKILLS" in payload.conditions and not payload.frozen_qd_skills:
            raise ServiceError("EVALUATION_FROZEN_SKILL_MISSING", "完整技能条件需要选定已发布技能。", status_code=422)
        manifest = self._manifest(payload, request_key)
        try:
            manifest.validate_experiment()
        except ValueError as error:
            raise ServiceError(str(error), "冻结评测配置未通过验证。", status_code=422) from error
        if "FULL_QD_SKILLS" in manifest.conditions:
            from photomatagent.scientific.evolution.qd.frozen_executor import preflight_frozen_skill
            assert self.frozen_backend is not None and self.environment_storage is not None
            try:
                for skill in manifest.frozen_skills:
                    preflight_frozen_skill(
                        self.workspace.resolve(skill.relative_path, must_exist=True),
                        skill.bundle_sha256, backend=self.frozen_backend,
                        storage_root=self.environment_storage,
                    )
            except (OSError, ValueError, TypeError) as error:
                raise ServiceError(
                    "EVALUATION_FROZEN_SKILL_NOT_EXECUTABLE",
                    "冻结技能的脚本、输入映射或依赖环境未通过预检。",
                    status_code=422,
                ) from error
        request_body: dict[str, object] = manifest.model_dump(mode="json")
        run_request = RunRequest(
            text="运行已冻结的科研产物对照评测。",
            budget=BudgetSpec(
                max_model_calls=payload.budget.max_model_calls,
                max_tokens=payload.budget.max_model_tokens,
                wall_seconds=math.ceil(payload.budget.max_wall_seconds),
            ),
            host_operation={
                "operation": "evaluation",
                "evaluation_id": manifest.evaluation_id,
                "manifest_sha256": manifest.content_sha256,
            },
        )
        try:
            reference = self.store.reserve_evaluation_run(
                task_id=payload.task_id,
                request=run_request,
                request_key=request_key,
                evaluation_id=manifest.evaluation_id,
                manifest=request_body,
            )
            if self.store.run(reference.run_id).status == "QUEUED":
                await self.manager.enqueue(reference.run_id)
        except IdempotencyConflict as error:
            raise ServiceError("IDEMPOTENCY_CONFLICT", "此请求键已用于不同评测配置。", status_code=409) from error
        except ActiveRunConflict as error:
            raise ServiceError("ACTIVE_RUN_EXISTS", str(error), status_code=409, retryable=True) from error
        return EvaluationStartResponse(
            evaluation_id=manifest.evaluation_id,
            task_id=payload.task_id,
            run_id=reference.run_id,
            status=self.store.run(reference.run_id).status,
        )

    def get(self, evaluation_id: str) -> EvaluationStatusResponse:
        try:
            record = self.store.evaluation(evaluation_id)
            run = self.store.run(str(record["run_id"]))
        except KeyError as error:
            raise ServiceError("NOT_FOUND", "评测记录不存在。", status_code=404) from error
        manifest = FrozenEvaluationManifestV2.model_validate(record["manifest"])
        status = str(record["status"])
        if run.status in {"QUEUED", "RUNNING", "INTERRUPTED", "CANCEL_REQUESTED", "CANCELLED", "FAILED"}:
            status = "INTERRUPTED" if run.status == "INTERRUPTED" else run.status
        return EvaluationStatusResponse(
            evaluation_id=evaluation_id,
            task_id=str(record["task_id"]),
            run_id=str(record["run_id"]),
            status=status,
            manifest_sha256=manifest.content_sha256,
            created_at=str(record["created_at"]),
            updated_at=str(record["updated_at"]),
            report=cast(dict[str, object] | None, record["report"]),
            error_code=cast(str | None, record["error_code"]),
        )

    def list_for_task(self, task_id: str) -> list[EvaluationStatusResponse]:
        try:
            records = self.store.evaluations_for_task(task_id)
        except KeyError as error:
            raise ServiceError("NOT_FOUND", "科研任务不存在。", status_code=404) from error
        return [self.get(str(record["evaluation_id"])) for record in records]

    def skill_reliability(self, bundle_sha256: str, *, task_id: str | None = None) -> dict[str, object]:
        """Summarize current-software held-out outcomes without returning cases or references."""
        if not re.fullmatch(r"[0-9a-f]{64}", bundle_sha256):
            raise ValueError("EVALUATION_SKILL_HASH_INVALID")
        from photomatagent.scientific.evolution.qd.evaluation_runtime import evaluation_software_sha256

        current_software = evaluation_software_sha256()
        records = (
            self.store.evaluations_for_task(task_id) if task_id is not None
            else [record for task in self.store.tasks()
                  for record in self.store.evaluations_for_task(task.task_id)]
        )
        successes = failures = unknown = 0
        groups: set[tuple[str, str]] = set()
        evaluation_ids: list[str] = []
        for record in records:
            if record["status"] != "SUCCEEDED" or not isinstance(record.get("report"), dict):
                continue
            try:
                manifest = FrozenEvaluationManifestV2.model_validate(record["manifest"])
                report = EvaluationReport.model_validate(record["report"])
                manifest.validate_experiment()
                if (manifest.software_sha256 != current_software
                        or report.manifest_sha256 != manifest.content_sha256
                        or report.evaluation_id != manifest.evaluation_id
                        or report.evidence_class != "runtime_observation"
                        or "FULL_QD_SKILLS" not in manifest.conditions
                        or bundle_sha256 not in {item.bundle_sha256 for item in manifest.frozen_skills}):
                    continue
                expected = {(case.case_id, case.task_group_id) for case in manifest.cases}
                outcomes = [item for item in report.results if item.condition == "FULL_QD_SKILLS"]
                if ({(item.case_id, item.task_group_id) for item in outcomes} != expected
                        or len(outcomes) != len(expected)
                        or any(item.provenance != "runtime" for item in outcomes)):
                    continue
                relative = f"user_output/{record['task_id']}/evaluations/{manifest.evaluation_id}/report.json"
                archive = self.workspace.resolve(relative, must_exist=True)
                archived = archive.read_bytes()
                canonical = _canonical(report.model_dump(mode="json")) + b"\n"
                artifact = self.store.artifact(
                    "evaluation_report_" + manifest.evaluation_id.removeprefix("eval_")
                )
                if (archive.is_symlink() or archived != canonical
                        or artifact["task_id"] != record["task_id"]
                        or artifact["sha256"] != hashlib.sha256(archived).hexdigest()):
                    continue
            except (OSError, ValueError, TypeError, KeyError):
                continue
            evaluation_ids.append(manifest.evaluation_id)
            for item in outcomes:
                groups.add((manifest.evaluation_id, item.task_group_id))
                if item.verdict == "PASS":
                    successes += 1
                elif item.verdict == "FAIL":
                    failures += 1
                else:
                    unknown += 1
        return {
            "bundle_sha256": bundle_sha256,
            "evaluations": len(evaluation_ids), "cases": successes + failures + unknown,
            "groups": len(groups), "successes": successes, "failures": failures,
            "unknown": unknown, "evidence_class": "runtime_observation" if evaluation_ids else "unavailable",
            "evaluation_ids": sorted(evaluation_ids),
        }

    async def execute(self, run: Any) -> None:
        operation = run.request.host_operation
        if operation is None or operation.get("operation") != "evaluation":
            raise ValueError("EVALUATION_OPERATION_INVALID")
        evaluation_id = str(operation.get("evaluation_id", ""))
        record = self.store.evaluation(evaluation_id)
        manifest = FrozenEvaluationManifestV2.model_validate(record["manifest"])
        expected_sha256 = str(operation.get("manifest_sha256", ""))
        if (record["task_id"] != run.task_id or record["run_id"] != run.run_id
                or manifest.content_sha256 != expected_sha256):
            raise ValueError("EVALUATION_BINDING_MISMATCH")
        self.store.update_evaluation(evaluation_id, status="RUNNING")
        runner = self._make_runner(run.run_id)
        from photomatagent.scientific.evolution.qd.evaluation import FrozenSkillEvaluator

        evaluator = FrozenSkillEvaluator(
            workspace=self.workspace,
            manifest=manifest,
            expected_manifest_sha256=expected_sha256,
            runner=runner,
            checker=lambda _case, output, reference: check_reference(manifest.checker, output, reference),
        )
        try:
            report = await evaluator.run()
        except BaseException as error:
            self.store.update_evaluation(
                evaluation_id, status="FAILED", error_code=_diagnostic_code(error),
            )
            raise
        report_json = report.model_dump(mode="json")
        self._archive_report(run, evaluation_id, report, report_json)
        cancelled = any(
            any(code in {"EVALUATION_CASE_CANCELLED", "EVALUATION_CANCELLED"} for code in item.diagnostics)
            for item in report.results
        )
        self.store.update_evaluation(
            evaluation_id,
            status="CANCELLED" if cancelled else "SUCCEEDED",
            report=report_json,
        )
        self.store.append_event_for_run(
            run.run_id,
            "evaluation_completed",
            {
                "evaluation_id": evaluation_id,
                "status": "CANCELLED" if cancelled else "SUCCEEDED",
                "evidence_class": report.evidence_class,
                "cost": report.total_cost.to_dict(),
            },
        )
        self.store.append_event_for_run(
            run.run_id, "evaluation_experience_recorded", {
                "evaluation_id": evaluation_id,
                "manifest_sha256": report.manifest_sha256,
                "frozen_skill_sha256": [item.bundle_sha256 for item in manifest.frozen_skills],
                "condition_success": dict(report.condition_success),
                "evidence_class": report.evidence_class,
            },
        )

    def _make_runner(self, run_id: str) -> Any:
        def event_sink(event_type: str, payload: dict[str, object]) -> None:
            self.store.append_event_for_run(run_id, event_type, payload)

        if self.runner_factory is not None:
            return self.runner_factory(event_sink=event_sink)
        from photomatagent.scientific.evolution.qd.evaluation_runtime import RuntimeEvaluationRunner
        from photomatagent.scientific.evolution.qd.runtime_adapter import _configured_model

        return RuntimeEvaluationRunner(
            model=_configured_model(self.workspace.root), event_sink=event_sink,
            frozen_backend=self.frozen_backend, environment_storage=self.environment_storage,
        )

    def _manifest(self, payload: EvaluationStartRequest, request_key: str) -> FrozenEvaluationManifestV2:
        cases: list[EvaluationCase] = []
        for selection in payload.cases:
            input_ref = self._artifact_ref(payload.task_id, selection.input_artifact_id)
            reference_ref = self._artifact_ref(payload.task_id, selection.reference_artifact_id)
            if selection.input_artifact_id == selection.reference_artifact_id:
                raise ServiceError("EVALUATION_CASE_ARTIFACTS_INVALID", "输入与参考答案必须是不同产物。", status_code=422)
            cases.append(EvaluationCase(
                case_id=selection.case_id,
                task_group_id=selection.task_group_id,
                signal_family=selection.signal_family,
                source_id=selection.source_id,
                input_ref=input_ref,
                reference_ref=reference_ref,
            ))
        frozen: list[FrozenSkillRef] = []
        for item in payload.frozen_qd_skills:
            relative = (
                Path(".photomatagent") / "qd" / "published-skills"
                / item.skill_id / item.bundle_sha256
            )
            source = self.workspace.resolve(relative.as_posix(), must_exist=True)
            from photomatagent.skills.published import PublishedSkillSource
            try:
                PublishedSkillSource(self.workspace.root / ".photomatagent/qd/published-skills")._verify_bundle(
                    source, item.bundle_sha256,
                )
            except (OSError, ValueError, TypeError) as error:
                raise ServiceError("EVALUATION_SKILL_VERSION_INVALID", "所选冻结技能版本不可验证。", status_code=422) from error
            frozen.append(FrozenSkillRef(
                skill_id=item.skill_id,
                bundle_sha256=item.bundle_sha256,
                relative_path=relative.as_posix(),
            ))
        from photomatagent.scientific.evolution.qd.evaluation_runtime import evaluation_software_sha256
        if self.model_id_factory is not None:
            model_id = self.model_id_factory()
        else:
            from photomatagent.scientific.evolution.qd.runtime_adapter import _configured_model
            model_id = _configured_model(self.workspace.root).model
        digest = hashlib.sha256(
            (payload.task_id + "\0" + request_key + "\0" + _canonical(payload.model_dump(mode="json")).decode()).encode()
        ).hexdigest()
        return FrozenEvaluationManifestV2(
            schema_version=2,
            evaluation_id="eval_" + digest[:32],
            model_id=model_id,
            software_sha256=evaluation_software_sha256(),
            dataset_sha256=dataset_identity_sha256(tuple(cases)),
            seed=payload.seed,
            budget=payload.budget,
            cases=tuple(cases),
            development_groups=payload.development_groups,
            frozen_skills=tuple(frozen),
            conditions=payload.conditions,
            checker=payload.checker,
        )

    def _artifact_ref(self, task_id: str, artifact_id: str) -> ArtifactDigest:
        try:
            row = self.store.artifact(artifact_id)
        except KeyError as error:
            raise ServiceError("EVALUATION_ARTIFACT_NOT_FOUND", "评测产物不存在。", status_code=404) from error
        if row["task_id"] != task_id:
            raise ServiceError("EVALUATION_ARTIFACT_SCOPE_VIOLATION", "输入和参考产物必须属于当前科研任务。", status_code=400)
        relative = Path(str(row["path"]))
        allowed_prefixes = (
            Path(".photomatagent") / "workbench" / "uploads" / task_id,
            Path("user_output") / task_id,
        )
        if (relative.is_absolute() or ".." in relative.parts or "\\" in str(row["path"])
                or not any(relative.is_relative_to(prefix) for prefix in allowed_prefixes)):
            raise ServiceError("EVALUATION_ARTIFACT_PATH_INVALID", "产物路径不在当前任务的附件或结果目录中。", status_code=422)
        if row["media_type"] != "application/json":
            raise ServiceError("EVALUATION_ARTIFACT_TYPE_INVALID", "当前冻结检查器只接受 JSON 输入与参考文件。", status_code=415)
        current = self.workspace.root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ServiceError("EVALUATION_ARTIFACT_PATH_INVALID", "评测产物不能经过符号链接。", status_code=422)
        path = self.workspace.resolve(relative.as_posix(), must_exist=True)
        if not path.is_file():
            raise ServiceError("EVALUATION_ARTIFACT_UNAVAILABLE", "评测产物文件不可读取。", status_code=422)
        data = path.read_bytes()
        if (len(data) != int(row["size_bytes"])
                or hashlib.sha256(data).hexdigest() != str(row["sha256"])):
            raise ServiceError("EVALUATION_ARTIFACT_CHANGED", "评测产物与保存的摘要不一致。", status_code=409)
        return ArtifactDigest(
            path=relative.as_posix(),
            sha256=str(row["sha256"]),
            size_bytes=int(row["size_bytes"]),
            media_type=str(row["media_type"]),
        )

    def _archive_report(
        self, run: Any, evaluation_id: str, report: EvaluationReport,
        report_json: dict[str, object],
    ) -> None:
        import uuid

        relative = Path("user_output") / run.task_id / "evaluations" / evaluation_id / "report.json"
        target = self.workspace.resolve(relative.as_posix(), must_exist=False)
        cursor = self.workspace.root
        for part in relative.parts[:-1]:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError("EVALUATION_REPORT_PATH_INVALID")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or (target.exists() and target.read_bytes() != (json.dumps(
            report_json, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ) + "\n").encode()):
            raise ValueError("EVALUATION_REPORT_PATH_CONFLICT")
        payload = (json.dumps(
            report_json, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ) + "\n").encode()
        if not target.exists():
            fd, temporary_name = tempfile.mkstemp(prefix=".evaluation-report-", dir=target.parent)
            try:
                with os.fdopen(fd, "wb") as output:
                    output.write(payload)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary_name, target)
            except BaseException:
                Path(temporary_name).unlink(missing_ok=True)
                raise
        artifact_id = "evaluation_report_" + evaluation_id.removeprefix("eval_")
        digest = hashlib.sha256(payload).hexdigest()
        try:
            existing = self.store.artifact(artifact_id)
        except KeyError:
            self.store.register_artifact(
                artifact_id=artifact_id,
                task_id=run.task_id,
                run_id=run.run_id,
                path=relative.as_posix(),
                filename="evaluation-report.json",
                sha256=digest,
                media_type="application/json",
                size_bytes=len(payload),
            )
        else:
            if existing["sha256"] != digest or existing["task_id"] != run.task_id:
                raise ValueError("EVALUATION_REPORT_ARTIFACT_CONFLICT")
        self.store.link_source(run.task_id, "evaluation", evaluation_id, digest)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def _configured_frozen_backend(workspace: Workspace) -> tuple[SandboxBackend | None, Path | None]:
    """Opt in through the existing QD config and its pinned environment store."""
    config_path = workspace.root / ".photomatagent" / "qd.json"
    if not config_path.is_file() or config_path.is_symlink():
        return None, None
    try:
        from photomatagent.scientific.evolution.qd.config import QDConfig
        config = QDConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
        if not config.enabled:
            return None, None
        storage = workspace.resolve(config.environment_storage, must_exist=True)
        if not storage.is_dir():
            return None, None
        if config.sandbox == "podman":
            from photomatagent.capability_acquisition.container_backend import PodmanSandboxBackend
            backend = PodmanSandboxBackend(storage)
            if backend.executable is None:
                return None, None
            return backend, storage
    except (OSError, ValueError, TypeError):
        return None, None
    return None, None


def _production_frozen_backend(backend: SandboxBackend | None) -> bool:
    from photomatagent.capability_acquisition.container_backend import PodmanSandboxBackend
    return isinstance(backend, PodmanSandboxBackend) and backend.executable is not None


def _diagnostic_code(error: BaseException) -> str:
    name = type(error).__name__.upper()
    if isinstance(error, asyncio.CancelledError):
        return "EVALUATION_INTERRUPTED"
    return "EVALUATION_FAILED_" + name[:48]


__all__ = [
    "EvaluationCapabilitiesResponse", "EvaluationCaseSelection", "EvaluationService",
    "EvaluationSkillSelection", "EvaluationStartRequest", "EvaluationStartResponse",
    "EvaluationStatusResponse",
]
