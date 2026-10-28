"""Source-bound capture of reviewed research procedures for the workbench."""
from __future__ import annotations

import hashlib
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from photomatagent.application.errors import ServiceError
from photomatagent.application.procedural_skills import ProcedureCreate, ProcedureService

if TYPE_CHECKING:
    from photomatagent.application.service import WorkbenchService


class ProcedureReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    review_note: str = Field(min_length=1, max_length=2000)


class ProcedureWorkflow:
    def __init__(self, service: WorkbenchService) -> None:
        self.workbench = service
        self.procedures = ProcedureService(service.workspace_root)

    def create(self, payload: ProcedureCreate) -> dict[str, Any]:
        provenance: dict[str, Any] = {}
        try:
            run = self.workbench.store.run(payload.run_id)
            if run.task_id != payload.task_id or run.status not in {"SUCCEEDED", "PARTIAL", "FAILED", "CANCELLED", "INTERRUPTED"}:
                raise ValueError("PROCEDURE_RUN_SOURCE_INVALID")
            provenance["source_run_status"] = run.status
            if payload.plan_id is not None:
                plan = self.workbench.store.revision_plan(payload.plan_id)
                source_runs = {plan.source_run_id, *(item.run_id for item in plan.run_attempts)}
                if plan.task_id != payload.task_id or payload.run_id not in source_runs:
                    raise ValueError("PROCEDURE_FEEDBACK_SOURCE_INVALID")
                provenance["feedback"] = {"plan_id": plan.plan_id, "source_digest": plan.source_digest,
                                          "feedback_sha256": hashlib.sha256(plan.feedback_text.encode()).hexdigest()}
            if payload.artifact_id is not None:
                artifact = self.workbench.store.artifact(payload.artifact_id)
                if artifact["task_id"] != payload.task_id or artifact["run_id"] != payload.run_id:
                    raise ValueError("PROCEDURE_ARTIFACT_SOURCE_INVALID")
                try:
                    snapshot, media_type, _ = self.workbench.artifacts.content(payload.artifact_id)
                except ServiceError as exc:
                    raise ValueError("PROCEDURE_ARTIFACT_UNAVAILABLE") from exc
                snapshot.close()
                if media_type not in {"text/markdown", "text/plain"}:
                    raise ValueError("PROCEDURE_TEXT_SOURCE_REQUIRED")
                provenance["artifact"] = {"artifact_id": payload.artifact_id, "sha256": artifact["sha256"]}
            return self.procedures.create(payload, provenance=provenance)
        except (KeyError, ValueError, OSError) as exc:
            raise ServiceError("PROCEDURE_SOURCE_INVALID", "方法经验来源不属于所选运行，或文件已经变化。", status_code=409) from exc

    def list_skills(self) -> list[dict[str, Any]]:
        return [item | {"retrieval_count": len(self.procedures.list_retrievals(item["skill_id"]))}
                for item in self.procedures.list_skills()]

    def read(self, skill_id: str) -> dict[str, Any]:
        try:
            return self.procedures.read(skill_id)
        except (ValueError, OSError) as exc:
            raise ServiceError("PROCEDURE_UNAVAILABLE", "方法经验不存在或完整性检查失败。", status_code=409) from exc

    def approve(self, skill_id: str, review_note: str) -> dict[str, Any]:
        self.read(skill_id)
        try:
            return self.procedures.approve(skill_id, review_note)
        except ValueError as exc:
            raise ServiceError("PROCEDURE_REVIEW_INVALID", "请填写审阅说明；已停用的经验不能重新启用。", status_code=422) from exc

    def disable(self, skill_id: str) -> dict[str, Any]:
        self.read(skill_id)
        return self.procedures.disable(skill_id)
