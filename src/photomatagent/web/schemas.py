"""Stable HTTP response contracts for the local workbench."""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Literal
from photomatagent.application.qd_strategy import QDRevisionOptions

from photomatagent.application.models import (
    BudgetSpec,
    WorkbenchRevisionPlan,
    ResearchTask,
    RunRecord,
    RunRef,
    WorkbenchLink,
)
from photomatagent.application.evaluations import (
    EvaluationCapabilitiesResponse,
    EvaluationStartRequest,
    EvaluationStartResponse,
    EvaluationStatusResponse,
)


class ErrorBody(BaseModel):
    code: str
    message: str
    run_id: str | None = None
    retryable: bool = False
    next_action: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class TaskIdResponse(BaseModel):
    task_id: str


class RunIdResponse(BaseModel):
    task_id: str
    run_id: str


class ApprovalDecisionRequest(BaseModel):
    approved: bool


class ApprovalDecisionResponse(BaseModel):
    request_id: str
    decision: Literal["approved", "denied"]


class HistoryImportRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=255)


class TaskSummaryResponse(ResearchTask):
    latest_run: RunRecord | None = None


class TaskListResponse(BaseModel):
    items: list[TaskSummaryResponse]


class ArtifactMetadataResponse(BaseModel):
    artifact_id: str
    task_id: str
    run_id: str | None
    filename: str
    media_type: str
    size_bytes: int
    sha256: str


class AttachmentResponse(ArtifactMetadataResponse):
    attachment_id: str


class TaskDetailResponse(BaseModel):
    task: ResearchTask
    runs: list[RunRecord]
    links: list[WorkbenchLink]
    artifacts: list[ArtifactMetadataResponse]
    evolution: dict[str, Any] = Field(default_factory=dict)


class HistoricalMessageResponse(BaseModel):
    role: str
    text: str


class HistoricalSourceResponse(BaseModel):
    session_id: str
    source_digest: str
    current: bool
    messages_current: bool | None


class HistoricalTaskResponse(BaseModel):
    source: HistoricalSourceResponse
    messages: list[HistoricalMessageResponse]
    messages_truncated: bool = False


class CancelRunResponse(BaseModel):
    run_id: str
    status: str


class HistorySessionResponse(BaseModel):
    session_id: str
    resumable: bool
    provider: str | None = None
    model: str | None = None
    iterations: int | None = None
    tool_calls: int | None = None
    stop_reason: str | None = None


class HistorySessionListResponse(BaseModel):
    items: list[HistorySessionResponse]


class CapabilityResponse(BaseModel):
    id: str
    available: bool
    reason: str | None = None


class CapabilityListResponse(BaseModel):
    items: list[CapabilityResponse]


class RunRefResponse(RunRef):
    pass


class RevisionRunRequest(BaseModel):
    strategy: Literal["standard", "qd"] = "standard"
    qd: QDRevisionOptions | None = None
    budget: BudgetSpec = Field(default_factory=BudgetSpec)


class RevisionPlanResponse(WorkbenchRevisionPlan):
    pass


__all__ = [
    "ArtifactMetadataResponse",
    "AttachmentResponse",
    "CancelRunResponse",
    "CapabilityListResponse",
    "EvaluationCapabilitiesResponse",
    "EvaluationStartRequest",
    "EvaluationStartResponse",
    "EvaluationStatusResponse",
    "RevisionPlanResponse",
    "RevisionRunRequest",
    "CapabilityResponse",
    "ErrorBody",
    "ErrorResponse",
    "HistoryImportRequest",
    "HistorySessionListResponse",
    "HistorySessionResponse",
    "HistoricalMessageResponse",
    "HistoricalSourceResponse",
    "HistoricalTaskResponse",
    "RunIdResponse",
    "RunRefResponse",
    "TaskDetailResponse",
    "TaskIdResponse",
    "TaskListResponse",
    "TaskSummaryResponse",
]
