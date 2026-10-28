"""Contracts for the persistent local scientific workbench index."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, field_validator, model_serializer
from photomatagent.application.skills import CheckerSpec
from photomatagent.scientific.loop.target import TargetSpec

from photomatagent.scientific.evolution.models import FeedbackCategory, FeedbackDelta, FeedbackItemStatus, FeedbackSeverity


class BudgetSpec(BaseModel):
    max_model_calls: int = Field(default=20, ge=1)
    max_tokens: int = Field(default=100_000, ge=1)
    wall_seconds: int = Field(default=1800, ge=1)


class TaskCreate(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    goal: str = Field(min_length=1, max_length=20_000)
    attachment_ids: list[str] = Field(default_factory=list)

    @field_validator("title", "goal")
    @classmethod
    def non_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("attachment_ids")
    @classmethod
    def attachment_ids_are_nonblank(cls, values: list[str]) -> list[str]:
        if any(not item.strip() for item in values):
            raise ValueError("attachment IDs must not be blank")
        return values


class EvolutionOptions(BaseModel):
    """Frozen checks and bounded automatic repairs for a single Web run."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    max_rounds: int = Field(default=3, ge=1, le=8)
    patience: int = Field(default=2, ge=1, le=4)
    required_artifacts: list[str] = Field(default_factory=list, max_length=20)
    checks: list[CheckerSpec] = Field(default_factory=list, max_length=20)
    input_artifact_ids: dict[str, str] = Field(default_factory=dict, max_length=20)
    target: TargetSpec | None = None

    @field_validator("required_artifacts")
    @classmethod
    def filenames_only(cls, values: list[str]) -> list[str]:
        if any(not v.strip() or v != v.strip() or v in {".", ".."} or "/" in v or "\\" in v or ":" in v or len(v) > 255 for v in values):
            raise ValueError("DELIVERY_FILENAME_INVALID")
        if len(set(values)) != len(values):
            raise ValueError("DELIVERY_FILENAME_DUPLICATE")
        return values


class RunRequest(BaseModel):
    text: str = Field(min_length=1, max_length=100_000)
    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    strategy: Literal["standard", "feedback", "qd", "skill_replay"] = "standard"
    host_operation: dict[str, object] | None = None
    evolution: EvolutionOptions = Field(default_factory=EvolutionOptions)

    @model_serializer(mode="wrap")
    def preserve_ordinary_request_identity(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        payload: dict[str, Any] = handler(self)
        if self.host_operation is None:
            payload.pop("host_operation", None)
        if self.evolution == EvolutionOptions():
            payload.pop("evolution", None)
        return payload

    @field_validator("text")
    @classmethod
    def non_blank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class FeedbackRequest(BaseModel):
    source_run_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=8_000)
    artifact_ids: list[str] = Field(default_factory=list, max_length=50)
    budget: BudgetSpec = Field(default_factory=BudgetSpec)

    @model_serializer(mode="wrap")
    def preserve_default_feedback_identity(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        payload: dict[str, Any] = handler(self)
        if self.budget == BudgetSpec():
            payload.pop("budget", None)
        return payload

    @field_validator("source_run_id", "text")
    @classmethod
    def feedback_text_is_nonblank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("artifact_ids")
    @classmethod
    def feedback_artifact_ids_are_unique(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("artifact IDs must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("artifact IDs must be unique")
        return normalized


class RevisionArtifactSource(BaseModel):
    artifact_id: str
    filename: str
    sha256: str
    media_type: str
    size_bytes: int
    path: str


class WorkbenchRevisionIssue(BaseModel):
    issue_id: str
    category: FeedbackCategory
    status: FeedbackItemStatus
    severity: FeedbackSeverity
    responsible_module: str
    problem: str
    original_feedback_span: str
    original_evidence: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    expected_artifacts: list[str] = Field(default_factory=list)
    check_method: str | None = None
    acceptance_test: str | None = None
    cannot_process_reason: str | None = None
    preserve: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class RevisionRunAttempt(BaseModel):
    attempt: int = Field(ge=1)
    run_id: str
    status: str


class WorkbenchRevisionPlan(BaseModel):
    plan_id: str
    task_id: str
    feedback_id: str
    source_run_id: str
    source_digest: str
    source_run_status: str
    source_artifacts: list[RevisionArtifactSource] = Field(default_factory=list)
    feedback_text: str
    status: Literal["PREPARING", "READY", "UNAVAILABLE"]
    issues: list[WorkbenchRevisionIssue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    compiler_provider: str | None = None
    compiler_model: str | None = None
    error: str | None = None
    instruction: str | None = None
    attempt: int = Field(ge=1)
    created_at: str
    run_id: str | None = None
    run_attempts: list[RevisionRunAttempt] = Field(default_factory=list)
    compiler_cost: dict[str, object] = Field(default_factory=dict)


class FeedbackAnalysis(BaseModel):
    items: list[FeedbackDelta] = Field(default_factory=list, max_length=100)
    warnings: list[str] = Field(default_factory=list, max_length=100)
    compiler_provider: str | None = None
    compiler_model: str | None = None


class RunRef(BaseModel):
    task_id: str
    run_id: str


class WorkbenchEvent(BaseModel):
    task_id: str
    run_id: str
    seq: int = Field(ge=1)
    type: str = Field(min_length=1, max_length=200)
    payload: dict[str, object]
    created_at: str = Field(min_length=1)


class ResearchTask(BaseModel):
    task_id: str
    session_id: str
    title: str
    goal: str
    attachment_ids: list[str]
    created_at: str


class RunRecord(BaseModel):
    run_id: str
    task_id: str
    request: RunRequest
    status: str
    created_at: str


class WorkbenchLink(BaseModel):
    task_id: str
    source_type: str
    source_id: str
    source_digest: str
    snapshot_digest: str | None = None
    created_at: str
