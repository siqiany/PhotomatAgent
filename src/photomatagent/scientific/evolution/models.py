"""Strict persistence models for expert-guided scientific evolution."""

from __future__ import annotations

import re
import secrets
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from photomatagent.scientific.loop import ScientificLoopSummary, TargetSpec

EvolutionStatus = Literal[
    "CREATED",
    "RUNNING",
    "AWAITING_EXPERT_FEEDBACK",
    "FEEDBACK_RECORDED",
    "REVISION_READY",
    "ACCEPTED",
    "STOPPED",
    "BUDGET_EXHAUSTED",
    "BLOCKED",
]
EpisodeStatus = Literal["RESERVED", "RUNNING", "COMPLETED", "FAILED"]
ExecutionMode = Literal["NORMAL", "CARRY_VERIFIED_EVIDENCE", "FRESH_EVALUATION"]
StrategyArm = Literal[
    "STATIC", "EVIDENCE_FIRST", "DIVERSITY_FIRST", "UNCERTAINTY_FIRST"
]
FeedbackItemStatus = Literal[
    "CORRECTION", "QUERY", "PREFERENCE", "POSITIVE_SIGNAL"
]
FeedbackSeverity = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
CompilationStatus = Literal["PENDING", "AVAILABLE", "UNAVAILABLE"]
AcceptanceStatus = Literal["PENDING", "PASS", "FAIL", "NEEDS_HUMAN_REVIEW"]
ExperienceMaturity = Literal[
    "OBSERVATION", "HYPOTHESIS", "VALIDATED_EXPERIENCE", "REUSABLE_SKILL"
]

_MANAGED_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def utc_now() -> datetime:
    """Return the single canonical timestamp form used by evolution records."""

    return datetime.now(UTC)


def validate_managed_id(value: str) -> str:
    """Reject IDs that cannot safely be used as one managed path component."""

    if not _MANAGED_ID_RE.fullmatch(value):
        raise ValueError("managed IDs may contain only letters, numbers, '_' and '-'")
    return value


def new_evolution_id() -> str:
    timestamp = utc_now().strftime("%Y%m%dT%H%M%S%fZ")
    return f"evo_{timestamp}_{secrets.token_hex(3)}"


def new_episode_id() -> str:
    return f"ep_{secrets.token_hex(5)}"


def new_feedback_id() -> str:
    return f"fb_{secrets.token_hex(5)}"


def new_revision_id() -> str:
    return f"rp_{secrets.token_hex(5)}"


def new_strategy_id() -> str:
    return f"strategy_{secrets.token_hex(5)}"


class StrictModel(BaseModel):
    """Base contract for JSON records owned by the evolution subsystem."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RubricScores(StrictModel):
    scientific_correctness: int = Field(ge=1, le=5)
    evidence_sufficiency: int = Field(ge=1, le=5)
    novelty: int = Field(ge=1, le=5)
    actionability: int = Field(ge=1, le=5)
    overall: int = Field(ge=1, le=5)


class RubricFlags(StrictModel):
    fabricated_source: bool = False
    conclusion_changing_error: bool = False
    abstract_only_core_evidence: bool = False
    unsupported_novelty: bool = False
    process_parameters_only: bool = False


class ExpertFeedbackDraft(StrictModel):
    scores: RubricScores
    flags: RubricFlags = Field(default_factory=RubricFlags)
    fatal_issue: bool = False
    comments: str = ""
    priority_corrections: list[str] = Field(default_factory=list, max_length=3)
    preserved_strengths: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)


class ArtifactRef(StrictModel):
    path: str
    media_type: str = "text/markdown"
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")


class CostSnapshot(StrictModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    runtime_seconds: float = Field(default=0.0, ge=0.0)
    hpc_cost: float | None = Field(default=None, ge=0.0)


class AcceptanceResult(StrictModel):
    acceptance_id: str
    status: AcceptanceStatus = "PENDING"
    detail: str = ""


class EvolutionTask(StrictModel):
    schema_version: int = Field(default=1, frozen=True)
    revision: int = Field(default=0, ge=0)
    evolution_id: str
    goal: str = Field(frozen=True)
    target: TargetSpec = Field(frozen=True)
    task_group_id: str = Field(frozen=True)
    input_sha256: str = Field(default="", frozen=True)
    status: EvolutionStatus = "CREATED"
    current_version: str | None = None
    last_completed_version: str | None = None
    accepted_version: str | None = None
    episode_ids: list[str] = Field(default_factory=list)
    feedback_ids: list[str] = Field(default_factory=list)
    compilation_ids: list[str] = Field(default_factory=list)
    revision_ids: list[str] = Field(default_factory=list)
    strategy_ids: list[str] = Field(default_factory=list)
    comparison_ids: list[str] = Field(default_factory=list)
    experience_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now, frozen=True)
    updated_at: datetime = Field(default_factory=utc_now)


class EpisodeRecord(StrictModel):
    schema_version: int = 1
    evolution_id: str
    episode_id: str
    version: str
    status: EpisodeStatus = "RESERVED"
    parent_version: str | None = None
    applied_feedback_id: str | None = None
    revision_plan_id: str | None = None
    runtime_session_id: str | None = None
    event_log_path: str | None = None
    execution_mode: ExecutionMode = "NORMAL"
    strategy_id: str | None = None
    strategy_arm: StrategyArm = "STATIC"
    scientific_state_path: str | None = None
    task_snapshot: dict[str, Any] = Field(default_factory=dict)
    target_snapshot: TargetSpec | None = None
    provider: str | None = None
    model: str | None = None
    tool_surface_fingerprint: str | None = None
    capability_fingerprint: str | None = None
    data_source_fingerprints: dict[str, str] = Field(default_factory=dict)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    summary: ScientificLoopSummary | None = None
    artifact: ArtifactRef | None = None
    cost: CostSnapshot = Field(default_factory=CostSnapshot)
    acceptance_results: list[AcceptanceResult] = Field(default_factory=list)
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class FeedbackDelta(StrictModel):
    item_id: str = ""
    category: str
    status: FeedbackItemStatus
    severity: FeedbackSeverity
    responsible_module: str
    problem: str
    requested_actions: list[str] = Field(default_factory=list)
    acceptance_test: str | None = None
    preserve: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    source_span: str


class FeedbackCompilation(StrictModel):
    schema_version: int = 1
    compilation_id: str = ""
    evolution_id: str = ""
    feedback_id: str = ""
    episode_version: str = ""
    status: CompilationStatus = "PENDING"
    items: list[FeedbackDelta] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class ExpertFeedbackRecord(ExpertFeedbackDraft):
    schema_version: int = 1
    feedback_id: str
    evolution_id: str
    episode_version: str
    result_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    rubric_version: str
    raw_input: str = ""
    suggested_scores: RubricScores | None = None
    hard_cap_reasons: list[str] = Field(default_factory=list)
    hard_cap_override_reason: str | None = None
    compilation_id: str | None = None
    confirmed_at: datetime = Field(default_factory=utc_now)
    supersedes_feedback_id: str | None = None


class RevisionPlan(StrictModel):
    schema_version: int = 1
    revision_id: str
    evolution_id: str
    source_version: str
    feedback_id: str
    contract_changes: list[str] = Field(default_factory=list)
    evidence_requirements: list[str] = Field(default_factory=list)
    output_schema_requirements: list[str] = Field(default_factory=list)
    preserved_facts: list[str] = Field(default_factory=list)
    preserved_evidence_ids: list[str] = Field(default_factory=list)
    prohibited_repeats: list[str] = Field(default_factory=list)
    invalidated_conclusions: list[str] = Field(default_factory=list)
    invalidated_evidence_ids: list[str] = Field(default_factory=list)
    machine_acceptance_tests: list[str] = Field(default_factory=list)
    human_acceptance_tests: list[str] = Field(default_factory=list)
    strategy_arm: StrategyArm = "STATIC"
    strategy_reason: str = ""
    warnings: list[str] = Field(default_factory=list)
    unresolved_ambiguities: list[str] = Field(default_factory=list)
    has_blocking_ambiguity: bool = False
    confirmed: bool = False
    confirmed_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)


class StrategyVersion(StrictModel):
    schema_version: int = 1
    strategy_id: str
    evolution_id: str
    arm: StrategyArm
    reason: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    strategy_sha256: str = ""
    cutoff_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)


class ComparisonReport(StrictModel):
    schema_version: int = 1
    comparison_id: str
    evolution_id: str
    previous_version: str
    current_version: str
    acceptance_results: list[AcceptanceResult] = Field(default_factory=list)
    closed_issue_ids: list[str] = Field(default_factory=list)
    recurring_issue_ids: list[str] = Field(default_factory=list)
    new_issue_ids: list[str] = Field(default_factory=list)
    closure_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    recurrence_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    new_issue_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    expert_utility_delta: float | None = None
    normalized_cost_increase: float | None = None
    reward: float | None = Field(default=None, ge=-1.0, le=1.0)
    components_used: list[str] = Field(default_factory=list)
    module_credit: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
