"""Strict persistence models for expert-guided scientific evolution."""

from __future__ import annotations

import re
import secrets
from copy import deepcopy
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

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
RubricDimension = Literal[
    "scientific_correctness",
    "evidence_sufficiency",
    "novelty",
    "actionability",
    "overall",
]
SchemaVersion = Literal[1]

_MANAGED_ID_PATTERN = r"^[A-Za-z0-9_-]+$"
_MANAGED_ID_RE = re.compile(_MANAGED_ID_PATTERN)
ManagedId = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, pattern=_MANAGED_ID_PATTERN),
]
EpisodeVersion = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^v\d{3}$"),
]
Sha256 = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-fA-F]{64}$"),
]
RubricScore = Annotated[int, Field(strict=True, ge=1, le=5)]


def _normalize_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


UtcDatetime = Annotated[AwareDatetime, AfterValidator(_normalize_utc)]


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
    scientific_correctness: RubricScore
    evidence_sufficiency: RubricScore
    novelty: RubricScore
    actionability: RubricScore
    overall: RubricScore


class RubricFlags(StrictModel):
    fabricated_source: bool = False
    conclusion_changing_error: bool = False
    abstract_only_core_evidence: bool = False
    unsupported_novelty: bool = False
    process_parameters_only: bool = False


_HARD_CAP_REASONS = {
    "fabricated_source": "存在伪造来源：证据充分性和总体等级最高 1 分",
    "conclusion_changing_error": (
        "存在会改变结论的科学错误：科学正确性和总体等级最高 2 分"
    ),
    "abstract_only_core_evidence": "核心结论只有摘要支持：证据充分性最高 2 分",
    "unsupported_novelty": "创新性没有定义、基线或证据：创新性最高 2 分",
    "process_parameters_only": "工艺只有路线名称和少数参数：可执行性最高 2 分",
}


def derive_hard_cap_suggestion(
    scores: RubricScores,
    flags: RubricFlags,
) -> tuple[RubricScores, list[str]]:
    """Return canonical cap data shared by validation and rubric presentation."""

    values = scores.model_dump()
    reasons: list[str] = []
    if flags.fabricated_source:
        values["evidence_sufficiency"] = min(values["evidence_sufficiency"], 1)
        values["overall"] = min(values["overall"], 1)
        reasons.append(_HARD_CAP_REASONS["fabricated_source"])
    if flags.conclusion_changing_error:
        values["scientific_correctness"] = min(
            values["scientific_correctness"], 2
        )
        values["overall"] = min(values["overall"], 2)
        reasons.append(_HARD_CAP_REASONS["conclusion_changing_error"])
    if flags.abstract_only_core_evidence:
        values["evidence_sufficiency"] = min(values["evidence_sufficiency"], 2)
        reasons.append(_HARD_CAP_REASONS["abstract_only_core_evidence"])
    if flags.unsupported_novelty:
        values["novelty"] = min(values["novelty"], 2)
        reasons.append(_HARD_CAP_REASONS["unsupported_novelty"])
    if flags.process_parameters_only:
        values["actionability"] = min(values["actionability"], 2)
        reasons.append(_HARD_CAP_REASONS["process_parameters_only"])
    return RubricScores.model_validate(values), reasons


class ExpertFeedbackDraft(StrictModel):
    scores: RubricScores
    flags: RubricFlags = Field(default_factory=RubricFlags)
    fatal_issue: bool = False
    comments: str = ""
    priority_corrections: list[str] = Field(default_factory=list, max_length=3)
    preserved_strengths: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)


class ArtifactRef(StrictModel):
    path: str = Field(min_length=1)
    media_type: str = Field(default="text/markdown", min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: Sha256


class CostSnapshot(StrictModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    runtime_seconds: float = Field(default=0.0, ge=0.0)
    hpc_cost: float | None = Field(default=None, ge=0.0)


class AcceptanceResult(StrictModel):
    acceptance_id: ManagedId
    status: AcceptanceStatus = "PENDING"
    detail: str = ""


class EvolutionTask(StrictModel):
    schema_version: SchemaVersion = Field(default=1, frozen=True)
    revision: int = Field(default=0, ge=0)
    evolution_id: ManagedId = Field(frozen=True)
    goal: str = Field(min_length=1, frozen=True)
    target: TargetSpec = Field(frozen=True)
    task_group_id: ManagedId = Field(frozen=True)
    input_sha256: Sha256 = Field(frozen=True)
    status: EvolutionStatus = "CREATED"
    current_version: EpisodeVersion | None = None
    last_completed_version: EpisodeVersion | None = None
    accepted_version: EpisodeVersion | None = None
    episode_ids: list[ManagedId] = Field(default_factory=list)
    feedback_ids: list[ManagedId] = Field(default_factory=list)
    compilation_ids: list[ManagedId] = Field(default_factory=list)
    revision_ids: list[ManagedId] = Field(default_factory=list)
    strategy_ids: list[ManagedId] = Field(default_factory=list)
    comparison_ids: list[ManagedId] = Field(default_factory=list)
    experience_ids: list[ManagedId] = Field(default_factory=list)
    created_at: UtcDatetime = Field(default_factory=utc_now, frozen=True)
    updated_at: UtcDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def isolate_target_snapshot(self) -> Self:
        object.__setattr__(self, "target", self.target.model_copy(deep=True))
        return self


class EpisodeRecord(StrictModel):
    schema_version: SchemaVersion = 1
    evolution_id: ManagedId = Field(frozen=True)
    episode_id: ManagedId = Field(frozen=True)
    version: EpisodeVersion = Field(frozen=True)
    status: EpisodeStatus = "RESERVED"
    parent_version: EpisodeVersion | None = Field(default=None, frozen=True)
    applied_feedback_id: ManagedId | None = Field(default=None, frozen=True)
    revision_plan_id: ManagedId | None = Field(default=None, frozen=True)
    runtime_session_id: ManagedId | None = None
    event_log_path: str | None = None
    execution_mode: ExecutionMode = Field(default="NORMAL", frozen=True)
    strategy_id: ManagedId | None = Field(default=None, frozen=True)
    strategy_arm: StrategyArm = Field(default="STATIC", frozen=True)
    scientific_state_path: str | None = None
    task_snapshot: dict[str, Any] = Field(frozen=True)
    target_snapshot: TargetSpec = Field(frozen=True)
    provider: str | None = Field(default=None, frozen=True)
    model: str | None = Field(default=None, frozen=True)
    tool_surface_fingerprint: Sha256 | None = Field(default=None, frozen=True)
    capability_fingerprint: Sha256 | None = Field(default=None, frozen=True)
    data_source_fingerprints: dict[str, Sha256] = Field(
        default_factory=dict,
        frozen=True,
    )
    started_at: UtcDatetime | None = None
    completed_at: UtcDatetime | None = None
    summary: ScientificLoopSummary | None = None
    artifact: ArtifactRef | None = None
    cost: CostSnapshot = Field(default_factory=CostSnapshot)
    acceptance_results: list[AcceptanceResult] = Field(default_factory=list)
    error: str | None = None
    created_at: UtcDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def isolate_input_snapshots(self) -> Self:
        object.__setattr__(self, "task_snapshot", deepcopy(self.task_snapshot))
        object.__setattr__(
            self, "target_snapshot", self.target_snapshot.model_copy(deep=True)
        )
        object.__setattr__(
            self,
            "data_source_fingerprints",
            deepcopy(self.data_source_fingerprints),
        )
        return self


class FeedbackDelta(StrictModel):
    item_id: ManagedId | None = None
    category: str = Field(min_length=1)
    status: FeedbackItemStatus
    severity: FeedbackSeverity
    responsible_module: str = Field(min_length=1)
    problem: str = Field(min_length=1)
    requested_actions: list[str] = Field(default_factory=list)
    acceptance_test: str | None = None
    preserve: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    source_span: str = Field(min_length=1)


class FeedbackCompilation(StrictModel):
    schema_version: SchemaVersion = 1
    compilation_id: ManagedId | None = None
    evolution_id: ManagedId | None = None
    feedback_id: ManagedId | None = None
    episode_version: EpisodeVersion | None = None
    status: CompilationStatus = "PENDING"
    items: list[FeedbackDelta] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    error: str | None = None
    created_at: UtcDatetime = Field(default_factory=utc_now)


class ExpertFeedbackRecord(ExpertFeedbackDraft):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)

    schema_version: SchemaVersion = 1
    feedback_id: ManagedId
    evolution_id: ManagedId
    episode_version: EpisodeVersion
    result_sha256: Sha256
    rubric_version: Literal["expert-review-v1"]
    raw_input: str = Field(min_length=1)
    suggested_scores: RubricScores | None = None
    hard_cap_reasons: list[str] = Field(default_factory=list)
    hard_cap_override_reason: str | None = None
    compilation_id: ManagedId | None = None
    confirmed_at: UtcDatetime = Field(default_factory=utc_now)
    supersedes_feedback_id: ManagedId | None = None

    @model_validator(mode="after")
    def validate_hard_cap_provenance(self) -> Self:
        suggested, reasons = derive_hard_cap_suggestion(self.scores, self.flags)
        if reasons:
            if self.suggested_scores is None:
                raise ValueError("hard-cap flags require suggested_scores")
            if self.suggested_scores != suggested:
                raise ValueError("suggested_scores do not match deterministic hard caps")
            if self.hard_cap_reasons != reasons:
                raise ValueError("hard_cap_reasons do not match deterministic hard caps")
            dimensions = RubricScores.model_fields
            exceeds_cap = any(
                getattr(self.scores, name) > getattr(suggested, name)
                for name in dimensions
            )
            if exceeds_cap and not (self.hard_cap_override_reason or "").strip():
                raise ValueError(
                    "scores above deterministic hard caps require an override reason"
                )
        else:
            if self.suggested_scores is not None and self.suggested_scores != self.scores:
                raise ValueError("suggested_scores must equal scores when no cap applies")
            if self.hard_cap_reasons:
                raise ValueError("hard_cap_reasons require an active hard-cap flag")

        object.__setattr__(self, "scores", self.scores.model_copy(deep=True))
        object.__setattr__(self, "flags", self.flags.model_copy(deep=True))
        object.__setattr__(self, "priority_corrections", list(self.priority_corrections))
        object.__setattr__(self, "preserved_strengths", list(self.preserved_strengths))
        object.__setattr__(self, "recommended_actions", list(self.recommended_actions))
        object.__setattr__(self, "hard_cap_reasons", list(self.hard_cap_reasons))
        return self


class RevisionPlan(StrictModel):
    schema_version: SchemaVersion = 1
    revision_id: ManagedId = Field(frozen=True)
    evolution_id: ManagedId = Field(frozen=True)
    source_version: EpisodeVersion = Field(frozen=True)
    feedback_id: ManagedId = Field(frozen=True)
    contract_changes: list[str] = Field(default_factory=list)
    evidence_requirements: list[str] = Field(default_factory=list)
    output_schema_requirements: list[str] = Field(default_factory=list)
    preserved_facts: list[str] = Field(default_factory=list)
    preserved_evidence_ids: list[ManagedId] = Field(default_factory=list)
    prohibited_repeats: list[str] = Field(default_factory=list)
    invalidated_conclusions: list[str] = Field(default_factory=list)
    invalidated_evidence_ids: list[ManagedId] = Field(default_factory=list)
    machine_acceptance_tests: list[str] = Field(default_factory=list)
    human_acceptance_tests: list[str] = Field(default_factory=list)
    strategy_arm: StrategyArm = "STATIC"
    strategy_reason: str = ""
    warnings: list[str] = Field(default_factory=list)
    unresolved_ambiguities: list[str] = Field(default_factory=list)
    has_blocking_ambiguity: bool = False
    confirmed: bool = False
    confirmed_at: UtcDatetime | None = None
    created_at: UtcDatetime = Field(default_factory=utc_now)


class StrategyVersion(StrictModel):
    schema_version: SchemaVersion = 1
    strategy_id: ManagedId = Field(frozen=True)
    evolution_id: ManagedId = Field(frozen=True)
    arm: StrategyArm
    reason: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    strategy_sha256: Sha256 | None = None
    cutoff_at: UtcDatetime | None = None
    created_at: UtcDatetime = Field(default_factory=utc_now)


class RubricScoreDelta(StrictModel):
    dimension: RubricDimension
    previous: RubricScore
    current: RubricScore
    delta: int = Field(strict=True, ge=-4, le=4)
    normalized_delta: float = Field(ge=-1.0, le=1.0)

    @model_validator(mode="after")
    def validate_delta(self) -> Self:
        expected = self.current - self.previous
        if self.delta != expected or self.normalized_delta != expected / 4:
            raise ValueError("score delta does not match previous/current scores")
        return self


class ConstraintChangeSummary(StrictModel):
    newly_passed: list[str] = Field(default_factory=list)
    newly_failed: list[str] = Field(default_factory=list)
    newly_unknown: list[str] = Field(default_factory=list)
    still_failed: list[str] = Field(default_factory=list)
    still_unknown: list[str] = Field(default_factory=list)


class EvidenceChangeSummary(StrictModel):
    added_ids: list[ManagedId] = Field(default_factory=list)
    removed_ids: list[ManagedId] = Field(default_factory=list)
    carried_ids: list[ManagedId] = Field(default_factory=list)
    invalidated_ids: list[ManagedId] = Field(default_factory=list)
    resolved_gaps: list[str] = Field(default_factory=list)
    new_gaps: list[str] = Field(default_factory=list)


class FidelityChangeSummary(StrictModel):
    upgraded_ids: list[ManagedId] = Field(default_factory=list)
    downgraded_ids: list[ManagedId] = Field(default_factory=list)
    unchanged_ids: list[ManagedId] = Field(default_factory=list)


class ArtifactDiff(StrictModel):
    previous_sha256: Sha256
    current_sha256: Sha256
    changed: bool
    size_bytes_delta: int = 0
    summary: str = ""

    @model_validator(mode="after")
    def validate_changed(self) -> Self:
        if self.changed != (self.previous_sha256 != self.current_sha256):
            raise ValueError("artifact changed flag does not match its hashes")
        return self


class CostDelta(StrictModel):
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    runtime_seconds: float = 0.0
    hpc_cost: float | None = None


class ComparisonReport(StrictModel):
    schema_version: SchemaVersion = 1
    comparison_id: ManagedId = Field(frozen=True)
    evolution_id: ManagedId = Field(frozen=True)
    previous_version: EpisodeVersion = Field(frozen=True)
    current_version: EpisodeVersion = Field(frozen=True)
    score_deltas: list[RubricScoreDelta] = Field(default_factory=list)
    acceptance_results: list[AcceptanceResult] = Field(default_factory=list)
    closed_issue_ids: list[ManagedId] = Field(default_factory=list)
    recurring_issue_ids: list[ManagedId] = Field(default_factory=list)
    new_issue_ids: list[ManagedId] = Field(default_factory=list)
    closure_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    recurrence_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    new_issue_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    constraint_changes: ConstraintChangeSummary = Field(
        default_factory=ConstraintChangeSummary
    )
    evidence_changes: EvidenceChangeSummary = Field(
        default_factory=EvidenceChangeSummary
    )
    fidelity_changes: FidelityChangeSummary = Field(
        default_factory=FidelityChangeSummary
    )
    artifact_diff: ArtifactDiff | None = None
    cost_delta: CostDelta = Field(default_factory=CostDelta)
    unresolved_human_checks: list[str] = Field(default_factory=list)
    expert_utility_delta: float | None = None
    normalized_cost_increase: float | None = None
    reward: float | None = Field(default=None, ge=-1.0, le=1.0)
    components_used: list[str] = Field(default_factory=list)
    module_credit: dict[str, float] = Field(default_factory=dict)
    created_at: UtcDatetime = Field(default_factory=utc_now)
