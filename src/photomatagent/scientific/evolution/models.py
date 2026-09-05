"""Strict persistence models for expert-guided scientific evolution."""

from __future__ import annotations

import math
import re
import secrets
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any, Literal, Never, Self

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)
from pydantic_core import core_schema

from photomatagent.scientific.loop.target import ConstraintSpec, TargetSpec

if TYPE_CHECKING:
    from photomatagent.scientific.loop.policy import ScientificLoopSummary
else:
    class _LazyScientificLoopSummary:
        """Import the concrete loop summary only when it is validated or schematized."""

        @classmethod
        def __get_pydantic_core_schema__(
            cls,
            source_type: Any,
            handler: Any,
        ) -> core_schema.CoreSchema:
            del source_type, handler
            return core_schema.no_info_plain_validator_function(
                _validate_scientific_loop_summary,
                serialization=core_schema.plain_serializer_function_ser_schema(
                    _serialize_scientific_loop_summary,
                ),
            )

        @classmethod
        def __get_pydantic_json_schema__(
            cls,
            schema: core_schema.CoreSchema,
            handler: Any,
        ) -> dict[str, Any]:
            del schema
            from photomatagent.scientific.loop.policy import (
                ScientificLoopSummary as ConcreteScientificLoopSummary,
            )

            concrete = ConcreteScientificLoopSummary.model_json_schema(
                mode=handler.mode,
            )
            return _inline_json_schema_refs(concrete)

    ScientificLoopSummary = _LazyScientificLoopSummary


def _validate_scientific_loop_summary(value: Any) -> Any:
    from photomatagent.scientific.loop.policy import ScientificLoopSummary

    return ScientificLoopSummary.model_validate(value)


def _serialize_scientific_loop_summary(value: Any) -> Any:
    return value.model_dump(mode="python")


def _inline_json_schema_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline one self-contained model schema for a lazy Pydantic adapter."""

    definitions = schema.get("$defs", {})

    def resolve(value: Any) -> Any:
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if not isinstance(value, dict):
            return value
        reference = value.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            name = reference.removeprefix("#/$defs/")
            target = definitions.get(name)
            if not isinstance(target, dict):
                raise ValueError(f"unresolved JSON schema reference: {reference}")
            return resolve(target)
        return {
            key: resolve(item)
            for key, item in value.items()
            if key != "$defs"
        }

    resolved = resolve(schema)
    if not isinstance(resolved, dict):  # pragma: no cover - Pydantic invariant
        raise TypeError("model JSON schema must be an object")
    return resolved

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
EvolutionResumeStatus = Literal[
    "CREATED",
    "AWAITING_EXPERT_FEEDBACK",
    "FEEDBACK_RECORDED",
    "REVISION_READY",
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
FeedbackCategory = Literal[
    "TASK_DEFINITION",
    "SCIENTIFIC_CORRECTNESS",
    "EVIDENCE_SUFFICIENCY",
    "NOVELTY",
    "DELIVERABLE_COMPLETENESS",
    "ACTIONABILITY",
    "SAFETY",
    "OTHER",
]
CompilationStatus = Literal["PENDING", "AVAILABLE", "UNAVAILABLE"]
AcceptanceStatus = Literal["PENDING", "PASS", "FAIL", "NEEDS_HUMAN_REVIEW"]
AcceptanceKind = Literal["MACHINE", "HUMAN"]
AcceptanceProvenance = Literal["DETERMINISTIC_EVALUATOR", "EXPERT_FEEDBACK"]
AcceptanceEvaluator = Literal[
    "CONSTRAINT_PASS", "EVIDENCE_PRESENT", "ARTIFACT_PRESENT", "UNREGISTERED"
]
ComparisonPhase = Literal["PRE_FEEDBACK", "POST_FEEDBACK"]
RewardComponent = Literal[
    "expert_utility_delta",
    "closure_rate",
    "recurrence_rate",
    "new_issue_rate",
    "normalized_cost_increase",
]
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
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=200,
        pattern=_MANAGED_ID_PATTERN,
    ),
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
FeedbackModule = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=200),
]
FeedbackText = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=4_000),
]
OptionalFeedbackText = Annotated[
    str,
    StringConstraints(strict=True, max_length=4_000),
]


def _normalize_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


UtcDatetime = Annotated[AwareDatetime, AfterValidator(_normalize_utc)]


class FrozenJsonDict(dict[str, Any]):
    """A JSON-object-compatible dict that rejects every in-place mutation."""

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> Never:
        raise TypeError("snapshot JSON values are immutable")

    __setitem__ = _immutable  # type: ignore[assignment]
    __delitem__ = _immutable  # type: ignore[assignment]
    clear = _immutable  # type: ignore[assignment]
    pop = _immutable  # type: ignore[assignment]
    popitem = _immutable  # type: ignore[assignment]
    setdefault = _immutable  # type: ignore[assignment]
    update = _immutable  # type: ignore[assignment]
    __ior__ = _immutable  # type: ignore[assignment]

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        memo[id(self)] = self
        return self


class FrozenJsonList(list[Any]):
    """A JSON-array-compatible list that rejects every in-place mutation."""

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> Never:
        raise TypeError("snapshot JSON values are immutable")

    __setitem__ = _immutable  # type: ignore[assignment]
    __delitem__ = _immutable  # type: ignore[assignment]
    append = _immutable  # type: ignore[assignment]
    clear = _immutable  # type: ignore[assignment]
    extend = _immutable  # type: ignore[assignment]
    insert = _immutable  # type: ignore[assignment]
    pop = _immutable  # type: ignore[assignment]
    remove = _immutable  # type: ignore[assignment]
    reverse = _immutable  # type: ignore[assignment]
    sort = _immutable  # type: ignore[assignment]
    __iadd__ = _immutable  # type: ignore[assignment]
    __imul__ = _immutable  # type: ignore[assignment]

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        memo[id(self)] = self
        return self


def _freeze_json(value: Any) -> Any:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python")
    if isinstance(value, dict):
        return FrozenJsonDict({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return FrozenJsonList(_freeze_json(item) for item in value)
    return value


def _freeze_json_dict(value: dict[str, Any]) -> dict[str, Any]:
    frozen = _freeze_json(value)
    if not isinstance(frozen, FrozenJsonDict):  # defensive type narrowing
        raise TypeError("snapshot must be a JSON object")
    return frozen


FrozenJsonObject = Annotated[
    dict[str, JsonValue],
    AfterValidator(_freeze_json_dict),
]


def _target_snapshot_input(value: Any) -> Any:
    if isinstance(value, TargetSpec):
        return value.model_dump(mode="python")
    return value


def _constraint_snapshot_input(value: Any) -> Any:
    if isinstance(value, ConstraintSpec):
        return value.model_dump(mode="python")
    return value


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


def new_compilation_id() -> str:
    return f"comp_{secrets.token_hex(5)}"


def new_revision_id() -> str:
    return f"rp_{secrets.token_hex(5)}"


def new_strategy_id() -> str:
    return f"strategy_{secrets.token_hex(5)}"


def new_episode_owner_token() -> str:
    """Return an unguessable, persistence-safe execution ownership token."""

    return f"owner_{secrets.token_hex(16)}"


class StrictModel(BaseModel):
    """Base contract for JSON records owned by the evolution subsystem."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class TargetConstraintSnapshot(ConstraintSpec):
    """Immutable form of one TargetSpec constraint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: JsonValue = None

    @model_validator(mode="before")
    @classmethod
    def accept_constraint_spec(cls, value: Any) -> Any:
        return _constraint_snapshot_input(value)

    @model_validator(mode="after")
    def freeze_constraint_value(self) -> Self:
        object.__setattr__(self, "value", _freeze_json(self.value))
        return self


class TargetSnapshot(TargetSpec):
    """A TargetSpec-compatible, recursively immutable execution snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operating_conditions: FrozenJsonObject = Field(default_factory=dict)
    metadata: FrozenJsonObject = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def accept_target_spec(cls, value: Any) -> Any:
        return _target_snapshot_input(value)

    @model_validator(mode="after")
    def freeze_target_values(self) -> Self:
        constraints = FrozenJsonList(
            TargetConstraintSnapshot.model_validate(item) for item in self.constraints
        )
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "objectives", FrozenJsonList(self.objectives))
        object.__setattr__(
            self,
            "operating_conditions",
            _freeze_json_dict(self.operating_conditions),
        )
        object.__setattr__(self, "metadata", _freeze_json_dict(self.metadata))
        return self

    def to_target_spec(self) -> TargetSpec:
        """Return a detached mutable TargetSpec for APIs that need one."""

        return TargetSpec.model_validate(self.model_dump(mode="python"))


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
    resolved_issue_ids: list[ManagedId] = Field(default_factory=list, max_length=100)

    @field_validator("resolved_issue_ids")
    @classmethod
    def validate_unique_resolutions(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("resolved issue IDs must be unique")
        return value


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
    kind: AcceptanceKind = "MACHINE"
    provenance: AcceptanceProvenance = "DETERMINISTIC_EVALUATOR"

    @model_validator(mode="after")
    def validate_provenance(self) -> Self:
        expected = (
            "DETERMINISTIC_EVALUATOR"
            if self.kind == "MACHINE"
            else "EXPERT_FEEDBACK"
        )
        if self.provenance != expected:
            raise ValueError("acceptance kind and provenance do not match")
        return self


class MachineAcceptanceCheck(StrictModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)

    acceptance_id: ManagedId
    description: str = Field(min_length=1, max_length=4_000)
    evaluator: AcceptanceEvaluator
    subject: str | None = Field(default=None, max_length=200)


class EvolutionTask(StrictModel):
    schema_version: SchemaVersion = Field(default=1, frozen=True)
    revision: int = Field(default=0, ge=0)
    evolution_id: ManagedId = Field(frozen=True)
    goal: str = Field(min_length=1, frozen=True)
    target: TargetSpec = Field(frozen=True)
    task_group_id: ManagedId = Field(frozen=True)
    input_sha256: Sha256 = Field(frozen=True)
    status: EvolutionStatus = "CREATED"
    resume_status: EvolutionResumeStatus | None = None
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

    @model_validator(mode="after")
    def validate_resume_checkpoint(self) -> Self:
        checkpoint_states = {"STOPPED", "BLOCKED", "BUDGET_EXHAUSTED"}
        if self.status in checkpoint_states and self.resume_status is None:
            raise ValueError(f"{self.status} tasks require resume_status")
        if self.status not in checkpoint_states and self.resume_status is not None:
            raise ValueError("resume_status is valid only for a paused task")
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
    owner_token: ManagedId | None = Field(default=None, frozen=True)
    runtime_session_id: ManagedId | None = None
    event_log_path: str | None = None
    execution_mode: ExecutionMode = Field(default="NORMAL", frozen=True)
    strategy_id: ManagedId | None = Field(default=None, frozen=True)
    strategy_arm: StrategyArm = Field(default="STATIC", frozen=True)
    scientific_state_path: str | None = None
    task_snapshot: FrozenJsonObject = Field(frozen=True)
    target_snapshot: TargetSnapshot = Field(frozen=True)
    provider: str | None = Field(default=None, frozen=True)
    model: str | None = Field(default=None, frozen=True)
    tool_surface_fingerprint: Sha256 | None = Field(default=None, frozen=True)
    capability_fingerprint: Sha256 | None = Field(default=None, frozen=True)
    data_source_fingerprints: Annotated[
        dict[str, Sha256], AfterValidator(_freeze_json_dict)
    ] = Field(
        default_factory=FrozenJsonDict,
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


class FeedbackDelta(StrictModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)

    item_id: ManagedId | None = None
    category: FeedbackCategory
    status: FeedbackItemStatus
    severity: FeedbackSeverity
    responsible_module: FeedbackModule
    problem: FeedbackText
    requested_actions: tuple[FeedbackText, ...] = Field(
        default_factory=tuple,
        max_length=20,
    )
    acceptance_test: OptionalFeedbackText | None = None
    preserve: tuple[FeedbackText, ...] = Field(
        default_factory=tuple,
        max_length=20,
    )
    confidence: float = Field(ge=0.0, le=1.0)
    source_span: FeedbackText


class FeedbackCompilation(StrictModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)

    schema_version: SchemaVersion = 1
    compilation_id: ManagedId | None = None
    evolution_id: ManagedId | None = None
    feedback_id: ManagedId | None = None
    episode_version: EpisodeVersion | None = None
    status: CompilationStatus = "PENDING"
    items: tuple[FeedbackDelta, ...] = Field(default_factory=tuple, max_length=100)
    warnings: tuple[OptionalFeedbackText, ...] = Field(
        default_factory=tuple,
        max_length=100,
    )
    provider: str | None = Field(default=None, max_length=200)
    model: str | None = Field(default=None, max_length=200)
    error: str | None = Field(default=None, max_length=1_000)
    created_at: UtcDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_compiled_record(self) -> Self:
        if self.status == "PENDING":
            return self
        required = {
            "compilation_id": self.compilation_id,
            "evolution_id": self.evolution_id,
            "feedback_id": self.feedback_id,
            "episode_version": self.episode_version,
            "provider": self.provider,
            "model": self.model,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(
                "completed compilation requires provenance: " + ", ".join(missing)
            )
        if self.status == "AVAILABLE" and self.error is not None:
            raise ValueError("available compilation cannot contain an error")
        if self.status == "UNAVAILABLE":
            if not self.error:
                raise ValueError("unavailable compilation requires an error")
            if self.items:
                raise ValueError("unavailable compilation cannot contain feedback items")
        return self


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
        object.__setattr__(self, "resolved_issue_ids", FrozenJsonList(self.resolved_issue_ids))
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
    machine_acceptance_checks: list[MachineAcceptanceCheck] = Field(
        default_factory=list
    )
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
    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)

    schema_version: SchemaVersion = 1
    comparison_id: ManagedId = Field(frozen=True)
    evolution_id: ManagedId = Field(frozen=True)
    previous_version: EpisodeVersion = Field(frozen=True)
    current_version: EpisodeVersion = Field(frozen=True)
    phase: ComparisonPhase = "PRE_FEEDBACK"
    current_feedback_id: ManagedId | None = None
    current_feedback_sha256: Sha256 | None = None
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
    expert_utility_delta: float | None = Field(default=None, ge=-1.0, le=1.0)
    normalized_cost_increase: float | None = Field(default=None, ge=-1.0, le=1.0)
    reward: float | None = Field(default=None, ge=-1.0, le=1.0)
    components_used: list[RewardComponent] = Field(
        default_factory=list,
        max_length=5,
    )
    module_credit: dict[str, float] = Field(default_factory=dict)
    created_at: UtcDatetime = Field(default_factory=utc_now)

    @field_validator("expert_utility_delta", "normalized_cost_increase", "reward")
    @classmethod
    def validate_finite_metric(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("comparison metrics must be finite")
        return value

    @field_validator("module_credit")
    @classmethod
    def validate_module_credit(cls, value: dict[str, float]) -> dict[str, float]:
        if any(
            not math.isfinite(score) or not -1.0 <= score <= 1.0
            for score in value.values()
        ):
            raise ValueError("module credit must be finite and within [-1, 1]")
        return value

    @model_validator(mode="after")
    def validate_comparison_consistency(self) -> Self:
        if len(self.components_used) != len(set(self.components_used)):
            raise ValueError("reward components must be unique")
        values = {
            "expert_utility_delta": self.expert_utility_delta,
            "closure_rate": self.closure_rate,
            "recurrence_rate": self.recurrence_rate,
            "new_issue_rate": self.new_issue_rate,
            "normalized_cost_increase": self.normalized_cost_increase,
        }
        expected = [name for name, value in values.items() if value is not None]
        if list(self.components_used) != expected:
            raise ValueError("reward components do not match available metrics")
        weighted = {
            "expert_utility_delta": 0.45,
            "closure_rate": 0.25,
            "recurrence_rate": -0.15,
            "new_issue_rate": -0.10,
            "normalized_cost_increase": -0.05,
        }
        if expected:
            denominator = sum(abs(weighted[name]) for name in expected)
            weighted_total = 0.0
            for name in expected:
                value = values[name]
                assert value is not None
                weighted_total += weighted[name] * value
            calculated = round(
                max(
                    -1.0,
                    min(
                        1.0,
                        weighted_total / denominator,
                    ),
                ),
                6,
            )
            if self.reward != calculated:
                raise ValueError("reward does not match its recorded components")
        elif self.reward is not None:
            raise ValueError("reward requires at least one recorded component")
        if self.phase == "PRE_FEEDBACK":
            if (
                self.current_feedback_id is not None
                or self.current_feedback_sha256 is not None
            ):
                raise ValueError("pre-feedback comparison cannot bind current feedback")
        elif self.current_feedback_id is None or self.current_feedback_sha256 is None:
            raise ValueError("post-feedback comparison requires exact feedback identity")
        machine = [
            result
            for result in self.acceptance_results
            if result.kind == "MACHINE" and result.status in {"PASS", "FAIL"}
        ]
        expected_closure = (
            sum(result.status == "PASS" for result in machine) / len(machine)
            if machine
            else None
        )
        if self.closure_rate != expected_closure:
            raise ValueError("closure rate must use only evaluated machine checks")
        object.__setattr__(self, "score_deltas", FrozenJsonList(self.score_deltas))
        object.__setattr__(
            self,
            "acceptance_results",
            FrozenJsonList(self.acceptance_results),
        )
        object.__setattr__(self, "closed_issue_ids", FrozenJsonList(self.closed_issue_ids))
        object.__setattr__(
            self,
            "recurring_issue_ids",
            FrozenJsonList(self.recurring_issue_ids),
        )
        object.__setattr__(self, "new_issue_ids", FrozenJsonList(self.new_issue_ids))
        object.__setattr__(
            self,
            "unresolved_human_checks",
            FrozenJsonList(self.unresolved_human_checks),
        )
        object.__setattr__(self, "components_used", FrozenJsonList(self.components_used))
        object.__setattr__(self, "module_credit", FrozenJsonDict(self.module_credit))
        return self
