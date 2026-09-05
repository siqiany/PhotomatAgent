"""Versioned experience observations and explicit maturity promotion rules."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from photomatagent.scientific.evolution.models import (
    ComparisonReport,
    EvolutionTask,
    ExperienceMaturity,
    ManagedId,
    SchemaVersion,
    Sha256,
    StrategyArm,
    StrategyVersion,
    StrictModel,
    UtcDatetime,
)
from photomatagent.scientific.loop import TargetSpec


class TaskContext(StrictModel):
    """Six bounded task features used by every strategy arm."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=True,
        allow_inf_nan=False,
    )

    intercept: float = Field(default=1.0, strict=True, ge=1.0, le=1.0)
    hard_constraint_ratio: float = Field(strict=True, ge=0.0, le=1.0)
    soft_constraint_ratio: float = Field(strict=True, ge=0.0, le=1.0)
    objective_ratio: float = Field(strict=True, ge=0.0, le=1.0)
    operating_condition_ratio: float = Field(strict=True, ge=0.0, le=1.0)
    previous_critical_gap_ratio: float = Field(strict=True, ge=0.0, le=1.0)

    @classmethod
    def from_target(
        cls,
        target: TargetSpec,
        *,
        previous_critical_gap_count: int = 0,
    ) -> Self:
        if (
            isinstance(previous_critical_gap_count, bool)
            or not isinstance(previous_critical_gap_count, int)
            or previous_critical_gap_count < 0
        ):
            raise ValueError(
                "previous_critical_gap_count must be a non-negative integer"
            )

        def scaled(count: int) -> float:
            return min(1.0, max(0.0, count / 10.0))

        return cls(
            hard_constraint_ratio=scaled(len(target.hard_constraints())),
            soft_constraint_ratio=scaled(len(target.soft_constraints())),
            objective_ratio=scaled(len(target.objectives)),
            operating_condition_ratio=scaled(len(target.operating_conditions)),
            previous_critical_gap_ratio=scaled(previous_critical_gap_count),
        )

    @property
    def values(self) -> tuple[float, float, float, float, float, float]:
        return (
            self.intercept,
            self.hard_constraint_ratio,
            self.soft_constraint_ratio,
            self.objective_ratio,
            self.operating_condition_ratio,
            self.previous_critical_gap_ratio,
        )


class StrategyObservation(StrictModel):
    """Immutable, provenance-complete reviewed reward for strategy learning."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=True,
        allow_inf_nan=False,
    )

    schema_version: SchemaVersion = 1
    observation_id: ManagedId = Field(frozen=True)
    observation_sha256: Sha256 = Field(frozen=True)
    evolution_id: ManagedId = Field(frozen=True)
    task_group_id: ManagedId = Field(frozen=True)
    comparison_id: ManagedId = Field(frozen=True)
    comparison_sha256: Sha256 = Field(frozen=True)
    experience_id: ManagedId = Field(frozen=True)
    experience_sha256: Sha256 = Field(frozen=True)
    previous_version: str = Field(frozen=True, pattern=r"^v\d{3}$")
    current_version: str = Field(frozen=True, pattern=r"^v\d{3}$")
    comparison_phase: Literal["POST_FEEDBACK"] = "POST_FEEDBACK"
    current_feedback_id: ManagedId = Field(frozen=True)
    current_feedback_sha256: Sha256 = Field(frozen=True)
    current_compilation_id: ManagedId = Field(frozen=True)
    current_compilation_sha256: Sha256 = Field(frozen=True)
    strategy_id: ManagedId = Field(frozen=True)
    strategy_arm: StrategyArm = Field(frozen=True)
    strategy_sha256: Sha256 = Field(frozen=True)
    strategy_record_sha256: Sha256 = Field(frozen=True)
    strategy_cutoff_at: UtcDatetime = Field(frozen=True)
    source_execution_mode: Literal["NORMAL", "CARRY_VERIFIED_EVIDENCE"]
    context: TaskContext
    reward: float = Field(strict=True, ge=-1.0, le=1.0)
    created_at: UtcDatetime

    @field_validator("reward")
    @classmethod
    def validate_finite_reward(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("strategy observation reward must be finite")
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        expected_hash = strategy_observation_sha256(self)
        if self.observation_sha256 != expected_hash:
            raise ValueError("observation_sha256 does not match observation content")
        if self.observation_id != f"obs_{expected_hash[:16]}":
            raise ValueError("observation_id does not match observation content")
        return self


def canonical_record_sha256(record: StrictModel) -> str:
    """Hash a strict evolution record's complete JSON representation."""

    return hashlib.sha256(
        json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def strategy_observation_sha256(
    observation: StrategyObservation | dict[str, Any],
) -> str:
    if isinstance(observation, StrategyObservation):
        payload = observation.model_dump(
            mode="json",
            exclude={"observation_id", "observation_sha256"},
        )
    else:
        payload = {
            key: value
            for key, value in observation.items()
            if key not in {"observation_id", "observation_sha256"}
        }

    def json_default(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        return value.isoformat()

    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=json_default,
        ).encode("utf-8")
    ).hexdigest()


def create_strategy_observation(
    *,
    task: EvolutionTask,
    comparison: ComparisonReport,
    experience: ExperienceRecord,
    strategy: StrategyVersion,
    context: TaskContext,
    source_execution_mode: Literal[
        "NORMAL", "CARRY_VERIFIED_EVIDENCE", "FRESH_EVALUATION"
    ],
) -> StrategyObservation:
    """Bind one reviewed Task-11 experience to its frozen strategy provenance."""

    if comparison.phase != "POST_FEEDBACK":
        raise ValueError("strategy learning requires a reviewed POST_FEEDBACK comparison")
    if source_execution_mode == "FRESH_EVALUATION":
        raise ValueError("fresh evaluations cannot train the strategy selector")
    if comparison.reward is None:
        raise ValueError("strategy learning requires a known Task-11 reward")
    if (
        comparison.expert_utility_delta is None
        or "expert_utility_delta" not in comparison.components_used
    ):
        raise ValueError(
            "strategy learning reward must include reviewed expert utility"
        )
    if (
        comparison.current_feedback_id is None
        or comparison.current_feedback_sha256 is None
    ):
        raise ValueError("strategy learning requires current expert feedback provenance")
    if (
        comparison.current_compilation_id is None
        or comparison.current_compilation_sha256 is None
    ):
        raise ValueError("strategy learning requires current feedback compilation provenance")
    if task.evolution_id != comparison.evolution_id:
        raise ValueError("comparison and task must belong to the same evolution task")
    if strategy.evolution_id != task.evolution_id:
        raise ValueError("strategy and task must belong to the same evolution task")
    if strategy.strategy_sha256 is None or strategy.cutoff_at is None:
        raise ValueError("strategy learning requires frozen strategy hash and cutoff")
    if experience.evolution_id != task.evolution_id:
        raise ValueError("experience and task must belong to the same evolution task")
    matching_evidence = [
        item
        for item in experience.observations
        if item.comparison_id == comparison.comparison_id
    ]
    if len(matching_evidence) != 1:
        raise ValueError("experience must contain exactly one matching comparison")
    evidence = matching_evidence[0]
    if (
        evidence.task_group_id != task.task_group_id
        or evidence.reward != comparison.reward
    ):
        raise ValueError("experience evidence does not match task group and reward")
    data: dict[str, Any] = {
        "schema_version": 1,
        "evolution_id": task.evolution_id,
        "task_group_id": task.task_group_id,
        "comparison_id": comparison.comparison_id,
        "comparison_sha256": canonical_record_sha256(comparison),
        "experience_id": experience.experience_id,
        "experience_sha256": canonical_record_sha256(experience),
        "previous_version": comparison.previous_version,
        "current_version": comparison.current_version,
        "comparison_phase": comparison.phase,
        "current_feedback_id": comparison.current_feedback_id,
        "current_feedback_sha256": comparison.current_feedback_sha256,
        "current_compilation_id": comparison.current_compilation_id,
        "current_compilation_sha256": comparison.current_compilation_sha256,
        "strategy_id": strategy.strategy_id,
        "strategy_arm": strategy.arm,
        "strategy_sha256": strategy.strategy_sha256,
        "strategy_record_sha256": canonical_record_sha256(strategy),
        "strategy_cutoff_at": strategy.cutoff_at,
        "source_execution_mode": source_execution_mode,
        "context": context,
        "reward": comparison.reward,
        "created_at": comparison.created_at,
    }
    draft = StrategyObservation.model_construct(
        **data,
        observation_id="obs_placeholder",
        observation_sha256="0" * 64,
    )
    digest = strategy_observation_sha256(draft)
    return StrategyObservation.model_validate(
        {
            **data,
            "observation_id": f"obs_{digest[:16]}",
            "observation_sha256": digest,
        }
    )


class ExperiencePromotionError(ValueError):
    """Raised when evidence does not satisfy an explicit maturity request."""


class ExperienceEvidence(StrictModel):
    """One episode-pair observation; module credit never multiplies this row."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)

    comparison_id: ManagedId
    task_group_id: ManagedId
    reward: float | None = Field(default=None, ge=-1.0, le=1.0)
    safety_or_fabrication_failure: bool = False


class ExperienceRecord(StrictModel):
    """Immutable snapshot of accumulated evidence at one maturity level."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)

    schema_version: SchemaVersion = 1
    experience_id: ManagedId
    evolution_id: ManagedId
    base_experience_id: ManagedId | None = None
    previous_maturity: ExperienceMaturity | None = None
    maturity: ExperienceMaturity = "OBSERVATION"
    observations: tuple[ExperienceEvidence, ...] = Field(
        default_factory=tuple,
        min_length=1,
    )
    user_approved_for_reuse: bool = False
    created_at: UtcDatetime

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        try:
            _validate_maturity(
                self.maturity,
                self.observations,
                user_approved=self.user_approved_for_reuse,
            )
        except ExperiencePromotionError as exc:
            raise ValueError(str(exc)) from exc
        if self.maturity == "OBSERVATION":
            if self.base_experience_id is not None or self.previous_maturity is not None:
                raise ValueError("OBSERVATION cannot reference a base experience")
        else:
            allowed = {
                "HYPOTHESIS": "OBSERVATION",
                "VALIDATED_EXPERIENCE": "HYPOTHESIS",
                "REUSABLE_SKILL": "VALIDATED_EXPERIENCE",
            }
            if (
                self.base_experience_id is None
                or self.previous_maturity != allowed[self.maturity]
            ):
                raise ValueError("experience requires a legal transition/base reference")
        if self.maturity == "REUSABLE_SKILL" and not self.user_approved_for_reuse:
            raise ValueError("REUSABLE_SKILL requires explicit user approval")
        if self.maturity != "REUSABLE_SKILL" and self.user_approved_for_reuse:
            raise ValueError("reuse approval is valid only for REUSABLE_SKILL")
        expected_id = derive_experience_id(
            evolution_id=self.evolution_id,
            maturity=self.maturity,
            observations=self.observations,
            base_experience_id=self.base_experience_id,
            previous_maturity=self.previous_maturity,
            user_approved_for_reuse=self.user_approved_for_reuse,
            created_at=self.created_at,
        )
        if self.experience_id != expected_id:
            raise ValueError("experience_id does not match record content")
        return self


def create_experience(
    comparison: ComparisonReport,
    *,
    task_group_id: str | None = None,
    safety_or_fabrication_failure: bool = False,
) -> ExperienceRecord:
    """Create exactly one low-confidence observation from one comparison."""

    observation = ExperienceEvidence(
        comparison_id=comparison.comparison_id,
        task_group_id=task_group_id or comparison.evolution_id,
        reward=comparison.reward,
        safety_or_fabrication_failure=safety_or_fabrication_failure,
    )
    return ExperienceRecord(
        experience_id=derive_experience_id(
            evolution_id=comparison.evolution_id,
            maturity="OBSERVATION",
            observations=(observation,),
            created_at=comparison.created_at,
        ),
        evolution_id=comparison.evolution_id,
        maturity="OBSERVATION",
        observations=(observation,),
        created_at=comparison.created_at,
    )


def promote_experience(
    experience: ExperienceRecord,
    *,
    to: ExperienceMaturity,
    evidence: Sequence[ExperienceEvidence | ExperienceRecord],
    user_approved: bool = False,
) -> ExperienceRecord:
    """Return a new immutable maturity snapshot after exact threshold checks.

    This function records approval only.  It intentionally has no filesystem
    or Skill loader dependency and can never edit a repository Skill.
    """

    allowed = {
        "OBSERVATION": "HYPOTHESIS",
        "HYPOTHESIS": "VALIDATED_EXPERIENCE",
        "VALIDATED_EXPERIENCE": "REUSABLE_SKILL",
        "REUSABLE_SKILL": None,
    }
    if allowed[experience.maturity] != to:
        raise ExperiencePromotionError(
            f"invalid experience maturity transition: {experience.maturity} -> {to}"
        )
    observations = _merge_observations(experience, evidence)
    _validate_maturity(to, observations, user_approved=user_approved)
    experience_id = derive_experience_id(
        evolution_id=experience.evolution_id,
        maturity=to,
        observations=observations,
        base_experience_id=experience.experience_id,
        previous_maturity=experience.maturity,
        user_approved_for_reuse=to == "REUSABLE_SKILL" and user_approved,
        created_at=experience.created_at,
    )
    return ExperienceRecord(
        experience_id=experience_id,
        evolution_id=experience.evolution_id,
        base_experience_id=experience.experience_id,
        previous_maturity=experience.maturity,
        maturity=to,
        observations=observations,
        user_approved_for_reuse=to == "REUSABLE_SKILL" and user_approved,
        created_at=experience.created_at,
    )


def _merge_observations(
    experience: ExperienceRecord,
    evidence: Sequence[ExperienceEvidence | ExperienceRecord],
) -> tuple[ExperienceEvidence, ...]:
    by_comparison = {
        item.comparison_id: item for item in experience.observations
    }
    for value in evidence:
        candidates = (
            value.observations
            if isinstance(value, ExperienceRecord)
            else (value,)
        )
        for item in candidates:
            existing = by_comparison.get(item.comparison_id)
            if existing is not None and existing != item:
                raise ExperiencePromotionError(
                    f"comparison {item.comparison_id} has conflicting "
                    "experience evidence"
                )
            by_comparison[item.comparison_id] = item
    return tuple(by_comparison[key] for key in sorted(by_comparison))


def derive_experience_id(
    *,
    evolution_id: str,
    maturity: ExperienceMaturity,
    observations: Sequence[ExperienceEvidence],
    base_experience_id: str | None = None,
    previous_maturity: ExperienceMaturity | None = None,
    user_approved_for_reuse: bool = False,
    created_at: UtcDatetime,
) -> str:
    payload = {
        "evolution_id": evolution_id,
        "maturity": maturity,
        "base_experience_id": base_experience_id,
        "previous_maturity": previous_maturity,
        "user_approved_for_reuse": user_approved_for_reuse,
        "created_at": created_at.isoformat(),
        "observations": [
            item.model_dump(mode="json")
            for item in sorted(observations, key=lambda value: value.comparison_id)
        ],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"exp_{digest[:16]}"


def _group_rewards(
    observations: Sequence[ExperienceEvidence],
) -> dict[str, float | None]:
    grouped: dict[str, list[float | None]] = {}
    for item in observations:
        grouped.setdefault(item.task_group_id, []).append(item.reward)
    return {
        group: (
            sum(value for value in rewards if value is not None) / len(rewards)
            if all(value is not None for value in rewards)
            else None
        )
        for group, rewards in grouped.items()
    }


def _validate_maturity(
    maturity: ExperienceMaturity,
    observations: Sequence[ExperienceEvidence],
    *,
    user_approved: bool,
) -> None:
    groups = _group_rewards(observations)
    if maturity == "OBSERVATION":
        if len(observations) != 1:
            raise ExperiencePromotionError("OBSERVATION requires exactly one comparison")
        return
    if maturity == "HYPOTHESIS":
        if len(groups) < 2 or any(
            value is None or value <= 0 for value in groups.values()
        ):
            raise ExperiencePromotionError(
                "HYPOTHESIS requires positive per-group results from at least two distinct task groups"
            )
        return
    if len(groups) < 5:
        raise ExperiencePromotionError(
            "validated experience requires at least five distinct task groups"
        )
    if any(item.safety_or_fabrication_failure for item in observations):
        raise ExperiencePromotionError(
            "validated experience cannot contain a safety or fabrication failure"
        )
    if any(value is None for value in groups.values()) or (
        sum(value for value in groups.values() if value is not None) / len(groups) <= 0
    ):
        raise ExperiencePromotionError(
            "validated experience requires a positive average known reward after averaging once per task group"
        )
    if maturity == "REUSABLE_SKILL" and not user_approved:
        raise ExperiencePromotionError("REUSABLE_SKILL requires explicit user approval")


__all__ = [
    "ExperienceEvidence",
    "ExperiencePromotionError",
    "ExperienceRecord",
    "StrategyObservation",
    "TaskContext",
    "canonical_record_sha256",
    "create_experience",
    "create_strategy_observation",
    "derive_experience_id",
    "promote_experience",
    "strategy_observation_sha256",
]
