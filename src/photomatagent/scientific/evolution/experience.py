"""Versioned experience observations and explicit maturity promotion rules."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Self

from pydantic import ConfigDict, Field, model_validator

from photomatagent.scientific.evolution.models import (
    ComparisonReport,
    ExperienceMaturity,
    ManagedId,
    SchemaVersion,
    StrictModel,
    UtcDatetime,
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
    maturity: ExperienceMaturity = "OBSERVATION"
    observations: tuple[ExperienceEvidence, ...] = Field(
        default_factory=tuple,
        min_length=1,
    )
    user_approved_for_reuse: bool = False
    created_at: UtcDatetime

    @model_validator(mode="after")
    def validate_reuse_approval(self) -> Self:
        if self.maturity == "REUSABLE_SKILL" and not self.user_approved_for_reuse:
            raise ValueError("REUSABLE_SKILL requires explicit user approval")
        if self.maturity != "REUSABLE_SKILL" and self.user_approved_for_reuse:
            raise ValueError("reuse approval is valid only for REUSABLE_SKILL")
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
        experience_id=_experience_id("OBSERVATION", (observation,)),
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
    distinct_tasks = {item.task_group_id for item in observations}
    rewards = [item.reward for item in observations]
    if to == "HYPOTHESIS":
        if len(distinct_tasks) < 2 or any(
            value is None or value <= 0 for value in rewards
        ):
            raise ExperiencePromotionError(
                "HYPOTHESIS requires positive rewards from at least two "
                "distinct task groups"
            )
    else:
        if to == "REUSABLE_SKILL" and not user_approved:
            raise ExperiencePromotionError(
                "REUSABLE_SKILL requires explicit user approval"
            )
        if len(distinct_tasks) < 5:
            raise ExperiencePromotionError(
                "validated experience requires at least five distinct tasks"
            )
        if any(item.safety_or_fabrication_failure for item in observations):
            raise ExperiencePromotionError(
                "validated experience cannot contain a safety or fabrication failure"
            )
        known_rewards = [value for value in rewards if value is not None]
        if (
            len(known_rewards) != len(rewards)
            or sum(known_rewards) / len(known_rewards) <= 0
        ):
            raise ExperiencePromotionError(
                "validated experience requires a positive average known reward"
            )
    return ExperienceRecord(
        experience_id=_experience_id(to, observations),
        evolution_id=experience.evolution_id,
        base_experience_id=experience.experience_id,
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


def _experience_id(
    maturity: ExperienceMaturity,
    observations: Sequence[ExperienceEvidence],
) -> str:
    payload = {
        "maturity": maturity,
        "observations": [
            item.model_dump(mode="json")
            for item in sorted(observations, key=lambda value: value.comparison_id)
        ],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"exp_{digest[:16]}"


__all__ = [
    "ExperienceEvidence",
    "ExperiencePromotionError",
    "ExperienceRecord",
    "create_experience",
    "promote_experience",
]
