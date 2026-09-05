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
    "create_experience",
    "derive_experience_id",
    "promote_experience",
]
