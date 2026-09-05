"""Deterministic baseline and gated Bayesian evolution strategy selection."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal, Self

import numpy as np
from pydantic import ConfigDict, Field, field_validator, model_validator

from photomatagent.scientific.evolution.experience import (
    StrategyObservation,
    TaskContext,
)
from photomatagent.scientific.evolution.models import (
    EvolutionTask,
    ManagedId,
    RevisionPlan,
    Sha256,
    StrategyArm,
    StrategyVersion,
    StrictModel,
    UtcDatetime,
    strategy_version_sha256,
)

ARM_ORDER: tuple[StrategyArm, ...] = (
    "STATIC",
    "EVIDENCE_FIRST",
    "DIVERSITY_FIRST",
    "UNCERTAINTY_FIRST",
)
CONTEXT_FEATURE_ORDER = (
    "context.intercept",
    "context.hard_constraint_ratio",
    "context.soft_constraint_ratio",
    "context.objective_ratio",
    "context.operating_condition_ratio",
    "context.previous_critical_gap_ratio",
)
ARM_FEATURE_ORDER = tuple(f"arm.{arm}" for arm in ARM_ORDER)
INTERACTION_FEATURE_ORDER = tuple(
    f"interaction.{context_name.removeprefix('context.')}*arm.{arm}"
    for context_name in CONTEXT_FEATURE_ORDER[1:]
    for arm in ARM_ORDER
)
FEATURE_SCHEMA = (
    *CONTEXT_FEATURE_ORDER,
    *ARM_FEATURE_ORDER,
    *INTERACTION_FEATURE_ORDER,
)
FEATURE_DIMENSION = len(FEATURE_SCHEMA)
_MIN_OBSERVATIONS = 20
_MIN_DISTINCT_TASKS = 8
PRODUCTION_SELECTOR_SEED = 0
PRIOR_PRECISION = 4.0
NOISE_VARIANCE = 0.25


def feature_vector(
    context: TaskContext,
    arm: StrategyArm,
) -> tuple[float, ...]:
    """Return the canonical 30-dimensional context/arm design row."""

    try:
        arm_index = ARM_ORDER.index(arm)
    except ValueError as exc:  # defensive for untyped callers
        raise ValueError(f"unsupported strategy arm: {arm!r}") from exc
    one_hot = tuple(1.0 if index == arm_index else 0.0 for index in range(4))
    interactions = tuple(
        value * indicator
        for value in context.values[1:]
        for indicator in one_hot
    )
    vector = (*context.values, *one_hot, *interactions)
    if len(vector) != FEATURE_DIMENSION:  # pragma: no cover - constant invariant
        raise RuntimeError("strategy feature schema dimension mismatch")
    return vector


class StrategyPosteriorSnapshot(StrictModel):
    """Immutable, self-verifying Bayesian linear posterior."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=True,
        allow_inf_nan=False,
    )

    schema_version: Literal[1] = 1
    posterior_id: ManagedId = Field(frozen=True)
    posterior_sha256: Sha256 = Field(frozen=True)
    feature_schema: tuple[str, ...]
    arm_order: tuple[StrategyArm, ...]
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    prior_precision: float = Field(strict=True, gt=0.0)
    noise_variance: float = Field(strict=True, gt=0.0)
    observation_count: int = Field(strict=True, ge=0)
    effective_training_rows: int = Field(strict=True, ge=0)
    distinct_task_groups: int = Field(strict=True, ge=0)
    training_observation_hashes: tuple[Sha256, ...]
    training_cutoff_at: UtcDatetime
    generated_at: UtcDatetime

    @field_validator("mean", "covariance", mode="before")
    @classmethod
    def validate_finite_arrays(cls, value: Any) -> Any:
        def numeric_entries(candidate: Any) -> Iterable[Any]:
            if isinstance(candidate, (list, tuple)):
                for item in candidate:
                    yield from numeric_entries(item)
            else:
                yield candidate

        entries = tuple(numeric_entries(value))
        if any(
            isinstance(item, bool) or not isinstance(item, Real)
            for item in entries
        ):
            raise ValueError("posterior arrays must contain only numeric values")
        array = np.asarray(value, dtype=float)
        if not np.isfinite(array).all():
            raise ValueError("posterior arrays must be finite")
        return value

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        if self.feature_schema != FEATURE_SCHEMA:
            raise ValueError("posterior feature schema is unsupported")
        if self.arm_order != ARM_ORDER:
            raise ValueError("posterior arm order is unsupported")
        if len(self.mean) != FEATURE_DIMENSION:
            raise ValueError("posterior mean dimension does not match feature schema")
        covariance = np.asarray(self.covariance, dtype=float)
        if covariance.shape != (FEATURE_DIMENSION, FEATURE_DIMENSION):
            raise ValueError("posterior covariance dimension does not match feature schema")
        if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12):
            raise ValueError("posterior covariance must be symmetric")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("posterior covariance must be positive definite") from exc
        if self.effective_training_rows > self.observation_count:
            raise ValueError("effective training rows cannot exceed observations")
        if self.distinct_task_groups > self.observation_count:
            raise ValueError("distinct task groups cannot exceed observations")
        if len(self.training_observation_hashes) != self.observation_count:
            raise ValueError("training observation hashes must match observation count")
        if tuple(sorted(set(self.training_observation_hashes))) != (
            self.training_observation_hashes
        ):
            raise ValueError("training observation hashes must be unique and sorted")
        if self.generated_at != self.training_cutoff_at:
            raise ValueError("posterior generated_at must equal training_cutoff_at")
        expected_hash = posterior_sha256(self)
        if self.posterior_sha256 != expected_hash:
            raise ValueError("posterior_sha256 does not match posterior content")
        if self.posterior_id != f"posterior_{expected_hash[:16]}":
            raise ValueError("posterior_id does not match posterior content")
        return self


def posterior_sha256(snapshot: StrategyPosteriorSnapshot | dict[str, Any]) -> str:
    if isinstance(snapshot, StrategyPosteriorSnapshot):
        payload = snapshot.model_dump(
            mode="json", exclude={"posterior_id", "posterior_sha256"}
        )
    else:
        payload = {
            key: value
            for key, value in snapshot.items()
            if key not in {"posterior_id", "posterior_sha256"}
        }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=lambda value: value.isoformat(),
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class BayesianSelectorDiagnostics:
    enabled: bool
    observation_count: int
    effective_training_rows: int
    distinct_tasks: int
    minimum_observations: int = _MIN_OBSERVATIONS
    minimum_distinct_tasks: int = _MIN_DISTINCT_TASKS
    incomplete_observation_chains: tuple[str, ...] = ()


class BayesianLinearStrategySelector:
    """Bayesian ridge Thompson selector with a conservative activation gate."""

    def __init__(
        self,
        *,
        seed: int = PRODUCTION_SELECTOR_SEED,
        prior_precision: float = PRIOR_PRECISION,
        noise_variance: float = NOISE_VARIANCE,
        fallback: FixedStrategySelector | None = None,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        for name, value in (
            ("prior_precision", prior_precision),
            ("noise_variance", noise_variance),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be a finite positive number")
        self.seed = seed
        self.prior_precision = float(prior_precision)
        self.noise_variance = float(noise_variance)
        self.fallback = fallback or FixedStrategySelector()
        self.posterior: StrategyPosteriorSnapshot | None = None
        self.diagnostics = BayesianSelectorDiagnostics(False, 0, 0, 0)

    @property
    def enabled(self) -> bool:
        return self.diagnostics.enabled

    @classmethod
    def from_posterior(
        cls,
        posterior: StrategyPosteriorSnapshot,
        *,
        seed: int = PRODUCTION_SELECTOR_SEED,
    ) -> Self:
        """Rehydrate a selector from one immutable historical posterior."""

        if (
            posterior.prior_precision != PRIOR_PRECISION
            or posterior.noise_variance != NOISE_VARIANCE
        ):
            raise ValueError("historical posterior uses non-production hyperparameters")
        selector = cls(
            seed=seed,
            prior_precision=PRIOR_PRECISION,
            noise_variance=NOISE_VARIANCE,
        )
        if (
            posterior.observation_count < _MIN_OBSERVATIONS
            or posterior.distinct_task_groups < _MIN_DISTINCT_TASKS
        ):
            raise ValueError("historical posterior does not satisfy the learning gate")
        selector.posterior = posterior
        selector.diagnostics = BayesianSelectorDiagnostics(
            True,
            posterior.observation_count,
            posterior.effective_training_rows,
            posterior.distinct_task_groups,
        )
        return selector

    def fit(
        self,
        observations: Iterable[StrategyObservation],
        *,
        incomplete_observation_chains: Iterable[str] = (),
    ) -> Self:
        by_comparison: dict[str, StrategyObservation] = {}
        by_id: dict[str, StrategyObservation] = {}
        for raw in observations:
            observation = StrategyObservation.model_validate(
                raw.model_dump(mode="python")
            )
            prior_id = by_id.get(observation.observation_id)
            if prior_id is not None and prior_id != observation:
                raise ValueError(
                    f"conflicting strategy observation {observation.observation_id}"
                )
            prior_comparison = by_comparison.get(observation.comparison_id)
            if prior_comparison is not None and prior_comparison != observation:
                raise ValueError(
                    f"conflicting strategy observation for comparison "
                    f"{observation.comparison_id}"
                )
            by_id[observation.observation_id] = observation
            by_comparison[observation.comparison_id] = observation
        unique = tuple(sorted(by_id.values(), key=lambda item: item.observation_sha256))
        distinct_tasks = len({item.task_group_id for item in unique})
        grouped: dict[
            tuple[str, StrategyArm, tuple[float, ...]], list[StrategyObservation]
        ] = defaultdict(list)
        for item in unique:
            grouped[(item.task_group_id, item.strategy_arm, item.context.values)].append(
                item
            )
        rows = sorted(grouped.items(), key=lambda item: item[0])
        effective_rows = len(rows)
        incomplete = tuple(sorted(set(incomplete_observation_chains)))
        enabled = (
            len(unique) >= _MIN_OBSERVATIONS
            and distinct_tasks >= _MIN_DISTINCT_TASKS
            and not incomplete
        )
        self.diagnostics = BayesianSelectorDiagnostics(
            enabled,
            len(unique),
            effective_rows,
            distinct_tasks,
            incomplete_observation_chains=incomplete,
        )
        self.posterior = None
        if not enabled:
            return self

        rows_per_group: dict[str, int] = defaultdict(int)
        for (group_id, _arm, _context), _members in rows:
            rows_per_group[group_id] += 1
        design_rows: list[tuple[float, ...]] = []
        rewards: list[float] = []
        weights: list[float] = []
        for (group_id, arm, _context_values), members in rows:
            design_rows.append(feature_vector(members[0].context, arm))
            rewards.append(sum(item.reward for item in members) / len(members))
            weights.append(1.0 / rows_per_group[group_id])
        design = np.asarray(design_rows, dtype=float)
        target = np.asarray(rewards, dtype=float)
        sqrt_weights = np.sqrt(np.asarray(weights, dtype=float))
        weighted_design = design * sqrt_weights[:, None]
        weighted_target = target * sqrt_weights
        precision = (
            self.prior_precision * np.eye(FEATURE_DIMENSION, dtype=float)
            + (weighted_design.T @ weighted_design) / self.noise_variance
        )
        try:
            np.linalg.cholesky(precision)
            covariance = np.linalg.solve(
                precision, np.eye(FEATURE_DIMENSION, dtype=float)
            )
        except np.linalg.LinAlgError as exc:  # pragma: no cover - ridge defensive path
            raise ValueError("Bayesian strategy precision is not positive definite") from exc
        covariance = (covariance + covariance.T) / 2.0
        mean = np.linalg.solve(
            precision,
            (weighted_design.T @ weighted_target) / self.noise_variance,
        )
        if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
            raise ValueError("Bayesian strategy posterior contains non-finite values")
        training_cutoff_at = max(item.created_at for item in unique)
        data: dict[str, Any] = {
            "schema_version": 1,
            "feature_schema": FEATURE_SCHEMA,
            "arm_order": ARM_ORDER,
            "mean": tuple(float(value) for value in mean),
            "covariance": tuple(
                tuple(float(value) for value in row) for row in covariance
            ),
            "prior_precision": self.prior_precision,
            "noise_variance": self.noise_variance,
            "observation_count": len(unique),
            "effective_training_rows": effective_rows,
            "distinct_task_groups": distinct_tasks,
            "training_observation_hashes": tuple(
                sorted(item.observation_sha256 for item in unique)
            ),
            "training_cutoff_at": training_cutoff_at,
            "generated_at": training_cutoff_at,
        }
        draft = StrategyPosteriorSnapshot.model_construct(
            **data,
            posterior_id="posterior_placeholder",
            posterior_sha256="0" * 64,
        )
        digest = posterior_sha256(draft)
        self.posterior = StrategyPosteriorSnapshot.model_validate(
            {
                **data,
                "posterior_id": f"posterior_{digest[:16]}",
                "posterior_sha256": digest,
            }
        )
        return self

    def select(
        self,
        task: EvolutionTask,
        plan: RevisionPlan,
        context: TaskContext,
    ) -> StrategyVersion:
        if plan.evolution_id != task.evolution_id:
            raise ValueError(
                "revision plan and task must belong to the same evolution task"
            )
        if not self.enabled or self.posterior is None:
            return self.fallback.select(task, plan)
        mean = np.asarray(self.posterior.mean, dtype=float)
        covariance = np.asarray(self.posterior.covariance, dtype=float)
        sampling_posterior_payload = {
            "feature_schema": self.posterior.feature_schema,
            "arm_order": self.posterior.arm_order,
            "mean": self.posterior.mean,
            "covariance": self.posterior.covariance,
            "prior_precision": self.posterior.prior_precision,
            "noise_variance": self.posterior.noise_variance,
            "training_observation_hashes": (
                self.posterior.training_observation_hashes
            ),
        }
        sampling_posterior_sha256 = hashlib.sha256(
            json.dumps(
                sampling_posterior_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        decision_payload = {
            "base_seed": self.seed,
            "posterior_sha256": self.posterior.posterior_sha256,
            "evolution_id": task.evolution_id,
            "revision_id": plan.revision_id,
            "context": context.model_dump(mode="json"),
        }
        decision_digest = hashlib.sha256(
            json.dumps(
                decision_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        decision_seed = int(decision_digest[:16], 16)
        rng = np.random.default_rng(decision_seed)
        coefficients = rng.multivariate_normal(mean, covariance, check_valid="raise")
        scores = {
            arm: float(np.dot(feature_vector(context, arm), coefficients))
            for arm in ARM_ORDER
        }
        selected_arm = max(
            ARM_ORDER,
            key=lambda arm: (scores[arm], -ARM_ORDER.index(arm)),
        )
        parameters: dict[str, Any] = {
            "selector": "bayesian-linear-thompson-v1",
            "revision_id": plan.revision_id,
            "posterior_id": self.posterior.posterior_id,
            "posterior_sha256": self.posterior.posterior_sha256,
            "seed": self.seed,
            "decision_seed": decision_seed,
            "sampling_posterior_sha256": sampling_posterior_sha256,
            "context": context.model_dump(mode="json"),
            "feature_schema": list(FEATURE_SCHEMA),
        }
        reason = (
            f"bayesian-linear-thompson-v1: selected {selected_arm} from "
            f"posterior {self.posterior.posterior_id}"
        )
        payload: dict[str, Any] = {
            "evolution_id": task.evolution_id,
            "revision_id": plan.revision_id,
            "arm": selected_arm,
            "reason": reason,
            "parameters": parameters,
            "cutoff_at": self.posterior.training_cutoff_at.isoformat(),
        }
        digest = strategy_version_sha256(payload)
        return StrategyVersion(
            strategy_id=f"strategy_{digest[:10]}",
            evolution_id=task.evolution_id,
            arm=selected_arm,
            reason=reason,
            parameters=parameters,
            strategy_sha256=digest,
            cutoff_at=self.posterior.training_cutoff_at,
            created_at=self.posterior.generated_at,
        )


class FixedStrategySelector:
    """Materialize the fixed-v1 choice already recorded by the planner."""

    def select(self, task: EvolutionTask, plan: RevisionPlan) -> StrategyVersion:
        if plan.evolution_id != task.evolution_id:
            raise ValueError(
                "revision plan and task must belong to the same evolution task"
            )
        parameters: dict[str, Any] = {
            "selector": "fixed-v1",
            "revision_id": plan.revision_id,
        }
        payload: dict[str, Any] = {
            "evolution_id": task.evolution_id,
            "revision_id": plan.revision_id,
            "arm": plan.strategy_arm,
            "reason": plan.strategy_reason[:1_000],
            "parameters": parameters,
        }
        digest = strategy_version_sha256(payload)
        return StrategyVersion(
            strategy_id=f"strategy_{digest[:10]}",
            evolution_id=task.evolution_id,
            arm=plan.strategy_arm,
            reason=plan.strategy_reason[:1_000],
            parameters=parameters,
            strategy_sha256=digest,
            cutoff_at=plan.created_at,
            created_at=plan.created_at,
        )


__all__ = [
    "ARM_ORDER",
    "BayesianLinearStrategySelector",
    "BayesianSelectorDiagnostics",
    "CONTEXT_FEATURE_ORDER",
    "FEATURE_DIMENSION",
    "FEATURE_SCHEMA",
    "FixedStrategySelector",
    "NOISE_VARIANCE",
    "PRIOR_PRECISION",
    "PRODUCTION_SELECTOR_SEED",
    "StrategyPosteriorSnapshot",
    "TaskContext",
    "feature_vector",
    "posterior_sha256",
]
