"""Persistent expert-feedback evolution contracts with lazy public exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from photomatagent.scientific.evolution.models import (
        AcceptanceResult,
        AcceptanceStatus,
        ArtifactDiff,
        ArtifactRef,
        ComparisonReport,
        CompilationStatus,
        ConstraintChangeSummary,
        CostDelta,
        CostSnapshot,
        EpisodeRecord,
        EpisodeStatus,
        EpisodeVersion,
        EvidenceChangeSummary,
        EvolutionResumeStatus,
        EvolutionStatus,
        EvolutionTask,
        ExecutionMode,
        ExperienceMaturity,
        ExpertFeedbackDraft,
        ExpertFeedbackRecord,
        FeedbackCategory,
        FeedbackCompilation,
        FeedbackDelta,
        FeedbackItemStatus,
        FeedbackSeverity,
        FidelityChangeSummary,
        ManagedId,
        RevisionPlan,
        RubricDimension,
        RubricFlags,
        RubricScoreDelta,
        RubricScores,
        SchemaVersion,
        Sha256,
        StrategyArm,
        StrategyVersion,
        TargetConstraintSnapshot,
        TargetSnapshot,
        UtcDatetime,
        new_compilation_id,
        new_episode_id,
        new_evolution_id,
        new_feedback_id,
        new_revision_id,
        new_strategy_id,
        utc_now,
        validate_managed_id,
    )
    from photomatagent.scientific.evolution.rubric import (
        RUBRIC_ANCHORS,
        RUBRIC_DIMENSIONS,
        RUBRIC_VERSION,
        HardCapAssessment,
        assess_hard_caps,
        expert_utility,
    )
    from photomatagent.scientific.evolution.service import (
        ALLOWED_TRANSITIONS,
        ArtifactMismatchError,
        EvolutionOperationConflict,
        EvolutionService,
        EvolutionServiceError,
        InvalidEvolutionTransition,
        MutationResult,
    )
    from photomatagent.scientific.evolution.store import (
        EvolutionAlreadyExistsError,
        EvolutionConflictError,
        EvolutionCorruptRecordError,
        EvolutionLockError,
        EvolutionStore,
        EvolutionStoreError,
        EvolutionTransaction,
        EvolutionUnsupportedSchemaError,
    )

_MODEL_EXPORTS = (
    "AcceptanceResult",
    "AcceptanceStatus",
    "ArtifactDiff",
    "ArtifactRef",
    "ComparisonReport",
    "CompilationStatus",
    "ConstraintChangeSummary",
    "CostDelta",
    "CostSnapshot",
    "EpisodeRecord",
    "EpisodeStatus",
    "EpisodeVersion",
    "EvidenceChangeSummary",
    "EvolutionStatus",
    "EvolutionResumeStatus",
    "EvolutionTask",
    "ExecutionMode",
    "ExperienceMaturity",
    "ExpertFeedbackDraft",
    "ExpertFeedbackRecord",
    "FeedbackCategory",
    "FeedbackCompilation",
    "FeedbackDelta",
    "FeedbackItemStatus",
    "FeedbackSeverity",
    "FidelityChangeSummary",
    "ManagedId",
    "RevisionPlan",
    "RubricDimension",
    "RubricFlags",
    "RubricScoreDelta",
    "RubricScores",
    "SchemaVersion",
    "Sha256",
    "StrategyArm",
    "StrategyVersion",
    "TargetConstraintSnapshot",
    "TargetSnapshot",
    "UtcDatetime",
    "new_episode_id",
    "new_compilation_id",
    "new_evolution_id",
    "new_feedback_id",
    "new_revision_id",
    "new_strategy_id",
    "utc_now",
    "validate_managed_id",
)
_RUBRIC_EXPORTS = (
    "RUBRIC_ANCHORS",
    "RUBRIC_DIMENSIONS",
    "RUBRIC_VERSION",
    "HardCapAssessment",
    "assess_hard_caps",
    "expert_utility",
)
_SERVICE_EXPORTS = (
    "ALLOWED_TRANSITIONS",
    "ArtifactMismatchError",
    "EvolutionOperationConflict",
    "EvolutionService",
    "EvolutionServiceError",
    "InvalidEvolutionTransition",
    "MutationResult",
)
_STORE_EXPORTS = (
    "EvolutionAlreadyExistsError",
    "EvolutionConflictError",
    "EvolutionCorruptRecordError",
    "EvolutionLockError",
    "EvolutionStore",
    "EvolutionStoreError",
    "EvolutionTransaction",
    "EvolutionUnsupportedSchemaError",
)

_EXPORT_MODULE = {
    **{
        name: "photomatagent.scientific.evolution.models"
        for name in _MODEL_EXPORTS
    },
    **{
        name: "photomatagent.scientific.evolution.rubric"
        for name in _RUBRIC_EXPORTS
    },
    **{
        name: "photomatagent.scientific.evolution.service"
        for name in _SERVICE_EXPORTS
    },
    **{
        name: "photomatagent.scientific.evolution.store"
        for name in _STORE_EXPORTS
    },
}

__all__ = [
    *_MODEL_EXPORTS,
    *_RUBRIC_EXPORTS,
    *_SERVICE_EXPORTS,
    *_STORE_EXPORTS,
]


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORT_MODULE[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
