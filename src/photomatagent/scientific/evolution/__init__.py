"""Persistent expert-feedback evolution contracts with lazy public exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from photomatagent.scientific.evolution.executor import (
        RuntimeFactory,
        run_fresh_evaluation,
    )
    from photomatagent.scientific.evolution.evidence import (
        EvidenceCarryDecision,
        build_inherited_scientific_state,
        select_carry_forward_evidence,
    )
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
        new_episode_owner_token,
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
    from photomatagent.scientific.evolution.revision import (
        build_revision_plan,
        format_revision_instruction,
    )
    from photomatagent.scientific.evolution.service import (
        ALLOWED_TRANSITIONS,
        ArtifactMismatchError,
        EvolutionOperationConflict,
        EvolutionService,
        EvolutionServiceError,
        FreshEvaluationClaim,
        InvalidEvolutionTransition,
        IterationContext,
        IterationClaim,
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
    from photomatagent.scientific.evolution.strategy import FixedStrategySelector

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
    "new_episode_owner_token",
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
    "FreshEvaluationClaim",
    "InvalidEvolutionTransition",
    "IterationContext",
    "IterationClaim",
    "MutationResult",
)
_EVIDENCE_EXPORTS = (
    "EvidenceCarryDecision",
    "build_inherited_scientific_state",
    "select_carry_forward_evidence",
)
_REVISION_EXPORTS = (
    "build_revision_plan",
    "format_revision_instruction",
)
_STRATEGY_EXPORTS = ("FixedStrategySelector",)
_EXECUTOR_EXPORTS = ("RuntimeFactory", "run_fresh_evaluation")
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
        name: "photomatagent.scientific.evolution.executor"
        for name in _EXECUTOR_EXPORTS
    },
    **{
        name: "photomatagent.scientific.evolution.evidence"
        for name in _EVIDENCE_EXPORTS
    },
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
        name: "photomatagent.scientific.evolution.revision"
        for name in _REVISION_EXPORTS
    },
    **{
        name: "photomatagent.scientific.evolution.strategy"
        for name in _STRATEGY_EXPORTS
    },
    **{
        name: "photomatagent.scientific.evolution.store"
        for name in _STORE_EXPORTS
    },
}

__all__ = [
    *_EVIDENCE_EXPORTS,
    *_EXECUTOR_EXPORTS,
    *_MODEL_EXPORTS,
    *_RUBRIC_EXPORTS,
    *_SERVICE_EXPORTS,
    *_REVISION_EXPORTS,
    *_STRATEGY_EXPORTS,
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
