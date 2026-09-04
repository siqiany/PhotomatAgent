"""Serializable event protocol emitted by the PhotomatAgent runtime."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RuntimeEvent(BaseModel):
    """Base envelope for one typed Agent Execution Trace event.

    ``kind`` is the stable event-type discriminator. ``run_id`` distinguishes
    individual user turns inside a longer-lived interactive session.
    """

    schema_version: Literal["1.0"] = "1.0"
    kind: str
    timestamp: datetime = Field(default_factory=_now)
    session_id: str | None = None
    run_id: str | None = None


class LoopStarted(RuntimeEvent):
    kind: Literal["loop_started"] = "loop_started"
    goal: str
    provider: str = "unknown"
    model: str = "unknown"
    workspace: str = "unknown"


class LoopIterationStarted(RuntimeEvent):
    kind: Literal["loop_iteration_started"] = "loop_iteration_started"
    iteration: int


class ModelRequestStarted(RuntimeEvent):
    kind: Literal["model_request_started"] = "model_request_started"
    iteration: int
    message_count: int
    provider: str = "unknown"
    model: str = "unknown"
    # Reserved for a reliable provider-reported value; no tokenizer estimate is
    # fabricated by the runtime.
    context_size: int | None = None
    registered_tools: int = 0
    direct_count: int = 0
    deferred_count: int = 0
    hidden_count: int = 0
    visible_schema_chars: int = 0
    manifest_chars: int = 0
    estimated_schema_tokens: int = 0
    estimated_direct_schema_tokens: int = 0
    estimated_deferred_schema_tokens: int = 0
    estimated_manifest_tokens: int = 0
    estimated_avoided_tokens: int = 0
    estimated_bridge_schema_tokens: int = 0
    estimated_current_prompt_tokens: int | None = None
    estimated_message_history_tokens: int | None = None
    estimated_tool_result_tokens: int | None = None
    model_context_limit: int | None = None
    working_context_chars: int | None = None
    durable_context_chars: int | None = None
    pruned_tool_results: int = 0
    compaction_count: int = 0


class ContextPruneStarted(RuntimeEvent):
    kind: Literal["context_prune_started"] = "context_prune_started"
    tokens_before: int
    chars_before: int
    messages_before: int
    protected_turns: int


class ContextPruneCompleted(RuntimeEvent):
    kind: Literal["context_prune_completed"] = "context_prune_completed"
    tokens_before: int
    tokens_after: int
    chars_before: int
    chars_after: int
    messages_before: int
    messages_after: int
    tool_results_pruned: int
    protected_turns: int
    duration_ms: float = 0.0


class ContextCompactionStarted(RuntimeEvent):
    kind: Literal["context_compaction_started"] = "context_compaction_started"
    tokens_before: int
    chars_before: int
    messages_before: int
    protected_turns: int


class ContextCompactionCompleted(RuntimeEvent):
    kind: Literal["context_compaction_completed"] = "context_compaction_completed"
    tokens_before: int
    tokens_after: int
    chars_before: int
    chars_after: int
    messages_before: int
    messages_after: int
    protected_turns: int
    duration_ms: float = 0.0
    usage: dict[str, int | None] = Field(default_factory=dict)


class ContextCompactionFailed(RuntimeEvent):
    kind: Literal["context_compaction_failed"] = "context_compaction_failed"
    tokens_before: int
    chars_before: int
    messages_before: int
    protected_turns: int
    error: str
    duration_ms: float = 0.0


class ModelStreamStarted(RuntimeEvent):
    kind: Literal["model_stream_started"] = "model_stream_started"
    iteration: int
    provider: str
    model: str
    response_id: str | None = None


class TextDelta(RuntimeEvent):
    kind: Literal["text_delta"] = "text_delta"
    iteration: int
    text: str


class ToolCallStarted(RuntimeEvent):
    kind: Literal["tool_call_started"] = "tool_call_started"
    iteration: int
    tool_call_id: str
    tool_name: str
    index: int


class ToolCallArgumentsDelta(RuntimeEvent):
    kind: Literal["tool_call_arguments_delta"] = "tool_call_arguments_delta"
    iteration: int
    tool_call_id: str
    delta: str
    index: int


class ToolCallCompleted(RuntimeEvent):
    kind: Literal["tool_call_completed"] = "tool_call_completed"
    iteration: int
    tool_call_id: str
    tool_name: str
    arguments: dict[str, object]
    index: int


class ModelResponseCompleted(RuntimeEvent):
    kind: Literal["model_response_completed"] = "model_response_completed"
    iteration: int
    provider: str = "unknown"
    model: str = "unknown"
    response_id: str | None = None
    finish_reason: str
    tool_call_count: int
    usage: dict[str, int | None] = Field(default_factory=dict)
    duration_ms: float = 0.0


class ProviderFailed(RuntimeEvent):
    kind: Literal["provider_failed"] = "provider_failed"
    iteration: int
    provider: str
    model: str
    error: str
    error_type: str | None = None
    duration_ms: float = 0.0


class ToolRequested(RuntimeEvent):
    kind: Literal["tool_requested"] = "tool_requested"
    iteration: int
    tool_call_id: str
    tool_name: str
    arguments: dict[str, object] = Field(default_factory=dict)
    bridge_tool: str | None = None
    underlying_tool: str | None = None


class ToolApprovalRequired(RuntimeEvent):
    kind: Literal["tool_approval_required"] = "tool_approval_required"
    iteration: int
    tool_call_id: str
    tool_name: str
    arguments: dict[str, object] = Field(default_factory=dict)
    reason: str = "permission policy requires approval"
    bridge_tool: str | None = None
    underlying_tool: str | None = None


class ToolPermissionDenied(RuntimeEvent):
    kind: Literal["tool_permission_denied"] = "tool_permission_denied"
    iteration: int
    tool_call_id: str
    tool_name: str
    reason: str
    tool_status: Literal["permission_denied"] = "permission_denied"
    bridge_tool: str | None = None
    underlying_tool: str | None = None


class SensitiveAccessBlocked(RuntimeEvent):
    kind: Literal["sensitive_access_blocked"] = "sensitive_access_blocked"
    iteration: int
    tool_call_id: str
    tool_name: str
    path: str
    reason: str = "sensitive path policy"


class ToolStarted(RuntimeEvent):
    kind: Literal["tool_started"] = "tool_started"
    iteration: int
    tool_name: str
    tool_call_id: str
    bridge_tool: str | None = None
    underlying_tool: str | None = None


class ToolCompleted(RuntimeEvent):
    kind: Literal["tool_completed"] = "tool_completed"
    iteration: int
    tool_name: str
    tool_call_id: str
    output: str
    duration_ms: float = 0.0
    tool_status: Literal["success"] = "success"
    bridge_tool: str | None = None
    underlying_tool: str | None = None
    truncated: bool = False
    original_chars: int | None = None
    delivered_chars: int | None = None
    redacted: bool = False


class ToolFailed(RuntimeEvent):
    kind: Literal["tool_failed"] = "tool_failed"
    iteration: int
    tool_name: str
    tool_call_id: str
    error: str
    duration_ms: float = 0.0
    tool_status: Literal["failed"] = "failed"
    error_type: str | None = None
    bridge_tool: str | None = None
    underlying_tool: str | None = None
    truncated: bool = False
    original_chars: int | None = None
    delivered_chars: int | None = None
    redacted: bool = False


class ScientificStateUpdated(RuntimeEvent):
    kind: Literal["scientific_state_updated"] = "scientific_state_updated"
    summary: str


class BudgetUpdated(RuntimeEvent):
    kind: Literal["budget_updated"] = "budget_updated"
    model_calls: int
    tool_calls: int
    iteration: int
    input_tokens: int = 0
    output_tokens: int = 0


class LoopCompleted(RuntimeEvent):
    kind: Literal["loop_completed"] = "loop_completed"
    iterations: int
    reason: str
    duration_ms: float = 0.0


class ScientificTraceMeta(RuntimeEvent):
    """Per-run innovation-oriented summary for scientific task analysis.

    Emitted once per run at loop completion. These fields exist for paper-level
    analysis (skills loaded, scientific tools used, evidence created, gaps
    identified, capability escalations); no separate ScientificState is added.
    """

    kind: Literal["scientific_trace_meta"] = "scientific_trace_meta"
    reason: str = ""
    skills_loaded: list[str] = Field(default_factory=list)
    scientific_tools_used: list[str] = Field(default_factory=list)
    evidence_created: int = 0
    evidence_sources: list[str] = Field(default_factory=list)
    evidence_gaps_identified: list[str] = Field(default_factory=list)
    capability_escalations: list[str] = Field(default_factory=list)

    @property
    def stop_reason(self) -> str:
        """Semantic alias used by trace-analysis consumers."""
        return self.reason


class LoopFailed(RuntimeEvent):
    kind: Literal["loop_failed"] = "loop_failed"
    error: str
    duration_ms: float
    error_type: str | None = None


class ScientificLoopStarted(RuntimeEvent):
    """Outer scientific feedback loop began a new controller run."""

    kind: Literal["scientific_loop_started"] = "scientific_loop_started"
    goal: str
    max_rounds: int = 0
    min_confidence: float = 0.0


class CandidateProposed(RuntimeEvent):
    kind: Literal["candidate_proposed"] = "candidate_proposed"
    round: int
    candidate_id: str
    label: str = ""
    fingerprint: str = ""
    generation_method: str = ""


class CandidateEvaluated(RuntimeEvent):
    kind: Literal["candidate_evaluated"] = "candidate_evaluated"
    round: int
    candidate_id: str
    score: float = 0.0
    verdict: str = "INCONCLUSIVE"
    violations: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)


class CandidateJudged(RuntimeEvent):
    """Advisory LLM Scientific Judge output for one candidate.

    Never authoritative: deterministic constraint results always win; a
    judge assessment can only hold back SUCCESS or add validation work.
    """

    kind: Literal["candidate_judged"] = "candidate_judged"
    round: int
    candidate_id: str
    status: str = "UNAVAILABLE"
    quality: float = 0.0
    issues: list[str] = Field(default_factory=list)
    summary: str = ""


class ScientificFeedbackGenerated(RuntimeEvent):
    kind: Literal["scientific_feedback_generated"] = "scientific_feedback_generated"
    round: int
    candidate_id: str
    decision: str = "CONTINUE"
    summary: str = ""


class ScientificLoopDecisionMade(RuntimeEvent):
    kind: Literal["scientific_loop_decision_made"] = "scientific_loop_decision_made"
    round: int
    action: str
    reason: str = ""
    best_candidate_id: str | None = None
    best_score: float = 0.0


class ScientificLoopCompleted(RuntimeEvent):
    kind: Literal["scientific_loop_completed"] = "scientific_loop_completed"
    status: str
    rounds: int
    candidate_count: int
    best_candidate_id: str | None = None
    best_score: float = 0.0
    termination_reason: str = ""


class ScientificLoopStalled(RuntimeEvent):
    kind: Literal["scientific_loop_stalled"] = "scientific_loop_stalled"
    rounds: int
    best_score: float = 0.0
    best_candidate_id: str | None = None
    no_progress_rounds: int = 0


EvolutionEventId = Annotated[
    str,
    Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
]
EvolutionEpisodeVersion = Annotated[str, Field(pattern=r"^v\d{3}$")]
EvolutionSha256 = Annotated[str, Field(pattern=r"^[0-9a-fA-F]{64}$")]
EVOLUTION_SUMMARY_MAX_CHARS = 240
EvolutionSummary = Annotated[str, Field(max_length=EVOLUTION_SUMMARY_MAX_CHARS)]
EvolutionScore = Annotated[int, Field(strict=True, ge=1, le=5)]
EvolutionScoreDimension = Literal[
    "scientific_correctness",
    "evidence_sufficiency",
    "novelty",
    "actionability",
    "overall",
]


class EvolutionRuntimeEvent(RuntimeEvent):
    """Base envelope for events emitted by one scientific evolution task."""

    model_config = ConfigDict(extra="forbid")

    evolution_id: EvolutionEventId


class EvolutionEpisodeEvent(EvolutionRuntimeEvent):
    """Base envelope for evolution events tied to one episode version."""

    episode_version: EvolutionEpisodeVersion


class EvolutionTaskCreated(EvolutionRuntimeEvent):
    kind: Literal["evolution_task_created"] = "evolution_task_created"
    goal_summary: EvolutionSummary = ""


class EvolutionEpisodeStarted(EvolutionEpisodeEvent):
    kind: Literal["evolution_episode_started"] = "evolution_episode_started"


class EvolutionEpisodeCompleted(EvolutionEpisodeEvent):
    kind: Literal["evolution_episode_completed"] = "evolution_episode_completed"


class ExpertFeedbackRecorded(EvolutionEpisodeEvent):
    kind: Literal["expert_feedback_recorded"] = "expert_feedback_recorded"
    feedback_id: EvolutionEventId
    result_sha256: EvolutionSha256
    scores: dict[EvolutionScoreDimension, EvolutionScore] = Field(
        default_factory=dict,
        max_length=5,
    )


class ExpertFeedbackCompiled(EvolutionEpisodeEvent):
    kind: Literal["expert_feedback_compiled"] = "expert_feedback_compiled"


class RevisionPlanConfirmed(EvolutionEpisodeEvent):
    kind: Literal["revision_plan_confirmed"] = "revision_plan_confirmed"


class EvolutionIterationStarted(EvolutionEpisodeEvent):
    kind: Literal["evolution_iteration_started"] = "evolution_iteration_started"


class EvolutionComparisonCompleted(EvolutionEpisodeEvent):
    kind: Literal["evolution_comparison_completed"] = (
        "evolution_comparison_completed"
    )


class ExperienceStateChanged(EvolutionRuntimeEvent):
    kind: Literal["experience_state_changed"] = "experience_state_changed"


class EvolutionTaskAccepted(EvolutionEpisodeEvent):
    kind: Literal["evolution_task_accepted"] = "evolution_task_accepted"


class EvolutionTaskStopped(EvolutionRuntimeEvent):
    kind: Literal["evolution_task_stopped"] = "evolution_task_stopped"


AnyRuntimeEvent = Annotated[
    Union[
        LoopStarted,
        LoopIterationStarted,
        ModelRequestStarted,
        ContextPruneStarted,
        ContextPruneCompleted,
        ContextCompactionStarted,
        ContextCompactionCompleted,
        ContextCompactionFailed,
        ModelStreamStarted,
        TextDelta,
        ToolCallStarted,
        ToolCallArgumentsDelta,
        ToolCallCompleted,
        ModelResponseCompleted,
        ProviderFailed,
        ToolRequested,
        ToolApprovalRequired,
        ToolPermissionDenied,
        SensitiveAccessBlocked,
        ToolStarted,
        ToolCompleted,
        ToolFailed,
        ScientificStateUpdated,
        ScientificTraceMeta,
        BudgetUpdated,
        LoopCompleted,
        LoopFailed,
        ScientificLoopStarted,
        CandidateProposed,
        CandidateEvaluated,
        CandidateJudged,
        ScientificFeedbackGenerated,
        ScientificLoopDecisionMade,
        ScientificLoopCompleted,
        ScientificLoopStalled,
        EvolutionTaskCreated,
        EvolutionEpisodeStarted,
        EvolutionEpisodeCompleted,
        ExpertFeedbackRecorded,
        ExpertFeedbackCompiled,
        RevisionPlanConfirmed,
        EvolutionIterationStarted,
        EvolutionComparisonCompleted,
        ExperienceStateChanged,
        EvolutionTaskAccepted,
        EvolutionTaskStopped,
    ],
    Field(discriminator="kind"),
]

_EVENT_ADAPTER: TypeAdapter[AnyRuntimeEvent] = TypeAdapter(AnyRuntimeEvent)


def parse_event(payload: dict[str, object]) -> RuntimeEvent:
    return _EVENT_ADAPTER.validate_python(payload)
