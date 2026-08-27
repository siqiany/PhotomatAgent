"""Typed experiment configuration, evaluation, and persisted results."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from photomatagent.observability.analyzer import SessionSummary
from photomatagent.scientific.loop.target import TargetSpec


class ScientificLoopVariant(BaseModel):
    """Drives each task through the Evidence-Guided Scientific Feedback Loop."""

    model_config = ConfigDict(extra="forbid")

    target: TargetSpec
    max_rounds: int = Field(default=5, ge=1)
    max_candidates: int = Field(default=10, ge=1)
    patience: int = Field(default=2, ge=1)
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)


class Expectations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_contains: list[str] = Field(default_factory=list)
    answer_not_contains: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    tools_not_used: list[str] = Field(default_factory=list)
    max_tool_calls: int | None = Field(default=None, ge=0)
    max_iterations: int | None = Field(default=None, ge=0)

    def is_empty(self) -> bool:
        return not any(
            (
                self.answer_contains,
                self.answer_not_contains,
                self.tools_used,
                self.tools_not_used,
                self.max_tool_calls is not None,
                self.max_iterations is not None,
            )
        )


class ExperimentTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    expect: Expectations | None = None


class ExperimentVariant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["fake", "openai", "anthropic"] | None = None
    model: str | None = None
    max_iterations: int = Field(default=10, ge=1)
    approval: Literal["auto", "deny"] = "auto"
    label: str | None = None
    tool_surface: Literal["progressive", "eager"] = "progressive"


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    tasks: list[ExperimentTask] = Field(min_length=1)
    variant: ExperimentVariant = Field(default_factory=ExperimentVariant)
    loop: ScientificLoopVariant | None = None


class ConfigurationSnapshot(BaseModel):
    provider: str
    model: str
    system_prompt: dict[str, object]
    stop_policy: dict[str, object]
    context_builder: dict[str, object]
    context_engine: dict[str, object] = Field(default_factory=dict)
    tool_surface: dict[str, object] = Field(default_factory=dict)
    task_set_sha256: str = ""
    skill_index_sha256: str = ""


class ExpectationCheck(BaseModel):
    name: str
    passed: bool
    detail: str


EvaluationStatus = Literal["PASS", "FAIL", "UNEVALUATED"]
RuntimeStatus = Literal["COMPLETED", "FAILED"]


class TaskEvaluation(BaseModel):
    status: EvaluationStatus
    checks: list[ExpectationCheck] = Field(default_factory=list)


class ExperimentTaskRun(BaseModel):
    task_id: str
    session_id: str
    runtime_status: RuntimeStatus
    evaluation: TaskEvaluation
    answer: str = ""
    error: str | None = None
    summary: SessionSummary


class ExperimentSummary(BaseModel):
    experiment_id: str
    name: str
    configuration: ConfigurationSnapshot
    tasks_total: int
    tasks_completed: int
    expectations_passed: int
    expectations_failed: int
    tasks_unevaluated: int
    expectation_pass_rate: float | None = None
    average_iterations: float
    average_model_calls: float
    average_tool_calls: float
    average_tool_failures: float
    tool_failure_rate: float
    repeated_tool_calls: int
    average_repeated_tool_calls: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    duration_seconds: float
    model_latency_seconds: float = 0.0
    estimated_tool_schema_tokens_per_call: float | None = None
    tool_search_calls: int = 0
    tool_describe_calls: int = 0
    tool_call_bridge_calls: int = 0
    peak_working_context_tokens: int | None = None
    pruned_tool_results: int = 0
    compaction_count: int = 0
    compaction_failures: int = 0


class ExperimentResult(BaseModel):
    experiment_id: str
    config: ExperimentConfig
    summary: ExperimentSummary
    runs: list[ExperimentTaskRun]
