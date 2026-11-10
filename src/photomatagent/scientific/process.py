"""Versioned, evidence-linked process proposals."""

from __future__ import annotations

import hashlib
from itertools import combinations
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, SerializerFunctionWrapHandler, StringConstraints, model_serializer, model_validator
from photomatagent.capability_acquisition.models import MethodNeed
from photomatagent.scientific.applicability import Regime, assess_method


ProcessNote = Annotated[str, Field(min_length=1, max_length=250)]
DeliveryNote = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=250)]
ProcessId = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")]
OperationName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
NumericRange = tuple[FiniteFloat, FiniteFloat]
TypedNumber = Annotated[FiniteFloat, Field(strict=True)]
TypedRange = tuple[TypedNumber, TypedNumber]


class ParameterValue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class QuantityValue(ParameterValue):
    kind: Literal["quantity"]
    amount: TypedNumber | TypedRange


class CategoryValue(ParameterValue):
    kind: Literal["category"]
    label: DeliveryNote


class TimelinePoint(ParameterValue):
    time: Annotated[TypedNumber, Field(ge=0)]
    value: TypedNumber


class TimelineValue(ParameterValue):
    kind: Literal["timeline"]
    time_unit: OperationName
    points: tuple[TimelinePoint, ...] = Field(min_length=2, max_length=16)

    @model_validator(mode="after")
    def validate_order(self) -> TimelineValue:
        if any(left.time >= right.time for left, right in zip(self.points, self.points[1:])):
            raise ValueError("timeline times must increase")
        return self


class EndpointValue(ParameterValue):
    kind: Literal["endpoint"]
    observable: DeliveryNote
    operator: Literal["lt", "le", "eq", "ge", "gt", "observed"]
    threshold: TypedNumber | None = None

    @model_validator(mode="after")
    def validate_threshold(self) -> EndpointValue:
        if (self.threshold is None) != (self.operator == "observed"):
            raise ValueError("numeric endpoints need a threshold")
        return self


class VariableValue(ParameterValue):
    kind: Literal["variable"]
    symbol: OperationName
    resolution: DeliveryNote


TypedParameterValue = Annotated[
    QuantityValue | CategoryValue | TimelineValue | EndpointValue | VariableValue,
    Field(discriminator="kind"),
]


class ProcessParameter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=80)
    unit: str = Field(default="", max_length=40)
    bounds: NumericRange | None = None
    basis: Literal["REPORTED", "TRANSFERRED", "PROPOSED"]
    evidence_id: str = Field(default="", max_length=100)
    transfer_assumption: str = Field(default="", max_length=250)
    sample_id: str = Field(default="", max_length=64)
    source_sample: str = Field(default="", max_length=100)
    source_conditions: dict[str, FiniteFloat | str | bool] = Field(default_factory=dict, max_length=8)
    control_experiment: str = Field(default="", max_length=250)
    value: TypedParameterValue | None = None
    intent: Literal["EXPLORATION", "STARTING_POINT", "VALIDATION_WINDOW", "SETPOINT"] | None = None
    measurement_basis: DeliveryNote | None = None
    rationale: DeliveryNote | None = None

    @model_serializer(mode="wrap")
    def serialize_parameter(self, handler: SerializerFunctionWrapHandler) -> dict:
        return _without_empty(handler(self), "sample_id", "source_sample", "source_conditions", "control_experiment",
                              "value", "intent", "measurement_basis", "rationale", "bounds", "unit")

    @model_validator(mode="after")
    def validate_bounds_and_basis(self) -> ProcessParameter:
        if (self.bounds is None) == (self.value is None):
            raise ValueError("declare one parameter value")
        numeric = self.numeric_bounds
        if numeric and numeric[0] > numeric[1]:
            raise ValueError("parameter bounds must be ordered")
        if self.bounds is not None and not self.unit:
            raise ValueError("legacy bounds require a unit")
        if self.value is not None and (numeric or isinstance(self.value, TimelineValue) or (
            isinstance(self.value, EndpointValue) and self.value.threshold is not None
        )) and not _declared(self.unit):
            raise ValueError("numeric parameters need a unit or dimensionless 1")
        if isinstance(self.value, VariableValue) and (self.basis == "REPORTED" or self.intent == "SETPOINT"):
            raise ValueError("an unresolved variable cannot be a reported value or setpoint")
        if self.intent == "SETPOINT" and numeric and numeric[0] != numeric[1]:
            raise ValueError("an interval needs a window or exploration intent")
        if self.basis != "PROPOSED" and not self.evidence_id:
            raise ValueError("reported and transferred parameters need an evidence reference")
        if self.basis == "TRANSFERRED" and not self.transfer_assumption:
            raise ValueError("transferred parameters need an assumption")
        return self

    @property
    def numeric_bounds(self) -> NumericRange | None:
        if self.bounds is not None:
            return self.bounds
        if isinstance(self.value, QuantityValue):
            amount = self.value.amount
            return amount if isinstance(amount, tuple) else (amount, amount)
        return None


class ProcessStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: OperationName
    purpose: str = Field(min_length=1, max_length=250)
    equipment: str = Field(default="", max_length=160)
    parameters: tuple[ProcessParameter, ...] = Field(default=(), max_length=12)
    checkpoint: str = Field(default="", max_length=250)
    failure_response: str = Field(default="", max_length=250)
    step_id: str = Field(default="", pattern=r"^(?:[A-Za-z][A-Za-z0-9_-]{0,63})?$")
    input_samples: tuple[str, ...] = Field(default=(), max_length=8)
    output_samples: tuple[str, ...] = Field(default=(), max_length=8)
    operation_tags: tuple[OperationName, ...] = Field(default=(), max_length=8)
    required_inputs: tuple[OperationName, ...] = Field(default=(), max_length=12)

    @model_serializer(mode="wrap")
    def serialize_step(self, handler: SerializerFunctionWrapHandler) -> dict:
        return _without_empty(handler(self), "step_id", "input_samples", "output_samples", "operation_tags", "required_inputs")


_OPERATION_REQUIREMENTS = {
    "synthesis": ("precursor",),
    "deposition": ("substrate",),
    "anneal": ("thermal_budget",),
}


def _declared(value: str) -> bool:
    return value.strip().casefold() not in {"", "unknown", "unspecified", "tbd", "未知", "待确认"}


def _without_empty(payload: dict, *keys: str) -> dict:
    for key in keys:
        if not payload.get(key):
            payload.pop(key, None)
    return payload


class ProcessSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    sample_id: ProcessId
    label: ProcessNote
    role: Literal["input", "intermediate", "product"]
    architecture: OperationName | None = None
    description: DeliveryNote | None = None
    regimes: tuple[Regime, ...] = Field(default=(), max_length=9)
    architecture_status: Literal["UNKNOWN", "HYPOTHESIS", "OBSERVED"] | None = None
    architecture_evidence_id: OperationName | None = None

    @model_serializer(mode="wrap")
    def serialize_sample(self, handler: SerializerFunctionWrapHandler) -> dict:
        return _without_empty(handler(self), "architecture", "description", "regimes",
                              "architecture_status", "architecture_evidence_id")

    @model_validator(mode="after")
    def validate_architecture_status(self) -> ProcessSample:
        if self.architecture_status == "UNKNOWN" and self.architecture:
            raise ValueError("unknown architecture cannot identify a concrete structure")
        if self.architecture_status in {"HYPOTHESIS", "OBSERVED"} and not self.architecture:
            raise ValueError("an architecture hypothesis or observation needs a label")
        if (self.architecture_status == "OBSERVED") != bool(self.architecture_evidence_id):
            raise ValueError("observed architecture needs a sample-bound evidence reference")
        return self


class ProcessTransition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source: ProcessId
    target: ProcessId
    outcome: Literal["always", "pass", "fail"]
    condition: ProcessNote


class MethodNeedRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    goal: str
    need: MethodNeed
    origin: str
    basis: str
    status: Literal["needs_search", "searched", "no_match", "source_unavailable", "incompatible", "selected", "not_needed"]
    decision_id: str = ""
    candidate_snapshot: dict[str, Any] = Field(default_factory=dict)

    @model_serializer(mode="wrap")
    def serialize_need_record(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return _without_empty(handler(self), "decision_id", "candidate_snapshot")


class ProcessReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["sample", "step", "result", "assumption", "requirement", "method", "decision"]
    id: DeliveryNote

    @property
    def key(self) -> tuple[str, str]:
        return self.kind, self.id


class ProcessLink(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: ProcessReference
    target: ProcessReference
    relation: Literal["DEPENDS_ON", "CORRELATION", "MECHANISM_HYPOTHESIS", "VALIDATION_REFERENCE"]
    explanation: DeliveryNote
    evidence_ids: tuple[OperationName, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def validate_reference(self) -> ProcessLink:
        if self.source == self.target:
            raise ValueError("a relation needs different endpoints")
        if self.relation == "VALIDATION_REFERENCE" and not self.evidence_ids:
            raise ValueError("a validation relation needs an evidence reference")
        return self


class ProcessOutcome(BaseModel):
    """A condition-scoped reading of a result or design expectation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    result_id: ProcessId
    metric: OperationName
    kind: Literal["TARGET", "REQUIREMENT", "IDEAL_LIMIT", "CONDITIONAL_CALCULATION",
                  "MEASURED", "PREDICTION", "QUALITATIVE", "UNAVAILABLE"]
    value: TypedNumber | TypedRange | None = None
    relation: Literal["MIN", "MAX", "EXACT", "RANGE"] | None = None
    unit: str = Field(default="", max_length=40)
    range_basis: Literal["DESIGN_WINDOW", "SCENARIO_ENVELOPE", "MEASURED_RANGE", "STATISTICAL_INTERVAL"] | None = None
    sample_id: ProcessId | None = None
    conditions: dict[OperationName, FiniteFloat | str | bool] = Field(default_factory=dict, max_length=12)
    basis: DeliveryNote
    limitations: DeliveryNote | None = None
    evidence_id: OperationName | None = None

    @model_validator(mode="after")
    def validate_identity(self) -> ProcessOutcome:
        if self.kind in {"QUALITATIVE", "UNAVAILABLE"}:
            if any(item is not None for item in (self.value, self.relation, self.range_basis)):
                raise ValueError("non-numerical outcomes cannot assert a numerical result")
            return self
        if self.value is None or self.relation is None or not _declared(self.unit):
            raise ValueError("numerical outcomes need value, relation and unit")
        if isinstance(self.value, tuple):
            if self.relation != "RANGE" or self.range_basis is None or self.value[0] > self.value[1]:
                raise ValueError("ranges need ordered bounds and a range basis")
        elif self.relation == "RANGE" or self.range_basis is not None:
            raise ValueError("range identity needs two bounds")
        if self.range_basis == "MEASURED_RANGE" and self.kind != "MEASURED":
            raise ValueError("a measured range requires measured evidence")
        if self.kind in {"MEASURED", "CONDITIONAL_CALCULATION"} or self.range_basis == "STATISTICAL_INTERVAL":
            if not self.evidence_id or not self.sample_id:
                raise ValueError("observations and calculations need a sample and evidence reference")
        if self.kind in {"IDEAL_LIMIT", "CONDITIONAL_CALCULATION", "PREDICTION"} and not self.limitations:
            raise ValueError("scoped estimates require explicit limitations")
        return self


class DecisionAlternative(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    description: ProcessNote
    method_id: str = Field(default="", max_length=100)
    assumptions: tuple[ProcessNote, ...] = Field(default=(), max_length=8)
    evidence_ids: tuple[ProcessNote, ...] = Field(default=(), max_length=8)
    potential: DeliveryNote | None = None
    preparation_basis: DeliveryNote | None = None

    @model_serializer(mode="wrap")
    def serialize_alternative(self, handler: SerializerFunctionWrapHandler) -> dict:
        return _without_empty(handler(self), "potential", "preparation_basis")


class ProposalUnknown(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cause: DeliveryNote
    impact: DeliveryNote
    resolution: DeliveryNote
    blocks_execution: bool = False
    blocks_validation: bool = True
    basis: Literal["OBSERVED", "MECHANISM", "HYPOTHESIS"] | None = None
    evidence_ids: tuple[OperationName, ...] = Field(default=(), max_length=8)
    affects: tuple[ProcessReference, ...] = Field(default=(), max_length=12)
    decision_effect: DeliveryNote | None = None
    mitigation_cost: DeliveryNote | None = None

    @model_serializer(mode="wrap")
    def serialize_unknown(self, handler: SerializerFunctionWrapHandler) -> dict:
        return _without_empty(handler(self), "basis", "evidence_ids", "affects", "decision_effect", "mitigation_cost")

    @model_validator(mode="after")
    def validate_observation(self) -> ProposalUnknown:
        if self.basis == "OBSERVED" and not self.evidence_ids:
            raise ValueError("observed risk needs an observation reference")
        return self


class ProposalValidation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    control: DeliveryNote
    procedure: DeliveryNote
    metric: DeliveryNote
    criterion: DeliveryNote
    next_action: DeliveryNote


class ProposalBranch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    condition: DeliveryNote
    approach: DeliveryNote
    information_action: DeliveryNote


class ProcessProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["CANDIDATE", "CONDITIONAL", "VALIDATION"]
    baseline: DeliveryNote
    change: DeliveryNote
    rationale: DeliveryNote
    variables: tuple[DeliveryNote, ...] = Field(min_length=1, max_length=8)
    unknowns: tuple[ProposalUnknown, ...] = Field(default=(), max_length=8)
    branches: tuple[ProposalBranch, ...] = Field(default=(), max_length=6)
    validation: ProposalValidation

    @model_validator(mode="after")
    def validate_conditions(self) -> ProcessProposal:
        if self.kind == "CONDITIONAL" and (not self.unknowns or not self.branches):
            raise ValueError("conditional proposal requires actionable unknowns and a branch")
        return self


class ResearchStopRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reason: Literal["DECISION_SUFFICIENT", "VALIDATION_REQUIRED", "HARD_CONFLICT"]
    rationale: DeliveryNote
    constraint_change: DeliveryNote | None = None

    @model_validator(mode="after")
    def validate_conflict(self) -> ResearchStopRequest:
        if self.reason == "HARD_CONFLICT" and self.constraint_change is None:
            raise ValueError("hard conflict requires a constraint change")
        return self


class ResearchDelivery(BaseModel):
    """Runtime-owned search and delivery state, persisted with the research."""

    model_config = ConfigDict(extra="forbid")

    goal: str
    request_sha256: str
    evidence_sha256: str
    attempts: int = 0
    no_gain: int = 0
    source_keys: list[str] = Field(default_factory=list)
    source_notes: list[str] = Field(default_factory=list)
    source_status: dict[str, str] = Field(default_factory=dict)
    search_tools: list[str] = Field(default_factory=list)
    stop_reason: str = ""
    status: Literal["SEARCHING", "READY", "PARTIAL", "CONFLICT"] = "SEARCHING"
    decision_id: str = ""
    plan_ids: tuple[str, ...] = ()
    superseded: bool = False


class DecisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    question: str = Field(min_length=1, max_length=500)
    alternatives: tuple[DecisionAlternative, ...] = Field(default=(), max_length=8)
    assumptions: tuple[ProcessNote, ...] = Field(default=(), max_length=8)
    method_need: MethodNeed | None = None
    method_need_basis: str = Field(default="", max_length=500)
    observed_action_summary: str = Field(default="", max_length=500)
    evidence_ids: tuple[ProcessNote, ...] = Field(default=(), max_length=12)
    process_plan_ids: tuple[ProcessNote, ...] = Field(default=(), max_length=8)
    previous_decision_id: str | None = Field(default=None, pattern=r"^decision_[0-9a-f]{32}$")
    method_disposition: Literal["selected", "not_needed"] | None = None
    selected_method_id: str = Field(default="", max_length=100)
    method_input_actions: dict[DeliveryNote, DeliveryNote] = Field(default_factory=dict, max_length=12)
    research_stop: ResearchStopRequest | None = None

    @model_serializer(mode="wrap")
    def serialize_decision(self, handler: SerializerFunctionWrapHandler) -> dict:
        return _without_empty(handler(self), "method_disposition", "selected_method_id",
                              "method_input_actions", "research_stop")

    @model_validator(mode="after")
    def validate_basis(self) -> DecisionRecord:
        if self.method_need and not self.method_need_basis.strip():
            raise ValueError("method need requires a stated basis")
        if self.observed_action_summary and not self.evidence_ids:
            raise ValueError("reported action requires evidence references")
        if self.method_disposition and self.method_need is None:
            raise ValueError("method disposition requires a method need")
        if bool(self.selected_method_id) != (self.method_disposition == "selected"):
            raise ValueError("selected disposition requires a method identity")
        if self.method_input_actions and self.method_disposition != "selected":
            raise ValueError("method input actions require a selected method")
        return self

    @property
    def id(self) -> str:
        return "decision_" + hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:32]


class ProcessPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    decision_question: str = Field(min_length=1, max_length=500)
    sample: str = Field(min_length=1, max_length=200)
    assumptions: tuple[ProcessNote, ...] = Field(default=(), max_length=12)
    alternatives: tuple[ProcessNote, ...] = Field(default=(), max_length=8)
    missing_inputs: tuple[ProcessNote, ...] = Field(default=(), max_length=12)
    steps: tuple[ProcessStep, ...] = Field(min_length=1, max_length=20)
    samples: tuple[ProcessSample, ...] = Field(default=(), max_length=30)
    transitions: tuple[ProcessTransition, ...] = Field(default=(), max_length=40)
    requirements: dict[OperationName, ProcessNote] = Field(default_factory=dict, max_length=24)
    previous_plan_id: str | None = Field(default=None, pattern=r"^proc_[0-9a-f]{32}$")
    revision_basis: str = Field(default="", max_length=500)
    proposal: ProcessProposal | None = None
    outcomes: tuple[ProcessOutcome, ...] = Field(default=(), max_length=16)
    links: tuple[ProcessLink, ...] = Field(default=(), max_length=40)

    @model_serializer(mode="wrap")
    def serialize_process(self, handler: SerializerFunctionWrapHandler) -> dict:
        return _without_empty(handler(self), "samples", "transitions", "requirements", "previous_plan_id",
                              "revision_basis", "proposal", "outcomes", "links")

    @model_validator(mode="after")
    def validate_graph(self) -> ProcessPlan:
        if bool(self.previous_plan_id) != bool(self.revision_basis.strip()):
            raise ValueError("process revision requires a predecessor and basis")
        samples = {sample.sample_id: sample for sample in self.samples}
        if len(samples) != len(self.samples):
            raise ValueError("process graph requires unique sample identities")
        outcomes = {item.result_id for item in self.outcomes}
        if len(outcomes) != len(self.outcomes) or any(
            item.sample_id is not None and item.sample_id not in samples for item in self.outcomes
        ):
            raise ValueError("process outcomes need unique IDs and available sample references")
        declared = {
            "step": {step.step_id for step in self.steps if step.step_id}, "sample": set(samples),
            "result": outcomes, "assumption": set(self.assumptions), "requirement": set(self.requirements),
        }
        references = [item for link in self.links for item in (link.source, link.target)]
        if self.proposal:
            references.extend(item for unknown in self.proposal.unknowns for item in unknown.affects)
        if any(item.kind in declared and item.id not in declared[item.kind] for item in references):
            raise ValueError("process links and risks must reference existing records")
        if not (self.samples or self.transitions or any(step.step_id or step.input_samples or step.output_samples for step in self.steps)):
            return self
        steps = {step.step_id: index for index, step in enumerate(self.steps)}
        if "" in steps or len(steps) != len(self.steps):
            raise ValueError("process graph requires unique step identities")
        incoming: dict[str, list[str]] = {}
        edge_outcomes: dict[str, set[str]] = {}
        for edge in self.transitions:
            if edge.source not in steps or edge.target not in steps or steps[edge.source] >= steps[edge.target]:
                raise ValueError("process edges must reference later steps")
            existing = edge_outcomes.setdefault(edge.source, set())
            if (edge.outcome in existing or "always" in existing or (edge.outcome == "always" and existing)
                    or (edge.outcome != "always" and not self.steps[steps[edge.source]].checkpoint.strip())):
                raise ValueError("process outcomes must be distinct and conditional edges need a checkpoint")
            existing.add(edge.outcome)
            incoming.setdefault(edge.target, []).append(edge.source)
        if not self.transitions:
            incoming = {step.step_id: [self.steps[index - 1].step_id] for index, step in enumerate(self.steps) if index}
        initial = {key for key, sample in samples.items() if sample.role == "input"}
        available: dict[str, set[str]] = {}
        produced: set[str] = set()
        for index, step in enumerate(self.steps):
            parents = incoming.get(step.step_id, [])
            if index and not parents:
                raise ValueError("process step is unreachable")
            inputs = set(step.input_samples)
            outputs = set(step.output_samples)
            current = set.intersection(*(available[parent] for parent in parents)) if parents else initial
            if (not inputs <= current or not outputs <= samples.keys() or outputs & (initial | produced)
                    or len(inputs) != len(step.input_samples) or len(outputs) != len(step.output_samples)
                    or any(p.sample_id and p.sample_id not in inputs | outputs for p in step.parameters)):
                raise ValueError("process sample is unavailable on this branch or has multiple producers")
            produced.update(outputs)
            available[step.step_id] = current | outputs
        return self

    @property
    def id(self) -> str:
        return "proc_" + hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:32]

    def revision_impact(self, revised: ProcessPlan) -> dict[str, Any]:
        """Trace only declared dependencies of results from this version."""
        before, after = self._revision_records(), revised._revision_records()
        changed = {key for key in before if before[key] != after.get(key)}
        old_steps = {step.step_id: step for step in self.steps if step.step_id}
        for link in set(self.links).symmetric_difference(revised.links):
            if link.relation == "DEPENDS_ON":
                changed.add(link.source.key)
        changed.update(("step", edge.target) for edge in self.transitions if edge not in revised.transitions)
        changed.update(("step", edge.target) for edge in revised.transitions if edge not in self.transitions)
        changed.update(("assumption", item) for item in self.assumptions if item not in revised.assumptions)
        changed.update(("requirement", key) for key, value in self.requirements.items()
                       if revised.requirements.get(key) != value)
        dependencies = [(link.source.key, link.target.key) for link in self.links if link.relation == "DEPENDS_ON"]
        dependencies.extend((("result", row.result_id), ("sample", row.sample_id))
                            for row in self.outcomes if row.sample_id)
        for key, step in old_steps.items():
            dependencies.extend((("step", key), ("sample", sample)) for sample in step.input_samples)
            dependencies.extend((("sample", sample), ("step", key)) for sample in step.output_samples)
        grew = True
        while grew:
            additions = {source for source, target in dependencies if target in changed} - changed
            grew = bool(additions)
            changed.update(additions)
        tracked = {source for source, _ in dependencies}
        untracked_change = self.steps != revised.steps and any(not step.step_id for step in (*self.steps, *revised.steps))
        needs_review = [row.result_id for row in self.outcomes if ("result", row.result_id) in changed]
        unknown = [row.result_id for row in self.outcomes if row.result_id not in needs_review and (
            untracked_change or ((changed or self.requirements != revised.requirements)
                                 and ("result", row.result_id) not in tracked)
        )]
        return {"previous_plan_id": self.id, "results_to_review": needs_review, "unknown_impact": unknown}

    def _revision_records(self) -> dict[tuple[str, str], BaseModel]:
        return {
            **{("step", row.step_id): row for row in self.steps if row.step_id},
            **{("sample", row.sample_id): row for row in self.samples},
            **{("result", row.result_id): row for row in self.outcomes},
        }

    def method_domain_gaps(self) -> list[dict[str, Any]]:
        samples = {row.sample_id: row for row in self.samples}
        result_samples = {row.result_id: row.sample_id for row in self.outcomes}
        warnings = []
        for link in self.links:
            if link.target.kind != "method":
                continue
            sample_id = link.source.id if link.source.kind == "sample" else result_samples.get(link.source.id)
            sample = samples.get(sample_id or "")
            if sample is None:
                continue
            architecture = (sample.architecture or "").casefold().replace("-", "_").replace(" ", "_")
            regimes = sample.regimes
            if architecture in {"core_shell", "radial_gradient", "graded_core_shell"} and "core_shell_dot" not in regimes:
                regimes += ("core_shell_dot",)
            domain = assess_method(link.target.id, regimes)
            if domain.status == "UNSUPPORTED":
                warnings.append({"reference": link.source.model_dump(), "sample_id": sample.sample_id,
                                 **domain.model_dump(mode="json")})
        return warnings

    def outcome_conflicts(self) -> list[list[str]]:
        targets = [row for row in self.outcomes if row.kind in {"TARGET", "REQUIREMENT"}]

        def interval(row: ProcessOutcome) -> tuple[float, float]:
            if isinstance(row.value, tuple):
                return row.value
            assert row.value is not None
            lower = -float("inf") if row.relation == "MAX" else row.value
            upper = float("inf") if row.relation == "MIN" else row.value
            return lower, upper

        conflicts = []
        for left, right in combinations(targets, 2):
            if (left.metric, left.unit, left.sample_id, left.conditions) != (
                right.metric, right.unit, right.sample_id, right.conditions
            ):
                continue
            a, b = interval(left), interval(right)
            if max(a[0], b[0]) > min(a[1], b[1]):
                conflicts.append([left.result_id, right.result_id])
        return conflicts

    def proposal_readiness(self) -> dict[str, Any]:
        """Derive gaps for a proposal without changing its serialized identity."""
        missing = list(self.missing_inputs)
        for step in self.steps:
            required = list(step.required_inputs)
            legacy = _OPERATION_REQUIREMENTS.get(step.operation, ())
            if legacy and (step.operation == "anneal" or not step.required_inputs):
                required = ["equipment", *(("temperature",) if step.operation == "anneal" else ()), *legacy, *required]
            declared = {key.casefold() for key, value in self.requirements.items() if _declared(value)}
            declared.update(parameter.name.strip().casefold() for parameter in step.parameters
                            if not isinstance(parameter.value, VariableValue))
            if _declared(step.equipment):
                declared.add("equipment")
            missing.extend(key for key in required if key.casefold() not in declared)
        method_gaps = self.method_domain_gaps()
        conflicts = self.outcome_conflicts()
        explanations = [
            {"step": step.step_id or str(index), "parameter": parameter.name, "reason": "PROPOSED_BASIS_UNEXPLAINED"}
            for index, step in enumerate(self.steps, 1) for parameter in step.parameters
            if parameter.value is not None and parameter.basis == "PROPOSED" and not (parameter.rationale or self.proposal)
        ]
        return {
            "completeness": "INCOMPLETE" if missing else "REQUIREMENTS_DECLARED",
            "missing_inputs": list(dict.fromkeys(missing)),
            "conditional": bool(missing or method_gaps or conflicts or self.transitions
                                or (self.proposal and self.proposal.unknowns)),
            "proposal_delivery": "READY" if self.proposal else "DRAFT",
            "execution_readiness": "BLOCKED" if missing or conflicts or (
                self.proposal and any(item.blocks_execution for item in self.proposal.unknowns)
            ) else "UNCHECKED",
            "method_domain_gaps": method_gaps,
            "outcome_conflicts": conflicts,
            "explanation_gaps": explanations,
        }
