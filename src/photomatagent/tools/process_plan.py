"""Register a process proposal through the runtime state-update boundary."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import BaseModel, ValidationError

from photomatagent.capability_acquisition.method_discovery import (
    MethodDiscoveryService, evaluate_method_candidate, method_input_identity,
)
from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.capabilities.literature.documents import DocumentService
from photomatagent.errors import ToolExecutionError
from photomatagent.scientific.process import (
    CategoryValue, DecisionRecord, MethodNeedRecord, ProcessParameter, ProcessPlan, VariableValue,
)
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.base import DeliveryReview, DeliveryReviewSource, Tool, ToolResult
from photomatagent.tools.exposure import ToolExposure
from photomatagent.workspace import Workspace


class RegisterDecisionTool(Tool):
    name = "decision.record"
    description = "Record a research decision question, alternatives, assumptions and references. A decision record does not establish scientific validity."
    short_description = "Record a research decision and its unresolved method needs."
    namespace = "research"
    source = "builtin"
    exposure = ToolExposure.DEFERRED
    tags = ("decision", "methods", "alternatives", "assumptions")
    input_schema = DecisionRecord.model_json_schema()

    def __init__(self, state: ScientificState, methods: MethodDiscoveryService | None = None) -> None:
        self._state = state
        self._methods = methods

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            decision = DecisionRecord.model_validate(arguments)
        except ValidationError:
            return ToolResult(output="DECISION_CONTRACT_INVALID", is_error=True)
        updates: list[BaseModel] = [decision]
        snapshot: dict[str, Any] = {}
        data: dict[str, Any] = {"decision_id": decision.id, "scientific_validation": "UNKNOWN"}
        if decision.method_need:
            if decision.method_disposition == "selected":
                method_id = decision.selected_method_id
                selected = await self._methods.inspect(method_id) if self._methods is not None else None
                if selected is None:
                    return ToolResult(output="METHOD_SELECTION_SNAPSHOT_UNAVAILABLE", is_error=True)
                snapshot, rejection = evaluate_method_candidate(decision.method_need, selected)
                if rejection:
                    error = {"diagnostic": "METHOD_SELECTION_INCOMPATIBLE", **rejection}
                    return ToolResult(output=json.dumps(error, ensure_ascii=False), data=error, is_error=True)
                actions = {method_input_identity(key) for key, action in decision.method_input_actions.items()
                           if action.strip()}
                missing_actions = [value for value in snapshot["unavailable_inputs"]
                                   if method_input_identity(value) not in actions]
                if missing_actions:
                    error = {"diagnostic": "METHOD_SELECTION_INPUT_ACTION_REQUIRED",
                             "reason": snapshot["execution_blocker"], "inputs": missing_actions}
                    return ToolResult(output=json.dumps(error, ensure_ascii=False), data=error, is_error=True)
                snapshot["input_actions"] = dict(decision.method_input_actions)
                data.update(conditional=snapshot["conditional"], unconfirmed_inputs=snapshot["unconfirmed_inputs"],
                            unconfirmed_constraints=snapshot["comparison"]["unconfirmed_constraints"],
                            execution_blocker=snapshot["execution_blocker"], input_actions=snapshot["input_actions"])
            updates.append(MethodNeedRecord(
                goal=self._state.goal, need=decision.method_need, origin=self.name,
                basis=decision.method_need_basis, status=decision.method_disposition or "needs_search",
                decision_id=decision.id, candidate_snapshot=snapshot,
            ))
        return ToolResult(output=json.dumps(data), data=data, state_updates=updates)


def _numeric_bounds(value: Any) -> tuple[float, float] | None:
    if type(value) in (int, float):
        return value, value
    if (isinstance(value, (tuple, list)) and len(value) == 2
            and all(type(item) in (int, float) for item in value)):
        return value[0], value[1]
    return None


def _parameter_binding(parameter: ProcessParameter, value: Any) -> tuple[bool, bool]:
    """Return comparable and equal, keeping type distinctions intact."""
    if parameter.numeric_bounds is not None:
        numeric = _numeric_bounds(value)
        return numeric is not None, numeric == parameter.numeric_bounds
    if isinstance(parameter.value, CategoryValue):
        return isinstance(value, str), value == parameter.value.label
    if isinstance(parameter.value, VariableValue):
        return value is not None, False
    structured = parameter.value
    return isinstance(value, dict) and structured is not None and value.get("kind") == structured.kind, (
        structured is not None and value == structured.model_dump(mode="json")
    )


class RegisterProcessPlanTool(Tool, DeliveryReviewSource):
    name = "process.register_plan"
    proposal_registration = True
    description = (
        "Register a candidate or validation plan. Use the operation names and parameters needed for this route, "
        "including typed proposed values, scoped outcomes and dependencies when useful. "
        "Execution readiness and scientific validation are reported separately."
    )
    short_description = "Register an evidence-linked process proposal."
    namespace = "research"
    source = "builtin"
    exposure = ToolExposure.DEFERRED
    tags = ("process", "materials", "recipe", "proposal", "design")
    input_schema = ProcessPlan.model_json_schema()

    def review_request(self, goal: str) -> DeliveryReview | None:
        from photomatagent.scientific.process_delivery import architecture_review

        return architecture_review(goal)

    def __init__(self, state: ScientificState, workspace: Workspace | None = None,
                 methods: MethodDiscoveryService | None = None) -> None:
        self._state = state
        self._documents = DocumentService(workspace) if workspace is not None else None
        self._methods = methods or MethodDiscoveryService(workspace)

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            plan = ProcessPlan.model_validate(arguments)
        except ValidationError as exc:
            invalid = {"diagnostic": "PROCESS_CONTRACT_INVALID", "issues": [
                {"field": ".".join(str(part) for part in item["loc"]), "message": item["msg"]}
                for item in exc.errors(include_input=False, include_context=False, include_url=False)[:8]
            ]}
            return ToolResult(output=json.dumps(invalid, ensure_ascii=False), data=invalid, is_error=True)
        previous = next((item for item in self._state.process_plans if item.id == plan.previous_plan_id), None)
        if plan.previous_plan_id and previous is None:
            return ToolResult(output="PROCESS_PREDECESSOR_UNAVAILABLE", is_error=True)
        samples = {sample.sample_id: sample.label for sample in plan.samples}
        required = {parameter.evidence_id for step in plan.steps for parameter in step.parameters
                    if parameter.basis != "PROPOSED"}
        observed = {item.evidence_id for item in plan.outcomes if item.evidence_id}
        links = [item for link in plan.links for item in (link.source, link.target)]
        cited = {item for link in plan.links for item in link.evidence_ids}
        verified = {item for link in plan.links if link.relation == "VALIDATION_REFERENCE" for item in link.evidence_ids}
        verified.update(sample.architecture_evidence_id for sample in plan.samples if sample.architecture_evidence_id)
        if plan.proposal:
            links.extend(item for risk in plan.proposal.unknowns for item in risk.affects)
            cited.update(item for risk in plan.proposal.unknowns for item in risk.evidence_ids)
            verified.update(item for risk in plan.proposal.unknowns if risk.basis == "OBSERVED" for item in risk.evidence_ids)
        if not (cited | observed) <= {item.id for item in self._state.evidence}:
            return ToolResult(output="PROCESS_SUPPORT_REFERENCE_UNAVAILABLE", is_error=True)
        if {item.id for item in links if item.kind == "decision"} - {item.id for item in self._state.decisions}:
            return ToolResult(output="PROCESS_DECISION_REFERENCE_UNAVAILABLE", is_error=True)
        for identifier in sorted({item.id for item in links if item.kind == "method"}):
            if await self._methods.inspect(identifier) is None:
                return ToolResult(output="PROCESS_METHOD_REFERENCE_UNAVAILABLE", is_error=True)
        evidence: dict[str, ScientificEvidence] = {}
        for item in self._state.evidence:
            if item.id not in required | observed | verified or not isinstance(item, ScientificEvidence) or item.assessment_role != "observation":
                continue
            attestation = self._state.verified_attestation(item.id)
            if (attestation is None or attestation.authority != "observation"
                    or item.provenance.get("source_kind") == "UNATTESTED_TEXT"):
                continue
            if item.source_type == "literature":
                if attestation.tool_name != "literature.extract_evidence" or self._documents is None:
                    continue
                try:
                    await asyncio.to_thread(self._documents.cached_anchored_object, *(item.provenance[key] for key in (
                        "source_record_id", "document_revision", "source_sha256", "object_id",
                    )))
                except (KeyError, TypeError, OSError, ValueError, ToolExecutionError):
                    continue
            evidence[item.id] = item
        if not verified <= evidence.keys():
            return ToolResult(output="PROCESS_OBSERVATION_NOT_ATTESTED", is_error=True)
        for sample in plan.samples:
            if sample.architecture_evidence_id:
                architecture_record = evidence[sample.architecture_evidence_id]
                if (architecture_record.property not in {"architecture", "composition_distribution"}
                        or architecture_record.conditions.get("sample_label", architecture_record.subject) != sample.label
                        or architecture_record.value != sample.architecture):
                    return ToolResult(output="PROCESS_ARCHITECTURE_EVIDENCE_NOT_BOUND", is_error=True)
        for outcome in plan.outcomes:
            if outcome.kind not in {"MEASURED", "CONDITIONAL_CALCULATION"} and outcome.range_basis != "STATISTICAL_INTERVAL":
                continue
            record = evidence.get(outcome.evidence_id or "")
            conditions = dict(record.conditions) if record else {}
            source_sample = conditions.pop("sample_label", record.subject if record else None)
            experimental = {"experiment", "experimental", "literature"}
            default_relation = "RANGE" if record and isinstance(record.value, (list, tuple)) else "EXACT"
            if (record is None or not record.method or record.property != outcome.metric or record.unit != outcome.unit
                    or source_sample != samples.get(outcome.sample_id or "") or conditions != outcome.conditions
                    or _numeric_bounds(record.value) != _numeric_bounds(outcome.value)
                    or record.provenance.get("value_relation", default_relation) != outcome.relation
                    or (outcome.kind == "MEASURED" and (
                        record.source_type not in experimental or (
                            record.source_type == "literature" and record.method != "reported experimental value"
                        )
                    ))
                    or (outcome.kind == "CONDITIONAL_CALCULATION" and record.source_type in experimental | {"user_parameter", "database"})
                    or (outcome.range_basis in {"MEASURED_RANGE", "STATISTICAL_INTERVAL"}
                        and record.provenance.get("range_basis") != outcome.range_basis)):
                return ToolResult(output="PROCESS_OUTCOME_EVIDENCE_NOT_BOUND", is_error=True,
                                  data={"result_id": outcome.result_id, "evidence_id": outcome.evidence_id})
        for step in plan.steps:
            for parameter in step.parameters:
                if parameter.basis == "PROPOSED":
                    if parameter.evidence_id:
                        return ToolResult(output="PROCESS_PROPOSED_PARAMETER_HAS_EVIDENCE_REFERENCE", is_error=True)
                    continue
                record = evidence.get(parameter.evidence_id)
                conditions = dict(record.conditions) if record is not None else {}
                source_sample = conditions.pop("sample_label", None)
                target_sample = samples.get(parameter.sample_id) if samples else plan.sample
                comparable, equal = _parameter_binding(parameter, record.value if record is not None else None)
                if (record is None or record.unit != parameter.unit or record.property != parameter.name
                        or record.source_type not in {"experiment", "experimental", "literature"}
                        or not record.method or record.method in {"UNKNOWN", "target or proposed value"}
                        or (record.source_type == "literature" and record.method != "reported experimental value")
                        or not source_sample or source_sample != parameter.source_sample
                        or conditions != parameter.source_conditions or not comparable or not target_sample
                        or (parameter.basis == "REPORTED" and (not equal or target_sample != source_sample))
                        or (parameter.basis == "TRANSFERRED" and not parameter.control_experiment.strip())):
                    return ToolResult(output="PROCESS_PARAMETER_EVIDENCE_NOT_BOUND", is_error=True,
                                      data={"parameter": parameter.name, "evidence_id": parameter.evidence_id})
        data: dict[str, Any] = {
            "plan_id": plan.id, "status": "PROPOSED", "scientific_validation": "UNKNOWN",
            "step_count": len(plan.steps), **plan.proposal_readiness(),
        }
        if previous is not None and previous.outcomes:
            data["revision_impact"] = previous.revision_impact(plan)
        if plan.outcomes:
            data["outcome_count"] = len(plan.outcomes)
        return ToolResult(output=json.dumps(data, ensure_ascii=False), data=data, state_updates=[plan])
