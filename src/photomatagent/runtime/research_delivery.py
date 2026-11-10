"""Bound discovery and render a useful handoff from registered research state."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from photomatagent.redaction import redact_text
from photomatagent.scientific.process import (
    CategoryValue, DecisionRecord, EndpointValue, ProcessParameter, ResearchDelivery, TimelineValue, VariableValue,
)
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.base import ToolResult


SEARCH_TOOLS = frozenset({
    "method_search", "method_search_external", "web.search", "web.search_repositories",
    "literature.search", "literature.search_arxiv", "literature.search_local",
    "literature.search_passages", "literature.resolve_fulltext", "literature.fetch_source",
})
_RESULT_FIELDS = ("hits", "results", "papers", "passages", "candidates",
                  "implementations", "literature_evidence_leads", "locations")
_SOURCE_FIELDS = ("doi", "arxiv_id", "document_id", "passage_id", "repository",
                  "normalized_name", "method_id", "canonical_id", "url", "id", "source", "path")
_SEARCH_LIMIT = 8
_REPEAT_LIMIT = 2


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _evidence_key(state: ScientificState) -> str:
    return digest("\n".join(sorted({
        item.id for item in state.evidence if getattr(item, "property", "") != "literature_reference"
    })))


def refresh_delivery(state: ScientificState, request_key: str, *, reopen_request: bool = False) -> None:
    delivery = state.research_delivery
    if delivery is None:
        return
    evidence_key = _evidence_key(state)
    if ((reopen_request and delivery.request_sha256 != request_key)
            or (delivery.stop_reason and delivery.evidence_sha256 != evidence_key)):
        delivery.superseded = True
    elif delivery.evidence_sha256 != evidence_key:
        delivery.evidence_sha256 = evidence_key
        delivery.no_gain = 0


def current_delivery(state: ScientificState, request_key: str) -> ResearchDelivery:
    delivery = state.research_delivery
    if delivery is None:
        delivery = ResearchDelivery(goal=state.goal, request_sha256=request_key, evidence_sha256=_evidence_key(state))
        state.research_deliveries.append(delivery)
    return delivery


def _source_identity(row: dict[str, Any]) -> tuple[str, str] | None:
    source = row.get("record") or row.get("method") or row.get("citation") or row
    if not isinstance(source, dict):
        return None
    for field in _SOURCE_FIELDS:
        value = source.get(field)
        if isinstance(value, (str, int)) and str(value).strip():
            identity = str(value).strip()
            if field in {"doi", "arxiv_id"}:
                identity = identity.casefold().removeprefix("https://doi.org/")
            version = source.get("document_revision") or source.get("version") or source.get("revision") or ""
            title = source.get("title") or source.get("name")
            label = f"{title} · {identity}" if isinstance(title, str) and title.strip() else identity
            if field in {"id", "canonical_id"}:
                identity = f"{row.get('provider', source.get('source', ''))}:{identity}"
            return digest(f"{field}:{identity}:{version}"), " ".join(redact_text(label).split())[:240]
    return None


def record_search(delivery: ResearchDelivery, name: str, result: ToolResult) -> None:
    delivery.attempts += 1
    if name not in delivery.search_tools:
        delivery.search_tools.append(name)
    statuses = result.data.get("source_status")
    if isinstance(statuses, dict):
        for source, value in list(statuses.items())[:16]:
            if isinstance(source, str) and isinstance(value, str):
                delivery.source_status[redact_text(source)[:64]] = redact_text(value)[:100]
    diagnostic = result.data.get("diagnostic") or result.data.get("error") or result.data.get("error_type")
    if result.is_error and not diagnostic:
        diagnostic = result.output if re.fullmatch(r"[A-Z_]{3,96}", result.output) else "SOURCE_ERROR"
    if isinstance(diagnostic, str):
        delivery.source_status[name] = redact_text(diagnostic)[:100]

    data = result.data
    rows = [data]
    for field in _RESULT_FIELDS:
        values = data.get(field)
        if isinstance(values, (list, tuple)):
            rows.extend(value for value in values[:32] if isinstance(value, dict))
    rows += [record for row in rows if not row.get("doi") and isinstance(row.get("records"), list)
             for record in row["records"][:8] if isinstance(record, dict)]
    found = dict(identity for row in rows if (identity := _source_identity(row)) is not None) if not result.is_error else {}
    new = [key for key in found if key not in delivery.source_keys]
    delivery.source_keys.extend(new)
    delivery.source_notes.extend(found[key] for key in new[:max(0, 8 - len(delivery.source_notes))])
    delivery.no_gain = 0 if new else delivery.no_gain + 1
    if delivery.attempts >= _SEARCH_LIMIT:
        delivery.stop_reason = "SEARCH_LIMIT"
    elif delivery.no_gain >= _REPEAT_LIMIT:
        degraded = any(value not in {"OK", "EMPTY"} for value in delivery.source_status.values())
        if degraded and not found:
            if delivery.attempts >= _REPEAT_LIMIT + 1:
                delivery.stop_reason = "SOURCE_LIMITATIONS"
        else:
            delivery.stop_reason = "NO_NEW_SOURCES"


def adopt_stop(delivery: ResearchDelivery, decision: DecisionRecord) -> None:
    if decision.research_stop is None:
        return
    delivery.stop_reason = decision.research_stop.reason
    delivery.decision_id = decision.id
    delivery.plan_ids = decision.process_plan_ids


def _describe_parameter(item: ProcessParameter) -> str:
    value = item.value
    if numeric := item.numeric_bounds:
        low, high = numeric
        display = f"{low:g}" if low == high else f"{low:g}–{high:g}"
    elif isinstance(value, CategoryValue):
        display = value.label
    elif isinstance(value, TimelineValue):
        display = ", ".join(f"{point.time:g} {value.time_unit} → {point.value:g}" for point in value.points)
    elif isinstance(value, EndpointValue):
        display = f"{value.observable} {value.operator}" + (f" {value.threshold:g}" if value.threshold is not None else "")
    elif isinstance(value, VariableValue):
        display = f"{value.symbol} · {value.resolution}"
    else:
        display = "—"
    qualifiers = " · ".join(filter(None, (item.basis, item.intent, item.measurement_basis, item.rationale)))
    return f"{item.name} {display} {item.unit}".rstrip() + f" [{qualifiers}]"


def render_delivery(state: ScientificState, delivery: ResearchDelivery, *, partial: bool = False) -> str:
    decision = next((item for item in state.decisions if item.id == delivery.decision_id), None)
    request = decision.research_stop if decision else None
    plans = [plan for plan in state.process_plans if plan.id in delivery.plan_ids]
    delivery.status = "PARTIAL" if partial else "CONFLICT" if request and request.reason == "HARD_CONFLICT" else (
        "READY" if any(plan.proposal for plan in plans) else "PARTIAL"
    )
    zh = bool(re.search(r"[\u3400-\u9fff]", delivery.goal))
    labels = (
        ("交付状态", "停止原因", "检索范围", "来源诊断", "科学验证", "基线", "改动", "假设", "设计变量", "待解决",
         "验证", "下一步", "操作就绪", "最小约束变更")
        if zh else
        ("Delivery", "Search stop", "Search scope", "Source diagnostics", "Scientific validation", "Baseline", "Change",
         "Hypothesis", "Design variables", "Unresolved", "Validation", "Next step", "Execution readiness", "Required constraint change")
    )
    (status, stop, scope, diagnostics, science, baseline, change, hypothesis,
     variables, unresolved, validation, next_step, execution, constraint) = labels
    lines = [f"**{status} — {delivery.status}**", f"{stop} — {delivery.stop_reason}",
             f"{scope} — {delivery.attempts} · {', '.join(delivery.search_tools) or '—'} · {len(delivery.source_keys)}",
             f"{science} — UNKNOWN"]
    if delivery.source_status:
        lines.append(f"{diagnostics} — " + ", ".join(f"{key}={value}" for key, value in delivery.source_status.items()))
    if delivery.source_notes:
        lines.append(("已有来源线索" if zh else "Source leads found") + "\n"
                     + "\n".join(f"- {item}" for item in delivery.source_notes))
    if request:
        lines.append(request.rationale)
        if request.constraint_change:
            lines.append(f"{constraint} — {request.constraint_change}")
    if decision:
        for alternative in decision.alternatives:
            dimensions = " · ".join(filter(None, (alternative.potential, alternative.preparation_basis)))
            if dimensions:
                lines.append(f"- {alternative.description} · {dimensions}")
    for plan in plans:
        proposal = plan.proposal
        readiness = plan.proposal_readiness()
        if proposal is None:
            lines.extend((
                f"\n**PROPOSED · {plan.id}**", plan.decision_question,
                *[f"- {step.operation} · {step.purpose}" for step in plan.steps],
                f"{execution} — {readiness['execution_readiness']} · {', '.join(readiness['missing_inputs']) or '—'}",
            ))
            continue
        lines.extend((
            f"\n**{proposal.kind} · {plan.id}**", f"{baseline} — {proposal.baseline}", f"{change} — {proposal.change}",
            f"{hypothesis} — {proposal.rationale}", f"{variables} — {', '.join(proposal.variables)}",
            f"{execution} — {readiness['execution_readiness']} · {', '.join(readiness['missing_inputs']) or '—'}",
        ))
        for step in plan.steps[:8]:
            parameters = ", ".join(_describe_parameter(item) for item in step.parameters[:6])
            lines.append(f"- {step.operation} · {step.purpose}" + (f" · {parameters}" if parameters else ""))
        for outcome in plan.outcomes:
            conditions = ", ".join(f"{key}={value}" for key, value in outcome.conditions.items())
            value = "–".join(f"{bound:g}" for bound in outcome.value) if isinstance(outcome.value, tuple) else (
                f"{outcome.value:g}" if outcome.value is not None else ""
            )
            fields = (outcome.metric, outcome.kind, outcome.relation, value, outcome.unit,
                      outcome.range_basis, conditions, outcome.basis, outcome.limitations)
            lines.append("- " + " · ".join(filter(None, fields)))
        for gap in readiness["method_domain_gaps"]:
            lines.append(f"- {gap['method_id']} · UNSUPPORTED · {', '.join(gap['reasons'])}")
        for left, right in readiness["outcome_conflicts"]:
            lines.append(f"- INCOMPATIBLE_DECLARED_BOUNDS · {left} / {right}")
        if plan.previous_plan_id:
            previous = next((item for item in state.process_plans if item.id == plan.previous_plan_id), None)
            if previous and previous.outcomes:
                impact = previous.revision_impact(plan)
                if impact["results_to_review"] or impact["unknown_impact"]:
                    lines.append(f"- REVALIDATION · {', '.join(impact['results_to_review']) or '—'} · "
                                 f"UNKNOWN IMPACT · {', '.join(impact['unknown_impact']) or '—'}")
        for item in proposal.unknowns:
            blocked = " / ".join(label for label, applies in (
                (execution, item.blocks_execution), (science, item.blocks_validation),
            ) if applies)
            lines.append(f"{unresolved} — {item.cause} · {item.impact} · {item.resolution} · {blocked or '—'}")
            if details := " · ".join(filter(None, (item.basis, item.decision_effect, item.mitigation_cost))):
                lines.append(f"- {details}")
        for branch in proposal.branches:
            lines.append(f"- {branch.condition} → {branch.approach} · {branch.information_action}")
        check = proposal.validation
        lines.extend((f"{validation} — {check.control} · {check.procedure} · {check.metric} · {check.criterion}",
                      f"{next_step} — {check.next_action}"))
    if delivery.status == "PARTIAL":
        gaps = state.evidence_gaps[-8:]
        lines.append(f"{unresolved} — " + ("; ".join(gaps) if gaps else (
            "现有记录不足以形成结构化方案。" if zh else "The available record does not yet support a structured proposal."
        )))
    return "\n\n".join(lines)
