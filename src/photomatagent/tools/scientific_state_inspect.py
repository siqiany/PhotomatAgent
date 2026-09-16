"""scientific_state_inspect: dump the current scientific state for the model."""

from __future__ import annotations

import json
from typing import Any

from photomatagent.runtime.context import format_scientific_state
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.base import Tool, ToolResult


class ScientificStateInspectTool(Tool):
    name = "scientific_state_inspect"
    description = (
        "Inspect the current scientific state (goal, hypotheses, claims, evidence, "
        "calculations)."
    )
    namespace = "scientific"
    tags = ("scientific", "state", "evidence", "claims")
    input_schema = {
        "type": "object",
        "properties": {
            "section": {
                "type": "string",
                "enum": ["all", "hypotheses", "evidence", "claims", "calculations"],
            },
            "offset": {"type": "integer", "minimum": 0, "default": 0},
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "default": 10,
            },
        },
        "required": [],
        "additionalProperties": False,
    }

    def __init__(self, scientific_state: ScientificState) -> None:
        self._state = scientific_state

    @staticmethod
    def _invalid_arguments(message: str) -> ToolResult:
        data = {"error_type": "INVALID_INSPECT_ARGUMENTS", "message": message}
        return ToolResult(
            output=json.dumps(data, ensure_ascii=False, separators=(",", ":")),
            is_error=True,
            data=data,
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        section = arguments.get("section", "all")
        valid_sections = {"all", "hypotheses", "evidence", "claims", "calculations"}
        if not isinstance(section, str) or section not in valid_sections:
            return self._invalid_arguments("unknown section")
        offset = arguments.get("offset", 0)
        limit = arguments.get("limit", 10)
        if type(offset) is not int or offset < 0:
            return self._invalid_arguments(
                "offset must be an integer greater than or equal to 0"
            )
        if type(limit) is not int or limit < 1 or limit > 50:
            return self._invalid_arguments("limit must be an integer between 1 and 50")
        if section == "all":
            text = format_scientific_state(self._state)
        elif section == "hypotheses":
            records = self._state.material_hypotheses[offset : offset + limit]
            items = [
                {
                    "hypothesis_id": record.id,
                    "candidate_id": record.candidate_id,
                    "request_id": record.proposal.request_id,
                    "formula": record.proposal.formula,
                    "statement": record.proposal.statement,
                    "design_operation": record.proposal.design_operation,
                    "parent_hypothesis_ids": list(
                        record.proposal.parent_hypothesis_ids
                    ),
                    "basis_evidence_ids": [
                        basis.evidence_id for basis in record.proposal.basis
                    ],
                    "validation_questions": list(
                        record.proposal.validation_questions
                    ),
                    "lineage": {
                        "candidate_id": record.lineage.candidate_id,
                        "generated_by": record.lineage.generated_by,
                        "validation_status": record.lineage.validation_status,
                    },
                }
                for record in records
            ]
            data = {
                "section": section,
                "offset": offset,
                "limit": limit,
                "total": len(self._state.material_hypotheses),
                "items": items,
            }
            return ToolResult(
                output=json.dumps(data, ensure_ascii=False, separators=(",", ":")),
                data=data,
            )
        elif section == "evidence":
            text = "\n".join(
                (
                    f"- ({getattr(e, 'type', getattr(e, 'source_type', 'observation'))} "
                    f"from {e.source}) "
                    f"{getattr(e, 'content', getattr(e, 'summary', ''))}"
                )
                for e in self._state.evidence
            )
        elif section == "claims":
            text = "\n".join(
                f"- [{c.status}] {c.statement} ({c.confidence})" for c in self._state.claims
            )
        elif section == "calculations":
            text = "\n".join(
                f"- [{c.status}] {c.task_type} -> {c.output_reference}" for c in self._state.calculations
            )
        return ToolResult(output=text or "(empty)", data={"section": section})
