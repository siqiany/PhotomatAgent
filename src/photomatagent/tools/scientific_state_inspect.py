"""scientific_state_inspect: dump the current scientific state for the model."""

from __future__ import annotations

from photomatagent.runtime.context import format_scientific_state
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.base import Tool, ToolResult


class ScientificStateInspectTool(Tool):
    name = "scientific_state_inspect"
    description = "Inspect the current scientific state (goal, claims, evidence, calculations)."
    input_schema = {
        "type": "object",
        "properties": {"section": {"type": "string", "enum": ["all", "evidence", "claims", "calculations"]}},
        "required": [],
    }

    def __init__(self, scientific_state: ScientificState) -> None:
        self._state = scientific_state

    async def execute(self, arguments: dict) -> ToolResult:
        section = arguments.get("section", "all")
        if section == "all":
            text = format_scientific_state(self._state)
        elif section == "evidence":
            text = "\n".join(
                f"- ({e.type} from {e.source}) {e.content}" for e in self._state.evidence
            )
        elif section == "claims":
            text = "\n".join(
                f"- [{c.status}] {c.statement} ({c.confidence})" for c in self._state.claims
            )
        elif section == "calculations":
            text = "\n".join(
                f"- [{c.status}] {c.task_type} -> {c.output_reference}" for c in self._state.calculations
            )
        else:
            text = "(invalid section)"
        return ToolResult(output=text or "(empty)", data={"section": section})
