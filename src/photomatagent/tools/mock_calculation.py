"""mock.run_calculation: a fake scientific tool that updates scientific state."""

from __future__ import annotations

import json

from photomatagent.scientific.backends.mock import MockBackend, run_mock_calculation
from photomatagent.scientific.calculations import CalculationRecord
from photomatagent.scientific.evidence import Evidence
from photomatagent.tools.base import Tool, ToolError, ToolResult


class MockCalculationTool(Tool):
    name = "mock.run_calculation"
    description = (
        "TEST-ONLY placeholder: run a fake scientific calculation for a "
        "material. Always returns hardcoded placeholder results (e.g. band "
        "gap 0.31 eV) and records evidence in the scientific state. Never "
        "call this for real research; it is excluded from tool_search. "
        "Types: band_structure, dos, relaxation."
    )
    namespace = "mock"
    tags = ("scientific", "materials", "calculation", "band structure", "dos")
    # Test fixture only: do not let the agent discover this via tool_search.
    searchable = False
    input_schema = {
        "type": "object",
        "properties": {
            "material": {"type": "string", "description": "Chemical formula, e.g. GaAs"},
            "calculation_type": {
                "type": "string",
                "enum": ["band_structure", "dos", "relaxation"],
            },
        },
        "required": ["material", "calculation_type"],
    }

    def __init__(self, backend: MockBackend | None = None) -> None:
        self.backend = backend or MockBackend()

    async def execute(self, arguments: dict) -> ToolResult:
        material = arguments["material"]
        calculation_type = arguments["calculation_type"]
        results = run_mock_calculation(material, calculation_type)

        task = await self.backend.submit(
            {"material": material, "calculation_type": calculation_type}
        )
        if task.status != "COMPLETED":
            raise ToolError(f"mock calculation did not complete: {task.status}")

        output_text = json.dumps(results, indent=2)
        record = CalculationRecord(
            backend=self.backend.name,
            task_type=calculation_type,
            status="completed",
            input_reference={"material": material, "calculation_type": calculation_type},
            output_reference=task.result_reference,
            metadata={"results": results},
        )
        evidence = Evidence(
            type="calculation",
            source=f"mock:{self.backend.name}",
            content=(
                f"Mock {calculation_type} calculation for {material} "
                f"returned {json.dumps(results)}"
            ),
            confidence=0.5,
            provenance={"tool": self.name, "task_id": task.task_id, "calculation_id": record.id},
        )
        return ToolResult(
            output=f"Calculation completed.\n{output_text}",
            data={"results": results, "calculation_id": record.id, "task_id": task.task_id},
            state_updates=[record, evidence],
        )
