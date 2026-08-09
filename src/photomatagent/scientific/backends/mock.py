"""MockBackend: instantly-completed fake calculations for loop testing."""

from __future__ import annotations

from typing import Any

from photomatagent.scientific.backends.base import ScientificCalculationBackend
from photomatagent.scientific.tasks import ScientificTask


MOCK_RESULTS: dict[str, dict[str, Any]] = {
    "band_structure": {"band_gap": 0.31, "gap_type": "direct", "method": "GGA-PBE (mock)"},
    "dos": {"band_gap": 0.31, "valence_band_max": -0.05, "conduction_band_min": 0.26},
    "relaxation": {"final_energy_ev": -12.44, "converged": True, "force_max_ev_a": 0.003},
}


class MockBackend(ScientificCalculationBackend):
    name = "mock"

    async def submit(self, request: dict[str, Any]) -> ScientificTask:
        task = ScientificTask(backend=self.name, status="RUNNING")
        # A real backend would park the task in a queue. The mock completes
        # synchronously so tests and demos are deterministic.
        task.status = "COMPLETED"
        task.result_reference = f"mock://{request.get('material', '?')}/{request.get('calculation_type', '?')}"
        return task

    async def status(self, task_id: str) -> ScientificTask:
        return ScientificTask(task_id=task_id, backend=self.name, status="COMPLETED")

    def parse(self, output_reference: str) -> dict[str, Any]:
        task_type = output_reference.rsplit("/", 1)[-1]
        return dict(MOCK_RESULTS.get(task_type, {"note": "unknown mock result type"}))


def run_mock_calculation(material: str, calculation_type: str) -> dict[str, Any]:
    """Direct helper used by the mock tool; returns structured mock results."""
    result = dict(MOCK_RESULTS.get(calculation_type, {}))
    result.setdefault("material", material)
    result.setdefault("calculation_type", calculation_type)
    result.setdefault("_mock", True)
    return result
