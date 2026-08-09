"""Build the default tool registry for a runtime."""

from __future__ import annotations

from photomatagent.scientific.state import ScientificState
from photomatagent.tools.calculator import CalculatorTool
from photomatagent.tools.echo import EchoTool
from photomatagent.tools.mock_calculation import MockCalculationTool
from photomatagent.tools.registry import ToolRegistry
from photomatagent.tools.scientific_state_inspect import ScientificStateInspectTool


def create_default_registry(scientific_state: ScientificState) -> ToolRegistry:
    """Standard tool set: echo, calculator, state inspect, mock calculation."""
    registry = ToolRegistry()
    registry.register_all(
        [
            EchoTool(),
            CalculatorTool(),
            ScientificStateInspectTool(scientific_state),
            MockCalculationTool(),
        ]
    )
    return registry
