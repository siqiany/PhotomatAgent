"""Build the default local tool registry."""

from __future__ import annotations

from pathlib import Path

from photomatagent.scientific.state import ScientificState
from photomatagent.tools.bash import BashTool
from photomatagent.tools.calculator import CalculatorTool
from photomatagent.tools.echo import EchoTool
from photomatagent.tools.edit import EditTool
from photomatagent.tools.glob import GlobTool
from photomatagent.tools.grep import GrepTool
from photomatagent.tools.mock_calculation import MockCalculationTool
from photomatagent.tools.read import ReadTool
from photomatagent.tools.registry import ToolRegistry
from photomatagent.tools.scientific_state_inspect import ScientificStateInspectTool
from photomatagent.tools.write import WriteTool
from photomatagent.workspace import Workspace


def create_default_registry(
    scientific_state: ScientificState,
    workspace: Workspace | Path | str | None = None,
) -> ToolRegistry:
    boundary = workspace if isinstance(workspace, Workspace) else Workspace(workspace or Path.cwd())
    registry = ToolRegistry()
    registry.register_all(
        [
            ReadTool(boundary),
            GlobTool(boundary),
            GrepTool(boundary),
            WriteTool(boundary),
            EditTool(boundary),
            BashTool(boundary),
            EchoTool(),
            CalculatorTool(),
            ScientificStateInspectTool(scientific_state),
            MockCalculationTool(),
        ]
    )
    return registry
