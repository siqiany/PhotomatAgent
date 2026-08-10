"""Build the default local tool registry."""

from __future__ import annotations

from pathlib import Path

from photomatagent.scientific.state import ScientificState
from photomatagent.skills.loader import SkillLoader
from photomatagent.tools.bash import BashTool
from photomatagent.tools.bridges import (
    SkillViewTool,
    ToolCallBridge,
    ToolDescribeTool,
    ToolSearchTool,
)
from photomatagent.tools.calculator import CalculatorTool
from photomatagent.tools.echo import EchoTool
from photomatagent.tools.edit import EditTool
from photomatagent.tools.glob import GlobTool
from photomatagent.tools.grep import GrepTool
from photomatagent.tools.mock_calculation import MockCalculationTool
from photomatagent.tools.read import ReadTool
from photomatagent.tools.registry import ToolRegistry
from photomatagent.tools.scientific_state_inspect import ScientificStateInspectTool
from photomatagent.tools.surface import ToolCatalog, ToolSurfaceConfig
from photomatagent.tools.write import WriteTool
from photomatagent.workspace import Workspace


def create_default_registry(
    scientific_state: ScientificState,
    workspace: Workspace | Path | str | None = None,
    *,
    skill_loader: SkillLoader | None = None,
    surface_config: ToolSurfaceConfig | None = None,
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
    config = surface_config or ToolSurfaceConfig()
    catalog = ToolCatalog(registry)
    registry.register_all(
        [
            ToolSearchTool(
                catalog,
                default_limit=config.search_default_limit,
                max_limit=config.search_max_limit,
            ),
            ToolDescribeTool(catalog),
            ToolCallBridge(),
            SkillViewTool(skill_loader or SkillLoader()),
        ]
    )
    return registry
