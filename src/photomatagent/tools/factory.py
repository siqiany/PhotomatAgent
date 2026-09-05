"""Build the default local tool registry."""

from __future__ import annotations

from pathlib import Path

from photomatagent.scientific.state import ScientificState
from photomatagent.scientific.capabilities.config import ScientificConfig
from photomatagent.scientific.capabilities.registry import build_scientific_tools
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
    application_approval_root: Path | str | None = None,
    evaluation_isolation: bool = False,
) -> ToolRegistry:
    boundary = workspace if isinstance(workspace, Workspace) else Workspace(workspace or Path.cwd())
    registry = ToolRegistry()
    safe_tools = [
        EchoTool(),
        CalculatorTool(),
        ScientificStateInspectTool(scientific_state),
        MockCalculationTool(),
    ]
    if evaluation_isolation:
        registry.register_all(safe_tools)
    else:
        registry.register_all(
            [
                ReadTool(boundary),
                GlobTool(boundary),
                GrepTool(boundary),
                WriteTool(boundary),
                EditTool(boundary),
                BashTool(boundary),
                *safe_tools,
            ]
        )
    scientific_config = ScientificConfig.from_environment(workspace=boundary.root)
    if not evaluation_isolation:
        registry.register_all(
            build_scientific_tools(
                scientific_config,
                boundary,
                vasp_approval_root=application_approval_root,
            )
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
            *([] if evaluation_isolation else [SkillViewTool(skill_loader or SkillLoader())]),
        ]
    )
    return registry
