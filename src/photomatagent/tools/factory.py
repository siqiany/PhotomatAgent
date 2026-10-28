"""Build the default local tool registry."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from photomatagent.scientific.state import ScientificState
from photomatagent.scientific.capabilities.config import ScientificConfig
from photomatagent.scientific.capabilities.registry import build_scientific_tools
from photomatagent.skills.loader import SkillLoader
from photomatagent.tools.bash import BashTool
from photomatagent.tools.base import Tool, ToolResult
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

if TYPE_CHECKING:
    from photomatagent.capability_acquisition.sandbox import SandboxService
    from photomatagent.scientific.evolution.qd.executor import QDExecutionContext


class _QDAttemptFileTool(Tool):
    """Keep all file paths inside one trial and freeze genome scripts against edits."""

    def __init__(self, delegate: Tool, workspace: Workspace, protected: frozenset[str]) -> None:
        self.delegate = delegate
        self.workspace = workspace
        self.protected = protected
        self.name = delegate.name
        self.description = delegate.description
        self.short_description = delegate.short_description
        self.input_schema = delegate.input_schema
        self.exposure = delegate.exposure
        self.namespace = delegate.namespace
        self.source = delegate.source
        self.tags = delegate.tags
        self.searchable = delegate.searchable
        self.cost_class = delegate.cost_class

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        for key in ("path", "glob"):
            value = arguments.get(key)
            if value is not None:
                path = Path(str(value))
                if path.is_absolute() or ".." in path.parts:
                    return ToolResult(output="QD_PATH_INVALID", is_error=True)
        if self.name in {"write", "edit"}:
            value = str(arguments["path"])
            try:
                path = self.workspace.resolve(value, must_exist=False)
                relative = self.workspace.relative(path)
            except (ValueError, OSError):
                return ToolResult(output="QD_PATH_INVALID", is_error=True)
            current = self.workspace.root
            for part in Path(value).parts:
                current /= part
                if current.is_symlink():
                    return ToolResult(output="QD_PATH_INVALID", is_error=True)
            if relative in self.protected or not relative.startswith(("drafts/", "tmp/")):
                return ToolResult(output="QD_PATH_READ_ONLY", is_error=True)
        return await self.delegate.execute(arguments)


class _QDFrozenServiceTool(Tool):
    """Reject changed script/environment bindings before any service backend call."""

    def __init__(self, delegate: Tool, context: QDExecutionContext) -> None:
        self.delegate = delegate
        self.delegate_execute = delegate.execute
        self.context = context
        self.name = delegate.name
        self.description = delegate.description
        self.short_description = delegate.short_description
        self.input_schema = delegate.input_schema
        self.exposure = delegate.exposure
        self.namespace = delegate.namespace
        self.source = delegate.source
        self.tags = delegate.tags
        self.searchable = delegate.searchable
        self.cost_class = delegate.cost_class

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        step_id = self.context.workflow_gate.begin(self.name, arguments)
        if step_id is None:
            return ToolResult(output="GENOME_MISMATCH", is_error=True)
        result = await self.delegate_execute(arguments)
        self.context.workflow_gate.finish(step_id, not result.is_error)
        return result


def _qd_registry(context: QDExecutionContext, config: ToolSurfaceConfig) -> ToolRegistry:
    boundary = context.attempt_workspace
    protected = frozenset(ref.path for ref in context.genome.script_refs) | context.protected_outputs
    registry = ToolRegistry()
    registry.register_all([
        _QDAttemptFileTool(tool, boundary, protected)
        for tool in (
            ReadTool(boundary), GlobTool(boundary), GrepTool(boundary),
            WriteTool(boundary), EditTool(boundary),
        )
    ])
    allowed = {step.binding_id for step in context.genome.steps if step.kind == "registered_tool"}
    if any(step.kind == "script" for step in context.genome.steps):
        allowed.add("script_run")
    reserved = {"read", "glob", "grep", "write", "edit", "bash", "tool_search", "tool_describe", "tool_call"}
    for name, tool in context.service_refs.items():
        if name != tool.name or name not in allowed or name in reserved:
            raise ValueError("QD_SERVICE_BINDING_UNAUTHORIZED")
        if isinstance(tool, BashTool) or tool.namespace in {"hpc", "scientific", "controller", "checker", "store", "reference", "mcp"}:
            raise ValueError("QD_SERVICE_BINDING_UNAUTHORIZED")
        if name.startswith(("vasp.", "namd.", "magus.", "hpc.", "scientific.", "generation.", "structure.", "mcp.")):
            raise ValueError("QD_SERVICE_BINDING_UNAUTHORIZED")
        registry.register(_QDFrozenServiceTool(tool, context))
    catalog = ToolCatalog(registry)
    registry.register_all([
        ToolSearchTool(catalog, default_limit=config.search_default_limit, max_limit=config.search_max_limit),
        ToolDescribeTool(catalog), ToolCallBridge(),
    ])
    registry.seal()
    return registry


def create_default_registry(
    scientific_state: ScientificState,
    workspace: Workspace | Path | str | None = None,
    *,
    skill_loader: SkillLoader | None = None,
    surface_config: ToolSurfaceConfig | None = None,
    application_approval_root: Path | str | None = None,
    evaluation_isolation: bool = False,
    qd_context: QDExecutionContext | None = None,
    capability_sandbox_service: SandboxService | None = None,
    resource_owners: list[object] | None = None,
    local_skill_task_id: str | None = None,
    local_skill_run_id: str | None = None,
    local_skill_service: Any = None,
) -> ToolRegistry:
    if qd_context is not None:
        if evaluation_isolation:
            raise ValueError("QD_REGISTRY_MODE_CONFLICT")
        return _qd_registry(qd_context, surface_config or ToolSurfaceConfig())
    boundary = workspace if isinstance(workspace, Workspace) else Workspace(workspace or Path.cwd())
    effective_skill_loader = skill_loader or SkillLoader(
        published_root=boundary.root / ".photomatagent" / "qd" / "published-skills",
        local_trial_workspace=boundary.root if local_skill_task_id else None,
        local_task_id=local_skill_task_id,
        local_run_id=local_skill_run_id,
    )
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
                scientific_state=scientific_state,
                sandbox_service=capability_sandbox_service,
                resource_owners=resource_owners,
                published_skill_source=getattr(effective_skill_loader, "published_source", None),
            )
        )
    if local_skill_task_id is not None and not evaluation_isolation:
        from photomatagent.application.skills import LocalSkillService
        from photomatagent.tools.skill_run import SkillRunTool
        registry.register(SkillRunTool(local_skill_service or LocalSkillService(boundary.root), local_skill_task_id, local_skill_run_id))
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
            *([] if evaluation_isolation else [SkillViewTool(effective_skill_loader)]),
        ]
    )
    return registry
