"""Build the deferred scientific tool set from capability packs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from photomatagent.scientific.capabilities.config import ScientificConfig
from photomatagent.tools.base import Tool
from photomatagent.workspace import Workspace


def build_scientific_tools(
    config: ScientificConfig | None = None,
    workspace: Workspace | None = None,
) -> list[Tool]:
    """Instantiate every capability pack and collect their tools.

    Never raises for missing dependencies: packs probe lazily and tools that
    need unavailable packages either are not registered or report
    prerequisites at call time.
    """
    from photomatagent.scientific.capabilities.defects import defects_pack
    from photomatagent.scientific.capabilities.device import device_pack
    from photomatagent.scientific.capabilities.electronic import electronic_pack
    from photomatagent.scientific.capabilities.interface import interface_pack
    from photomatagent.scientific.capabilities.ir import ir_pack
    from photomatagent.scientific.capabilities.kp import kp_pack
    from photomatagent.scientific.capabilities.literature import literature_pack
    from photomatagent.scientific.capabilities.materials import materials_pack
    from photomatagent.scientific.capabilities.optics import optics_pack
    from photomatagent.scientific.capabilities.photodetector import photodetector_pack
    from photomatagent.scientific.capabilities.quantum_dot import (
        alloy_pack,
        quantum_dot_pack,
    )
    from photomatagent.scientific.capabilities.structure import structure_pack
    from photomatagent.scientific.capabilities.transport import transport_pack
    from photomatagent.scientific.applications.vasp.tools import vasp_pack
    from photomatagent.scientific.applications.namd.tools import namd_pack
    from photomatagent.scientific.applications.magus.tools import magus_pack
    from photomatagent.scientific.capabilities.generation.tools import (
        generation_pack,
    )
    from photomatagent.scientific.capabilities.chemistry.tools import (
        chemistry_pack,
    )

    effective_config = config or ScientificConfig.from_environment(
        workspace=workspace.root if workspace else None
    )
    effective_workspace = workspace or Workspace(Path.cwd())
    packs = [
        materials_pack(effective_config),
        literature_pack(effective_config, effective_workspace),
        structure_pack(effective_config, effective_workspace),
        electronic_pack(effective_config, effective_workspace),
        defects_pack(effective_config, effective_workspace),
        transport_pack(effective_config, effective_workspace),
        device_pack(effective_config, effective_workspace),
        optics_pack(effective_config, effective_workspace),
        ir_pack(),
        quantum_dot_pack(),
        alloy_pack(),
        photodetector_pack(),
        interface_pack(),
        kp_pack(effective_workspace),
        vasp_pack(effective_workspace),
        namd_pack(effective_workspace),
        magus_pack(effective_workspace),
        generation_pack(effective_config),
        chemistry_pack(),
    ]
    tools: list[Tool] = []
    for pack in packs:
        try:
            tools.extend(pack.tools())
        except Exception:
            # A broken pack must never take down agent startup.
            continue
    builtin_names = {getattr(tool, "name", "") for tool in tools}
    tools.extend(
        _mcp_gateway_tools(
            effective_config, effective_workspace, builtin_names
        )
    )
    return tools


def _mcp_gateway_tools(
    config: ScientificConfig,
    workspace: Workspace,
    builtin_tool_names: set[str] | None = None,
) -> list[Tool]:
    """Register configured MCP servers as deferred tools through the gateway.

    Uses the workspace ``.photomatagent/mcp.json`` configuration. A failing
    server degrades to a ``<namespace>.status`` stub; it never raises.
    Adapters that merely duplicate tools the local packs already register
    (the SCNet MCP ``vasp_molecule_*`` family) are skipped by default so the
    agent never sees two entry points for the same capability; set
    ``PHOTOMATAGENT_MCP_INCLUDE_DUPLICATE_MOLECULAR=1`` to keep the MCP
    copies.
    """
    from photomatagent.mcp.manager import MCPServerManager

    manager = MCPServerManager(config.mcp_servers, workspace=workspace.root)
    try:
        return manager.register_tools(builtin_tool_names)
    except Exception:
        return []


def scientific_tools_for_namespace(
    tools: list[Tool], namespace: str
) -> list[Tool]:
    return [tool for tool in tools if tool.namespace == namespace]


def _namespace(tool: Any) -> str:
    return getattr(tool, "namespace", "core")
