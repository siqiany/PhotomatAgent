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
    from photomatagent.scientific.capabilities.ir import ir_pack
    from photomatagent.scientific.capabilities.literature import literature_pack
    from photomatagent.scientific.capabilities.materials import materials_pack
    from photomatagent.scientific.capabilities.optics import optics_pack
    from photomatagent.scientific.capabilities.structure import structure_pack
    from photomatagent.scientific.capabilities.transport import transport_pack

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
    ]
    tools: list[Tool] = []
    for pack in packs:
        try:
            tools.extend(pack.tools())
        except Exception:
            # A broken pack must never take down agent startup.
            continue
    tools.extend(_mcp_materials_tools(effective_config))
    return tools


def _mcp_materials_tools(config: ScientificConfig) -> list[Tool]:
    """Register configured MCP servers under ``materials_mcp`` (optional)."""
    from photomatagent.mcp.client import connect

    tools: list[Tool] = []
    for server in config.mcp_servers:
        try:
            tools.extend(connect(server))
        except Exception:
            # MCP failure must never prevent agent startup.
            continue
    return tools


def scientific_tools_for_namespace(
    tools: list[Tool], namespace: str
) -> list[Tool]:
    return [tool for tool in tools if tool.namespace == namespace]


def _namespace(tool: Any) -> str:
    return getattr(tool, "namespace", "core")
