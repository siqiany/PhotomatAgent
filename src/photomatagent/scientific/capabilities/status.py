"""Capability status reporting for ``photomatagent scientific status``."""

from __future__ import annotations

from pathlib import Path

from photomatagent.scientific.capabilities.base import CapabilityInfo
from photomatagent.scientific.capabilities.base import CapabilityStatus
from photomatagent.scientific.capabilities.config import ScientificConfig
from photomatagent.workspace import Workspace


def probe_all_capabilities(
    config: ScientificConfig | None = None,
    workspace: Workspace | None = None,
) -> list[CapabilityInfo]:
    """Probe every pack without raising; returns one row per pack."""
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
    ]
    infos: list[CapabilityInfo] = []
    for pack in packs:
        try:
            infos.append(pack.info())
        except Exception as exc:
            infos.append(
                CapabilityInfo(
                    name=pack.name,
                    status=CapabilityStatus.ERROR,
                    detail=f"probe raised: {type(exc).__name__}: {exc}",
                )
            )
    mcp_configured = len(effective_config.mcp_servers)
    if mcp_configured:
        from photomatagent.mcp.manager import MCPServerManager
        from photomatagent.mcp.manager import MCPServerState

        manager = MCPServerManager(
            effective_config.mcp_servers, workspace=effective_workspace.root
        )
        rows = manager.status_rows()
        ready = sum(1 for row in rows if row.state is MCPServerState.READY)
        enabled = sum(1 for row in rows if row.state is not MCPServerState.DISABLED)
        infos.append(
            CapabilityInfo(
                name="materials_mcp",
                status=(
                    CapabilityStatus.AVAILABLE
                    if ready == enabled and enabled > 0
                    else CapabilityStatus.UNCONFIGURED
                ),
                detail=(
                    f"{mcp_configured} MCP server(s) configured; "
                    f"{ready}/{enabled} ready; tools registered per-server "
                    "namespace (default materials_mcp); run "
                    "`photomatagent mcp status` for a live probe"
                ),
                tools=sorted(
                    {
                        f"{row.namespace}.*"
                        for row in rows
                        if row.state is MCPServerState.READY
                    }
                    or ["materials_mcp.*"]
                ),
            )
        )
    else:
        infos.append(
            CapabilityInfo(
                name="materials_mcp",
                status=CapabilityStatus.UNCONFIGURED,
                detail="no MCP servers configured (.photomatagent/mcp.json)",
            )
        )
    return infos


def format_status_table(infos: list[CapabilityInfo]) -> list[tuple[str, str, str, str]]:
    """Return rows (name, status, version, detail) for the CLI table."""
    rows = []
    for info in sorted(infos, key=lambda item: item.name):
        rows.append((info.name, info.status.value, info.version, info.detail))
    return rows
