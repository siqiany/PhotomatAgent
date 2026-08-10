"""Capability status reporting for ``photomatagent scientific status``."""

from __future__ import annotations

from pathlib import Path

from photomatagent.scientific.capabilities.base import CapabilityInfo
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
    infos: list[CapabilityInfo] = []
    for pack in packs:
        try:
            infos.append(pack.info())
        except Exception as exc:
            infos.append(
                CapabilityInfo(
                    name=pack.name,
                    status="ERROR",
                    detail=f"probe raised: {type(exc).__name__}: {exc}",
                )
            )
    mcp_configured = len(effective_config.mcp_servers)
    if mcp_configured:
        infos.append(
            CapabilityInfo(
                name="materials_mcp",
                status="AVAILABLE" if _mcp_ok(effective_config) else "UNCONFIGURED",
                detail=(
                    f"{mcp_configured} MCP server(s) configured; tools registered "
                    "under namespace materials_mcp"
                ),
                tools=["materials_mcp.*"],
            )
        )
    else:
        infos.append(
            CapabilityInfo(
                name="materials_mcp",
                status="UNCONFIGURED",
                detail="no MCP servers configured (.photomatagent/mcp.json)",
            )
        )
    return infos


def _mcp_ok(config: ScientificConfig) -> bool:
    try:
        from photomatagent.mcp.client import connect

        return all(connect(server) for server in config.mcp_servers)
    except Exception:
        return False


def format_status_table(infos: list[CapabilityInfo]) -> list[tuple[str, str, str, str]]:
    """Return rows (name, status, version, detail) for the CLI table."""
    rows = []
    for info in sorted(infos, key=lambda item: item.name):
        rows.append((info.name, info.status.value, info.version, info.detail))
    return rows
