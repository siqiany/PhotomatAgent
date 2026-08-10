"""Scientific capability configuration (limits, secrets, MCP servers)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from photomatagent.mcp.client import MCPServerSpec


def _boolish(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name, "")
    try:
        return int(value) if value.strip() else default
    except ValueError:
        return default


@dataclass(frozen=True)
class ScientificConfig:
    """Hard limits and integration settings for scientific capabilities."""

    materials_api_key_env: str = "MATERIALS_API_KEY"
    materials_max_results: int = 10
    literature_max_papers: int = 5
    literature_max_chars: int = 4000
    structure_output_dir: str = "output/scientific"
    mcp_servers: list[MCPServerSpec] = field(default_factory=list)

    @classmethod
    def from_environment(
        cls, *, workspace: Path | str | None = None
    ) -> "ScientificConfig":
        root = Path(workspace or Path.cwd())
        servers = _load_mcp_servers(root)
        return cls(
            materials_api_key_env=os.environ.get(
                "PHOTOMATAGENT_MATERIALS_KEY_ENV", "MATERIALS_API_KEY"
            ),
            materials_max_results=_int_env(
                "PHOTOMATAGENT_MATERIALS_MAX_RESULTS", 10
            ),
            literature_max_papers=_int_env(
                "PHOTOMATAGENT_LITERATURE_MAX_PAPERS", 5
            ),
            literature_max_chars=_int_env(
                "PHOTOMATAGENT_LITERATURE_MAX_CHARS", 4000
            ),
            mcp_servers=servers,
        )

    def materials_api_key(self) -> str:
        return os.environ.get(self.materials_api_key_env, "").strip()


def _load_mcp_servers(workspace: Path) -> list[MCPServerSpec]:
    """Read optional MCP server config from .photomatagent/mcp.json.

    MCP is never required: a missing or malformed config yields no servers and
    a broken configured server is reported by the probe, not by startup.
    """
    candidates = [
        workspace / ".photomatagent" / "mcp.json",
        workspace / ".photomatagent" / "mcp.yaml",
        workspace / "mcp.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            raw = _parse_file(path)
        except Exception:
            return []
        servers = raw.get("servers", []) if isinstance(raw, dict) else None
        if not isinstance(servers, list):
            return []
        specs: list[MCPServerSpec] = []
        for entry in servers:
            if not isinstance(entry, dict) or not entry.get("name"):
                continue
            specs.append(
                MCPServerSpec(
                    name=str(entry["name"]),
                    command=str(entry["command"]) if entry.get("command") else None,
                    args=[str(item) for item in entry.get("args", [])]
                    if isinstance(entry.get("args"), list)
                    else None,
                    url=str(entry["url"]) if entry.get("url") else None,
                    env=dict(entry.get("env", {})),
                )
            )
        return specs
    return []


def _parse_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read MCP YAML config") from exc
    value = yaml.safe_load(text)
    return value if isinstance(value, dict) else {}

