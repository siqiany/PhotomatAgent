"""MCP server configuration schema and workspace file loading.

The gateway is configured from ``.photomatagent/mcp.json`` (or ``mcp.yaml``),
with two accepted shapes:

* new style: ``{"servers": {"<name>": { ... }}}`` (dict keyed by server name)
* legacy style: ``{"servers": [ {"name": ..., ...} ]}`` (list of entries)

Environment references of the form ``${VAR}`` inside ``env`` values (and in
``command``/``args``/``url``) are expanded at start time from the process
environment, falling back to the workspace ``.env`` file when a variable is
not exported (process environment always wins). Expanded secrets are never
logged by the gateway.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from photomatagent.tools.exposure import ToolExposure

Transport = Literal["stdio", "http"]
TrustLevel = Literal["local_trusted", "local_isolated", "remote"]

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True)
class MCPServerConfig:
    """Declarative description of one MCP server."""

    name: str
    workspace: str | None = field(default=None, compare=False)
    enabled: bool = True
    transport: Transport = "stdio"
    command: str | None = None
    args: list[str] = field(default_factory=list)
    url: str | None = None
    namespace: str = ""
    env: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    startup_timeout_seconds: float = 20.0
    tool_exposure: str = "deferred"
    description: str = ""
    source: str = ""
    version: str = ""
    trust_level: TrustLevel = "local_isolated"
    healthcheck_tool: str | None = None

    @property
    def effective_namespace(self) -> str:
        return self.namespace or _namespace_from_name(self.name)

    @property
    def exposure(self) -> ToolExposure:
        value = (self.tool_exposure or "deferred").strip().lower()
        if value == "direct":
            return ToolExposure.DIRECT
        if value == "hidden":
            return ToolExposure.HIDDEN
        return ToolExposure.DEFERRED

    def resolved_env(
        self, environ: dict[str, str] | None = None
    ) -> tuple[dict[str, str], list[str]]:
        """Expand ``${VAR}`` references; returns (env, missing variable names)."""
        source = self._env_source() if environ is None else dict(environ)
        missing: list[str] = []
        resolved: dict[str, str] = {}
        for key, value in self.env.items():
            expanded, refs = _expand(value, source)
            for ref in refs:
                if ref not in source or not source[ref]:
                    missing.append(ref)
            resolved[key] = expanded
        return resolved, sorted(set(missing))

    def resolved_command(self) -> str | None:
        """Expand env references in the command string."""
        if not self.command:
            return None
        expanded, _ = _expand(self.command, self._env_source())
        return expanded

    def resolved_args(self) -> list[str]:
        expanded, _ = _expand_args(self.args, self._env_source())
        return expanded

    def resolved_url(self) -> str | None:
        if not self.url:
            return None
        expanded, _ = _expand(self.url, self._env_source())
        return expanded

    def _env_source(self) -> dict[str, str]:
        """Process environment overlaid with workspace ``.env`` values."""
        source = dict(os.environ)
        if not self.workspace:
            return source
        try:
            from dotenv import dotenv_values

            dotenv = dotenv_values(Path(self.workspace) / ".env") or {}
        except Exception:
            return source
        for key, value in dotenv.items():
            if value and key not in source:
                source[key] = value
        return source


def _expand(value: str, environ: dict[str, str]) -> tuple[str, list[str]]:
    refs: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        refs.append(name)
        return environ.get(name, "")

    return _ENV_REF.sub(replace, value), refs


def _expand_args(
    args: list[str], environ: dict[str, str]
) -> tuple[list[str], list[str]]:
    refs: list[str] = []
    expanded: list[str] = []
    for arg in args:
        value, found = _expand(arg, environ)
        refs.extend(found)
        expanded.append(value)
    return expanded, refs


def _namespace_from_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.]+", "_", name.strip()).strip("_")
    cleaned = cleaned.replace("-", "_")
    return cleaned or "mcp"


def load_mcp_servers(workspace: Path) -> list[MCPServerConfig]:
    """Read optional MCP server config; never raises.

    A missing or malformed config yields no servers. Individual invalid
    entries are skipped so one typo cannot disable the rest.
    """
    candidates = [
        workspace / ".photomatagent" / "mcp.json",
        workspace / ".photomatagent" / "mcp.yaml",
        workspace / "mcp.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        raw = _parse_file(path)
        servers = _servers_section(raw)
        if servers is None:
            continue
        specs: list[MCPServerConfig] = []
        for entry in servers:
            parsed = _parse_server_entry(entry, workspace=workspace)
            if parsed is not None:
                specs.append(parsed)
        return specs
    return []


def _servers_section(raw: Any) -> list[Any] | None:
    if not isinstance(raw, dict):
        return None
    section = raw.get("servers")
    if isinstance(section, dict):
        entries: list[Any] = []
        for name, value in section.items():
            if isinstance(value, dict):
                entries.append({"name": name, **value})
        return entries
    if isinstance(section, list):
        return section
    return None


def _parse_server_entry(
    entry: Any, *, workspace: Path | None = None
) -> MCPServerConfig | None:
    if not isinstance(entry, dict) or not entry.get("name"):
        return None
    name = str(entry["name"])
    transport = str(entry.get("transport", "stdio")).strip().lower()
    if transport in {"sse", "streamable-http"}:
        transport = "http"
    if transport not in {"stdio", "http"}:
        transport = "stdio"
    args = entry.get("args", [])
    env = entry.get("env", {})
    return MCPServerConfig(
        name=name,
        workspace=str(workspace) if workspace is not None else None,
        enabled=_truthy(entry.get("enabled", True)),
        transport=transport,  # type: ignore[arg-type]
        command=str(entry["command"]) if entry.get("command") else None,
        args=[str(item) for item in args] if isinstance(args, list) else [],
        url=str(entry["url"]) if entry.get("url") else None,
        namespace=str(entry.get("namespace", "")),
        env=(
            {str(key): str(value) for key, value in env.items()}
            if isinstance(env, dict)
            else {}
        ),
        timeout_seconds=_float(entry.get("timeout"), 30.0),
        startup_timeout_seconds=_float(entry.get("startup_timeout"), 20.0),
        tool_exposure=str(entry.get("tool_exposure", "deferred")),
        description=str(entry.get("description", "")),
        source=str(entry.get("source", "")),
        version=str(entry.get("version", "")),
        trust_level=str(entry.get("trust_level", "local_isolated")),  # type: ignore[arg-type]
        healthcheck_tool=(
            str(entry["healthcheck_tool"]) if entry.get("healthcheck_tool") else None
        ),
    )


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
