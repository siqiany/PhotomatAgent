"""Stable direct bridge tools for deferred capability discovery and use."""

from __future__ import annotations

import json
from typing import Any

from photomatagent.skills.loader import SkillLoader
from photomatagent.tools.base import Tool, ToolResult
from photomatagent.tools.exposure import ToolExposure
from photomatagent.tools.surface import ToolCatalog, compact_parameter_help


class ToolSearchTool(Tool):
    name = "tool_search"
    description = "Search deferred capabilities by keywords without loading their full schemas."
    exposure = ToolExposure.DIRECT
    tags = ("capability", "search", "discovery")
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Capability keywords."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            "namespace": {"type": "string"},
        },
        "required": ["query"],
    }

    def __init__(self, catalog: ToolCatalog, *, default_limit: int = 5, max_limit: int = 20) -> None:
        self.catalog = catalog
        self.default_limit = default_limit
        self.max_limit = max_limit

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        limit = min(int(arguments.get("limit", self.default_limit)), self.max_limit)
        namespace = arguments.get("namespace")
        matches = self.catalog.search(
            str(arguments["query"]),
            limit=limit,
            namespace=str(namespace) if namespace else None,
        )
        cards = [
            {
                "name": match.entry.name,
                "description": match.entry.short_description,
                "namespace": match.entry.namespace,
                "required_parameters": match.entry.required_parameters,
            }
            for match in matches
        ]
        return ToolResult(
            output=json.dumps({"matches": cards}, ensure_ascii=False, separators=(",", ":")),
            data={"matches": cards},
        )


class ToolDescribeTool(Tool):
    name = "tool_describe"
    description = "Load calling instructions for one deferred capability returned by tool_search."
    exposure = ToolExposure.DIRECT
    tags = ("capability", "schema", "discovery")
    input_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    def __init__(self, catalog: ToolCatalog) -> None:
        self.catalog = catalog

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        name = str(arguments["name"])
        entry = self.catalog.get(name)
        if entry is None:
            payload = {
                "error": "not_deferred_or_unavailable",
                "tool": name,
                "hint": "Use tool_search; direct and hidden tools cannot be described here.",
            }
            return ToolResult(
                output=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                is_error=True,
                data=payload,
            )
        definition = entry.full_schema_reference
        description_payload: dict[str, object] = {
            "name": entry.name,
            "description": entry.full_description,
            "parameters": compact_parameter_help(definition),
            "required_parameters": entry.required_parameters,
            "schema": definition.input_schema,
        }
        return ToolResult(
            output=json.dumps(
                description_payload, ensure_ascii=False, separators=(",", ":")
            ),
            data=description_payload,
        )


class ToolCallBridge(Tool):
    """Marker tool; AgentRuntime unwraps it before normal execution."""

    name = "tool_call"
    description = "Invoke one deferred capability using its name and JSON arguments."
    exposure = ToolExposure.DIRECT
    tags = ("capability", "call", "bridge")
    input_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Deferred tool name."},
            "arguments": {"type": "object", "description": "Arguments for that tool."},
        },
        "required": ["name", "arguments"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        raise RuntimeError("tool_call must be unwrapped by AgentRuntime")


class SkillViewTool(Tool):
    name = "skill_view"
    description = "Load one skill's SKILL.md or a named reference only when needed."
    exposure = ToolExposure.DIRECT
    namespace = "skills"
    tags = ("skill", "instructions", "reference")
    input_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "path": {"type": "string", "description": "Optional path below the skill directory."},
        },
        "required": ["name"],
    }

    def __init__(self, loader: SkillLoader) -> None:
        self.loader = loader

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        name = str(arguments["name"])
        path = arguments.get("path")
        try:
            content, resolved = self.loader.view(name, str(path) if path else None)
        except (KeyError, OSError, ValueError) as exc:
            return ToolResult(output=f"skill_view failed: {exc}", is_error=True)
        return ToolResult(
            output=content,
            data={"skill": name, "path": resolved},
        )
