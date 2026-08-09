"""ToolRegistry: registration, lookup, listing, and argument validation."""

from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter, ValidationError

from photomatagent.tools.base import Tool, ToolError


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def register_all(self, tools: list[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name]

    def list_tools(self) -> list[Tool]:
        return sorted(self._tools.values(), key=lambda t: t.name)

    def tool_metadata_list(self) -> list[dict[str, Any]]:
        return [t.tool_metadata() for t in self.list_tools()]

    def validate_arguments(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate arguments against the tool's JSON Schema.

        Uses ``additionalProperties: false`` semantics via a per-call adapter
        so extra/missing keys fail loudly rather than silently pass.
        """
        tool = self.get(name)
        schema = tool.input_schema
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise ToolError(f"tool {name} has an invalid input_schema")
        adapter = TypeAdapter(dict[str, Any])
        try:
            adapter.validate_python(arguments)
        except ValidationError as exc:
            raise ToolError(f"invalid arguments for {name}: {exc.errors()[:3]}") from exc
        if not isinstance(arguments, dict):
            raise ToolError(f"arguments for {name} must be an object")
        required = schema.get("required", [])
        for key in required:
            if key not in arguments:
                raise ToolError(f"missing required argument {key!r} for {name}")
        properties = schema.get("properties", {})
        for key in arguments:
            if key not in properties:
                raise ToolError(f"unexpected argument {key!r} for {name}")
        return arguments
