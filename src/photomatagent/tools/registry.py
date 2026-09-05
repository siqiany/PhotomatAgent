"""ToolRegistry: registration, lookup, listing, and argument validation."""

from __future__ import annotations

from typing import Any

from photomatagent.errors import ToolValidationError
from photomatagent.models.types import ToolDefinition
from photomatagent.tools.base import Tool
from photomatagent.tools.exposure import ToolExposure


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

    def definition(self, name: str) -> ToolDefinition:
        tool = self.get(name)
        return ToolDefinition(
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
            namespace=tool.namespace,
        )

    def definitions(
        self, exposure: ToolExposure | None = None
    ) -> list[ToolDefinition]:
        return [
            self.definition(tool.name)
            for tool in self.list_tools()
            if exposure is None or tool.exposure is exposure
        ]

    def tools_for_exposure(self, exposure: ToolExposure) -> list[Tool]:
        return [tool for tool in self.list_tools() if tool.exposure is exposure]

    def validate_arguments(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate arguments against the tool's JSON Schema.

        Uses ``additionalProperties: false`` semantics via a per-call adapter
        so extra/missing keys fail loudly rather than silently pass.
        """
        tool = self.get(name)
        schema = tool.input_schema
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise ToolValidationError(f"tool {name} has an invalid input_schema")
        if not isinstance(arguments, dict):
            raise ToolValidationError(f"arguments for {name} must be an object")
        required = schema.get("required", [])
        for key in required:
            if key not in arguments:
                raise ToolValidationError(f"missing required argument {key!r} for {name}")
        properties = schema.get("properties", {})
        for key in arguments:
            if key not in properties:
                raise ToolValidationError(f"unexpected argument {key!r} for {name}")
            self._validate_value(name, key, arguments[key], properties[key])
        return arguments

    def _validate_value(
        self, tool_name: str, key: str, value: Any, schema: dict[str, Any]
    ) -> None:
        expected = schema.get("type")
        checks = {
            "string": lambda item: isinstance(item, str),
            "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
            "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
            "boolean": lambda item: isinstance(item, bool),
            "array": lambda item: isinstance(item, list),
            "object": lambda item: isinstance(item, dict),
        }
        if expected in checks and not checks[expected](value):
            raise ToolValidationError(
                f"argument {key!r} for {tool_name} must be {expected}"
            )
        if "enum" in schema and value not in schema["enum"]:
            raise ToolValidationError(
                f"argument {key!r} for {tool_name} must be one of {schema['enum']}"
            )
        if isinstance(value, (int, float)):
            if schema.get("minimum") is not None and value < schema["minimum"]:
                raise ToolValidationError(f"argument {key!r} is below minimum")
            if schema.get("maximum") is not None and value > schema["maximum"]:
                raise ToolValidationError(f"argument {key!r} is above maximum")
