"""ToolRegistry: registration, lookup, listing, and argument validation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from photomatagent.errors import ToolValidationError
from photomatagent.models.types import ToolDefinition
from photomatagent.tools.base import Tool
from photomatagent.tools.exposure import ToolExposure


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Mapping[str, Tool] = {}
        self._sealed = False
        self._sealed_contracts: dict[str, tuple[object, ...]] = {}

    @property
    def sealed(self) -> bool:
        return self._sealed

    def seal(self) -> None:
        """Make registrations and registered tool instances immutable.

        Sealing is intentionally explicit: normal interactive registries may
        still be assembled incrementally, while security-bound runtimes can
        close the validation-to-execution mutation window.
        """

        if self._sealed:
            self._assert_integrity()
            return
        tools = dict(self._tools)
        for tool in tools.values():
            Tool._seal(tool)
        self._sealed_contracts = {
            name: self._tool_contract(tool) for name, tool in tools.items()
        }
        self._tools = MappingProxyType(tools)
        self._sealed = True

    def register(self, tool: Tool) -> None:
        if self._sealed:
            raise RuntimeError("sealed tool registry cannot be modified")
        if not isinstance(self._tools, dict):
            raise RuntimeError("sealed tool registry cannot be modified")
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def register_all(self, tools: list[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Tool:
        self._assert_integrity()
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name]

    def list_tools(self) -> list[Tool]:
        self._assert_integrity()
        return sorted(self._tools.values(), key=lambda t: t.name)

    @staticmethod
    def _tool_contract(tool: Tool) -> tuple[object, ...]:
        execute = tool.execute
        execute_owner = getattr(execute, "__self__", None)
        execute_function = getattr(execute, "__func__", execute)
        try:
            schema = json.dumps(
                tool.input_schema,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        except (TypeError, ValueError):
            schema = repr(tool.input_schema)
        metadata = (
            tool.name,
            tool.description,
            tool.short_description,
            schema,
            tool.exposure,
            tool.namespace,
            tool.source,
            tool.tags,
            tool.searchable,
            tool.cost_class,
        )
        instance_state = tuple(
            sorted((key, id(value)) for key, value in vars(tool).items())
        )
        return (
            type(tool),
            id(tool),
            id(execute_owner),
            id(execute_function),
            metadata,
            instance_state,
        )

    def _assert_integrity(self) -> None:
        if not self._sealed:
            return
        if set(self._tools) != set(self._sealed_contracts):
            raise RuntimeError("sealed tool registry mapping was mutated")
        for name, expected in self._sealed_contracts.items():
            tool = self._tools.get(name)
            if tool is None or self._tool_contract(tool) != expected:
                raise RuntimeError(f"sealed tool {name!r} was mutated")

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
