"""Tool abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from photomatagent.errors import ToolError


class ToolResult(BaseModel):
    """Unified result of a tool execution.

    ``output`` is the text the model sees. ``data`` is optional structured
    payload for consumers. ``state_updates`` are applied by the runtime to
    the ScientificState (e.g. new evidence or calculation records).
    """

    output: str
    is_error: bool = False
    data: dict[str, Any] = Field(default_factory=dict)
    state_updates: list[BaseModel] = Field(default_factory=list)


class Tool(ABC):
    """A callable capability. Name, description and input schema are exposed
    to the model as tool metadata."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = {"type": "object", "properties": {}}

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the tool with validated arguments."""

    def tool_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
