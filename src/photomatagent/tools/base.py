"""Tool abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from photomatagent.errors import ToolError
from photomatagent.tools.exposure import ToolExposure


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
    """A callable capability in the registry's authorized universe.

    Registration does not imply that the full schema is exposed to the model.
    ``exposure`` is consumed by the provider-independent surface planner.
    """

    name: str
    description: str = ""
    short_description: str = ""
    input_schema: dict[str, Any] = {"type": "object", "properties": {}}
    exposure: ToolExposure = ToolExposure.DEFERRED
    namespace: str = "core"
    source: str = "builtin"
    tags: tuple[str, ...] = ()

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the tool with validated arguments."""

    def tool_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "exposure": self.exposure.value,
            "namespace": self.namespace,
            "source": self.source,
            "tags": list(self.tags),
        }
