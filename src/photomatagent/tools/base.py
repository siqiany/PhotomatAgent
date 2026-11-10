"""Tool abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, SerializerFunctionWrapHandler, model_serializer

from photomatagent.errors import ToolError
from photomatagent.models.types import ImageReference
from photomatagent.tools.exposure import ToolExposure


@dataclass(frozen=True)
class DeliveryReview:
    """A scoped presentation check from a registered capability."""

    key: str
    check: Callable[[str], str | None]
    guidance: str = ""
    summary: str = ""
    ambiguous_scope: str = ""
    explicit_scope: str = ""
    check_explicit: Callable[[str], str | None] | None = None


class DeliveryReviewSource:
    """Opt-in hooks for host-owned tools. Tool payloads cannot install hooks."""

    proposal_registration = False

    def review_request(self, goal: str) -> DeliveryReview | None:
        return None

    def review_result(self, result: ToolResult) -> DeliveryReview | None:
        return None


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
    images: tuple[ImageReference, ...] = Field(default=(), max_length=4)

    @model_serializer(mode="wrap")
    def serialize_result(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        payload: dict[str, object] = handler(self)
        if not self.images:
            payload.pop("images", None)
        return payload


class Tool(ABC):
    """A callable capability in the registry's authorized universe.

    Registration does not imply that the full schema is exposed to the model.
    ``exposure`` is consumed by the provider-independent surface planner.
    """

    name: str
    effect_class: str = "UNKNOWN"
    description: str = ""
    short_description: str = ""
    input_schema: dict[str, Any] = {"type": "object", "properties": {}}
    exposure: ToolExposure = ToolExposure.DEFERRED
    namespace: str = "core"
    source: str = "builtin"
    tags: tuple[str, ...] = ()
    searchable: bool = True
    cost_class: str = "CHEAP"  # CHEAP | MODERATE | EXPENSIVE | VERY_EXPENSIVE

    def __setattr__(self, name: str, value: Any) -> None:
        """Reject instance grafting once an authoritative registry is sealed."""

        if self.__dict__.get("_tool_sealed", False):
            raise RuntimeError(f"sealed tool {self.name!r} cannot be mutated")
        object.__setattr__(self, name, value)

    def _seal(self) -> None:
        """Freeze this instance for use by a sealed authoritative registry."""

        object.__setattr__(self, "_tool_sealed", True)

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
            "searchable": self.searchable,
            "cost_class": self.cost_class,
        }
