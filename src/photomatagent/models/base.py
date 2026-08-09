"""ModelProvider protocol."""

from __future__ import annotations

from typing import Protocol

from photomatagent.models.types import ModelResponse
from photomatagent.runtime.state import Message
from photomatagent.tools.base import Tool


class ModelProvider(Protocol):
    """A model the runtime can call.

    The runtime only knows this protocol; it never sees OpenAI/Anthropic
    objects. A provider maps its native streaming/tool-calling semantics into
    the unified ModelResponse.
    """

    name: str

    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool],
    ) -> ModelResponse: ...
