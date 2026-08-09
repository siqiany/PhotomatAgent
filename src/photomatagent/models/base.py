"""Provider-neutral streaming model interface."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from photomatagent.models.types import ModelRequest, ModelStreamEvent


class ModelProvider(Protocol):
    provider: str
    model: str

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        """Stream canonical events; never execute tools inside the provider."""
        ...
