"""ConversationState stores canonical messages, separate from scientific state."""

from __future__ import annotations

from pydantic import BaseModel, Field

from photomatagent.models.types import ModelMessage


class ConversationState(BaseModel):
    messages: list[ModelMessage] = Field(default_factory=list)

    def add(self, message: ModelMessage) -> None:
        self.messages.append(message)

    def as_context(self) -> list[ModelMessage]:
        return list(self.messages)
