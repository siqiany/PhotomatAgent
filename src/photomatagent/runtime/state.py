"""ConversationState: the raw message history.

This is deliberately distinct from ScientificState. The conversation is what
the model literally saw; scientific facts/claims/evidence live elsewhere and
are folded into the context by the ContextBuilder.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from photomatagent.models.types import ToolCall

MessageRole = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


class ConversationState(BaseModel):
    """Ordered message history exchanged with the model."""

    messages: list[Message] = Field(default_factory=list)

    def add(self, message: Message) -> None:
        self.messages.append(message)

    def as_openai_messages(self) -> list[dict[str, object]]:
        """Minimal adapter so providers see plain dicts, not SDK objects."""
        return [m.model_dump(exclude_none=True) for m in self.messages]

    def as_context(self) -> list[Message]:
        return self.messages
