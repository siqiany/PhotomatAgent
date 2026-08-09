from __future__ import annotations

from photomatagent.runtime.context import ContextBuilder
from photomatagent.runtime.state import ConversationState, Message
from photomatagent.scientific.evidence import Evidence
from photomatagent.scientific.state import ScientificState


def test_context_builder_injects_scientific_state():
    conversation = ConversationState()
    conversation.add(Message(role="user", content="investigate GaAs"))
    scientific = ScientificState(goal="investigate GaAs")
    scientific.add_evidence(
        Evidence(
            type="calculation",
            source="mock",
            content="band gap 0.31 eV",
            confidence=0.5,
        )
    )
    context = ContextBuilder().build(conversation, scientific)
    assert context[0].role == "system"
    assert "Current scientific state" in context[0].content
    assert "band gap 0.31 eV" in context[0].content
    assert context[1].content == "investigate GaAs"


def test_context_builder_empty_state():
    context = ContextBuilder().build(ConversationState(), ScientificState())
    assert "Goal: (none yet)" in context[0].content
