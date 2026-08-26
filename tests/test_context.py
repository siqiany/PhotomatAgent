from __future__ import annotations

from photomatagent.models.types import SystemMessage, UserMessage
from photomatagent.runtime.context import ContextBuilder
from photomatagent.runtime.state import ConversationState
from photomatagent.scientific.evidence import Evidence
from photomatagent.scientific.state import ScientificState


def test_context_builder_injects_scientific_state():
    conversation = ConversationState()
    conversation.add(UserMessage(content="investigate GaAs"))
    scientific = ScientificState(goal="investigate GaAs")
    scientific.add_evidence(
        Evidence(type="calculation", source="mock", content="band gap 0.31 eV", confidence=0.5)
    )
    context = ContextBuilder().build(conversation, scientific)
    assert isinstance(context[0], SystemMessage)
    # State must NOT live in the static system prompt (it would break prompt caching).
    assert "Current scientific state" not in context[0].content
    assert "band gap 0.31 eV" not in context[0].content
    assert isinstance(context[1], UserMessage)
    assert context[1].content == "investigate GaAs"
    # The latest state snapshot is appended as the final message.
    assert isinstance(context[-1], UserMessage)
    assert "Current scientific state" in context[-1].content
    assert "band gap 0.31 eV" in context[-1].content


def test_context_builder_empty_state():
    context = ContextBuilder().build(ConversationState(), ScientificState())
    assert isinstance(context[0], SystemMessage)
    assert "Goal: (none yet)" in context[-1].content


def test_state_updates_only_replace_trailing_line_for_cache_hits():
    conversation = ConversationState()
    conversation.add(UserMessage(content="investigate GaAs"))
    builder = ContextBuilder()
    scientific = ScientificState(goal="investigate GaAs")

    first = builder.build(conversation, scientific)
    scientific.add_evidence(
        Evidence(type="calculation", source="mock", content="band gap 0.31 eV", confidence=0.5)
    )
    second = builder.build(conversation, scientific)

    # System prompt and conversation prefix are byte-identical, so provider
    # prompt-cache prefixes survive state updates; only the tail changed.
    assert first[:-1] == second[:-1]
    assert first[0].content == second[0].content
    assert first[-1].content != second[-1].content
    assert "band gap 0.31 eV" in second[-1].content


def test_system_prompt_enforces_user_output_and_tmp_layout():
    context = ContextBuilder().build(ConversationState(), ScientificState())
    assert "user_output/" in context[0].content
    assert "tmp/" in context[0].content
