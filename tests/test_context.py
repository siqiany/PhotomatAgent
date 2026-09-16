from __future__ import annotations

from photomatagent.models.types import SystemMessage, UserMessage
from photomatagent.runtime.context import ContextBuilder
from photomatagent.runtime.state import ConversationState
from photomatagent.scientific.evidence import Evidence
from photomatagent.scientific.discovery.models import HypothesisOrigin, HypothesisProposal
from photomatagent.scientific.discovery.registration import build_hypothesis
from photomatagent.scientific.state import ScientificState


def _material_hypothesis(index: int):
    return build_hypothesis(
        HypothesisProposal(
            request_id=f"context-{index}",
            formula=f"Na{index + 1}BiS2",
            statement=f"structured statement {index}",
            design_operation="isovalent_substitution",
            basis=[
                {
                    "evidence_id": f"evidence-{index}",
                    "relation": "supports",
                    "anchor": f"FULL BASIS TEXT {index} " + "private detail " * 30,
                }
            ],
            validation_questions=[f"validation gap {index}"],
        ),
        HypothesisOrigin(
            tool_name="generation.register_hypothesis",
            tool_call_id=f"call-{index}",
            session_id="session",
            run_id="run",
            provider="fake",
            model="fake",
        ),
    )


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


def test_context_shows_only_three_recent_structured_hypothesis_summaries():
    scientific = ScientificState(
        material_hypotheses=[_material_hypothesis(index) for index in range(4)]
    )
    complete_before = scientific.model_dump(mode="json")

    latest_state = ContextBuilder().build(ConversationState(), scientific)[-1].content

    assert scientific.model_dump(mode="json") == complete_before
    assert "context-0" not in latest_state
    for index in range(1, 4):
        record = scientific.material_hypotheses[index]
        assert record.id in latest_state
        assert record.proposal.design_operation in latest_state
        assert f"validation gap {index}" in latest_state
    assert "FULL BASIS TEXT" not in latest_state
