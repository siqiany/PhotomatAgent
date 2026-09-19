from __future__ import annotations

import pytest

from photomatagent.models.types import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from photomatagent.runtime.context_segments import (
    CompactionCut,
    UnsafeContextHistory,
    select_compaction_cut,
    split_atomic_spans,
    user_anchors,
)


def _tool_transaction(index: int, *, protected: bool = False) -> list:
    call = ToolCall(id=f"c{index}", name="read", arguments={"path": f"r{index}"})
    return [
        AssistantMessage(tool_calls=[call]),
        ToolResultMessage(
            tool_call_id=call.id,
            tool_name="read",
            content=f"result {index}",
            protected=protected,
        ),
    ]


def test_split_groups_multi_tool_call_as_one_atomic_span() -> None:
    first = ToolCall(id="a", name="read", arguments={"path": "a"})
    second = ToolCall(id="b", name="read", arguments={"path": "b"})
    messages = [
        UserMessage(content="goal"),
        AssistantMessage(tool_calls=[first, second]),
        ToolResultMessage(tool_call_id="a", tool_name="read", content="A"),
        ToolResultMessage(tool_call_id="b", tool_name="read", content="B", protected=True),
        AssistantMessage(text="done"),
    ]
    spans = split_atomic_spans(messages)
    assert [(span.start, span.end, span.has_tools, span.protected) for span in spans] == [
        (0, 1, False, False),
        (1, 4, True, True),
        (4, 5, False, False),
    ]


@pytest.mark.parametrize(
    "messages",
    [
        [ToolResultMessage(tool_call_id="orphan", tool_name="read", content="x")],
        [
            AssistantMessage(tool_calls=[ToolCall(id="a", name="read")]),
            ToolResultMessage(tool_call_id="a", tool_name="read", content="A"),
            ToolResultMessage(tool_call_id="a", tool_name="read", content="A2"),
        ],
        [
            AssistantMessage(tool_calls=[ToolCall(id="a", name="read")]),
            UserMessage(content="interrupt"),
            ToolResultMessage(tool_call_id="a", tool_name="read", content="A"),
        ],
    ],
)
def test_split_rejects_unsafe_tool_pairing(messages) -> None:
    with pytest.raises(UnsafeContextHistory):
        split_atomic_spans(messages)


def test_select_intra_turn_fallback_supports_single_user_goal() -> None:
    messages = [UserMessage(content="one long goal")]
    for index in range(5):
        messages.extend(_tool_transaction(index))
    raw_indices = list(range(len(messages)))

    cut = select_compaction_cut(
        messages,
        raw_indices,
        len(messages),
        protect_recent_turns=2,
        recent_transaction_count=2,
        prefer_intra_turn=True,
    )

    assert cut == CompactionCut(
        prefix_end=7,
        raw_prefix_end=7,
        used_intra_turn_fallback=True,
    )
    suffix = messages[cut.prefix_end :]
    assert [message.tool_call_id for message in suffix if isinstance(message, ToolResultMessage)] == [
        "c3",
        "c4",
    ]
    assert user_anchors(messages, cut.raw_prefix_end, turns=2) == [
        UserMessage(content="one long goal")
    ]


def test_select_turn_boundary_is_not_used_for_single_user_goal() -> None:
    messages = [UserMessage(content="one long goal")]
    for index in range(3):
        messages.extend(_tool_transaction(index))
    raw_indices = list(range(len(messages)))
    assert (
        select_compaction_cut(
            messages,
            raw_indices,
            len(messages),
            protect_recent_turns=2,
            recent_transaction_count=2,
            prefer_intra_turn=False,
        )
        is None
    )


def test_select_retreats_from_protected_transaction() -> None:
    messages = [UserMessage(content="goal"), AssistantMessage(text="old")]
    messages.extend(_tool_transaction(1, protected=True))
    messages.extend(_tool_transaction(2))
    messages.extend(_tool_transaction(3))
    raw_indices = list(range(len(messages)))

    cut = select_compaction_cut(
        messages,
        raw_indices,
        len(messages),
        protect_recent_turns=1,
        recent_transaction_count=1,
        prefer_intra_turn=True,
    )

    # The protected group starts at index 2 and must remain entirely in suffix.
    assert cut is not None
    assert cut.prefix_end == 2
    assert messages[cut.prefix_end].tool_calls[0].id == "c1"


def test_raw_prefix_end_uses_raw_indices_not_filtered_length() -> None:
    # Active excludes durable message 1 (an abandoned transaction), so raw
    # mappings must advance to 5 rather than len(active).
    active = [UserMessage(content="goal"), AssistantMessage(text="old")]
    active.extend(_tool_transaction(1))
    active.extend(_tool_transaction(2))
    raw_indices = [0, 1, 2, 3, 5, 6]
    cut = select_compaction_cut(
        active,
        raw_indices,
        8,
        protect_recent_turns=1,
        recent_transaction_count=1,
        prefer_intra_turn=True,
    )
    assert cut is not None
    assert cut.prefix_end == 4
    assert cut.raw_prefix_end == raw_indices[4] == 5


def test_user_anchors_are_bounded_and_before_cursor() -> None:
    durable = [
        UserMessage(content="first"),
        AssistantMessage(text="work"),
        UserMessage(content="second"),
        UserMessage(content="third"),
        AssistantMessage(text="current"),
    ]
    assert [item.content for item in user_anchors(durable, 4, turns=2)] == [
        "second",
        "third",
    ]
    assert user_anchors(durable, 1, turns=2) == []
    assert user_anchors(durable, 0, turns=2) == []
