"""Deterministic history grouping and compaction-cut selection.

This module deliberately contains no model calls and no mutable global state.
The engine uses it to decide which durable prefix may be summarized while
keeping tool-call/result transactions atomic and preserving the recent suffix.
"""

from __future__ import annotations

from dataclasses import dataclass

from photomatagent.models.types import (
    AssistantMessage,
    ModelMessage,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)


class UnsafeContextHistory(ValueError):
    """The conversation cannot be grouped without guessing tool pairing."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class MessageSpan:
    start: int
    end: int
    protected: bool
    has_tools: bool


@dataclass(frozen=True)
class CompactionCut:
    prefix_end: int
    raw_prefix_end: int
    used_intra_turn_fallback: bool


def split_atomic_spans(messages: list[ModelMessage]) -> list[MessageSpan]:
    """Split messages into atomic, non-overlapping transaction groups.

    Ordinary user/system messages and assistant messages without tool calls are
    single-message groups.  An assistant message with tool calls owns the entire
    contiguous block of matching tool results; incomplete, duplicated, orphaned,
    or interleaved results raise :class:`UnsafeContextHistory` rather than being
    guessed at.
    """

    spans: list[MessageSpan] = []
    index = 0
    total = len(messages)
    while index < total:
        message = messages[index]
        if isinstance(message, (SystemMessage, UserMessage)):
            spans.append(
                MessageSpan(
                    start=index,
                    end=index + 1,
                    protected=False,
                    has_tools=False,
                )
            )
            index += 1
            continue
        if isinstance(message, AssistantMessage):
            if not message.tool_calls:
                spans.append(
                    MessageSpan(
                        start=index,
                        end=index + 1,
                        protected=False,
                        has_tools=False,
                    )
                )
                index += 1
                continue

            expected_ids: list[str] = []
            for call in message.tool_calls:
                if call.id in expected_ids:
                    raise UnsafeContextHistory(
                        f"duplicate tool call id in assistant group: {call.id}"
                    )
                expected_ids.append(call.id)
            expected = set(expected_ids)

            start = index
            index += 1
            results: dict[str, ToolResultMessage] = {}
            while index < total:
                candidate = messages[index]
                if not isinstance(candidate, ToolResultMessage):
                    if len(results) < len(expected):
                        raise UnsafeContextHistory(
                            "assistant tool call group was interrupted before all "
                            "results arrived"
                        )
                    break
                if candidate.tool_call_id not in expected:
                    raise UnsafeContextHistory(
                        f"orphaned tool result: {candidate.tool_call_id}"
                    )
                if candidate.tool_call_id in results:
                    raise UnsafeContextHistory(
                        f"duplicate tool result: {candidate.tool_call_id}"
                    )
                results[candidate.tool_call_id] = candidate
                index += 1
                if len(results) == len(expected):
                    break
            if len(results) != len(expected):
                missing = sorted(expected - set(results))
                raise UnsafeContextHistory(
                    "assistant tool call group is missing results for: "
                    + ", ".join(missing)
                )
            spans.append(
                MessageSpan(
                    start=start,
                    end=index,
                    protected=any(result.protected for result in results.values()),
                    has_tools=True,
                )
            )
            continue
        if isinstance(message, ToolResultMessage):
            raise UnsafeContextHistory(
                f"orphaned tool result without preceding assistant call: "
                f"{message.tool_call_id}"
            )
        raise UnsafeContextHistory(
            f"unsupported message type in history: {type(message).__name__}"
        )

    # Defensive invariant: spans must fully and exactly cover the input.
    cursor = 0
    for span in spans:
        if span.start != cursor or span.end <= span.start:
            raise AssertionError("atomic span coverage invariant violated")
        cursor = span.end
    if cursor != total:
        raise AssertionError("atomic span coverage invariant violated")
    return spans


def user_anchors(
    durable: list[ModelMessage], cursor: int, *, turns: int
) -> list[UserMessage]:
    """Return the most recent real user messages before ``cursor``.

    The returned messages are deep copies.  Callers may render them as anchors
    without mutating durable history.  Synthetic trailing state messages are not
    present in the durable conversation and therefore cannot be selected here.
    """

    if cursor <= 0 or turns <= 0:
        return []
    users: list[tuple[int, UserMessage]] = [
        (index, message)
        for index, message in enumerate(durable)
        if isinstance(message, UserMessage)
    ]
    if not users:
        return []
    return [
        message.model_copy(deep=True)
        for index, message in users[-turns:]
        if index < cursor
    ]


def select_compaction_cut(
    active: list[ModelMessage],
    raw_indices: list[int],
    raw_length: int,
    *,
    protect_recent_turns: int,
    recent_transaction_count: int,
    prefer_intra_turn: bool,
) -> CompactionCut | None:
    """Select the largest safe summarizable prefix for one strategy.

    ``prefer_intra_turn=False`` implements the normal user-turn boundary;
    ``prefer_intra_turn=True`` implements the single-long-task fallback that
    keeps the most recent complete transaction groups.  The function returns
    ``None`` when the chosen strategy cannot produce a safe, eligible prefix.
    """

    if len(active) != len(raw_indices):
        raise ValueError("active and raw_indices must have the same length")
    if not active:
        return None

    spans = split_atomic_spans(active)
    if not spans:
        return None

    if prefer_intra_turn:
        candidate = _intra_turn_candidate(
            spans, recent_transaction_count=recent_transaction_count
        )
    else:
        candidate = _recent_user_turn_candidate(
            active, protect_recent_turns=protect_recent_turns
        )

    if candidate is None or candidate <= 0:
        return None

    # A candidate split point may not cross a protected transaction.  If a
    # protected group begins before the candidate, retreat to its start so the
    # protected text remains in the untouched suffix.  Take the earliest such
    # barrier to guarantee we never split any protected group.
    for span in spans:
        if span.protected and span.start < candidate:
            candidate = span.start
            break

    if candidate <= 0:
        return None
    if candidate not in {span.start for span in spans}:
        raise AssertionError("compaction cut is not on an atomic span boundary")

    prefix = active[:candidate]
    if not any(
        isinstance(message, (AssistantMessage, ToolResultMessage))
        for message in prefix
    ):
        return None

    raw_prefix_end = (
        raw_indices[candidate] if candidate < len(raw_indices) else raw_length
    )
    if raw_prefix_end <= 0:
        return None
    return CompactionCut(
        prefix_end=candidate,
        raw_prefix_end=raw_prefix_end,
        used_intra_turn_fallback=prefer_intra_turn,
    )


def _recent_user_turn_candidate(
    active: list[ModelMessage], *, protect_recent_turns: int
) -> int | None:
    user_indices = [
        index for index, message in enumerate(active) if isinstance(message, UserMessage)
    ]
    if len(user_indices) < protect_recent_turns:
        return 0
    return user_indices[-protect_recent_turns]


def _intra_turn_candidate(
    spans: list[MessageSpan], *, recent_transaction_count: int
) -> int | None:
    tool_spans = [span for span in spans if span.has_tools]
    if tool_spans:
        keep = tool_spans[-recent_transaction_count:]
        return keep[0].start
    keep = spans[-recent_transaction_count:]
    if not keep:
        return None
    return keep[0].start


__all__ = [
    "CompactionCut",
    "MessageSpan",
    "UnsafeContextHistory",
    "select_compaction_cut",
    "split_atomic_spans",
    "user_anchors",
]
