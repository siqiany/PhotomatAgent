from __future__ import annotations

import asyncio
import json

import pytest

from photomatagent.models.types import (
    AssistantMessage,
    ModelCompleted,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ModelStreamStarted,
    ModelUsage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from photomatagent.runtime.context_engine import (
    CompactionState,
    ContextEngineConfig,
    ProviderContextSummarizer,
    SummaryError,
)


def _last_history(request: ModelRequest) -> str:
    message = request.messages[-1]
    assert isinstance(message, UserMessage)
    return message.content


class ScriptedProvider:
    provider = "fake"
    model = "fake"

    def __init__(self, responses):
        self.requests: list[ModelRequest] = []
        self._responses = list(responses)

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        item = self._responses.pop(0)
        yield ModelStreamStarted(provider=self.provider, model=self.model)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, str):
            response = ModelResponse(text=item, finish_reason="stop")
        else:
            response = item
        yield ModelCompleted(response=response)


class SlowProvider:
    provider = "fake"
    model = "fake"

    def __init__(self):
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        yield ModelStreamStarted(provider=self.provider, model=self.model)
        await asyncio.sleep(3600)
        yield ModelCompleted(response=ModelResponse(text="{}", finish_reason="stop"))


def _transaction(index: int, *, content_size: int = 700) -> list[ModelMessage]:
    call = ToolCall(id=f"c{index}", name="read", arguments={"path": f"r{index}"})
    return [
        AssistantMessage(tool_calls=[call]),
        ToolResultMessage(
            tool_call_id=call.id,
            tool_name="read",
            content=("x" * content_size) + f" result {index}",
        ),
    ]


def _config(**overrides) -> ContextEngineConfig:
    values = {
        "context_limit_tokens": 32_000,
        "response_reserve_tokens": 512,
        "safety_margin_tokens": 128,
        "compact_trigger_tokens": 12_000,
        "compact_target_tokens": 6_000,
        "summary_max_tokens": 256,
        "summary_chunk_tokens": 1_200,
        "summary_chunk_output_tokens": 256,
        "summary_max_calls": 4,
        "summary_timeout_seconds": 5.0,
    }
    values.update(overrides)
    return ContextEngineConfig(**values)


@pytest.mark.asyncio
async def test_provider_summarizer_uses_bounded_request_and_deduces_usage() -> None:
    state = CompactionState(goal="preserved", progress=["one"])
    provider = ScriptedProvider([state.model_dump_json()])
    summarizer = ProviderContextSummarizer(provider, _config())

    result = await summarizer.summarize([UserMessage(content="history")], None)

    assert result.goal == "preserved"
    request = provider.requests[0]
    assert request.tools == []
    assert request.max_output_tokens == 256
    assert len(request.messages) == 2
    assert summarizer.last_model_calls == 1
    assert summarizer.last_usage is not None


@pytest.mark.asyncio
async def test_provider_summarizer_chunks_history_and_passes_previous_state() -> None:
    messages = [UserMessage(content="goal")]
    for index in range(8):
        messages.extend(_transaction(index, content_size=700))
    first = CompactionState(goal="first chunk")
    second = CompactionState(goal="second chunk")
    provider = ScriptedProvider(
        [
            first.model_dump_json(),
            second.model_dump_json(),
            second.model_dump_json(),
            second.model_dump_json(),
        ]
    )
    summarizer = ProviderContextSummarizer(provider, _config(summary_chunk_tokens=1_500))

    result = await summarizer.summarize(messages, None)

    assert result.goal == "second chunk"
    assert summarizer.last_model_calls >= 2
    assert len(provider.requests) == summarizer.last_model_calls
    for request in provider.requests:
        assert request.tools == []
        assert request.max_output_tokens == 256
    assert "first chunk" in provider.requests[1].messages[-1].content


@pytest.mark.asyncio
async def test_provider_summarizer_records_usage_before_json_validation_failure() -> None:
    usage = ModelUsage(input_tokens=10, output_tokens=4)
    provider = ScriptedProvider(
        [ModelResponse(text="not json", finish_reason="stop", usage=usage)]
    )
    summarizer = ProviderContextSummarizer(provider, _config())

    with pytest.raises(SummaryError) as exc_info:
        await summarizer.summarize([UserMessage(content="history")], None)

    assert exc_info.value.reason == "invalid_summary"
    assert summarizer.last_model_calls == 1
    assert summarizer.last_usages == [usage]
    assert summarizer.last_usage is not None
    assert summarizer.last_usage.input_tokens == 10
    assert summarizer.last_usage.output_tokens == 4


@pytest.mark.asyncio
async def test_provider_summarizer_rejects_non_stop_or_tool_call_responses() -> None:
    provider = ScriptedProvider(
        [
            ModelResponse(text="{}", finish_reason="max_tokens"),
            ModelResponse(
                text="{}",
                finish_reason="stop",
                tool_calls=[ToolCall(name="read", arguments={})],
            ),
        ]
    )
    summarizer = ProviderContextSummarizer(provider, _config())
    for expected_reason in ("truncated_summary", "invalid_summary"):
        with pytest.raises(SummaryError) as exc_info:
            await summarizer.summarize([UserMessage(content="history")], None)
        assert exc_info.value.reason == expected_reason


@pytest.mark.asyncio
async def test_provider_summarizer_salvages_truncated_summary_prefix() -> None:
    """A summary cut off by the output cap keeps its complete fields.

    Regression: a reasoning-heavy model can emit reasoning tokens and then be
    truncated mid-JSON. Discarding that whole prefix forced compaction to fail
    with "expected exactly one completed summary response, got 0"/empty output
    and the session could never compact again.
    """
    text = (
        '{"goal":"keep the MWIR design on track","standing_instructions":["no '
        'unverified claims"],"progress":["relax submitted for vasp_044aefb"],'
        '"key_findings":["Eg ~0.25 eV predicted"],"decisions":["wait for the '
        'relax job before static"],"next_actions":["collect and vali'
    )
    provider = ScriptedProvider(
        [ModelResponse(text=text, finish_reason="max_tokens")]
    )
    summarizer = ProviderContextSummarizer(provider, _config())

    state = await summarizer.summarize([UserMessage(content="history")], None)

    assert state.goal == "keep the MWIR design on track"
    assert state.progress == ["relax submitted for vasp_044aefb"]
    assert state.key_findings == ["Eg ~0.25 eV predicted"]
    # The unfinished final entry is dropped, never invented.
    assert state.next_actions == []


@pytest.mark.asyncio
async def test_provider_summarizer_shrinks_oversized_atomic_group() -> None:
    """A giant tool result is truncated inside its group, not abandoned.

    Regression: raising ``summary_unit_too_large`` here made an oversized
    tool output a permanent, unrecoverable compaction failure.
    """
    call = ToolCall(id="big", name="read", arguments={"path": "big"})
    messages = [
        AssistantMessage(tool_calls=[call]),
        ToolResultMessage(
            tool_call_id="big", tool_name="read", content="y" * 60_000
        ),
    ]
    provider = ScriptedProvider([CompactionState(goal="shrunk").model_dump_json()])
    summarizer = ProviderContextSummarizer(provider, _config(summary_chunk_tokens=200))

    result = await summarizer.summarize(messages, None)

    assert result.goal == "shrunk"
    assert summarizer.last_model_calls == 1
    request = provider.requests[0]
    history = _last_history(request)
    assert len(history) < 60_000
    assert "truncated for compaction summary" in history
    # The call/result transaction is preserved: the assistant tool call and
    # its matching tool_call_id both survive in the request.
    assert "big" in history


@pytest.mark.asyncio
async def test_provider_summarizer_retries_truncated_chunk_with_less_history() -> None:
    """Repeated truncation shrinks the request instead of giving up at once."""
    call = ToolCall(id="c", name="read", arguments={"path": "r"})
    messages = [
        AssistantMessage(tool_calls=[call]),
        ToolResultMessage(tool_call_id="c", tool_name="read", content="z" * 40_000),
    ]
    truncated = ModelResponse(text="{\"goal\":\"partial", finish_reason="max_tokens")
    provider = ScriptedProvider(
        [
            truncated,
            truncated,
            CompactionState(goal="recovered").model_dump_json(),
        ]
    )
    summarizer = ProviderContextSummarizer(provider, _config(summary_chunk_tokens=200, summary_max_calls=4))

    result = await summarizer.summarize(messages, None)

    assert result.goal == "recovered"
    assert summarizer.last_model_calls == 3
    sizes = [len(_last_history(request)) for request in provider.requests]
    assert sizes[1] < sizes[0]
    assert sizes[2] < sizes[1]


@pytest.mark.asyncio
async def test_provider_summarizer_timeout_is_bounded() -> None:
    provider = SlowProvider()
    summarizer = ProviderContextSummarizer(
        provider, _config(summary_timeout_seconds=0.01)
    )
    with pytest.raises(SummaryError) as exc_info:
        await summarizer.summarize([UserMessage(content="history")], None)
    assert exc_info.value.reason == "summary_timeout"
    assert summarizer.last_model_calls == 1
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_provider_summarizer_rejects_oversized_rendered_summary() -> None:
    huge = CompactionState(key_findings=["x" * 4_000])
    provider = ScriptedProvider([huge.model_dump_json()])
    summarizer = ProviderContextSummarizer(provider, _config(summary_max_tokens=64))
    with pytest.raises(SummaryError) as exc_info:
        await summarizer.summarize([UserMessage(content="history")], None)
    assert exc_info.value.reason == "summary_too_large"


@pytest.mark.asyncio
async def test_provider_summarizer_never_exceeds_max_call_limit() -> None:
    messages = [UserMessage(content="goal")]
    for index in range(8):
        messages.extend(_transaction(index, content_size=700))
    provider = ScriptedProvider(
        [CompactionState(goal="first").model_dump_json()]
    )
    summarizer = ProviderContextSummarizer(
        provider,
        _config(summary_chunk_tokens=1_500, summary_max_calls=1),
    )
    with pytest.raises(SummaryError) as exc_info:
        await summarizer.summarize(messages, None)
    assert exc_info.value.reason == "summary_call_limit"
    assert summarizer.last_model_calls == 1
    assert len(provider.requests) == 1
