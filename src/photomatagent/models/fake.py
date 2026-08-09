"""Deterministic streaming provider for tests and offline development."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

from photomatagent.models.types import (
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ModelStreamStarted,
    ModelTextDelta,
    ModelToolCallArgumentsDelta,
    ModelToolCallCompleted,
    ModelToolCallStarted,
    ModelUsage,
    ModelUsageUpdated,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

DEFAULT_MATERIAL = "GaAs"


@dataclass
class FakeResponse:
    text: str = ""
    tool_calls: list[ToolCall] | None = None
    text_deltas: list[str] | None = None
    usage: ModelUsage | None = None

    def to_response(self) -> ModelResponse:
        calls = self.tool_calls or []
        return ModelResponse(
            text=self.text,
            tool_calls=calls,
            finish_reason="tool_calls" if calls else "stop",
            usage=self.usage or ModelUsage(),
        )


def scripted_tool_call(
    name: str, arguments: dict[str, object], *, tool_call_id: str | None = None
) -> FakeResponse:
    call = ToolCall(name=name, arguments=arguments)
    if tool_call_id is not None:
        call.id = tool_call_id
    return FakeResponse(tool_calls=[call])


def _extract_material(goal: str) -> str:
    match = re.search(r"material\s+([A-Za-z0-9_\-]+)", goal)
    return match.group(1) if match else DEFAULT_MATERIAL


class FakeModelProvider:
    provider = "fake"
    model = "fake"

    def __init__(self, responses: list[FakeResponse] | None = None, *, auto: bool = False) -> None:
        self._responses = list(responses or [])
        self.auto = auto
        self.requests: list[ModelRequest] = []

    def set_responses(self, responses: list[FakeResponse]) -> None:
        self._responses = list(responses)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        scripted = self._responses.pop(0) if self._responses else None
        response = scripted.to_response() if scripted else self._auto_response(request)

        yield ModelStreamStarted(provider=self.provider, model=self.model)
        deltas = scripted.text_deltas if scripted and scripted.text_deltas is not None else None
        if deltas is None and response.text:
            deltas = [response.text]
        for delta in deltas or []:
            yield ModelTextDelta(text=delta)

        for index, call in enumerate(response.tool_calls):
            yield ModelToolCallStarted(
                tool_call_id=call.id, tool_name=call.name, index=index
            )
            arguments_json = json.dumps(call.arguments)
            if arguments_json:
                yield ModelToolCallArgumentsDelta(
                    tool_call_id=call.id, delta=arguments_json, index=index
                )
            yield ModelToolCallCompleted(tool_call=call, index=index)

        yield ModelUsageUpdated(usage=response.usage)
        yield ModelCompleted(response=response)

    def _auto_response(self, request: ModelRequest) -> ModelResponse:
        user_messages = [m for m in request.messages if isinstance(m, UserMessage)]
        goal = user_messages[-1].content if user_messages else ""
        material = _extract_material(goal)
        if any(isinstance(m, ToolResultMessage) for m in request.messages):
            return ModelResponse(
                text=(
                    f"The mock calculation for {material} suggests a direct band gap "
                    "of 0.31 eV. This is a placeholder result from the mock backend; "
                    "validate with a real calculation before drawing conclusions."
                ),
                finish_reason="stop",
            )
        return ModelResponse(
            tool_calls=[
                ToolCall(
                    name="mock.run_calculation",
                    arguments={"material": material, "calculation_type": "band_structure"},
                )
            ],
            finish_reason="tool_calls",
        )
