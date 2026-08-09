"""Anthropic Messages API adapter with canonical streaming events."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from photomatagent.errors import ProviderError
from photomatagent.models.types import (
    AssistantMessage,
    FinishReason,
    ModelCompleted,
    ModelMessage,
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
    SystemMessage,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
    UserMessage,
)
from photomatagent.models.utils import as_dict, safe_provider_message


def anthropic_input(
    messages: Sequence[ModelMessage],
) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []

    def append(role: str, blocks: list[dict[str, Any]]) -> None:
        if converted and converted[-1]["role"] == role:
            converted[-1]["content"].extend(blocks)
        else:
            converted.append({"role": role, "content": blocks})

    for message in messages:
        if isinstance(message, SystemMessage):
            system_parts.append(message.content)
        elif isinstance(message, UserMessage):
            append("user", [{"type": "text", "text": message.content}])
        elif isinstance(message, AssistantMessage):
            blocks: list[dict[str, Any]] = []
            if message.text:
                blocks.append({"type": "text", "text": message.text})
            blocks.extend(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                }
                for call in message.tool_calls
            )
            append("assistant", blocks)
        elif isinstance(message, ToolResultMessage):
            append(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": message.content,
                        "is_error": message.is_error,
                    }
                ],
            )
    return "\n\n".join(system_parts), converted


def anthropic_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        for tool in tools
    ]


@dataclass
class _ToolBuilder:
    tool_call_id: str
    name: str
    index: int
    arguments_text: str = ""


class AnthropicStreamMapper:
    def __init__(self, model: str) -> None:
        self.model = model
        self.response_id: str | None = None
        self.text_parts: list[str] = []
        self.builders: dict[int, _ToolBuilder] = {}
        self.completed_calls: dict[str, ToolCall] = {}
        self.usage = ModelUsage()
        self.stop_reason: str | None = None

    def feed(self, raw: Any) -> list[ModelStreamEvent]:
        event = as_dict(raw)
        event_type = str(event.get("type", ""))
        mapped: list[ModelStreamEvent] = []

        if event_type == "message_start":
            message = as_dict(event.get("message", {}))
            self.response_id = message.get("id")
            self._merge_usage(message.get("usage"))
            mapped.extend(
                [
                    ModelStreamStarted(
                        provider="anthropic",
                        model=self.model,
                        response_id=self.response_id,
                    ),
                    ModelUsageUpdated(usage=self.usage.model_copy()),
                ]
            )
        elif event_type == "content_block_start":
            index = int(event.get("index", 0))
            block = as_dict(event.get("content_block", {}))
            if block.get("type") == "tool_use":
                builder = _ToolBuilder(
                    tool_call_id=str(block.get("id", "")),
                    name=str(block.get("name", "")),
                    index=index,
                    arguments_text=json.dumps(block.get("input")) if block.get("input") else "",
                )
                self.builders[index] = builder
                mapped.append(
                    ModelToolCallStarted(
                        tool_call_id=builder.tool_call_id,
                        tool_name=builder.name,
                        index=index,
                    )
                )
            elif block.get("type") == "text" and block.get("text"):
                text = str(block["text"])
                self.text_parts.append(text)
                mapped.append(ModelTextDelta(text=text))
        elif event_type == "content_block_delta":
            index = int(event.get("index", 0))
            delta = as_dict(event.get("delta", {}))
            if delta.get("type") == "text_delta":
                text = str(delta.get("text", ""))
                self.text_parts.append(text)
                mapped.append(ModelTextDelta(text=text))
            elif delta.get("type") == "input_json_delta":
                delta_builder = self.builders.get(index)
                if delta_builder is None:
                    raise ProviderError("anthropic", f"input delta for unknown block {index}")
                partial = str(delta.get("partial_json", ""))
                delta_builder.arguments_text += partial
                mapped.append(
                    ModelToolCallArgumentsDelta(
                        tool_call_id=delta_builder.tool_call_id,
                        delta=partial,
                        index=index,
                    )
                )
        elif event_type == "content_block_stop":
            index = int(event.get("index", 0))
            stopped_builder = self.builders.get(index)
            if (
                stopped_builder is not None
                and stopped_builder.tool_call_id not in self.completed_calls
            ):
                call = ToolCall(
                    id=stopped_builder.tool_call_id,
                    name=stopped_builder.name,
                    arguments=self._parse_arguments(
                        stopped_builder.arguments_text, stopped_builder.tool_call_id
                    ),
                )
                self.completed_calls[call.id] = call
                mapped.append(ModelToolCallCompleted(tool_call=call, index=index))
        elif event_type == "message_delta":
            delta = as_dict(event.get("delta", {}))
            self.stop_reason = delta.get("stop_reason") or self.stop_reason
            self._merge_usage(event.get("usage"))
            mapped.append(ModelUsageUpdated(usage=self.usage.model_copy()))
        elif event_type == "message_stop":
            calls = [
                self.completed_calls[builder.tool_call_id]
                for _, builder in sorted(self.builders.items())
                if builder.tool_call_id in self.completed_calls
            ]
            mapped.append(
                ModelCompleted(
                    response=ModelResponse(
                        text="".join(self.text_parts),
                        tool_calls=calls,
                        finish_reason=self._finish_reason(calls),
                        usage=self.usage,
                        response_id=self.response_id,
                    )
                )
            )
        elif event_type == "error":
            raise ProviderError("anthropic", str(event.get("error") or "stream failed"))
        return mapped

    def _merge_usage(self, raw: Any) -> None:
        if not raw:
            return
        usage = as_dict(raw)
        if usage.get("input_tokens") is not None:
            self.usage.input_tokens = int(usage["input_tokens"])
        if usage.get("output_tokens") is not None:
            self.usage.output_tokens = int(usage["output_tokens"])
        self.usage.total_tokens = self.usage.input_tokens + self.usage.output_tokens

    def _finish_reason(self, calls: list[ToolCall]) -> FinishReason:
        if calls or self.stop_reason == "tool_use":
            return "tool_calls"
        if self.stop_reason == "max_tokens":
            return "max_tokens"
        if self.stop_reason in {"end_turn", "stop_sequence", None}:
            return "stop"
        return "unknown"

    @staticmethod
    def _parse_arguments(raw: str, call_id: str) -> dict[str, object]:
        try:
            value = json.loads(raw or "{}")
        except json.JSONDecodeError as exc:
            raise ProviderError(
                "anthropic", f"invalid JSON arguments for {call_id}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ProviderError("anthropic", f"tool arguments for {call_id} are not an object")
        return value


class AnthropicProvider:
    provider = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        max_tokens: int = 4096,
        client: Any = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        if client is None:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(api_key=api_key)
        self._client = client

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        system, messages = anthropic_input(request.messages)
        mapper = AnthropicStreamMapper(self.model)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "tools": anthropic_tools(request.tools),
        }
        if system:
            kwargs["system"] = system
        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for sdk_event in stream:
                    for event in mapper.feed(sdk_event):
                        yield event
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(self.provider, safe_provider_message(exc)) from exc
