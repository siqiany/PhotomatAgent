"""OpenAI Responses API adapter.

The adapter translates SDK stream events into canonical model events. It
does not execute tools and never owns an agent loop.
"""

from __future__ import annotations

import json
import hashlib
import re
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

_OPENAI_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class OpenAIToolNameCodec:
    """Map canonical tool names to OpenAI-safe names and back.

    PhotomatAgent allows namespaced names such as ``mock.run_calculation``.
    OpenAI-compatible APIs commonly only allow letters, numbers, ``_`` and ``-``.
    A hash suffix prevents collisions while keeping already-valid names unchanged.
    """

    def __init__(self, tools: Sequence[ToolDefinition]) -> None:
        canonical_names = {tool.name for tool in tools}
        self._encode: dict[str, str] = {}
        self._decode: dict[str, str] = {}
        reserved = {
            name for name in canonical_names if _OPENAI_TOOL_NAME_PATTERN.fullmatch(name)
        }
        for name in sorted(canonical_names):
            encoded = name if name in reserved else self._safe_name(name, reserved)
            self._encode[name] = encoded
            self._decode[encoded] = name
            reserved.add(encoded)

    @staticmethod
    def _safe_name(name: str, reserved: set[str]) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", name) or "tool"
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
        for digest_length in range(8, len(digest) + 1):
            suffix = f"__{digest[:digest_length]}"
            candidate = f"{cleaned[: 64 - len(suffix)]}{suffix}"
            if candidate not in reserved:
                return candidate
        raise ValueError(f"could not encode OpenAI tool name: {name}")

    def encode(self, canonical_name: str) -> str:
        return self._encode.get(canonical_name, canonical_name)

    def decode(self, provider_name: str) -> str:
        return self._decode.get(provider_name, provider_name)


def openai_input(
    messages: Sequence[ModelMessage], tool_names: OpenAIToolNameCodec | None = None
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            items.append({"role": "system", "content": message.content})
        elif isinstance(message, UserMessage):
            items.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            if message.text:
                items.append({"role": "assistant", "content": message.text})
            for call in message.tool_calls:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call.id,
                        "name": tool_names.encode(call.name) if tool_names else call.name,
                        "arguments": json.dumps(call.arguments),
                    }
                )
        elif isinstance(message, ToolResultMessage):
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.content,
                }
            )
    return items


def openai_tools(
    tools: list[ToolDefinition], tool_names: OpenAIToolNameCodec | None = None
) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": tool_names.encode(tool.name) if tool_names else tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
            "strict": False,
        }
        for tool in tools
    ]


@dataclass
class _CallBuilder:
    item_id: str
    call_id: str
    name: str
    index: int
    arguments_text: str = ""


class OpenAIStreamMapper:
    """Stateful pure mapper; tests feed it mocked SDK event dictionaries."""

    def __init__(
        self, model: str, tool_names: OpenAIToolNameCodec | None = None
    ) -> None:
        self.model = model
        self.tool_names = tool_names
        self.response_id: str | None = None
        self.text_parts: list[str] = []
        self.calls_by_item: dict[str, _CallBuilder] = {}
        self.completed_calls: dict[str, ToolCall] = {}
        self.usage = ModelUsage()
        self.started = False

    def feed(self, raw: Any) -> list[ModelStreamEvent]:
        event = as_dict(raw)
        event_type = str(event.get("type", ""))
        mapped: list[ModelStreamEvent] = []

        if event_type == "response.created":
            response = as_dict(event.get("response", {}))
            self.response_id = response.get("id")
            self.started = True
            mapped.append(
                ModelStreamStarted(
                    provider="openai", model=self.model, response_id=self.response_id
                )
            )
        elif event_type == "response.output_text.delta":
            delta = str(event.get("delta", ""))
            self.text_parts.append(delta)
            mapped.append(ModelTextDelta(text=delta))
        elif event_type == "response.output_item.added":
            item = as_dict(event.get("item", {}))
            if item.get("type") == "function_call":
                item_id = str(item.get("id") or event.get("item_id") or "")
                call_id = str(item.get("call_id") or item_id)
                builder = _CallBuilder(
                    item_id=item_id,
                    call_id=call_id,
                    name=self._decode_tool_name(str(item.get("name", ""))),
                    index=int(event.get("output_index", 0)),
                    arguments_text=str(item.get("arguments") or ""),
                )
                self.calls_by_item[item_id] = builder
                mapped.append(
                    ModelToolCallStarted(
                        tool_call_id=call_id,
                        tool_name=builder.name,
                        index=builder.index,
                    )
                )
        elif event_type == "response.function_call_arguments.delta":
            item_id = str(event.get("item_id", ""))
            delta_builder = self.calls_by_item.get(item_id)
            if delta_builder is None:
                raise ProviderError("openai", f"arguments delta for unknown item {item_id}")
            delta = str(event.get("delta", ""))
            delta_builder.arguments_text += delta
            mapped.append(
                ModelToolCallArgumentsDelta(
                    tool_call_id=delta_builder.call_id,
                    delta=delta,
                    index=delta_builder.index,
                )
            )
        elif event_type == "response.function_call_arguments.done":
            item_id = str(event.get("item_id", ""))
            done_builder = self.calls_by_item.get(item_id)
            if done_builder is None:
                raise ProviderError("openai", f"arguments completed for unknown item {item_id}")
            done_name = str(event.get("name") or done_builder.name)
            done_builder.name = self._decode_tool_name(done_name)
            final_arguments = str(event.get("arguments") or done_builder.arguments_text)
            call = ToolCall(
                id=done_builder.call_id,
                name=done_builder.name,
                arguments=self._parse_arguments(final_arguments, done_builder.call_id),
            )
            self.completed_calls[call.id] = call
            mapped.append(ModelToolCallCompleted(tool_call=call, index=done_builder.index))
        elif event_type == "response.completed":
            response = as_dict(event.get("response", {}))
            self.response_id = str(response.get("id") or self.response_id or "") or None
            self._recover_final_calls(response, mapped)
            self.usage = self._usage(response.get("usage"))
            mapped.append(ModelUsageUpdated(usage=self.usage))
            calls = sorted(
                self.completed_calls.values(),
                key=lambda call: next(
                    (b.index for b in self.calls_by_item.values() if b.call_id == call.id), 0
                ),
            )
            finish: FinishReason = "tool_calls" if calls else "stop"
            if response.get("status") == "incomplete":
                finish = "max_tokens"
            mapped.append(
                ModelCompleted(
                    response=ModelResponse(
                        text="".join(self.text_parts),
                        tool_calls=calls,
                        finish_reason=finish,
                        usage=self.usage,
                        response_id=self.response_id,
                    )
                )
            )
        elif event_type in {"response.failed", "error"}:
            error = event.get("error") or event.get("response") or "stream failed"
            raise ProviderError("openai", str(error))
        return mapped

    def _recover_final_calls(
        self, response: dict[str, Any], mapped: list[ModelStreamEvent]
    ) -> None:
        for index, raw_item in enumerate(response.get("output") or []):
            item = as_dict(raw_item)
            if item.get("type") != "function_call":
                continue
            call_id = str(item.get("call_id") or item.get("id") or "")
            if call_id in self.completed_calls:
                continue
            call = ToolCall(
                id=call_id,
                name=self._decode_tool_name(str(item.get("name", ""))),
                arguments=self._parse_arguments(str(item.get("arguments") or "{}"), call_id),
            )
            self.completed_calls[call.id] = call
            mapped.append(ModelToolCallCompleted(tool_call=call, index=index))

    def _decode_tool_name(self, name: str) -> str:
        return self.tool_names.decode(name) if self.tool_names else name

    @staticmethod
    def _parse_arguments(raw: str, call_id: str) -> dict[str, object]:
        try:
            value = json.loads(raw or "{}")
        except json.JSONDecodeError as exc:
            raise ProviderError("openai", f"invalid JSON arguments for {call_id}: {exc}") from exc
        if not isinstance(value, dict):
            raise ProviderError("openai", f"tool arguments for {call_id} are not an object")
        return value

    @staticmethod
    def _usage(raw: Any) -> ModelUsage:
        if not raw:
            return ModelUsage()
        usage = as_dict(raw)
        return ModelUsage(
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            total_tokens=(
                int(usage["total_tokens"])
                if usage.get("total_tokens") is not None
                else None
            ),
        )


class OpenAIProvider:
    provider = "openai"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any = None,
    ) -> None:
        self.model = model
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._client = client

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        tool_names = OpenAIToolNameCodec(request.tools)
        mapper = OpenAIStreamMapper(self.model, tool_names)
        system_messages = [
            message.content for message in request.messages if isinstance(message, SystemMessage)
        ]
        conversation = [
            message for message in request.messages if not isinstance(message, SystemMessage)
        ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            # Stateless continuation: send the complete conversation each round.
            # function_call and function_call_output items travel together with
            # matching call_ids, which works on official Responses API and on
            # OpenAI-compatible endpoints that do not support previous_response_id.
            "input": openai_input(conversation, tool_names),
            "tools": openai_tools(request.tools, tool_names),
            "stream": True,
        }
        if system_messages:
            kwargs["instructions"] = "\n\n".join(system_messages)
        try:
            stream = await self._client.responses.create(**kwargs)
            async for sdk_event in stream:
                for event in mapper.feed(sdk_event):
                    yield event
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(self.provider, safe_provider_message(exc)) from exc
