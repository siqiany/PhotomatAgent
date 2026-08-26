"""Bounded working-context lifecycle over an immutable durable conversation."""

from __future__ import annotations

import json
import time
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from photomatagent.models.base import ModelProvider
from photomatagent.models.types import (
    AssistantMessage,
    ModelCompleted,
    ModelMessage,
    ModelRequest,
    ModelUsage,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from photomatagent.redaction import redact_secrets, redact_text
from photomatagent.runtime.context import ContextBuilder
from photomatagent.runtime.events import (
    ContextCompactionCompleted,
    ContextCompactionFailed,
    ContextCompactionStarted,
    ContextPruneCompleted,
    ContextPruneStarted,
    RuntimeEvent,
)
from photomatagent.runtime.ledger import derive_working_ledger, format_working_ledger
from photomatagent.runtime.state import ConversationState
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.surface import ToolSurfaceStats, estimate_tokens


class ContextEngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context_limit_tokens: int = Field(default=128_000, ge=1_024)
    prune_trigger_ratio: float = Field(default=0.70, gt=0, lt=1)
    compact_trigger_ratio: float = Field(default=0.82, gt=0, lt=1)
    target_ratio: float = Field(default=0.60, gt=0, lt=1)
    protect_recent_turns: int = Field(default=2, ge=1)
    ledger_max_chars: int = Field(default=1_200, ge=128)

    @model_validator(mode="after")
    def validate_ratios(self) -> "ContextEngineConfig":
        if not self.target_ratio < self.prune_trigger_ratio <= self.compact_trigger_ratio:
            raise ValueError(
                "expected target_ratio < prune_trigger_ratio <= compact_trigger_ratio"
            )
        return self


class RelevantResource(BaseModel):
    reference: str
    relevance: str = ""


class CompactionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = ""
    standing_instructions: list[str] = Field(default_factory=list)
    progress: list[str] = Field(default_factory=list)
    key_findings: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    failed_approaches: list[str] = Field(default_factory=list)
    relevant_resources: list[RelevantResource] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)


class ContextSize(BaseModel):
    chars: int
    tokens: int
    messages: int


class ContextBuildResult(BaseModel):
    messages: list[ModelMessage]
    events: list[RuntimeEvent] = Field(default_factory=list)
    size: ContextSize
    durable_size: ContextSize
    pruned_tool_results: int = 0
    compaction_count: int = 0
    inflight_tool_transaction: bool = False
    compaction_usage: ModelUsage | None = None


class ContextSummarizer(Protocol):
    async def summarize(
        self, messages: list[ModelMessage], previous: CompactionState | None
    ) -> CompactionState: ...


class ProviderContextSummarizer:
    """One ordinary, tool-free provider request returning CompactionState JSON."""

    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider
        self.last_usage: ModelUsage | None = None

    async def summarize(
        self, messages: list[ModelMessage], previous: CompactionState | None
    ) -> CompactionState:
        schema = json.dumps(CompactionState.model_json_schema(), separators=(",", ":"))
        history = json.dumps(
            [message.model_dump(mode="json") for message in messages],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        previous_text = previous.model_dump_json() if previous else "null"
        request = ModelRequest(
            messages=[
                SystemMessage(
                    content=(
                        "Compact agent history into strict JSON matching the supplied schema. "
                        "Preserve facts, failures, decisions, standing instructions, open work, "
                        "and provenance references. Do not invent facts. Return JSON only.\n"
                        f"Schema: {schema}"
                    )
                ),
                UserMessage(
                    content=f"Previous compaction: {previous_text}\nHistory: {history}"
                ),
            ],
            tools=[],
        )
        text = ""
        self.last_usage = None
        async for event in self.provider.stream(request):
            if isinstance(event, ModelCompleted):
                text = event.response.text
                self.last_usage = event.response.usage
        if not text:
            raise ValueError("compaction provider returned no completed text")
        return CompactionState.model_validate_json(_extract_json(text))


class ContextEngine:
    """Build bounded working messages without mutating the durable transcript."""

    def __init__(
        self,
        *,
        config: ContextEngineConfig | None = None,
        summarizer: ContextSummarizer | None = None,
    ) -> None:
        self.config = config or ContextEngineConfig()
        self.summarizer = summarizer
        self._compaction_state: CompactionState | None = None
        self._compacted_message_count = 0
        self._compaction_count = 0

    @property
    def compaction_state(self) -> CompactionState | None:
        return self._compaction_state

    @property
    def compaction_count(self) -> int:
        return self._compaction_count

    def snapshot(self) -> dict[str, Any]:
        """Serialize the compaction cursor for session persistence."""
        return {
            "compaction_state": (
                self._compaction_state.model_dump(mode="json")
                if self._compaction_state is not None
                else None
            ),
            "compacted_message_count": self._compacted_message_count,
            "compaction_count": self._compaction_count,
        }

    def restore(self, **snapshot: Any) -> None:
        """Restore a previously saved compaction cursor (from ``snapshot()``)."""
        raw_state = snapshot.get("compaction_state")
        self._compaction_state = (
            CompactionState.model_validate(raw_state) if raw_state else None
        )
        self._compacted_message_count = int(
            snapshot.get("compacted_message_count") or 0
        )
        self._compaction_count = int(snapshot.get("compaction_count") or 0)

    async def build(
        self,
        *,
        conversation: ConversationState,
        scientific: ScientificState,
        context_builder: ContextBuilder,
        capability_manifest: str,
        surface: ToolSurfaceStats,
        session_id: str,
        force_compaction: bool = False,
    ) -> ContextBuildResult:
        durable = list(conversation.messages)
        ledger = derive_working_ledger(
            durable, max_chars=self.config.ledger_max_chars
        )
        active_raw = durable[self._compacted_message_count :]
        active, raw_indices = _working_copy(active_raw)
        context_messages = context_builder.build_messages(
            active,
            scientific,
            capability_manifest=capability_manifest,
            investigation_state=format_working_ledger(ledger),
            compaction_state=self._compaction_state,
        )
        durable_messages = context_builder.build_messages(
            durable,
            scientific,
            capability_manifest=capability_manifest,
            investigation_state=format_working_ledger(ledger),
        )
        before = _measure(context_messages, surface)
        durable_size = _measure(durable_messages, surface)
        events: list[RuntimeEvent] = []
        protected_start = _protected_start(active, self.config.protect_recent_turns)
        protected_raw_start = (
            raw_indices[protected_start] if protected_start < len(raw_indices) else len(active_raw)
        )
        protected_turns = _count_user_turns(active[protected_start:])
        inflight = has_inflight_tool_transaction(active)
        pruned = 0
        compaction_usage: ModelUsage | None = None

        if before.tokens >= self._threshold(self.config.prune_trigger_ratio):
            started = time.monotonic()
            events.append(
                ContextPruneStarted(
                    tokens_before=before.tokens,
                    chars_before=before.chars,
                    messages_before=before.messages,
                    protected_turns=protected_turns,
                )
            )
            context_messages, pruned = self._prune_tool_results(
                context_messages,
                active_offset=1 + (1 if self._compaction_state else 0),
                protected_start=protected_start,
                session_id=session_id,
                surface=surface,
            )
            after_prune = _measure(context_messages, surface)
            events.append(
                ContextPruneCompleted(
                    tokens_before=before.tokens,
                    tokens_after=after_prune.tokens,
                    chars_before=before.chars,
                    chars_after=after_prune.chars,
                    messages_before=before.messages,
                    messages_after=after_prune.messages,
                    tool_results_pruned=pruned,
                    protected_turns=protected_turns,
                    duration_ms=(time.monotonic() - started) * 1000,
                )
            )

        current = _measure(context_messages, surface)
        should_compact = force_compaction or (
            current.tokens >= self._threshold(self.config.compact_trigger_ratio)
        )
        if should_compact and not inflight and protected_start > 0 and self.summarizer:
            compaction_started = time.monotonic()
            events.append(
                ContextCompactionStarted(
                    tokens_before=current.tokens,
                    chars_before=current.chars,
                    messages_before=current.messages,
                    protected_turns=protected_turns,
                )
            )
            active_offset = 1 + (1 if self._compaction_state else 0)
            prefix = context_messages[
                active_offset : active_offset + protected_start
            ]
            try:
                raw_state = await self.summarizer.summarize(
                    prefix, self._compaction_state
                )
                state = CompactionState.model_validate(
                    redact_secrets(raw_state.model_dump())
                )
                usage = getattr(self.summarizer, "last_usage", None)
                compaction_usage = usage if isinstance(usage, ModelUsage) else None
                candidate_active = active[protected_start:]
                candidate = context_builder.build_messages(
                    candidate_active,
                    scientific,
                    capability_manifest=capability_manifest,
                    investigation_state=format_working_ledger(ledger),
                    compaction_state=state,
                )
                after = _measure(candidate, surface)
                self._compaction_state = state
                self._compacted_message_count += protected_raw_start
                self._compaction_count += 1
                context_messages = candidate
                events.append(
                    ContextCompactionCompleted(
                        tokens_before=current.tokens,
                        tokens_after=after.tokens,
                        chars_before=current.chars,
                        chars_after=after.chars,
                        messages_before=current.messages,
                        messages_after=after.messages,
                        protected_turns=protected_turns,
                        duration_ms=(time.monotonic() - compaction_started) * 1000,
                        usage=(
                            compaction_usage.model_dump()
                            if compaction_usage is not None
                            else {}
                        ),
                    )
                )
            except Exception as exc:
                events.append(
                    ContextCompactionFailed(
                        tokens_before=current.tokens,
                        chars_before=current.chars,
                        messages_before=current.messages,
                        protected_turns=protected_turns,
                        error=redact_text(f"{type(exc).__name__}: {exc}"),
                        duration_ms=(time.monotonic() - compaction_started) * 1000,
                    )
                )

        final_size = _measure(context_messages, surface)
        if final_size.tokens >= self.config.context_limit_tokens:
            raise RuntimeError(
                "working context exceeds configured context limit after safe pruning/compaction: "
                f"{final_size.tokens} >= {self.config.context_limit_tokens} estimated tokens"
            )
        return ContextBuildResult(
            messages=context_messages,
            events=events,
            size=final_size,
            durable_size=durable_size,
            pruned_tool_results=pruned,
            compaction_count=self._compaction_count,
            inflight_tool_transaction=inflight,
            compaction_usage=compaction_usage,
        )

    def _prune_tool_results(
        self,
        messages: list[ModelMessage],
        *,
        active_offset: int,
        protected_start: int,
        session_id: str,
        surface: ToolSurfaceStats,
    ) -> tuple[list[ModelMessage], int]:
        working = list(messages)
        calls = _tool_calls_by_id(working)
        candidates: list[tuple[int, ToolResultMessage]] = []
        protected_index = active_offset + protected_start
        for index, message in enumerate(working):
            if (
                index >= protected_index
                or not isinstance(message, ToolResultMessage)
                or message.protected
            ):
                continue
            candidates.append((index, message))
        candidates.sort(key=lambda item: len(item[1].content), reverse=True)
        pruned = 0
        target = self._threshold(self.config.target_ratio)
        for index, message in candidates:
            if _measure(working, surface).tokens <= target:
                break
            call = calls.get(message.tool_call_id)
            args = _arguments_summary(call.arguments if call else {})
            status = "failure" if message.is_error else "success"
            placeholder = (
                "[Previous tool output omitted from active context]\n"
                f"tool: {message.tool_name}\n"
                f"arguments: {args}\n"
                f"status: {status}\n"
                f"reference: session://{session_id}/tool-call/{message.tool_call_id}"
            )
            working[index] = message.model_copy(update={"content": placeholder})
            pruned += 1
        return working, pruned

    def _threshold(self, ratio: float) -> int:
        return max(1, int(self.config.context_limit_tokens * ratio))


def has_inflight_tool_transaction(messages: list[ModelMessage]) -> bool:
    requested = {
        call.id
        for message in messages
        if isinstance(message, AssistantMessage)
        for call in message.tool_calls
    }
    completed = {
        message.tool_call_id
        for message in messages
        if isinstance(message, ToolResultMessage)
    }
    return bool(requested - completed)


def format_compaction_state(state: CompactionState) -> str:
    return "Compaction Summary (structured):\n" + state.model_dump_json(indent=2)


def _protected_start(messages: list[ModelMessage], turns: int) -> int:
    user_indices = [
        index for index, message in enumerate(messages) if isinstance(message, UserMessage)
    ]
    return user_indices[-turns] if len(user_indices) >= turns else 0


def _count_user_turns(messages: list[ModelMessage]) -> int:
    return sum(isinstance(message, UserMessage) for message in messages)


def _tool_calls_by_id(messages: list[ModelMessage]) -> dict[str, ToolCall]:
    return {
        call.id: call
        for message in messages
        if isinstance(message, AssistantMessage)
        for call in message.tool_calls
    }


def _working_copy(messages: list[ModelMessage]) -> tuple[list[ModelMessage], list[int]]:
    """Hide abandoned call/result pairs from the provider without rewriting durable history."""
    last_user = max(
        (index for index, message in enumerate(messages) if isinstance(message, UserMessage)),
        default=-1,
    )
    abandoned: set[str] = set()
    results = {
        message.tool_call_id: message
        for message in messages
        if isinstance(message, ToolResultMessage)
    }
    for index, message in enumerate(messages):
        if index >= last_user or not isinstance(message, AssistantMessage):
            continue
        for call in message.tool_calls:
            result = results.get(call.id)
            if result is None or result.content.startswith("not executed because"):
                abandoned.add(call.id)

    working: list[ModelMessage] = []
    raw_indices: list[int] = []
    for index, message in enumerate(messages):
        if isinstance(message, AssistantMessage):
            calls = [call for call in message.tool_calls if call.id not in abandoned]
            if not calls and not message.text:
                continue
            message = message.model_copy(update={"tool_calls": calls})
        elif isinstance(message, ToolResultMessage) and message.tool_call_id in abandoned:
            continue
        working.append(message)
        raw_indices.append(index)
    return working, raw_indices


def _arguments_summary(arguments: dict[str, object], max_chars: int = 240) -> str:
    rendered = json.dumps(
        arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    rendered = redact_text(rendered)
    return rendered if len(rendered) <= max_chars else rendered[: max_chars - 1] + "…"


def _measure(messages: list[ModelMessage], surface: ToolSurfaceStats) -> ContextSize:
    chars = sum(
        len(json.dumps(message.model_dump(), ensure_ascii=False, separators=(",", ":")))
        for message in messages
    ) + surface.visible_schema_chars
    return ContextSize(chars=chars, tokens=estimate_tokens(chars), messages=len(messages))


def _extract_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1]
        stripped = stripped.rsplit("```", 1)[0].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("compaction response did not contain a JSON object")
    return stripped[start : end + 1]
