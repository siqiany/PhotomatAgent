"""The complete, visible PhotomatAgent control loop.

Read this file top-to-bottom to follow:
context -> provider stream -> tool calls -> permission -> execution -> results
-> next iteration -> stop. No SDK or UI owns any part of that sequence.
"""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from photomatagent.errors import ProviderError, ToolError, ToolValidationError
from photomatagent.models.base import ModelProvider
from photomatagent.models.types import (
    AssistantMessage,
    ModelCompleted,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelStreamStarted as ProviderStreamStarted,
    ModelTextDelta,
    ModelToolCallArgumentsDelta,
    ModelToolCallCompleted,
    ModelToolCallStarted,
    ModelUsageUpdated,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from photomatagent.runtime.budget import BudgetState
from photomatagent.runtime.context import ContextBuilder, format_scientific_state
from photomatagent.runtime.context_engine import (
    ContextEngine,
    ContextEngineConfig,
    ProviderContextSummarizer,
)
from photomatagent.runtime.context_budget import account_context
from photomatagent.runtime.events import (
    BudgetUpdated,
    LoopCompleted,
    LoopFailed,
    LoopIterationStarted,
    LoopStarted,
    ModelRequestStarted,
    ModelResponseCompleted,
    ModelStreamStarted,
    ProviderFailed,
    RuntimeEvent,
    ScientificStateUpdated,
    SensitiveAccessBlocked,
    TextDelta,
    ToolApprovalRequired,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallStarted,
    ToolCompleted,
    ToolFailed,
    ToolPermissionDenied,
    ToolRequested,
    ToolStarted,
)
from photomatagent.runtime.permissions import (
    ApprovalHandler,
    ApprovalRequest,
    PermissionDecision,
    PermissionPolicy,
    default_permission_policy,
)
from photomatagent.runtime.observation import ObservationPolicy
from photomatagent.runtime.sensitive import SensitiveAccessError, SensitivePathPolicy
from photomatagent.runtime.state import ConversationState
from photomatagent.runtime.stop_policy import StopPolicy
from photomatagent.scientific.calculations import CalculationRecord
from photomatagent.scientific.claims import ScientificClaim
from photomatagent.scientific.evidence import Evidence
from photomatagent.scientific.state import ScientificState
from photomatagent.scientific.tasks import ScientificTask
from photomatagent.tools.registry import ToolRegistry
from photomatagent.tools.exposure import ToolExposure
from photomatagent.tools.surface import (
    ToolSurfaceConfig,
    ToolSurfacePlanner,
    compact_parameter_help,
)
from photomatagent.workspace import Workspace

EventSink = Callable[[RuntimeEvent], Awaitable[None] | None]


class AgentRuntime:
    def __init__(
        self,
        *,
        model: ModelProvider,
        tools: ToolRegistry,
        workspace: Workspace | Path | str | None = None,
        scientific_state: ScientificState | None = None,
        context_builder: ContextBuilder | None = None,
        tool_surface_planner: ToolSurfacePlanner | None = None,
        tool_surface_config: ToolSurfaceConfig | None = None,
        observation_policy: ObservationPolicy | None = None,
        context_engine: ContextEngine | None = None,
        context_engine_config: ContextEngineConfig | None = None,
        sensitive_path_policy: SensitivePathPolicy | None = None,
        model_context_limit: int | None = None,
        permission_policy: PermissionPolicy | None = None,
        stop_policy: StopPolicy | None = None,
        budget: BudgetState | None = None,
        approval_handler: ApprovalHandler | None = None,
        event_sinks: list[EventSink] | None = None,
        session_id: str | None = None,
    ) -> None:
        self._model = model
        self._tools = tools
        self._workspace = (
            workspace
            if isinstance(workspace, Workspace)
            else Workspace(workspace or Path.cwd())
        )
        self._scientific = scientific_state or ScientificState()
        self._context_builder = context_builder or ContextBuilder()
        self._tool_surface = tool_surface_planner or ToolSurfacePlanner(
            tools, tool_surface_config
        )
        self._observation = observation_policy or ObservationPolicy()
        engine_config = context_engine_config or ContextEngineConfig()
        if model_context_limit is not None:
            engine_config = engine_config.model_copy(
                update={"context_limit_tokens": model_context_limit}
            )
        self._context_engine = context_engine or ContextEngine(
            config=engine_config,
            summarizer=ProviderContextSummarizer(model),
        )
        self._model_context_limit = self._context_engine.config.context_limit_tokens
        self._sensitive_paths = sensitive_path_policy or SensitivePathPolicy()
        self._permission = permission_policy or default_permission_policy()
        self._stop_policy = stop_policy or StopPolicy()
        self._budget = budget or BudgetState()
        self._approval_handler = approval_handler
        self._event_sinks = list(event_sinks or [])
        self._conversation = ConversationState()
        self._session_id = session_id or uuid4().hex
        self._run_id: str | None = None

    @property
    def scientific_state(self) -> ScientificState:
        return self._scientific

    @property
    def conversation_state(self) -> ConversationState:
        return self._conversation

    @property
    def budget(self) -> BudgetState:
        return self._budget

    @property
    def context_engine(self) -> ContextEngine:
        return self._context_engine

    async def run(self, goal: str) -> AsyncIterator[RuntimeEvent]:
        """Run one user turn while preserving conversation and scientific state."""
        self._close_pending_tool_calls()
        self._run_id = uuid4().hex
        run_started = time.monotonic()
        self._scientific.goal = goal
        self._conversation.add(UserMessage(content=goal))
        iteration = 0
        try:
            yield await self._emit(
                LoopStarted(
                    goal=goal,
                    provider=self._model.provider,
                    model=self._model.model,
                    workspace=str(self._workspace.root),
                )
            )
            while True:
                iteration += 1
                self._budget.record_iteration()
                yield await self._emit(LoopIterationStarted(iteration=iteration))

                surface = self._tool_surface.plan()
                context = await self._context_engine.build(
                    conversation=self._conversation,
                    scientific=self._scientific,
                    context_builder=self._context_builder,
                    capability_manifest=surface.manifest.text,
                    surface=surface.stats,
                    session_id=self._session_id,
                )
                for context_event in context.events:
                    yield await self._emit(context_event)
                if context.compaction_usage is not None:
                    self._budget.record_model_call(context.compaction_usage)
                messages = context.messages
                context_budget = account_context(
                    messages,
                    surface.stats,
                    model_context_limit=self._model_context_limit,
                )
                request = ModelRequest(messages=messages, tools=surface.definitions)
                yield await self._emit(
                    ModelRequestStarted(
                        iteration=iteration,
                        message_count=len(messages),
                        provider=self._model.provider,
                        model=self._model.model,
                        registered_tools=surface.stats.registered_tools,
                        direct_count=surface.stats.direct_tools,
                        deferred_count=surface.stats.deferred_tools,
                        hidden_count=surface.stats.hidden_tools,
                        visible_schema_chars=surface.stats.visible_schema_chars,
                        manifest_chars=surface.stats.manifest_chars,
                        estimated_schema_tokens=surface.stats.estimated_visible_schema_tokens,
                        estimated_direct_schema_tokens=(
                            surface.stats.estimated_direct_schema_tokens
                        ),
                        estimated_deferred_schema_tokens=(
                            surface.stats.estimated_deferred_schema_tokens
                        ),
                        estimated_manifest_tokens=surface.stats.estimated_manifest_tokens,
                        estimated_avoided_tokens=surface.stats.estimated_avoided_tokens,
                        estimated_bridge_schema_tokens=(
                            surface.stats.estimated_bridge_schema_tokens
                        ),
                        estimated_current_prompt_tokens=(
                            context_budget.estimated_current_prompt_tokens
                        ),
                        estimated_message_history_tokens=(
                            context_budget.estimated_message_history_tokens
                        ),
                        estimated_tool_result_tokens=(
                            context_budget.estimated_tool_result_tokens
                        ),
                        model_context_limit=context_budget.model_context_limit,
                        working_context_chars=context.size.chars,
                        durable_context_chars=context.durable_size.chars,
                        pruned_tool_results=context.pruned_tool_results,
                        compaction_count=context.compaction_count,
                    )
                )

                response: ModelResponse | None = None
                model_started = time.monotonic()
                try:
                    async for model_event in self._model.stream(request):
                        if isinstance(model_event, ProviderStreamStarted):
                            yield await self._emit(
                                ModelStreamStarted(
                                    iteration=iteration,
                                    provider=model_event.provider,
                                    model=model_event.model,
                                    response_id=model_event.response_id,
                                )
                            )
                        elif isinstance(model_event, ModelTextDelta):
                            yield await self._emit(
                                TextDelta(iteration=iteration, text=model_event.text)
                            )
                        elif isinstance(model_event, ModelToolCallStarted):
                            yield await self._emit(
                                ToolCallStarted(
                                    iteration=iteration,
                                    tool_call_id=model_event.tool_call_id,
                                    tool_name=model_event.tool_name,
                                    index=model_event.index,
                                )
                            )
                        elif isinstance(model_event, ModelToolCallArgumentsDelta):
                            yield await self._emit(
                                ToolCallArgumentsDelta(
                                    iteration=iteration,
                                    tool_call_id=model_event.tool_call_id,
                                    delta=model_event.delta,
                                    index=model_event.index,
                                )
                            )
                        elif isinstance(model_event, ModelToolCallCompleted):
                            call = model_event.tool_call
                            yield await self._emit(
                                ToolCallCompleted(
                                    iteration=iteration,
                                    tool_call_id=call.id,
                                    tool_name=call.name,
                                    arguments=call.arguments,
                                    index=model_event.index,
                                )
                            )
                        elif isinstance(model_event, ModelUsageUpdated):
                            # The final aggregate is booked once below to avoid
                            # double-counting incremental usage snapshots.
                            continue
                        elif isinstance(model_event, ModelCompleted):
                            response = model_event.response
                except ProviderError as exc:
                    yield await self._emit(
                        ProviderFailed(
                            iteration=iteration,
                            provider=self._model.provider,
                            model=self._model.model,
                            error=str(exc),
                            error_type=type(exc).__name__,
                            duration_ms=(time.monotonic() - model_started) * 1000,
                        )
                    )
                    raise

                if response is None:
                    no_completion_error = ProviderError(
                        self._model.provider, "stream ended without ModelCompleted"
                    )
                    yield await self._emit(
                        ProviderFailed(
                            iteration=iteration,
                            provider=self._model.provider,
                            model=self._model.model,
                            error=str(no_completion_error),
                            error_type=type(no_completion_error).__name__,
                            duration_ms=(time.monotonic() - model_started) * 1000,
                        )
                    )
                    raise no_completion_error

                self._budget.record_model_call(response.usage)
                yield await self._emit(
                    ModelResponseCompleted(
                        iteration=iteration,
                        provider=self._model.provider,
                        model=self._model.model,
                        response_id=response.response_id,
                        finish_reason=response.finish_reason,
                        tool_call_count=len(response.tool_calls),
                        usage=response.usage.model_dump(),
                        duration_ms=(time.monotonic() - model_started) * 1000,
                    )
                )
                self._conversation.add(
                    AssistantMessage(text=response.text, tool_calls=response.tool_calls)
                )

                decision = self._stop_policy.should_stop(
                    iteration=iteration, response=response, budget=self._budget
                )
                if decision.should_stop:
                    self._close_tool_calls(
                        response.tool_calls,
                        f"not executed because loop stopped: {decision.reason}",
                    )
                    yield await self._emit_budget(iteration)
                    yield await self._emit(
                        LoopCompleted(
                            iterations=iteration,
                            reason=decision.reason,
                            duration_ms=(time.monotonic() - run_started) * 1000,
                        )
                    )
                    return

                for tool_call in response.tool_calls:
                    async for event in self._handle_tool_call(tool_call, iteration):
                        yield event
                yield await self._emit_budget(iteration)
        except Exception as exc:
            yield await self._emit(
                LoopFailed(
                    error=f"{type(exc).__name__}: {exc}",
                    duration_ms=(time.monotonic() - run_started) * 1000,
                    error_type=type(exc).__name__,
                )
            )
            raise

    async def compact_working_context(self) -> list[RuntimeEvent]:
        """Developer hook used by the interactive ``/compact`` command."""
        surface = self._tool_surface.plan()
        result = await self._context_engine.build(
            conversation=self._conversation,
            scientific=self._scientific,
            context_builder=self._context_builder,
            capability_manifest=surface.manifest.text,
            surface=surface.stats,
            session_id=self._session_id,
            force_compaction=True,
        )
        emitted: list[RuntimeEvent] = []
        for event in result.events:
            emitted.append(await self._emit(event))
        return emitted

    def _close_pending_tool_calls(self) -> None:
        """Append factual terminal results for abandoned calls; never rewrite history."""
        messages = self._conversation.messages
        fulfilled = {
            message.tool_call_id
            for message in messages
            if isinstance(message, ToolResultMessage)
        }
        unfulfilled = {
            call.id
            for message in messages
            if isinstance(message, AssistantMessage)
            for call in message.tool_calls
        } - fulfilled
        if not unfulfilled:
            return
        calls = [
            call
            for message in messages
            if isinstance(message, AssistantMessage)
            for call in message.tool_calls
            if call.id in unfulfilled
        ]
        self._close_tool_calls(calls, "not executed because the previous turn ended")

    def _close_tool_calls(self, calls: list[ToolCall], reason: str) -> None:
        fulfilled = {
            message.tool_call_id
            for message in self._conversation.messages
            if isinstance(message, ToolResultMessage)
        }
        for call in calls:
            if call.id not in fulfilled:
                self._conversation.add(
                    ToolResultMessage(
                        tool_call_id=call.id,
                        tool_name=call.name,
                        content=reason,
                        is_error=True,
                    )
                )

    async def _handle_tool_call(
        self, tool_call: ToolCall, iteration: int
    ) -> AsyncIterator[RuntimeEvent]:
        bridge_tool: str | None = None
        protocol_tool_name = tool_call.name
        if tool_call.name == "tool_call":
            try:
                bridge_arguments = self._tools.validate_arguments(
                    "tool_call", tool_call.arguments
                )
                target_name = str(bridge_arguments["name"])
                target_arguments = bridge_arguments["arguments"]
                target = self._tools.get(target_name)
                if target.exposure is not ToolExposure.DEFERRED:
                    raise ToolValidationError(
                        f"tool_call only accepts deferred tools; {target_name!r} is "
                        f"{target.exposure.value}"
                    )
                if not isinstance(target_arguments, dict):
                    raise ToolValidationError("tool_call arguments must be an object")
                tool_call = tool_call.model_copy(
                    update={"name": target_name, "arguments": target_arguments}
                )
                bridge_tool = "tool_call"
            except (ToolValidationError, KeyError) as exc:
                async for event in self._record_tool_failure(
                    tool_call,
                    iteration,
                    self._bridge_error(tool_call, str(exc)),
                    error_type=type(exc).__name__,
                ):
                    yield event
                return

        name = tool_call.name
        try:
            registered = self._tools.get(name)
        except KeyError as exc:
            async for event in self._record_tool_failure(
                tool_call, iteration, str(exc), error_type=type(exc).__name__
            ):
                yield event
            return
        if registered.exposure is ToolExposure.HIDDEN:
            async for event in self._record_tool_failure(
                tool_call,
                iteration,
                f"tool {name!r} is hidden and unavailable",
                error_type="ToolUnavailable",
                bridge_tool=bridge_tool,
                protocol_tool_name=protocol_tool_name,
            ):
                yield event
            return
        try:
            self._sensitive_paths.check_tool_call(name, tool_call.arguments)
        except SensitiveAccessError as exc:
            yield await self._emit(
                SensitiveAccessBlocked(
                    iteration=iteration,
                    tool_call_id=tool_call.id,
                    tool_name=name,
                    path=exc.path,
                )
            )
            self._conversation.add(
                ToolResultMessage(
                    tool_call_id=tool_call.id,
                    tool_name=protocol_tool_name,
                    content=f"Blocked access to sensitive file: {exc.path}",
                    is_error=True,
                )
            )
            return
        yield await self._emit(
            ToolRequested(
                iteration=iteration,
                tool_call_id=tool_call.id,
                tool_name=name,
                arguments=tool_call.arguments,
                bridge_tool=bridge_tool,
                underlying_tool=name if bridge_tool else None,
            )
        )
        if (
            registered.exposure is ToolExposure.DEFERRED
            and bridge_tool is None
            and self._tool_surface.config.mode == "progressive"
        ):
            payload = json.dumps(
                {
                    "error": "deferred_tool_requires_bridge",
                    "tool": name,
                    "hint": "Use tool_search/tool_describe, then tool_call.",
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            async for event in self._record_tool_failure(
                tool_call,
                iteration,
                payload,
                error_type="DeferredToolDirectCall",
            ):
                yield event
            return

        permission = await self._permission.check(name, tool_call.arguments)
        if permission.decision is PermissionDecision.DENY:
            async for event in self._deny_tool(
                tool_call,
                iteration,
                permission.reason,
                bridge_tool=bridge_tool,
                protocol_tool_name=protocol_tool_name,
            ):
                yield event
            return
        if permission.decision is PermissionDecision.ASK:
            yield await self._emit(
                ToolApprovalRequired(
                    iteration=iteration,
                    tool_call_id=tool_call.id,
                    tool_name=name,
                    arguments=tool_call.arguments,
                    reason=permission.reason,
                    bridge_tool=bridge_tool,
                    underlying_tool=name if bridge_tool else None,
                )
            )
            approved = (
                self._approval_handler is not None
                and await self._approval_handler.request_approval(
                    ApprovalRequest(name, tool_call.arguments, permission.reason)
                )
            )
            if not approved:
                async for event in self._deny_tool(
                    tool_call,
                    iteration,
                    "approval denied",
                    bridge_tool=bridge_tool,
                    protocol_tool_name=protocol_tool_name,
                ):
                    yield event
                return

        try:
            arguments = self._tools.validate_arguments(name, tool_call.arguments)
        except (ToolValidationError, KeyError) as exc:
            error = (
                self._deferred_validation_error(name, tool_call.arguments, str(exc))
                if bridge_tool
                else str(exc)
            )
            async for event in self._record_tool_failure(
                tool_call,
                iteration,
                error,
                error_type=type(exc).__name__,
                bridge_tool=bridge_tool,
                protocol_tool_name=protocol_tool_name,
            ):
                yield event
            return

        yield await self._emit(
            ToolStarted(
                iteration=iteration,
                tool_name=name,
                tool_call_id=tool_call.id,
                bridge_tool=bridge_tool,
                underlying_tool=name if bridge_tool else None,
            )
        )
        self._budget.record_tool_call()
        tool_started = time.monotonic()
        try:
            result = await self._tools.get(name).execute(arguments)
        except Exception as exc:
            observation = self._observation.apply(
                name, f"{type(exc).__name__}: {exc}"
            )
            async for event in self._record_tool_failure(
                tool_call,
                iteration,
                observation.content,
                (time.monotonic() - tool_started) * 1000,
                error_type=type(exc).__name__,
                bridge_tool=bridge_tool,
                protocol_tool_name=protocol_tool_name,
                observation=observation,
            ):
                yield event
            return

        duration_ms = (time.monotonic() - tool_started) * 1000
        if result.is_error:
            observation = self._observation.apply(name, result.output)
            async for event in self._record_tool_failure(
                tool_call,
                iteration,
                observation.content,
                duration_ms,
                bridge_tool=bridge_tool,
                protocol_tool_name=protocol_tool_name,
                observation=observation,
            ):
                yield event
            return

        observation = self._observation.apply(name, result.output)
        yield await self._emit(
            ToolCompleted(
                iteration=iteration,
                tool_name=name,
                tool_call_id=tool_call.id,
                output=observation.content,
                duration_ms=duration_ms,
                bridge_tool=bridge_tool,
                underlying_tool=name if bridge_tool else None,
                truncated=observation.truncated,
                original_chars=observation.original_chars,
                delivered_chars=observation.delivered_chars,
                redacted=observation.redacted,
            )
        )
        if result.state_updates:
            for update in result.state_updates:
                self._apply_state_update(update)
            yield await self._emit(
                ScientificStateUpdated(summary=format_scientific_state(self._scientific))
            )
        self._conversation.add(
            ToolResultMessage(
                tool_call_id=tool_call.id,
                tool_name=protocol_tool_name,
                content=observation.content,
                is_error=False,
            )
        )

    async def _deny_tool(
        self,
        tool_call: ToolCall,
        iteration: int,
        reason: str,
        *,
        bridge_tool: str | None = None,
        protocol_tool_name: str | None = None,
    ) -> AsyncIterator[RuntimeEvent]:
        yield await self._emit(
            ToolPermissionDenied(
                iteration=iteration,
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                reason=reason,
                bridge_tool=bridge_tool,
                underlying_tool=tool_call.name if bridge_tool else None,
            )
        )
        self._conversation.add(
            ToolResultMessage(
                tool_call_id=tool_call.id,
                tool_name=protocol_tool_name or tool_call.name,
                content=f"permission denied: {reason}",
                is_error=True,
            )
        )

    async def _record_tool_failure(
        self,
        tool_call: ToolCall,
        iteration: int,
        error: str,
        duration_ms: float = 0.0,
        error_type: str | None = None,
        *,
        bridge_tool: str | None = None,
        protocol_tool_name: str | None = None,
        observation: Any | None = None,
    ) -> AsyncIterator[RuntimeEvent]:
        yield await self._emit(
            ToolFailed(
                iteration=iteration,
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                error=error,
                duration_ms=duration_ms,
                error_type=error_type,
                bridge_tool=bridge_tool,
                underlying_tool=tool_call.name if bridge_tool else None,
                truncated=bool(observation and observation.truncated),
                original_chars=(observation.original_chars if observation else None),
                delivered_chars=(observation.delivered_chars if observation else None),
                redacted=bool(observation and observation.redacted),
            )
        )
        self._conversation.add(
            ToolResultMessage(
                tool_call_id=tool_call.id,
                tool_name=protocol_tool_name or tool_call.name,
                content=error,
                is_error=True,
            )
        )

    def _bridge_error(self, tool_call: ToolCall, detail: str) -> str:
        payload = {
            "error": "invalid_tool_call_bridge",
            "detail": detail,
            "required": ["name", "arguments"],
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _deferred_validation_error(
        self, name: str, arguments: dict[str, object], detail: str
    ) -> str:
        definition = self._tools.definition(name)
        raw_required = definition.input_schema.get("required", [])
        required = raw_required if isinstance(raw_required, list) else []
        missing = [item for item in required if item not in arguments]
        payload = {
            "error": "missing_required_arguments" if missing else "validation_error",
            "tool": name,
            "missing": missing,
            "detail": detail,
            "parameter_help": compact_parameter_help(definition),
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _apply_state_update(self, update: Any) -> None:
        if isinstance(update, Evidence):
            self._scientific.add_evidence(update)
        elif isinstance(update, ScientificClaim):
            self._scientific.add_claim(update)
        elif isinstance(update, CalculationRecord):
            self._scientific.add_calculation(update)
        elif isinstance(update, ScientificTask):
            self._scientific.add_task(update)
        else:
            raise ToolError(f"unsupported scientific state update: {type(update).__name__}")

    async def _emit_budget(self, iteration: int) -> RuntimeEvent:
        return await self._emit(
            BudgetUpdated(
                model_calls=self._budget.model_calls,
                tool_calls=self._budget.tool_calls,
                iteration=iteration,
                input_tokens=self._budget.input_tokens,
                output_tokens=self._budget.output_tokens,
            )
        )

    async def _emit(self, event: RuntimeEvent) -> RuntimeEvent:
        event.session_id = self._session_id
        event.run_id = self._run_id
        for sink in self._event_sinks:
            result = sink(event)
            if inspect.isawaitable(result):
                await result
        return event
