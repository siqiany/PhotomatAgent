"""The complete, visible PhotomatAgent control loop.

Read this file top-to-bottom to follow:
context -> provider stream -> tool calls -> permission -> execution -> results
-> next iteration -> stop. No SDK or UI owns any part of that sequence.
"""

from __future__ import annotations

import inspect
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
from photomatagent.runtime.state import ConversationState
from photomatagent.runtime.stop_policy import StopPolicy
from photomatagent.scientific.calculations import CalculationRecord
from photomatagent.scientific.claims import ScientificClaim
from photomatagent.scientific.evidence import Evidence
from photomatagent.scientific.state import ScientificState
from photomatagent.scientific.tasks import ScientificTask
from photomatagent.tools.registry import ToolRegistry
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
        self._permission = permission_policy or default_permission_policy()
        self._stop_policy = stop_policy or StopPolicy()
        self._budget = budget or BudgetState()
        self._approval_handler = approval_handler
        self._event_sinks = list(event_sinks or [])
        self._conversation = ConversationState()
        self._session_id = session_id or uuid4().hex

    @property
    def scientific_state(self) -> ScientificState:
        return self._scientific

    @property
    def conversation_state(self) -> ConversationState:
        return self._conversation

    @property
    def budget(self) -> BudgetState:
        return self._budget

    async def run(self, goal: str) -> AsyncIterator[RuntimeEvent]:
        """Run one user turn while preserving conversation and scientific state."""
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

                messages = self._context_builder.build(self._conversation, self._scientific)
                request = ModelRequest(messages=messages, tools=self._tools.definitions())
                yield await self._emit(
                    ModelRequestStarted(
                        iteration=iteration,
                        message_count=len(messages),
                        provider=self._model.provider,
                        model=self._model.model,
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
                )
            )
            raise

    async def _handle_tool_call(
        self, tool_call: ToolCall, iteration: int
    ) -> AsyncIterator[RuntimeEvent]:
        name = tool_call.name
        yield await self._emit(
            ToolRequested(
                iteration=iteration,
                tool_call_id=tool_call.id,
                tool_name=name,
                arguments=tool_call.arguments,
            )
        )

        permission = await self._permission.check(name, tool_call.arguments)
        if permission.decision is PermissionDecision.DENY:
            async for event in self._deny_tool(tool_call, iteration, permission.reason):
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
                )
            )
            approved = (
                self._approval_handler is not None
                and await self._approval_handler.request_approval(
                    ApprovalRequest(name, tool_call.arguments, permission.reason)
                )
            )
            if not approved:
                async for event in self._deny_tool(tool_call, iteration, "approval denied"):
                    yield event
                return

        try:
            arguments = self._tools.validate_arguments(name, tool_call.arguments)
        except (ToolValidationError, KeyError) as exc:
            async for event in self._record_tool_failure(tool_call, iteration, str(exc)):
                yield event
            return

        yield await self._emit(
            ToolStarted(iteration=iteration, tool_name=name, tool_call_id=tool_call.id)
        )
        self._budget.record_tool_call()
        tool_started = time.monotonic()
        try:
            result = await self._tools.get(name).execute(arguments)
        except Exception as exc:
            async for event in self._record_tool_failure(
                tool_call,
                iteration,
                f"{type(exc).__name__}: {exc}",
                (time.monotonic() - tool_started) * 1000,
            ):
                yield event
            return

        duration_ms = (time.monotonic() - tool_started) * 1000
        if result.is_error:
            async for event in self._record_tool_failure(
                tool_call, iteration, result.output, duration_ms
            ):
                yield event
            return

        yield await self._emit(
            ToolCompleted(
                iteration=iteration,
                tool_name=name,
                tool_call_id=tool_call.id,
                output=result.output,
                duration_ms=duration_ms,
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
                tool_name=name,
                content=result.output,
                is_error=False,
            )
        )

    async def _deny_tool(
        self, tool_call: ToolCall, iteration: int, reason: str
    ) -> AsyncIterator[RuntimeEvent]:
        yield await self._emit(
            ToolPermissionDenied(
                iteration=iteration,
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                reason=reason,
            )
        )
        self._conversation.add(
            ToolResultMessage(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
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
    ) -> AsyncIterator[RuntimeEvent]:
        yield await self._emit(
            ToolFailed(
                iteration=iteration,
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                error=error,
                duration_ms=duration_ms,
            )
        )
        self._conversation.add(
            ToolResultMessage(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                content=error,
                is_error=True,
            )
        )

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
        for sink in self._event_sinks:
            result = sink(event)
            if inspect.isawaitable(result):
                await result
        return event
