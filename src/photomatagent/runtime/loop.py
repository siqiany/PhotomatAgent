"""AgentRuntime: the event-driven agent loop.

This is the only loop in the codebase. It is deliberately readable end-to-end
in one file: context -> model -> tools -> state -> stop, with every step
announced through RuntimeEvents. Consumers (CLI, TUI, API, JSONL logger)
subscribe via ``async for event in runtime.run(goal)``.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from photomatagent.models.base import ModelProvider
from photomatagent.models.types import ToolCall
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
    RuntimeEvent,
    ScientificStateUpdated,
    TextDelta,
    ToolApprovalRequired,
    ToolCompleted,
    ToolFailed,
    ToolRequested,
    ToolStarted,
)
from photomatagent.runtime.permissions import AllowAllPolicy, ApprovalHandler, PermissionDecision, PermissionPolicy
from photomatagent.runtime.state import ConversationState, Message
from photomatagent.runtime.stop_policy import StopDecision, StopPolicy
from photomatagent.scientific.calculations import CalculationRecord
from photomatagent.scientific.claims import ScientificClaim
from photomatagent.scientific.evidence import Evidence
from photomatagent.scientific.state import ScientificState
from photomatagent.scientific.tasks import ScientificTask
from photomatagent.tools.base import Tool, ToolError
from photomatagent.tools.registry import ToolRegistry

EventSink = Callable[[RuntimeEvent], Awaitable[None] | None]


class AgentRuntime:
    def __init__(
        self,
        *,
        model: ModelProvider,
        tools: ToolRegistry,
        scientific_state: ScientificState | None = None,
        context_builder: ContextBuilder | None = None,
        permission_policy: PermissionPolicy | None = None,
        stop_policy: StopPolicy | None = None,
        budget: BudgetState | None = None,
        approval_handler: ApprovalHandler | None = None,
        event_sinks: list[EventSink] | None = None,
    ) -> None:
        self._model = model
        self._tools = tools
        self._scientific = scientific_state or ScientificState()
        self._context_builder = context_builder or ContextBuilder()
        self._permission = permission_policy or AllowAllPolicy()
        self._stop_policy = stop_policy or StopPolicy()
        self._budget = budget or BudgetState()
        self._approval_handler = approval_handler
        self._event_sinks = list(event_sinks or [])
        # Conversation persists across run() calls on the same runtime, so a
        # chat session accumulates history; scientific state persists too.
        self._conversation = ConversationState()

    @property
    def scientific_state(self) -> ScientificState:
        return self._scientific

    @property
    def budget(self) -> BudgetState:
        return self._budget

    async def run(self, goal: str) -> AsyncIterator[RuntimeEvent]:
        """Run the loop for one user goal. Yields events as they happen."""
        self._scientific.goal = goal
        self._conversation.add(Message(role="user", content=goal))
        iteration = 0
        try:
            yield await self._emit(LoopStarted(goal=goal))
            while True:
                iteration += 1
                yield await self._emit(LoopIterationStarted(iteration=iteration))
                self._budget.record_iteration()

                context = self._context_builder.build(self._conversation, self._scientific)
                yield await self._emit(
                    ModelRequestStarted(iteration=iteration, message_count=len(context))
                )
                response = await self._model.complete(context, self._tools.list_tools())
                self._budget.record_model_call(response.usage)

                if response.text:
                    yield await self._emit(TextDelta(text=response.text))
                yield await self._emit(
                    ModelResponseCompleted(
                        iteration=iteration,
                        finish_reason=response.finish_reason,
                        tool_call_count=len(response.tool_calls),
                        usage=response.usage.model_dump(),
                    )
                )
                self._conversation.add(
                    Message(
                        role="assistant",
                        content=response.text,
                        tool_calls=response.tool_calls or None,
                    )
                )

                decision = self._stop_policy.should_stop(
                    iteration=iteration, response=response, budget=self._budget
                )
                if decision.should_stop:
                    yield await self._emit_budget(iteration)
                    yield await self._emit(
                        LoopCompleted(iterations=iteration, reason=decision.reason)
                    )
                    return

                async for event in self._handle_tool_calls(response.tool_calls):
                    yield event
                yield await self._emit_budget(iteration)
        except Exception as exc:
            yield await self._emit(LoopFailed(error=f"{type(exc).__name__}: {exc}"))
            raise

    async def _handle_tool_calls(self, tool_calls: list[ToolCall]) -> AsyncIterator[RuntimeEvent]:
        for tool_call in tool_calls:
            async for event in self._handle_tool_call(tool_call):
                yield event

    async def _handle_tool_call(self, tool_call: ToolCall) -> AsyncIterator[RuntimeEvent]:
        name = tool_call.name
        yield await self._emit(ToolRequested(tool_name=name, arguments=tool_call.arguments))

        permission = await self._permission.check(name, tool_call.arguments)
        if permission.decision is PermissionDecision.DENY:
            error = f"permission denied: {permission.reason}"
            async for event in self._fail_tool(name, tool_call, error):
                yield event
            return
        if permission.decision is PermissionDecision.ASK:
            yield await self._emit(
                ToolApprovalRequired(tool_name=name, arguments=tool_call.arguments)
            )
            approved = False
            if self._approval_handler is not None:
                approved = await self._approval_handler(name, tool_call.arguments)
            if not approved:
                async for event in self._fail_tool(
                    name, tool_call, "user declined tool execution"
                ):
                    yield event
                return

        try:
            arguments = self._tools.validate_arguments(name, tool_call.arguments)
        except (ToolError, KeyError) as exc:
            async for event in self._fail_tool(name, tool_call, str(exc)):
                yield event
            return

        yield await self._emit(ToolStarted(tool_name=name, tool_call_id=tool_call.id))
        self._budget.record_tool_call()
        try:
            tool = self._tools.get(name)
            result = await tool.execute(arguments)
        except Exception as exc:
            async for event in self._fail_tool(
                name, tool_call, f"{type(exc).__name__}: {exc}"
            ):
                yield event
            return

        yield await self._emit(
            ToolCompleted(tool_name=name, tool_call_id=tool_call.id, output=result.output)
        )
        if result.state_updates:
            for update in result.state_updates:
                self._apply_state_update(update)
            yield await self._emit(
                ScientificStateUpdated(summary=format_scientific_state(self._scientific))
            )
        self._conversation.add(
            Message(role="tool", content=result.output, name=name, tool_call_id=tool_call.id)
        )

    async def _fail_tool(
        self, name: str, tool_call: ToolCall, error: str
    ) -> AsyncIterator[RuntimeEvent]:
        yield await self._emit(
            ToolFailed(tool_name=name, tool_call_id=tool_call.id, error=error)
        )
        self._conversation.add(
            Message(role="tool", content=f"ERROR: {error}", name=name, tool_call_id=tool_call.id)
        )

    def _apply_state_update(self, update: Any) -> None:
        """Apply a structured update produced by a tool to ScientificState."""
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
            )
        )

    async def _emit(self, event: RuntimeEvent) -> RuntimeEvent:
        """Notify sinks, then hand the event to the consumer (via yield in run())."""
        for sink in self._event_sinks:
            result = sink(event)
            if inspect.isawaitable(result):
                await result
        return event
