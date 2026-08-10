"""Shared runner for the three IR vertical slices.

Two modes:
- scripted: drives the real AgentRuntime with a deterministic step plan
  (fake provider), so every tool call executes against the real scientific
  capability packs and the trace is fully reproducible.
- llm: runs the real configured LLM end-to-end on the goal.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from photomatagent.logging.event_logger import EventLogger, default_sessions_dir
from photomatagent.models.fake import FakeModelProvider, FakeResponse, scripted_tool_call
from photomatagent.models.types import ToolCall
from photomatagent.runtime.budget import BudgetState
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import AllowAllPolicy
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.factory import create_default_registry
from photomatagent.workspace import Workspace


@dataclass
class ScriptedStep:
    """One scripted agent step: reasoning text plus the tool calls it makes."""

    reasoning: str
    tool_calls: list[ToolCall] = field(default_factory=list)

    def fake_responses(self) -> list[FakeResponse]:
        # One response per step: text + calls in the same frame, so the loop
        # continues (finish_reason=tool_calls) until the final text-only step.
        return [FakeResponse(text=self.reasoning, tool_calls=list(self.tool_calls))]


def scripted_call(name: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(name=name, arguments=arguments)


async def run_scripted(
    *,
    goal: str,
    steps: list[ScriptedStep],
    workspace_root: Path,
    session_dir: Path,
) -> dict[str, Any]:
    responses: list[FakeResponse] = []
    for step in steps:
        responses.extend(step.fake_responses())
    workspace = Workspace(workspace_root)
    scientific = ScientificState()
    registry = create_default_registry(scientific, workspace)
    model = FakeModelProvider(responses)
    logger = EventLogger(session_dir)
    runtime = AgentRuntime(
        model=model,
        tools=registry,
        workspace=workspace,
        scientific_state=scientific,
        permission_policy=AllowAllPolicy(),
        budget=BudgetState(max_iterations=40),
        event_sinks=[logger.log],
        session_id=logger.session_id,
    )
    events = [event async for event in runtime.run(goal)]
    trajectory = _trajectory_from_events(events, steps)
    meta = next(
        (event for event in events if event.kind == "scientific_trace_meta"), None
    )
    return {
        "mode": "scripted",
        "goal": goal,
        "trace_path": str(logger.events_path),
        "session_id": logger.session_id,
        "final_answer": steps[-1].reasoning,
        "evidence_count": len(scientific.evidence),
        "evidence": [
            {
                "subject": ev.subject,
                "property": ev.property,
                "value": ev.value,
                "unit": ev.unit,
                "source": ev.source,
                "source_type": ev.source_type,
                "summary": ev.summary,
            }
            for ev in scientific.evidence
        ],
        "trajectory": trajectory,
        "trace_meta": meta.model_dump() if meta else None,
    }


async def run_llm(
    *,
    goal: str,
    workspace_root: Path,
    session_dir: Path,
    provider: str,
    model_name: str,
    max_iterations: int = 14,
) -> dict[str, Any]:
    from photomatagent.models.factory import create_provider

    workspace = Workspace(workspace_root)
    scientific = ScientificState()
    registry = create_default_registry(scientific, workspace)
    model = create_provider(provider, model_name)
    logger = EventLogger(session_dir)
    runtime = AgentRuntime(
        model=model,
        tools=registry,
        workspace=workspace,
        scientific_state=scientific,
        permission_policy=AllowAllPolicy(),
        budget=BudgetState(max_iterations=max_iterations),
        event_sinks=[logger.log],
        session_id=logger.session_id,
    )
    events = [event async for event in runtime.run(goal)]
    final = next(
        (event for event in reversed(events) if event.kind == "model_response_completed"),
        None,
    )
    meta = next(
        (event for event in events if event.kind == "scientific_trace_meta"), None
    )
    tool_events = [event for event in events if event.kind == "tool_completed"]
    return {
        "mode": "llm",
        "goal": goal,
        "trace_path": str(logger.events_path),
        "session_id": logger.session_id,
        "provider": provider,
        "model": model_name,
        "iterations": max((event.iteration for event in events if hasattr(event, "iteration") and event.iteration), default=0),
        "tool_calls": [event.tool_name for event in tool_events],
        "usage": final.usage if final else None,
        "evidence_count": len(scientific.evidence),
        "evidence": [
            {
                "subject": ev.subject,
                "property": ev.property,
                "value": ev.value,
                "unit": ev.unit,
                "source": ev.source,
                "summary": ev.summary,
            }
            for ev in scientific.evidence
        ],
        "trace_meta": meta.model_dump() if meta else None,
        "event_count": len(events),
    }


def _trajectory_from_events(events: list, steps: list[ScriptedStep]) -> list[dict]:
    """Collapse the event stream into a step-wise evidence-gap trajectory."""
    tool_events: list[tuple[str, str]] = []
    for event in events:
        if event.kind == "tool_completed":
            tool_events.append((event.tool_name, event.output))
        elif event.kind == "tool_failed":
            tool_events.append((event.tool_name, f"ERROR: {event.error}"))
    trajectory = []
    cursor = 0
    for index, step in enumerate(steps):
        results: dict[str, str] = {}
        for call in step.tool_calls:
            resolved_name = (
                str(call.arguments.get("name", "tool_call"))
                if call.name == "tool_call"
                else call.name
            )
            if cursor < len(tool_events):
                actual_name, output = tool_events[cursor]
                cursor += 1
            else:
                actual_name, output = "", "(absent)"
            if actual_name not in {call.name, resolved_name}:
                output = f"(order mismatch: saw {actual_name})"
            results[f"{call.name}->{resolved_name}" if call.name == "tool_call" else call.name] = output
        trajectory.append(
            {
                "step": index + 1,
                "reasoning": step.reasoning,
                "tool_calls": [
                    {"name": call.name, "arguments": call.arguments}
                    for call in step.tool_calls
                ],
                "results": results,
            }
        )
    return trajectory


def save_result(result: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def default_session_dir() -> Path:
    return Path(default_sessions_dir()) / "vertical"
