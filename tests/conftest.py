from __future__ import annotations

import pytest

from photomatagent.models.fake import FakeModelProvider
from photomatagent.runtime.budget import BudgetState
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import AllowAllPolicy
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.factory import create_default_registry
from photomatagent.workspace import Workspace


def make_runtime(
    model,
    *,
    max_iterations: int = 10,
    permission_policy=None,
    approval_handler=None,
    event_sinks=None,
    workspace=None,
) -> AgentRuntime:
    """Build a standard test runtime: default tools, allow-all, fresh state."""
    scientific = ScientificState()
    boundary = workspace or Workspace(".")
    registry = create_default_registry(scientific, boundary)
    budget = BudgetState(max_iterations=max_iterations)
    return AgentRuntime(
        model=model,
        tools=registry,
        workspace=boundary,
        scientific_state=scientific,
        permission_policy=permission_policy or AllowAllPolicy(),
        approval_handler=approval_handler,
        event_sinks=event_sinks,
        budget=budget,
    )


async def collect(runtime: AgentRuntime, goal: str) -> list:
    return [e async for e in runtime.run(goal)]


@pytest.fixture
def fake_model():
    return FakeModelProvider()
