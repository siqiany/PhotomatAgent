"""Task 18: vasp.wait defers polling to an internal timer (token saver).

vasp.wait must consume exactly ONE model round-trip regardless of how long
the workflow keeps running: polling happens inside the tool via
``asyncio.sleep``, so the waiting agent is not charged a model call per
vasp.status. These tests exercise the tool against a fake service.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from photomatagent.scientific.applications.vasp.unified.executors import (
    StatusResult,
)
from photomatagent.scientific.applications.vasp.unified.models import (
    WorkflowState,
)
from photomatagent.scientific.capabilities.contracts import ScientificToolResult


class _WaitTestService:
    """Minimal fake service whose status transitions shared mutable state.

    The tool only calls ``service.status(workflow_id)``, so this one method
    is enough to drive both the settle and the timeout paths.
    """

    def __init__(self, sequence: list[WorkflowState]) -> None:
        self._sequence = list(sequence)
        self._calls = 0

    @property
    def status_calls(self) -> int:
        return self._calls

    async def status(self, workflow_id: str) -> "_FakeServiceResult":
        self._calls += 1
        state = self._sequence[min(self._calls - 1, len(self._sequence) - 1)]
        return _FakeServiceResult(state)


class _FakeServiceResult:
    def __init__(self, state: WorkflowState) -> None:
        self.state = state
        self.workflow_id = "vasp_0123456789abcdef"

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "state": self.state.value,
            "ok": True,
            "data": {},
            "errors": [],
        }


def _make_tool(service: _WaitTestService):
    from photomatagent.scientific.applications.vasp.unified.tool_pack import (
        VaspWaitTool,
    )

    return VaspWaitTool(service)


@pytest.mark.asyncio
async def test_wait_settles_with_one_tool_and_bounded_polls():
    service = _WaitTestService(
        [WorkflowState.SUBMITTED, WorkflowState.RUNNING, WorkflowState.FAILED]
    )
    tool = _make_tool(service)
    result = await tool.execute(
        {"workflow_id": "vasp_0123456789abcdef", "poll_interval_seconds": 5}
    )
    assert isinstance(result, ScientificToolResult)
    assert not result.is_error
    data = result.data
    assert data["state"] == "FAILED"
    assert data["wait"]["settled"] is True
    assert service.status_calls == 3


@pytest.mark.asyncio
async def test_wait_timeout_returns_unsettled_without_error():
    service = _WaitTestService([WorkflowState.SUBMITTED, WorkflowState.RUNNING])
    tool = _make_tool(service)
    result = await tool.execute(
        {
            "workflow_id": "vasp_0123456789abcdef",
            "timeout_seconds": 1,
            "poll_interval_seconds": 5,
        }
    )
    assert not result.is_error
    data = result.data
    assert data["wait"]["settled"] is False
    assert data["wait"]["timeout_seconds"] == 1
    # At least one status poll happened, and the tool returned on its own.
    assert service.status_calls >= 1


@pytest.mark.asyncio
async def test_wait_missing_workflow_id_is_an_error():
    tool = _make_tool(_WaitTestService([]))
    result = await tool.execute({})
    assert result.is_error
    assert "workflow_id" in str(result.output)


@pytest.mark.asyncio
async def test_wait_schema_bounds_poll_and_timeout():
    from photomatagent.scientific.applications.vasp.unified.tool_pack import (
        VaspWaitTool,
    )

    schema = VaspWaitTool(_WaitTestService([])).input_schema
    props = schema["properties"]
    assert props["timeout_seconds"]["minimum"] == 5
    assert props["timeout_seconds"]["maximum"] == 1800
    assert props["poll_interval_seconds"]["minimum"] == 5
    assert props["poll_interval_seconds"]["maximum"] == 300


@pytest.mark.asyncio
async def test_wait_is_async_and_can_be_cancelled_cleanly():
    """A long wait must not block the event loop while polling."""
    service = _WaitTestService([WorkflowState.SUBMITTED])
    tool = _make_tool(service)
    started = asyncio.get_running_loop().time()
    task = asyncio.create_task(
        tool.execute(
            {
                "workflow_id": "vasp_0123456789abcdef",
                "timeout_seconds": 300,
                "poll_interval_seconds": 5,
            }
        )
    )
    # Let one poll happen, then cancel: no exception should leak.
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert service.status_calls >= 1