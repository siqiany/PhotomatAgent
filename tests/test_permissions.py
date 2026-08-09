from __future__ import annotations

import pytest

from photomatagent.models.fake import FakeModelProvider, scripted_tool_call
from photomatagent.runtime.permissions import AskPolicy, DenyAllPolicy
from photomatagent.tools.echo import EchoTool

from conftest import collect, make_runtime


class SpyEcho(EchoTool):
    def __init__(self) -> None:
        super().__init__()
        self.executions = 0

    async def execute(self, arguments: dict) -> object:
        self.executions += 1
        return await super().execute(arguments)


def _runtime_with_spy(policy, approval_handler=None):
    spy = SpyEcho()
    runtime = make_runtime(
        FakeModelProvider([scripted_tool_call("echo", {"text": "x"})]),
        permission_policy=policy,
        approval_handler=approval_handler,
    )
    runtime._tools._tools["echo"] = spy  # replace echo with spy under the same name
    return runtime, spy


@pytest.mark.asyncio
async def test_deny_prevents_execution():
    runtime, spy = _runtime_with_spy(DenyAllPolicy())
    events = await collect(runtime, "say x")
    assert spy.executions == 0
    failed = [e for e in events if e.kind == "tool_failed"]
    assert failed and "permission denied" in failed[0].error


@pytest.mark.asyncio
async def test_ask_declined_prevents_execution():
    async def decline(name, arguments):
        return False

    runtime, spy = _runtime_with_spy(AskPolicy(), approval_handler=decline)
    events = await collect(runtime, "say x")
    assert spy.executions == 0
    assert any(e.kind == "tool_approval_required" for e in events)
    failed = [e for e in events if e.kind == "tool_failed"]
    assert failed and "declined" in failed[0].error


@pytest.mark.asyncio
async def test_ask_approved_executes():
    async def approve(name, arguments):
        return True

    runtime, spy = _runtime_with_spy(AskPolicy(), approval_handler=approve)
    events = await collect(runtime, "say x")
    assert spy.executions == 1
    assert any(e.kind == "tool_completed" for e in events)


@pytest.mark.asyncio
async def test_ask_without_handler_denies():
    runtime, spy = _runtime_with_spy(AskPolicy())
    await collect(runtime, "say x")
    assert spy.executions == 0
