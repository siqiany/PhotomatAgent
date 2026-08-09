from __future__ import annotations

import pytest

from photomatagent.models.fake import FakeModelProvider, FakeResponse, scripted_tool_call
from photomatagent.runtime.permissions import AskPolicy, AutoApproveHandler, DenyAllPolicy, DenyHandler
from photomatagent.tools.echo import EchoTool
from photomatagent.workspace import Workspace

from conftest import collect, make_runtime


class SpyEcho(EchoTool):
    def __init__(self) -> None:
        self.executions = 0

    async def execute(self, arguments: dict):
        self.executions += 1
        return await super().execute(arguments)


def _runtime_with_spy(policy, approval_handler=None):
    spy = SpyEcho()
    runtime = make_runtime(
        FakeModelProvider(
            [scripted_tool_call("echo", {"text": "x"}), FakeResponse(text="done")]
        ),
        permission_policy=policy,
        approval_handler=approval_handler,
    )
    runtime._tools._tools["echo"] = spy
    return runtime, spy


@pytest.mark.asyncio
async def test_deny_prevents_execution_and_records_tool_result():
    runtime, spy = _runtime_with_spy(DenyAllPolicy())
    events = await collect(runtime, "say x")
    assert spy.executions == 0
    assert any(event.kind == "tool_permission_denied" for event in events)
    result = runtime.conversation_state.messages[-2]
    assert result.is_error is True
    assert "permission denied" in result.content


@pytest.mark.asyncio
async def test_ask_declined_prevents_execution():
    runtime, spy = _runtime_with_spy(AskPolicy(), DenyHandler())
    events = await collect(runtime, "say x")
    assert spy.executions == 0
    assert any(event.kind == "tool_approval_required" for event in events)
    assert any(event.kind == "tool_permission_denied" for event in events)


@pytest.mark.asyncio
async def test_ask_approved_executes():
    runtime, spy = _runtime_with_spy(AskPolicy(), AutoApproveHandler())
    events = await collect(runtime, "say x")
    assert spy.executions == 1
    assert any(event.kind == "tool_completed" for event in events)


@pytest.mark.asyncio
async def test_write_ask_deny_leaves_file_unchanged(tmp_path):
    target = tmp_path / "new.txt"
    model = FakeModelProvider(
        [
            scripted_tool_call("write", {"path": "new.txt", "content": "secret"}),
            FakeResponse(text="denied"),
        ]
    )
    runtime = make_runtime(
        model,
        workspace=Workspace(tmp_path),
        permission_policy=AskPolicy(),
        approval_handler=DenyHandler(),
    )
    await collect(runtime, "write")
    assert not target.exists()


@pytest.mark.asyncio
async def test_write_ask_allow_changes_file(tmp_path):
    target = tmp_path / "new.txt"
    model = FakeModelProvider(
        [
            scripted_tool_call("write", {"path": "new.txt", "content": "hello"}),
            FakeResponse(text="written"),
        ]
    )
    runtime = make_runtime(
        model,
        workspace=Workspace(tmp_path),
        permission_policy=AskPolicy(),
        approval_handler=AutoApproveHandler(),
    )
    await collect(runtime, "write")
    assert target.read_text(encoding="utf-8") == "hello"
