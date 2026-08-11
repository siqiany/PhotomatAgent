from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from photomatagent.cli.commands import ChatCommandRouter
from photomatagent.models.fake import FakeModelProvider
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import (
    ApprovalScope,
    ApprovalSettings,
    DenyAllPolicy,
    SwitchablePermissionPolicy,
)
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.factory import create_default_registry
from photomatagent.workspace import Workspace


def _router(tmp_path):
    workspace = Workspace(tmp_path)
    scientific = ScientificState()
    policy = SwitchablePermissionPolicy(
        DenyAllPolicy(), settings=ApprovalSettings(tmp_path)
    )
    runtime = AgentRuntime(
        model=FakeModelProvider([]),
        tools=create_default_registry(scientific, workspace),
        workspace=workspace,
        scientific_state=scientific,
        permission_policy=policy,
    )
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=120)
    return ChatCommandRouter(console, runtime, workspace), policy, stream


@pytest.mark.asyncio
async def test_help_lists_commands(tmp_path):
    router, _, stream = _router(tmp_path)
    await router.execute("/help")
    output = stream.getvalue()
    assert "/doctor" in output
    assert "/approve -o" in output
    assert "/compact" in output


@pytest.mark.asyncio
async def test_approve_commands_change_scope(tmp_path):
    router, policy, _ = _router(tmp_path)
    await router.execute("/approve -o")
    assert policy.scope is ApprovalScope.SESSION
    await router.execute("/approve -a")
    assert policy.scope is ApprovalScope.ALWAYS
    await router.execute("/approve -b")
    assert policy.scope is ApprovalScope.DEFAULT


@pytest.mark.asyncio
async def test_unknown_slash_command_is_not_a_model_goal(tmp_path):
    router, _, stream = _router(tmp_path)
    handled = await router.execute("/does-not-exist")
    assert handled
    assert "未知命令" in stream.getvalue()


@pytest.mark.asyncio
async def test_skills_command_reuses_cli_without_invalid_workspace_option(tmp_path):
    router, _, stream = _router(tmp_path)
    await router.execute("/skills")
    assert "命令失败" not in stream.getvalue()
