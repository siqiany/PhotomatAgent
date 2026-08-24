from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from photomatagent.cli.commands import ChatCommandRouter, _print_cli_capture, strip_ansi_codes
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


def test_strip_ansi_codes_removes_csi_and_osc_escapes():
    sample = (
        "\x1b[1;2;36m10\x1b[0m \x1b[1;2;35mskill\x1b[0m"
        "\n\x1b]8;;https://example.com\x1b\\/skills/a/SKILL.md\x1b]8;;\x1b\\"
        "\n[1;2;33mdiagnostic\x1b[0m"
    )
    cleaned = strip_ansi_codes(sample)
    assert "\x1b" not in cleaned
    assert cleaned == "10 skill\n/skills/a/SKILL.md\n[1;2;33mdiagnostic"


def test_cli_capture_is_printed_without_ansi_escapes():
    """Regression: captured CLI stdout may carry ANSI styles (the CLI module
    console fixes its color system at import time while stdout is still the
    interactive TTY). Re-printing those raw escapes through another Rich console
    corrupts them into literal "[1;2;36m…[0m" text, so the interactive output
    must only ever receive clean plain text."""
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=120)
    captured = (
        "10 skill(s) from sources: photomat\n"
        "\x1b[1;2;36mcarrier-transport-analysis\x1b[0m — assess transport\n"
        "\x1b[2;35m/home/skills/carrier-transport-analysis/SKILL.md\x1b[0m"
    )
    _print_cli_capture(console, captured)
    output = stream.getvalue()
    assert "\x1b" not in output
    assert "10 skill(s) from sources: photomat" in output
    assert "carrier-transport-analysis — assess transport" in output
    assert "SKILL.md" in output
