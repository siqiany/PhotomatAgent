from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from photomatagent.cli.commands import ChatCommandRouter, _print_cli_capture, strip_ansi_codes
from photomatagent.logging.event_logger import EventLogger
from photomatagent.models.fake import FakeModelProvider
from photomatagent.models.types import UserMessage
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.context_engine import ContextEngine
from photomatagent.runtime.state import ConversationState
from photomatagent.runtime.permissions import (
    ApprovalScope,
    ApprovalSettings,
    DenyAllPolicy,
    SwitchablePermissionPolicy,
)
from photomatagent.scientific.evidence import Evidence
from photomatagent.scientific.state import ScientificState
from photomatagent.sessions.store import load_session_snapshot, save_session_snapshot
from photomatagent.tools.registry import ToolRegistry
from photomatagent.workspace import Workspace


def _router(tmp_path):
    workspace = Workspace(tmp_path)
    scientific = ScientificState()
    policy = SwitchablePermissionPolicy(
        DenyAllPolicy(), settings=ApprovalSettings(tmp_path)
    )
    runtime = AgentRuntime(
        model=FakeModelProvider([]),
        tools=ToolRegistry(),
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
    assert "/resume" in output
    assert "/evolve" in output


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


def _historical_session(tmp_path, name: str = "session_old") -> Path:
    session_dir = tmp_path / name
    conversation = ConversationState()
    conversation.add(UserMessage(content="historical goal"))
    scientific = ScientificState(goal="historical goal")
    scientific.add_evidence(
        Evidence(
            type="calculation",
            source="mock",
            content="historical evidence",
            confidence=0.7,
        )
    )
    save_session_snapshot(
        session_dir,
        conversation=conversation,
        scientific=scientific,
        engine=ContextEngine().snapshot(),
    )
    return session_dir


@pytest.mark.asyncio
async def test_resume_command_restores_historical_session_state(tmp_path):
    session_dir = _historical_session(tmp_path)
    router, _, stream = _router(tmp_path)

    await router.execute(f"/resume {session_dir}")

    output = stream.getvalue()
    assert "已回溯到 session" in output
    user_goals = [
        message.content
        for message in router.runtime.conversation_state.messages
        if isinstance(message, UserMessage)
    ]
    assert user_goals == ["historical goal"]
    assert len(router.runtime.scientific_state.evidence) == 1


@pytest.mark.asyncio
async def test_resume_command_switches_logger_into_resumed_session(tmp_path):
    session_dir = _historical_session(tmp_path)
    logger = EventLogger(tmp_path, session_id="current_session")
    router, _, _ = _router(tmp_path)
    router.sessions_dir = tmp_path
    router.logger = logger

    await router.execute(f"/resume {session_dir.name}")

    assert logger.session_id == "session_old"
    assert logger.session_dir == tmp_path / "session_old"
    assert logger.events_path == tmp_path / "session_old" / "events.jsonl"
    # The current in-chat session was snapshotted before switching away.
    assert load_session_snapshot(tmp_path / "current_session") is not None


@pytest.mark.asyncio
async def test_resume_command_usage_and_unresumable_errors(tmp_path):
    router, _, stream = _router(tmp_path)
    router.sessions_dir = tmp_path

    await router.execute("/resume")
    assert "用法：/resume" in stream.getvalue()

    stream.truncate(0)
    stream.seek(0)
    (tmp_path / "session_without_snapshot").mkdir()
    await router.execute("/resume session_without_snapshot")
    assert "没有可恢复的状态" in stream.getvalue()


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
