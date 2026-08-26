from __future__ import annotations

import json

import pytest

from photomatagent.logging.event_logger import EventLogger
from photomatagent.models.fake import FakeModelProvider, FakeResponse, scripted_tool_call
from photomatagent.models.types import (
    AssistantMessage,
    ToolResultMessage,
    UserMessage,
)
from photomatagent.runtime.context_engine import (
    CompactionState,
    ContextEngine,
    RelevantResource,
)
from photomatagent.scientific.evidence import Evidence
from photomatagent.scientific.state import ScientificState
from photomatagent.sessions.store import (
    SESSION_STATE_FILENAME,
    load_session_snapshot,
    save_session_snapshot,
    session_is_resumable,
)

from conftest import collect, make_runtime


def test_snapshot_roundtrip_preserves_all_resume_state(tmp_path):
    conversation = make_runtime(FakeModelProvider()).conversation_state
    conversation.add(UserMessage(content="compute GaAs"))
    scientific = ScientificState(goal="compute GaAs")
    scientific.add_evidence(
        Evidence(type="calculation", source="mock", content="E_g = 0.31 eV", confidence=0.7)
    )
    engine = ContextEngine()
    engine.restore(
        compaction_state=CompactionState(
            goal="goal",
            progress=["inspected"],
            relevant_resources=[RelevantResource(reference="a.py")],
        ).model_dump(mode="json"),
        compacted_message_count=3,
        compaction_count=1,
    )

    path = save_session_snapshot(
        tmp_path,
        conversation=conversation,
        scientific=scientific,
        engine=engine.snapshot(),
    )
    assert path.name == SESSION_STATE_FILENAME
    assert session_is_resumable(tmp_path)

    restored = load_session_snapshot(tmp_path)
    assert restored.conversation == conversation
    assert restored.scientific == scientific
    assert restored.engine is not None
    assert restored.engine.compacted_message_count == 3
    assert restored.engine.compaction_count == 1
    assert restored.engine.compaction_state is not None
    assert restored.engine.compaction_state.progress == ["inspected"]


@pytest.mark.asyncio
async def test_restored_session_keeps_tool_state_and_continues(tmp_path):
    first_model = FakeModelProvider(
        [
            scripted_tool_call(
                "tool_call",
                {
                    "name": "mock.run_calculation",
                    "arguments": {
                        "material": "GaAs",
                        "calculation_type": "band_structure",
                    },
                },
                tool_call_id="resume-call-1",
            ),
            FakeResponse(text="gap is 0.31 eV"),
        ]
    )
    runtime = make_runtime(first_model, workspace=tmp_path)
    await collect(runtime, "compute GaAs band gap")
    assert len(runtime.scientific_state.evidence) == 1

    save_session_snapshot(
        tmp_path / "session-state",
        conversation=runtime.conversation_state,
        scientific=runtime.scientific_state,
        engine=runtime.context_engine.snapshot(),
    )
    snapshot = load_session_snapshot(tmp_path / "session-state")

    # A brand-new runtime (fresh scientific state) resumes the old session.
    second_runtime = make_runtime(
        FakeModelProvider([FakeResponse(text="continuing from previous session")]),
        workspace=tmp_path,
    )
    assert len(second_runtime.scientific_state.evidence) == 0
    second_runtime.restore_session(snapshot)
    assert len(second_runtime.scientific_state.evidence) == 1

    # Tools registered against the live scientific instance must see the
    # restored evidence (in-place mutation, not instance replacement).
    inspect = second_runtime._tools.get("scientific_state_inspect")
    result = await inspect.execute({"section": "all"})
    assert "Mock band_structure calculation for GaAs" in result.output

    # The restored conversation is preserved and a new turn continues on top.
    await collect(second_runtime, "follow-up question")
    assert any(
        isinstance(message, AssistantMessage)
        and message.text == "continuing from previous session"
        for message in second_runtime.conversation_state.messages
    )
    assert any(
        isinstance(message, AssistantMessage)
        and message.text == "gap is 0.31 eV"
        for message in second_runtime.conversation_state.messages
    )
    assert any(
        isinstance(message, UserMessage)
        and message.content == "compute GaAs band gap"
        for message in second_runtime.conversation_state.messages
    )
    assert any(
        isinstance(message, UserMessage) and message.content == "follow-up question"
        for message in second_runtime.conversation_state.messages
    )


def test_snapshot_redacts_secrets_before_write(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_RESUME_API_KEY", "sk-secret-value-to-redact-12345")
    conversation, scientific = make_runtime(FakeModelProvider()).conversation_state, ScientificState()
    conversation.add(UserMessage(content="leak"))
    conversation.add(
        ToolResultMessage(
            tool_call_id="id-1",
            tool_name="bash",
            content="TEST_RESUME_API_KEY=sk-secret-value-to-redact-12345",
        )
    )
    save_session_snapshot(
        tmp_path,
        conversation=conversation,
        scientific=scientific,
        engine=None,
    )
    raw = (tmp_path / SESSION_STATE_FILENAME).read_text(encoding="utf-8")
    assert "sk-secret-value-to-redact-12345" not in raw
    assert "[REDACTED]" in raw


def test_event_logger_accepts_explicit_session_id(tmp_path):
    logger = EventLogger(tmp_path, session_id="20260801T000000_abc123")
    assert logger.session_id == "20260801T000000_abc123"
    assert logger.session_dir == tmp_path / "20260801T000000_abc123"
    assert logger.events_path == logger.session_dir / "events.jsonl"
    assert logger.session_dir.is_dir()


@pytest.mark.asyncio
async def test_run_chat_resume_continues_into_the_same_session(tmp_path):
    from photomatagent.cli.chat import run_chat

    await run_chat(
        provider="fake",
        approval="auto",
        goal="first goal",
        log_events=True,
        sessions_dir=tmp_path,
    )
    only_session = next(path for path in sorted(tmp_path.iterdir()) if path.is_dir())
    assert (only_session / SESSION_STATE_FILENAME).is_file()
    prior_events = (only_session / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert any("first goal" in line for line in prior_events)

    await run_chat(
        provider="fake",
        approval="auto",
        goal="second goal",
        resume=only_session.name,
        log_events=True,
        sessions_dir=tmp_path,
    )
    # No second directory: the resumed turn appends to the same trace.
    assert [path for path in sorted(tmp_path.iterdir()) if path.is_dir()] == [only_session]
    after_events = (only_session / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(after_events) > len(prior_events)
    assert any("second goal" in line for line in after_events)
    snapshot = load_session_snapshot(only_session)
    user_goals = [
        message.content
        for message in snapshot.conversation.messages
        if isinstance(message, UserMessage)
    ]
    assert user_goals == ["first goal", "second goal"]
