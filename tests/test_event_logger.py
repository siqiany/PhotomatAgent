from __future__ import annotations

import pytest

from photomatagent.logging.event_logger import EventLogger
from photomatagent.runtime.events import LoopCompleted, LoopStarted, ToolCompleted


@pytest.mark.asyncio
async def test_events_roundtrip_via_jsonl(tmp_path):
    logger = EventLogger(tmp_path)
    await logger.log(LoopStarted(goal="hello"))
    await logger.log(ToolCompleted(tool_name="echo", tool_call_id="c1", output="hi"))
    await logger.log(LoopCompleted(iterations=1, reason="final_response"))

    events = logger.read_events()
    assert [e.kind for e in events] == [
        "loop_started",
        "tool_completed",
        "loop_completed",
    ]
    assert events[0].goal == "hello"
    assert events[2].reason == "final_response"


@pytest.mark.asyncio
async def test_logger_creates_session_dir(tmp_path):
    logger = EventLogger(tmp_path)
    assert logger.events_path.exists()
    assert logger.session_dir.parent == tmp_path
