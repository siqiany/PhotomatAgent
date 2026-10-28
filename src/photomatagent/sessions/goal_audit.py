"""Read-only audit of a session's (possibly overwritten) research goal.

A continuation run can legitimately replace the active subtask, but it must
never silently replace the durable research objective.  When an older session
was written before that separation existed, the active goal may have been
overwritten and cannot be recovered automatically.  This module only *reports*
the sources it can see; it never rewrites the session or the manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from photomatagent.models.types import AssistantMessage, UserMessage
from photomatagent.sessions.store import SessionSnapshot, load_session_snapshot


class GoalSource:
    """One candidate goal source with its provenance."""

    def __init__(self, kind: str, text: str, origin: str) -> None:
        self.kind = kind
        self.text = text
        self.origin = origin

    def public_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "text": self.text, "origin": self.origin}


def audit_goal_history(session_dir: Path | str) -> dict[str, Any]:
    """Return every goal source found in one persisted session, read-only.

    Sources reported (when present):

    * ``durable_goal`` -- ``ScientificState.goal``
    * ``active_subtask`` -- the most recent instruction
    * ``first_user_message`` -- the original user request in the transcript
    * ``later_user_messages`` -- subsequent explicit instructions
    * ``compaction_goal`` -- the goal captured by context compaction, when the
      engine snapshot carries one

    ``status`` is ``UNIQUE`` only when the durable goal and the active subtask
    agree; otherwise it is ``NEEDS_REVIEW`` and the caller must not assume the
    active subtask is the research objective.
    """
    snapshot = load_session_snapshot(session_dir)
    return audit_goal_snapshot(snapshot)


def audit_goal_snapshot(snapshot: SessionSnapshot) -> dict[str, Any]:
    """Same audit as :func:`audit_goal_history` but from a loaded snapshot."""
    scientific = snapshot.scientific
    sources: list[GoalSource] = []
    if scientific.goal:
        sources.append(GoalSource("durable_goal", scientific.goal, "scientific_state"))
    if scientific.active_subtask and scientific.active_subtask != scientific.goal:
        sources.append(
            GoalSource(
                "active_subtask",
                scientific.active_subtask,
                "scientific_state",
            )
        )
    user_messages = [
        message
        for message in snapshot.conversation.messages
        if isinstance(message, UserMessage)
    ]
    if user_messages:
        sources.append(
            GoalSource(
                "first_user_message",
                user_messages[0].content,
                "conversation.messages[0]",
            )
        )
    for index, message in enumerate(user_messages[1:], start=1):
        sources.append(
            GoalSource(
                "later_user_message",
                message.content,
                f"conversation.messages[{index}]",
            )
        )
    compaction_goal = _compaction_goal(snapshot)
    if compaction_goal:
        sources.append(
            GoalSource("compaction_goal", compaction_goal, "engine.compaction")
        )

    durable = next(
        (item.text for item in sources if item.kind == "durable_goal"), ""
    )
    active = scientific.active_subtask or ""
    first_user = next(
        (item.text for item in sources if item.kind == "first_user_message"), ""
    )
    if not sources:
        status = "NEEDS_REVIEW"
        reason = "no goal source is available in this session"
    elif not durable:
        status = "NEEDS_REVIEW"
        reason = (
            "no durable goal was persisted; the first user message is the only "
            "objective evidence"
        )
    elif first_user and durable != first_user:
        status = "NEEDS_REVIEW"
        reason = (
            "the persisted durable goal differs from the original user message, "
            "which is the historical overwrite signature; confirm the real "
            "research objective before continuing"
        )
    elif active and durable != active:
        status = "NEEDS_REVIEW"
        reason = (
            "the active subtask differs from the durable goal; do not treat the "
            "subtask as the research objective"
        )
    else:
        status = "UNIQUE"
        reason = "the durable goal matches the original user message"
    return {
        "status": status,
        "reason": reason,
        "durable_goal": durable or "UNKNOWN",
        "active_subtask": active or "UNKNOWN",
        "sources": [item.public_dict() for item in sources],
        "later_instructions": [
            item.public_dict()
            for item in sources
            if item.kind == "later_user_message"
        ],
        "explicit_task_switches": [
            item.public_dict()
            for item in sources
            if is_likely_task_switch(item.text)
        ],
        "read_only": True,
        "note": (
            "goal audit reads persisted facts only; it never rewrites the "
            "session, the manifest, or the job registry"
        ),
    }


def is_likely_task_switch(text: str) -> bool:
    """True only for an explicit "start a different task" instruction.

    Continuation/refinement messages (for example "继续完成之前的VASP验证环节")
    are deliberately *not* switches: the runtime keeps the durable goal for
    them, and the audit must not misreport them as objective replacements.
    """
    from photomatagent.runtime.loop import _is_explicit_task_switch

    return _is_explicit_task_switch(text)


def _compaction_goal(snapshot: SessionSnapshot) -> str:
    """Return the goal captured by context compaction, when persisted."""
    engine = snapshot.engine
    compaction = getattr(engine, "compaction_state", None)
    goal = getattr(compaction, "goal", "")
    return goal.strip() if isinstance(goal, str) else ""


def _assistant_texts(snapshot: SessionSnapshot) -> list[str]:
    return [
        message.text
        for message in snapshot.conversation.messages
        if isinstance(message, AssistantMessage)
    ]
