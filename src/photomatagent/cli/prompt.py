"""prompt_toolkit-based input helpers."""

from __future__ import annotations

import json

from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.shortcuts import PromptSession

from photomatagent.runtime.permissions import ApprovalHandler


def make_prompt_session() -> PromptSession:
    return PromptSession(history=InMemoryHistory())


def make_approval_handler(session: PromptSession) -> ApprovalHandler:
    """Ask the user y/n for each tool call the policy marks 'ask'."""

    async def handler(tool_name: str, arguments: dict) -> bool:
        args = json.dumps(arguments, ensure_ascii=False)
        answer = await session.prompt_async(f"Approve {tool_name} {args}? [y/N] ")
        return answer.strip().lower() in {"y", "yes"}

    return handler
