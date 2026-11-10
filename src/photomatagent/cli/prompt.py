"""prompt_toolkit input and CLI-only approval UI."""

from __future__ import annotations

import asyncio

from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.shortcuts import PromptSession

from photomatagent.runtime.permissions import ApprovalRequest


def make_prompt_session() -> PromptSession:
    return PromptSession(history=InMemoryHistory())


class CLIApprovalHandler:
    def __init__(self, session: PromptSession) -> None:
        self.session = session

    async def request_approval(self, request: ApprovalRequest) -> bool:
        try:
            answer = await self.session.prompt_async("  [y] Allow once  [n] Deny > ")
        except (KeyboardInterrupt, EOFError) as exc:
            # Convert before the task exits: asyncio treats raw KeyboardInterrupt
            # as process-level shutdown even if the outer UI appears to recover.
            raise asyncio.CancelledError("approval interrupted by user") from exc
        return answer.strip().lower() in {"y", "yes"}


def make_approval_handler(session: PromptSession) -> CLIApprovalHandler:
    return CLIApprovalHandler(session)
