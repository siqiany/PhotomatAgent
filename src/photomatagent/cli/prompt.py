"""prompt_toolkit input and CLI-only approval UI."""

from __future__ import annotations

from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.shortcuts import PromptSession

from photomatagent.runtime.permissions import ApprovalRequest


def make_prompt_session() -> PromptSession:
    return PromptSession(history=InMemoryHistory())


class CLIApprovalHandler:
    def __init__(self, session: PromptSession) -> None:
        self.session = session

    async def request_approval(self, request: ApprovalRequest) -> bool:
        answer = await self.session.prompt_async("  [y] Allow once  [n] Deny > ")
        return answer.strip().lower() in {"y", "yes"}


def make_approval_handler(session: PromptSession) -> CLIApprovalHandler:
    return CLIApprovalHandler(session)
