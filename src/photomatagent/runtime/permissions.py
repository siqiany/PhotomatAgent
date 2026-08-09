"""Permission policies: allow / deny / ask.

The runtime never hard-codes which tools are permitted. It asks the policy
for a decision and, for ``ask``, delegates the interactive prompt to an
injected async approval handler.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum


class PermissionDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass(frozen=True)
class PermissionResult:
    decision: PermissionDecision
    reason: str = ""


class PermissionPolicy:
    """Base class. Subclasses decide per-tool-call permissions."""

    async def check(self, tool_name: str, arguments: dict[str, object]) -> PermissionResult:
        raise NotImplementedError


class AllowAllPolicy(PermissionPolicy):
    async def check(self, tool_name: str, arguments: dict[str, object]) -> PermissionResult:
        return PermissionResult(PermissionDecision.ALLOW, "allow-all policy")


class DenyAllPolicy(PermissionPolicy):
    async def check(self, tool_name: str, arguments: dict[str, object]) -> PermissionResult:
        return PermissionResult(PermissionDecision.DENY, "deny-all policy")


class AskPolicy(PermissionPolicy):
    """Ask the user for every tool call."""

    async def check(self, tool_name: str, arguments: dict[str, object]) -> PermissionResult:
        return PermissionResult(PermissionDecision.ASK, "ask policy")


class PolicyRule(PermissionPolicy):
    """Static allow/deny/ask rules keyed by tool name; default rule applies otherwise."""

    def __init__(
        self,
        rules: dict[str, PermissionDecision] | None = None,
        default: PermissionDecision = PermissionDecision.ALLOW,
    ) -> None:
        self._rules = rules or {}
        self._default = default

    async def check(self, tool_name: str, arguments: dict[str, object]) -> PermissionResult:
        decision = self._rules.get(tool_name, self._default)
        return PermissionResult(decision, f"policy rule: {decision.value}")


ApprovalHandler = Callable[[str, dict[str, object]], Awaitable[bool]]
