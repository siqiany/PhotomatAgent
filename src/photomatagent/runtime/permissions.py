"""Permission decisions and UI-neutral approval handlers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class PermissionDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass(frozen=True)
class PermissionResult:
    decision: PermissionDecision
    reason: str = ""


@dataclass(frozen=True)
class ApprovalRequest:
    tool_name: str
    arguments: dict[str, object]
    reason: str


class ApprovalHandler(Protocol):
    async def request_approval(self, request: ApprovalRequest) -> bool: ...


class AutoApproveHandler:
    async def request_approval(self, request: ApprovalRequest) -> bool:
        return True


class DenyHandler:
    async def request_approval(self, request: ApprovalRequest) -> bool:
        return False


class PermissionPolicy:
    async def check(self, tool_name: str, arguments: dict[str, object]) -> PermissionResult:
        raise NotImplementedError


class AllowAllPolicy(PermissionPolicy):
    async def check(self, tool_name: str, arguments: dict[str, object]) -> PermissionResult:
        return PermissionResult(PermissionDecision.ALLOW, "allow-all policy")


class DenyAllPolicy(PermissionPolicy):
    async def check(self, tool_name: str, arguments: dict[str, object]) -> PermissionResult:
        return PermissionResult(PermissionDecision.DENY, "deny-all policy")


class AskPolicy(PermissionPolicy):
    async def check(self, tool_name: str, arguments: dict[str, object]) -> PermissionResult:
        return PermissionResult(PermissionDecision.ASK, "ask policy")


class PolicyRule(PermissionPolicy):
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


def default_permission_policy() -> PermissionPolicy:
    """Safe default UX: inspection is allowed; mutation/process tools ask."""
    return PolicyRule(
        rules={
            "read": PermissionDecision.ALLOW,
            "glob": PermissionDecision.ALLOW,
            "grep": PermissionDecision.ALLOW,
            "write": PermissionDecision.ASK,
            "edit": PermissionDecision.ASK,
            "bash": PermissionDecision.ASK,
            "mock.run_calculation": PermissionDecision.ASK,
        },
        default=PermissionDecision.ALLOW,
    )
