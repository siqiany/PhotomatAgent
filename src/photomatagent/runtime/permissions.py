"""Permission decisions and UI-neutral approval handlers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
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
        # New tools must be classified deliberately.  Falling back to ASK
        # keeps newly installed scientific/MCP tools from silently gaining
        # process, network, or filesystem authority.
        default=PermissionDecision.ASK,
    )


class ApprovalScope(str, Enum):
    DEFAULT = "default"
    SESSION = "session"
    ALWAYS = "always"


class ApprovalSettings:
    """Workspace-local persistence for the explicit always-allow switch."""

    def __init__(self, workspace: Path | str) -> None:
        self.path = (
            Path(workspace).expanduser().resolve()
            / ".photomatagent"
            / "settings.json"
        )

    def always_allow(self) -> bool:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return False
        return data.get("approval") == ApprovalScope.ALWAYS.value

    def set_always_allow(self, enabled: bool) -> None:
        data: dict[str, object] = {}
        if self.path.is_file():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data = loaded
            except (OSError, ValueError, TypeError):
                pass
        if enabled:
            data["approval"] = ApprovalScope.ALWAYS.value
        else:
            data.pop("approval", None)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.path)


class SwitchablePermissionPolicy(PermissionPolicy):
    """Delegates to the initial policy unless an explicit override is active."""

    def __init__(
        self,
        base: PermissionPolicy,
        *,
        settings: ApprovalSettings | None = None,
    ) -> None:
        self._base = base
        self._settings = settings
        self._session_allow_all = False

    @property
    def scope(self) -> ApprovalScope:
        if self._settings is not None and self._settings.always_allow():
            return ApprovalScope.ALWAYS
        if self._session_allow_all:
            return ApprovalScope.SESSION
        return ApprovalScope.DEFAULT

    def allow_for_session(self) -> None:
        self._session_allow_all = True

    def allow_always(self) -> None:
        if self._settings is None:
            # A fresh execution scope deliberately has no durable settings.
            # Treat an explicit "always" choice as allow-all only for this
            # runtime instance so it cannot authorize a later episode.
            self._session_allow_all = True
            return
        self._settings.set_always_allow(True)
        self._session_allow_all = False

    def reset(self) -> None:
        self._session_allow_all = False
        if self._settings is not None:
            self._settings.set_always_allow(False)

    async def check(self, tool_name: str, arguments: dict[str, object]) -> PermissionResult:
        scope = self.scope
        if scope is not ApprovalScope.DEFAULT:
            return PermissionResult(
                PermissionDecision.ALLOW, f"explicit {scope.value} allow-all override"
            )
        return await self._base.check(tool_name, arguments)
