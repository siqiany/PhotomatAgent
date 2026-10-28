"""Bounded, cache-aware trigger policy for the advisory Scientific Judge."""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from photomatagent.scientific.loop.judge import JudgeReport

JudgeMode = Literal["off", "on_demand", "every_round"]


class JudgeDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    call: bool
    reason: str
    cache_key: str | None = None


class JudgePolicy(BaseModel):
    """Decide whether to judge, using one finite per-run call budget."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    mode: JudgeMode = "on_demand"
    require_judge: bool = False
    max_calls: int = Field(default=1, ge=0)

    _calls: int = PrivateAttr(default=0)
    _reports: dict[str, JudgeReport] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def required_judge_must_be_enabled(self) -> JudgePolicy:
        if self.require_judge and self.mode == "off":
            raise ValueError("REQUIRED_JUDGE_DISABLED")
        return self

    @property
    def calls_used(self) -> int:
        return self._calls

    def decide(
        self,
        *,
        artifact_sha256: str,
        contract_sha256: str,
        provider: str,
        model: str,
        prompt_version: str,
        schema_version: str,
        user_requested: bool = False,
        non_machine_issue: bool = False,
        stagnated: bool = False,
        required_success: bool = False,
    ) -> JudgeDecision:
        key_payload = {
            "artifact_sha256": artifact_sha256,
            "contract_sha256": contract_sha256,
            "provider": provider,
            "model": model,
            "prompt_version": prompt_version,
            "schema_version": schema_version,
        }
        cache_key = hashlib.sha256(json.dumps(
            key_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")).hexdigest()
        if cache_key in self._reports:
            return JudgeDecision(call=False, reason="cache_hit", cache_key=cache_key)
        if self.mode == "off":
            return JudgeDecision(call=False, reason="mode_off", cache_key=cache_key)
        triggered = self.mode == "every_round" or any((
            user_requested, non_machine_issue, stagnated,
            self.require_judge and required_success,
        ))
        if not triggered:
            return JudgeDecision(call=False, reason="not_needed", cache_key=cache_key)
        if self._calls >= self.max_calls:
            reason = "required_budget_exhausted" if self.require_judge and required_success else "budget_exhausted"
            return JudgeDecision(call=False, reason=reason, cache_key=cache_key)
        self._calls += 1
        if self.require_judge and required_success:
            reason = "required_success_gate"
        elif self.mode == "every_round":
            reason = "every_round"
        else:
            reason = "on_demand_trigger"
        return JudgeDecision(call=True, reason=reason, cache_key=cache_key)

    def remember(self, cache_key: str, report: JudgeReport) -> None:
        if not cache_key:
            raise ValueError("JUDGE_CACHE_KEY_REQUIRED")
        self._reports[cache_key] = report.model_copy(deep=True)

    def cached(self, cache_key: str) -> JudgeReport | None:
        report = self._reports.get(cache_key)
        return report.model_copy(deep=True) if report is not None else None


__all__ = ["JudgeDecision", "JudgeMode", "JudgePolicy"]
