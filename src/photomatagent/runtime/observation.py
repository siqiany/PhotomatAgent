"""Bound raw tool results before they enter model-visible history."""

from __future__ import annotations

from pydantic import BaseModel, Field

from photomatagent.redaction import redact_text
from photomatagent.tools.surface import estimate_tokens


class ObservationPolicyConfig(BaseModel):
    default_max_chars: int = Field(default=12_000, ge=256)
    read_max_chars: int = Field(default=16_000, ge=256)
    grep_max_chars: int = Field(default=12_000, ge=256)
    glob_max_chars: int = Field(default=8_000, ge=256)
    bash_max_chars: int = Field(default=16_000, ge=256)


class ToolObservation(BaseModel):
    content: str
    truncated: bool
    original_chars: int
    delivered_chars: int
    estimated_original_tokens: int
    estimated_delivered_tokens: int
    redacted: bool = False


class ObservationPolicy:
    """Deterministic, explicit output budgeting with tool-specific strategies."""

    def __init__(self, config: ObservationPolicyConfig | None = None) -> None:
        self.config = config or ObservationPolicyConfig()

    def apply(self, tool_name: str, output: str) -> ToolObservation:
        limit = self._limit(tool_name)
        original_chars = len(output)
        safe_output = redact_text(output)
        redacted = safe_output != output
        if len(safe_output) <= limit:
            return ToolObservation(
                content=safe_output,
                truncated=False,
                original_chars=original_chars,
                delivered_chars=len(safe_output),
                estimated_original_tokens=estimate_tokens(original_chars),
                estimated_delivered_tokens=estimate_tokens(len(safe_output)),
                redacted=redacted,
            )
        marker = (
            f"\n[output truncated: original {original_chars} chars, "
            f"~{estimate_tokens(original_chars)} estimated tokens; "
            "request a narrower range/query to continue]\n"
        )
        available = max(0, limit - len(marker))
        if tool_name == "bash":
            head = available * 2 // 3
            tail = available - head
            content = safe_output[:head] + marker + safe_output[-tail:]
        else:
            content = safe_output[:available] + marker
        return ToolObservation(
            content=content,
            truncated=True,
            original_chars=original_chars,
            delivered_chars=len(content),
            estimated_original_tokens=estimate_tokens(original_chars),
            estimated_delivered_tokens=estimate_tokens(len(content)),
            redacted=redacted,
        )

    def _limit(self, tool_name: str) -> int:
        return {
            "read": self.config.read_max_chars,
            "grep": self.config.grep_max_chars,
            "glob": self.config.glob_max_chars,
            "bash": self.config.bash_max_chars,
        }.get(tool_name, self.config.default_max_chars)
