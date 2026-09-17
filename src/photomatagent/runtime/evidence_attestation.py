"""Host-owned evidence authority derived from the executed registered tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from photomatagent.scientific.state import (
    DEFAULT_TRUSTED_EVIDENCE_TOOLS,
    EvidenceAttestation,
)
from photomatagent.tools.base import Tool

DEFAULT_TRUSTED_BUILTIN_TOOLS = DEFAULT_TRUSTED_EVIDENCE_TOOLS


@dataclass(frozen=True)
class EvidenceAttestationPolicy:
    """Classify evidence by actual registry origin, never producer payload fields."""

    trusted_builtin_tools: frozenset[str] = DEFAULT_TRUSTED_BUILTIN_TOOLS
    synthetic_test_tools: frozenset[str] = field(default_factory=frozenset)

    def attest(
        self, *, tool: Tool, evidence_id: str, tool_call_id: str
    ) -> EvidenceAttestation:
        name = tool.name
        source = tool.source.strip().casefold()
        blocked = (
            name == "generation"
            or name.startswith("generation.")
            or name == "mock"
            or name.startswith("mock.")
            or source.startswith("mcp:")
        )
        if blocked:
            authority: Literal["observation", "synthetic", "background"] = (
                "background"
            )
            origin: Literal["trusted_builtin", "synthetic_test", "untrusted_tool"] = (
                "untrusted_tool"
            )
        elif name in self.synthetic_test_tools:
            authority = "synthetic"
            origin = "synthetic_test"
        elif name in self.trusted_builtin_tools:
            authority = "observation"
            origin = "trusted_builtin"
        else:
            authority = "background"
            origin = "untrusted_tool"
        return EvidenceAttestation.host_create(
            evidence_id=evidence_id,
            authority=authority,
            origin=origin,
            tool_name=name,
            tool_call_id=tool_call_id,
        )
