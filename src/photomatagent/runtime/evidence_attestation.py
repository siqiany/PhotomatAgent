"""Host-owned evidence authority derived from the executed registered tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from photomatagent.scientific.state import (
    DEFAULT_TRUSTED_EVIDENCE_TOOLS,
    EvidenceAttestation,
)
from photomatagent.tools.base import Tool

if TYPE_CHECKING:
    from photomatagent.scientific.state import ScientificState

DEFAULT_TRUSTED_BUILTIN_TOOLS = DEFAULT_TRUSTED_EVIDENCE_TOOLS


class _RuntimeEvidenceAuthority:
    """Host-internal capability for one runtime's nonpersistent authority ledger.

    Host application code is trusted. Model output, tool state-update payloads,
    and serialized snapshots never receive this object or its capability.
    """

    def __init__(self) -> None:
        self.__capability = object()

    def bind(self, state: ScientificState, *, replace: bool = False) -> None:
        state._bind_runtime_authority(self.__capability, replace=replace)

    def attest(
        self, state: ScientificState, attestation: EvidenceAttestation
    ) -> EvidenceAttestation:
        return state._attest_evidence(attestation, capability=self.__capability)

    def copy_ledger(
        self, source: ScientificState, destination: ScientificState
    ) -> None:
        destination._copy_runtime_attestations_from(
            source, capability=self.__capability
        )


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
        return EvidenceAttestation(
            evidence_id=evidence_id,
            authority=authority,
            origin=origin,
            tool_name=name,
            tool_call_id=tool_call_id,
        )
