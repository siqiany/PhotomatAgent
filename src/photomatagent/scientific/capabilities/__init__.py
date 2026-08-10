"""Scientific capability packs built on the existing ToolRegistry.

Packs do not introduce a second executor: every capability is a normal
``Tool`` with ``exposure = DEFERRED`` and a pack namespace, discovered through
``tool_search`` and invoked through ``tool_call``. Dependencies are probed
without ever failing agent startup.
"""

from photomatagent.scientific.capabilities.base import (
    CapabilityInfo,
    CapabilityPack,
    CapabilityStatus,
    ProbeResult,
)
from photomatagent.scientific.capabilities.contracts import (
    ScientificEvidence,
    ScientificToolResult,
)
from photomatagent.scientific.capabilities.registry import build_scientific_tools
from photomatagent.scientific.capabilities.status import probe_all_capabilities

__all__ = [
    "CapabilityInfo",
    "CapabilityPack",
    "CapabilityStatus",
    "ProbeResult",
    "ScientificEvidence",
    "ScientificToolResult",
    "build_scientific_tools",
    "probe_all_capabilities",
]

