"""Dependency probing and capability pack contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CapabilityStatus(str, Enum):
    """Runtime state of one scientific capability pack."""

    AVAILABLE = "AVAILABLE"
    MISSING_DEPENDENCY = "MISSING_DEPENDENCY"
    UNCONFIGURED = "UNCONFIGURED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of one dependency probe."""

    status: CapabilityStatus
    detail: str = ""
    version: str = ""


class CapabilityInfo(BaseModel):
    """Stable status report entry for one pack."""

    name: str
    status: CapabilityStatus
    detail: str = ""
    version: str = ""
    tools: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CapabilityPack:
    """A named group of deferred scientific tools with a dependency probe.

    ``probe()`` must never raise: it returns a ``ProbeResult`` describing
    AVAILABLE / MISSING_DEPENDENCY / UNCONFIGURED / ERROR. ``tools()`` returns
    only the tools whose dependencies are present; metadata-only tools (e.g.
    ``<pack>.capabilities``) are always returned.
    """

    name: str
    description: str = ""
    execution_mode: str = "local"  # local | subprocess | mcp/scnet | mcp | ...
    backend_name: str = ""  # SCNet | local | isolated env | ...

    def probe(self) -> ProbeResult:
        raise NotImplementedError

    def tools(self) -> list[Any]:
        raise NotImplementedError

    def info(self) -> CapabilityInfo:
        result = self.probe()
        return CapabilityInfo(
            name=self.name,
            status=result.status,
            detail=result.detail,
            version=result.version,
            tools=sorted(tool.name for tool in self.tools()),
            metadata={
                "execution_mode": self.execution_mode,
                "backend": self.backend_name,
            },
        )
