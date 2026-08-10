"""Scientific result contract: ToolObservation + ScientificEvidence + artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from photomatagent.tools.base import ToolResult


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ScientificEvidence(BaseModel):
    """Minimal provenance carrier for one scientific observation.

    Deliberately not an EvidenceGraph: it only records what was measured, from
    where, and how, so downstream claims can be traced back.
    """

    id: str = Field(default_factory=lambda: f"sev_{uuid4().hex[:12]}")
    subject: str
    property: str
    value: Any = None
    unit: str = ""
    source: str = ""
    source_type: Literal[
        "database", "literature", "calculation", "experiment", "derived", "model"
    ] = "calculation"
    method: str = ""
    summary: str = ""
    limitations: str = ""
    provenance: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_now)


class ScientificToolResult(ToolResult):
    """Tool result for scientific capabilities.

    Adds structured evidence and optional artifact references on top of the
    normal ``ToolResult`` contract. The runtime keeps applying
    ``state_updates``; evidence items are appended there automatically so the
    ScientificState records them without a second executor.
    """

    evidence: list[ScientificEvidence] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        if not self.state_updates:
            self.state_updates = list(self.evidence)
        elif not any(isinstance(item, ScientificEvidence) for item in self.state_updates):
            self.state_updates = [*self.state_updates, *self.evidence]

