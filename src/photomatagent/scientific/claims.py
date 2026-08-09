"""ScientificClaim: an assertion grounded in evidence."""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

ClaimStatus = Literal["proposed", "supported", "contradicted", "rejected"]


class ScientificClaim(BaseModel):
    id: str = Field(default_factory=lambda: f"cl_{uuid4().hex[:12]}")
    statement: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    status: ClaimStatus = "proposed"
