"""Lightweight candidate provenance chain (section 50).

Not a knowledge graph: just a parent/child chain recording which generator
produced a candidate from what. Example:

    VAE-F01 -> MatterGen-MG01 -> CHGNet-R01 -> VASP-D01
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class CandidateLineage(BaseModel):
    """One node in a candidate provenance chain."""

    candidate_id: str = Field(default_factory=lambda: f"cand_{uuid4().hex[:10]}")
    parent_candidate_id: str | None = None
    generated_by: str = ""  # vae_formula | mattergen | magus | chgnet | vasp
    generation_parameters: dict[str, Any] = Field(default_factory=dict)
    source_artifacts: list[str] = Field(default_factory=list)
    transformation: str = ""
    validation_status: str = "UNVALIDATED_GENERATED_STRUCTURE"
    created_at: datetime = Field(default_factory=_now)

    def child(
        self,
        *,
        generated_by: str,
        transformation: str,
        generation_parameters: dict[str, Any] | None = None,
        source_artifacts: list[str] | None = None,
    ) -> "CandidateLineage":
        return CandidateLineage(
            parent_candidate_id=self.candidate_id,
            generated_by=generated_by,
            generation_parameters=generation_parameters or {},
            source_artifacts=source_artifacts or [],
            transformation=transformation,
        )

    def to_evidence_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
