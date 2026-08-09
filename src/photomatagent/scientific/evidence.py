"""Evidence: a single recorded observation with provenance."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Evidence(BaseModel):
    """One piece of evidence (calculation result, literature claim, ...).

    ``type`` is a free-form category ("calculation", "literature",
    "experiment"). ``provenance`` records how it was obtained so claims can
    be traced back to a source.
    """

    id: str = Field(default_factory=lambda: f"ev_{uuid4().hex[:12]}")
    type: str
    source: str
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=_now)
    provenance: dict[str, Any] = Field(default_factory=dict)
