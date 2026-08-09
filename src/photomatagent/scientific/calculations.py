"""CalculationRecord: an immutable log entry for a scientific computation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


CalculationStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


class CalculationRecord(BaseModel):
    """Record of one calculation, independent of how long it ran."""

    id: str = Field(default_factory=lambda: f"calc_{uuid4().hex[:12]}")
    backend: str
    task_type: str
    status: CalculationStatus
    input_reference: dict[str, Any] = Field(default_factory=dict)
    output_reference: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_now)
