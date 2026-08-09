"""ScientificTask: a long-running computation handled by a backend."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

TaskStatus = Literal["PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ScientificTask(BaseModel):
    """Conceptual handle for a long-running scientific calculation.

    A future HPC/VASP backend may keep this task alive for minutes or hours;
    nothing in the runtime assumes ``submit`` returns a final result quickly.
    """

    task_id: str = Field(default_factory=lambda: f"task_{uuid4().hex[:12]}")
    backend: str
    status: TaskStatus = "PENDING"
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    result_reference: str = ""
    error: str = ""
