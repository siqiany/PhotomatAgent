"""Deterministic capability exposure metadata."""

from __future__ import annotations

from enum import Enum


class ToolExposure(str, Enum):
    """How a registered capability may appear in model context."""

    DIRECT = "direct"
    DEFERRED = "deferred"
    HIDDEN = "hidden"
