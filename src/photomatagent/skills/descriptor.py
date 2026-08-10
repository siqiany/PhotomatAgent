"""SkillDescriptor: provenance-aware metadata for skills from any source."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SkillDescriptor:
    """Identity and provenance of one skill (native or third-party).

    ``priority`` orders duplicate names across sources: a higher value wins.
    ``source`` and ``license`` are carried for provenance reporting and are
    never used to decide whether a skill is callable.
    """

    name: str
    description: str
    path: Path
    source: str = "native"
    license: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    priority: int = 100
    category: str = ""


@dataclass(frozen=True)
class SkillDiagnostic:
    """A non-fatal problem encountered while scanning one skill source."""

    code: str
    message: str
    source: str = ""
    path: Path | None = None

