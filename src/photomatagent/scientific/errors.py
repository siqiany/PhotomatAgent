"""Typed scientific failures for no-hallucination tool contracts.

Every deterministic scientific tool must return one of these typed outcomes
instead of a vague exception, so the model can see *cannot compute, because
..., required inputs ...*. Tools surface them as structured
``ScientificToolResult`` payloads; the classes below are the programmatic
equivalents for solver code that raises.
"""

from __future__ import annotations

from typing import Any


class MissingScientificPrerequisite(ValueError):
    """Required inputs/parameters are absent or unsourced."""

    def __init__(self, message: str, missing: list[str] | None = None) -> None:
        super().__init__(message)
        self.missing = list(missing or [])


class UnsupportedScientificRegime(ValueError):
    """Inputs are valid but outside the model's stated validity range."""


class InsufficientParameterEvidence(ValueError):
    """Parameters exist but lack source/provenance required by policy."""


class ExternalSolverUnavailable(RuntimeError):
    """An external solver (kdotpy, DFT, GPAW, ...) is not reachable."""


def prerequisite_failure(
    message: str,
    missing: list[str] | None = None,
    *,
    tool: str = "",
    evidence: list[Any] | None = None,
) -> dict[str, Any]:
    """Build the structured failure payload carried by tool results."""
    return {
        "error_type": "missing_prerequisites",
        "message": message,
        "missing": list(missing or []),
        "tool": tool,
        "hint": "provide the listed inputs with sources, or use a "
        "higher-fidelity solver that can supply them",
    }
