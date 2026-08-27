"""Machine-verifiable scientific targets.

A :class:`TargetSpec` turns a natural-language research goal into a list of
deterministic constraints the loop can check without asking the LLM to
compare numbers. Everything here is pure data + deterministic rules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

ConstraintOperator = Literal["lt", "le", "gt", "ge", "eq", "between"]
ConstraintSeverity = Literal["HARD", "SOFT"]


class ConstraintSpec(BaseModel):
    """One verifiable threshold on a scientific property."""

    property: str
    operator: ConstraintOperator
    value: Any = None
    unit: str = ""
    severity: ConstraintSeverity = "HARD"
    weight: float = Field(default=1.0, ge=0.0)
    description: str = ""


class TargetSpec(BaseModel):
    """Structured, machine-verifiable research target.

    ``goal`` keeps the original natural-language user goal verbatim (it is
    never replaced by the compiled constraints). ``constraints`` are checked
    deterministically; ``objectives`` and ``operating_conditions`` frame
    the search space without being boolean gates.
    """

    goal: str
    constraints: list[ConstraintSpec] = Field(default_factory=list)
    objectives: list[str] = Field(default_factory=list)
    operating_conditions: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def hard_constraints(self) -> list[ConstraintSpec]:
        return [c for c in self.constraints if c.severity == "HARD"]

    def soft_constraints(self) -> list[ConstraintSpec]:
        return [c for c in self.constraints if c.severity == "SOFT"]

    def constraint(self, property_name: str) -> ConstraintSpec | None:
        for constraint in self.constraints:
            if constraint.property == property_name:
                return constraint
        return None


def canonical_lwir_detector_target() -> TargetSpec:
    """8--14 um infrared photodetector demo target (explicit mode A).

    The spectral band and operating temperature are declared as operating
    conditions; the boolean gates are the material band-gap ceiling and the
    device responsivity floor. None of these constraints is satisfied by
    assertion -- each needs evidence before it can PASS.
    """
    return TargetSpec(
        goal=(
            "Design an LWIR photodetector material: spectral range 8-14 um, "
            "band gap <= 0.155 eV, responsivity >= 1 A/W at 77 K."
        ),
        constraints=[
            ConstraintSpec(
                property="band_gap",
                operator="le",
                value=0.155,
                unit="eV",
                severity="HARD",
                description="material band gap must be small enough for 8-14 um detection",
            ),
            ConstraintSpec(
                property="responsivity",
                operator="ge",
                value=1.0,
                unit="A/W",
                severity="HARD",
                description="device responsivity must reach 1 A/W in band",
            ),
        ],
        objectives=[
            "spectral compatibility with the 8-14 um atmospheric window",
            "operable at 77 K (dark-current suppression)",
        ],
        operating_conditions={
            "spectral_range_um": [8.0, 14.0],
            "temperature_k": 77,
        },
        metadata={"demo": "photodetector vertical slice"},
    )


class ConstraintOutcome(BaseModel):
    """Evaluator result for one constraint (evidence-resolved)."""

    property: str
    operator: ConstraintOperator
    observed_value: Any = None
    target_value: Any = None
    unit: str = ""
    severity: ConstraintSeverity = "HARD"
    result: Literal["PASS", "FAIL", "UNKNOWN"] = "UNKNOWN"
    evidence_found: bool = False
    evidence_ids: list[str] = Field(default_factory=list)
    fidelity: str | None = None
    confidence: float = 0.0
    soft_score: float = 1.0
    reason: str = ""


@dataclass(frozen=True)
class ConstraintCheck:
    """Deterministic result of one numeric constraint comparison.

    ``passed`` is ``None`` when the observed value is missing or not
    comparable -- the only legitimate way to get an UNKNOWN outcome.
    """

    passed: bool | None
    detail: str
    soft_score: float = 1.0

    @property
    def evaluable(self) -> bool:
        return self.passed is not None


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_bounds(value: Any) -> tuple[float, float] | None:
    if value is None or isinstance(value, (str, bool)):
        return None
    try:
        low, high = value
    except (TypeError, ValueError):
        return None
    low_n = _as_number(low)
    high_n = _as_number(high)
    if low_n is None or high_n is None:
        return None
    return low_n, high_n


def _relative_margin(observed: float, target: float) -> float:
    """Relative gap between observed and target, used for continuous soft scores."""
    scale = max(abs(target), abs(observed), 1e-12)
    return abs(observed - target) / scale


def evaluate_constraint(
    constraint: ConstraintSpec, observed_value: Any
) -> ConstraintCheck:
    """Deterministically compare ``observed_value`` against the constraint.

    This is the single place that answers "is 0.21 eV <= 0.155 eV" -- never
    the LLM, and never duplicated anywhere else in the loop.
    """
    if observed_value is None:
        return ConstraintCheck(None, "no observed value")
    if isinstance(observed_value, str):
        stripped = observed_value.strip()
        if not stripped:
            return ConstraintCheck(None, "empty observed value")
        if constraint.operator == "eq":
            return ConstraintCheck(
                stripped == str(constraint.value),
                f"{stripped!r} == {constraint.value!r}",
            )
        return ConstraintCheck(None, f"string value not comparable with {constraint.operator}")

    operator = constraint.operator
    if operator == "between":
        bounds = _as_bounds(constraint.value)
        observed = _as_number(observed_value)
        if bounds is None or observed is None:
            return ConstraintCheck(
                None,
                "between requires numeric observed value and [low, high] target",
            )
        low, high = bounds
        passed = low <= observed <= high
        if passed:
            return ConstraintCheck(True, f"{observed} in [{low}, {high}]")
        margin = min(_relative_margin(observed, low), _relative_margin(observed, high))
        return ConstraintCheck(False, f"{observed} outside [{low}, {high}]", max(0.0, 1.0 - margin))

    observed = _as_number(observed_value)
    target = _as_number(constraint.value)
    if observed is None or target is None:
        return ConstraintCheck(
            None, f"numeric comparison requires numeric values, got {observed_value!r}"
        )
    if operator == "lt":
        passed = observed < target
        margin = _relative_margin(observed, target)
    elif operator == "le":
        passed = observed <= target
        margin = _relative_margin(observed, target)
    elif operator == "gt":
        passed = observed > target
        margin = _relative_margin(observed, target)
    elif operator == "ge":
        passed = observed >= target
        margin = _relative_margin(observed, target)
    else:  # eq with a small deterministic relative tolerance
        passed = math.isclose(observed, target, rel_tol=1e-6, abs_tol=1e-9)
        margin = 0.0 if passed else _relative_margin(observed, target)
    detail = f"{observed} {operator} {target}"
    soft_score = 1.0 if passed else max(0.0, 1.0 - margin)
    return ConstraintCheck(passed, detail, soft_score)


class ConstraintViolation(BaseModel):
    """One failed constraint; the basic building block of the feedback loop."""

    property: str
    observed_value: Any = None
    target_value: Any = None
    unit: str = ""
    severity: ConstraintSeverity = "HARD"
    message: str = ""
    evidence_ids: list[str] = Field(default_factory=list)

    @classmethod
    def from_constraint(
        cls,
        constraint: ConstraintSpec,
        *,
        observed_value: Any,
        evidence_ids: list[str] | None = None,
    ) -> "ConstraintViolation":
        return cls(
            property=constraint.property,
            observed_value=observed_value,
            target_value=constraint.value,
            unit=constraint.unit,
            severity=constraint.severity,
            message=(
                f"{constraint.property} = {observed_value} {constraint.unit}"
                f" violates {constraint.operator} {constraint.value}"
            ),
            evidence_ids=list(evidence_ids or []),
        )

    def short(self) -> str:
        return self.message or f"{self.property} constraint violated"