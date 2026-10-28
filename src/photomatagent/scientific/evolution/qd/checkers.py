"""Registered, deterministic reference checkers for frozen QD evaluations.

Checker configuration is data validated by this module. Uploaded artifacts can
select a registered checker, but they cannot provide executable checker code.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class NumericFieldRule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    value_path: str = Field(min_length=1, max_length=300)
    unit_path: str = Field(min_length=1, max_length=300)
    units: str = Field(min_length=1, max_length=100)
    absolute_tolerance: float = Field(default=0.0, ge=0.0)
    relative_tolerance: float = Field(default=0.0, ge=0.0)

    @field_validator("value_path", "unit_path")
    @classmethod
    def valid_pointer(cls, value: str) -> str:
        if not value.startswith("/") or re.search(r"~(?![01])", value):
            raise ValueError("CHECKER_JSON_POINTER_INVALID")
        return value

    @field_validator("units")
    @classmethod
    def nonblank_units(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("CHECKER_UNITS_REQUIRED")
        return value.strip()


class EvaluationCheckerSpec(BaseModel):
    """A versioned checker ID plus finite JSON-only configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checker_id: Literal["json_exact_v1", "json_numeric_tolerance_v1"]
    version: Literal["1"]
    required_fields: tuple[str, ...] = ()
    numeric_fields: dict[str, NumericFieldRule] = Field(default_factory=dict, max_length=32)

    @field_validator("required_fields")
    @classmethod
    def valid_required_fields(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) > 64 or len(values) != len(set(values)):
            raise ValueError("CHECKER_REQUIRED_FIELDS_INVALID")
        for value in values:
            if not value.startswith("/") or re.search(r"~(?![01])", value):
                raise ValueError("CHECKER_JSON_POINTER_INVALID")
        return values

    @field_validator("numeric_fields")
    @classmethod
    def valid_numeric_field_names(cls, values: dict[str, NumericFieldRule]) -> dict[str, NumericFieldRule]:
        if any(not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", name) for name in values):
            raise ValueError("CHECKER_NUMERIC_FIELDS_INVALID")
        return values

    @model_validator(mode="after")
    def checker_configuration_matches_id(self) -> EvaluationCheckerSpec:
        if self.checker_id == "json_exact_v1" and self.numeric_fields:
            raise ValueError("CHECKER_CONFIGURATION_INVALID")
        if self.checker_id == "json_numeric_tolerance_v1" and not self.numeric_fields:
            raise ValueError("CHECKER_NUMERIC_FIELDS_REQUIRED")
        return self


class CheckedNumericField(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    name: str
    observed: float
    reference: float
    units: str
    absolute_error: float
    allowed_error: float
    verdict: Literal["PASS", "FAIL"]


class EvaluationCheckerResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    checker_id: str
    checker_version: str
    verdict: Literal["PASS", "FAIL", "UNKNOWN"]
    reason_codes: tuple[str, ...]
    evidence_refs: tuple[str, ...] = ()
    checked: tuple[CheckedNumericField, ...] = ()


def check_reference(
    spec: EvaluationCheckerSpec,
    output: bytes,
    reference: bytes,
) -> EvaluationCheckerResult:
    """Check candidate JSON against reference bytes using a pinned checker."""
    output_digest = hashlib.sha256(output).hexdigest()
    reference_digest = hashlib.sha256(reference).hexdigest()
    evidence = (f"sha256:{output_digest}", f"sha256:{reference_digest}")
    expected, reference_error = _parse(reference)
    if reference_error:
        return _result(spec, "UNKNOWN", ("REFERENCE_JSON_UNREADABLE",), evidence)
    actual, output_error = _parse(output)
    if output_error:
        return _result(spec, "FAIL", ("OUTPUT_JSON_INVALID",), evidence)
    if spec.checker_id == "json_exact_v1":
        return _result(
            spec,
            "PASS" if _json_equal(actual, expected) else "FAIL",
            ("JSON_EXACT_MATCH" if _json_equal(actual, expected) else "JSON_EXACT_MISMATCH",),
            evidence,
        )

    missing = [
        path for path in spec.required_fields
        if _pointer(actual, path) is _MISSING or _pointer(expected, path) is _MISSING
    ]
    field_results: list[CheckedNumericField] = []
    missing_numeric: list[str] = []
    failed_units: list[str] = []
    for name, rule in spec.numeric_fields.items():
        observed = _pointer(actual, rule.value_path)
        reference_value = _pointer(expected, rule.value_path)
        observed_unit = _pointer(actual, rule.unit_path)
        reference_unit = _pointer(expected, rule.unit_path)
        if (observed is _MISSING or reference_value is _MISSING
                or observed_unit is _MISSING or reference_unit is _MISSING):
            missing_numeric.append(name)
            continue
        if observed_unit != rule.units or reference_unit != rule.units:
            failed_units.append(name)
            continue
        if not _finite(observed) or not _finite(reference_value):
            missing_numeric.append(name)
            continue
        absolute_error = abs(float(observed) - float(reference_value))
        allowed_error = rule.absolute_tolerance + rule.relative_tolerance * abs(float(reference_value))
        field_results.append(CheckedNumericField(
            name=name,
            observed=float(observed),
            reference=float(reference_value),
            units=rule.units,
            absolute_error=absolute_error,
            allowed_error=allowed_error,
            verdict="PASS" if absolute_error <= allowed_error else "FAIL",
        ))
    reasons: list[str] = []
    if missing:
        reasons.append("REQUIRED_FIELD_MISSING")
    if missing_numeric:
        reasons.append("NUMERIC_EVIDENCE_MISSING")
    if failed_units:
        reasons.append("UNIT_MISMATCH")
    if any(item.verdict == "FAIL" for item in field_results):
        reasons.append("TOLERANCE_EXCEEDED")
    if missing or missing_numeric:
        verdict: Literal["PASS", "FAIL", "UNKNOWN"] = "UNKNOWN"
    elif failed_units or any(item.verdict == "FAIL" for item in field_results):
        verdict = "FAIL"
    else:
        verdict = "PASS"
        reasons.append("NUMERIC_TOLERANCE_WITHIN_LIMIT")
    return _result(spec, verdict, tuple(reasons), evidence, tuple(field_results))


_MISSING = object()


def _parse(payload: bytes) -> tuple[Any, bool]:
    try:
        return json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        ), False
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return None, True


def _pointer(document: Any, pointer: str) -> Any:
    current = document
    for encoded in pointer.split("/")[1:]:
        key = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping) and key in current:
            current = current[key]
        elif isinstance(current, list) and key.isdecimal() and (key == "0" or not key.startswith("0")):
            try:
                current = current[int(key)]
            except IndexError:
                return _MISSING
        else:
            return _MISSING
    return current


def _finite(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_json_equal(left[key], right[key]) for key in right)
    if isinstance(left, list):
        return len(left) == len(right) and all(_json_equal(a, b) for a, b in zip(left, right))
    return left == right


def _result(
    spec: EvaluationCheckerSpec,
    verdict: Literal["PASS", "FAIL", "UNKNOWN"],
    reason_codes: tuple[str, ...],
    evidence_refs: tuple[str, ...],
    checked: tuple[CheckedNumericField, ...] = (),
) -> EvaluationCheckerResult:
    return EvaluationCheckerResult(
        checker_id=spec.checker_id,
        checker_version=spec.version,
        verdict=verdict,
        reason_codes=reason_codes,
        evidence_refs=evidence_refs,
        checked=checked,
    )


__all__ = [
    "CheckedNumericField", "EvaluationCheckerResult", "EvaluationCheckerSpec",
    "NumericFieldRule", "check_reference",
]
