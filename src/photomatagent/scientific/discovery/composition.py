"""Stable, bounded composition identities backed by lazy pymatgen parsing."""

from __future__ import annotations

import hashlib
import json
import math
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import reduce
from typing import Any

MAX_FORMULA_LENGTH = 256
MAX_ELEMENTS = 16
MAX_DENOMINATOR = 10_000
MAX_TOTAL_PARSED_AMOUNT = 1_000_000
MAX_TOTAL_NORMALIZED_AMOUNT = 1_000_000_000
COMPOSITION_ERROR_TOLERANCE = Fraction(1, 100_000_000)

_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_NON_FINITE_TOKEN = re.compile(r"(?:NaN|nan|Inf|inf|Infinity|infinity)")


class CompositionCapabilityError(RuntimeError):
    """Raised when composition canonicalization lacks its parser dependency."""


def _lazy_pymatgen_types() -> tuple[Any, Any]:
    try:
        from pymatgen.core import Composition, Element
    except ImportError as exc:  # pragma: no cover - exercised with an import guard
        raise CompositionCapabilityError(
            "composition normalization requires the pymatgen capability"
        ) from exc
    return Composition, Element


def _validate_numeric_literals(formula: str) -> None:
    if _NON_FINITE_TOKEN.search(formula):
        raise ValueError("composition amounts must be finite")
    for literal in _NUMBER.findall(formula):
        try:
            value = Decimal(literal)
        except InvalidOperation as exc:
            raise ValueError(f"invalid numeric amount {literal!r}") from exc
        if not value.is_finite():
            raise ValueError("composition amounts must be finite")
        if value <= 0:
            raise ValueError("composition amounts must be positive")


def _least_common_multiple(left: int, right: int) -> int:
    return left // math.gcd(left, right) * right


def normalize_composition(formula: str) -> tuple[tuple[str, int], ...]:
    """Return an element-sorted, reduced positive-integer composition."""

    if not isinstance(formula, str):
        raise ValueError("formula must be a string")
    if len(formula) > MAX_FORMULA_LENGTH:
        raise ValueError(f"formula exceeds {MAX_FORMULA_LENGTH} characters")
    formula = formula.strip()
    if not formula:
        raise ValueError("formula must not be empty")
    _validate_numeric_literals(formula)

    Composition, Element = _lazy_pymatgen_types()
    try:
        composition = Composition(formula)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid composition formula {formula!r}: {exc}") from exc

    elements = composition.elements
    if not elements:
        raise ValueError("formula must contain at least one element")
    if len(elements) > MAX_ELEMENTS:
        raise ValueError(f"formula contains more than {MAX_ELEMENTS} elements")
    if any(not isinstance(element, Element) for element in elements):
        raise ValueError("formula contains an unknown or unsupported element")

    raw_amounts = composition.get_el_amt_dict()
    total_amount = sum(float(amount) for amount in raw_amounts.values())
    if not math.isfinite(total_amount):
        raise ValueError("composition amounts must be finite")
    if total_amount <= 0 or total_amount > MAX_TOTAL_PARSED_AMOUNT:
        raise ValueError("total composition amount is outside the supported range")

    rational_amounts: dict[str, Fraction] = {}
    for symbol, raw_amount in raw_amounts.items():
        numeric_amount = float(raw_amount)
        if not math.isfinite(numeric_amount):
            raise ValueError("composition amounts must be finite")
        if numeric_amount <= 0:
            raise ValueError("composition amounts must be positive")
        exact = Fraction(Decimal(str(numeric_amount)))
        bounded = exact.limit_denominator(MAX_DENOMINATOR)
        if abs(exact - bounded) > COMPOSITION_ERROR_TOLERANCE:
            raise ValueError(
                f"composition amount for {symbol} cannot be represented with "
                f"denominator <= {MAX_DENOMINATOR} within 1e-8"
            )
        rational_amounts[symbol] = bounded

    common_denominator = reduce(
        _least_common_multiple,
        (amount.denominator for amount in rational_amounts.values()),
        1,
    )
    integer_amounts = {
        symbol: amount.numerator * (common_denominator // amount.denominator)
        for symbol, amount in rational_amounts.items()
    }
    divisor = reduce(math.gcd, integer_amounts.values())
    normalized = tuple(
        sorted((symbol, amount // divisor) for symbol, amount in integer_amounts.items())
    )
    if sum(amount for _, amount in normalized) > MAX_TOTAL_NORMALIZED_AMOUNT:
        raise ValueError("normalized composition amount exceeds the supported range")
    return normalized


def composition_key(formula: str) -> str:
    """Return the deterministic SHA256 identity of a normalized composition."""

    normalized = normalize_composition(formula)
    payload = json.dumps(normalized, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
