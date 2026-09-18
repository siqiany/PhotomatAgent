"""Pure, bounded crystal-structure construction operations.

The functions in this module operate on pymatgen objects supplied by the
caller. They validate the complete request before copying or mutating any
structure, and never write files or alter scientific state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import comb
from typing import Any, Iterable

from photomatagent.scientific.capabilities.structure.construction_models import (
    ConstructionLimits,
    OrderingRequest,
    SiteReplacement,
)
from photomatagent.scientific.discovery.composition import composition_key


class StructureConstructionError(ValueError):
    """A deterministic construction failure with a model-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class OrderingEnumeration(list[Any]):
    """List-compatible bounded enumeration result with audit counters."""

    total: int = 0
    scanned: int = 0
    discarded: int = 0
    truncated: bool = False
    matcher: dict[str, float] = field(
        default_factory=lambda: {"ltol": 0.2, "stol": 0.3, "angle_tol": 5}
    )

    def __init__(
        self,
        values: Iterable[Any] = (),
        *,
        total: int = 0,
        scanned: int = 0,
        discarded: int = 0,
        truncated: bool = False,
        matcher: dict[str, float] | None = None,
    ) -> None:
        list.__init__(self, values)
        self.total = total
        self.scanned = scanned
        self.discarded = discarded
        self.truncated = truncated
        self.matcher = matcher or {"ltol": 0.2, "stol": 0.3, "angle_tol": 5}


def _validate_ordered_structure(structure: Any) -> None:
    if len(structure) == 0:
        raise StructureConstructionError(
            "EMPTY_STRUCTURE", "construction requires at least one input site"
        )
    if not getattr(structure, "is_ordered", True):
        raise StructureConstructionError(
            "PARTIAL_OCCUPANCY_UNSUPPORTED",
            "disordered structures are unsupported by construction operations",
        )


def _validated_scaling(scaling: tuple[int, int, int] | Iterable[int]) -> tuple[int, int, int]:
    values = tuple(scaling)
    if len(values) != 3 or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values):
        raise StructureConstructionError(
            "INVALID_SCALING", "scaling must contain three positive integers"
        )
    return values  # type: ignore[return-value]


def make_supercell(
    structure: Any,
    scaling: tuple[int, int, int],
    limits: ConstructionLimits,
) -> Any:
    """Return a bounded supercell while leaving ``structure`` unchanged."""

    _validate_ordered_structure(structure)
    factors = _validated_scaling(scaling)
    requested_atoms = len(structure) * factors[0] * factors[1] * factors[2]
    if requested_atoms > limits.max_atoms:
        raise StructureConstructionError(
            "ATOM_LIMIT_EXCEEDED",
            f"supercell would contain {requested_atoms} atoms; limit is {limits.max_atoms}",
        )

    # pymatgen's make_supercell mutates its receiver. Copy only after all
    # validation succeeds so failed operations cannot alter caller state.
    result = structure.copy()
    try:
        result.make_supercell(factors)
    except OverflowError as exc:
        raise StructureConstructionError(
            "SCALING_LIMIT_EXCEEDED",
            "supercell scaling exceeds the supported integer bound",
        ) from exc
    if len(result) > limits.max_atoms:
        raise StructureConstructionError(
            "ATOM_LIMIT_EXCEEDED",
            f"supercell contains {len(result)} atoms; limit is {limits.max_atoms}",
        )
    return result


def substitute_sites(
    structure: Any,
    replacements: list[SiteReplacement],
    expected_formula: str,
) -> Any:
    """Return a copy with explicitly selected host sites substituted.

    Every index and host species is checked against the original input before
    the copy is changed. Formula comparison uses the shared canonical
    composition identity, so equivalent fractional and reduced formulas agree.
    """

    _validate_ordered_structure(structure)
    try:
        replacements = [
            item if isinstance(item, SiteReplacement) else SiteReplacement.model_validate(item)
            for item in replacements
        ]
    except Exception as exc:
        raise StructureConstructionError("INVALID_REPLACEMENTS", str(exc)) from exc
    if not replacements:
        raise StructureConstructionError("INVALID_REPLACEMENTS", "at least one replacement is required")

    seen: set[int] = set()
    for replacement in replacements:
        index = replacement.index
        if index in seen:
            raise StructureConstructionError(
                "DUPLICATE_SITE_INDEX", f"site index {index} occurs more than once"
            )
        if not 0 <= index < len(structure):
            raise StructureConstructionError(
                "SITE_INDEX_OUT_OF_RANGE",
                f"site index {index} is outside the input structure (0..{len(structure) - 1})",
            )
        actual_element = structure[index].specie.symbol
        if actual_element != replacement.from_element:
            raise StructureConstructionError(
                "HOST_ELEMENT_MISMATCH",
                f"input site {index} is {actual_element}, expected {replacement.from_element}",
            )
        seen.add(index)

    try:
        expected = composition_key(expected_formula)
    except (TypeError, ValueError) as exc:
        raise StructureConstructionError(
            "INVALID_EXPECTED_FORMULA", str(exc)
        ) from exc

    result = structure.copy()
    try:
        for replacement in replacements:
            result.replace(replacement.index, replacement.to_element)
    except (TypeError, ValueError, IndexError, KeyError) as exc:
        raise StructureConstructionError(
            "INVALID_REPLACEMENT", f"replacement could not be applied: {exc}"
        ) from exc
    try:
        actual = composition_key(result.composition.formula)
    except (TypeError, ValueError) as exc:
        raise StructureConstructionError(
            "INVALID_CONSTRUCTED_COMPOSITION", str(exc)
        ) from exc
    if actual != expected:
        raise StructureConstructionError(
            "COMPOSITION_MISMATCH",
            f"constructed composition {result.composition.reduced_formula} does not match expected_formula {expected_formula}",
        )
    return result


def ordering_count(site_count: int, replacement_count: int) -> int:
    """Return the number of fixed-size site combinations after validation."""

    if (
        isinstance(site_count, bool)
        or isinstance(replacement_count, bool)
        or not isinstance(site_count, int)
        or not isinstance(replacement_count, int)
        or site_count < 1
        or replacement_count < 1
        or replacement_count > site_count
    ):
        raise ValueError("replacement_count must be between 1 and site_count")
    return comb(site_count, replacement_count)


def enumerate_orderings(
    structure: Any,
    request: OrderingRequest,
    limits: ConstructionLimits,
) -> OrderingEnumeration:
    """Enumerate bounded fixed-count substitutions in deterministic index order.

    The complete combination count is computed before any structure copy.  The
    output cap is applied after exact hash and fixed-tolerance structure-match
    deduplication; a capped result is explicitly marked non-exhaustive.
    """

    _validate_ordered_structure(structure)
    if not isinstance(request, OrderingRequest):
        try:
            request = OrderingRequest.model_validate(request)
        except Exception as exc:
            raise StructureConstructionError("INVALID_ORDERING", str(exc)) from exc
    eligible = tuple(sorted(request.eligible_indices))
    if len(eligible) != len(set(eligible)) or any(
        index < 0 or index >= len(structure) for index in eligible
    ):
        raise StructureConstructionError(
            "INVALID_ORDERING", "eligible_indices must be unique input-structure indices"
        )
    for index in eligible:
        if structure[index].specie.symbol != request.from_element:
            raise StructureConstructionError(
                "HOST_ELEMENT_MISMATCH",
                f"input site {index} is {structure[index].specie.symbol}, expected {request.from_element}",
            )
    try:
        from pymatgen.core import Composition, Element

        Element(request.from_element)
        Element(request.to_element)
        expected_composition = composition_key(request.expected_formula)
        amounts = structure.composition.get_el_amt_dict()
        amounts[request.from_element] = amounts.get(request.from_element, 0.0) - request.replacement_count
        amounts[request.to_element] = amounts.get(request.to_element, 0.0) + request.replacement_count
        constructed_composition = composition_key(Composition(amounts).formula)
    except (TypeError, ValueError, KeyError) as exc:
        raise StructureConstructionError("INVALID_ORDERING", str(exc)) from exc
    if constructed_composition != expected_composition:
        raise StructureConstructionError(
            "COMPOSITION_MISMATCH",
            "expected_formula does not match the fixed replacement count",
        )
    total = ordering_count(len(eligible), request.replacement_count)
    if total > limits.max_raw_configurations:
        raise StructureConstructionError(
            "ENUMERATION_LIMIT_EXCEEDED",
            f"ordering request has {total} raw configurations; limit is {limits.max_raw_configurations}",
        )

    # Imports remain lazy so a missing optional pymatgen installation keeps the
    # base runtime and capability discovery available.
    try:
        from photomatagent.scientific.capabilities.structure.artifacts import structure_hash
        from pymatgen.analysis.structure_matcher import StructureMatcher
    except ImportError as exc:
        raise StructureConstructionError(
            "MISSING_DEPENDENCY", "pymatgen is required for ordering enumeration"
        ) from exc
    matcher_parameters = {"ltol": 0.2, "stol": 0.3, "angle_tol": 5}
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    outputs: list[Any] = []
    hashes: set[str] = set()
    scanned = 0
    discarded = 0
    for selected in combinations(eligible, request.replacement_count):
        scanned += 1
        replacements = [
            SiteReplacement(
                index=index,
                from_element=request.from_element,
                to_element=request.to_element,
            )
            for index in selected
        ]
        try:
            candidate = substitute_sites(structure, replacements, request.expected_formula)
            candidate_hash = structure_hash(candidate)
            if candidate_hash in hashes or any(
                matcher.fit(candidate, existing) for existing in outputs
            ):
                discarded += 1
                continue
        except StructureConstructionError:
            discarded += 1
            continue
        outputs.append(candidate)
        hashes.add(candidate_hash)
        if len(outputs) >= limits.max_outputs and scanned < total:
            break
    truncated = scanned < total
    return OrderingEnumeration(
        outputs,
        total=total,
        scanned=scanned,
        discarded=discarded,
        truncated=truncated,
        matcher=matcher_parameters,
    )
