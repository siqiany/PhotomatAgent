"""Pure, bounded crystal-structure construction operations.

The functions in this module operate on pymatgen objects supplied by the
caller. They validate the complete request before copying or mutating any
structure, and never write files or alter scientific state.
"""

from __future__ import annotations

from typing import Any, Iterable

from photomatagent.scientific.capabilities.structure.construction_models import (
    ConstructionLimits,
    SiteReplacement,
)
from photomatagent.scientific.discovery.composition import composition_key


class StructureConstructionError(ValueError):
    """A deterministic construction failure with a model-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _validate_ordered_structure(structure: Any) -> None:
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
    result.make_supercell(factors)
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
    for replacement in replacements:
        result.replace(replacement.index, replacement.to_element)
    try:
        actual = composition_key(result.composition.formula)
    except (TypeError, ValueError) as exc:
        raise StructureConstructionError(
            "INVALID_EXPECTED_FORMULA", str(exc)
        ) from exc
    if actual != expected:
        raise StructureConstructionError(
            "COMPOSITION_MISMATCH",
            f"constructed composition {result.composition.reduced_formula} does not match expected_formula {expected_formula}",
        )
    return result
