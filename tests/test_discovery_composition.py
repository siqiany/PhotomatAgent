from __future__ import annotations

import builtins

import pytest

from photomatagent.scientific.discovery.composition import (
    CompositionCapabilityError,
    composition_key,
    normalize_composition,
)


def test_equivalent_fractional_formulas_have_same_identity() -> None:
    assert composition_key("Na0.75Ag0.25BiS2") == composition_key("Na3AgBi4S8")
    assert composition_key("Na0.5Ag0.5BiS2") != composition_key("Na3AgBi4S8")


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("NaBiS2", (("Bi", 1), ("Na", 1), ("S", 2))),
        ("S2BiNa", (("Bi", 1), ("Na", 1), ("S", 2))),
        ("Na(ClO3)2", (("Cl", 2), ("Na", 1), ("O", 6))),
        ("Na0.5Ag0.5BiS2", (("Ag", 1), ("Bi", 2), ("Na", 1), ("S", 4))),
        ("Na0.333333333333Cl", (("Cl", 3), ("Na", 1))),
    ],
)
def test_normalize_composition_reduces_and_sorts_elements(
    formula: str,
    expected: tuple[tuple[str, int], ...],
) -> None:
    assert normalize_composition(formula) == expected


@pytest.mark.parametrize(
    "formula",
    [
        "",
        "   ",
        "Na1-xAgxBiS2",
        "Na-1Cl2",
        "Na0Cl",
        "NaInfCl",
        "Xx2O",
        "Na" + "1" * 256,
        "Na0.00005Cl",
        "HHeLiBeBCNOFNeNaMgAlSiPSClArK",
        "Na1000000000Cl",
    ],
)
def test_normalize_composition_rejects_invalid_or_unbounded_formulas(formula: str) -> None:
    with pytest.raises(ValueError):
        normalize_composition(formula)


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("NaN", (("N", 1), ("Na", 1))),
        ("NaNCl", (("Cl", 1), ("N", 1), ("Na", 1))),
        ("NaNaNCl", (("Cl", 1), ("N", 1), ("Na", 2))),
    ],
)
def test_element_tokens_that_spell_nan_remain_valid(
    formula: str,
    expected: tuple[tuple[str, int], ...],
) -> None:
    assert normalize_composition(formula) == expected


def test_nonfinite_parsed_amount_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite|invalid"):
        normalize_composition("Na1e999Cl")


def test_composition_module_import_is_independent_of_pymatgen() -> None:
    real_import = builtins.__import__

    def reject_pymatgen(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("pymatgen"):
            raise ImportError("pymatgen intentionally unavailable")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = reject_pymatgen  # type: ignore[assignment]
    try:
        with pytest.raises(CompositionCapabilityError, match="pymatgen"):
            normalize_composition("NaCl")
    finally:
        builtins.__import__ = real_import
