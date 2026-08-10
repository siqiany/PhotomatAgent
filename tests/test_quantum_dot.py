"""Numerical validation of the Brus QD solvers (no LLM arithmetic)."""

from __future__ import annotations

import math

import pytest

from photomatagent.scientific.capabilities.quantum_dot.brus import (
    HC_EV_UM,
    MIN_RADIUS_NM,
    excitonic_regime,
    exciton_bohr_radius_nm,
    size_sweep,
    solve_size_for_transition,
    transition_energy,
)
from photomatagent.scientific.errors import MissingScientificPrerequisite


GAAS = dict(
    bulk_band_gap_eV=1.424,
    electron_effective_mass_m0=0.067,
    hole_effective_mass_m0=0.45,
    relative_dielectric_constant=12.9,
)


def test_larger_radius_less_confinement():
    small = transition_energy(radius_nm=2.0, **GAAS)
    large = transition_energy(radius_nm=5.0, **GAAS)
    assert small["confinement_shift_eV"] > large["confinement_shift_eV"]
    assert small["transition_energy_eV"] > large["transition_energy_eV"]


def test_infinite_radius_approaches_bulk_limit():
    huge = transition_energy(radius_nm=200.0, **GAAS)
    assert abs(huge["transition_energy_eV"] - GAAS["bulk_band_gap_eV"]) < 1e-3


def test_known_gaas_numerical_value():
    # Hand-computed from the formula with the module constants (unit guard):
    # shift = hbar^2 e pi^2 (1/me+1/mh) / (2 m0 R^2), R = 3 nm.
    result = transition_energy(radius_nm=3.0, include_coulomb_term=False, **GAAS)
    expected_shift = (
        6.582119569e-16**2
        * 1.602176634e-19
        * math.pi**2
        * (1 / 0.067 + 1 / 0.45)
        / (2 * 9.1093837015e-31 * (3e-9) ** 2)
    )
    assert result["confinement_shift_eV"] == pytest.approx(expected_shift, rel=1e-9)
    assert result["transition_energy_eV"] == pytest.approx(
        GAAS["bulk_band_gap_eV"] + expected_shift, rel=1e-9
    )


def test_coulomb_term_reduces_transition_energy():
    with_coulomb = transition_energy(radius_nm=3.0, **GAAS)
    without = transition_energy(radius_nm=3.0, include_coulomb_term=False, **GAAS)
    assert with_coulomb["transition_energy_eV"] < without["transition_energy_eV"]
    assert with_coulomb["coulomb_correction_eV"] > 0


def test_wavelength_energy_roundtrip():
    result = transition_energy(radius_nm=4.0, **GAAS)
    energy = result["transition_energy_eV"]
    wavelength = result["transition_wavelength_um"]
    assert wavelength == pytest.approx(HC_EV_UM / energy, rel=1e-6)


def test_negative_mass_rejected():
    with pytest.raises(ValueError):
        transition_energy(
            radius_nm=3.0, bulk_band_gap_eV=1.0, electron_effective_mass_m0=-0.1,
            hole_effective_mass_m0=0.4,
        )
    with pytest.raises(ValueError):
        transition_energy(
            radius_nm=3.0, bulk_band_gap_eV=1.0, electron_effective_mass_m0=0.1,
            hole_effective_mass_m0=0.0,
        )


def test_invalid_radius_rejected():
    with pytest.raises(ValueError):
        transition_energy(radius_nm=-1.0, **GAAS)
    with pytest.raises(ValueError):
        transition_energy(radius_nm=0.0, **GAAS)


def test_coulomb_without_dielectric_is_prerequisite():
    with pytest.raises(MissingScientificPrerequisite):
        transition_energy(
            radius_nm=3.0,
            bulk_band_gap_eV=1.0,
            electron_effective_mass_m0=0.1,
            hole_effective_mass_m0=0.4,
            include_coulomb_term=True,
        )


def test_exciton_bohr_radius_gaas_magnitude():
    # GaAs: a_B* ~ 10-12 nm with these masses.
    a_star = exciton_bohr_radius_nm(
        electron_effective_mass_m0=0.067,
        hole_effective_mass_m0=0.45,
        relative_dielectric_constant=12.9,
    )
    assert 9.0 < a_star < 13.0


def test_excitonic_regime_labels():
    strong = excitonic_regime(radius_nm=2.0, **{k: v for k, v in GAAS.items() if k != "bulk_band_gap_eV"})
    assert strong["confinement_regime"] == "strong"
    weak = excitonic_regime(
        radius_nm=60.0,
        electron_effective_mass_m0=0.067,
        hole_effective_mass_m0=0.45,
        relative_dielectric_constant=12.9,
    )
    assert weak["confinement_regime"] == "weak"


def test_inverse_roundtrip_with_coulomb():
    forward = transition_energy(radius_nm=2.0, **GAAS)
    inverse = solve_size_for_transition(
        target_energy_eV=forward["transition_energy_eV"], **GAAS
    )
    assert inverse["outcome"] == "SOLVED"
    assert inverse["candidate_radius_nm"] == pytest.approx(2.0, abs=1e-3)
    assert inverse["residual_eV"] < 1e-6


def test_inverse_roundtrip_without_coulomb():
    forward = transition_energy(radius_nm=2.5, include_coulomb_term=False, **GAAS)
    inverse = solve_size_for_transition(
        target_energy_eV=forward["transition_energy_eV"],
        include_coulomb_term=False,
        **GAAS,
    )
    assert inverse["outcome"] == "SOLVED"
    assert inverse["candidate_radius_nm"] == pytest.approx(2.5, abs=1e-3)


def test_inverse_by_wavelength():
    target_um = 0.65
    inverse = solve_size_for_transition(
        target_wavelength_um=target_um, **GAAS
    )
    assert inverse["outcome"] == "SOLVED"
    assert inverse["predicted_transition_wavelength_um"] == pytest.approx(
        target_um, rel=1e-3
    )


def test_inverse_below_gap_returns_no_solution():
    inverse = solve_size_for_transition(
        target_energy_eV=1.0,  # below GaAs bulk gap 1.424
        **GAAS,
    )
    assert inverse["outcome"] == "NO_PHYSICAL_SOLUTION"


def test_inverse_exactly_at_gap_without_coulomb_returns_no_solution():
    inverse = solve_size_for_transition(
        target_energy_eV=GAAS["bulk_band_gap_eV"],
        include_coulomb_term=False,
        **GAAS,
    )
    assert inverse["outcome"] == "NO_PHYSICAL_SOLUTION"


def test_inverse_requires_target():
    with pytest.raises(ValueError):
        solve_size_for_transition(**GAAS)
    with pytest.raises(ValueError):
        solve_size_for_transition(target_energy_eV=1.5, target_wavelength_um=1.0, **GAAS)


def test_size_sweep_monotonic_and_bounded():
    result = size_sweep(
        min_size_nm=1.0,
        max_size_nm=10.0,
        points=50,
        **GAAS,
    )
    assert result["count"] == 50
    assert len(result["selected_rows"]) == 50
    energies = [row["transition_energy_eV"] for row in result["selected_rows"]]
    assert energies == sorted(energies, reverse=True)
    assert result["summary"]["wavelength_max_um"] >= result["summary"]["wavelength_min_um"]


def test_size_sweep_downsampled():
    result = size_sweep(min_size_nm=1.0, max_size_nm=10.0, points=1000, **GAAS)
    assert result["count"] == 1000
    assert len(result["selected_rows"]) <= 100
    assert "showing" in result["note"]


def test_narrow_gap_warning_emitted():
    result = transition_energy(
        radius_nm=3.0,
        bulk_band_gap_eV=-0.30,  # HgTe-like inverted gap
        electron_effective_mass_m0=0.03,
        hole_effective_mass_m0=0.5,
        relative_dielectric_constant=21.0,
    )
    assert any("inverted" in warning for warning in result["warnings"])
    assert result["fidelity_level"] == "L1"


def test_parameter_registry_lookup_with_sources():
    from photomatagent.scientific.capabilities.quantum_dot.models import default_registry

    registry = default_registry()
    me = registry.get("InAs", "electron_effective_mass")
    assert me is not None
    assert me.value == pytest.approx(0.026)
    assert me.unit == "m0"
    assert "Vurgaftman" in me.source
    assert registry.get("InAs", "missing_property") is None
    assert registry.get("UnknownMaterial", "band_gap") is None
