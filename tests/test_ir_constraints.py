"""Deterministic physics tests for ir.compile_constraints."""

from __future__ import annotations

import math

import pytest

from photomatagent.scientific.capabilities.ir import (
    HC_EV_UM,
    blackbody_photon_flux,
    compile_ir_constraints,
)


def test_lwir_8_14um_cutoff_energy():
    constraints = compile_ir_constraints(spectral_min_um=8, spectral_max_um=14)
    # Eg must be <= hc / 14 um to reach 14 um.
    expected = HC_EV_UM / 14.0
    assert constraints["cutoff_energy_requirement_eV"] == pytest.approx(expected, rel=1e-4)
    assert constraints["band_gap_upper_bound_eV"] == pytest.approx(expected, rel=1e-4)
    # Photon energy range: [hc/14, hc/8].
    assert constraints["photon_energy_range_eV"][0] == pytest.approx(
        HC_EV_UM / 14.0, rel=1e-4
    )
    assert constraints["photon_energy_range_eV"][1] == pytest.approx(
        HC_EV_UM / 8.0, rel=1e-4
    )


def test_mwir_3_5um_cutoff_energy():
    constraints = compile_ir_constraints(spectral_min_um=3, spectral_max_um=5)
    assert constraints["cutoff_energy_requirement_eV"] == pytest.approx(
        HC_EV_UM / 5.0, rel=1e-4
    )


def test_ideal_responsivity_at_ten_micron():
    constraints = compile_ir_constraints(spectral_min_um=8, spectral_max_um=14)
    # Geometric mean wavelength ~ 10.58 um; ideal R = lambda / 1.2398.
    expected = math.sqrt(8 * 14) / HC_EV_UM
    assert constraints["ideal_responsivity_A_per_W"]["at_peak_wavelength_A_per_W"] == pytest.approx(
        expected, rel=1e-4
    )
    assert constraints["ideal_responsivity_A_per_W"]["at_shortest_A_per_W"] == pytest.approx(
        8.0 / HC_EV_UM, rel=1e-4
    )


def test_blackbody_photon_flux_increases_with_temperature():
    cold = blackbody_photon_flux(8.0, 14.0, 200.0)
    hot = blackbody_photon_flux(8.0, 14.0, 300.0)
    assert hot > cold > 0


def test_blackbody_photon_flux_physical_magnitude():
    # At 300 K the 8-14 um photon flux is on the order of 1e21-1e22
    # photons/s/m2 (~1e17-1e18 photons/s/cm2).
    flux = blackbody_photon_flux(8.0, 14.0, 300.0)
    assert 1e20 < flux < 1e23


def test_thermal_guideline_and_ratio():
    constraints = compile_ir_constraints(spectral_min_um=8, spectral_max_um=14, temperature_k=300)
    kt = 8.617333262e-5 * 300
    assert constraints["thermal"]["kBT_eV"] == pytest.approx(kt, rel=1e-4)
    cutoff = HC_EV_UM / 14.0
    assert constraints["thermal"]["thermal_suppression_ratio_at_cutoff"] == pytest.approx(
        math.exp(-cutoff / kt), rel=1e-4
    )
    assert constraints["thermal"]["thermal_gap_guideline_eV"] == pytest.approx(4 * kt, rel=1e-4)


def test_required_evidence_base_and_detector_specific():
    constraints = compile_ir_constraints(
        spectral_min_um=8, spectral_max_um=14, detector_type="photodiode"
    )
    evidence = constraints["required_evidence"]
    assert any("band gap" in item for item in evidence)
    assert any("R0A" in item for item in evidence)
    constraints_general = compile_ir_constraints(spectral_min_um=8, spectral_max_um=14)
    assert not any("R0A" in item for item in constraints_general["required_evidence"])


def test_targets_flow_into_evidence():
    constraints = compile_ir_constraints(
        spectral_min_um=8,
        spectral_max_um=14,
        target_detectivity=1e9,
        target_netd=0.05,
        target_dark_current=1e-9,
    )
    assert constraints["targets"]["detectivity_cm_Hz_W"] == 1e9
    assert constraints["targets"]["netd_K"] == 0.05
    assert constraints["targets"]["dark_current_A"] == 1e-9
    assert any("NETD" in item for item in constraints["required_evidence"])
    assert any("dark-current budget" in item for item in constraints["required_evidence"])


def test_blip_detectivity_is_positive():
    constraints = compile_ir_constraints(spectral_min_um=8, spectral_max_um=14, temperature_k=300)
    assert constraints["blip_detectivity_cm_Hz_W"] > 0


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        compile_ir_constraints(spectral_min_um=0, spectral_max_um=14)
    with pytest.raises(ValueError):
        compile_ir_constraints(spectral_min_um=14, spectral_max_um=8)
    with pytest.raises(ValueError):
        compile_ir_constraints(spectral_min_um=8, spectral_max_um=14, temperature_k=-5)
    with pytest.raises(ValueError):
        compile_ir_constraints(spectral_min_um=8, spectral_max_um=14, detector_type="plasma")


def test_compile_constraints_is_deterministic():
    kwargs = dict(
        spectral_min_um=8.0,
        spectral_max_um=14.0,
        temperature_k=250.0,
        detector_type="photoconductor",
    )
    first = compile_ir_constraints(**kwargs)
    second = compile_ir_constraints(**kwargs)
    assert first == second
