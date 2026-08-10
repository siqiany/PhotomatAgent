"""Alloy bowing numerical tests."""

from __future__ import annotations

import pytest

from photomatagent.scientific.capabilities.quantum_dot.alloy import bandgap_bowing
from photomatagent.scientific.errors import MissingScientificPrerequisite


def test_endpoints_exact():
    at_a = bandgap_bowing(x=0.0, band_gap_a_eV=0.354, band_gap_b_eV=1.475, bowing_parameter_eV=0.3)
    at_b = bandgap_bowing(x=1.0, band_gap_a_eV=0.354, band_gap_b_eV=1.475, bowing_parameter_eV=0.3)
    assert at_a["band_gap_eV"] == pytest.approx(0.354)
    assert at_b["band_gap_eV"] == pytest.approx(1.475)


def test_bowing_dips_below_linear_interpolation():
    linear = 0.5 * (0.354 + 1.475)
    bowed = bandgap_bowing(x=0.5, band_gap_a_eV=0.354, band_gap_b_eV=1.475, bowing_parameter_eV=0.5)
    assert bowed["band_gap_eV"] < linear


def test_bowing_effect_scales_with_parameter():
    small = bandgap_bowing(x=0.5, band_gap_a_eV=0.0, band_gap_b_eV=2.0, bowing_parameter_eV=0.1)
    large = bandgap_bowing(x=0.5, band_gap_a_eV=0.0, band_gap_b_eV=2.0, bowing_parameter_eV=1.0)
    assert large["band_gap_eV"] < small["band_gap_eV"]


def test_out_of_range_x_rejected():
    with pytest.raises(ValueError):
        bandgap_bowing(x=1.2, band_gap_a_eV=1.0, band_gap_b_eV=2.0, bowing_parameter_eV=0.3)
    with pytest.raises(ValueError):
        bandgap_bowing(x=-0.1, band_gap_a_eV=1.0, band_gap_b_eV=2.0, bowing_parameter_eV=0.3)


def test_wavelength_from_positive_gap():
    result = bandgap_bowing(x=0.3, band_gap_a_eV=0.354, band_gap_b_eV=1.475, bowing_parameter_eV=0.4)
    assert result["wavelength_um"] == pytest.approx(1.239841984 / result["band_gap_eV"])


def test_varshni_missing_parameters_is_prerequisite():
    with pytest.raises(MissingScientificPrerequisite):
        bandgap_bowing(
            x=0.5,
            band_gap_a_eV=0.3,
            band_gap_b_eV=1.4,
            bowing_parameter_eV=0.2,
            temperature_k=300.0,
        )


def test_varshni_shift_applied():
    shifted = bandgap_bowing(
        x=0.0,
        band_gap_a_eV=0.354,
        band_gap_b_eV=1.475,
        bowing_parameter_eV=0.0,
        temperature_k=300.0,
        varshni_alpha_a=0.000276,
        varshni_beta_a=93.0,
        varshni_alpha_b=0.0,
        varshni_beta_b=1.0,
    )
    expected = 0.354 - 0.000276 * 300.0**2 / (300.0 + 93.0)
    assert shifted["band_gap_eV"] == pytest.approx(expected, rel=1e-6)
