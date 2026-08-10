"""Photodetector R/EQE/gain numerical tests."""

from __future__ import annotations

import json
import asyncio

import pytest

from photomatagent.scientific.capabilities.photodetector import (
    CheckTargetsTool,
    EQEFromResponsivityTool,
    ResponsivityFromEQETool,
)


def _run(tool, **arguments):
    return asyncio.run(tool.execute(arguments))


def test_eqe_to_responsivity_known_value():
    tool = ResponsivityFromEQETool()
    result = _run(tool, wavelength_um=1.0, eqe_fraction=1.0)
    assert not result.is_error
    payload = json.loads(result.output)
    assert payload["responsivity_a_w"] == pytest.approx(1.0 / 1.239841984, rel=1e-4)
    assert payload["ideal_unity_gain_responsivity_a_w"] == payload["responsivity_a_w"]


def test_fraction_percent_not_confused():
    tool = ResponsivityFromEQETool()
    by_fraction = _run(tool, wavelength_um=2.0, eqe_fraction=0.2)
    by_percent = _run(tool, wavelength_um=2.0, eqe_percent=20.0)
    assert json.loads(by_fraction.output)["responsivity_a_w"] == pytest.approx(
        json.loads(by_percent.output)["responsivity_a_w"], rel=1e-4
    )


def test_both_eqe_forms_rejected():
    tool = ResponsivityFromEQETool()
    result = _run(tool, wavelength_um=2.0, eqe_fraction=0.2, eqe_percent=20.0)
    assert result.is_error
    assert "exactly one" in result.output


def test_eqe_out_of_range_rejected():
    tool = ResponsivityFromEQETool()
    assert _run(tool, wavelength_um=2.0, eqe_fraction=1.2).is_error
    assert _run(tool, wavelength_um=2.0, eqe_percent=101.0).is_error


def test_responsivity_to_eqe_roundtrip():
    r_tool = ResponsivityFromEQETool()
    e_tool = EQEFromResponsivityTool()
    r_result = json.loads(_run(r_tool, wavelength_um=2.5, eqe_fraction=0.5).output)
    e_result = json.loads(
        _run(
            e_tool,
            wavelength_um=2.5,
            responsivity_a_w=r_result["responsivity_a_w"],
        ).output
    )
    assert e_result["eqe_fraction"] == pytest.approx(0.5, rel=1e-4)


def test_gain_scales_responsivity():
    tool = ResponsivityFromEQETool()
    unity = json.loads(_run(tool, wavelength_um=1.5, eqe_fraction=0.5).output)
    gained = json.loads(
        _run(tool, wavelength_um=1.5, eqe_fraction=0.5, photoconductive_gain=10.0).output
    )
    assert gained["responsivity_a_w"] == pytest.approx(
        unity["responsivity_a_w"] * 10.0, rel=1e-4
    )


def test_eqe_over_100_percent_flagged():
    tool = EQEFromResponsivityTool()
    # R = 2 A/W at 1.24 um requires EQE = 200% at unity gain -> flagged.
    result = _run(tool, wavelength_um=1.239841984, responsivity_a_w=2.0)
    payload = json.loads(result.output)
    assert payload["eqe_percent"] == pytest.approx(200.0, rel=1e-4)
    assert "gain" in payload["note"]


def test_check_targets_consistent_case():
    tool = CheckTargetsTool()
    result = _run(
        tool,
        spectral_min_um=2.0,
        spectral_max_um=5.0,
        target_responsivity_a_w=0.8,
        eqe_fraction=0.5,
    )
    payload = json.loads(result.output)
    # R = 0.5 * lambda / 1.2398: at 2 um -> 0.806 A/W >= 0.8 target.
    assert payload["mutually_consistent"] is True
    assert payload["statement"]


def test_check_targets_inconsistent_requires_gain():
    tool = CheckTargetsTool()
    result = _run(
        tool,
        spectral_min_um=2.0,
        spectral_max_um=5.0,
        target_responsivity_a_w=0.8,
        eqe_fraction=0.1,
    )
    payload = json.loads(result.output)
    assert payload["mutually_consistent"] is False
    assert payload["required_gain_if_not_consistent"] > 1.0


def test_check_targets_wavelength_dependence():
    tool = CheckTargetsTool()
    result = _run(
        tool,
        spectral_min_um=2.0,
        spectral_max_um=5.0,
        target_responsivity_a_w=0.8,
        eqe_percent=20.0,
    )
    payload = json.loads(result.output)
    required = payload["required_eqe_for_r_target"]
    assert required["2.0"] > required["5.0"]  # shorter wavelength needs more EQE


def test_check_targets_evidence_attached():
    tool = CheckTargetsTool()
    result = _run(
        tool,
        spectral_min_um=2.0,
        spectral_max_um=5.0,
        target_responsivity_a_w=0.8,
        eqe_fraction=0.5,
    )
    assert len(result.evidence) == 1
    assert result.evidence[0].property == "required_eqe_for_responsivity"
    assert result.evidence[0].source_type == "analytical_model"
