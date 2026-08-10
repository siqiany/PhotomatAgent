"""Anderson-rule band alignment tests."""

from __future__ import annotations

import json
import asyncio

from photomatagent.scientific.capabilities.interface import (
    AndersonBandAlignmentTool,
)


def _align(**kwargs):
    tool = AndersonBandAlignmentTool()
    return asyncio.run(tool.execute(kwargs))


def test_silicon_ge_like_case():
    # A = Si-like (chi 4.05, Eg 1.12), B = Ge-like (chi 4.0, Eg 0.67).
    result = _align(
        electron_affinity_a_eV=4.05,
        band_gap_a_eV=1.12,
        electron_affinity_b_eV=4.0,
        band_gap_b_eV=0.67,
    )
    assert not result.is_error
    payload = json.loads(result.output)
    # CBO = 4.05 - 4.0 = 0.05 eV; VBO = 0.05 + (1.12 - 0.67) = 0.50 eV.
    assert payload["cbo_eV"] == 0.05
    assert payload["vbo_eV"] == 0.50
    assert payload["alignment_type"].startswith("Type I")


def test_staggered_alignment():
    # chi_A < chi_B but Eg_A > Eg_B -> CBO < 0, VBO > 0 -> Type II.
    result = _align(
        electron_affinity_a_eV=4.0,
        band_gap_a_eV=1.5,
        electron_affinity_b_eV=4.3,
        band_gap_b_eV=0.7,
    )
    payload = json.loads(result.output)
    assert payload["cbo_eV"] < 0
    assert payload["vbo_eV"] > 0
    assert payload["alignment_type"].startswith("Type II")


def test_low_fidelity_flag_present():
    result = _align(
        electron_affinity_a_eV=4.05,
        band_gap_a_eV=1.12,
        electron_affinity_b_eV=4.0,
        band_gap_b_eV=0.67,
    )
    payload = json.loads(result.output)
    assert payload["fidelity_note"] == "LOW FIDELITY"
    assert any("dipole" in item for item in payload["assumptions"])


def test_evidence_attached():
    result = _align(
        electron_affinity_a_eV=4.05,
        band_gap_a_eV=1.12,
        electron_affinity_b_eV=4.0,
        band_gap_b_eV=0.67,
    )
    assert len(result.evidence) == 1
    assert result.evidence[0].fidelity == "analytical"
