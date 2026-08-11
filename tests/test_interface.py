"""Anderson-rule band alignment tests (vacuum-level convention).

Convention under test (Sprint 3 hotfix): vacuum level = 0 eV, Ec = -chi,
Ev = -(chi + Eg), CBO = Ec(A) - Ec(B) = chi_B - chi_A, VBO = Ev(A) - Ev(B)
= CBO + (Eg_B - Eg_A). Positive CBO -> electrons collected in B; positive
VBO -> holes collected in B.
"""

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
    # Vacuum convention: CBO = chi_B - chi_A = 4.0 - 4.05 = -0.05 eV;
    # VBO = CBO + (Eg_B - Eg_A) = -0.05 + (0.67 - 1.12) = -0.50 eV.
    # Both edges of Ge lie above the Si edges -> staggered Type II.
    assert payload["cbo_eV"] == -0.05
    assert payload["vbo_eV"] == -0.50
    assert payload["alignment_type"].startswith("Type II")
    # Vacuum-level convention sanity: Ec = -chi, Ev = -(chi + Eg).
    assert payload["conduction_band_edges_eV"] == {
        "Ec_A": -4.05,
        "Ec_B": -4.0,
    }
    assert payload["valence_band_edges_eV"] == {
        "Ev_A": -5.17,
        "Ev_B": -4.67,
    }


def test_type_i_straddling_fixture():
    # B's gap contains A's gap (B: wide gap, low chi).
    # A: chi=4.0, Eg=1.0 -> Ec_A=-4.0, Ev_A=-5.0
    # B: chi=3.0, Eg=3.0 -> Ec_B=-3.0, Ev_B=-6.0
    # CBO = -4.0 - (-3.0) = -1.0; VBO = -5.0 - (-6.0) = +1.0 -> Type I.
    result = _align(
        electron_affinity_a_eV=4.0,
        band_gap_a_eV=1.0,
        electron_affinity_b_eV=3.0,
        band_gap_b_eV=3.0,
    )
    payload = json.loads(result.output)
    assert payload["cbo_eV"] == -1.0
    assert payload["vbo_eV"] == 1.0
    assert payload["alignment_type"].startswith("Type I")


def test_type_ii_staggered_fixture():
    # Both edges of B shifted the same way relative to A -> Type II.
    # A: chi=4.05, Eg=1.12 (Si-like); B: chi=4.0, Eg=0.67 (Ge-like).
    result = _align(
        electron_affinity_a_eV=4.05,
        band_gap_a_eV=1.12,
        electron_affinity_b_eV=4.0,
        band_gap_b_eV=0.67,
    )
    payload = json.loads(result.output)
    assert payload["cbo_eV"] * payload["vbo_eV"] > 0
    assert payload["alignment_type"].startswith("Type II")


def test_type_iii_broken_gap_fixture():
    # A: chi=4.0, Eg=0.2 -> Ev_A = -4.2; B: chi=4.5, Eg=1.0 -> Ec_B = -4.5.
    # Ev_A (-4.2) > Ec_B (-4.5): valence band of A overlaps conduction band
    # of B -> broken gap (Type III).
    result = _align(
        electron_affinity_a_eV=4.0,
        band_gap_a_eV=0.2,
        electron_affinity_b_eV=4.5,
        band_gap_b_eV=1.0,
    )
    payload = json.loads(result.output)
    assert payload["alignment_type"].startswith("Type III")


def test_identical_materials_zero_offsets_not_type_iii():
    # material A == material B -> zero offsets, and never Type III.
    result = _align(
        electron_affinity_a_eV=4.05,
        band_gap_a_eV=1.12,
        electron_affinity_b_eV=4.05,
        band_gap_b_eV=1.12,
    )
    assert not result.is_error
    payload = json.loads(result.output)
    assert payload["cbo_eV"] == 0.0
    assert payload["vbo_eV"] == 0.0
    assert "Type I" in payload["alignment_type"]
    assert "Type III" not in payload["alignment_type"]


def test_sign_convention_electrons_fall_into_lower_cbm():
    # Electrons fall from the higher CBM into the lower CBM.
    # A: chi=4.0 -> Ec_A = -4.0; B: chi=4.5 -> Ec_B = -4.5.
    # CBO = Ec(A) - Ec(B) = +0.5 -> electrons collected in B.
    result = _align(
        electron_affinity_a_eV=4.0,
        band_gap_a_eV=1.5,
        electron_affinity_b_eV=4.5,
        band_gap_b_eV=1.5,
    )
    payload = json.loads(result.output)
    assert payload["cbo_eV"] == 0.5
    assert payload["conduction_band_edges_eV"]["Ec_B"] < payload[
        "conduction_band_edges_eV"
    ]["Ec_A"]


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
    assert payload["vacuum_level_eV"] == 0.0


def test_evidence_attached():
    result = _align(
        electron_affinity_a_eV=4.05,
        band_gap_a_eV=1.12,
        electron_affinity_b_eV=4.0,
        band_gap_b_eV=0.67,
    )
    assert len(result.evidence) == 1
    assert result.evidence[0].fidelity == "analytical"
