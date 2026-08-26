"""Offline tests for the generic chemistry structure capability.

RDKit is used for deterministic 3D embeddings; no network, SSH or VASP is
ever touched. Real POTCAR content never appears anywhere.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore", message=".*explicit Hs.*")

from photomatagent.scientific.capabilities.chemistry.conformers import (
    ChemistryError,
    generate_conformer_candidates,
    mol_formula,
    mol_from_smiles,
)
from photomatagent.scientific.capabilities.chemistry.models import (
    ProvenanceStatus,
    ReliabilityGrade,
)
from photomatagent.scientific.capabilities.chemistry.registry import (
    APPROVED_ALIAS_REGISTRY,
    lookup_alias,
)
from photomatagent.scientific.capabilities.chemistry.resolver import (
    StructureRequest,
    resolve_structure,
    validate_generated,
)


def test_alias_registry_charges_are_explicit():
    assert lookup_alias("TFPMA").total_charge == 0
    assert lookup_alias("DME-Li+").total_charge == 1
    assert lookup_alias("TVM-Li+").total_charge == 1
    assert lookup_alias("TVM-TFSI-").total_charge == -1
    assert lookup_alias("TFSI-").total_charge == -1
    assert lookup_alias("Li+").total_charge == 1
    # Case-insensitive, never guessed: unknown names return None.
    assert lookup_alias("dme_li") is not None
    assert lookup_alias("some-random-polymer") is None


def test_alias_formulas_match_verified_plan():
    formulas = {
        entry.system_id: entry.formula for entry in APPROVED_ALIAS_REGISTRY
    }
    assert formulas["tfpma"] == "C7H8F4O2"
    assert formulas["vec"] == "C5H6O3"
    assert formulas["mba"] == "C7H10N2O2"
    assert formulas["dme"] == "C4H10O2"
    assert formulas["tfsi"] == "C2F6NO4S2"


def test_smiles_parsing_and_charge_contract():
    mol = mol_from_smiles(
        "C=C(C)C(=O)OCC(F)(F)C(F)F", expected_charge=0, name="TFPMA"
    )
    assert mol_formula(mol) == "C7H8F4O2"
    with pytest.raises(ChemistryError) as excinfo:
        mol_from_smiles("[Li+]", expected_charge=0, name="wrong")
    assert "CHARGE_MISMATCH" in excinfo.value.code


def test_conformer_generation_is_deterministic(tmp_path):
    mol = mol_from_smiles("COCCOC", expected_charge=0, name="DME")
    first = generate_conformer_candidates(
        mol, n_conformers=6, seed=20260825, max_returned=2
    )
    second = generate_conformer_candidates(
        mol, n_conformers=6, seed=20260825, max_returned=2
    )
    assert len(first) >= 1
    assert [candidate.energy_kcal_mol for candidate in first] == [
        candidate.energy_kcal_mol for candidate in second
    ]
    # Energy-sorted rank order and collision-free heavy pairs.
    assert first[0].energy_kcal_mol is not None
    assert first[0].min_heavy_distance >= 0.9
    # xyz render round-trip keeps the formula's atom count (Hs included).
    from photomatagent.scientific.capabilities.chemistry.conformers import (
        mol_to_xyz,
    )
    from photomatagent.scientific.capabilities.chemistry.storage import read_xyz

    path = tmp_path / "dme.xyz"
    path.write_text(mol_to_xyz(first[0].mol, first[0].conf_id), encoding="utf-8")
    symbols, _, _ = read_xyz(path)
    assert len(symbols) == first[0].mol.GetNumAtoms()


def test_smiles_resolution_grades_and_manifest(tmp_path):
    structures = resolve_structure(
        StructureRequest(
            system_id="tfpma",
            display_name="TFPMA",
            total_charge=0,
            max_candidates=2,
        ),
        tmp_path / "structures",
    )
    assert len(structures) == 2
    first = structures[0]
    assert first.identity.formula == "C7H8F4O2"
    assert first.atom_count == 21  # TFPMA C7H8F4O2
    assert first.reliability_grade() is ReliabilityGrade.B
    assert first.provenance.status is ProvenanceStatus.GENERATED_FROM_SMILES
    assert first.provenance.random_seed == 20260825
    assert first.structure_path.is_file()
    assert validate_generated(first) == []


def test_complex_build_multiple_candidates_and_charge_sum(tmp_path):
    structures = resolve_structure(
        StructureRequest(
            system_id="dme_li",
            display_name="DME-Li+",
            total_charge=1,
            max_candidates=2,
        ),
        tmp_path / "structures",
    )
    assert len(structures) >= 1
    first = structures[0]
    assert first.identity.total_charge == 1
    assert first.reliability_grade() is ReliabilityGrade.C
    assert first.provenance.status is ProvenanceStatus.HEURISTIC_COMPLEX
    # Charge = fragment sum: DME (0) + Li+ (+1).
    assert "fragment charge sum" in first.provenance.assumptions[0]
    assert first.identity.system_id == "dme_li"


def test_proxy_and_blocked_never_guess(tmp_path):
    # Explicit proxy marker when nothing is known.
    structures = resolve_structure(
        StructureRequest(
            system_id="unknown-polymer-x",
            display_name="UNKNOWNPOLY",
            allow_assumed=True,
        ),
        tmp_path / "structures",
    )
    assert structures[0].provenance.status is ProvenanceStatus.ASSUMED_PROXY
    assert structures[0].reliability_grade() is ReliabilityGrade.D
    assert structures[0].atom_count == 0
    assert not (tmp_path / "structures" / "unknown-polymer-x_POSCAR.xyz").exists()

    # allow_assumed=False -> typed BLOCKED, never a guessed structure.
    blocked = resolve_structure(
        StructureRequest(
            system_id="unknown-polymer-y",
            allow_assumed=False,
        ),
        tmp_path / "structures2",
    )
    assert blocked[0].provenance.status is ProvenanceStatus.GENERATION_FAILED
    assert "BLOCKED_MISSING_STRUCTURE" in blocked[0].validation


def test_oligomer_proxy_explicit_defaults(tmp_path):
    structures = resolve_structure(
        StructureRequest(
            system_id="vm",
            display_name="VM",
            total_charge=0,
            max_candidates=1,
        ),
        tmp_path / "structures",
    )
    first = structures[0]
    assert first.provenance.status is ProvenanceStatus.ASSUMED_REPRESENTATIVE
    assert first.reliability_grade() is ReliabilityGrade.C
    joined = "\n".join(first.provenance.assumptions)
    assert "repeat_counts" in joined
    assert "end_caps" in joined
    assert "crosslink_position" in joined
    assert first.identity.formula  # real chain formula, not guessed
    # The proxy must never claim to be the real polymer.
    assert first.identity.role.value == "oligomer"


def test_user_provided_structure_grade_a(tmp_path):
    xyz = tmp_path / "user.xyz"
    xyz.write_text(
        "5\nuser structure\nLi 0.0 0.0 0.0\nN 1.0 0.0 0.0\n"
        "O 2.0 0.0 0.0\nO 1.0 1.0 0.0\nO 1.0 -1.0 0.0\n",
        encoding="utf-8",
    )
    structures = resolve_structure(
        StructureRequest(
            system_id="lino3_user",
            display_name="LiNO3",
            structure_path=xyz,
            total_charge=0,
        ),
        tmp_path / "structures",
    )
    first = structures[0]
    assert first.provenance.status is ProvenanceStatus.USER_PROVIDED
    assert first.reliability_grade() is ReliabilityGrade.A
    assert first.atom_count == 5
