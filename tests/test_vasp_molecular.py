"""Offline tests for the isolated-molecule VASP data models + preflight.

All POTCAR fixtures are synthetic (TITEL/ZVAL/ENMAX header lines only); no
real pseudopotential content is ever used, copied or asserted. The Li ZVAL
fixture mirrors the user's real potpaw_PBE.64 library (Li ZVAL = 1.0), which
is why DME-Li+ (C4H10O2Li, q=+1) carries NELECT = 39 - 1 = 38.

No SSH, Slurm or VASP execution is involved anywhere in this module.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from photomatagent.scientific.applications.vasp.molecular.generator import (
    MolecularVaspGenerator,
)
from photomatagent.scientific.applications.vasp.molecular.models import (
    CorrectionPolicy,
    MoleculeSpec,
    MonopoleMethod,
    PolymerKind,
    Polymerization,
    PreflightReport,
    ResourceCeiling,
    StageName,
    WorkflowSpec,
)
from photomatagent.scientific.applications.vasp.molecular.preflight import (
    render_agent_text,
    run_molecular_preflight,
)
from photomatagent.scientific.applications.vasp.molecular.templates import (
    esp_incar,
    make_stage,
    orbital_incar,
    relax_incar,
    restart_incar,
    static_incar,
)


DME_SYMBOLS = ["C"] * 4 + ["O"] * 2 + ["H"] * 10

ZVAL_ENMAX = {
    "C": (4.0, 400.0),
    "O": (6.0, 400.0),
    "H": (1.0, 250.0),
    "Li": (1.0, 140.0),
    "N": (5.0, 400.0),
    "F": (7.0, 400.0),
    "S": (6.0, 258.689),
}


def write_dataset(library: Path, element: str, zval: float, enmax: float) -> Path:
    dataset = library / element
    dataset.mkdir(parents=True, exist_ok=True)
    potcar = dataset / "POTCAR"
    potcar.write_text(
        f"   TITEL  = PAW_PBE {element} 08Apr2002\n"
        f"   POMASS =    12.011; ZVAL   =    {zval:.3f}    mass and valenz\n"
        f"   ENMAX  =  {enmax:.3f}; ENMIN  =  300.000 eV\n"
        f"   EMMIN  =  100.000 eV\n",
        encoding="utf-8",
    )
    return potcar


def make_psp(tmp_path: Path, elements=("C", "O", "H", "Li")) -> Path:
    library = tmp_path / "psp"
    for element in elements:
        zval, enmax = ZVAL_ENMAX[element]
        write_dataset(library, element, zval, enmax)
    return library


def write_xyz(
    tmp_path: Path,
    name: str,
    symbols: list[str],
    coords: np.ndarray,
    comment: str,
) -> Path:
    path = tmp_path / name
    lines = [str(len(symbols)), comment]
    for symbol, position in zip(symbols, coords, strict=True):
        lines.append(
            f"{symbol} {position[0]:.6f} {position[1]:.6f} {position[2]:.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def dme_coords(seed: int = 10) -> np.ndarray:
    return np.random.default_rng(seed).uniform(10.0, 20.0, size=(16, 3))


def dme_molecule(
    tmp_path: Path,
    *,
    charge: int,
    multiplicity: int = 1,
    box: float = 30.0,
    symbols: list[str] | None = None,
    coords: np.ndarray | None = None,
    name: str = "DME",
    polymer_kind: PolymerKind = PolymerKind.NONE,
    polymerization: Polymerization | None = None,
) -> MoleculeSpec:
    symbols = symbols or list(DME_SYMBOLS)
    coords = coords if coords is not None else dme_coords()
    path = write_xyz(
        tmp_path, f"{name}.xyz", symbols, coords, f"{name} q={charge:+d}"
    )
    return MoleculeSpec(
        name=name,
        structure_path=path,
        total_charge=charge,
        spin_multiplicity=multiplicity,
        box_ang=box,
        calculation_purpose="binding",
        polymer_kind=polymer_kind,
        polymerization=polymerization,
    )


def two_stage_workflow(
    molecule: MoleculeSpec,
    *,
    spin: int = 1,
    encut: float = 520.0,
    nelect: float | None = None,
    lmono: bool = False,
    dipole: bool = True,
    ncore: int = 2,
    tasks_per_node: int = 8,
    stages: list | None = None,
) -> WorkflowSpec:
    if stages is None:
        stages = [
            make_stage(
                StageName.RELAX,
                incar=relax_incar(
                    spin=spin,
                    encut=encut,
                    nelect=nelect,
                    lmono=lmono,
                    dipole=dipole,
                    ncore=ncore,
                ),
            ),
            make_stage(
                StageName.STATIC,
                depends_on=StageName.RELAX,
                incar=static_incar(
                    spin=spin,
                    encut=encut,
                    nelect=nelect,
                    lmono=lmono,
                    dipole=dipole,
                    ncore=ncore,
                ),
                required_upstream_outputs=["CONTCAR", "CHGCAR"],
                produced_outputs=["WAVECAR", "CHGCAR"],
            ),
        ]
    return WorkflowSpec(
        molecule=molecule,
        stages=stages,
        scientific_method="isolated-molecule PBE-D3(BJ); NELECT = neutral - q",
        correction_policy=CorrectionPolicy(
            monopole_method=(
                MonopoleMethod.LMONO if lmono else MonopoleMethod.NONE
            ),
            dipole=dipole,
        ),
        resource_ceiling=ResourceCeiling(
            nodes=1, tasks_per_node=tasks_per_node, walltime_minutes=20
        ),
    )


def generate_and_report(
    tmp_path: Path, workflow: WorkflowSpec, *, write_potcar: bool = False
) -> tuple[PreflightReport, Path, dict]:
    library = make_psp(tmp_path)
    generator = MolecularVaspGenerator(psp_dir=library)
    result = generator.generate(
        workflow, tmp_path / "inputs", write_potcar=write_potcar
    )
    report = PreflightReport.model_validate(result["preflight"])
    return report, tmp_path / "inputs", result["workflow"]


def codes(report: PreflightReport) -> set[str]:
    return {issue.code for issue in report.errors}


# -- electron bookkeeping -----------------------------------------------------


def test_nelect_signs_for_q0_qplus_qminus(tmp_path):
    expectations = {0: 38.0, 1: 37.0, -1: 39.0}
    for charge, expected in expectations.items():
        lmono = charge != 0
        spin = 1 if charge == 0 else 2
        multiplicity = 1 if charge == 0 else 2
        molecule = dme_molecule(tmp_path, charge=charge, multiplicity=multiplicity)
        workflow = two_stage_workflow(
            molecule, spin=spin, nelect=expected, lmono=lmono, dipole=not lmono
        )
        report, _, _ = generate_and_report(tmp_path, workflow)
        assert report.passed, report.errors
        assert report.summary is not None
        assert report.summary.nelect == expected
        assert report.summary.neutral_valence_electrons == 38.0
        assert report.summary.charge == charge
        # q=+1 removes one electron, q=-1 adds one.
        assert report.summary.nelect == report.summary.neutral_valence_electrons - charge


def test_dme_li_plus_reports_38_paw_electrons(tmp_path):
    symbols = list(DME_SYMBOLS) + ["Li"]
    positions = np.concatenate([dme_coords(), np.asarray([[12.0, 12.0, 12.0]])])
    molecule = dme_molecule(
        tmp_path,
        charge=1,
        symbols=symbols,
        coords=positions,
        name="DME_Li_plus",
    )
    workflow = two_stage_workflow(
        molecule, spin=1, nelect=38.0, lmono=True, dipole=False
    )
    report, _, _ = generate_and_report(tmp_path, workflow)
    assert report.passed, report.errors
    assert report.summary is not None
    # C4H10O2 = 38, Li ZVAL = 1 -> neutral 39, q=+1 -> NELECT = 38.
    assert report.summary.neutral_valence_electrons == 39.0
    assert report.summary.nelect == 38.0
    assert "Li" in report.summary.elements
    assert report.summary.formula == "C4O2H10Li1"


def test_odd_electron_rejected_with_ispin1(tmp_path):
    molecule = dme_molecule(tmp_path, charge=1, multiplicity=1)
    workflow = two_stage_workflow(
        molecule, spin=1, nelect=37.0, lmono=True, dipole=False
    )
    report, _, _ = generate_and_report(tmp_path, workflow)
    assert not report.passed
    assert "ELECTRON_PARITY_MISMATCH" in codes(report)


def test_spin_multiplicity_parity_mismatch_rejected(tmp_path):
    # Even electron count (38) with an even spin multiplicity (2) is not a
    # valid spin eigenstate.
    molecule = dme_molecule(tmp_path, charge=0, multiplicity=2)
    workflow = two_stage_workflow(molecule, spin=2)
    report, _, _ = generate_and_report(tmp_path, workflow)
    assert not report.passed
    assert "ELECTRON_PARITY_MISMATCH" in codes(report)


def test_explicit_nelect_contradicting_charge_rejected(tmp_path):
    molecule = dme_molecule(tmp_path, charge=1, multiplicity=2)
    workflow = two_stage_workflow(
        molecule, spin=2, nelect=41.0, lmono=True, dipole=False
    )
    report, _, _ = generate_and_report(tmp_path, workflow)
    assert not report.passed
    assert "NELECT_MISMATCH" in codes(report)


# -- POTCAR order / duplicates / metadata ------------------------------------


def test_potcar_order_mismatch_rejected(tmp_path):
    library = make_psp(tmp_path)
    wrong = tmp_path / "potcar_wrong_order"
    wrong.write_bytes(
        (library / "O" / "POTCAR").read_bytes()
        + (library / "C" / "POTCAR").read_bytes()
        + (library / "H" / "POTCAR").read_bytes()
    )
    molecule = dme_molecule(tmp_path, charge=0)
    workflow = two_stage_workflow(molecule)
    report = run_molecular_preflight(
        workflow, psp_dir=library, potcar_path=wrong
    )
    assert not report.passed
    assert "POTCAR_ORDER_MISMATCH" in codes(report)


def test_duplicate_element_blocks_rejected(tmp_path):
    library = make_psp(tmp_path, elements=("C", "H"))
    duplicated = tmp_path / "potcar_dup"
    duplicated.write_bytes(
        (library / "H" / "POTCAR").read_bytes()
        + (library / "H" / "POTCAR").read_bytes()
    )
    # Molecule with elements [H, C]: two H blocks are invalid duplicates.
    symbols = ["H", "C", "H"]
    coords = np.asarray(
        [[10.0, 15.0, 15.0], [20.0, 15.0, 15.0], [10.0, 20.0, 15.0]]
    )
    molecule = dme_molecule(
        tmp_path,
        charge=0,
        symbols=symbols,
        coords=coords,
        name="HCH",
    )
    workflow = two_stage_workflow(molecule)
    report = run_molecular_preflight(workflow, psp_dir=library, potcar_path=duplicated)
    assert not report.passed
    assert "POTCAR_DUPLICATE_ELEMENT" in codes(report)


def test_potcar_block_count_mismatch_rejected(tmp_path):
    library = make_psp(tmp_path, elements=("C", "O"))
    short = tmp_path / "potcar_short"
    short.write_bytes((library / "C" / "POTCAR").read_bytes())
    molecule = dme_molecule(tmp_path, charge=0)
    workflow = two_stage_workflow(molecule)
    report = run_molecular_preflight(workflow, psp_dir=library, potcar_path=short)
    assert not report.passed
    assert "POTCAR_BLOCK_COUNT_MISMATCH" in codes(report)


def test_missing_psp_dataset_rejected(tmp_path):
    library = make_psp(tmp_path, elements=("C", "H"))
    molecule = dme_molecule(tmp_path, charge=0)
    workflow = two_stage_workflow(molecule)
    report = run_molecular_preflight(workflow, psp_dir=library)
    assert not report.passed
    assert "PSP_DATASET_MISSING" in codes(report)
    assert any("O" in issue.message for issue in report.errors)


def test_psp_unresolved_rejected(tmp_path):
    molecule = dme_molecule(tmp_path, charge=0)
    workflow = two_stage_workflow(molecule)
    report = run_molecular_preflight(workflow, psp_dir=None)
    assert not report.passed
    assert "PSP_UNRESOLVED" in codes(report)


# -- ENCUT / KPOINTS / DIPOL --------------------------------------------------


def test_encut_below_floor_rejected(tmp_path):
    molecule = dme_molecule(tmp_path, charge=0)
    workflow = two_stage_workflow(molecule, encut=400.0)
    report, _, _ = generate_and_report(tmp_path, workflow)
    assert not report.passed
    assert "ENCUT_BELOW_FLOOR" in codes(report)


def test_encut_below_enmax_ratio_rejected(tmp_path):
    library = make_psp(tmp_path)
    # Raise C ENMAX to 500 -> 1.3 x 500 = 650 eV > 520.
    write_dataset(library, "C", 4.0, 500.0)
    molecule = dme_molecule(tmp_path, charge=0)
    workflow = two_stage_workflow(molecule, encut=520.0)
    generator = MolecularVaspGenerator(psp_dir=library)
    result = generator.generate(workflow, tmp_path / "inputs")
    report = PreflightReport.model_validate(result["preflight"])
    assert not report.passed
    assert "ENCUT_BELOW_ENMAX_RATIO" in codes(report)
    assert "ENCUT_BELOW_FLOOR" not in codes(report)


def test_kpoints_gamma_only_enforced(tmp_path):
    molecule = dme_molecule(tmp_path, charge=0)
    workflow = two_stage_workflow(molecule)
    report, root, _ = generate_and_report(tmp_path, workflow)
    assert report.passed
    (root / "01_relax" / "KPOINTS").write_text(
        "Automatic mesh\n0\nMonkhorst-Pack\n2 2 2\n0 0 0\n",
        encoding="utf-8",
    )
    stage_dirs = {
        StageName.RELAX: root / "01_relax",
        StageName.STATIC: root / "02_static",
    }
    report = run_molecular_preflight(
        workflow, psp_dir=make_psp(tmp_path / "psp2"), stage_dirs=stage_dirs
    )
    assert not report.passed
    assert "KPOINTS_NOT_GAMMA_ONLY" in codes(report)

    # A Gamma 2x2x2 automatic grid must also be rejected.
    (root / "01_relax" / "KPOINTS").write_text(
        "Gamma\n0\nGamma\n2 2 2\n0 0 0\n", encoding="utf-8"
    )
    report = run_molecular_preflight(
        workflow, psp_dir=tmp_path / "psp2", stage_dirs=stage_dirs
    )
    assert "KPOINTS_NOT_GAMMA_ONLY" in codes(report)


def test_kpoints_explicit_single_gamma_accepted(tmp_path):
    molecule = dme_molecule(tmp_path, charge=0)
    workflow = two_stage_workflow(molecule)
    report, root, _ = generate_and_report(tmp_path, workflow)
    (root / "01_relax" / "KPOINTS").write_text(
        "point\n1\n0.0 0.0 0.0 1.0\n", encoding="utf-8"
    )
    stage_dirs = {
        StageName.RELAX: root / "01_relax",
        StageName.STATIC: root / "02_static",
    }
    report = run_molecular_preflight(
        workflow, psp_dir=tmp_path / "psp", stage_dirs=stage_dirs
    )
    assert "KPOINTS_NOT_GAMMA_ONLY" not in codes(report)


def test_dipol_rendered_as_space_separated_string(tmp_path):
    molecule = dme_molecule(tmp_path, charge=0)
    workflow = two_stage_workflow(molecule)
    report, root, _ = generate_and_report(tmp_path, workflow)
    assert report.passed
    for stage_dir in (root / "01_relax", root / "02_static"):
        incar = (stage_dir / "INCAR").read_text(encoding="utf-8")
        assert "DIPOL = 0.5 0.5 0.5" in incar
        assert "[0.5" not in incar
        assert ", 0.5" not in incar


def test_dipol_list_input_renders_correctly(tmp_path):
    molecule = dme_molecule(tmp_path, charge=0)
    workflow = two_stage_workflow(molecule)
    workflow.stages[0].incar["DIPOL"] = [0.5, 0.5, 0.5]
    workflow.stages[1].incar["DIPOL"] = [0.5, 0.5, 0.5]
    _, root, _ = generate_and_report(tmp_path, workflow)
    incar = (root / "01_relax" / "INCAR").read_text(encoding="utf-8")
    assert "DIPOL = 0.5 0.5 0.5" in incar
    assert "[" not in incar


def test_dipol_list_format_in_handwritten_incar_rejected(tmp_path):
    molecule = dme_molecule(tmp_path, charge=0)
    workflow = two_stage_workflow(molecule)
    report, root, _ = generate_and_report(tmp_path, workflow)
    incar = (root / "01_relax" / "INCAR").read_text(encoding="utf-8")
    (root / "01_relax" / "INCAR").write_text(
        incar.replace("DIPOL = 0.5 0.5 0.5", "DIPOL = [0.5, 0.5, 0.5]"),
        encoding="utf-8",
    )
    stage_dirs = {
        StageName.RELAX: root / "01_relax",
        StageName.STATIC: root / "02_static",
    }
    report = run_molecular_preflight(
        workflow, psp_dir=tmp_path / "psp", stage_dirs=stage_dirs
    )
    assert not report.passed
    assert "DIPOL_LIST_FORMAT" in codes(report)
    assert any("Python" in issue.message for issue in report.errors)


def test_ldipol_without_dipol_warns(tmp_path):
    molecule = dme_molecule(tmp_path, charge=0)
    workflow = two_stage_workflow(molecule)
    del workflow.stages[0].incar["DIPOL"]
    report, _, _ = generate_and_report(tmp_path, workflow)
    assert any(issue.code == "LDIPOL_WITHOUT_DIPOL" for issue in report.warnings)


# -- box geometry / collisions ------------------------------------------------


def test_vacuum_insufficient_rejected(tmp_path):
    molecule = dme_molecule(tmp_path, charge=0, box=20.0)
    workflow = two_stage_workflow(molecule)
    report, _, _ = generate_and_report(tmp_path, workflow)
    assert not report.passed
    assert "BOX_VACUUM_INSUFFICIENT" in codes(report)


def test_vacuum_sufficient_accepted(tmp_path):
    molecule = dme_molecule(tmp_path, charge=0, box=30.0)
    workflow = two_stage_workflow(molecule)
    report, _, _ = generate_and_report(tmp_path, workflow)
    assert "BOX_VACUUM_INSUFFICIENT" not in codes(report)


def test_interatomic_collision_rejected(tmp_path):
    coords = np.asarray(
        [[12.0, 12.0, 12.0], [12.0, 12.0, 12.2], [20.0, 20.0, 20.0]]
    )
    molecule = dme_molecule(
        tmp_path,
        charge=0,
        symbols=["C", "H", "O"],
        coords=coords,
        name="collide",
    )
    workflow = two_stage_workflow(molecule)
    report, _, _ = generate_and_report(tmp_path, workflow)
    assert not report.passed
    assert "INTERATOMIC_COLLISION" in codes(report)
    assert any("0.20" in issue.message for issue in report.errors)


# -- stage dependency contracts ----------------------------------------------


def test_orbital_requires_iband_and_wavecar(tmp_path):
    molecule = dme_molecule(tmp_path, charge=0)
    no_iband = orbital_incar(iband=10)
    del no_iband["IBAND"]
    stages = [
        make_stage(StageName.RELAX, incar=relax_incar()),
        make_stage(
            StageName.STATIC,
            depends_on=StageName.RELAX,
            incar=static_incar(),
            required_upstream_outputs=["CONTCAR", "CHGCAR", "WAVECAR"],
            produced_outputs=["WAVECAR", "CHGCAR"],
        ),
        make_stage(
            StageName.ORBITAL,
            depends_on=StageName.STATIC,
            incar=no_iband,
            required_upstream_outputs=["CONTCAR", "CHGCAR", "WAVECAR"],
            produced_outputs=["PARCHG"],
        ),
    ]
    workflow = two_stage_workflow(molecule, stages=stages)
    report, _, _ = generate_and_report(tmp_path, workflow)
    assert not report.passed
    assert "ORBITAL_IBAND_MISSING" in codes(report)

    stages[2].incar["IBAND"] = 39
    stages[1].produced_outputs = ["CHGCAR"]  # no WAVECAR available upstream
    stages[2].required_upstream_outputs = ["CONTCAR", "CHGCAR"]
    workflow = two_stage_workflow(molecule, stages=stages)
    report, _, _ = generate_and_report(tmp_path, workflow)
    assert not report.passed
    assert "ORBITAL_WAVECAR_MISSING" in codes(report)


def test_orbital_with_iband_and_wavecar_passes(tmp_path):
    molecule = dme_molecule(tmp_path, charge=0)
    stages = [
        make_stage(StageName.RELAX, incar=relax_incar()),
        make_stage(
            StageName.STATIC,
            depends_on=StageName.RELAX,
            incar=static_incar(),
            required_upstream_outputs=["CONTCAR", "CHGCAR"],
            produced_outputs=["WAVECAR", "CHGCAR"],
        ),
        make_stage(
            StageName.ORBITAL,
            depends_on=StageName.STATIC,
            incar=orbital_incar(iband=39),
            required_upstream_outputs=["CONTCAR", "CHGCAR", "WAVECAR"],
            produced_outputs=["PARCHG"],
        ),
        make_stage(
            StageName.ESP,
            depends_on=StageName.STATIC,
            incar=esp_incar(),
            required_upstream_outputs=["CONTCAR", "CHGCAR"],
            produced_outputs=["LOCPOT", "CHGCAR"],
        ),
    ]
    workflow = two_stage_workflow(molecule, stages=stages)
    report, _, _ = generate_and_report(tmp_path, workflow)
    assert report.passed, report.errors


def test_relax_downstream_requires_conticar(tmp_path):
    molecule = dme_molecule(tmp_path, charge=0)
    stages = [
        make_stage(StageName.RELAX, incar=relax_incar()),
        make_stage(
            StageName.STATIC,
            depends_on=StageName.RELAX,
            incar=static_incar(),
            required_upstream_outputs=["CHGCAR"],
        ),
    ]
    workflow = two_stage_workflow(molecule, stages=stages)
    report, _, _ = generate_and_report(tmp_path, workflow)
    assert not report.passed
    assert "RELAX_DOWNSTREAM_MISSING_CONTCAR" in codes(report)


def test_restart_requires_wavecar_and_chgcar(tmp_path):
    molecule = dme_molecule(tmp_path, charge=0)
    stages = [
        make_stage(StageName.RELAX, incar=relax_incar()),
        make_stage(
            StageName.STATIC,
            depends_on=StageName.RELAX,
            incar=static_incar(),
            required_upstream_outputs=["CONTCAR", "CHGCAR"],
            produced_outputs=["WAVECAR", "CHGCAR"],
        ),
        make_stage(
            StageName.RESTART,
            depends_on=StageName.STATIC,
            incar=restart_incar(),
            required_upstream_outputs=["CONTCAR"],
        ),
    ]
    workflow = two_stage_workflow(molecule, stages=stages)
    report, _, _ = generate_and_report(tmp_path, workflow)
    assert not report.passed
    assert "RESTART_MISSING_RESTART_FILES" in codes(report)


def test_esp_requires_lvhar_and_relaxed_structure(tmp_path):
    molecule = dme_molecule(tmp_path, charge=0)
    stages = [
        make_stage(StageName.RELAX, incar=relax_incar()),
        make_stage(
            StageName.ESP,
            depends_on=StageName.RELAX,
            incar={
                key: value
                for key, value in esp_incar().items()
                if key != "LVHAR"
            },
            required_upstream_outputs=["CHGCAR"],
        ),
    ]
    workflow = two_stage_workflow(molecule, stages=stages)
    report, _, _ = generate_and_report(tmp_path, workflow)
    assert not report.passed
    assert "ESP_SPEC_MISSING" in codes(report)
    assert "ESP_STRUCTURE_SOURCE_MISSING" in codes(report)


# -- charged-correction policy ------------------------------------------------


def test_charged_without_declared_correction_rejected(tmp_path):
    molecule = dme_molecule(tmp_path, charge=1, multiplicity=2)
    workflow = two_stage_workflow(
        molecule, spin=2, nelect=37.0, lmono=False, dipole=False
    )
    report, _, _ = generate_and_report(tmp_path, workflow)
    assert not report.passed
    assert "CHARGED_CORRECTION_UNDECLARED" in codes(report)


def test_lmono_with_dipole_conflict_rejected(tmp_path):
    molecule = dme_molecule(tmp_path, charge=1, multiplicity=2)
    workflow = two_stage_workflow(
        molecule, spin=2, nelect=37.0, lmono=True, dipole=True
    )
    report, _, _ = generate_and_report(tmp_path, workflow)
    assert not report.passed
    assert "CONFLICTING_CORRECTIONS" in codes(report)


def test_lmono_declared_but_incar_mismatch_rejected(tmp_path):
    molecule = dme_molecule(tmp_path, charge=1, multiplicity=2)
    stages = [
        make_stage(
            StageName.RELAX,
            incar=relax_incar(spin=2, nelect=37.0, lmono=False, dipole=False),
        ),
        make_stage(
            StageName.STATIC,
            depends_on=StageName.RELAX,
            incar=static_incar(spin=2, nelect=37.0, lmono=False, dipole=False),
            required_upstream_outputs=["CONTCAR", "CHGCAR"],
            produced_outputs=["WAVECAR", "CHGCAR"],
        ),
    ]
    workflow = WorkflowSpec(
        molecule=molecule,
        stages=stages,
        scientific_method="static",
        correction_policy=CorrectionPolicy(
            monopole_method=MonopoleMethod.LMONO, dipole=False
        ),
    )
    report, _, _ = generate_and_report(tmp_path, workflow)
    assert not report.passed
    assert "INCAR_CORRECTION_MISMATCH" in codes(report)


# -- VM/TVM blocking ----------------------------------------------------------


def test_vm_tvm_missing_structure_blocked(tmp_path):
    molecule = MoleculeSpec(
        name="TVM",
        structure_path=None,
        total_charge=0,
        spin_multiplicity=1,
        polymer_kind=PolymerKind.TVM,
        blocked_reason="user must provide connectivity, sites, units, caps",
    )
    workflow = two_stage_workflow(molecule)
    report = run_molecular_preflight(workflow, psp_dir=None)
    assert not report.passed
    assert "BLOCKED_MISSING_STRUCTURE" in codes(report)

    # Complete structure but incomplete polymer definition is still blocked.
    molecule = dme_molecule(
        tmp_path,
        charge=0,
        polymer_kind=PolymerKind.VM,
        polymerization=Polymerization(connectivity=""),
        name="VM",
    )
    workflow = two_stage_workflow(molecule)
    report = run_molecular_preflight(workflow, psp_dir=make_psp(tmp_path / "p2"))
    assert not report.passed
    assert "BLOCKED_MISSING_STRUCTURE" in codes(report)


def test_vm_tvm_with_full_definition_not_blocked(tmp_path):
    molecule = dme_molecule(
        tmp_path,
        charge=0,
        polymer_kind=PolymerKind.VM,
        polymerization=Polymerization(
            connectivity="VEC-MBA alternating",
            polymerization_sites=["C3-C4 vinyl"],
            repeat_units=["VEC", "MBA"],
            end_caps=["acetyl", "methyl"],
        ),
        name="VM",
    )
    workflow = two_stage_workflow(molecule)
    report, _, _ = generate_and_report(tmp_path, workflow)
    assert "BLOCKED_MISSING_STRUCTURE" not in codes(report)


def test_generator_refuses_missing_structure(tmp_path):
    molecule = MoleculeSpec(name="TVM", structure_path=None, total_charge=0)
    workflow = two_stage_workflow(molecule)
    generator = MolecularVaspGenerator()
    with pytest.raises(ValueError, match="BLOCKED_MISSING_STRUCTURE"):
        generator.generate(workflow, tmp_path / "out")


# -- POTCAR content policy ----------------------------------------------------


def test_potcar_content_never_written_by_default(tmp_path):
    molecule = dme_molecule(tmp_path, charge=0)
    workflow = two_stage_workflow(molecule)
    report, root, _ = generate_and_report(tmp_path, workflow)
    assert report.passed
    for stage_dir in (root / "01_relax", root / "02_static"):
        assert not (stage_dir / "POTCAR").exists()
        assert (stage_dir / "POTCAR.meta").is_file()
        meta = json.loads((stage_dir / "POTCAR.meta").read_text(encoding="utf-8"))
        assert set(meta["datasets"][0]) == {
            "element",
            "title",
            "zval",
            "enmax",
            "enmin",
        }
    # The persisted report must not leak raw POTCAR-only lines.
    report_text = (root / "preflight.json").read_text(encoding="utf-8")
    assert "mass and valenz" not in report_text
    assert "POTCAR" in report_text  # metadata note is fine


def test_undecleared_materialized_potcar_rejected(tmp_path):
    molecule = dme_molecule(tmp_path, charge=0)
    workflow = two_stage_workflow(molecule)
    report, _, _ = generate_and_report(tmp_path, workflow, write_potcar=True)
    assert not report.passed
    assert "POTCAR_CONTENT_PRESENT" in codes(report)


def test_declared_materialized_potcar_accepted(tmp_path):
    molecule = dme_molecule(tmp_path, charge=0)
    workflow = two_stage_workflow(molecule)
    workflow.potcar_materialized = True
    report, _, _ = generate_and_report(tmp_path, workflow, write_potcar=True)
    assert "POTCAR_CONTENT_PRESENT" not in codes(report)


# -- resources -----------------------------------------------------------------


def test_ncore_incompatible_with_slurm_tasks_rejected(tmp_path):
    molecule = dme_molecule(tmp_path, charge=0)
    stages = [
        make_stage(StageName.RELAX, incar=relax_incar(ncore=6)),
        make_stage(
            StageName.STATIC,
            depends_on=StageName.RELAX,
            incar=static_incar(ncore=6),
            required_upstream_outputs=["CONTCAR", "CHGCAR"],
            produced_outputs=["WAVECAR", "CHGCAR"],
        ),
    ]
    workflow = two_stage_workflow(molecule, stages=stages, tasks_per_node=8)
    report, _, _ = generate_and_report(tmp_path, workflow)
    assert not report.passed
    assert "NCORE_INCOMPATIBLE" in codes(report)
    assert any("8" in issue.message for issue in report.errors)


def test_ncore_compatible_accepted(tmp_path):
    molecule = dme_molecule(tmp_path, charge=0)
    workflow = two_stage_workflow(molecule, ncore=4, tasks_per_node=8)
    report, _, _ = generate_and_report(tmp_path, workflow)
    assert "NCORE_INCOMPATIBLE" not in codes(report)


def test_npar_incompatible_with_slurm_tasks_rejected(tmp_path):
    molecule = dme_molecule(tmp_path, charge=0)
    stages = [
        make_stage(
            StageName.RELAX,
            incar={
                key: value
                for key, value in relax_incar().items()
                if key != "NCORE"
            }
            | {"NPAR": 6},
        ),
    ]
    workflow = two_stage_workflow(molecule, stages=stages, tasks_per_node=8)
    report, _, _ = generate_and_report(tmp_path, workflow)
    assert not report.passed
    assert "NPAR_INCOMPATIBLE" in codes(report)


# -- stage DAG -----------------------------------------------------------------


def test_stage_dag_duplicate_names_rejected():
    molecule = MoleculeSpec(name="x", structure_path=None, total_charge=0)
    stages = [
        make_stage(StageName.RELAX, incar=relax_incar()),
        make_stage(StageName.RELAX, incar=relax_incar()),
    ]
    with pytest.raises(ValueError, match="unique"):
        WorkflowSpec(molecule=molecule, stages=stages, scientific_method="test")


def test_stage_dag_unknown_or_backward_dependency_rejected():
    molecule = MoleculeSpec(name="x", structure_path=None, total_charge=0)
    stages = [
        make_stage(
            StageName.STATIC,
            depends_on=StageName.RELAX,
            incar=static_incar(),
        ),
        make_stage(StageName.RELAX, incar=relax_incar()),
    ]
    with pytest.raises(ValueError, match="earlier"):
        WorkflowSpec(molecule=molecule, stages=stages, scientific_method="test")
    stages = [
        make_stage(
            StageName.RELAX,
            depends_on=StageName.ESP,
            incar=relax_incar(),
        ),
    ]
    with pytest.raises(ValueError, match="unknown stage"):
        WorkflowSpec(molecule=molecule, stages=stages, scientific_method="test")


# -- structure readers --------------------------------------------------------


def test_mol_and_sdf_structures_readable(tmp_path):
    mol_path = tmp_path / "water.mol"
    mol_path.write_text(
        "water\n"
        "  molfile\n"
        "  RDKit          2D\n"
        "\n"
        "  3  2  0  0  0  0  0  0  0  0999 V2000\n"
        "    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "    0.9559    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "   -0.2400    0.9259    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "  1  2  1  0  0  0  0\n"
        "  1  3  1  0  0  0  0\n"
        "M  END\n",
        encoding="utf-8",
    )
    molecule = MoleculeSpec(
        name="water",
        structure_path=mol_path,
        total_charge=0,
        spin_multiplicity=1,
    )
    workflow = two_stage_workflow(molecule)
    report, _, _ = generate_and_report(tmp_path, workflow)
    assert report.summary is not None
    assert report.summary.formula == "O1H2"

    sdf_path = tmp_path / "water.sdf"
    content = mol_path.read_text(encoding="utf-8") + "$$$$\n"
    content += (
        "ethanol\n  molfile\n  RDKit\n\n"
        "  3  2  0  0  0  0  0  0  0  0999 V2000\n"
        "    1.0000    1.0000    1.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "    2.4000    1.0000    1.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "    3.4000    1.0000    1.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "  1  2  1  0  0  0  0\n"
        "  2  3  1  0  0  0  0\n"
        "M  END\n"
        "$$$$\n"
    )
    sdf_path.write_text(content, encoding="utf-8")
    molecule = MoleculeSpec(
        name="ethanol_second_block",
        structure_path=sdf_path,
        total_charge=0,
        spin_multiplicity=1,
        conformer_id="1",
    )
    workflow = two_stage_workflow(molecule)
    report, _, _ = generate_and_report(tmp_path, workflow)
    assert report.summary is not None
    assert report.summary.formula == "C2O1"


def test_xyz_atom_count_mismatch_rejected(tmp_path):
    path = tmp_path / "bad.xyz"
    path.write_text("16\ncomment\n" + "C 10 10 10\n" * 15, encoding="utf-8")
    molecule = MoleculeSpec(name="bad", structure_path=path, total_charge=0)
    workflow = two_stage_workflow(molecule)
    report = run_molecular_preflight(workflow)
    assert not report.passed
    assert "STRUCTURE_ATOM_COUNT_MISMATCH" in codes(report)


def test_poscar_per_atom_element_line_rejected(tmp_path):
    path = tmp_path / "dme_per_atom.POSCAR"
    symbols = ["C", "O", "C", "C", "O", "C"] + ["H"] * 10
    counts = " ".join(["1"] * len(symbols))
    rng = np.random.default_rng(4)
    frac = rng.uniform(0.3, 0.7, size=(len(symbols), 3))
    rows = "\n".join(" ".join(f"{value:.9f}" for value in row) for row in frac)
    path.write_text(
        "bad per-atom element line\n"
        "1.0\n"
        "30.0 0.0 0.0\n0.0 30.0 0.0\n0.0 0.0 30.0\n"
        + " ".join(symbols)
        + "\n"
        + counts
        + "\nDirect\n"
        + rows
        + "\n",
        encoding="utf-8",
    )
    molecule = MoleculeSpec(
        name="bad_poscar",
        structure_path=path,
        structure_kind="poscar",
        total_charge=0,
    )
    workflow = two_stage_workflow(molecule)
    report = run_molecular_preflight(workflow)
    assert not report.passed
    assert "POSCAR_ELEMENT_BLOCKS_INVALID" in codes(report)


def test_poscar_vasp4_without_element_line_rejected(tmp_path):
    path = tmp_path / "vasp4.POSCAR"
    path.write_text(
        "vasp4 style\n"
        "1.0\n"
        "30.0 0.0 0.0\n0.0 30.0 0.0\n0.0 0.0 30.0\n"
        "16\n"
        "Direct\n"
        + "\n".join("0.4 0.5 0.5" for _ in range(16))
        + "\n",
        encoding="utf-8",
    )
    molecule = MoleculeSpec(
        name="vasp4",
        structure_path=path,
        structure_kind="poscar",
        total_charge=0,
    )
    workflow = two_stage_workflow(molecule)
    report = run_molecular_preflight(workflow)
    assert not report.passed
    assert "POSCAR_ELEMENT_BLOCKS_INVALID" in codes(report)


# -- report artifacts ---------------------------------------------------------


def test_preflight_json_written_and_agent_text_is_short(tmp_path):
    molecule = dme_molecule(tmp_path, charge=0)
    workflow = two_stage_workflow(molecule)
    report, root, _ = generate_and_report(tmp_path, workflow)
    assert (root / "preflight.json").is_file()
    persisted = json.loads((root / "preflight.json").read_text(encoding="utf-8"))
    assert persisted["passed"] is True
    assert persisted["summary"]["nelect"] == 38.0
    assert (root / "workflow.json").is_file()
    text = render_agent_text(report)
    assert "PREFLIGHT PASSED" in text
    assert len(text) < 2000
    # Agent text must never dump source code or full file content.
    assert "INCAR = " not in text
    assert "Direct" not in text


def test_charge_is_never_inferred_from_name(tmp_path):
    # A file named DME_Li must NOT silently pick q=+1: total_charge is
    # explicit, and omitting it raises immediately at model construction.
    with pytest.raises(ValueError):
        MoleculeSpec(name="DME_Li", structure_path=tmp_path / "x.xyz")  # type: ignore[call-arg]
