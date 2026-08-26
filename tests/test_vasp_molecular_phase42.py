"""Phase 4.2 offline regressions: long-cycle realism for real VASP runs.

Covers, with synthetic fixtures only (no SSH, no Slurm, no VASP execution):

* Slurm COMPLETED with an NSW-exhausted relax can NEVER reach VALIDATED;
* relax acceptance is OUTCAR-grounded (formal marker + max force vs EDIFFG;
  adjacent-step total-energy differences are never the criterion);
* vasprun.xml reads the LAST <calculation> (multi-step relax) and counts
  all ionic steps without building the whole tree;
* real dipol+quadrupol correction lines (plain decimal AND scientific
  notation, VASP 5.4/6.x wording);
* spin semantics: spin_multiplicity is never ISPIN (triplet -> ISPIN=2),
  NUPDOWN mapping assumptions are recorded, ISPIN is {1,2} only;
* streaming/chunked LOCPOT (x-fastest planar accumulation, six vacuum
  faces for 0.5/1.0/1.5/2.0 A) with no 40,000,000-point failure.
"""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from photomatagent.scientific.applications.vasp.molecular.models import (
    MoleculeSpec,
    StageName,
    WorkflowSpec,
)
from photomatagent.scientific.applications.vasp.molecular.preflight import (
    run_molecular_preflight,
)
from photomatagent.scientific.applications.vasp.molecular.results import (
    MAX_IN_MEMORY_GRID_POINTS,
    analyze_outcar_convergence,
    analyze_result_dir,
    esp_metadata,
    parse_vasprun,
    read_locpot,
    read_locpot_header,
    scientific_evidence,
    stream_locpot_planar,
    vacuum_summary_all_thicknesses,
)
from photomatagent.scientific.applications.vasp.molecular.templates import (
    make_stage,
    relax_incar,
    static_incar,
)
from photomatagent.scientific.applications.vasp.molecular.workflow import (
    build_molecule_workflow,
)


def read_outcar_corrections(path: Path) -> dict:
    from photomatagent.scientific.applications.vasp.molecular.results import (
        read_outcar_corrections,
    )

    return read_outcar_corrections(path)


# ---------------------------------------------------------------------------
# shared synthetic-result helpers
# ---------------------------------------------------------------------------


def cluster_coords(n: int, box: float, spacing: float = 1.6) -> np.ndarray:
    pattern = [(i % 4, (i // 4) % 2, (i // 8) % 3) for i in range(n)]
    points = np.asarray(pattern, dtype=float) * spacing
    points -= points.mean(axis=0)
    points += box / 2
    return points


def write_poscar(path: Path, *, n_atoms: int = 16, box: float = 30.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    symbols = ["C"] * 4 + ["O"] * 2 + ["H"] * 10
    coords = cluster_coords(len(symbols), box)
    lines = [
        "phase42 synthetic",
        "1.0",
        f"{box} 0.0 0.0",
        f"0.0 {box} 0.0",
        f"0.0 0.0 {box}",
        "C O H",
        "4 2 10",
        "Direct",
    ]
    lines += [
        f"{x / box:.6f} {y / box:.6f} {z / box:.6f}"
        for x, y, z in coords
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_eigenval(path: Path, *, nelect: int = 38) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    homo = nelect // 2
    nbands = max(40, homo + 6)
    lines = [
        "   17   17    1    1",
        "  0.2000000E+03  0.2000000E-08  0.2000000E-08  0.2000000E-08",
        "  1.000000000000000E-004",
        "  CAR ",
        "  synthetic-eigenval",
        f"     {nelect}     1    {nbands}",
        "",
        "  0.0000000E+00  0.0000000E+00  0.0000000E+00  0.1000000E+01",
    ]
    for band in range(1, nbands + 1):
        if band <= homo:
            lines.append(
                f"{band:6d} {-6.4 - (homo - band) * 0.18:16.6f} {1.0:10.6f}"
            )
        else:
            lines.append(
                f"{band:6d} {-2.2 + (band - homo - 1) * 0.12:16.6f} {0.0:10.6f}"
            )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_oszicar(path: Path, *, e0: float = -122.27635471) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "DAV:   1    -0.121765040995E+03   -0.12177E+03   -0.71368E-05   144   0.905E-02    0.696E-03\n"
        "DAV:   2    -0.121765040523E+03    0.47271E-06   -0.20602E-05   120   0.197E-02    0.462E-03\n"
        "DAV:   3    -0.121765040541E+03   -0.18559E-07   -0.19949E-06   104   0.999E-03\n"
        f"   1 F= {e0: 14.8E} E0= {e0: 14.8E}  d E =0.000E+00\n"
        f"   2 F= {e0: 14.8E} E0= {e0: 14.8E}  d E =0.000E+00\n"
        f"   3 F= {e0: 14.8E} E0= {e0: 14.8E}  d E =0.000E+00\n",
        encoding="utf-8",
    )


def write_vasprun_multi(
    path: Path,
    *,
    energies: list[float],
    scsteps_per_calc: int = 2,
) -> None:
    """vasprun.xml with one <calculation> PER ionic step (relax format)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("modeling")
    for value in energies:
        calc = ET.SubElement(root, "calculation")
        for _ in range(scsteps_per_calc):
            scstep = ET.SubElement(calc, "scstep")
            energy = ET.SubElement(scstep, "energy")
            ET.SubElement(energy, "v", {"name": "e_fr_energy"}).text = f"{value:.8f}"
            ET.SubElement(energy, "v", {"name": "e_0_energy"}).text = f"{value:.8f}"
        energy = ET.SubElement(calc, "energy")
        ET.SubElement(energy, "v", {"name": "e_fr_energy"}).text = f"{value:.8f}"
        ET.SubElement(energy, "v", {"name": "e_0_energy"}).text = f"{value:.8f}"
        ET.SubElement(energy, "v", {"name": "eentropy"}).text = "0.00000000"
        structure = ET.SubElement(calc, "structure", {"name": "finalpos"})
        varray = ET.SubElement(structure, "varray", {"name": "positions"})
        for index in range(16):
            ET.SubElement(varray, "v").text = (
                f"{0.4 + index * 0.001:.6f} {0.5:.6f} {0.5:.6f}"
            )
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def write_outcar(
    path: Path,
    *,
    n_atoms: int = 16,
    max_force: float = 0.005,
    reached: bool = True,
    force_blocks: int = 1,
    error_tokens: list[str] | None = None,
) -> None:
    """Realistic OUTCAR fixture: TOTAL-FORCE blocks per ionic step."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        " vasp.5.4.4.18Apr17 (build Mar 03 2024 15:47:24) complex parallel",
        "   NSW      =      200     number of steps for ionic motion",
        "   IBRION   =      2     ionic relax: 1=quasi-Newton, 2=damped",
        "   EDIFFG   = -0.02E+00  force-criterion for ionic relax",
    ]
    for token in error_tokens or []:
        lines.append(token)
    if reached:
        lines.append(
            "  reached required accuracy - stopping structural energy minimisation"
        )
    for block in range(force_blocks):
        if block == force_blocks - 1:
            component = max_force / max(1.0, (3 * n_atoms) ** 0.5)
        else:
            component = 0.4  # earlier steps far from convergence
        lines.append("POSITION                                       TOTAL-FORCE (eV/Angst)")
        lines.append("-" * 90)
        for index in range(n_atoms):
            lines.append(
                f"{index + 1:6d} {0.0:17.10f} {0.0:17.10f} {0.0:17.10f}"
                f"{component:14.8f} {component:14.8f} {component:14.8f}"
            )
        lines.append("-" * 90)
    lines.append("   1 F= -0.122276354712E+03 E0= -0.122276354712E+03")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def seed_result_dir(
    tmp_path: Path,
    *,
    relax: bool,
    max_force: float = 0.005,
    reached: bool = True,
    nsw: int = 200,
    force_blocks: int = 1,
    error_tokens: list[str] | None = None,
) -> Path:
    """A complete result directory satisfying static or relax contracts."""
    result_dir = tmp_path / ("relax" if relax else "static")
    result_dir.mkdir(parents=True, exist_ok=True)
    write_poscar(result_dir / "POSCAR")
    shutil.copy2(result_dir / "POSCAR", result_dir / "CONTCAR")
    write_eigenval(result_dir / "EIGENVAL", nelect=38)
    write_oszicar(result_dir / "OSZICAR")
    if relax:
        (result_dir / "INCAR").write_text(
            "SYSTEM = phase42 relax\n"
            "ENCUT = 400\n"
            "EDIFF = 1E-6\n"
            f"NSW = {nsw}\n"
            "EDIFFG = -0.02\n"
            "ISPIN = 1\n"
            "IBRION = 2\n",
            encoding="utf-8",
        )
        write_outcar(
            result_dir / "OUTCAR",
            max_force=max_force,
            reached=reached,
            force_blocks=force_blocks,
            error_tokens=error_tokens,
        )
    else:
        (result_dir / "INCAR").write_text(
            "SYSTEM = phase42 static\n"
            "ENCUT = 400\n"
            "EDIFF = 1E-6\n"
            "NSW = 0\n"
            "ISPIN = 1\n"
            "IBRION = -1\n",
            encoding="utf-8",
        )
    write_vasprun_multi(result_dir / "vasprun.xml", energies=[-121.0, -122.1, -122.27635471])
    (result_dir / "POTCAR.meta").write_text(
        json.dumps({"nelect": 38.0, "neutral_valence_electrons": 38.0}),
        encoding="utf-8",
    )
    return result_dir


# ---------------------------------------------------------------------------
# A2: relax acceptance via OUTCAR force criterion
# ---------------------------------------------------------------------------


def test_slurm_completed_with_nsw_exhausted_never_validated(tmp_path):
    result_dir = seed_result_dir(
        tmp_path,
        relax=True,
        max_force=0.5,          # far above |EDIFFG| = 0.02
        reached=False,          # no formal marker
        nsw=3,
        force_blocks=4,         # 4 force blocks -> >= NSW=3 consumed
    )
    results = analyze_result_dir(
        result_dir, charge=0, spin_multiplicity=1, box_ang=30.0
    )
    assert results["validated"] is False
    convergence = results["convergence"]
    assert convergence["exhausted_nsw"] is True
    assert convergence["ionic_converged"] is False
    assert convergence["max_force_ev_ang"] > 0.02
    assert any("IONIC_NOT_CONVERGED" in error for error in results["errors"])
    assert any("NSW" in error for error in results["errors"])
    # No scientific evidence may exist for a Slurm-COMPLETED but unconverged
    # relax: the registry state would stay COLLECTED at the workflow layer.
    assert scientific_evidence(results) == []


def test_outcar_formal_force_convergence_validates(tmp_path):
    result_dir = seed_result_dir(tmp_path, relax=True, max_force=0.005, reached=True)
    results = analyze_result_dir(
        result_dir, charge=0, spin_multiplicity=1, box_ang=30.0
    )
    assert results["validated"] is True, results["errors"]
    convergence = results["convergence"]
    assert convergence["reached_required_accuracy"] is True
    assert convergence["ionic_converged"] is True
    assert convergence["max_force_ev_ang"] <= 0.02
    assert results["ionic"]["converged"] is True


def test_max_force_above_ediffg_fails_even_with_marker(tmp_path):
    result_dir = seed_result_dir(
        tmp_path, relax=True, max_force=0.5, reached=True
    )
    results = analyze_result_dir(
        result_dir, charge=0, spin_multiplicity=1, box_ang=30.0
    )
    assert results["validated"] is False
    assert results["convergence"]["reached_required_accuracy"] is True
    assert results["convergence"]["ionic_converged"] is False
    assert any("IONIC_NOT_CONVERGED" in error for error in results["errors"])
    assert results["convergence"]["max_force_ev_ang"] > 0.02
    assert scientific_evidence(results) == []


def test_detected_vasp_errors_never_validated(tmp_path):
    result_dir = seed_result_dir(
        tmp_path,
        relax=True,
        reached=False,
        error_tokens=[
            "BRMIX: serious error",
            "ZHEGV: some diagonalization failed",
        ],
    )
    results = analyze_result_dir(
        result_dir, charge=0, spin_multiplicity=1, box_ang=30.0
    )
    assert results["validated"] is False
    assert any("DETECTED_VASP_ERROR" in error for error in results["errors"])
    detected = results["convergence"]["detected_errors"]
    assert any("brmix" in token for token in detected)
    assert any("zhegv" in token for token in detected)


def test_adjacent_step_denergy_is_never_the_criterion(tmp_path):
    # Ionic F values in OSZICAR are IDENTICAL between steps (dE = 0) but the
    # OUTCAR forces are large and no formal marker exists: the run must be
    # judged by the FORCE criterion, not by dE.
    result_dir = seed_result_dir(tmp_path, relax=True, max_force=0.5, reached=False)
    results = analyze_result_dir(
        result_dir, charge=0, spin_multiplicity=1, box_ang=30.0
    )
    assert results["validated"] is False
    assert any("IONIC_NOT_CONVERGED" in error for error in results["errors"])
    assert results["convergence"]["ionic_converged"] is False


def test_static_ionic_convergence_is_not_applicable(tmp_path):
    result_dir = seed_result_dir(tmp_path, relax=False)
    results = analyze_result_dir(
        result_dir, charge=0, spin_multiplicity=1, box_ang=30.0
    )
    assert results["validated"] is True, results["errors"]
    assert results["ionic"]["static_single_point"] is True
    assert results["convergence"]["applicable"] is False
    assert results["ionic"]["converged"] is True


# ---------------------------------------------------------------------------
# A3: vasprun.xml last calculation + ionic step count
# ---------------------------------------------------------------------------


def test_vasprun_multi_calculation_uses_last_energy_and_structure(tmp_path):
    path = tmp_path / "vasprun.xml"
    write_vasprun_multi(path, energies=[-121.0, -122.1, -122.27635471])
    data = parse_vasprun(path)
    assert data.final_f_ev == pytest.approx(-122.27635471)
    assert data.final_e0_ev == pytest.approx(-122.27635471)
    assert data.ionic_steps == 2  # three calculations -> two ionic steps
    assert data.n_atoms == 16


def test_vasprun_single_calculation_static_compat(tmp_path):
    path = tmp_path / "vasprun.xml"
    write_vasprun_multi(path, energies=[-122.27635471])
    data = parse_vasprun(path)
    assert data.final_f_ev == pytest.approx(-122.27635471)
    assert data.ionic_steps == 0
    assert data.n_atoms == 16


# ---------------------------------------------------------------------------
# A4: dipol+quadrupol correction parsing (real wording)
# ---------------------------------------------------------------------------


def test_dipol_quadrupol_energy_correction_plain_decimal(tmp_path):
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(
        "dipol+quadrupol energy correction           -0.000154 eV\n"
        "dipole moment                            -0.000021 eAng\n",
        encoding="utf-8",
    )
    parsed = read_outcar_corrections(outcar)
    assert parsed["dipole_quadrupole_ev"] == pytest.approx(-0.000154)
    assert parsed["dipole_quadrupole_moment_eang"] is None


def test_dipol_quadrupol_correction_scientific_notation(tmp_path):
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(
        "dipol+quadrupol energy correction           -1.54000E-04 eV\n",
        encoding="utf-8",
    )
    parsed = read_outcar_corrections(outcar)
    assert parsed["dipole_quadrupole_ev"] == pytest.approx(-1.54e-4)


def test_dipol_quadrupol_moment_vasp54_wording(tmp_path):
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(
        "dipol+quadrupol moment           -0.000161 eAng\n",
        encoding="utf-8",
    )
    parsed = read_outcar_corrections(outcar)
    assert parsed["dipole_quadrupole_ev"] is None
    assert parsed["dipole_quadrupole_moment_eang"] == pytest.approx(-0.000161)


def test_monopole_correction_wording(tmp_path):
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(
        "monopole energy correction            0.000123 eV\n"
        "monopole moment                       0.12345678E+00 eAng\n",
        encoding="utf-8",
    )
    parsed = read_outcar_corrections(outcar)
    assert parsed["monopole_ev"] == pytest.approx(0.000123)
    assert parsed["monopole_moment_eang"] == pytest.approx(0.12345678)


# ---------------------------------------------------------------------------
# A5: spin semantics
# ---------------------------------------------------------------------------


def _molecule(tmp_path: Path, *, charge: int, multiplicity: int, **extra) -> MoleculeSpec:
    path = tmp_path / f"spin_{charge}_{multiplicity}.xyz"
    coords = cluster_coords(16, 30.0)
    path.write_text(
        "\n".join(
            [str(16), "spin fixture"]
            + [
                f"{'C' if i < 16 else 'H'} {x:.5f} {y:.5f} {z:.5f}"
                for i, (x, y, z) in enumerate(coords)
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return MoleculeSpec(
        name="spin_fixture",
        structure_path=path,
        total_charge=charge,
        spin_multiplicity=multiplicity,
        box_ang=30.0,
        **extra,
    )


def test_singlet_defaults_to_ispin1(tmp_path):
    molecule = _molecule(tmp_path, charge=0, multiplicity=1)
    assert molecule.effective_ispin() == 1
    workflow = build_molecule_workflow(molecule, psp_dir=None)
    relax = workflow.stages[0]
    assert relax.incar["ISPIN"] == 1
    assert workflow.provenance["spin"]["ispin"] == 1


def test_doublet_defaults_to_ispin2(tmp_path):
    molecule = _molecule(tmp_path, charge=1, multiplicity=2)
    assert molecule.effective_ispin() == 2
    workflow = build_molecule_workflow(molecule, psp_dir=None)
    assert workflow.stages[0].incar["ISPIN"] == 2
    assert workflow.provenance["spin"]["ispin"] == 2


def test_triplet_never_becomes_ispin3(tmp_path):
    molecule = _molecule(tmp_path, charge=0, multiplicity=3)
    assert molecule.effective_ispin() == 2  # multiplicity 3 -> polarized, not 3
    workflow = build_molecule_workflow(molecule, psp_dir=None)
    assert workflow.stages[0].incar["ISPIN"] == 2
    assert workflow.stages[0].incar.get("ISPIN") != 3
    # The model layer itself refuses ISPIN outside {1, 2}.
    with pytest.raises(ValueError, match="Input should be 1 or 2"):
        _molecule(tmp_path, charge=0, multiplicity=3, ispin=3)


def test_preflight_rejects_ispin_outside_1_2(tmp_path):
    molecule = _molecule(tmp_path, charge=0, multiplicity=1)
    stages = [
        make_stage(StageName.RELAX, incar=relax_incar(spin=3)),
    ]
    workflow = WorkflowSpec(
        molecule=molecule, stages=stages, scientific_method="spin test"
    )
    report = run_molecular_preflight(workflow)
    assert not report.passed
    assert any(issue.code == "ISPIN_INVALID" for issue in report.errors)


def test_nupdown_mapping_assumption_recorded_and_validated(tmp_path):
    molecule = _molecule(tmp_path, charge=0, multiplicity=3, nupdown=2)
    assert any("multiplicity - 1 = 2" in note for note in molecule.spin_assumptions())
    workflow = build_molecule_workflow(molecule, psp_dir=None)
    assert workflow.stages[0].incar["NUPDOWN"] == 2
    report = run_molecular_preflight(workflow, psp_dir=psp_library(tmp_path))
    assert report.passed, report.errors

    # A stray NUPDOWN that contradicts the multiplicity mapping warns.
    # (nupdown=4 keeps NELECT parity but breaks the nupdown = multiplicity-1
    # mapping assumption, so it must warn instead of silently passing.)
    mismatch = _molecule(tmp_path, charge=0, multiplicity=3, nupdown=4)
    assert any("does NOT match" in note for note in mismatch.spin_assumptions())
    workflow_m = build_molecule_workflow(mismatch, psp_dir=None)
    report_m = run_molecular_preflight(workflow_m, psp_dir=psp_library(tmp_path / "m"))
    assert report_m.passed
    assert any(
        issue.code == "NUPDOWN_MAPPING_MISMATCH" for issue in report_m.warnings
    )


def psp_library(tmp_path: Path) -> Path:
    library = tmp_path / "psp"
    for element, zval, enmax in (
        ("C", 4.0, 400.0),
        ("O", 6.0, 400.0),
        ("H", 1.0, 250.0),
    ):
        dataset = library / element
        dataset.mkdir(parents=True, exist_ok=True)
        (dataset / "POTCAR").write_text(
            f"   TITEL  = PAW_PBE {element} 08Apr2002\n"
            f"   POMASS =     1.000; ZVAL   =    {zval:.3f}    mass and valenz\n"
            f"   ENMAX  =  {enmax:.3f}; ENMIN  =  300.000 eV\n",
            encoding="utf-8",
        )
    return library


def test_nupdown_parity_inconsistent_rejected(tmp_path):
    molecule = _molecule(tmp_path, charge=0, multiplicity=3, nupdown=2)
    stages = [
        make_stage(
            StageName.RELAX,
            incar=relax_incar(spin=2, nupdown=5),  # exceed NELECT parity
        ),
    ]
    workflow = WorkflowSpec(
        molecule=molecule,
        stages=stages,
        scientific_method="spin test",
        correction_policy={
            "monopole_method": "none",
            "dipole": True,
        },
    )
    report = run_molecular_preflight(workflow, psp_dir=psp_library(tmp_path / "p"))
    # NELECT=38, NUPDOWN=5 -> parity (38-5)%2==1 -> inconsistent
    assert any(
        issue.code == "NUPDOWN_PARITY_INCONSISTENT" for issue in report.errors
    )


# ---------------------------------------------------------------------------
# A1: streaming LOCPOT
# ---------------------------------------------------------------------------


def _write_locpot(path: Path, box: float = 20.0, grid=(8, 8, 8)) -> None:
    nx, ny, nz = grid
    flat = np.arange(nx * ny * nz, dtype=np.float64)  # value == flat index
    lines = [
        "phase42 synthetic LOCPOT",
        f"{1.0:20.12E}",
        f"{box} 0.0 0.0",
        f"0.0 {box} 0.0",
        f"0.0 0.0 {box}",
        "C O H Li",
        "4 2 10 1",
        "Direct",
    ]
    for _ in range(17):
        lines.append("0.5 0.5 0.5")
    lines.append(f"{nx} {ny} {nz}")
    for start in range(0, flat.size, 5):
        lines.append(
            " ".join(f"{value:20.12E}" for value in flat[start : start + 5])
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_streaming_planar_matches_in_memory_x_fastest(tmp_path):
    grid = (4, 3, 2)
    path = tmp_path / "LOCPOT"
    _write_locpot(path, box=12.0, grid=grid)
    header = read_locpot_header(path)
    assert header.grid == grid
    assert header.lattice_lengths_ang == pytest.approx([12.0, 12.0, 12.0])
    planar = stream_locpot_planar(path, header)
    legacy = read_locpot(path, box_ang=12.0)
    for stats in planar:
        axis = stats.axis
        other = [a for a in range(3) if a != axis]
        expected = legacy.data.mean(axis=tuple(other))
        assert stats.plane_means == pytest.approx(expected)


def test_six_vacuum_faces_and_thicknesses(tmp_path):
    path = tmp_path / "LOCPOT_flat"
    nx = ny = nz = 10
    flat = np.full(nx * ny * nz, 2.0)
    lines = [
        "synthetic LOCPOT",
        f"{1.0:20.12E}",
        "20.0 0.0 0.0",
        "0.0 20.0 0.0",
        "0.0 0.0 20.0",
        "C O H Li",
        "4 2 10 1",
        "Direct",
    ]
    for _ in range(17):
        lines.append("0.5 0.5 0.5")
    lines.append(f"{nx} {ny} {nz}")
    for start in range(0, flat.size, 5):
        lines.append(" ".join(f"{v:20.12E}" for v in flat[start : start + 5]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    by_thickness = vacuum_summary_all_thicknesses(path)
    assert set(by_thickness) == {0.5, 1.0, 1.5, 2.0}
    summary = by_thickness[1.0]
    assert len(summary.faces) == 6
    assert summary.mean_ev == pytest.approx(2.0)
    assert summary.std_ev == pytest.approx(0.0)
    assert summary.range_ev == pytest.approx(0.0)
    assert summary.stability == "stable"
    for face in summary.faces:
        assert face.thickness_ang == 1.0
        assert face.side in {"low", "high"}


def test_unstable_six_face_vacuum_flagged(tmp_path):
    path = tmp_path / "LOCPOT_tilted"
    n = 8
    flat = np.zeros(n**3)
    iz = 0
    for k in range(n**3):
        iz = k // (n * n)
        flat[k] = 0.05 * iz
    lines = [
        "synthetic LOCPOT",
        f"{1.0:20.12E}",
        "20.0 0.0 0.0",
        "0.0 20.0 0.0",
        "0.0 0.0 20.0",
        "C O H Li",
        "4 2 10 1",
        "Direct",
    ]
    for _ in range(17):
        lines.append("0.5 0.5 0.5")
    lines.append(f"{n} {n} {n}")
    for start in range(0, flat.size, 5):
        lines.append(" ".join(f"{v:20.12E}" for v in flat[start : start + 5]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = vacuum_summary_all_thicknesses(path)[1.0]
    assert summary.stability == "unstable"
    assert summary.std_ev > 0.1


def test_large_grid_streams_beyond_40m_cap_and_header_only_esp(tmp_path):
    """A >40,000,000-point LOCPOT must not fail via the streaming API and
    must never be rejected by the old in-memory limit masquerading as a fix."""
    grid = (256, 400, 400)  # 40,960,000 points > 40,000,000
    assert np.prod(grid) > MAX_IN_MEMORY_GRID_POINTS
    path = tmp_path / "LOCPOT"
    header_text = (
        "synthetic LOCPOT\n"
        f"{1.0:20.12E}\n"
        "20.0 0.0 0.0\n"
        "0.0 20.0 0.0\n"
        "0.0 0.0 20.0\n"
        "C O H Li\n"
        "4 2 10 1\n"
        "Direct\n"
        + "".join("0.5 0.5 0.5\n" for _ in range(17))
        + f"{grid[0]} {grid[1]} {grid[2]}\n"
    )
    with path.open("wb") as handle:
        handle.write(header_text.encode("ascii"))
        chunk = b"0 " * (4 * 1024 * 1024)
        written = 0
        total_tokens = int(np.prod(grid))
        while written < total_tokens:
            n_tokens = min(total_tokens - written, 4 * 1024 * 1024)
            handle.write(b"0 " * n_tokens)
            written += n_tokens
            del chunk
            chunk = b"0 " * min(4 * 1024 * 1024, total_tokens - written)
    header = read_locpot_header(path)
    assert header.grid == grid
    planar = stream_locpot_planar(path, header)
    assert planar[0].plane_means == pytest.approx(np.zeros(grid[0]))
    by_thickness = vacuum_summary_all_thicknesses(path)
    assert by_thickness[1.0].mean_ev == pytest.approx(0.0)
    # esp_metadata only touches the header, never the ~82 MB body.
    meta = esp_metadata(tmp_path)
    assert meta["has_locpot"] is True
    assert meta["grid"] == list(grid)
    assert meta["data_offset_bytes"] > 0
    # The legacy in-memory reader still refuses (cap unchanged by design).
    with pytest.raises(ValueError, match="too large for the in-memory reader"):
        read_locpot(path)
