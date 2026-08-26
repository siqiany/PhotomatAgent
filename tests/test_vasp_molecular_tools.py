"""Offline tests for the molecular VASP tool surface, DAG runner and parses.

Golden regression fixtures come from the verified TFPMA smoke run
(gel_electrolyte_dft/codex_run/results/tfpma_smoke_corrected_static_clean),
copied here as small license-free output files (no POTCAR/WAVECAR/CHGCAR).
All other inputs are synthetic; no SSH, Slurm or VASP is ever invoked.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from photomatagent.scientific.applications.vasp.molecular.binding import (
    BindingEnergyInput,
    BindingReference,
)
from photomatagent.scientific.applications.vasp.molecular.models import (
    MoleculeSpec,
    PolymerKind,
    Polymerization,
    ResourceProfile,
    ScreenDecision,
    StageName,
)
from photomatagent.scientific.applications.vasp.molecular.preflight import (
    run_molecular_preflight,
)
from photomatagent.scientific.applications.vasp.molecular.results import (
    analyze_result_dir,
    determine_orbital_bands,
    parse_eigenval,
    parse_oszicar,
    read_locpot,
    scientific_evidence,
    vacuum_level,
)
from photomatagent.scientific.applications.vasp.molecular.tools import (
    MolecularVaspTools,
)
from photomatagent.scientific.applications.vasp.molecular.workflow import (
    StageTask,
    build_molecule_workflow,
    determine_ibands,
    load_task_state,
    run_molecule_workflow,
)
from photomatagent.scientific.remote.fake import FakeSCNetBackend
from photomatagent.scientific.remote.lifecycle import SubmitOnceSession
from photomatagent.scientific.remote.models import HPCJobState
from photomatagent.scientific.remote.registry import (
    JobLifecycleState,
    JobRegistry,
)

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "molecule"
    / "tfpma_smoke_corrected_static_clean"
)

ZVAL_ENMAX = {
    "C": (4.0, 400.0),
    "O": (6.0, 400.0),
    "H": (1.0, 250.0),
    "Li": (1.0, 140.0),
    "F": (7.0, 400.0),
}


def write_dataset(library: Path, element: str) -> None:
    dataset = library / element
    dataset.mkdir(parents=True, exist_ok=True)
    zval, enmax = ZVAL_ENMAX[element]
    (dataset / "POTCAR").write_text(
        f"   TITEL  = PAW_PBE {element} 08Apr2002\n"
        f"   POMASS =     1.000; ZVAL   =    {zval:.3f}    mass and valenz\n"
        f"   ENMAX  =  {enmax:.3f}; ENMIN  =  300.000 eV\n",
        encoding="utf-8",
    )


def make_psp(tmp_path: Path, elements=("C", "O", "H", "Li", "F")) -> Path:
    library = tmp_path / "psp"
    for element in elements:
        write_dataset(library, element)
    return library


def cluster_coords(n: int, box: float, spacing: float = 1.6) -> np.ndarray:
    """A compact 3D cluster centred in the box (no PBC collisions)."""
    pattern = [(i % 4, (i // 4) % 2, (i // 8) % 3) for i in range(n)]
    points = np.asarray(pattern, dtype=float) * spacing
    points -= points.mean(axis=0)
    points += box / 2
    return points


def dme_li_symbols() -> list[str]:
    return ["C"] * 4 + ["O"] * 2 + ["H"] * 10 + ["Li"]


def dme_li_molecule(
    tmp_path: Path,
    *,
    charge: int = 1,
    box: float = 20.0,
    conformer_id: str | None = None,
    polymer_kind: PolymerKind = PolymerKind.NONE,
) -> MoleculeSpec:
    symbols = dme_li_symbols()
    coords = cluster_coords(len(symbols), box)
    path = tmp_path / "dme_li.xyz"
    lines = [str(len(symbols)), "DME-Li+ q=+1"]
    lines += [
        f"{symbol} {x:.5f} {y:.5f} {z:.5f}"
        for symbol, (x, y, z) in zip(symbols, coords, strict=True)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return MoleculeSpec(
        name="DME_Li",
        structure_path=path,
        total_charge=charge,
        spin_multiplicity=1,
        box_ang=box,
        calculation_purpose="binding",
        conformer_id=conformer_id,
        polymer_kind=polymer_kind,
    )


# --------------------------------------------------------------------------
# synthetic VASP output writers
# --------------------------------------------------------------------------


def write_eigenval(
    path: Path,
    *,
    nelect: int,
    nbands: int = 30,
    ispin: int = 1,
    homo_energy: float = -6.4,
    lumo_energy: float = -2.2,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    homo = nelect // 2
    lines = [
        "   17   17    1    1",
        "  0.2000000E+03  0.2000000E-08  0.2000000E-08  0.2000000E-08",
        "  1.000000000000000E-004",
        "  CAR ",
        "  synthetic-eigenval",
        f"     {nelect}     1    {nbands}",
        "",
    ]
    for spin in range(ispin):
        lines.append("  0.0000000E+00  0.0000000E+00  0.0000000E+00  0.1000000E+01")
        for band in range(1, nbands + 1):
            if band <= homo:
                energy = homo_energy - (homo - band) * 0.18
                occ = 1.0
            else:
                energy = lumo_energy + (band - homo - 1) * 0.12
                occ = 0.0
            lines.append(f"{band:6d} {energy:16.6f} {occ:10.6f}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_oszicar(
    path: Path, *, e0: float = -122.27635471, converged: bool = True
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "DAV:   1    -0.121765040995E+03   -0.12177E+03   -0.71368E-05   144   0.905E-02    0.696E-03",
        "DAV:   2    -0.121765040523E+03    0.47271E-06   -0.20602E-05   120   0.197E-02    0.462E-03",
    ]
    if converged:
        lines.append(
            "DAV:   3    -0.121765040541E+03   -0.18559E-07   -0.19949E-06   104   0.999E-03"
        )
    de = "0.000E+00" if converged else "0.123E-03"
    lines.append(f"   1 F= {e0: 14.8E} E0= {e0: 14.8E}  d E ={de}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_minimal_vasprun(
    path: Path,
    *,
    e_fr: float = -122.27635471,
    e_0: float = -122.27635471,
    n_atoms: int = 17,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("modeling")
    calc = ET.SubElement(root, "calculation")
    for iteration, value in enumerate((e_fr + 0.01, e_fr)):
        scstep = ET.SubElement(calc, "scstep")
        energy = ET.SubElement(scstep, "energy")
        ET.SubElement(energy, "v", {"name": "e_fr_energy"}).text = f"{value:.8f}"
        ET.SubElement(energy, "v", {"name": "e_0_energy"}).text = f"{value:.8f}"
    energy = ET.SubElement(calc, "energy")
    ET.SubElement(energy, "v", {"name": "e_fr_energy"}).text = f"{e_fr:.8f}"
    ET.SubElement(energy, "v", {"name": "e_0_energy"}).text = f"{e_0:.8f}"
    ET.SubElement(energy, "v", {"name": "eentropy"}).text = "0.00000000"
    structure = ET.SubElement(calc, "structure", {"name": "finalpos"})
    varray = ET.SubElement(structure, "varray", {"name": "positions"})
    for index in range(n_atoms):
        ET.SubElement(varray, "v").text = (
            f"{0.4 + index * 0.001:.6f} {0.5:.6f} {0.5:.6f}"
        )
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def write_locpot(
    path: Path,
    *,
    box: float = 20.0,
    grid: tuple[int, int, int] = (8, 8, 8),
    epsilon: float = 0.0,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    """LOCPOT with a centered gaussian well; vacuum ~ 0 outside."""
    nx, ny, nz = grid
    scale = 1.0
    coords = [
        np.linspace(0, box, n, endpoint=False) for n in (nx, ny, nz)
    ]
    data = np.zeros((nx, ny, nz))
    center = box / 2
    sigma = box / 12
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                r2 = (
                    (coords[0][ix] - center) ** 2
                    + (coords[1][iy] - center) ** 2
                    + (coords[2][iz] - center) ** 2
                )
                data[ix, iy, iz] = -2.5 * np.exp(-r2 / (2 * sigma**2))
                data[ix, iy, iz] += epsilon * (ix / nx + iy / ny + iz / nz)
    lines = [
        "synthetic LOCPOT",
        f"{scale:20.12E}",
        f"{box} 0.0 0.0",
        f"0.0 {box} 0.0",
        f"0.0 0.0 {box}",
        "C O H Li",
        "4 2 10 1",
        "Direct",
    ]
    for index in range(17):
        lines.append("0.5 0.5 0.5")
    lines.append(f"{nx} {ny} {nz}")
    flat = data.reshape(-1, order="C")
    for start in range(0, flat.size, 5):
        lines.append(
            " ".join(f"{value:20.12E}" for value in flat[start : start + 5])
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_outcar(
    path: Path,
    *,
    n_atoms: int = 17,
    max_force: float = 0.001,
    reached: bool = True,
) -> Path:
    """Synthetic force-converged OUTCAR (relax validation fixture)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        " vasp.5.4.4.18Apr17 (build Mar 03 2024 15:47:24) complex parallel",
        "   NSW      =     200     number of steps for ionic motion",
        "   IBRION   =      2     ionic relax: 1=quasi-Newton, 2=damped",
        "   EDIFFG   = -0.02E+00  force-criterion for ionic relax",
    ]
    if reached:
        lines.append(
            "  reached required accuracy - stopping structural energy minimisation"
        )
        lines.append("")
    lines.append("POSITION                                       TOTAL-FORCE (eV/Angst)")
    lines.append("-" * 90)
    component = max_force / max(1.0, (3 * n_atoms) ** 0.5)
    for index in range(n_atoms):
        lines.append(
            f"{index + 1:6d} {0.0:17.10f} {0.0:17.10f} {0.0:17.10f}"
            f"{component:14.8f} {component:14.8f} {component:14.8f}"
        )
    lines.append("-" * 90)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def stage_results_bundle(
    stage_dir: Path,
    *,
    nelect: int,
    e0: float,
    box: float,
    include_vasprun: bool = True,
    include_locpot: bool = False,
    include_contcar: bool = False,
) -> None:
    write_eigenval(stage_dir / "EIGENVAL", nelect=nelect)
    write_oszicar(stage_dir / "OSZICAR", e0=e0)
    write_outcar(stage_dir / "OUTCAR")
    if include_vasprun:
        write_minimal_vasprun(stage_dir / "vasprun.xml", e_fr=e0, e_0=e0)
    if include_locpot:
        write_locpot(stage_dir / "LOCPOT", box=box)
    if include_contcar:
        shutil.copy2(stage_dir / "POSCAR", stage_dir / "CONTCAR")


def result_seeding_backend(psp: Path, tmp_path: Path) -> FakeSCNetBackend:
    """Fake backend that seeds plausible results into every job directory."""
    backend = FakeSCNetBackend()
    original_upload = backend.upload_files

    async def upload_with_results(local_paths, remote_directory):
        names = await original_upload(local_paths, remote_directory)
        bulk = tmp_path / "bulk"
        bulk.mkdir(exist_ok=True)
        stage_results_bundle(bulk, nelect=38, e0=-122.277, box=20.0)
        if "orbital" in remote_directory or remote_directory.endswith("/esp") or "-esp-" in remote_directory:
            write_locpot(bulk / "LOCPOT", box=20.0)
            backend.add_remote_file(remote_directory, "LOCPOT", (bulk / "LOCPOT").read_bytes())
        for name in ("EIGENVAL", "OSZICAR", "vasprun.xml", "OUTCAR"):
            backend.add_remote_file(remote_directory, name, (bulk / name).read_bytes())
        if "-relax-" in remote_directory:
            contcar = next(
                (Path(str(path)) for path in local_paths if str(path).endswith("POSCAR")),
                None,
            )
            backend.add_remote_file(
                remote_directory, "CONTCAR", contcar.read_bytes() if contcar else b""
            )
        return names

    backend.upload_files = upload_with_results
    return backend


def make_session(
    tmp_path: Path, backend: FakeSCNetBackend
) -> SubmitOnceSession:
    return SubmitOnceSession(
        JobRegistry(tmp_path / "jobs.sqlite3"),
        backend,
        marker_temp_dir=tmp_path / "markers",
    )


def tools(
    tmp_path: Path,
    backend: FakeSCNetBackend,
    *,
    psp: Path | None = None,
    workflow_dir: Path | None = None,
    remote_psp_dir: str = "~/photomatagent/psp",
) -> MolecularVaspTools:
    return MolecularVaspTools(
        session=make_session(tmp_path, backend),
        backend=backend,
        psp_dir=psp,
        workflow_dir=workflow_dir,
        log_dir=tmp_path / "logs",
        remote_psp_dir=remote_psp_dir,
    )


# --------------------------------------------------------------------------
# golden fixture regressions
# --------------------------------------------------------------------------


def test_golden_eigenval_parses_like_run_record():
    data = parse_eigenval(FIXTURE / "EIGENVAL")
    assert data.nelect == 76
    assert data.nbnds == 52
    assert data.ispin == 1
    assert len(data.kpoints) == 1
    bands = determine_orbital_bands(data)
    assert bands.homo_band == 38
    assert bands.lumo_band == 39
    assert bands.homo_raw_ev == pytest.approx(-6.601280, abs=1e-4)
    assert bands.lumo_raw_ev == pytest.approx(-2.454867, abs=1e-4)
    assert bands.ks_gap_ev == pytest.approx(4.146413, abs=1e-4)


def test_golden_oszicar_convergence_and_e0():
    oszi = parse_oszicar(FIXTURE / "OSZICAR")
    assert oszi.scf_steps == 3
    assert oszi.last_scf_de_ev == pytest.approx(1.8559e-08, abs=1e-12)
    assert oszi.final_f_ev == pytest.approx(-122.27635471, abs=1e-4)
    assert oszi.final_e0_ev == pytest.approx(-122.27635471, abs=1e-4)


def test_golden_analyze_full_pipeline(tmp_path):
    result_dir = tmp_path / "golden"
    result_dir.mkdir()
    for name in ("INCAR", "KPOINTS", "POSCAR", "EIGENVAL", "OSZICAR", "vasprun.xml"):
        shutil.copy2(FIXTURE / name, result_dir / name)
    # The real audit values: C7O2F4H8 neutral -> NELECT 76.
    (result_dir / "POTCAR.meta").write_text(
        json.dumps(
            {"nelect": 76.0, "neutral_valence_electrons": 76.0,
             "sequence": ["C", "O", "F", "H"]}
        ),
        encoding="utf-8",
    )
    results = analyze_result_dir(
        result_dir, charge=0, spin_multiplicity=1, box_ang=20.0
    )
    assert results["validated"] is True
    assert results["errors"] == []
    energy = results["energy"]
    # vasprun.xml's e_0_energy is zero in this VASP version: E0 falls back
    # to the OSZICAR F=/E0= row and F vs E0 are both reported.
    assert energy["e_0_ev"] == pytest.approx(-122.27635471, abs=1e-5)
    assert energy["e_fr_ev"] == pytest.approx(-122.27635471, abs=1e-5)
    assert energy["source"] == "vasprun.xml <energy> final ionic step"
    assert "OSZICAR" in energy["note"]  # E0 fell back to the OSZICAR row
    assert results["identity"]["formula"] == "C7O2F4H8"
    assert results["identity"]["nelect_declared"] == 76.0
    assert results["scf"]["converged"] is True
    assert results["ionic"]["static_single_point"] is True
    assert results["ionic"]["converged"] is True
    assert results["orbitals"]["homo_band"] == 38
    assert results["geometry"]["n_atoms"] == 21
    evidence = scientific_evidence(results)
    properties = [item.property for item in evidence]
    assert "total_energy_E0" in properties
    assert "HOMO_energy_raw" in properties
    assert "Kohn_Sham_gap" in properties
    assert len(evidence) >= 4


def test_golden_without_meta_is_not_validated_no_evidence(tmp_path):
    result_dir = tmp_path / "golden_nometa"
    result_dir.mkdir()
    for name in ("INCAR", "POSCAR", "EIGENVAL", "OSZICAR"):
        shutil.copy2(FIXTURE / name, result_dir / name)
    results = analyze_result_dir(
        result_dir, charge=0, spin_multiplicity=1, box_ang=20.0
    )
    assert results["validated"] is False
    assert any("ELECTRON_COUNT_UNDECLARED" in error for error in results["errors"])
    assert scientific_evidence(results) == []


def test_static_single_point_is_not_reported_as_ionic_failure(tmp_path):
    result_dir = tmp_path / "sp"
    result_dir.mkdir()
    write_eigenval(result_dir / "EIGENVAL", nelect=38)
    write_oszicar(result_dir / "OSZICAR")
    write_minimal_vasprun(result_dir / "vasprun.xml", n_atoms=17)
    cluster = cluster_coords(17, 20.0)
    poscar_lines = [
        "dme li",
        "1.0",
        "20 0 0",
        "0 20 0",
        "0 0 20",
        "C O H Li",
        "4 2 10 1",
        "Direct",
    ]
    poscar_lines += [
        f"{x/20:.6f} {y/20:.6f} {z/20:.6f}" for x, y, z in cluster
    ]
    (result_dir / "POSCAR").write_text("\n".join(poscar_lines) + "\n", encoding="utf-8")
    (result_dir / "POTCAR.meta").write_text(
        json.dumps({"nelect": 38.0, "neutral_valence_electrons": 39.0}),
        encoding="utf-8",
    )
    results = analyze_result_dir(
        result_dir, charge=1, spin_multiplicity=1, box_ang=20.0
    )
    assert results["validated"] is True
    assert results["ionic"]["static_single_point"] is True
    assert results["ionic"]["converged"] is True
    assert "not applicable" in results["ionic"]["note"]


def test_e0_vs_free_energy_are_distinguished(tmp_path):
    result_dir = tmp_path / "fe"
    result_dir.mkdir()
    write_oszicar(result_dir / "OSZICAR", e0=-10.05)
    write_minimal_vasprun(result_dir / "vasprun.xml", e_fr=-9.95, e_0=-10.05)
    results = analyze_result_dir(
        result_dir, charge=0, spin_multiplicity=1, box_ang=20.0
    )
    assert results["energy"]["e_fr_ev"] == pytest.approx(-9.95)
    assert results["energy"]["e_0_ev"] == pytest.approx(-10.05)
    assert "OSZICAR" not in results["energy"]["source"]  # used vasprun
    # oops: e0 comes from vasprun e_0 field in this case
    assert results["energy"]["source"] == "vasprun.xml <energy> final ionic step"


# --------------------------------------------------------------------------
# LOCPOT / vacuum alignment
# --------------------------------------------------------------------------


def test_read_locpot_grid_and_esp_metadata(tmp_path):
    locpot = write_locpot(tmp_path / "LOCPOT", box=20.0, grid=(8, 8, 8))
    grid = read_locpot(locpot, box_ang=20.0)
    assert grid.grid == (8, 8, 8)
    assert grid.spacing_ang == pytest.approx((2.5, 2.5, 2.5))
    assert grid.data.shape == (8, 8, 8)
    assert not np.isnan(grid.data).any()


def test_vacuum_alignment_from_locpot(tmp_path):
    from photomatagent.scientific.applications.vasp.molecular.results import (
        vacuum_aligned,
    )

    locpot = write_locpot(tmp_path / "LOCPOT", box=20.0)
    grid = read_locpot(locpot, box_ang=20.0)
    level, std, count = vacuum_level(grid)
    assert level == pytest.approx(0.0, abs=0.2)
    assert std < 0.1
    assert count > 0
    bands = determine_orbital_bands(parse_eigenval(
        write_eigenval(tmp_path / "EIGENVAL", nelect=38)
    ))
    aligned = vacuum_aligned(bands, level)
    assert aligned["aligned_homo_ev"] == pytest.approx(bands.homo_raw_ev - level, abs=1e-9)


def test_unstable_plateau_warns(tmp_path):
    locpot = write_locpot(tmp_path / "LOCPOT", box=20.0, epsilon=0.5)
    grid = read_locpot(locpot, box_ang=20.0)
    _, std, _ = vacuum_level(grid)
    assert std > 0.1


# --------------------------------------------------------------------------
# DAG builder + preflight gates
# --------------------------------------------------------------------------


def test_builder_dag_contracts(tmp_path, ):
    psp = make_psp(tmp_path)
    workflow = build_molecule_workflow(
        dme_li_molecule(tmp_path), psp_dir=psp
    )
    names = [stage.name for stage in workflow.stages]
    assert names[0] is StageName.RELAX
    assert names[1] is StageName.STATIC_PRECONVERGE
    assert names[2] is StageName.CORRECTED_STATIC
    assert StageName.ORBITAL_HOMO in names
    assert StageName.ORBITAL_LUMO in names
    assert names[-1] is StageName.ESP
    by_name = {stage.name: stage for stage in workflow.stages}
    assert by_name[StageName.STATIC_PRECONVERGE].depends_on is StageName.RELAX
    assert "CONTCAR" in by_name[StageName.STATIC_PRECONVERGE].required_upstream_outputs
    corrected = by_name[StageName.CORRECTED_STATIC]
    assert corrected.depends_on is StageName.STATIC_PRECONVERGE
    assert {"WAVECAR", "CHGCAR"} <= set(corrected.required_upstream_outputs)
    assert corrected.incar["EDIFF"] == 1e-6
    assert by_name[StageName.STATIC_PRECONVERGE].incar["EDIFF"] == 1e-5
    assert by_name[StageName.STATIC_PRECONVERGE].incar.get("LDIPOL") is None
    assert corrected.incar["ISTART"] == 1 and corrected.incar["ICHARG"] == 1
    for orbital in (StageName.ORBITAL_HOMO, StageName.ORBITAL_LUMO):
        spec = by_name[orbital]
        assert spec.incar["LPARD"] is True
        assert spec.incar["LVHAR"] is True
        assert spec.incar["IBAND"] == 0  # replaced by the runner from EIGENVAL
    esp = by_name[StageName.ESP]
    assert esp.incar["LVHAR"] is True
    assert {"CONTCAR", "CHGCAR"} <= set(esp.required_upstream_outputs)


def test_builder_smoke_and_production_profiles(tmp_path):
    psp = make_psp(tmp_path)
    molecule = dme_li_molecule(tmp_path)
    smoke = build_molecule_workflow(molecule, psp_dir=psp)
    assert smoke.resource_plan.tasks_per_node == 8  # never 32 by default
    assert smoke.resource_plan.profile is ResourceProfile.SMOKE
    assert smoke.preflight_config.encut_floor_ev == 400.0
    production = build_molecule_workflow(
        molecule, psp_dir=psp,
        resource_profile=ResourceProfile.PRODUCTION,
    )
    assert production.preflight_config.encut_floor_ev == 520.0
    assert not production.resource_plan.resource_calibrated
    report = run_molecular_preflight(production, psp_dir=psp)
    assert any(
        issue.code == "RESOURCE_PLAN_VIOLATION" for issue in report.errors
    )


def test_preconverge_dipole_exemption_warns_not_fails(tmp_path):
    psp = make_psp(tmp_path)
    # Neutral DME (C4H10O2, 38 valence electrons, even) keeps ISPIN=1 and a
    # dipole-only policy, so the preconvergence dipole exemption is exercised.
    path = tmp_path / "dme_neutral.xyz"
    symbols = ["C"] * 4 + ["O"] * 2 + ["H"] * 10
    coords = cluster_coords(len(symbols), 20.0)
    path.write_text(
        "\n".join(
            [str(len(symbols)), "DME neutral"]
            + [
                f"{s} {x:.5f} {y:.5f} {z:.5f}"
                for s, (x, y, z) in zip(symbols, coords, strict=True)
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    molecule = MoleculeSpec(
        name="DME",
        structure_path=path,
        total_charge=0,
        spin_multiplicity=1,
        box_ang=20.0,
    )
    workflow = build_molecule_workflow(molecule, psp_dir=psp)
    assert workflow.correction_policy.dipole is True
    report = run_molecular_preflight(workflow, psp_dir=psp)
    assert report.passed is True
    assert any(
        issue.code == "PRECONVERGE_WITHOUT_DIPOLE" for issue in report.warnings
    )


def test_corrected_static_missing_restart_files(tmp_path):
    psp = make_psp(tmp_path)
    workflow = build_molecule_workflow(dme_li_molecule(tmp_path), psp_dir=psp)
    corrected = next(
        stage for stage in workflow.stages
        if stage.name is StageName.CORRECTED_STATIC
    )
    corrected.required_upstream_outputs = ["CONTCAR"]
    report = run_molecular_preflight(workflow, psp_dir=psp)
    assert any(
        issue.code == "CORRECTED_STATIC_MISSING_RESTART_FILES"
        for issue in report.errors
    )


def test_hse06_gate_requires_screen_decision(tmp_path):
    psp = make_psp(tmp_path)
    molecule = dme_li_molecule(tmp_path, conformer_id="conf-3")
    blocked = build_molecule_workflow(
        molecule, psp_dir=psp, include_hse06=True
    )
    report = run_molecular_preflight(blocked, psp_dir=psp)
    assert any(
        issue.code == "HSE06_SCREENING_REQUIRED" for issue in report.errors
    )
    allowed = build_molecule_workflow(
        molecule,
        psp_dir=psp,
        include_hse06=True,
        screen_decision=ScreenDecision(
            conformer_id="conf-3", pbe_e0_ev=-122.0
        ),
    )
    report2 = run_molecular_preflight(allowed, psp_dir=psp)
    assert not any(
        issue.code == "HSE06_SCREENING_REQUIRED" for issue in report2.errors
    )


def test_vm_tvm_missing_structure_stays_blocked(tmp_path):
    psp = make_psp(tmp_path)
    molecule = dme_li_molecule(
        tmp_path, polymer_kind=PolymerKind.VM
    )
    workflow = build_molecule_workflow(molecule, psp_dir=psp)
    report = run_molecular_preflight(workflow, psp_dir=psp)
    assert report.passed is False
    assert any(
        issue.code == "BLOCKED_MISSING_STRUCTURE" for issue in report.errors
    )
    backend = FakeSCNetBackend()
    tool = tools(tmp_path, backend, psp=psp, workflow_dir=tmp_path / "vm")
    prepared = asyncio.run(tool.prepare(workflow, output_dir=tmp_path / "vm"))
    assert prepared["ok"] is False
    submitted = asyncio.run(tool.submit(StageName.RELAX))
    assert submitted["ok"] is False
    assert backend.uploaded == []  # no guessed model, no upload, no sbatch


# --------------------------------------------------------------------------
# workflow runner (resume + failure propagation)
# --------------------------------------------------------------------------


async def test_run_workflow_end_to_end_and_resume(tmp_path):
    psp = make_psp(tmp_path)
    backend = result_seeding_backend(psp, tmp_path)
    session = make_session(tmp_path, backend)
    workflow = build_molecule_workflow(dme_li_molecule(tmp_path), psp_dir=psp)
    root = tmp_path / "wf"
    first = await run_molecule_workflow(
        workflow, root, session=session, backend=backend, psp_dir=psp,
        remote_psp_dir="~/photomatagent/psp",
        wait_timeout_seconds=30,
    )
    assert first.get("error") is None
    assert first["blocked"] == []
    assert sorted(first["completed"]) == sorted(
        stage.name.value for stage in workflow.stages
    )
    assert first["evidence_count"] > 0
    jobs_after_first = len(backend.submitted_scripts)
    state = load_task_state(root)
    assert state is not None
    assert all(
        item.state == JobLifecycleState.VALIDATED.value for item in state.stages
    )
    assert all(item.validated for item in state.stages)
    # resume: completed stages are not resubmitted
    second = await run_molecule_workflow(
        workflow, root, session=session, backend=backend, psp_dir=psp,
        remote_psp_dir="~/photomatagent/psp",
        wait_timeout_seconds=30,
    )
    assert second["resumed"] == [stage.name.value for stage in workflow.stages]
    assert len(backend.submitted_scripts) == jobs_after_first


async def test_run_workflow_failure_blocks_dependents(tmp_path):
    psp = make_psp(tmp_path)
    backend = result_seeding_backend(psp, tmp_path)
    backend.scripted_states = [
        HPCJobState.PENDING,
        HPCJobState.RUNNING,
        HPCJobState.FAILED,
    ]
    session = make_session(tmp_path, backend)
    workflow = build_molecule_workflow(dme_li_molecule(tmp_path), psp_dir=psp)
    report = await run_molecule_workflow(
        workflow, tmp_path / "wf_fail", session=session, backend=backend,
        psp_dir=psp, remote_psp_dir="~/photomatagent/psp",
        wait_timeout_seconds=20,
    )
    relax_entry = next(
        item for item in report["stages"] if item["stage"] == "relax"
    )
    assert relax_entry["state"] == "FAILED"
    assert report["blocked"]
    assert report["evidence_count"] == 0
    assert len(backend.submitted_scripts) == 1  # dependents never submitted


async def test_oom_timeout_produce_no_evidence(tmp_path):
    psp = make_psp(tmp_path)
    molecule = dme_li_molecule(tmp_path)
    from photomatagent.scientific.applications.vasp.molecular.workflow import (
        build_molecule_workflow,
    )

    workflow = build_molecule_workflow(molecule, psp_dir=psp)
    root = tmp_path / "wf_oom"
    # prepare + task_state as a human would
    backend = result_seeding_backend(psp, tmp_path)
    session = make_session(tmp_path, backend)
    from photomatagent.scientific.applications.vasp.molecular.tools import (
        MolecularVaspTools,
    )

    tool = MolecularVaspTools(
        session=session, backend=backend, psp_dir=psp,
        workflow_dir=root, log_dir=tmp_path / "logs",
    )
    await tool.prepare(workflow, output_dir=root)
    state = load_task_state(root)
    assert state is not None
    from photomatagent.scientific.applications.vasp.molecular.workflow import (
        save_task_state,
    )

    state.stages = [
        StageTask(
            stage="relax",
            state=JobLifecycleState.OUT_OF_MEMORY.value,
            request_id="oom-request",
            remote_directory="~/photomatagent/oom-dir",
        )
    ]
    save_task_state(root, state)
    collected = await tool.collect(StageName.RELAX)
    assert collected["ok"] is False
    assert collected["evidence_count"] == 0
    assert not (root / "results" / "relax" / "evidence.json").exists()


def test_determine_ibands_from_static(tmp_path):
    static = tmp_path / "static"
    static.mkdir()
    write_eigenval(static / "EIGENVAL", nelect=38)
    assert determine_ibands(static) == (19, 20)
    assert determine_ibands(tmp_path) == (None, None)


def test_eigenval_ispin2_parsing(tmp_path):
    path = tmp_path / "EIGENVAL"
    nelect = 39  # one unpaired electron: alpha has 20, beta has 19
    nbands = 30
    lines = [
        "   17   17    2    1",
        "  0.2000000E+03  0.2000000E-08  0.2000000E-08  0.2000000E-08",
        "  1.000000000000000E-004",
        "  CAR ",
        "  synthetic-eigenval-spin2",
        f"     {nelect}     1    {nbands}",
        "",
    ]
    for spin, occupied in ((1, 20), (2, 19)):
        lines.append("  0.0000000E+00  0.0000000E+00  0.0000000E+00  0.1000000E+01")
        for band in range(1, nbands + 1):
            energy = -6.0 - (occupied - band) * 0.1 if band <= occupied else -1.0 + (band - occupied) * 0.1
            occ = 1.0 if band <= occupied else 0.0
            lines.append(f"{band:6d} {energy:16.6f} {occ:10.6f}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    data = parse_eigenval(path)
    assert data.ispin == 2
    assert len(data.kpoints) == 2
    bands = determine_orbital_bands(data)
    assert bands.homo_band == 20
    assert bands.lumo_band == 21


def test_magnetization_parsed_from_outcar(tmp_path):
    from photomatagent.scientific.applications.vasp.molecular.results import (
        magnetization_from_outcar,
    )

    outcar = tmp_path / "OUTCAR"
    outcar.write_text(
        "number of electron     38.0000000 magnetization      1.0100000\n",
        encoding="utf-8",
    )
    assert magnetization_from_outcar(outcar) == pytest.approx(1.01)
    spin1 = tmp_path / "OUTCAR_s1"
    spin1.write_text("ISPIN = 1\nno magnetization line\n", encoding="utf-8")
    assert magnetization_from_outcar(spin1) == 0.0


# --------------------------------------------------------------------------
# tool surface
# --------------------------------------------------------------------------


async def test_tool_prepare_bounded_payload(tmp_path):
    psp = make_psp(tmp_path)
    backend = FakeSCNetBackend()
    tool = tools(tmp_path, backend, psp=psp)
    workflow = build_molecule_workflow(dme_li_molecule(tmp_path), psp_dir=psp)
    out = await tool.prepare(workflow, output_dir=tmp_path / "prep")
    assert out["ok"] is True
    assert out["summary"]["preflight_passed"] is True
    assert (tmp_path / "prep" / "preflight.json").is_file()
    assert (tmp_path / "prep" / "task_state.json").is_file()
    assert (tmp_path / "prep" / "workflow.json").is_file()
    assert out["chars"] <= 4000


async def test_tool_submit_idempotent_and_status(tmp_path):
    psp = make_psp(tmp_path)
    backend = result_seeding_backend(psp, tmp_path)
    tool = tools(tmp_path, backend, psp=psp, workflow_dir=tmp_path / "toolwf")
    workflow = build_molecule_workflow(dme_li_molecule(tmp_path), psp_dir=psp)
    await tool.prepare(workflow, output_dir=tmp_path / "toolwf")
    first = await tool.submit(StageName.RELAX)
    assert first["ok"] is True
    assert first["summary"]["job_id"] == "1001"
    second = await tool.submit(StageName.RELAX)
    assert second["ok"] is False
    assert len(backend.submitted_scripts) == 1  # submit-once held
    status = await tool.status(StageName.RELAX)
    assert status["summary"]["lifecycle_state"] == JobLifecycleState.COMPLETED.value
    assert status["chars"] <= 4000


async def test_tool_collect_analysis_and_evidence(tmp_path):
    psp = make_psp(tmp_path)
    backend = result_seeding_backend(psp, tmp_path)
    tool = tools(tmp_path, backend, psp=psp, workflow_dir=tmp_path / "tw2")
    workflow = build_molecule_workflow(dme_li_molecule(tmp_path), psp_dir=psp)
    root = tmp_path / "tw2"
    await tool.prepare(workflow, output_dir=root)
    submitted = await tool.submit(StageName.CORRECTED_STATIC, wait=True)
    assert submitted["ok"] is True
    collected = await tool.collect(StageName.CORRECTED_STATIC)
    assert collected["ok"] is True
    assert collected["evidence_count"] > 0
    assert (root / "results" / "corrected_static" / "results.json").is_file()
    orbitals = await tool.analyze_orbitals(root / "results" / "corrected_static")
    assert orbitals["summary"]["homo_band"] == 19
    assert orbitals["summary"]["lumo_band"] == 20
    assert orbitals["summary"]["ks_gap_ev"] is not None


async def test_tool_analyze_esp(tmp_path):
    from photomatagent.scientific.applications.vasp.molecular.results import (
        esp_metadata,
    )

    result_dir = tmp_path / "espres"
    result_dir.mkdir()
    write_locpot(result_dir / "LOCPOT", box=20.0)
    (result_dir / "INCAR").write_text(
        "LVHAR = .TRUE.\nENCUT = 400\n", encoding="utf-8"
    )
    backend = FakeSCNetBackend()
    tool = tools(tmp_path, backend)
    out = await tool.analyze_esp(result_dir)
    assert out["ok"] is True
    assert out["summary"]["has_locpot"] is True
    assert out["summary"]["grid"] == [8, 8, 8]
    assert out["summary"]["lvhar_declared"] is True


async def test_tool_binding_energy_consistency(tmp_path):
    def quick_results(directory: Path, e0: float, encut: float) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "results.json").write_text(
            json.dumps(
                {
                    "validated": True,
                    "errors": [],
                    "warnings": [],
                    "identity": {"formula": "X"},
                    "method": {
                        "box_ang": 20.0,
                        "functional": "PE",
                        "encut_ev": encut,
                    },
                    "energy": {"e_0_ev": e0},
                    "corrections": {"dipole_quadrupole_ev": -0.001},
                }
            ),
            encoding="utf-8",
        )
        return directory

    complex_dir = quick_results(tmp_path / "complex", e0=-122.0, encut=400.0)
    ref_dir = quick_results(tmp_path / "ref", e0=-100.0, encut=400.0)
    backend = FakeSCNetBackend()
    tool = tools(tmp_path, backend)
    ok = await tool.binding_energy(
        BindingEnergyInput(
            complex_name="DME-Li+",
            complex_dir=str(complex_dir),
            charge=1,
            references=[
                BindingReference(name="DME", results_dir=str(ref_dir), charge=1)
            ],
            alternative_references=[
                BindingReference(
                    name="DME(neutral)", results_dir=str(ref_dir), charge=0
                ),
                BindingReference(
                    name="Li+", results_dir=str(ref_dir), charge=1
                ),
            ],
        )
    )
    assert ok["ok"] is True
    assert ok["summary"]["delta_e_ev"] == pytest.approx(-22.0)
    assert ok["summary"]["electronic_only"] is True

    # ENCUT mismatch must refuse
    bad_ref = quick_results(tmp_path / "ref_bad", e0=-100.0, encut=350.0)
    rejected = await tool.binding_energy(
        BindingEnergyInput(
            complex_name="DME-Li+",
            complex_dir=str(complex_dir),
            charge=1,
            references=[
                BindingReference(name="DME", results_dir=str(bad_ref), charge=1)
            ],
        )
    )
    assert rejected["ok"] is False
    assert any("ENCUT" in error for error in rejected["errors"])

    # charge non-conservation must refuse
    bad_charge = await tool.binding_energy(
        BindingEnergyInput(
            complex_name="DME-Li+",
            complex_dir=str(complex_dir),
            charge=1,
            references=[
                BindingReference(name="DME", results_dir=str(ref_dir), charge=0)
            ],
        )
    )
    assert bad_charge["ok"] is False
    assert any("conserved" in error for error in bad_charge["errors"])


def test_tool_outputs_are_bounded_and_logged(tmp_path, ):
    from photomatagent.scientific.applications.vasp.molecular.tools import (
        bounded_payload,
    )

    payload = bounded_payload(
        ok=True,
        summary={"a": 1, "b": [2, 3, 4, 5, 6]},
        errors=["x" * 120] * 30,
        warnings=[],
        artifacts=[],
    )
    assert payload["chars"] <= 4000
    assert len(payload["errors"]) <= 10


async def test_tool_logs_written_for_every_call(tmp_path):
    psp = make_psp(tmp_path)
    backend = FakeSCNetBackend()
    tool = tools(tmp_path, backend, psp=psp, workflow_dir=tmp_path / "logwf")
    workflow = build_molecule_workflow(dme_li_molecule(tmp_path), psp_dir=psp)
    await tool.prepare(workflow, output_dir=tmp_path / "logwf")
    await tool.preflight(workflow)
    await tool.submit(StageName.RELAX)
    await tool.status(StageName.RELAX)
    await tool.analyze_esp(tmp_path)  # empty dir: gracefully reports no LOCPOT
    log_names = {path.name for path in Path(tmp_path / "logs").glob("*.log")}
    assert "prepare.log" in log_names
    assert "preflight.log" in log_names
    assert "submit_relax.log" in log_names
    assert "status_relax.log" in log_names
