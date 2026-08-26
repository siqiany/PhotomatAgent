"""Phase 4.1 offline regressions: real-workspace integration blockers.

Covers, with synthetic POTCARs only (TITEL/ZVAL/ENMAX header lines, never
real library content):

* local pseudopotential layout resolution (direct / potpaw_PBE /
  potpaw_PBE.64) with the real ZVAL values: TFPMA -> 76 valence electrons,
  DME-Li+ -> NELECT 38;
* the local POTCAR submission strategy (assemble from the resolved local
  library, upload to the unique remote job directory, clean up afterwards,
  never leak POTCAR bulk into logs/registry/payloads);
* extensionless POSCAR/CONTCAR auto-detection through the tool surface;
* the resume contract: scheduler COMPLETED is not scientific completion,
  COLLECTED stages are re-validated without resubmission, only VALIDATED
  satisfies downstream dependencies, and validation failures block
  dependents;
* PARCHG orbital artifacts (VASP 5.4.4 naming variants) survive collect.

No real SSH, Slurm or VASP is ever touched.
"""

from __future__ import annotations

import asyncio
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from photomatagent.scientific.applications.vasp.molecular.models import (
    MoleculeSpec,
    StageName,
)
from photomatagent.scientific.applications.vasp.molecular.psp_metadata import (
    resolve_potcar_metadata,
)
from photomatagent.scientific.applications.vasp.molecular.runtime import (
    MolecularVaspRuntime,
)
from photomatagent.scientific.applications.vasp.molecular.slurm import (
    materialize_stage_potcar,
    potcar_mode_of_stage,
)
from photomatagent.scientific.applications.vasp.molecular.tool_pack import (
    MolecularVaspPrepareTool,
)
from photomatagent.scientific.applications.vasp.molecular.tools import (
    MolecularVaspTools,
)
from photomatagent.scientific.applications.vasp.molecular.workflow import (
    StageTask,
    _download_parchg_artifacts,
    build_molecule_workflow,
    load_task_state,
    needs_revalidation,
    run_molecule_workflow,
    save_task_state,
    stage_done,
)
from photomatagent.scientific.remote.fake import FakeSCNetBackend
from photomatagent.scientific.remote.lifecycle import SubmitOnceSession
from photomatagent.scientific.remote.models import ResourcePolicy
from photomatagent.scientific.remote.registry import (
    JobLifecycleState,
    JobRegistry,
)


# Real PAW-PBE.64 ZVAL values (metadata only; synthetic header lines).
REAL_ZVAL_ENMAX = {
    "C": (4.0, 400.0),
    "O": (6.0, 400.0),
    "H": (1.0, 250.0),
    "Li": (1.0, 140.0),
    "F": (7.0, 400.0),
}


def write_dataset(library: Path, element: str) -> None:
    dataset = library / element
    dataset.mkdir(parents=True, exist_ok=True)
    zval, enmax = REAL_ZVAL_ENMAX[element]
    (dataset / "POTCAR").write_text(
        f"   TITEL  = PAW_PBE {element} 08Apr2002\n"
        f"   POMASS =     1.000; ZVAL   =    {zval:.3f}    mass and valenz\n"
        f"   ENMAX  =  {enmax:.3f}; ENMIN  =  300.000 eV\n"
        f"# FIXTURE-BULK-MARKER-9f3a {element}\n",
        encoding="utf-8",
    )


def make_psp(tmp_path: Path, layout: str = "direct") -> Path:
    """Build a local library root in one of the three supported layouts."""
    root = tmp_path / "psp"
    library = root if layout == "direct" else root / layout
    for element in REAL_ZVAL_ENMAX:
        write_dataset(library, element)
    return root


def _molecule(
    tmp_path: Path,
    symbols: list[str],
    *,
    name: str,
    charge: int,
    box: float = 20.0,
    suffix: str = "xyz",
) -> MoleculeSpec:
    pattern = [(i % 4, (i // 4) % 2, (i // 8) % 3) for i in range(len(symbols))]
    points = np.asarray(pattern, dtype=float) * 1.6
    points -= points.mean(axis=0)
    points += box / 2
    path = tmp_path / f"{name}.{suffix}"
    lines = [str(len(symbols)), f"{name} q={charge:+d}"]
    lines += [
        f"{symbol} {x:.5f} {y:.5f} {z:.5f}"
        for symbol, (x, y, z) in zip(symbols, points, strict=True)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return MoleculeSpec(
        name=name,
        structure_path=path,
        total_charge=charge,
        spin_multiplicity=1,
        box_ang=box,
        calculation_purpose="regression",
    )


def dme_li(tmp_path: Path, *, charge: int = 1) -> MoleculeSpec:
    return _molecule(
        tmp_path,
        ["C"] * 4 + ["O"] * 2 + ["H"] * 10 + ["Li"],
        name="DME_Li",
        charge=charge,
    )


def tfpma(tmp_path: Path) -> MoleculeSpec:
    # Verified smoke formula: C7H8F4O2, neutral, 76 PAW valence electrons.
    return _molecule(
        tmp_path,
        ["C"] * 7 + ["H"] * 8 + ["F"] * 4 + ["O"] * 2,
        name="TFPMA",
        charge=0,
    )


def make_runtime(
    tmp_path: Path,
    *,
    psp: Path | None,
    remote_psp_dir: str = "",
    strict: bool = True,
    workflow_dir: Path | None = None,
) -> MolecularVaspRuntime:
    return MolecularVaspRuntime(
        backend=FakeSCNetBackend(
            policy=ResourcePolicy(allow_hpc_submit=strict),
            strict=strict,
        ),
        configured=True,
        psp_dir=psp,
        workflow_dir=workflow_dir or (tmp_path / "mol"),
        log_dir=tmp_path / "logs",
        registry_path=tmp_path / "state" / "jobs.sqlite3",
        module_name="vasp-5.4.4",
        env_script="",
        remote_psp_dir=remote_psp_dir,
    )


def make_session(
    tmp_path: Path, backend: FakeSCNetBackend
) -> SubmitOnceSession:
    return SubmitOnceSession(
        JobRegistry(tmp_path / "jobs.sqlite3"),
        backend,
        marker_temp_dir=tmp_path / "markers",
    )


def write_eigenval(path: Path, *, nelect: int = 38) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    homo = nelect // 2
    lines = [
        "   17   17    1    1",
        "  0.2000000E+03  0.2000000E-08  0.2000000E-08  0.2000000E-08",
        "  1.000000000000000E-004",
        "  CAR ",
        "  synthetic-eigenval",
        f"     {nelect}     1    40",
        "",
        "  0.0000000E+00  0.0000000E+00  0.0000000E+00  0.1000000E+01",
    ]
    for band in range(1, 41):
        if band <= homo:
            energy = -6.4 - (homo - band) * 0.18
            occ = 1.0
        else:
            energy = -2.2 + (band - homo - 1) * 0.12
            occ = 0.0
        lines.append(f"{band:6d} {energy:16.6f} {occ:10.6f}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_oszicar(
    path: Path, *, e0: float = -122.27635471, converged: bool = True
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "DAV:   1    -0.121765040995E+03   -0.12177E+03   -0.71368E-05   144   0.905E-02    0.696E-03",
        "DAV:   2    -0.121765040523E+03    0.47271E-06   -0.20602E-05   120   0.197E-02    0.462E-03",
    ]
    if converged:
        lines.append(
            "DAV:   3    -0.121765040541E+03   -0.18559E-07   -0.19949E-06   104   0.999E-03"
        )
    else:
        # A final SCF step whose dE is far above EDIFF=1E-6: the run must be
        # flagged SCF_NOT_CONVERGED and produce no evidence.
        lines.append(
            "DAV:   3    -0.121765040541E+03    0.123E-03   -0.19949E-06   104   0.999E-03"
        )
    de = "0.000E+00" if converged else "0.123E-03"
    lines.append(f"   1 F= {e0: 14.8E} E0= {e0: 14.8E}  d E ={de}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_minimal_vasprun(path: Path, *, e0: float = -122.27635471) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("modeling")
    calc = ET.SubElement(root, "calculation")
    for value in (e0 + 0.01, e0):
        scstep = ET.SubElement(calc, "scstep")
        energy = ET.SubElement(scstep, "energy")
        ET.SubElement(energy, "v", {"name": "e_fr_energy"}).text = f"{value:.8f}"
        ET.SubElement(energy, "v", {"name": "e_0_energy"}).text = f"{value:.8f}"
    energy = ET.SubElement(calc, "energy")
    ET.SubElement(energy, "v", {"name": "e_fr_energy"}).text = f"{e0:.8f}"
    ET.SubElement(energy, "v", {"name": "e_0_energy"}).text = f"{e0:.8f}"
    ET.SubElement(energy, "v", {"name": "eentropy"}).text = "0.00000000"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def write_locpot(path: Path, *, box: float = 20.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nx = ny = nz = 8
    lines = [
        "synthetic LOCPOT",
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
    for _ in range((nx * ny * nz + 4) // 5):
        lines.append(" ".join(f"{0.0:20.12E}" for _ in range(5)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def seed_results(
    backend: FakeSCNetBackend,
    tmp_path: Path,
    *,
    converged: bool = True,
    parchg: bool = False,
) -> None:
    """Seed plausible results (optionally PARCHG) into every job directory."""
    original_upload = backend.upload_files

    async def upload_with_results(local_paths, remote_directory):
        names = await original_upload(local_paths, remote_directory)
        bulk = tmp_path / "bulk"
        bulk.mkdir(exist_ok=True)
        write_eigenval(bulk / "EIGENVAL", nelect=38)
        write_oszicar(bulk / "OSZICAR", converged=converged)
        write_minimal_vasprun(bulk / "vasprun.xml")
        for name in ("EIGENVAL", "OSZICAR", "vasprun.xml"):
            backend.add_remote_file(
                remote_directory, name, (bulk / name).read_bytes()
            )
        # Restart artifacts are staged between remote directories on SCNet
        # and never downloaded locally; placeholders prove staging works.
        backend.add_remote_file(
            remote_directory, "WAVECAR", b"fake-wave-epoch-1"
        )
        backend.add_remote_file(
            remote_directory, "CHGCAR", b"fake-chg-epoch-1"
        )
        if "-relax-" in remote_directory:
            source = next(
                (
                    Path(str(path))
                    for path in local_paths
                    if str(path).endswith("POSCAR")
                ),
                None,
            )
            backend.add_remote_file(
                remote_directory,
                "CONTCAR",
                source.read_bytes() if source is not None else b"",
            )
        if "orbital" in remote_directory or "-esp-" in remote_directory:
            write_locpot(bulk / "LOCPOT")
            backend.add_remote_file(
                remote_directory, "LOCPOT", (bulk / "LOCPOT").read_bytes()
            )
        if parchg and "-orbital_" in remote_directory:
            backend.add_remote_file(
                remote_directory, "PARCHG.19", b"fake parchg band 19"
            )
            backend.add_remote_file(
                remote_directory, "PARCHG", b"fake parchg plain"
            )
        return names

    backend.upload_files = upload_with_results


def make_tools(
    tmp_path: Path,
    backend: FakeSCNetBackend,
    *,
    psp: Path | None,
    workflow_dir: Path,
    remote_psp_dir: str = "",
) -> MolecularVaspTools:
    return MolecularVaspTools(
        session=make_session(tmp_path, backend),
        backend=backend,
        psp_dir=psp,
        workflow_dir=workflow_dir,
        log_dir=tmp_path / "logs",
        remote_psp_dir=remote_psp_dir,
    )


# ---------------------------------------------------------------------------
# 1. local pseudopotential layout resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", ["direct", "potpaw_PBE", "potpaw_PBE.64"])
def test_psp_layouts_resolve_dme_li_nelect_and_tfpma_76(tmp_path, layout):
    from photomatagent.scientific.applications.vasp.molecular.preflight import (
        run_molecular_preflight,
    )

    psp = make_psp(tmp_path, layout)
    dme = dme_li(tmp_path)
    resolution = resolve_potcar_metadata(dme, ["C", "O", "H", "Li"], psp_dir=psp)
    assert resolution.layout == layout
    assert resolution.library.name == (
        "psp" if layout == "direct" else layout
    )
    neutral = sum(
        block.zval * count
        for block, count in zip(resolution.blocks, [4, 2, 10, 1], strict=True)
    )
    assert neutral == 39
    workflow = build_molecule_workflow(dme, psp_dir=psp)
    report = run_molecular_preflight(workflow, psp_dir=psp)
    assert report.passed is True
    assert report.summary is not None
    assert report.summary.nelect == 38
    assert report.summary.neutral_valence_electrons == 39

    t = tfpma(tmp_path)
    workflow_t = build_molecule_workflow(t, psp_dir=psp)
    report_t = run_molecular_preflight(workflow_t, psp_dir=psp)
    assert report_t.passed is True
    assert report_t.summary is not None
    assert report_t.summary.neutral_valence_electrons == 76
    assert report_t.summary.nelect == 76


def test_unresolved_layout_is_reported_not_guessed(tmp_path):
    from photomatagent.scientific.applications.vasp.molecular.preflight import (
        run_molecular_preflight,
    )

    root = tmp_path / "not-a-psp-root"
    root.mkdir()
    (root / "README.txt").write_text("no potcar here", encoding="utf-8")
    # Building the workflow without a PSP dir keeps the typed DAG intact;
    # the preflight itself reports the unresolved layout deterministically.
    workflow = build_molecule_workflow(dme_li(tmp_path), psp_dir=None)
    report = run_molecular_preflight(workflow, psp_dir=root)
    assert report.passed is False
    assert any(
        issue.code in {"PSP_UNRESOLVED", "PSP_METADATA_UNREADABLE"}
        for issue in report.errors
    )


# ---------------------------------------------------------------------------
# 2. local POTCAR submission strategy
# ---------------------------------------------------------------------------


def test_potcar_mode_selection_and_capabilities(tmp_path):
    psp = make_psp(tmp_path)
    runtime_local = make_runtime(tmp_path, psp=psp, remote_psp_dir="")
    caps = runtime_local.capabilities_payload()
    assert caps["selected_potcar_mode"] == "local"
    assert caps["psp_layout"] == "direct"

    runtime_remote = make_runtime(
        tmp_path, psp=psp, remote_psp_dir="~/photomatagent/psp"
    )
    assert runtime_remote.capabilities_payload()["selected_potcar_mode"] == "remote"

    runtime_none = make_runtime(tmp_path, psp=None, remote_psp_dir="")
    assert runtime_none.capabilities_payload()["selected_potcar_mode"] == "none"

    # potcar_mode_of_stage reports local when the library resolves even
    # though no POTCAR file exists yet (materialization happens at submit).
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "POTCAR.meta").write_text(
        json.dumps({"sequence": ["C", "O", "H", "Li"]}), encoding="utf-8"
    )
    assert (
        potcar_mode_of_stage(stage, remote_psp_dir="", psp_dir=psp)
        == "local"
    )
    assert (
        potcar_mode_of_stage(stage, remote_psp_dir="", psp_dir=None)
        == "none"
    )
    assert (
        potcar_mode_of_stage(
            stage, remote_psp_dir="~/photomatagent/psp", psp_dir=psp
        )
        == "remote"
    )


async def test_local_potcar_submit_assembles_uploads_cleans_no_leak(tmp_path):
    psp = make_psp(tmp_path)
    molecule = dme_li(tmp_path)
    workflow = build_molecule_workflow(molecule, psp_dir=psp)
    root = tmp_path / "wf_local"
    backend = FakeSCNetBackend(
        policy=ResourcePolicy(allow_hpc_submit=True), strict=True
    )
    tool = make_tools(
        tmp_path, backend, psp=psp, workflow_dir=root, remote_psp_dir=""
    )
    prepared = await tool.prepare(workflow, output_dir=root)
    assert prepared["ok"] is True

    stage_dir = root / "inputs" / "01_relax"
    assert not (stage_dir / "POTCAR").is_file()  # never pre-materialized
    submitted = await tool.submit(StageName.RELAX, wait=True)
    assert submitted["ok"] is True
    assert submitted["summary"]["state"] == JobLifecycleState.COMPLETED.value

    # The assembled POTCAR reached the unique remote job directory...
    remote = submitted["summary"]["remote_directory"]
    remote_files = backend.remote_files.get(remote, {})
    assert "POTCAR" in remote_files
    assert b"FIXTURE-BULK-MARKER-9f3a" in remote_files["POTCAR"]
    # ...and was removed locally afterwards (never committed to Git).
    assert not (stage_dir / "POTCAR").is_file()

    # POTCAR bulk never entered logs, the registry or tool payloads.
    marker = "FIXTURE-BULK-MARKER-9f3a"
    logs = "".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (tmp_path / "logs").glob("*.log")
    )
    assert marker not in logs
    payload_text = json.dumps(submitted, ensure_ascii=False)
    assert marker not in payload_text
    record = tool.session.registry.get(submitted["summary"]["request_id"])
    assert record is not None
    assert marker not in json.dumps(record.public_dict(), ensure_ascii=False)
    # Metadata (TITEL/ZVAL/ENMAX) summaries are the only POTCAR-derived data
    # that may be present.
    meta = (stage_dir / "POTCAR.meta").read_text(encoding="utf-8")
    assert "PAW_PBE C" in meta  # metadata summary only
    assert "FIXTURE-BULK-MARKER" not in meta


# ---------------------------------------------------------------------------
# 3. extensionless POSCAR auto-detection
# ---------------------------------------------------------------------------


def _write_poscar_file(path: Path, *, box: float = 20.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    symbols = ["C"] * 4 + ["O"] * 2 + ["H"] * 10 + ["Li"]
    pattern = [(i % 4, (i // 4) % 2, (i // 8) % 3) for i in range(len(symbols))]
    points = np.asarray(pattern, dtype=float) * 1.6
    points -= points.mean(axis=0)
    points += box / 2
    frac = points / box
    lines = [
        "DME-Li isolated",
        "1.0",
        f"{box:.10f} 0.0 0.0",
        f"0.0 {box:.10f} 0.0",
        f"0.0 0.0 {box:.10f}",
        "C O H Li",
        "4 2 10 1",
        "Direct",
    ]
    lines.extend(" ".join(f"{value:.12f}" for value in row) for row in frac)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


async def test_extensionless_poscar_prepare_preflight(tmp_path):
    psp = make_psp(tmp_path)
    poscar = _write_poscar_file(tmp_path / "POSCAR")
    runtime = make_runtime(tmp_path, psp=psp, remote_psp_dir="")
    tool = MolecularVaspPrepareTool(runtime)
    result = await tool.execute(
        {
            "structure_path": str(poscar),
            "name": "DME_Li",
            "total_charge": 1,
            "spin_multiplicity": 1,
            "box_ang": 20.0,
        }
    )
    assert result.data["ok"] is True
    assert result.data["summary"]["nelect"] == 38
    assert (runtime.workflow_dir / "preflight.json").is_file()
    assert len(result.output) <= 4000

    # Explicit structure_kind hint is accepted too.
    contcar = _write_poscar_file(tmp_path / "CONTCAR")
    runtime2 = make_runtime(
        tmp_path,
        psp=psp,
        remote_psp_dir="",
        workflow_dir=tmp_path / "mol_contcar",
    )
    tool2 = MolecularVaspPrepareTool(runtime2)
    result2 = await tool2.execute(
        {
            "structure_path": str(contcar),
            "structure_kind": "poscar",
            "name": "DME_Li",
            "total_charge": 1,
            "spin_multiplicity": 1,
            "box_ang": 20.0,
        }
    )
    assert result2.data["ok"] is True


# ---------------------------------------------------------------------------
# 4. resume contract (COMPLETED is not scientific completion)
# ---------------------------------------------------------------------------


def test_stage_done_only_validated():
    assert stage_done("COMPLETED") is False
    assert stage_done("COLLECTED") is False
    assert stage_done("VALIDATED") is True
    assert needs_revalidation("COMPLETED") is True
    assert needs_revalidation("COLLECTED") is True
    assert needs_revalidation("VALIDATED") is False


async def test_resume_after_submit_then_process_exit(tmp_path):
    psp = make_psp(tmp_path)
    molecule = dme_li(tmp_path)
    workflow = build_molecule_workflow(molecule, psp_dir=psp)
    root = tmp_path / "wf_exit"
    backend = FakeSCNetBackend(
        policy=ResourcePolicy(allow_hpc_submit=True), strict=True
    )
    seed_results(backend, tmp_path)
    session = make_session(tmp_path, backend)

    # Run the relax stage but do NOT collect: the process "exits" right
    # after submission with the stage scheduler-COMPLETED in task_state.json
    # and the SQLite registry (and no scientific completion at all).
    first = await run_molecule_workflow(
        workflow, root, session=session, backend=backend, psp_dir=psp,
        remote_psp_dir="",
        wait=True, collect=False, wait_timeout_seconds=30,
        only=["relax"],
    )
    assert first.get("error") is None
    assert first["evidence_count"] == 0
    state = load_task_state(root)
    assert state is not None
    assert state.stage_map()["relax"].state == JobLifecycleState.COMPLETED.value
    assert state.stage_map()["relax"].validated is False
    jobs = len(backend.submitted_scripts)
    assert jobs == 1

    # "Restart the process": the session object is gone, but the cluster
    # (backend) and the SQLite registry survive. A brand-new session over
    # the same backend+registry must collect/validate the completed relax
    # without resubmitting it, then finish the rest of the DAG.
    session2 = SubmitOnceSession(
        JobRegistry(tmp_path / "jobs.sqlite3"),
        backend,
        marker_temp_dir=tmp_path / "markers",
    )
    second = await run_molecule_workflow(
        workflow, root, session=session2, backend=backend, psp_dir=psp,
        remote_psp_dir="",
        wait=True, collect=True, wait_timeout_seconds=30,
    )
    assert second.get("error") is None
    assert second["resumed"] == ["relax"]
    assert second["evidence_count"] > 0
    state2 = load_task_state(root)
    assert state2 is not None
    assert all(
        item.state == JobLifecycleState.VALIDATED.value
        for item in state2.stages
    )
    # The relax stage was never resubmitted: exactly one job per stage.
    registry = session2.registry
    records = registry.list()
    assert len(records) == len(workflow.stages)
    relax_records = [
        record for record in records
        if record.workflow_stage == "relax"
    ]
    assert len(relax_records) == 1
    assert relax_records[0].state is JobLifecycleState.VALIDATED


async def test_resume_revalidates_collected_without_resubmit(tmp_path):
    psp = make_psp(tmp_path)
    molecule = dme_li(tmp_path)
    workflow = build_molecule_workflow(molecule, psp_dir=psp)
    root = tmp_path / "wf_reval"
    backend = FakeSCNetBackend(
        policy=ResourcePolicy(allow_hpc_submit=True), strict=True
    )
    seed_results(backend, tmp_path, converged=False)
    session = make_session(tmp_path, backend)

    first = await run_molecule_workflow(
        workflow, root, session=session, backend=backend, psp_dir=psp,
        remote_psp_dir="",
        wait=True, collect=True, wait_timeout_seconds=30,
        only=["relax"],
    )
    assert first["evidence_count"] == 0
    assert first["stages"][0]["state"] == JobLifecycleState.COLLECTED.value
    assert first["blocked"]
    jobs = len(backend.submitted_scripts)
    assert jobs == 1

    # Resume: the COLLECTED stage is re-validated from disk, never resubmitted.
    second = await run_molecule_workflow(
        workflow, root, session=session, backend=backend, psp_dir=psp,
        remote_psp_dir="",
        wait=True, collect=True, wait_timeout_seconds=30,
        only=["relax"],
    )
    assert second["resumed"] == ["relax"]
    assert second["stages"][0]["state"] == JobLifecycleState.COLLECTED.value
    assert second["evidence_count"] == 0
    assert second["blocked"]
    assert len(backend.submitted_scripts) == jobs  # no resubmission

    # The results on disk were transiently bad; once they are fixed, the same
    # resume re-validates to VALIDATED with no new submission.
    result_dir = root / "results" / "relax"
    write_oszicar(result_dir / "OSZICAR", converged=True)
    third = await run_molecule_workflow(
        workflow, root, session=session, backend=backend, psp_dir=psp,
        remote_psp_dir="",
        wait=True, collect=True, wait_timeout_seconds=30,
        only=["relax"],
    )
    assert third["stages"][0]["state"] == JobLifecycleState.VALIDATED.value
    assert third["evidence_count"] > 0
    assert len(backend.submitted_scripts) == jobs


# ---------------------------------------------------------------------------
# 5. PARCHG orbital artifacts
# ---------------------------------------------------------------------------


def test_orbital_stages_declare_parchg_outputs(tmp_path):
    psp = make_psp(tmp_path)
    workflow = build_molecule_workflow(dme_li(tmp_path), psp_dir=psp)
    orbital = {
        stage.name: stage.produced_outputs
        for stage in workflow.stages
        if stage.name in {StageName.ORBITAL_HOMO, StageName.ORBITAL_LUMO}
    }
    assert "PARCHG" in orbital[StageName.ORBITAL_HOMO]
    assert "PARCHG" in orbital[StageName.ORBITAL_LUMO]


async def test_download_parchg_discovery_variants(tmp_path):
    backend = FakeSCNetBackend()
    remote = "~/photomatagent/orb-homo"
    backend.add_remote_file(remote, "PARCHG", b"plain")
    backend.add_remote_file(remote, "PARCHG.19", b"band 19")
    backend.add_remote_file(remote, "PARCHG.0001.0019", b"band 19 padded")
    backend.add_remote_file(remote, "EIGENVAL", b"x")
    result_dir = tmp_path / "res"
    downloaded = await _download_parchg_artifacts(backend, remote, result_dir)
    names = {path.name for path in downloaded}
    assert {"PARCHG", "PARCHG.19", "PARCHG.0001.0019"} <= names
    assert (result_dir / "PARCHG.19").read_bytes() == b"band 19"
    assert not (result_dir / "EIGENVAL").exists()


async def test_workflow_collect_saves_parchg_artifacts(tmp_path):
    psp = make_psp(tmp_path)
    workflow = build_molecule_workflow(dme_li(tmp_path), psp_dir=psp)
    root = tmp_path / "wf_parchg"
    backend = FakeSCNetBackend(
        policy=ResourcePolicy(allow_hpc_submit=True), strict=True
    )
    seed_results(backend, tmp_path, parchg=True)
    session = make_session(tmp_path, backend)
    report = await run_molecule_workflow(
        workflow, root, session=session, backend=backend, psp_dir=psp,
        remote_psp_dir="",
        wait=True, collect=True, wait_timeout_seconds=30,
    )
    assert report.get("error") is None
    assert report["evidence_count"] > 0
    for stage in (StageName.ORBITAL_HOMO, StageName.ORBITAL_LUMO):
        artifact = root / "results" / stage.value / "PARCHG.19"
        assert artifact.is_file()
        assert artifact.read_bytes() == b"fake parchg band 19"
        assert (root / "results" / stage.value / "PARCHG").is_file()


async def test_tool_collect_saves_parchg_variants(tmp_path):
    """vasp_molecule.collect keeps PARCHG.<band> files for orbital stages."""
    psp = make_psp(tmp_path)
    workflow = build_molecule_workflow(dme_li(tmp_path), psp_dir=psp)
    root = tmp_path / "wf_tool_collect"
    backend = FakeSCNetBackend(
        policy=ResourcePolicy(allow_hpc_submit=True), strict=True
    )
    session = make_session(tmp_path, backend)
    tool = make_tools(
        tmp_path, backend, psp=psp, workflow_dir=root, remote_psp_dir=""
    )
    await tool.prepare(workflow, output_dir=root)

    # A completed orbital job whose remote directory holds results plus a
    # VASP 5.4.4 PARCHG.<band> density.
    remote_dir = "~/photomatagent/DME_Li-orbital_homo-fake"
    bulk = tmp_path / "bulk_orb"
    bulk.mkdir(exist_ok=True)
    write_eigenval(bulk / "EIGENVAL", nelect=38)
    write_oszicar(bulk / "OSZICAR", converged=True)
    write_minimal_vasprun(bulk / "vasprun.xml")
    write_locpot(bulk / "LOCPOT")
    for name in ("EIGENVAL", "OSZICAR", "vasprun.xml", "LOCPOT"):
        backend.add_remote_file(remote_dir, name, (bulk / name).read_bytes())
    backend.add_remote_file(remote_dir, "PARCHG.19", b"band 19 density")

    from photomatagent.scientific.applications.vasp.molecular.workflow import (
        TaskState,
    )

    task_state = TaskState(
        workflow_dir=str(root), molecule_name="DME_Li",
        stages=[
            StageTask(
                stage="orbital_homo",
                state=JobLifecycleState.COMPLETED.value,
                request_id="orb-request-1",
                job_id="2001",
                stage_dir=str(root / "inputs" / "04_orbital_homo"),
                remote_directory=remote_dir,
            )
        ],
    )
    save_task_state(root, task_state)

    collected = await tool.collect(StageName.ORBITAL_HOMO)
    assert collected["ok"] is True
    assert collected["summary"]["validated"] is True
    artifact = root / "results" / "orbital_homo" / "PARCHG.19"
    assert artifact.is_file()
    assert artifact.read_bytes() == b"band 19 density"
    # task_state advanced COMPLETED -> COLLECTED -> VALIDATED in lockstep
    # with the SQLite registry (the record stays in sync when it exists).
    state = load_task_state(root)
    assert state is not None
    entry = state.stage_map()["orbital_homo"]
    assert entry.state == JobLifecycleState.VALIDATED.value
    assert entry.validated is True
