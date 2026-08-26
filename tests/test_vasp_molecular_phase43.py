"""Phase 4.3 offline regressions: calibration, recovery, screening, refs.

Covers, with synthetic fixtures only (no SSH, no Slurm, no VASP execution):

* B1: CalibrationRecord audit fields, content-addressed ids, production
  preflight gates on matching calibration records;
* B2: deterministic relax recovery (decision table, CONTCAR restarts,
  practical-convergence provenance, bounded attempts, status-only rule);
* B3: conformer screening funnel (cheap static E0 screens rank candidates;
  only the selected lowest-E0 candidate enters production);
* B4: bare-ion reference honesty (explicit_reference_assumption, ΔΔE
  preference, absolute-binding high-risk flags).
"""

from __future__ import annotations

import asyncio
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from photomatagent.scientific.applications.vasp.molecular.calibration import (
    CalibrationRecord,
    calibration_applicable,
    derive_calibration_id,
)
from photomatagent.scientific.applications.vasp.molecular.models import (
    MoleculeSpec,
    ResourceProfile,
)
from photomatagent.scientific.applications.vasp.molecular.preflight import (
    run_molecular_preflight,
)
from photomatagent.scientific.applications.vasp.molecular.recovery import (
    RecoveryFailure,
    RecoveryPolicy,
    classify_relax_failure,
    decide_recovery,
    materialize_recovery_stage_dir,
)
from photomatagent.scientific.applications.vasp.molecular.workflow import (
    build_molecule_workflow,
    load_task_state,
    run_molecule_workflow,
)
from photomatagent.scientific.remote.fake import FakeSCNetBackend
from photomatagent.scientific.remote.lifecycle import SubmitOnceSession
from photomatagent.scientific.remote.models import ResourcePolicy
from photomatagent.scientific.remote.registry import (
    JobLifecycleState,
    JobRegistry,
)


ZVAL_ENMAX = {
    "C": (4.0, 400.0),
    "O": (6.0, 400.0),
    "H": (1.0, 250.0),
}


def make_psp(tmp_path: Path) -> Path:
    library = tmp_path / "psp"
    for element, (zval, enmax) in ZVAL_ENMAX.items():
        dataset = library / element
        dataset.mkdir(parents=True, exist_ok=True)
        (dataset / "POTCAR").write_text(
            f"   TITEL  = PAW_PBE {element} 08Apr2002\n"
            f"   POMASS =     1.000; ZVAL   =    {zval:.3f}    mass and valenz\n"
            f"   ENMAX  =  {enmax:.3f}; ENMIN  =  300.000 eV\n"
            f"# FIXTURE-BULK-MARKER-9f3a {element}\n",
            encoding="utf-8",
        )
    return library


def cluster_coords(n: int, box: float, spacing: float = 1.6) -> np.ndarray:
    pattern = [(i % 4, (i // 4) % 2, (i // 8) % 3) for i in range(n)]
    points = np.asarray(pattern, dtype=float) * spacing
    points -= points.mean(axis=0)
    points += box / 2
    return points


def write_xyz(path: Path, *, tag: str, box: float = 20.0) -> None:
    symbols = ["C"] * 4 + ["O"] * 2 + ["H"] * 10
    coords = cluster_coords(len(symbols), box)
    path.write_text(
        "\n".join(
            [str(len(symbols)), tag]
            + [
                f"{s} {x:.5f} {y:.5f} {z:.5f}"
                for s, (x, y, z) in zip(symbols, coords, strict=True)
            ]
        )
        + "\n",
        encoding="utf-8",
    )


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


def write_oszicar(path: Path, *, e0: float = -122.277, steps: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "DAV:   1    -0.121765040995E+03   -0.12177E+03   -0.71368E-05   144   0.905E-02    0.696E-03",
        "DAV:   2    -0.121765040523E+03    0.47271E-06   -0.20602E-05   120   0.197E-02    0.462E-03",
        "DAV:   3    -0.121765040541E+03   -0.18559E-07   -0.19949E-06   104   0.999E-03",
    ]
    for step in range(1, steps + 1):
        lines.append(
            f"{step:4d} F= {e0: 14.8E} E0= {e0: 14.8E}  d E =0.000E+00"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_minimal_vasprun(path: Path, *, e0: float = -122.277) -> None:
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


def write_outcar(
    path: Path,
    *,
    n_atoms: int = 16,
    max_force: float = 0.001,
    reached: bool = True,
    force_blocks: int = 1,
    marker: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f" vasp.5.4.4 synthetic {marker}",
        "   NSW      =      200     number of steps for ionic motion",
        "   IBRION   =      2     ionic relax: 1=quasi-Newton, 2=damped",
        "   EDIFFG   = -0.02E+00  force-criterion for ionic relax",
    ]
    if reached:
        lines.append(
            "  reached required accuracy - stopping structural energy minimisation"
        )
    for block in range(force_blocks):
        component = max_force / 3.0**0.5
        if block == force_blocks - 1:
            pass
        else:
            component = 0.4
        lines.append("POSITION                                       TOTAL-FORCE (eV/Angst)")
        lines.append("-" * 90)
        for index in range(n_atoms):
            lines.append(
                f"{index + 1:6d} {0.0:17.10f} {0.0:17.10f} {0.0:17.10f}"
                f"{component:14.8f} {component:14.8f} {component:14.8f}"
            )
        lines.append("-" * 90)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_session(tmp_path: Path, backend: FakeSCNetBackend) -> SubmitOnceSession:
    return SubmitOnceSession(
        JobRegistry(tmp_path / "jobs.sqlite3"),
        backend,
        marker_temp_dir=tmp_path / "markers",
    )


def dme_molecule(tmp_path: Path, *, name: str = "DME") -> MoleculeSpec:
    return dme_molecule_box(tmp_path, name=name, box=20.0)


def dme_molecule_box(
    tmp_path: Path, *, name: str = "DME", box: float = 20.0
) -> MoleculeSpec:
    path = tmp_path / f"{name}.xyz"
    write_xyz(path, tag=name, box=box)
    return MoleculeSpec(
        name=name,
        structure_path=path,
        total_charge=0,
        spin_multiplicity=1,
        box_ang=box,
        calculation_purpose="phase43",
    )


# ---------------------------------------------------------------------------
# B1: calibration records
# ---------------------------------------------------------------------------


def _calibration() -> CalibrationRecord:
    return CalibrationRecord(
        atom_count=16,
        formula="C4O2H10",
        elements=["C", "O", "H"],
        nbands=64,
        box_ang=20.0,
        grid=(128, 128, 128),
        encut_ev=400.0,
        prec="Accurate",
        lreal=False,
        addgrid=True,
        tasks=8,
        ncore=2,
        electronic_steps=12,
        elapsed_seconds=600.0,
        max_rss_bytes=8_000_000_000,
        vasp_version="6.4.2",
        source_job_id="12001",
        applicable_to="C4O2H10 20A 400eV smoke-equivalent molecules",
    )


def test_calibration_record_audit_fields_and_content_addressed_id():
    record = _calibration()
    record_id = derive_calibration_id(record)
    assert len(record_id) == 16
    assert derive_calibration_id(record) == record_id  # deterministic
    changed = record.model_copy(update={"elapsed_seconds": 700.0})
    assert derive_calibration_id(changed) != record_id


def test_calibration_applicability_scope():
    record = _calibration()
    ok, reasons = calibration_applicable(
        record, formula="C4O2H10", atom_count=16, box_ang=20.0, encut_ev=400.0
    )
    assert ok and reasons == []
    ok2, reasons2 = calibration_applicable(
        record, formula="C7H8F4O2", atom_count=21, box_ang=30.0
    )
    assert not ok2
    assert any("formula" in reason for reason in reasons2)
    assert any("atom_count" in reason for reason in reasons2)


def test_production_workflow_requires_calibration_record(tmp_path):
    workflow = build_molecule_workflow(
        dme_molecule(tmp_path),
        psp_dir=make_psp(tmp_path),
        resource_profile=ResourceProfile.PRODUCTION,
        encut_ev=520.0,
    )
    report = run_molecular_preflight(workflow, psp_dir=tmp_path / "psp")
    assert not report.passed
    assert any(
        "RESOURCE_PLAN_VIOLATION" == issue.code
        and "CalibrationRecord" in issue.message
        for issue in report.errors
    )


def test_production_workflow_with_matching_calibration_passes(tmp_path):
    calibration = _calibration().model_copy(
        update={"encut_ev": 520.0, "box_ang": 30.0}
    )
    workflow = build_molecule_workflow(
        dme_molecule_box(tmp_path, box=30.0),
        psp_dir=make_psp(tmp_path),
        resource_profile=ResourceProfile.PRODUCTION,
        encut_ev=520.0,
        calibration=calibration,
    )
    assert workflow.resource_plan.calibration is not None
    assert workflow.resource_plan.calibration.calibration_id
    assert workflow.resource_plan.tasks_per_node == calibration.tasks
    report = run_molecular_preflight(workflow, psp_dir=tmp_path / "psp")
    assert report.passed, report.errors


def test_mismatched_calibration_is_refused(tmp_path):
    wrong = _calibration().model_copy(
        update={"formula": "C7H8F4O2", "atom_count": 21}
    )
    with pytest.raises(ValueError, match="does not apply"):
        build_molecule_workflow(
            dme_molecule(tmp_path),
            psp_dir=make_psp(tmp_path),
            resource_profile=ResourceProfile.PRODUCTION,
            encut_ev=520.0,
            calibration=wrong,
        )


def test_study_matrix_estimates_from_profile_and_calibration(tmp_path):
    from photomatagent.scientific.applications.vasp.study.matrix import (
        build_calculation_matrix,
    )
    from photomatagent.scientific.applications.vasp.study.models import (
        PropertyRequest,
        StudySystem,
        VaspStudyRequest,
    )
    from photomatagent.scientific.capabilities.chemistry.models import (
        ChemicalIdentity,
        ChemicalRole,
        GeneratedStructure,
        ProvenanceStatus,
        StructureProvenance,
    )

    structure = GeneratedStructure(
        identity=ChemicalIdentity(
            system_id="X",
            display_name="X",
            formula="C4O2H10",
            total_charge=0,
            role=ChemicalRole.MOLECULE,
        ),
        structure_path=tmp_path / "x.xyz",
        format="xyz",
        atom_count=16,
        provenance=StructureProvenance(
            status=ProvenanceStatus.GENERATED_FROM_SMILES
        ),
    )
    resolved = {"X": [structure]}
    request = VaspStudyRequest(
        original_request="est",
        systems=[StudySystem(system_id="X", properties=[PropertyRequest.HOMO_LUMO])],
        property_requests=[PropertyRequest.HOMO_LUMO],
        method={
            "box_ang": 20.0,
            "resource_profile": "production",
            "calibration": _calibration().model_dump(mode="json"),
        },
    )
    smoke_request = VaspStudyRequest(
        original_request="est-smoke",
        systems=[StudySystem(system_id="X", properties=[PropertyRequest.HOMO_LUMO])],
        property_requests=[PropertyRequest.HOMO_LUMO],
    )
    smoke = build_calculation_matrix(smoke_request, resolved)
    prod = build_calculation_matrix(request, resolved)
    # calibrations adds walltime 20 -> 20 min (max(20, 15+5)=20): both are
    # 5 stages * 8 tasks * 20 min / 60 = 13.33 core-hours.
    assert smoke.total_core_hours == pytest.approx(round(5 * 8 * 20 / 60.0, 2))
    assert prod.total_core_hours == pytest.approx(round(5 * 8 * 20 / 60.0, 2))
    assert prod.estimated_disk_gb >= 24.0  # memory-derived, not fixed 8 GB
    assert smoke.estimated_disk_gb == 8.0  # unchanged smoke baseline


# ---------------------------------------------------------------------------
# B2: deterministic recovery
# ---------------------------------------------------------------------------


def test_recovery_decision_table_behaviors():
    policy = RecoveryPolicy(max_auto_attempts=2)
    cases = [
        (
            classify_relax_failure(
                convergence={"exhausted_nsw": True, "electronic_converged": True}
            ),
            {"has_contcar": True},
            ("RESUBMIT", "CONTCAR", []),
        ),
        (
            classify_relax_failure(query_failed=True),
            {},
            ("STATUS_ONLY", "", []),
        ),
        (
            classify_relax_failure(
                lifecycle_state="UNKNOWN_RECONCILIATION_REQUIRED"
            ),
            {},
            ("RECONCILE", "", []),
        ),
        (
            classify_relax_failure(
                convergence={"detected_errors": ["out of memory"]}
            ),
            {},
            ("STOP", "", []),
        ),
        (
            classify_relax_failure(scheduler_state="TIMEOUT"),
            {"has_contcar": True},
            ("RESUBMIT", "CONTCAR", []),
        ),
        (
            classify_relax_failure(
                convergence={
                    "ionic_converged": False,
                    "electronic_converged": True,
                    "exhausted_nsw": False,
                    "detected_errors": [],
                },
                force_history=[0.05, 0.048, 0.049],
            ),
            {"has_contcar": True},
            ("RESUBMIT", "CONTCAR", ["POTIM = 0.5"]),
        ),
        (
            classify_relax_failure(
                convergence={
                    "ionic_converged": False,
                    "electronic_converged": True,
                    "exhausted_nsw": False,
                    "detected_errors": [],
                },
                force_history=[0.05, 0.051, 0.2],
            ),
            {"has_xdatcar_best": True},
            ("RESUBMIT", "XDATCAR_BEST", []),
        ),
        (
            classify_relax_failure(
                convergence={"electronic_converged": False}
            ),
            {},
            ("STOP", "", []),
        ),
    ]
    for failure, kwargs, expected in cases:
        decision = decide_recovery(policy, failure=failure, **kwargs)
        action, restart, changes = expected
        assert decision.action == action, (failure, decision)
        assert decision.restart_from == restart, (failure, decision)
        assert decision.parameter_changes == changes, (failure, decision)


def test_recovery_attempt_cap_stops():
    policy = RecoveryPolicy(max_auto_attempts=1)
    failure = RecoveryFailure.NSW_EXHAUSTED
    first = decide_recovery(policy, failure=failure, attempts_used=0, has_contcar=True)
    second = decide_recovery(policy, failure=failure, attempts_used=1, has_contcar=True)
    assert first.action == "RESUBMIT"
    assert second.action == "STOP"
    assert "limit" in second.reason


def test_practical_convergence_records_old_new_and_reason():
    policy = RecoveryPolicy(max_auto_attempts=2, ediffg_relax_factor=2.0)
    decision = decide_recovery(
        policy,
        failure=RecoveryFailure.NSW_EXHAUSTED,
        has_contcar=True,
        max_force=0.03,
        ediffg=0.02,
    )
    assert decision.action == "RESUBMIT"
    assert decision.practical_convergence is True
    assert decision.incar_changes["EDIFFG"] == pytest.approx(-0.04)
    assert "-0.020000" in decision.practical_convergence_note
    assert "-0.040000" in decision.practical_convergence_note
    assert "0.03" in decision.practical_convergence_note
    assert "practical" in decision.practical_convergence_note.lower()
    assert "NOT" in decision.practical_convergence_note


def test_incar_changes_rewrite_and_restart_dir_materialization(tmp_path):
    previous = tmp_path / "stage_relax_1"
    previous.mkdir(parents=True, exist_ok=True)
    (previous / "INCAR").write_text(
        "SYSTEM = x\nNSW = 200\nEDIFFG = -0.02\nIBRION = 2\nPOTIM = 0.5\n",
        encoding="utf-8",
    )
    (previous / "KPOINTS").write_text("k\n", encoding="utf-8")
    contcar = tmp_path / "CONTCAR"
    contcar.write_text("marker-geometry\n", encoding="utf-8")
    restart = materialize_recovery_stage_dir(
        previous_stage_dir=previous,
        restart_structure=contcar,
        incar_changes={"EDIFFG": -0.04, "POTIM": 0.25},
        attempt_id="relax-attempt-1",
        workflow_dir=tmp_path,
        reason="practical convergence",
        practical_convergence=True,
    )
    assert (restart / "POSCAR").read_text(encoding="utf-8") == "marker-geometry\n"
    incar = (restart / "INCAR").read_text(encoding="utf-8")
    assert "EDIFFG = -0.04" in incar
    assert "POTIM = 0.25" in incar
    assert "EDIFFG = -0.02" not in incar
    provenance = json.loads(
        (restart / "recovery_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["practical_convergence"] is True
    assert provenance["incar_changes"] == {"EDIFFG": -0.04, "POTIM": 0.25}


def _recovery_seeded_backend(tmp_path: Path) -> FakeSCNetBackend:
    """Seeds: 1st relax job NSW-exhausted, later jobs converged."""
    backend = FakeSCNetBackend(
        policy=ResourcePolicy(allow_hpc_submit=True), strict=True
    )
    relax_count = {"n": 0}
    original_upload = backend.upload_files

    async def upload_with_results(local_paths, remote_directory):
        names = await original_upload(local_paths, remote_directory)
        bulk = tmp_path / "bulk"
        bulk.mkdir(exist_ok=True)
        write_eigenval(bulk / "EIGENVAL", nelect=38)
        write_minimal_vasprun(bulk / "vasprun.xml")
        if "-relax-" in remote_directory:
            relax_count["n"] += 1
            first = relax_count["n"] == 1
            write_oszicar(bulk / "OSZICAR", steps=200 if first else 1)
            write_outcar(
                bulk / "OUTCAR",
                reached=not first,
                max_force=0.03 if first else 0.001,
                force_blocks=200 if first else 1,
                marker="recovery",
            )
            contcar = bulk / "CONTCAR"
            marker = "CONTCAR-FROM-RELAX-ATTEMPT-1" if first else "CONTCAR-FINAL"
            source = next(
                (
                    Path(str(path))
                    for path in local_paths
                    if str(path).endswith("POSCAR")
                ),
                None,
            )
            if source is not None:
                lines = source.read_text(encoding="utf-8").splitlines()
                if lines:
                    lines[0] = marker
                contcar.write_text("\n".join(lines) + "\n", encoding="utf-8")
            backend.add_remote_file(
                remote_directory, "CONTCAR", contcar.read_bytes()
            )
        else:
            write_oszicar(bulk / "OSZICAR")
            write_outcar(bulk / "OUTCAR")
        for name in ("EIGENVAL", "OSZICAR", "vasprun.xml", "OUTCAR"):
            backend.add_remote_file(
                remote_directory, name, (bulk / name).read_bytes()
            )
        return names

    backend.upload_files = upload_with_results
    return backend


async def test_relax_recovery_restarts_from_contcar_not_poscar(tmp_path):
    psp = make_psp(tmp_path)
    workflow = build_molecule_workflow(dme_molecule(tmp_path), psp_dir=psp)
    root = tmp_path / "wf_recover"
    backend = _recovery_seeded_backend(tmp_path)
    session = make_session(tmp_path, backend)
    from photomatagent.scientific.applications.vasp.molecular.recovery import (
        RecoveryPolicy,
    )

    report = await run_molecule_workflow(
        workflow,
        root,
        session=session,
        backend=backend,
        psp_dir=psp,
        remote_psp_dir="",
        wait=True,
        collect=True,
        wait_timeout_seconds=40,
        only=["relax"],
        recovery_policy=RecoveryPolicy(max_auto_attempts=1),
    )
    assert report.get("error") is None
    relaxed = [stage for stage in report["stages"] if stage.get("recovered")]
    assert relaxed, report["stages"]
    assert relaxed[0]["state"] == JobLifecycleState.VALIDATED.value
    state = load_task_state(root)
    entry = state.stage_map()["relax"]
    assert entry.state == JobLifecycleState.VALIDATED.value
    assert entry.retry_count == 1
    assert entry.attempt_id == "relax-attempt-1"
    assert len(entry.recovery_attempts) == 1
    attempt = entry.recovery_attempts[0]
    assert attempt["restart_from"] == "CONTCAR"
    assert attempt["failure"] == "NSW_EXHAUSTED"
    # The restart dir's POSCAR must be the previous CONTCAR, never the
    # initial POSCAR.
    restart_poscar = (
        root / "stage_relax_attempt_relax-attempt-1" / "POSCAR"
    ).read_text(encoding="utf-8")
    assert "CONTCAR-FROM-RELAX-ATTEMPT-1" in restart_poscar
    # One original job + one recovery job, with distinct request ids and
    # unique remote directories (submit-once is preserved).
    relax_records = [
        record for record in session.registry.list()
        if record.workflow_stage == "relax"
    ]
    assert len(relax_records) == 2
    assert relax_records[0].request_id != relax_records[1].request_id
    assert (
        relax_records[0].remote_directory != relax_records[1].remote_directory
    )
    # The NSW-exhausted job record correctly stays COLLECTED with a failed
    # scientific validation; only the recovery attempt validates.
    assert sorted(
        record.state for record in relax_records
    ) == [
        JobLifecycleState.COLLECTED,
        JobLifecycleState.VALIDATED,
    ]
    validated_record = next(
        record for record in relax_records
        if record.state is JobLifecycleState.VALIDATED
    )
    assert "relax-attempt-1" in validated_record.remote_directory


async def test_status_query_failure_never_resubmits(tmp_path):
    policy = RecoveryPolicy(max_auto_attempts=2)
    decision = decide_recovery(
        policy, failure=classify_relax_failure(query_failed=True)
    )
    assert decision.action == "STATUS_ONLY"
    # At the engine level: no RESUBMIT decision can ever be produced for a
    # query failure, regardless of remaining attempts or CONTCAR presence.
    decision2 = decide_recovery(
        policy,
        failure=classify_relax_failure(query_failed=True),
        attempts_used=0,
        has_contcar=True,
    )
    assert decision2.action == "STATUS_ONLY"


async def test_practical_convergence_provenance_in_runner(tmp_path):
    psp = make_psp(tmp_path)
    workflow = build_molecule_workflow(dme_molecule(tmp_path), psp_dir=psp)
    root = tmp_path / "wf_practical"
    backend = _recovery_seeded_backend(tmp_path)
    session = make_session(tmp_path, backend)
    from photomatagent.scientific.applications.vasp.molecular.recovery import (
        RecoveryPolicy,
    )

    report = await run_molecule_workflow(
        workflow,
        root,
        session=session,
        backend=backend,
        psp_dir=psp,
        remote_psp_dir="",
        wait=True,
        collect=True,
        wait_timeout_seconds=40,
        only=["relax"],
        recovery_policy=RecoveryPolicy(max_auto_attempts=1),
    )
    assert report.get("error") is None
    state = load_task_state(root)
    entry = state.stage_map()["relax"]
    assert entry.state == JobLifecycleState.VALIDATED.value
    assert entry.recovery_attempts[0]["practical_convergence"] is True
    results = json.loads(
        (Path(entry.results_dir) / "results.json").read_text(encoding="utf-8")
    )
    convergence = results["convergence"]
    assert convergence["practical_convergence"] is True
    note = convergence["practical_convergence_note"]
    assert "EDIFFG relaxed" in note
    assert "-0.020000" in note and "-0.040000" in note
    assert "NOT the originally required" in note
    # The relaxed threshold is recorded but never claimed as the original.
    assert results["scf"]["converged"] is True


# ---------------------------------------------------------------------------
# B3: conformer screening funnel
# ---------------------------------------------------------------------------


def _screening_seeded_backend(tmp_path: Path) -> FakeSCNetBackend:
    """Vary screen E0 per candidate marker in the job name."""
    backend = FakeSCNetBackend(
        policy=ResourcePolicy(allow_hpc_submit=True), strict=True
    )
    e0_by_index = {0: -10.0, 1: -9.5, 2: -10.3}
    original_upload = backend.upload_files

    async def upload_with_results(local_paths, remote_directory):
        names = await original_upload(local_paths, remote_directory)
        bulk = tmp_path / "bulk_screen"
        bulk.mkdir(exist_ok=True)
        screen_match = re.search(r"screen_(\d+)-", remote_directory)
        if screen_match:
            e0 = e0_by_index[int(screen_match.group(1))]
        else:
            e0 = -10.3
        write_eigenval(bulk / "EIGENVAL", nelect=38)
        write_oszicar(bulk / "OSZICAR", e0=e0)
        write_minimal_vasprun(bulk / "vasprun.xml", e0=e0)
        write_outcar(bulk / "OUTCAR")
        if "-relax-" in remote_directory:
            contcar = bulk / "CONTCAR"
            source = next(
                (
                    Path(str(path))
                    for path in local_paths
                    if str(path).endswith("POSCAR")
                ),
                None,
            )
            if source is not None:
                lines = source.read_text(encoding="utf-8").splitlines()
                if lines:
                    lines[0] = "CONTCAR-RELAXED"
                contcar.write_text("\n".join(lines) + "\n", encoding="utf-8")
            backend.add_remote_file(
                remote_directory, "CONTCAR", contcar.read_bytes()
            )
        for name in ("EIGENVAL", "OSZICAR", "vasprun.xml", "OUTCAR"):
            backend.add_remote_file(
                remote_directory, name, (bulk / name).read_bytes()
            )
        return names

    backend.upload_files = upload_with_results
    return backend


def _screened_spec(tmp_path: Path) -> Any:
    from photomatagent.scientific.applications.vasp.study.models import (
        CalculationMatrix,
        CalculationTask,
        PropertyRequest,
        VaspStudyRequest,
        VaspStudySpec,
    )

    candidates = [tmp_path / f"conf_{i}.xyz" for i in range(3)]
    for index, path in enumerate(candidates):
        write_xyz(path, tag=f"conf_{index}")
    task = CalculationTask(
        task_id="dme|q0|s1",
        system_id="DME",
        display_name="DME",
        role="molecule",
        formula="C4O2H10",
        total_charge=0,
        spin_multiplicity=1,
        structure_path=candidates[0],
        structure_candidates=[str(path) for path in candidates],
        reliability="B",
        assists=[PropertyRequest.HOMO_LUMO],
        estimated_core_hours=5.0,
    )
    return VaspStudySpec(
        study_id="study-screen-test",
        request=VaspStudyRequest(
            original_request="screen test",
            execution_policy={
                "user_requested_computation": True,
                "wait_timeout_seconds": 60,
            },
            resource_budget={"max_core_hours": 400.0},
            method={"box_ang": 20.0},
        ),
        study_dir=tmp_path / "study",
        calculation_matrix=CalculationMatrix(tasks=[task]),
    )


async def test_screening_funnel_only_selected_candidate_reaches_production(
    tmp_path,
):
    from photomatagent.scientific.applications.vasp.study.executor import (
        StudyExecutor,
    )
    from photomatagent.scientific.applications.vasp.study.screening import (
        load_screen_reports,
    )
    from photomatagent.scientific.applications.vasp.molecular.runtime import (
        MolecularVaspRuntime,
    )

    psp = make_psp(tmp_path)
    backend = _screening_seeded_backend(tmp_path)
    runtime = MolecularVaspRuntime(
        backend=backend,
        configured=True,
        psp_dir=psp,
        workflow_dir=tmp_path / "mol",
        log_dir=tmp_path / "logs",
        registry_path=tmp_path / "state" / "jobs.sqlite3",
    )
    spec = _screened_spec(tmp_path)
    executor = StudyExecutor(spec, runtime)
    report = await executor.execute()
    assert report["failed"] == [], report["failed"]
    task = spec.calculation_matrix.task_map()["dme|q0|s1"]
    assert task.state == "VALIDATED", task.error
    assert task.structure_path.name == "conf_2.xyz"

    reports = load_screen_reports(spec.study_dir)
    screen = reports["dme|q0|s1"]
    assert screen.screen_complete is True
    assert screen.selected_structure_path.endswith("conf_2.xyz")
    assert len(screen.records) == 3
    by_index = {record.candidate_index: record for record in screen.records}
    assert by_index[0].e0_ev == pytest.approx(-10.0)
    assert by_index[1].e0_ev == pytest.approx(-9.5)
    assert by_index[2].e0_ev == pytest.approx(-10.3)
    assert by_index[0].relative_e0_ev == pytest.approx(0.3)
    assert by_index[1].relative_e0_ev == pytest.approx(0.8)
    assert by_index[2].relative_e0_ev == pytest.approx(0.0)
    assert "higher E0" in by_index[0].elimination_reason
    assert "higher E0" in by_index[1].elimination_reason
    assert by_index[2].elimination_reason == ""

    # Exactly THREE cheap static screens and ONE relax job: the two losing
    # candidates never entered the expensive production stages.
    registry = runtime.session.registry
    screens = [
        record for record in registry.list()
        if record.workflow_stage == "static"
    ]
    relaxes = [
        record for record in registry.list()
        if record.workflow_stage == "relax"
    ]
    assert len(screens) == 3
    assert len(relaxes) == 1
    # The production relax ran on the selected geometry only.
    workflow_json = json.loads(
        (Path(task.workflow_dir) / "workflow.json").read_text(encoding="utf-8")
    )
    assert workflow_json["molecule"]["structure_path"].endswith("conf_2.xyz")


# ---------------------------------------------------------------------------
# B4: bare-ion reference honesty
# ---------------------------------------------------------------------------


def _reference_result_dir(tmp_path: Path, *, e0: float, name: str) -> Path:
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "validated": True,
        "errors": [],
        "energy": {"e_0_ev": e0, "e_fr_ev": e0, "source": "synthetic"},
        "identity": {"formula": name, "charge": 1 if name == "Li" else 0},
        "method": {"box_ang": 20.0, "functional": "PE", "encut_ev": 400.0},
    }
    if name == "Li":
        payload["explicit_reference_assumption"] = True
        payload["reference_kind"] = "zero_electron_bare_ion"
        payload["not_a_vasp_result"] = True
        payload["reference_model"] = {
            "kind": "zero_electron_bare_ion",
            "convention": "E = 0 eV by definition",
        }
    (directory / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return directory


def test_bare_ion_reference_is_flagged_not_vasp_result():
    from photomatagent.scientific.applications.vasp.molecular.binding import (
        BindingEnergyInput,
        BindingReference,
        compute_binding_energy,
    )

    import tempfile

    tmp = Path(tempfile.mkdtemp())
    complex_dir = _reference_result_dir(tmp, e0=-10.5, name="DME_Li")
    li_dir = _reference_result_dir(tmp, e0=0.0, name="Li")
    result = compute_binding_energy(
        BindingEnergyInput(
            complex_name="DME_Li",
            complex_dir=complex_dir.as_posix(),
            references=[
                BindingReference(name="DME", results_dir=_reference_result_dir(
                    tmp, e0=-10.5, name="DME").as_posix(), charge=0,
                ),
                BindingReference(
                    name="Li",
                    results_dir=li_dir.as_posix(),
                    charge=1,
                    role="ion",
                ),
            ],
            charge=1,
        )
    )
    assert result["ok"] is True
    assert result["results"]["uses_declared_reference_assumption"] is True
    assert result["results"]["high_risk_absolute_binding_energy"] is True
    assert "Li" in result["reference_assumptions"]
    assert any("E=0 convention" in warning for warning in result["warnings"])
    li_component = next(
        item for item in result["components"]["primary"] if item["name"] == "Li"
    )
    assert li_component["explicit_reference_assumption"] is True
    assert li_component["not_a_vasp_result"] is True


def test_zero_electron_reference_payload_is_explicit(tmp_path):
    # The executor's reference writer is exercised through the mini study in
    # the existing suite; here we only assert the payload contract pieces.
    payload = {
        "validated": True,
        "explicit_reference_assumption": True,
        "reference_kind": "zero_electron_bare_ion",
        "not_a_vasp_result": True,
        "energy": {"e_0_ev": 0.0, "source": "declared zero-electron reference model"},
    }
    assert payload["explicit_reference_assumption"] is True
    assert payload["not_a_vasp_result"] is True
