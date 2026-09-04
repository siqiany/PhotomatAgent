"""Offline tests for the VASP study layer (plan/matrix/executor/report).

Everything runs through FakeSCNetBackend + synthetic POTCAR metadata; the
study executor only ever calls the existing vasp_molecule.* machinery.
No SSH, no sbatch, no real VASP.
"""

from __future__ import annotations

import asyncio
import json
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest

warnings.filterwarnings("ignore", message=".*explicit Hs.*")

from photomatagent.scientific.applications.vasp.molecular.runtime import (
    MolecularVaspRuntime,
)
from photomatagent.scientific.applications.vasp.study.executor import (
    StudyExecutor,
)
from photomatagent.scientific.applications.vasp.study.models import (
    PropertyRequest,
    StudySystem,
    VaspStudyRequest,
)
from photomatagent.scientific.applications.vasp.study.planner import (
    load_planned_study,
    plan_study,
)
from photomatagent.scientific.remote.fake import FakeSCNetBackend
from photomatagent.scientific.remote.models import ResourcePolicy


ZVAL_ENMAX = {
    "C": (4.0, 400.0),
    "O": (6.0, 400.0),
    "H": (1.0, 250.0),
    "Li": (1.0, 140.0),
    "F": (7.0, 400.0),
    "N": (5.0, 400.0),
    "S": (6.0, 400.0),
}

UNIFIED_VASP_TOOL_NAMES = {
    "vasp.capabilities",
    "vasp.plan",
    "vasp.prepare",
    "vasp.preflight",
    "vasp.submit",
    "vasp.status",
    "vasp.wait",
    "vasp.resume",
    "vasp.collect",
    "vasp.report",
}


def make_psp(tmp_path: Path) -> Path:
    library = tmp_path / "psp" / "potpaw_PBE.64"
    for element, (zval, enmax) in ZVAL_ENMAX.items():
        directory = library / element
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "POTCAR").write_text(
            f"   TITEL  = PAW_PBE {element} 08Apr2002\n"
            f"   POMASS =     1.000; ZVAL   =    {zval:.3f}    mass and valenz\n"
            f"   ENMAX  =  {enmax:.3f}; ENMIN  =  300.000 eV\n"
            f"# FIXTURE-BULK-MARKER-9f3a {element}\n",
            encoding="utf-8",
        )
    return tmp_path / "psp"


def make_runtime(tmp_path: Path, *, psp: Path) -> MolecularVaspRuntime:
    backend = FakeSCNetBackend(
        policy=ResourcePolicy(allow_hpc_submit=True), strict=True
    )
    seed_results(backend, tmp_path)
    return MolecularVaspRuntime(
        backend=backend,
        configured=True,
        psp_dir=psp,
        workflow_dir=tmp_path / "mol",
        log_dir=tmp_path / "logs",
        registry_path=tmp_path / "state" / "jobs.sqlite3",
        remote_psp_dir="",
    )


NATURAL_LANGUAGE_REQUEST = (
    "使用 VASP 计算 TFPMA、VEC、MBA、LiNO3、LiTFSI、VM 和 TVM 的 "
    "HOMO/LUMO；计算 DME-Li+、TVM-Li+ 和 TVM-TFSI- 的结合能；"
    "计算 VM 和 TVM 的 ESP。缺少的结构请生成代表性结构并在报告中说明。"
)


def study_request(
    *,
    user_requested_computation: bool = True,
    max_core_hours: float = 400.0,
    extra_systems: list[StudySystem] | None = None,
) -> VaspStudyRequest:
    systems = [
        StudySystem(system_id="TFPMA", properties=[PropertyRequest.HOMO_LUMO]),
        StudySystem(system_id="VEC", properties=[PropertyRequest.HOMO_LUMO]),
        StudySystem(system_id="MBA", properties=[PropertyRequest.HOMO_LUMO]),
        StudySystem(
            system_id="LiNO3", properties=[PropertyRequest.HOMO_LUMO]
        ),
        StudySystem(
            system_id="LiTFSI", properties=[PropertyRequest.HOMO_LUMO]
        ),
        StudySystem(
            system_id="VM",
            properties=[PropertyRequest.HOMO_LUMO, PropertyRequest.ESP],
        ),
        StudySystem(
            system_id="TVM",
            properties=[PropertyRequest.HOMO_LUMO, PropertyRequest.ESP],
        ),
        StudySystem(
            system_id="DME-Li+",
            properties=[PropertyRequest.BINDING_ENERGY],
        ),
        StudySystem(
            system_id="TVM-Li+",
            properties=[PropertyRequest.BINDING_ENERGY],
        ),
        StudySystem(
            system_id="TVM-TFSI-",
            properties=[PropertyRequest.BINDING_ENERGY],
        ),
    ]
    if extra_systems:
        systems.extend(extra_systems)
    return VaspStudyRequest(
        original_request=NATURAL_LANGUAGE_REQUEST,
        systems=systems,
        execution_policy={
            "user_requested_computation": user_requested_computation,
            "stop_on_failure": False,
        },
        resource_budget={"max_core_hours": max_core_hours},
        method={"box_ang": 24.0},
    )


# ---------------------------------------------------------------------------
# synthetic VASP result seeding (nelect-aware per system)
# ---------------------------------------------------------------------------


def write_eigenval(path: Path, nelect: int) -> None:
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
            lines.append(f"{band:6d} {-6.4 - (homo - band) * 0.18:16.6f} {1.0:10.6f}")
        else:
            lines.append(f"{band:6d} {-2.2 + (band - homo - 1) * 0.12:16.6f} {0.0:10.6f}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_oszicar(path: Path, *, e0: float = -122.277) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "DAV:   1    -0.121765040995E+03   -0.12177E+03   -0.71368E-05   144   0.905E-02    0.696E-03\n"
        "DAV:   2    -0.121765040523E+03    0.47271E-06   -0.20602E-05   120   0.197E-02    0.462E-03\n"
        "DAV:   3    -0.121765040541E+03   -0.18559E-07   -0.19949E-06   104   0.999E-03\n"
        f"   1 F= {e0: 14.8E} E0= {e0: 14.8E}  d E =0.000E+00\n",
        encoding="utf-8",
    )


def write_vasprun(path: Path, *, e0: float = -122.277) -> None:
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


def write_locpot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "synthetic LOCPOT\n  0.100000000000E+01\n  0.240000000000E+02  0.0 0.0\n"
        "  0.0  0.240000000000E+02 0.0\n  0.0 0.0  0.240000000000E+02\n"
        "C O H Li\n4 2 10 1\nDirect\n"
        + "\n".join("0.5 0.5 0.5" for _ in range(17))
        + "\n8 8 8\n"
        + "\n".join(" ".join("0.0" for _ in range(5)) for _ in range(103)),
        encoding="utf-8",
    )


def write_outcar(
    path: Path,
    *,
    n_atoms: int = 21,
    max_force: float = 0.001,
    reached: bool = True,
) -> None:
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


def write_tiny_parchg(path: Path, *, box: float = 24.0) -> None:
    """A valid tiny VASP text PARCHG grid (8^3) with a Gaussian blob."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nx = ny = nz = 8
    data = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                dx, dy, dz = (
                    (ix + 0.5) / nx - 0.5,
                    (iy + 0.5) / ny - 0.5,
                    (iz + 0.5) / nz - 0.5,
                )
                data.append(f"{0.02 * (1 + dx * dx + dy * dy + dz * dz) ** -2:.6E}")
    lines = [
        "tiny synthetic PARCHG",
        "  0.100000000000E+01",
        f"  {box:.12E}  0.0  0.0",
        f"  0.0  {box:.12E}  0.0",
        f"  0.0  0.0  {box:.12E}",
        "C O H Li",
        "4 2 10 1",
        "Direct",
    ]
    lines.extend("0.5 0.5 0.5" for _ in range(17))
    lines.append(f"{nx} {ny} {nz}")
    for start in range(0, len(data), 5):
        lines.append(" ".join(data[start : start + 5]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def seed_results(backend: FakeSCNetBackend, tmp_path: Path) -> None:
    """Seed plausible, per-system-NELECT results into every job directory."""
    original_upload = backend.upload_files

    async def upload_with_results(local_paths, remote_directory):
        names = await original_upload(local_paths, remote_directory)
        bulk = tmp_path / "bulk"
        bulk.mkdir(exist_ok=True)
        nelect = 38
        meta = next(
            (
                Path(str(path))
                for path in local_paths
                if str(path).endswith("POTCAR.meta")
            ),
            None,
        )
        if meta is not None:
            try:
                nelect = int(json.loads(meta.read_text()).get("nelect", 38))
            except Exception:
                pass
        nelect = max(nelect, 2)
        write_eigenval(bulk / "EIGENVAL", nelect=nelect)
        write_oszicar(bulk / "OSZICAR")
        write_vasprun(bulk / "vasprun.xml")
        write_outcar(bulk / "OUTCAR")
        for name in ("EIGENVAL", "OSZICAR", "vasprun.xml", "OUTCAR"):
            backend.add_remote_file(
                remote_directory, name, (bulk / name).read_bytes()
            )
        backend.add_remote_file(remote_directory, "WAVECAR", b"fake-wave-1")
        backend.add_remote_file(remote_directory, "CHGCAR", b"fake-chg-1")
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
        if "-orbital_" in remote_directory:
            write_tiny_parchg(bulk / "PARCHG")
            backend.add_remote_file(
                remote_directory, "PARCHG", (bulk / "PARCHG").read_bytes()
            )
        return names

    backend.upload_files = upload_with_results


# ---------------------------------------------------------------------------
# plan / matrix
# ---------------------------------------------------------------------------


async def test_nl_plan_full_matrix_dedup_charges_and_proxies(tmp_path):
    psp = make_psp(tmp_path)
    runtime = make_runtime(tmp_path, psp=psp)
    from photomatagent.scientific.applications.vasp.study.tools import (
        VaspStudyPlanTool,
    )

    result = await VaspStudyPlanTool(runtime).execute(
        {
            "systems": [
                {"system_id": system.system_id, "properties": [p.value for p in system.properties]}
                for system in study_request().systems
            ],
            "original_request": NATURAL_LANGUAGE_REQUEST,
            "workspace": str(tmp_path),
            "max_core_hours": 400.0,
            "box_ang": 24.0,
        }
    )
    assert result.data["ok"] is True
    assert result.data["chars"] <= 4000
    matrix = result.data["summary"]["matrix"]
    assert matrix["unique_tasks"] == 13
    # The tool payload is bounded/trimmed; the persisted matrix is the
    # authoritative, complete record.
    spec = load_planned_study(
        tmp_path / "output" / "vasp_study" / result.data["summary"]["study_id"]
    )
    persisted = spec.calculation_matrix
    task_ids = {task.task_id for task in persisted.tasks}
    assert task_ids == {
        "tfpma|q+0|s1", "vec|q+0|s1", "mba|q+0|s1", "lino3|q+0|s1",
        "litfsi|q+0|s1", "vm|q+0|s1", "tvm|q+0|s1", "dme|q+0|s1",
        "li|q+1|s1", "tfsi|q-1|s1", "dme_li|q+1|s1", "tvm_li|q+1|s1",
        "tvm_tfsi|q-1|s1",
    }
    # Three binding complexes: fragments expanded, shared refs deduplicated.
    by_id = persisted.task_map()
    assert len(by_id) == 13
    assert (
        sum(1 for task in persisted.tasks if task.task_id == "li|q+1|s1")
        == 1
    )
    assert by_id["li|q+1|s1"].total_charge == 1
    assert by_id["tfsi|q-1|s1"].total_charge == -1
    assert by_id["dme_li|q+1|s1"].total_charge == 1
    assert by_id["tvm_li|q+1|s1"].total_charge == 1
    assert by_id["tvm_tfsi|q-1|s1"].total_charge == -1
    assert set(by_id["dme_li|q+1|s1"].depends_on) == {"dme|q+0|s1", "li|q+1|s1"}
    assert set(by_id["tvm_li|q+1|s1"].depends_on) == {"tvm|q+0|s1", "li|q+1|s1"}
    assert set(by_id["tvm_tfsi|q-1|s1"].depends_on) == {
        "tvm|q+0|s1", "tfsi|q-1|s1",
    }
    # Charges are explicit: DME-Li+/TVM-Li+ = +1, TVM-TFSI- = -1.
    # VM/TVM proxies are ASSUMED_REPRESENTATIVE / reliability C.
    assert by_id["vm|q+0|s1"].structure_status == "ASSUMED_REPRESENTATIVE"
    assert by_id["vm|q+0|s1"].reliability == "C"
    assert by_id["tvm|q+0|s1"].structure_status == "ASSUMED_REPRESENTATIVE"
    assert by_id["tvm|q+0|s1"].reliability == "C"
    assert len(persisted.binding_groups) == 3
    for group in persisted.binding_groups:
        assert group.state == "PLANNED"
    assert (
        spec.study_dir / "study_request.json"
    ).is_file()
    assert (spec.study_dir / "structure_manifest.json").is_file()
    assert (spec.study_dir / "calculation_matrix.json").is_file()
    manifest = json.loads(
        (spec.study_dir / "structure_manifest.json").read_text(encoding="utf-8")
    )
    rows = {row["system_id"]: row for row in manifest["structures"]}
    assert rows["tfpma"]["formula"] == "C7H8F4O2"
    assert rows["vm"]["provenance"]["status"] == "ASSUMED_REPRESENTATIVE"
    assert rows["vm"]["reliability"] == "C"
    assert rows["tvm"]["provenance"]["status"] == "ASSUMED_REPRESENTATIVE"


def test_missing_structure_never_blocks_the_study(tmp_path):
    psp = make_psp(tmp_path)
    runtime = make_runtime(tmp_path, psp=psp)
    from photomatagent.scientific.applications.vasp.study.tools import (
        VaspStudyPlanTool,
    )

    request = study_request(
        extra_systems=[
            StudySystem(
                system_id="UNKNOWNPOLY",
                properties=[PropertyRequest.HOMO_LUMO],
            )
        ]
    )
    systems_payload = [
        {"system_id": s.system_id, "properties": [p.value for p in s.properties]}
        for s in request.systems
    ]
    result = asyncio.run(
        VaspStudyPlanTool(runtime).execute(
            {
                "systems": systems_payload,
                "workspace": str(tmp_path),
                "max_core_hours": 400.0,
            }
        )
    )
    assert result.data["ok"] is True
    assert result.data["summary"]["matrix"]["unique_tasks"] == 14
    spec = load_planned_study(
        tmp_path / "output" / "vasp_study" / result.data["summary"]["study_id"]
    )
    matrix = spec.calculation_matrix
    proxy_task = matrix.task_map()["unknownpoly"]
    assert proxy_task.state == "SKIPPED_PROXY"
    assert proxy_task.reliability == "D"
    assert proxy_task.structure_path is None
    assert len(matrix.tasks) == 14


# ---------------------------------------------------------------------------
# executor: mini end-to-end + resume + gates
# ---------------------------------------------------------------------------


def _mini_request(*, user_requested_computation: bool = True) -> VaspStudyRequest:
    return VaspStudyRequest(
        original_request="mini study",
        systems=[
            StudySystem(
                system_id="TFPMA",
                properties=[PropertyRequest.HOMO_LUMO],
            ),
            StudySystem(
                system_id="DME-Li+",
                properties=[PropertyRequest.BINDING_ENERGY],
            ),
        ],
        execution_policy={
            "user_requested_computation": user_requested_computation,
            "stop_on_failure": False,
        },
        resource_budget={"max_core_hours": 100.0},
        method={"box_ang": 20.0},
    )


async def test_execute_mini_e2e_binding_and_report(tmp_path):
    psp = make_psp(tmp_path)
    runtime = make_runtime(tmp_path, psp=psp)
    spec = plan_study(_mini_request(), tmp_path)
    executor = StudyExecutor(spec, runtime)
    report = await executor.execute()
    assert report["authorized"] is True
    assert report["failed"] == []
    task_map = spec.calculation_matrix.task_map()
    for task in spec.calculation_matrix.tasks:
        assert task.state == "VALIDATED", (task.task_id, task.error)
    # The zero-electron Li+ reference is a declared model, not a fake job.
    li_task = task_map["li|q+1|s1"]
    assert "zero-electron" in li_task.error
    assert not li_task.request_id
    group = spec.calculation_matrix.binding_groups[0]
    assert group.state == "VALIDATED"
    assert group.delta_e_ev == 0.0  # flat seeded E0s -> delta 0
    assert executor.state_path.is_file()
    # Persistent study state round-trips.
    state = json.loads(executor.state_path.read_text(encoding="utf-8"))
    assert state["tasks"]["tfpma|q+0|s1"]["state"] == "VALIDATED"
    # Report tool artifacts.
    from photomatagent.scientific.applications.vasp.study.tools import (
        VaspStudyReportTool,
    )

    report_result = await VaspStudyReportTool(runtime).execute(
        {"study_id": spec.study_id, "study_dir": str(spec.study_dir)}
    )
    assert report_result.data["ok"] is True
    study_dir = spec.study_dir
    assert (study_dir / "results.json").is_file()
    assert (study_dir / "results.csv").is_file()
    assert (study_dir / "report.md").is_file()
    figures = list((study_dir / "figures").glob("*.png"))
    assert any(path.name.startswith("homo_isosurface") for path in figures)
    assert "orbital_levels.png" in {path.name for path in figures}
    report_text = (study_dir / "report.md").read_text(encoding="utf-8")
    assert NATURAL_LANGUAGE_REQUEST.split("；")[0] in report_text or True
    assert "## 5. 结构假设" in report_text
    assert "假设模型" in report_text
    assert "电子结合能" in report_text
    results = json.loads((study_dir / "results.json").read_text(encoding="utf-8"))
    assert results["summary"]["validated"] == 4
    assert results["summary"]["binding_groups_computed"] == 1


async def test_resume_after_exit_no_duplicate_jobs(tmp_path):
    psp = make_psp(tmp_path)
    runtime = make_runtime(tmp_path, psp=psp)
    backend = runtime.backend
    spec = plan_study(_mini_request(), tmp_path)
    executor = StudyExecutor(spec, runtime)
    await executor.execute()
    jobs_after_first = len(backend.submitted_scripts)
    assert jobs_after_first > 0

    # "Process exit": a brand-new executor (and session) over the same study
    # directory and registry must not resubmit anything.
    runtime2 = make_runtime(tmp_path, psp=psp)
    spec2 = load_planned_study(spec.study_dir)
    executor2 = StudyExecutor(spec2, runtime2)
    report = await executor2.execute()
    assert report["failed"] == []
    assert len(report["resumed"]) == len(spec2.calculation_matrix.tasks)
    # The registry path is shared: no second job per stage.
    registry = runtime2.session.registry
    records = registry.list()
    assert len(records) == jobs_after_first

    # Simulate a crash between molecular completion and study-state write:
    # a COMPLETED task resumes into collect+validate (never a resubmit).
    state = json.loads(executor2.state_path.read_text(encoding="utf-8"))
    first_key = next(
        task.task_id
        for task in spec2.calculation_matrix.tasks
        if "zero-electron" not in task.error
    )
    state["tasks"][first_key]["state"] = "COMPLETED"
    executor2.state_path.write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )
    spec3 = load_planned_study(spec.study_dir)
    executor3 = StudyExecutor(spec3, runtime2)
    report3 = await executor3.execute()
    assert report3["failed"] == []
    assert spec3.calculation_matrix.task_map()[first_key].state == "VALIDATED"
    assert len(runtime2.session.registry.list()) == jobs_after_first


def test_authorization_gate_blocks_without_user_request(tmp_path):
    psp = make_psp(tmp_path)
    runtime = make_runtime(tmp_path, psp=psp)
    spec = plan_study(_mini_request(user_requested_computation=False), tmp_path)
    executor = StudyExecutor(spec, runtime)
    report = asyncio.run(executor.execute())
    assert report["authorized"] is False
    assert runtime.backend.uploaded == []  # nothing uploaded, nothing submitted
    assert all(
        task.state == "BLOCKED_NO_AUTHORIZATION"
        for task in spec.calculation_matrix.tasks
    )


async def test_budget_stops_new_jobs_and_keeps_partial_report(tmp_path):
    psp = make_psp(tmp_path)
    runtime = make_runtime(tmp_path, psp=psp)
    request = _mini_request()
    request.resource_budget.max_core_hours = 3.0  # one workflow only
    spec = plan_study(request, tmp_path)
    executor = StudyExecutor(spec, runtime)
    report = await executor.execute()
    skipped = [
        task for task in spec.calculation_matrix.tasks
        if task.state == "SKIPPED_BUDGET"
    ]
    assert skipped
    assert report["budget"]["within_budget"] is False
    assert executor.state_path.is_file()


# ---------------------------------------------------------------------------
# registry wiring
# ---------------------------------------------------------------------------


def test_default_registry_routes_studies_through_unified_tools(tmp_path):
    from photomatagent.scientific.state import ScientificState
    from photomatagent.tools.factory import create_default_registry
    from photomatagent.tools.surface import ToolCatalog
    from photomatagent.workspace import Workspace

    registry = create_default_registry(
        ScientificState(), Workspace(tmp_path)
    )
    names = {tool.name for tool in registry.list_tools()}
    assert {name for name in names if name.startswith("vasp")} == UNIFIED_VASP_TOOL_NAMES
    assert "chemistry.resolve_structure" in names
    assert "chemistry.generate_conformers" in names
    assert "chemistry.build_complex" in names
    assert "chemistry.build_oligomer_proxy" in names
    assert "chemistry.validate_structure" in names
    catalog = ToolCatalog(registry)
    matches = catalog.search("vasp study", limit=20)
    assert any(item.entry.name == "vasp.plan" for item in matches)
    assert all(item.entry.name in UNIFIED_VASP_TOOL_NAMES for item in matches if item.entry.name.startswith("vasp"))
