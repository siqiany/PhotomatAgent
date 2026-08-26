"""End-to-end wiring tests for the registered ``vasp_molecule.*`` tools.

These prove the Phase-4 delivery contract: the tools are discoverable and
describable through the default registry (tool_search / tool_describe), they
can be invoked offline through the tool surface (prepare/preflight), and the
strict submission path rejects the historical false-positive (submit with no
run.slurm, no vasp_std launcher or no POTCAR strategy). No real SSH, Slurm or
VASP is ever touched; POTCAR fixtures are synthetic header lines only.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from photomatagent.scientific.applications.vasp.molecular.models import (
    MoleculeSpec,
    StageName,
)
from photomatagent.scientific.applications.vasp.molecular.runtime import (
    MolecularVaspRuntime,
)
from photomatagent.scientific.applications.vasp.molecular.tool_pack import (
    MolecularVaspCapabilitiesTool,
    MolecularVaspCollectTool,
    MolecularVaspPrepareTool,
    MolecularVaspPreflightTool,
    MolecularVaspResumeWorkflowTool,
    MolecularVaspSubmitTool,
)
from photomatagent.scientific.applications.vasp.molecular.workflow import (
    load_task_state,
)
from photomatagent.scientific.remote.fake import FakeSCNetBackend
from photomatagent.scientific.remote.lifecycle import SubmissionGate
from photomatagent.scientific.remote.models import (
    ResourcePolicy,
    ResourceRequest,
)
from photomatagent.scientific.remote.registry import JobLifecycleState
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.exposure import ToolExposure
from photomatagent.tools.factory import create_default_registry
from photomatagent.tools.surface import ToolCatalog
from photomatagent.workspace import Workspace


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


def make_psp(tmp_path: Path) -> Path:
    library = tmp_path / "psp"
    for element in ZVAL_ENMAX:
        write_dataset(library, element)
    return library


def dme_li_symbols() -> list[str]:
    return ["C"] * 4 + ["O"] * 2 + ["H"] * 10 + ["Li"]


def dme_li_molecule(tmp_path: Path, *, charge: int = 1, box: float = 20.0) -> MoleculeSpec:
    symbols = dme_li_symbols()
    points = np.asarray(
        [(i % 4, (i // 4) % 2, (i // 8) % 3) for i in range(len(symbols))],
        dtype=float,
    )
    points -= points.mean(axis=0)
    points += box / 2
    path = tmp_path / "dme_li.xyz"
    lines = [str(len(symbols)), "DME-Li+ q=+1"]
    lines += [
        f"{symbol} {x:.5f} {y:.5f} {z:.5f}"
        for symbol, (x, y, z) in zip(symbols, points, strict=True)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return MoleculeSpec(
        name="DME_Li",
        structure_path=path,
        total_charge=charge,
        spin_multiplicity=1,
        box_ang=box,
        calculation_purpose="binding",
    )


def make_runtime(tmp_path: Path, *, psp: Path, strict: bool = True) -> MolecularVaspRuntime:
    return MolecularVaspRuntime(
        backend=FakeSCNetBackend(
            policy=ResourcePolicy(allow_hpc_submit=strict),
            strict=strict,
        ),
        configured=True,
        psp_dir=psp,
        workflow_dir=tmp_path / "mol",
        log_dir=tmp_path / "logs",
        registry_path=tmp_path / "state" / "jobs.sqlite3",
        module_name="vasp-5.4.4",
        env_script="",
        remote_psp_dir="~/photomatagent/psp",
    )


def _resource() -> ResourceRequest:
    return ResourceRequest(
        partition="kshcnormal", nodes=1, tasks_per_node=8, walltime_minutes=20
    )


def write_minimal_eigenval(path: Path, *, nelect: int = 38) -> None:
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


def _write_minimal_vasprun(path: Path, *, e0: float = -122.277) -> None:
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


def seed_results(
    backend: FakeSCNetBackend, tmp_path: Path, *, converged: bool = True
) -> None:
    """Seed plausible results into every job directory created afterwards."""
    original_upload = backend.upload_files

    async def upload_with_results(local_paths, remote_directory):
        names = await original_upload(local_paths, remote_directory)
        bulk = tmp_path / "seeds"
        bulk.mkdir(exist_ok=True)
        write_minimal_eigenval(bulk / "EIGENVAL", nelect=38)
        last_de = "0.47271E-06" if converged else "0.123E-03"
        last_de_3 = "-0.18559E-07" if converged else "0.123E-03"
        (bulk / "OSZICAR").write_text(
            "DAV:   1    -0.121765040995E+03   -0.12177E+03   "
            "-0.71368E-05   144   0.905E-02    0.696E-03\n"
            f"DAV:   2    -0.121765040523E+03    {last_de}   "
            "-0.20602E-05   120   0.197E-02    0.462E-03\n"
            f"DAV:   3    -0.121765040541E+03    {last_de_3}   "
            "-0.19949E-06   104   0.999E-03\n"
            "   1 F= -0.122277000000E+03 E0= -0.122277000000E+03  "
            "d E =0.000E+00\n",
            encoding="utf-8",
        )
        _write_minimal_vasprun(bulk / "vasprun.xml", e0=-122.277)
        # Relax validation is OUTCAR-grounded (max force vs EDIFFG); seed a
        # force-converged OUTCAR so the synthetic results satisfy the same
        # scientific contract as real VASP outputs.
        write_outcar(bulk / "OUTCAR", reached=converged)
        for name in ("EIGENVAL", "OSZICAR", "vasprun.xml", "OUTCAR"):
            backend.add_remote_file(
                remote_directory, name, (bulk / name).read_bytes()
            )
        # Large restart artifacts are staged between remote dirs on SCNet and
        # never downloaded; fake placeholders prove the staging path works.
        backend.add_remote_file(
            remote_directory, "WAVECAR", b"fake-wave-epoch-1"
        )
        backend.add_remote_file(
            remote_directory, "CHGCAR", b"fake-chg-epoch-1"
        )
        if "-relax-" in remote_directory:
            source = next(
                (
                    Path(str(p))
                    for p in local_paths
                    if str(p).endswith("POSCAR")
                ),
                None,
            )
            backend.add_remote_file(
                remote_directory,
                "CONTCAR",
                source.read_bytes() if source is not None else b"",
            )
        return names

    backend.upload_files = upload_with_results


def write_outcar(
    path: Path, *, n_atoms: int = 17, reached: bool = True
) -> None:
    """Synthetic force-converged OUTCAR (relax validation fixture)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        " vasp.5.4.4.18Apr17 (build Mar 03 2024 15:47:24) complex parallel",
        "   NSW      =      200     number of steps for ionic motion",
        "   IBRION   =      2     ionic relax: 1=quasi-Newton, 2=damped",
        "   EDIFFG   = -0.02E+00  force-criterion for ionic relax",
    ]
    if reached:
        lines.append(
            "  reached required accuracy - stopping structural energy minimisation"
        )
    lines.append("POSITION                                       TOTAL-FORCE (eV/Angst)")
    lines.append("-" * 90)
    for index in range(n_atoms):
        lines.append(
            f"{index + 1:6d} {0.0:17.10f} {0.0:17.10f} {0.0:17.10f}"
            f"{0.00014:14.8f} {0.00014:14.8f} {0.00014:14.8f}"
        )
    lines.append("-" * 90)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# registry discovery + description
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registry(tmp_path_factory) -> Any:
    root = tmp_path_factory.mktemp("wire-registry")
    return create_default_registry(ScientificState(), Workspace(root))


def test_default_registry_registers_all_molecular_tools(registry):
    tool_names = {tool.name for tool in registry.list_tools()}
    assert {
        "vasp_molecule.capabilities",
        "vasp_molecule.prepare",
        "vasp_molecule.preflight",
        "vasp_molecule.submit",
        "vasp_molecule.status",
        "vasp_molecule.collect",
        "vasp_molecule.analyze_orbitals",
        "vasp_molecule.analyze_esp",
        "vasp_molecule.binding_energy",
        "vasp_molecule.resume_workflow",
    } <= tool_names


def test_tool_search_finds_molecular_vasp(registry):
    catalog = ToolCatalog(registry)
    matches = catalog.search("isolated molecule VASP", limit=20)
    names = {match.entry.name for match in matches}
    assert "vasp_molecule.prepare" in names
    assert "vasp_molecule.preflight" in names
    assert "vasp_molecule.submit" in names


def test_tool_describe_returns_schema(registry):
    catalog = ToolCatalog(registry)
    entry = catalog.get("vasp_molecule.prepare")
    assert entry is not None
    required = entry.required_parameters
    assert "structure_path" in required
    assert "total_charge" in required
    schema = entry.full_schema_reference.input_schema
    assert schema["properties"]["total_charge"]["type"] == "integer"
    assert entry.namespace == "vasp_molecule"


def test_all_molecular_tools_are_deferred(registry):
    for tool in registry.list_tools():
        if tool.name.startswith("vasp_molecule."):
            assert tool.exposure is ToolExposure.DEFERRED


# --------------------------------------------------------------------------
# tool-surface offline prepare / preflight
# --------------------------------------------------------------------------


async def test_tool_bridge_prepare_and_preflight(tmp_path):
    psp = make_psp(tmp_path)
    runtime = make_runtime(tmp_path, psp=psp)
    molecule = dme_li_molecule(tmp_path)
    prepare = MolecularVaspPrepareTool(runtime)
    result = await prepare.execute(
        {
            "structure_path": str(molecule.structure_path),
            "name": "DME_Li",
            "total_charge": 1,
            "spin_multiplicity": 1,
            "box_ang": 20.0,
            "calculation_purpose": "binding",
        }
    )
    assert result.data["ok"] is True
    assert result.data["summary"]["preflight_passed"] is True
    assert (runtime.workflow_dir / "workflow.json").is_file()
    assert (runtime.workflow_dir / "preflight.json").is_file()
    assert len(result.output) <= 4000

    preflight = MolecularVaspPreflightTool(runtime)
    pres = await preflight.execute(
        {"workflow_dir": str(runtime.workflow_dir)}
    )
    assert pres.data["ok"] is True
    # DME-Li+ : neutral valence 39 (C4/O2/H10/Li) minus q=+1 -> 38 NELECT.
    assert pres.data["summary"]["nelect"] == 38
    assert len(pres.output) <= 4000


async def test_tool_capabilities_bounded(tmp_path):
    psp = make_psp(tmp_path)
    runtime = make_runtime(tmp_path, psp=psp)
    caps = MolecularVaspCapabilitiesTool(runtime)
    result = await caps.execute({})
    assert result.data["configured"] is True
    assert result.data["remote_psp_dir_configured"] is True
    assert ".ssh" not in result.output  # no private-path leakage
    assert len(result.output) <= 4000


# --------------------------------------------------------------------------
# strict submission path: historical false-positives must fail
# --------------------------------------------------------------------------


def _stage_inputs(tmp_path: Path, name: str) -> Path:
    directory = tmp_path / name
    directory.mkdir(exist_ok=True)
    for filename, content in {
        "POSCAR": "C O H Li\n4 2 10 1\nDirect\n",
        "INCAR": "ENCUT = 400\nISPIN = 1\nNELECT = 38\n",
        "KPOINTS": "Gamma\n0\nGamma\n1 1 1\n0 0 0\n",
    }.items():
        (directory / filename).write_text(content, encoding="utf-8")
    return directory


async def test_strict_backend_rejects_submit_without_run_slurm(tmp_path):
    runtime = make_runtime(tmp_path, psp=make_psp(tmp_path))
    session = runtime.session
    input_dir = _stage_inputs(tmp_path, "stage1")
    # No script_renderer -> no run.slurm -> strict backend must refuse.
    result = await session.submit_once(
        application="vasp_molecular",
        workflow_stage="relax",
        job_name="dme-relax",
        local_input_dir=input_dir,
        gate=SubmissionGate(passed=True),
        resource=_resource(),
        executable="vasp_std",
        script_name="run.slurm",
        potcar_mode="remote",
        potcar_symbols=["C", "O", "H", "Li"],
        remote_psp_dir="~/photomatagent/psp",
    )
    assert result.submitted is False
    assert "run.slurm" in result.error


async def test_strict_backend_rejects_script_without_vasp_std(tmp_path):
    runtime = make_runtime(tmp_path, psp=make_psp(tmp_path))
    session = runtime.session
    input_dir = _stage_inputs(tmp_path, "stage2")
    result = await session.submit_once(
        application="vasp_molecular",
        workflow_stage="relax",
        job_name="dme-relax",
        local_input_dir=input_dir,
        gate=SubmissionGate(passed=True),
        resource=_resource(),
        executable="vasp_std",
        script_name="run.slurm",
        script_renderer=lambda job_name, resource: "#!/bin/bash\necho hi\n",
        potcar_mode="remote",
        potcar_symbols=["C", "O", "H", "Li"],
        remote_psp_dir="~/photomatagent/psp",
    )
    assert result.submitted is False
    assert "vasp_std" in result.error


async def test_strict_backend_rejects_missing_local_potcar(tmp_path):
    runtime = make_runtime(tmp_path, psp=make_psp(tmp_path))
    session = runtime.session
    input_dir = _stage_inputs(tmp_path, "stage3")
    result = await session.submit_once(
        application="vasp_molecular",
        workflow_stage="relax",
        job_name="dme-relax",
        local_input_dir=input_dir,
        gate=SubmissionGate(passed=True),
        resource=_resource(),
        executable="vasp_std",
        script_name="run.slurm",
        script_renderer=lambda job_name, resource: (
            "#!/bin/bash\nsrun --mpi=pmi2 vasp_std\n"
        ),
        potcar_mode="local",
        potcar_symbols=[],
        remote_psp_dir="",
    )
    assert result.submitted is False
    assert "POTCAR" in result.error


async def test_strict_backend_rejects_remote_without_assembly_preamble(tmp_path):
    runtime = make_runtime(tmp_path, psp=make_psp(tmp_path))
    session = runtime.session
    input_dir = _stage_inputs(tmp_path, "stage4")
    result = await session.submit_once(
        application="vasp_molecular",
        workflow_stage="relax",
        job_name="dme-relax",
        local_input_dir=input_dir,
        gate=SubmissionGate(passed=True),
        resource=_resource(),
        executable="vasp_std",
        script_name="run.slurm",
        script_renderer=lambda job_name, resource: (
            "#!/bin/bash\nsrun --mpi=pmi2 vasp_std\n"
        ),
        potcar_mode="remote",
        potcar_symbols=["C", "O", "H", "Li"],
        remote_psp_dir="~/photomatagent/psp",
    )
    assert result.submitted is False
    assert "POTCAR" in result.error or "assembly" in result.error


# --------------------------------------------------------------------------
# corrected tool submit -> run.slurm + POTCAR assembly
# --------------------------------------------------------------------------


async def test_tool_submit_generates_run_slurm_and_remote_assembly(tmp_path):
    psp = make_psp(tmp_path)
    runtime = make_runtime(tmp_path, psp=psp, strict=True)
    molecule = dme_li_molecule(tmp_path)
    prepare = MolecularVaspPrepareTool(runtime)
    assert (await prepare.execute({
        "structure_path": str(molecule.structure_path),
        "name": "DME_Li", "total_charge": 1, "box_ang": 20.0,
    })).data["ok"] is True
    submit = MolecularVaspSubmitTool(runtime)
    result = await submit.execute(
        {
            "workflow_dir": str(runtime.workflow_dir),
            "stage": StageName.RELAX.value,
            "wait": False,
        }
    )
    assert result.data["ok"] is True, result.data.get("errors")
    assert result.data["summary"]["job_id"] is not None
    remote_dir = result.data["summary"]["remote_directory"]
    remote = runtime.backend.remote_files[remote_dir]
    assert "run.slurm" in remote
    script = remote["run.slurm"].decode("utf-8")
    assert "vasp_std" in script
    assert "srun --mpi=pmi2" in script
    assert "psp_base=" in script  # remote POTCAR assembly preamble
    assert "POTCAR.meta" in remote
    assert "POTCAR" not in remote  # content never uploaded in remote mode
    # submit-once: a second call creates no second job
    second = await submit.execute(
        {
            "workflow_dir": str(runtime.workflow_dir),
            "stage": StageName.RELAX.value,
        }
    )
    assert second.data["ok"] is False
    assert len(runtime.backend.submitted_scripts) == 1


# --------------------------------------------------------------------------
# lifecycle: COMPLETED -> COLLECTED -> VALIDATED
# --------------------------------------------------------------------------


async def test_collect_advances_validated_in_registry_and_task_state(tmp_path):
    psp = make_psp(tmp_path)
    runtime = make_runtime(tmp_path, psp=psp, strict=True)
    seed_results(runtime.backend, tmp_path)
    molecule = dme_li_molecule(tmp_path)
    prepare = MolecularVaspPrepareTool(runtime)
    assert (await prepare.execute({
        "structure_path": str(molecule.structure_path),
        "name": "DME_Li", "total_charge": 1, "box_ang": 20.0,
    })).data["ok"] is True
    submit = MolecularVaspSubmitTool(runtime)
    submitted = await submit.execute(
        {
            "workflow_dir": str(runtime.workflow_dir),
            "stage": StageName.RELAX.value,
            "wait": True,
            "wait_timeout_seconds": 30,
        }
    )
    assert submitted.data["ok"] is True
    request_id = submitted.data["summary"]["request_id"]
    record = runtime.session.registry.get(request_id)
    assert record is not None
    assert record.state is JobLifecycleState.COMPLETED

    collect = MolecularVaspCollectTool(runtime)
    collected = await collect.execute(
        {
            "workflow_dir": str(runtime.workflow_dir),
            "stage": StageName.RELAX.value,
        }
    )
    assert collected.data["ok"] is True
    assert collected.data["evidence_count"] > 0
    after = runtime.session.registry.get(request_id)
    assert after is not None
    assert after.state is JobLifecycleState.VALIDATED
    assert after.scientific_validation_state == "passed"
    state = load_task_state(runtime.workflow_dir)
    assert state is not None
    relax = state.stage_map()[StageName.RELAX.value]
    assert relax.state == JobLifecycleState.VALIDATED.value
    assert relax.validated is True


async def test_resume_workflow_tool_resumes_completed_stages(tmp_path):
    psp = make_psp(tmp_path)
    runtime = make_runtime(tmp_path, psp=psp, strict=True)
    seed_results(runtime.backend, tmp_path)
    molecule = dme_li_molecule(tmp_path)
    prepare = MolecularVaspPrepareTool(runtime)
    assert (await prepare.execute({
        "structure_path": str(molecule.structure_path),
        "name": "DME_Li", "total_charge": 1, "box_ang": 20.0,
    })).data["ok"] is True
    resume = MolecularVaspResumeWorkflowTool(runtime)
    first = await resume.execute(
        {"workflow_dir": str(runtime.workflow_dir), "wait_timeout_seconds": 30}
    )
    assert first.data["ok"] is True, first.data.get("errors")
    jobs_first = len(runtime.backend.submitted_scripts)
    second = await resume.execute(
        {"workflow_dir": str(runtime.workflow_dir)}
    )
    assert second.data["ok"] is True
    assert set(second.data["summary"]["resumed"]) == set(
        second.data["summary"]["stages"]
    )
    assert len(runtime.backend.submitted_scripts) == jobs_first


async def test_collect_failed_validation_stays_collected_without_evidence(tmp_path):
    psp = make_psp(tmp_path)
    runtime = make_runtime(tmp_path, psp=psp, strict=True)
    seed_results(runtime.backend, tmp_path, converged=False)
    molecule = dme_li_molecule(tmp_path)
    prepare = MolecularVaspPrepareTool(runtime)
    assert (await prepare.execute({
        "structure_path": str(molecule.structure_path),
        "name": "DME_Li", "total_charge": 1, "box_ang": 20.0,
    })).data["ok"] is True
    submit = MolecularVaspSubmitTool(runtime)
    submitted = await submit.execute(
        {
            "workflow_dir": str(runtime.workflow_dir),
            "stage": StageName.RELAX.value,
            "wait": True,
            "wait_timeout_seconds": 30,
        }
    )
    assert submitted.data["ok"] is True
    request_id = submitted.data["summary"]["request_id"]
    collect = MolecularVaspCollectTool(runtime)
    collected = await collect.execute(
        {
            "workflow_dir": str(runtime.workflow_dir),
            "stage": StageName.RELAX.value,
        }
    )
    assert collected.data["ok"] is False
    assert collected.data["evidence_count"] == 0
    assert any("SCF_NOT_CONVERGED" in item for item in collected.data["errors"])
    after = runtime.session.registry.get(request_id)
    assert after is not None
    assert after.state is JobLifecycleState.COLLECTED  # never VALIDATED
    assert after.scientific_validation_state == "failed"
    state = load_task_state(runtime.workflow_dir)
    assert state is not None
    relax = state.stage_map()[StageName.RELAX.value]
    assert relax.state == JobLifecycleState.COLLECTED.value
    assert relax.validated is False


# --------------------------------------------------------------------------
# SCNet MCP boundary (direct function calls, no MCP handshake)
# --------------------------------------------------------------------------


async def test_scnet_mcp_boundary_prepare(tmp_path):
    psp = make_psp(tmp_path)
    runtime = make_runtime(tmp_path, psp=psp)
    molecule = dme_li_molecule(tmp_path)
    from photomatagent.mcp_servers.scnet import server

    tool = MolecularVaspPrepareTool(runtime)
    with patch.object(server, "_molecular_tool", return_value=tool):
        result = await server._call_molecular(
            "vasp_molecule.prepare",
            {
                "structure_path": str(molecule.structure_path),
                "name": "DME_Li",
                "total_charge": 1,
                "box_ang": 20.0,
                "workflow_dir": str(tmp_path / "mcpflow"),
            },
        )
    assert result.get("is_error") is False
    assert result["summary"]["preflight_passed"] is True
