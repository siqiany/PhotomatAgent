"""Offline VASP application tests: inputs, POTCAR policy, validation, fake-backend flow."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from photomatagent.scientific.applications.vasp.application import VaspApplication
from photomatagent.scientific.applications.vasp.inputs import _kpoint_grid_density
from photomatagent.scientific.applications.vasp.validation import (
    check_vasprun,
    parse_result,
    validate_output,
)
from photomatagent.scientific.remote.fake import FakeSCNetBackend
from photomatagent.scientific.remote.models import ResourcePolicy
from photomatagent.scientific.remote.scnet import RemoteSubmissionBlocked


def write_vasprun(
    directory: Path,
    *,
    converged: bool = True,
    nsw: int = 100,
    ionic_steps: int = 1,
    with_dielectric: bool = True,
    invalid_xml: bool = False,
) -> Path:
    """Write a synthetic VASP 5.4.4-style vasprun.xml fixture."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "vasprun.xml"
    if invalid_xml:
        path.write_text("<modeling><broken", encoding="utf-8")
        return path
    scsteps = []
    for index in range(3):
        marker = "<c/><v/>" if converged else ""
        scsteps.append(
            f"      <scstep>\n"
            f"        <energy><i name=\"e_fr_energy\">{-12.3 - index * 1e-4}</i></energy>\n"
            f"        {marker}\n"
            f"      </scstep>"
        )
    ionic = "\n".join(
        f'    <i name="ionic step" type="int">{i}</i>'
        for i in range(1, ionic_steps + 1)
    )
    dielectric = ""
    if with_dielectric:
        dielectric = (
            "    <dielectricfunction>\n"
            "      <varray name=\"real\">\n"
            "        <v> 10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0 </v>\n"
            "        <v> 9.5 0.0 0.0 0.0 9.5 0.0 0.0 0.0 9.5 </v>\n"
            "      </varray>\n"
            "      <varray name=\"imag\">\n"
            "        <v> 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 </v>\n"
            "        <v> 0.8 0.0 0.0 0.0 0.8 0.0 0.0 0.0 0.8 </v>\n"
            "      </varray>\n"
            "    </dielectricfunction>\n"
        )
    path.write_text(
        "<modeling>\n"
        "  <generator><i name=\"program\" type=\"string\">vasp.6.4.2</i></generator>\n"
        f'  <incar><i name="NSW" type="int">{nsw}</i></incar>\n'
        "  <calculation>\n"
        + "\n".join(scsteps)
        + "\n"
        + ionic
        + "\n"
        + dielectric
        + "  </calculation>\n"
        "</modeling>\n",
        encoding="utf-8",
    )
    return path


def make_psp_dir(root: Path, *symbols: str) -> Path:
    psp = root / "potpaw_PBE.64"
    for symbol in symbols:
        directory = psp / symbol
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "POTCAR").write_text(f"POTCAR {symbol} (test fixture)\n", encoding="utf-8")
    return root


def make_application(*, psp: Path | None = None, allow: bool = True) -> VaspApplication:
    backend = FakeSCNetBackend(policy=ResourcePolicy(allow_hpc_submit=allow))
    return VaspApplication(
        backend, psp_dir=str(psp) if psp else None, jobs_local_dir="output/vasp_inputs"
    )


# -- input generation ---------------------------------------------------------


def test_prepare_workflow_generates_all_stages(tmp_path):
    app = make_application()
    structure = tmp_path / "inAs.cif"
    structure.write_text(
        """# InAs zincblende test fixture
data_InAs
_symmetry_space_group_name_H-M   'F -4 3 m'
_cell_length_a   6.0583
_cell_length_b   6.0583
_cell_length_c   6.0583
_cell_angle_alpha   90
_cell_angle_beta    90
_cell_angle_gamma   90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
In  In  0.0 0.0 0.0
As  As  0.25 0.25 0.25
""",
        encoding="utf-8",
    )
    out = tmp_path / "workflow"
    manifest = app.prepare_inputs(
        structure_path=str(structure),
        profile_name="standard_semiconductor",
        output_dir=out,
    )
    assert manifest["profile"] == "standard_semiconductor"
    stage_names = [stage["stage"] for stage in manifest["stages"]]
    assert stage_names == ["relax", "static", "band", "dos"]
    for stage in manifest["stages"]:
        stage_dir = Path(stage["directory"])
        for name in ("POSCAR", "INCAR", "KPOINTS", "POTCAR.policy"):
            assert (stage_dir / name).is_file(), f"{name} missing in {stage_dir}"
        assert not (stage_dir / "POTCAR").exists()  # never generated
    assert (out / "workflow.json").is_file()
    workflow = json.loads((out / "workflow.json").read_text(encoding="utf-8"))
    assert workflow["stages"][1]["depends_on"] == "relax"
    assert workflow["stages"][1]["required_outputs"] == ["CONTCAR"]


def test_kpoint_density_matches_kpoints_per_atom_scaling():
    grid = _kpoint_grid_density(
        5.431**3, 1000, [5.431, 5.431, 5.431], num_sites=2
    )
    assert grid == [7, 7, 7]
    assert max(grid) < 20


def test_narrow_gap_soc_incar_contains_soc_settings(tmp_path):
    app = make_application()
    structure = tmp_path / "hgTe.cif"
    structure.write_text(
        """data_HgTe
_cell_length_a   6.46
_cell_length_b   6.46
_cell_length_c   6.46
_cell_angle_alpha   90
_cell_angle_beta    90
_cell_angle_gamma   90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Hg  Hg  0.0 0.0 0.0
Te  Te  0.25 0.25 0.25
""",
        encoding="utf-8",
    )
    out = tmp_path / "soc"
    manifest = app.prepare_inputs(
        structure_path=str(structure),
        profile_name="narrow_gap_soc",
        output_dir=out,
    )
    assert manifest["soc"] is True
    assert manifest["executable"] == "vasp_ncl"
    incar = (Path(manifest["stages"][0]["directory"]) / "INCAR").read_text(
        encoding="utf-8"
    )
    assert "LSORBIT = .TRUE." in incar
    assert "LNONCOLLINEAR = .TRUE." in incar
    assert "GGA_COMPAT = .FALSE." in incar
    assert "ISYM = -1" in incar


def test_potcar_policy_resolution(tmp_path):
    psp = make_psp_dir(tmp_path / "psp", "In", "As")
    app = make_application(psp=psp)
    structure = tmp_path / "inAs.cif"
    structure.write_text(
        """data_InAs
_cell_length_a   6.0583
_cell_length_b   6.0583
_cell_length_c   6.0583
_cell_angle_alpha   90
_cell_angle_beta    90
_cell_angle_gamma   90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
In  In  0.0 0.0 0.0
As  As  0.25 0.25 0.25
""",
        encoding="utf-8",
    )
    out = tmp_path / "w"
    manifest = app.prepare_inputs(
        structure_path=str(structure),
        profile_name="standard_semiconductor",
        output_dir=out,
    )
    stage_dir = Path(manifest["stages"][0]["directory"])
    policy = (stage_dir / "POTCAR.policy").read_text(encoding="utf-8")
    assert "In: resolved" in policy
    assert "As: resolved" in policy
    potcar = app.resolve_potcar(stage_dir)
    assert potcar is not None and potcar.is_file()
    content = potcar.read_text(encoding="utf-8")
    assert "POTCAR In" in content and "POTCAR As" in content


def test_submit_refuses_without_potcar(tmp_path, monkeypatch):
    monkeypatch.delenv("PMG_VASP_PSP_DIR", raising=False)
    app = make_application(psp=None)
    structure = tmp_path / "inAs.cif"
    structure.write_text(
        """data_InAs
_cell_length_a   6.0583
_cell_length_b   6.0583
_cell_length_c   6.0583
_cell_angle_alpha   90
_cell_angle_beta    90
_cell_angle_gamma   90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
In  In  0.0 0.0 0.0
As  As  0.25 0.25 0.25
""",
        encoding="utf-8",
    )
    out = tmp_path / "w"
    manifest = app.prepare_inputs(
        structure_path=str(structure),
        profile_name="standard_semiconductor",
        output_dir=out,
    )

    async def scenario():
        with pytest.raises(ValueError, match="POTCAR"):
            await app.submit_stage(
                job_name="no-psp",
                input_dir=manifest["stages"][0]["directory"],
                profile_name="standard_semiconductor",
            )

    asyncio.run(scenario())


def test_submit_rejects_crlf_incar_and_incompatible_ncore(tmp_path):
    from photomatagent.scientific.remote.models import ResourceRequest

    incar = tmp_path / "INCAR"
    incar.write_bytes(b"ENCUT = 520\r\n")
    request = ResourceRequest(
        partition="p", nodes=1, tasks_per_node=32, walltime_minutes=10
    )
    with pytest.raises(ValueError, match="dos2unix"):
        VaspApplication._validate_incar_for_resource(incar, request)
    incar.write_text("ENCUT = 520\nNCORE = 5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must divide"):
        VaspApplication._validate_incar_for_resource(incar, request)
    incar.write_text("ENCUT = 520\nNCORE = 8\n", encoding="utf-8")
    VaspApplication._validate_incar_for_resource(incar, request)


def test_render_slurm_can_source_scnet_product_environment():
    from photomatagent.scientific.applications.vasp.profiles import get_profile
    from photomatagent.scientific.remote.models import ResourceRequest

    app = VaspApplication(
        module_name="broken-module",
        env_script="/public/home/u/apprepo/vasp/v/scripts/env.sh",
    )
    script = app.render_slurm(
        job_name="smoke",
        profile=get_profile("standard_semiconductor"),
        resource=ResourceRequest(
            partition="p", nodes=1, tasks_per_node=2, walltime_minutes=10
        ),
    )
    assert "module load broken-module" not in script
    assert "source /public/home/u/apprepo/vasp/v/scripts/env.sh" in script
    assert "command -v vasp_std >/dev/null" in script


# -- validation ---------------------------------------------------------------


def test_check_vasprun_valid_fixture(tmp_path):
    path = write_vasprun(tmp_path)
    check = check_vasprun(path)
    assert check.exists and check.well_formed_xml and check.has_scf_blocks
    assert check.electronic_converged is True
    assert check.ionic_converged is True
    assert check.reasons == []


def test_check_vasprun_missing_file(tmp_path):
    check = check_vasprun(tmp_path / "vasprun.xml")
    assert check.exists is False
    assert any("missing or empty" in reason for reason in check.reasons)


def test_check_vasprun_invalid_xml(tmp_path):
    path = write_vasprun(tmp_path, invalid_xml=True)
    check = check_vasprun(path)
    assert check.well_formed_xml is False
    assert any("not well-formed" in reason for reason in check.reasons)


def test_check_vasprun_unconverged_marker(tmp_path):
    path = write_vasprun(tmp_path, converged=False)
    check = check_vasprun(path)
    assert check.electronic_converged is None
    assert any("convergence marker" in reason for reason in check.reasons)


def test_check_vasprun_accepts_vasp54_outcar_ediff_marker(tmp_path):
    path = write_vasprun(tmp_path, converged=False)
    (tmp_path / "OUTCAR").write_text(
        "aborting loop because EDIFF is reached\n"
        "reached required accuracy - stopping structural energy minimisation\n",
        encoding="utf-8",
    )
    check = check_vasprun(path)
    assert check.electronic_converged is True
    assert check.ionic_converged is True
    assert not check.reasons


def test_validate_output_relax_requires_ionic_convergence(tmp_path):
    write_vasprun(tmp_path, nsw=1, ionic_steps=1)  # ran all NSW steps
    problems = validate_output(tmp_path, profile_name="relax")
    assert any("ionic convergence" in problem for problem in problems)


def test_validate_output_band_requires_chgcar(tmp_path):
    write_vasprun(tmp_path)
    problems = validate_output(tmp_path, profile_name="band")
    assert any("CHGCAR" in problem for problem in problems)
    (tmp_path / "CHGCAR").write_text("fake chgcar", encoding="utf-8")
    assert validate_output(tmp_path, profile_name="band") == []


def test_parse_result_extracts_energy_and_dielectric(tmp_path):
    write_vasprun(tmp_path)
    parsed = parse_result(tmp_path)
    assert "final_energy_eV" in parsed
    assert parsed["dielectric"]["points"] == 2
    assert parsed["dielectric"]["imag_xx_max"] == pytest.approx(0.8)


# -- fake backend end-to-end --------------------------------------------------


def test_submit_status_collect_flow(tmp_path):
    psp = make_psp_dir(tmp_path / "psp", "In", "As")
    app = make_application(psp=psp)
    structure = tmp_path / "inAs.cif"
    structure.write_text(
        """data_InAs
_cell_length_a   6.0583
_cell_length_b   6.0583
_cell_length_c   6.0583
_cell_angle_alpha   90
_cell_angle_beta    90
_cell_angle_gamma   90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
In  In  0.0 0.0 0.0
As  As  0.25 0.25 0.25
""",
        encoding="utf-8",
    )
    out = tmp_path / "w"
    manifest = app.prepare_inputs(
        structure_path=str(structure),
        profile_name="standard_semiconductor",
        output_dir=out,
    )
    stage_dir = manifest["stages"][0]["directory"]

    async def scenario():
        ref = await app.submit_stage(
            job_name="inas-static",
            input_dir=stage_dir,
            profile_name="standard_semiconductor",
        )
        assert ref.job_id.isdigit()
        # fake backend stores what was uploaded
        uploaded = app.backend.remote_files.get(ref.remote_directory, {})
        for name in ("POSCAR", "INCAR", "KPOINTS", "POTCAR", "vasp.slurm"):
            assert name in uploaded
        assert await app.status(ref.job_id) is not None
        # pre-populate remote results, then collect
        write_vasprun(tmp_path / "remote_vasprun")
        vasprun = (tmp_path / "remote_vasprun" / "vasprun.xml").read_bytes()
        app.backend.add_remote_file(ref.remote_directory, "vasprun.xml", vasprun)
        app.backend.add_remote_file(ref.remote_directory, "OUTCAR", "reached required accuracy\n")
        report = await app.collect(
            job_ref=ref,
            local_dir=tmp_path / "results",
            profile_name="standard_semiconductor",
        )
        assert report["scientifically_valid"] is True
        assert report["parsed"]["final_energy_eV"] == pytest.approx(-12.3, abs=1e-3)

    asyncio.run(scenario())


def test_submit_blocked_without_authorization(tmp_path):
    psp = make_psp_dir(tmp_path / "psp", "In", "As")
    app = make_application(psp=psp, allow=False)
    structure = tmp_path / "inAs.cif"
    structure.write_text(
        """data_InAs
_cell_length_a   6.0583
_cell_length_b   6.0583
_cell_length_c   6.0583
_cell_angle_alpha   90
_cell_angle_beta    90
_cell_angle_gamma   90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
In  In  0.0 0.0 0.0
As  As  0.25 0.25 0.25
""",
        encoding="utf-8",
    )
    out = tmp_path / "w"
    manifest = app.prepare_inputs(
        structure_path=str(structure),
        profile_name="standard_semiconductor",
        output_dir=out,
    )

    async def scenario():
        with pytest.raises(RemoteSubmissionBlocked):
            await app.submit_stage(
                job_name="blocked",
                input_dir=manifest["stages"][0]["directory"],
                profile_name="standard_semiconductor",
            )

    asyncio.run(scenario())


def test_workflow_runner_all_stages_valid(tmp_path):
    psp = make_psp_dir(tmp_path / "psp", "In", "As")
    app = make_application(psp=psp)
    structure = tmp_path / "inAs.cif"
    structure.write_text(
        """data_InAs
_cell_length_a   6.0583
_cell_length_b   6.0583
_cell_length_c   6.0583
_cell_angle_alpha   90
_cell_angle_beta    90
_cell_angle_gamma   90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
In  In  0.0 0.0 0.0
As  As  0.25 0.25 0.25
""",
        encoding="utf-8",
    )
    workflow_dir = tmp_path / "wf"
    app.prepare_inputs(
        structure_path=str(structure),
        profile_name="standard_semiconductor",
        output_dir=workflow_dir,
    )
    vasprun_bytes = write_vasprun(tmp_path / "fixture").read_bytes()
    # Every stage now lands in its own unique remote directory (the runner
    # consumes ``job_ref.remote_directory``), so results are seeded at upload
    # time instead of guessing directory names.
    original_upload = app.backend.upload_files

    async def upload_with_results(local_paths, remote_directory):
        names = await original_upload(local_paths, remote_directory)
        app.backend.add_remote_file(remote_directory, "vasprun.xml", vasprun_bytes)
        app.backend.add_remote_file(
            remote_directory, "OUTCAR", "reached required accuracy\n"
        )
        app.backend.add_remote_file(remote_directory, "CONTCAR", "fake contcar")
        app.backend.add_remote_file(remote_directory, "CHGCAR", "fake")
        return names

    app.backend.upload_files = upload_with_results

    async def scenario():
        report = await app.submit_workflow(
            workflow_dir=workflow_dir, profile_name="standard_semiconductor"
        )
        assert report["all_valid"] is True
        assert [stage["stage"] for stage in report["stages"]] == [
            "relax",
            "static",
            "band",
            "dos",
        ]

    asyncio.run(scenario())


def test_workflow_local_timeout_cancels_without_collecting(tmp_path):
    from photomatagent.scientific.applications.vasp.workflow import run_vasp_workflow
    from photomatagent.scientific.remote.models import HPCJobState

    workflow = tmp_path / "workflow"
    stage_dir = workflow / "relax"
    stage_dir.mkdir(parents=True)
    (workflow / "workflow.json").write_text(
        json.dumps({"stages": [{"stage": "relax", "directory": str(stage_dir)}]}),
        encoding="utf-8",
    )

    class Backend:
        def __init__(self):
            self.cancelled: list[str] = []

        async def cancel_job(self, job_id: str):
            self.cancelled.append(job_id)

    class Application:
        def __init__(self):
            self.backend = Backend()
            self.collected = False

        async def submit_stage(self, **kwargs):
            return SimpleNamespace(job_id="42")

        async def status(self, job_id: str):
            return HPCJobState.RUNNING

        async def collect(self, **kwargs):
            self.collected = True
            raise AssertionError("timed-out jobs must not be collected")

    application = Application()
    with pytest.raises(TimeoutError, match="job 42"):
        asyncio.run(
            run_vasp_workflow(
                application=application,
                workflow_dir=workflow,
                profile_name="test",
                poll_interval_seconds=0,
                timeout_seconds=0,
            )
        )
    assert application.backend.cancelled == ["42"]
    assert application.collected is False


def test_tool_search_finds_vasp_tools():
    from photomatagent.scientific.applications.vasp.tools import (
        VaspCapabilityPack,
    )

    pack = VaspCapabilityPack(application=None)
    names = [tool.name for tool in pack.tools()]
    assert set(names) == {
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
    assert "vasp.inspect_result" not in names
    assert "vasp.run_workflow" not in names
    assert "vasp_molecule.capabilities" not in names
    assert "vasp_study.plan" not in names
    assert all(tool.exposure.value == "deferred" for tool in pack.tools())
