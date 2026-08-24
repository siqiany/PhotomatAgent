"""Offline tests for the MAGUS Real Integration Sprint (Sprint 4).

Nothing here touches SCNet: the FakeSCNetBackend scripts SSH replies and
holds an in-memory remote filesystem. Live probes are gated behind
``PHOTOMATAGENT_RUN_LIVE_SCIENCE=1``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from photomatagent.scientific.applications.magus.application import (
    MagusApplication,
    MagusPseudopotentialMissingError,
    MagusSubmissionBlockedError,
    MagusUnconfiguredError,
    default_magus_application,
)
from photomatagent.scientific.applications.magus.models import (
    MagusComposition,
    MagusGenerateRequest,
    MagusSlabConfig,
    MagusSearchRequest,
)
from photomatagent.scientific.applications.magus.probe import (
    parse_checkpack_calculators,
    parse_example_dirs,
    parse_example_structure_types,
    parse_magus_help_commands,
    parse_magus_version,
)
from photomatagent.scientific.applications.magus.render import (
    magus_arguments,
    render_generate_input,
    render_magus_slurm,
    render_search_input,
)
from photomatagent.scientific.remote.fake import FakeSCNetBackend
from photomatagent.scientific.remote.models import (
    HPCJobState,
    RemoteJobRef,
    ResourcePolicy,
    ResourceRequest,
)


CHECKPACK_OUTPUT = """calculator
-------------------
emt            : EMTCalculator
lj             : LJCalculator
vasp           : VaspCalculator
vaspc          : VaspCalculator
-------------------
"""

CHECKPACK_STDERR = """Fail when try to import magus.calculators.dpmd, because:
ModuleNotFoundError: No module named 'deepmd'
Fail when try to import magus.calculators.mace, because:
ModuleNotFoundError: No module named 'mace'
"""


def probe_backend() -> FakeSCNetBackend:
    backend = FakeSCNetBackend()
    backend.add_ssh_script("test -d ~/magus", "EXISTS")
    backend.add_ssh_script("test -d /opt/magus", "EXISTS")
    backend.add_ssh_script("test -x /opt/magus/bin/magus", "OK")
    backend.add_ssh_script("test -x ~/magus/bin/magus", "OK")
    backend.add_ssh_script("test -d ~/magus/bin", "bin-EXISTS")
    backend.add_ssh_script("-v 2>&1", "2.1.0")
    backend.add_ssh_script(
        "-h 2>&1",
        "usage: magus [-h] [-v]\n"
        "             {search,summary,clean,prepare,calculate,generate,checkpack,"
        "test,update,tool,mutate}\n",
    )
    backend.add_ssh_script("checkpack calculators", CHECKPACK_OUTPUT, ok=True)
    backend.add_ssh_script("test -f ~/magus/examples.zip", "ZIP")
    backend.add_ssh_script(
        "unzip -l ~/magus/examples.zip",
        "     428  12-19-2025 16:01   magus-master-examples/examples/01--1-B12/input.yaml\n"
        "       0  12-19-2025 16:01   magus-master-examples/examples/01--1-B12/\n"
        "       0  12-19-2025 16:01   magus-master-examples/examples/05--1-LJ26/\n"
        "       0  12-19-2025 16:01   magus-master-examples/examples/06--1-C_2x1_100/\n",
    )
    backend.add_ssh_script(
        "grep -m1 '^structureType:'",
        "structureType: bulk\nstructureType: cluster\nstructureType: surface\n",
    )
    backend.add_ssh_script("examples.zip", "magus-master-examples/examples/")
    return backend


def scripted_application(**kwargs) -> MagusApplication:
    backend = probe_backend()
    return MagusApplication(backend, **kwargs)


# -- composition / models -----------------------------------------------------


def test_formula_parsing_deterministic():
    composition = MagusComposition.from_formula("InAs")
    assert composition.symbols == ["In", "As"]
    assert composition.formula == [1, 1]
    assert composition.formula_type == "fix"


def test_formula_parsing_fixed_composition():
    composition = MagusComposition.from_formula("HgTe")
    assert composition.symbols == ["Hg", "Te"]
    assert composition.formula == [1, 1]


def test_formula_parsing_variable_composition():
    composition = MagusComposition.from_formula("PbTe", formula_type="var")
    assert composition.formula_type == "var"


def test_formula_parsing_rejects_garbage():
    with pytest.raises(Exception):
        MagusComposition.from_formula("NotAFormula!?")


def test_request_max_atoms_ge_min():
    with pytest.raises(Exception):
        MagusSearchRequest.from_composition(
            "InAs", min_atoms=8, max_atoms=4
        )


# -- renderers -----------------------------------------------------------------


def test_generate_input_yaml_matches_installed_keys():
    request = MagusGenerateRequest.from_composition(
        "InAs", structure_type="bulk", number=5, min_atoms=8, max_atoms=8
    )
    text = render_generate_input(request)
    assert "formulaType: fix" in text
    assert "structureType: bulk" in text
    assert "symbols: ['In', 'As']" in text
    assert "formula: [1, 1]" in text
    assert "min_n_atoms: 8" in text
    assert "max_n_atoms: 8" in text
    assert "spacegroup: ['1-230']" in text
    assert "d_ratio: 0.5" in text
    assert "volume_ratio: 8" in text


def test_bulk_search_input_yaml():
    request = MagusSearchRequest.from_composition(
        "Al",
        structure_type="bulk",
        calculator="vasp",
        init_size=4,
        population_size=4,
        generations=1,
        save_good=2,
        min_atoms=4,
        max_atoms=4,
    )
    text = render_search_input(request)
    assert "initSize: 4" in text
    assert "popSize: 4" in text
    assert "numGen: 1" in text
    assert "saveGood: 2" in text
    assert "pressure: 0" in text
    assert "MainCalculator:" in text
    assert "calculator: 'vasp'" in text
    assert "mode: 'serial'" in text
    assert "ppLabel: ['']" in text
    assert "add_sym: True" in text


def test_cluster_search_input_yaml():
    request = MagusSearchRequest.from_composition(
        "H",
        structure_type="cluster",
        calculator="lj",
        min_atoms=26,
        max_atoms=26,
    )
    text = render_search_input(request)
    assert "structureType: cluster" in text
    assert "calculator: 'lj'" in text


def test_surface_search_input_yaml_with_slab():
    request = MagusSearchRequest.from_composition(
        "C",
        structure_type="surface",
        calculator="vasp",
    )
    request.slab = MagusSlabConfig(bulk_file="diamond.vasp")
    text = render_search_input(request)
    assert "structureType: surface" in text
    assert "slabinfo:" in text
    assert "bulk_file: 'diamond.vasp'" in text


def test_magus_arguments_match_installed_cli():
    generate = MagusGenerateRequest.from_composition("B", number=10)
    assert magus_arguments("generate", generate) == [
        "generate", "-i", "input.yaml", "-o", "gen.traj", "-n", "10",
    ]
    search = MagusSearchRequest.from_composition("Al")
    assert magus_arguments("search", search) == ["search", "-i", "input.yaml"]


def test_slurm_rendering_launcher_empty_and_job_system():
    script = render_magus_slurm(
        job_name="magus-gen-x",
        executable="~/magus/bin/magus",
        args=["generate", "-i", "input.yaml", "-o", "gen.traj", "-n", "5"],
        resource=ResourceRequest(
            partition="kshcnormal", nodes=1, tasks_per_node=8, walltime_minutes=60
        ),
        magus_root="~/magus",
        vasp_pp_path="~/psp",
    )
    assert "srun --mpi=pmi2" not in script
    assert script.rstrip().endswith(
        "magus generate -i input.yaml -o gen.traj -n 5"
    )
    assert "export JOB_SYSTEM=SLURM" in script
    # PATH is a deterministic double-quoted preamble (single quotes would
    # prevent $PATH expansion); generate never loads the VASP environment.
    assert 'export PATH="$HOME/magus/bin:$PATH"' in script
    assert "export VASP_PP_PATH" not in script


def test_slurm_rendering_sources_env_scripts():
    script = render_magus_slurm(
        job_name="magus-search-x",
        executable="/opt/magus/bin/magus",
        args=["search", "-i", "input.yaml"],
        resource=ResourceRequest(),
        vasp_script="/public/home/u/apprepo/vasp/env.sh",
        needs_vasp=True,
        ase_vasp_command="srun --mpi=pmi2 vasp_std",
    )
    assert "source /public/home/u/apprepo/vasp/env.sh" in script
    assert 'export ASE_VASP_COMMAND="srun --mpi=pmi2 vasp_std"' in script


def test_slurm_generate_never_loads_vasp_environment():
    """magus generate must not carry VASP env / PP path / ASE command."""
    script = render_magus_slurm(
        job_name="magus-gen-y",
        executable="/opt/magus/bin/magus",
        args=["generate", "-i", "input.yaml", "-o", "gen.traj", "-n", "5"],
        resource=ResourceRequest(),
        magus_root="/opt/magus",
        vasp_script="/public/home/u/apprepo/vasp/env.sh",
        vasp_pp_path="/public/home/u",
        ase_vasp_command="srun --mpi=pmi2 vasp_std",
    )
    assert "export JOB_SYSTEM=SLURM" in script
    assert 'export PATH="/opt/magus/bin:$PATH"' in script
    assert "VASP_PP_PATH" not in script
    assert "ASE_VASP_COMMAND" not in script
    assert "source /public/home/u/apprepo/vasp/env.sh" not in script


def test_slurm_non_vasp_search_never_loads_vasp_environment():
    """LJ/EMT searches load MAGUS only, exactly like generate."""
    script = render_magus_slurm(
        job_name="magus-lj-search",
        executable="/opt/magus/bin/magus",
        args=["search", "-i", "input.yaml"],
        resource=ResourceRequest(),
        magus_root="/opt/magus",
        vasp_script="/public/home/u/apprepo/vasp/env.sh",
        vasp_pp_path="/public/home/u",
        ase_vasp_command="srun --mpi=pmi2 vasp_std",
    )
    assert "VASP_PP_PATH" not in script
    assert "ASE_VASP_COMMAND" not in script
    assert "source /public/home/u/apprepo/vasp/env.sh" not in script
    assert 'export PATH="/opt/magus/bin:$PATH"' in script


def test_slurm_vasp_search_loads_full_vasp_environment():
    script = render_magus_slurm(
        job_name="magus-vasp-search",
        executable="/opt/magus/bin/magus",
        args=["search", "-i", "input.yaml"],
        resource=ResourceRequest(),
        magus_root="/opt/magus",
        vasp_script="/public/home/u/apprepo/vasp/env.sh",
        vasp_pp_path="/public/home/u",
        ase_vasp_command="srun --mpi=pmi2 vasp_std",
        needs_vasp=True,
    )
    assert "export VASP_PP_PATH=/public/home/u" in script
    assert 'export ASE_VASP_COMMAND="srun --mpi=pmi2 vasp_std"' in script
    assert "source /public/home/u/apprepo/vasp/env.sh" in script


def test_slurm_path_expansion_never_single_quoted():
    script = render_magus_slurm(
        job_name="magus-path",
        executable="~/magus/bin/magus",
        args=["generate", "-i", "input.yaml", "-o", "gen.traj", "-n", "3"],
        resource=ResourceRequest(),
        magus_root="~/magus",
    )
    assert 'export PATH="$HOME/magus/bin:$PATH"' in script
    assert "export PATH='~/magus/bin:$PATH'" not in script
    # $PATH must remain expandable (double quotes), never a literal.
    assert "':$PATH'" not in script


def test_slurm_tilde_executable_invoked_via_path():
    """A ~/ executable cannot be quoted (tilde does not expand inside
    quotes): it must be invoked by basename through the $HOME-based PATH
    preamble instead."""
    script = render_magus_slurm(
        job_name="magus-tilde-exe",
        executable="~/magus/bin/magus",
        args=["generate", "-i", "input.yaml", "-o", "gen.traj", "-n", "3"],
        resource=ResourceRequest(),
        magus_root="~/magus",
    )
    assert script.rstrip().endswith(
        "magus generate -i input.yaml -o gen.traj -n 3"
    )
    assert "'~/magus/bin/magus' generate" not in script
    assert 'export PATH="$HOME/magus/bin:$PATH"' in script


def test_validate_configured_vasp_command():
    from photomatagent.scientific.applications.magus.render import (
        validate_configured_vasp_command,
    )

    assert (
        validate_configured_vasp_command("srun --mpi=pmi2 vasp_std")
        == "srun --mpi=pmi2 vasp_std"
    )
    assert validate_configured_vasp_command("  vasp_std  ") == "vasp_std"
    for bad in (
        "srun\nvasp_std",
        "srun\rvasp_std",
        "vasp_std; rm -rf /",
        "vasp_std && echo hi",
        "vasp_std || true",
        "vasp_std > /tmp/x",
        "vasp_std < /dev/null",
        "vasp_std | grep x",
        "`vasp_std`",
        "$(vasp_std)",
        "echo $HOME",
        'echo "hi"',
        "",
        "   ",
    ):
        with pytest.raises(ValueError):
            validate_configured_vasp_command(bad)


# -- default application -------------------------------------------------------


def test_default_magus_application_from_environment(monkeypatch, tmp_path):
    key = tmp_path / "id_key"
    key.write_text("key")
    monkeypatch.setenv("SCNET_HOST", "cancon.example")
    monkeypatch.setenv("SCNET_USERNAME", "scniv4a4go")
    monkeypatch.setenv("SCNET_PORT", "65023")
    monkeypatch.setenv("SCNET_PRIVATE_KEY_PATH", str(key))
    monkeypatch.setenv("SCNET_REMOTE_ROOT", "~/myjob")
    monkeypatch.setenv("SCNET_MAGUS_ROOT", "/opt/magus")
    monkeypatch.setenv("SCNET_MAGUS_VASP_PP_PATH", "/opt/psp")
    monkeypatch.setenv("SCNET_MAGUS_ASE_VASP_COMMAND", "srun --mpi=pmi2 vasp_std")
    monkeypatch.setenv("SCNET_VASP_ENV_SCRIPT", "/opt/vasp/env.sh")
    app = default_magus_application()
    assert app is not None
    assert app.magus_root == "/opt/magus"
    assert app.vasp_pp_path == "/opt/psp"
    assert app.ase_vasp_command == "srun --mpi=pmi2 vasp_std"
    assert app.vasp_script == "/opt/vasp/env.sh"
    assert app.backend is not None
    assert app.backend.config.host == "cancon.example"


def test_default_magus_application_unconfigured(monkeypatch):
    for name in ("SCNET_HOST", "SCNET_USERNAME", "SUPERCOMPUTING_HOST", "SUPERCOMPUTING_USERNAME"):
        monkeypatch.delenv(name, raising=False)
    assert default_magus_application() is None


# -- remote probe ---------------------------------------------------------------


def test_probe_reports_available_with_real_values():
    app = scripted_application()
    report = asyncio.run(app.probe_environment_async())
    assert report["status"] == "AVAILABLE"
    assert report["version"] == "2.1.0"
    assert report["executable"] == "~/magus/bin/magus"
    assert "generate" in report["commands"]
    assert "vasp" in report["calculators"]
    assert "surface" in report["structure_types"]
    assert report["job_system"] == "SLURM"
    assert report["bin_exists"] is True


def test_probe_missing_root():
    backend = FakeSCNetBackend()
    backend.add_ssh_script("test -d ~/magus", "", ok=False)
    app = MagusApplication(backend)
    report = asyncio.run(app.probe_environment_async())
    assert report["status"] == "MISSING_DEPENDENCY"
    assert report["error_type"] == "MISSING_DEPENDENCY"


def test_probe_explicit_executable_override():
    backend = FakeSCNetBackend()
    backend.add_ssh_script("test -d ~/magus", "EXISTS")
    backend.add_ssh_script("test -x /opt/magus/bin/magus", "OK")
    backend.add_ssh_script("-v 2>&1", "2.1.0")
    app = MagusApplication(backend, executable="/opt/magus/bin/magus")
    report = asyncio.run(app.probe_environment_async())
    assert report["executable"] == "/opt/magus/bin/magus"


def test_probe_vasp_readiness_layered_ready():
    backend = FakeSCNetBackend()
    backend.add_ssh_script("test -f ~/psp/In/POTCAR", "OK")
    backend.add_ssh_script(
        'p=$(command -v "srun" 2>/dev/null) && '
        'echo "FOUND-srun=$p" || echo "MISSING-srun"; '
        'p=$(command -v "vasp_std" 2>/dev/null) && '
        'echo "FOUND-vasp_std=$p" || echo "MISSING-vasp_std"',
        "FOUND-srun=/usr/bin/srun\nFOUND-vasp_std=/opt/vasp/vasp_std",
    )
    app = MagusApplication(
        backend,
        vasp_pp_path="~/psp",
        vasp_script="/opt/vasp/env.sh",
        ase_vasp_command="srun --mpi=pmi2 vasp_std",
    )
    psp = asyncio.run(app._probe_psp_readiness())
    report = asyncio.run(app._probe_vasp_readiness(["vasp"], psp))
    assert report["calculator"] == "READY"
    assert report["environment"] == "READY"
    assert report["ase_command_configured"] is True
    assert report["ase_command_verified"] is True
    assert report["pseudopotential_path"] == "READY"
    assert report["overall"] == "READY"


def test_probe_vasp_readiness_partial_without_ase_command():
    backend = FakeSCNetBackend()
    backend.add_ssh_script("test -f ~/psp/In/POTCAR", "OK")
    app = MagusApplication(
        backend, vasp_pp_path="~/psp", vasp_script="/opt/vasp/env.sh"
    )
    psp = asyncio.run(app._probe_psp_readiness())
    report = asyncio.run(app._probe_vasp_readiness(["vasp"], psp))
    assert report["calculator"] == "READY"
    assert report["environment"] == "READY"
    assert report["ase_command_configured"] is False
    assert report["ase_command_verified"] is False
    assert report["overall"] == "PARTIAL"


def test_probe_vasp_readiness_missing_calculator():
    backend = FakeSCNetBackend()
    app = MagusApplication(backend)
    psp = asyncio.run(app._probe_psp_readiness())
    report = asyncio.run(app._probe_vasp_readiness(["emt"], psp))
    assert report["calculator"] == "MISSING"
    assert report["overall"] == "MISSING"


def test_probe_bounded_find_fallback():
    backend = FakeSCNetBackend()
    backend.add_ssh_script("test -d ~/magus", "EXISTS")
    backend.add_ssh_script("test -x ~/magus/bin/magus", "", ok=False)
    backend.add_ssh_script("test -x ~/magus/magus", "", ok=False)
    backend.add_ssh_script(
        "find ~/magus -maxdepth 4",
        "/public/home/u/magus/envs/magus/bin/magus\n",
    )
    backend.add_ssh_script("-v 2>&1", "2.1.0")
    app = MagusApplication(backend)
    report = asyncio.run(app.probe_environment_async())
    assert report["executable"] == "/public/home/u/magus/envs/magus/bin/magus"


def test_probe_missing_executable():
    backend = FakeSCNetBackend()
    backend.add_ssh_script("test -d ~/magus", "EXISTS")
    backend.add_ssh_script("test -x", "", ok=False)
    backend.add_ssh_script("find ~/magus -maxdepth 4", "")
    app = MagusApplication(backend)
    report = asyncio.run(app.probe_environment_async())
    assert report["status"] == "MISSING_DEPENDENCY"


def test_probe_unconfigured_without_backend():
    app = MagusApplication(executable="magus-x")
    report = app.probe_environment()
    assert report["status"] == "UNCONFIGURED"
    assert "UNVALIDATED" in report["candidate_validity"]


def test_probe_environment_no_nested_asyncio_run():
    """probe_environment() inside a running loop must not asyncio.run."""
    app = scripted_application()

    async def scenario():
        report = app.probe_environment()
        assert report["status"] == "UNKNOWN"
        assert "probe_environment_async" in report["detail"]

    asyncio.run(scenario())


def test_probe_sync_uses_remote_outside_loop():
    app = scripted_application()
    report = app.probe_environment()
    assert report["status"] == "AVAILABLE"


# -- parsers ---------------------------------------------------------------------


def test_version_parsing():
    assert parse_magus_version("2.1.0\n") == "2.1.0"
    assert parse_magus_version(" 1.0.3rc1 \n") == "1.0.3rc1"
    assert parse_magus_version("junk") == ""


def test_help_parsing():
    commands = parse_magus_help_commands(
        "usage: magus [-h] [-v]\n"
        "             {search,summary,clean,prepare,calculate,generate,checkpack}\n"
    )
    assert "generate" in commands and "search" in commands


def test_checkpack_parsing():
    parsed = parse_checkpack_calculators(CHECKPACK_OUTPUT, CHECKPACK_STDERR)
    assert "vasp" in parsed["available"]
    assert "emt" in parsed["available"]
    assert "dpmd" in parsed["failed"]
    assert "mace" in parsed["failed"]


def test_example_discovery_parsing():
    dirs = parse_example_dirs(
        "     428  12-19-2025 16:01   magus-master-examples/examples/01--1-B12/input.yaml\n"
        "       0  12-19-2025 16:01   magus-master-examples/examples/05--1-LJ26/\n"
    )
    assert any("05--1-LJ26" in item for item in dirs)


def test_example_structure_type_parsing():
    types = parse_example_structure_types(
        "structureType: bulk\nstructureType: cluster\nstructureType: surface\n"
    )
    assert types == ["bulk", "cluster", "surface"]


# -- preparation lifecycle ---------------------------------------------------------


def test_prepare_generate_creates_required_files(tmp_path):
    app = MagusApplication()
    job = tmp_path / "gen"
    request = MagusGenerateRequest.from_composition("B", number=5)
    manifest = app.prepare_generate(request, job)
    assert (job / "input.yaml").is_file()
    assert (job / "magus.slurm").is_file()
    assert (job / "photomat_manifest.json").is_file()
    assert (job / "magus_manifest.json").is_file()
    assert manifest["operation"] == "generate"
    assert manifest["input_hash"]
    assert manifest["candidate_lineage_root"].startswith("magus_generate_")


def test_prepare_search_creates_required_files(tmp_path):
    app = MagusApplication()
    job = tmp_path / "search"
    request = MagusSearchRequest.from_composition(
        "Al", calculator="vasp", min_atoms=4, max_atoms=4
    )
    manifest = app.prepare_search(request, job)
    assert (job / "input.yaml").is_file()
    assert (job / "inputFold" / "VASP" / "INCAR").is_file()
    assert (job / "Seeds").is_dir()
    assert (job / "magus.slurm").is_file()
    assert manifest["operation"] == "search"
    assert "pseudopotentials" in manifest
    assert manifest["pseudopotentials"][0]["element"] == "Al"


def test_prepare_search_surface_without_slab_rejected(tmp_path):
    app = MagusApplication()
    request = MagusSearchRequest.from_composition(
        "C", structure_type="surface", calculator="vasp"
    )
    with pytest.raises(Exception, match="slab"):
        app.prepare_search(request, tmp_path / "surface")


def test_legacy_prepare_builds_full_tree(tmp_path):
    app = MagusApplication()
    target = tmp_path / "target"
    target.mkdir()
    out = tmp_path / "out"
    manifest = app.prepare(
        search_type="bulk",
        composition="HgTe",
        target_dir=target,
        output_dir=out,
    )
    assert manifest["status"] == "PREPARED"
    assert (out / "input.yaml").is_file()
    assert (out / "magus.slurm").is_file()


# -- submit / status / collect -------------------------------------------------------


def test_submit_rejects_manifest_only_tree(tmp_path):
    backend = FakeSCNetBackend()
    app = MagusApplication(backend)
    job = tmp_path / "job"
    job.mkdir()
    (job / "photomat_manifest.json").write_text("{}", encoding="utf-8")

    async def scenario():
        with pytest.raises(ValueError, match="input.yaml"):
            await app.submit(job_name="x", prepared_dir=job)

    asyncio.run(scenario())


def test_submit_rejects_missing_slurm(tmp_path):
    backend = FakeSCNetBackend()
    app = MagusApplication(backend)
    job = tmp_path / "job"
    job.mkdir()
    (job / "input.yaml").write_text("x", encoding="utf-8")
    (job / "photomat_manifest.json").write_text("{}", encoding="utf-8")

    async def scenario():
        with pytest.raises(ValueError, match="magus.slurm"):
            await app.submit(job_name="x", prepared_dir=job)

    asyncio.run(scenario())


def test_submit_success_uploads_tree_and_returns_ref(tmp_path):
    backend = FakeSCNetBackend()
    app = MagusApplication(
        backend, remote_root="~/science", executable="~/magus/bin/magus"
    )
    job = tmp_path / "job"
    app.prepare_generate(MagusGenerateRequest.from_composition("B", number=3), job)

    async def scenario():
        ref = await app.submit(
            job_name="tiny-gen",
            prepared_dir=job,
            resource=ResourceRequest(
                partition="kshcnormal",
                nodes=1,
                tasks_per_node=8,
                walltime_minutes=60,
            ),
        )
        assert ref.backend == "fake-scnet"
        assert ref.application == "magus"
        assert ref.remote_directory.startswith("~/science/magus/tiny-gen-")
        remote = backend.remote_files[ref.remote_directory]
        assert "input.yaml" in remote
        assert "magus.slurm" in remote
        assert "photomat_manifest.json" in remote

    asyncio.run(scenario())


def test_submit_blocked_by_policy(tmp_path):
    backend = FakeSCNetBackend(policy=ResourcePolicy(allow_hpc_submit=False))
    app = MagusApplication(backend)
    job = tmp_path / "job"
    app.prepare_generate(MagusGenerateRequest.from_composition("B", number=3), job)

    async def scenario():
        with pytest.raises(MagusSubmissionBlockedError):
            await app.submit(job_name="x", prepared_dir=job)

    asyncio.run(scenario())


def test_submit_vasp_search_checks_pseudopotentials(tmp_path):
    backend = FakeSCNetBackend()
    backend.add_ssh_script("test -f ~/psp/In/POTCAR", "", ok=False)
    backend.add_ssh_script("test -f ~/psp/potpaw_PBE/In/POTCAR", "OK")
    backend.add_ssh_script("test -f ~/psp/potpaw_PBE/Al/POTCAR", "OK")
    app = MagusApplication(
        backend, vasp_pp_path="~/psp", executable="~/magus/bin/magus"
    )
    job = tmp_path / "job"
    app.prepare_search(
        MagusSearchRequest.from_composition("Al", calculator="vasp"), job
    )

    async def scenario():
        ref = await app.submit(job_name="vasp-search", prepared_dir=job)
        assert ref.application == "magus"

    asyncio.run(scenario())


def test_submit_vasp_search_missing_pseudopotential(tmp_path):
    backend = FakeSCNetBackend()
    backend.add_ssh_script("test -f ~/psp/Al/POTCAR", "", ok=False)
    app = MagusApplication(
        backend, vasp_pp_path="~/psp", executable="~/magus/bin/magus"
    )
    job = tmp_path / "job"
    app.prepare_search(
        MagusSearchRequest.from_composition("Al", calculator="vasp"), job
    )

    async def scenario():
        with pytest.raises(MagusPseudopotentialMissingError) as exc_info:
            await app.submit(job_name="vasp-search", prepared_dir=job)
        assert exc_info.value.missing == ["Al"]

    asyncio.run(scenario())


def test_submit_vasp_search_requires_pp_path(tmp_path):
    backend = FakeSCNetBackend()
    app = MagusApplication(backend)
    job = tmp_path / "job"
    app.prepare_search(
        MagusSearchRequest.from_composition("Al", calculator="vasp"), job
    )

    async def scenario():
        with pytest.raises(Exception, match="SCNET_MAGUS_VASP_PP_PATH"):
            await app.submit(job_name="x", prepared_dir=job)

    asyncio.run(scenario())


def test_status_mapping():
    backend = FakeSCNetBackend(scripted_states=[HPCJobState.RUNNING])
    app = MagusApplication(backend)

    async def scenario():
        assert await app.status("1001") == HPCJobState.RUNNING

    asyncio.run(scenario())


def test_status_unknown_without_backend():
    app = MagusApplication()

    async def scenario():
        assert await app.status("1") == HPCJobState.UNKNOWN

    asyncio.run(scenario())


def test_collect_downloads_bounded_artifacts(tmp_path):
    import io

    from ase import Atoms
    from ase.io import write

    backend = FakeSCNetBackend()
    remote = "~/science/magus/job-12345678"
    backend.add_remote_file(remote, "input.yaml", "formulaType: fix\n")
    backend.add_remote_file(remote, "magus.slurm", "#!/bin/bash\n")
    backend.add_remote_file(remote, "photomat_manifest.json", json.dumps(
        {"application": "magus", "operation": "search"}
    ))
    backend.add_remote_file(remote, "log.txt", "some log\n")
    backend.add_remote_file(remote, "summary", "symmetry enthalpy formula priFormula\n")
    buffer = io.BytesIO()
    write(buffer, Atoms("Al", positions=[[0, 0, 0]], cell=[4, 4, 4]), format="traj")
    backend.add_remote_file(remote, "results/best.traj", buffer.getvalue())
    backend.add_remote_file(remote, "huge.bin", b"y" * 10_000_000)
    app = MagusApplication(backend)
    ref = RemoteJobRef(
        backend="fake-scnet",
        application="magus",
        job_id="1001",
        remote_directory=remote,
    )

    async def scenario():
        report = await app.collect(job_ref=ref, local_dir=tmp_path / "out")
        names = {path.name for path in (tmp_path / "out").rglob("*") if path.is_file()}
        assert "input.yaml" in names
        assert "summary" in names
        assert "huge.bin" not in names
        # Nested artifacts must keep their relative structure (scp
        # basename-in-target semantics), not double-nest.
        assert (tmp_path / "out" / "results" / "best.traj").is_file()
        assert report["candidate_count"] == 1
        assert report["artifact_candidate_counts"] == {"results/best.traj": 1}

    asyncio.run(scenario())


def test_interesting_artifact_filter():
    app = MagusApplication()
    assert app._interesting_artifact("summary")
    assert app._interesting_artifact("results/best.traj")
    assert app._interesting_artifact("gen.traj")
    assert app._interesting_artifact("inputFold/VASP/INCAR")
    assert not app._interesting_artifact("WAVECAR")
    assert not app._interesting_artifact("CHGCAR")


def test_summary_parsing():
    app = MagusApplication()
    text = (
        "symmetry enthalpy formula priFormula\n"
        "  I4_1md (109)     None     B12         B6\n"
        "  Pbam (55)     None     B12        B12\n"
    )
    rows = app._parse_summary_rows(text)
    assert len(rows) == 2
    assert rows[0]["symmetry"] == "I4_1md (109)"


def test_inspect_results_reports_unverified_generate(tmp_path):
    app = MagusApplication()
    out = tmp_path / "res"
    out.mkdir()
    report = app.inspect_results(out, operation="generate", expected_number=5)
    assert report["candidates"] == [{"requested": 5, "verified_from_artifact": False}]
    # requested != verified: no artifact means no verified count.
    assert report["candidate_count"] is None


def test_inspect_results_no_fabrication_when_nothing_downloaded(tmp_path):
    app = MagusApplication()
    out = tmp_path / "empty"
    out.mkdir()
    report = app.inspect_results(out, operation="search")
    assert report["candidates"] == []
    assert report["candidate_count"] is None
    assert report["summary"] == ""


def test_generate_candidate_count_uses_traj_frames(tmp_path):
    import io

    from ase import Atoms
    from ase.io import write

    app = MagusApplication()
    out = tmp_path / "res"
    out.mkdir()
    buffer = io.BytesIO()
    write(
        buffer,
        [Atoms("B", positions=[[0, 0, 0]], cell=[4, 4, 4]) for _ in range(3)],
        format="traj",
    )
    (out / "gen.traj").write_bytes(buffer.getvalue())
    report = app.inspect_results(out, operation="generate", expected_number=3)
    assert report["candidate_count"] == 3
    assert report["artifact_candidate_counts"] == {"gen.traj": 3}
    assert len(report["candidates"]) == 1
    assert report["candidates"][0] == {"artifact": "gen.traj", "frames": 3}


def test_search_candidate_count_summary_priority(tmp_path):
    import io

    from ase import Atoms
    from ase.io import write

    app = MagusApplication()
    out = tmp_path / "res"
    results = out / "results"
    results.mkdir(parents=True)
    (out / "summary").write_text(
        "symmetry enthalpy formula priFormula\n"
        "  I4_1md (109)     None     B12         B6\n"
        "  Pbam (55)     None     B12        B12\n",
        encoding="utf-8",
    )
    buffer = io.BytesIO()
    write(
        buffer,
        [Atoms("B", positions=[[0, 0, 0]], cell=[4, 4, 4]) for _ in range(3)],
        format="traj",
    )
    (results / "best.traj").write_bytes(buffer.getvalue())
    report = app.inspect_results(out, operation="search")
    # summary rows (2) win over best.traj frames (3); never summed.
    assert report["candidate_count"] == 2
    assert report["artifact_candidate_counts"] == {"results/best.traj": 3}


def test_search_candidate_count_best_traj_fallback(tmp_path):
    import io

    from ase import Atoms
    from ase.io import write

    app = MagusApplication()
    out = tmp_path / "res"
    results = out / "results"
    results.mkdir(parents=True)
    buffer = io.BytesIO()
    write(
        buffer,
        [Atoms("Al", positions=[[0, 0, 0]], cell=[4, 4, 4]) for _ in range(3)],
        format="traj",
    )
    (results / "best.traj").write_bytes(buffer.getvalue())
    (results / "good.traj").write_bytes(buffer.getvalue())
    report = app.inspect_results(out, operation="search")
    assert report["candidate_count"] == 3


def test_manifest_provenance(tmp_path):
    app = MagusApplication()
    job = tmp_path / "job"
    app.prepare_generate(MagusGenerateRequest.from_composition("B", number=2), job)
    manifest = json.loads((job / "photomat_manifest.json").read_text())
    assert manifest["application"] == "magus"
    assert manifest["backend"] == "scnet"
    assert manifest["input_hash"]
    assert "created_at" in manifest
    assert manifest["candidate_lineage_root"].startswith("magus_generate_")
    assert "UNVALIDATED" in manifest["execution"]["limitations"][0]


# -- tools surface ------------------------------------------------------------------


def test_magus_tool_pack_lifecycle_surface():
    from photomatagent.scientific.applications.magus.tools import (
        MagusCapabilityPack,
    )

    pack = MagusCapabilityPack()
    names = [tool.name for tool in pack.tools()]
    for expected in (
        "magus.capabilities",
        "magus.prepare_generate",
        "magus.prepare_search",
        "magus.submit",
        "magus.status",
        "magus.collect",
        "magus.inspect_results",
        "magus.search_bulk",
        "magus.search_cluster",
        "magus.search_surface",
    ):
        assert expected in names
    assert all(tool.exposure.value == "deferred" for tool in pack.tools())


def test_mcp_scnet_server_registers_magus_lifecycle_tools():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "photomatagent.mcp_servers.scnet.server",
            "--doctor",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd="/home/shiqiany/AIagent/PhomatAgent",
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert "vasp" in report and "namd" in report and "magus" in report


def test_live_magus_probe_gated():
    """Gated live probe (read-only, no submission)."""
    import os

    if os.environ.get("PHOTOMATAGENT_RUN_LIVE_SCIENCE") != "1":
        pytest.skip("set PHOTOMATAGENT_RUN_LIVE_SCIENCE=1 for the live probe")
    app = default_magus_application()
    assert app is not None
    report = asyncio.run(app.probe_environment_async())
    assert report["status"] == "AVAILABLE"
    assert report["version"]


def test_live_magus_generate_acceptance(tmp_path):
    """Real tiny MAGUS generate acceptance (no VASP). Gated on BOTH
    PHOTOMATAGENT_RUN_LIVE_SCIENCE=1 and PHOTOMATAGENT_ALLOW_HPC_SUBMIT=1.

    Run::

        PHOTOMATAGENT_RUN_LIVE_SCIENCE=1 PHOTOMATAGENT_ALLOW_HPC_SUBMIT=1 \\
            uv run pytest tests/test_magus_sprint.py -k live_magus_generate

    Execution acceptance only -- not a scientific result.
    """
    import os
    import time

    if os.environ.get("PHOTOMATAGENT_RUN_LIVE_SCIENCE") != "1" or (
        os.environ.get("PHOTOMATAGENT_ALLOW_HPC_SUBMIT") != "1"
    ):
        pytest.skip(
            "set PHOTOMATAGENT_RUN_LIVE_SCIENCE=1 and "
            "PHOTOMATAGENT_ALLOW_HPC_SUBMIT=1 for the live generate acceptance"
        )
    app = default_magus_application()
    assert app is not None
    request = MagusGenerateRequest.from_composition(
        "B",
        structure_type="bulk",
        number=3,
        min_atoms=12,
        max_atoms=12,
    )
    job = tmp_path / "magus-generate-acceptance"
    app.prepare_generate(request, job)
    partition = os.environ.get("SCNET_PARTITION", "kshcnormal")
    ref = asyncio.run(
        app.submit(
            job_name="acceptance-gen",
            prepared_dir=job,
            resource=ResourceRequest(
                partition=partition,
                nodes=1,
                tasks_per_node=2,
                walltime_minutes=10,
            ),
        )
    )

    async def wait_terminal() -> HPCJobState:
        state = HPCJobState.SUBMITTED
        deadline = time.monotonic() + 20 * 60
        while not state.terminal and time.monotonic() < deadline:
            await asyncio.sleep(20)
            state = await app.status(ref.job_id)
        return state

    state = asyncio.run(wait_terminal())
    assert state == HPCJobState.COMPLETED, f"job {ref.job_id} ended {state.value}"
    report = asyncio.run(
        app.collect(job_ref=ref, local_dir=tmp_path / "results")
    )
    assert report["candidate_count"] == 3
    assert report["artifact_candidate_counts"] == {"gen.traj": 3}
    assert "gen.traj" in {
        Path(artifact["name"]).name for artifact in report["artifacts"]
    }
    print(
        "MAGUS generate acceptance: job",
        ref.job_id,
        "state",
        state.value,
        "candidates",
        report["candidate_count"],
    )


def test_live_magus_vasp_tiny_acceptance(tmp_path):
    """Real tiny MAGUS+VASP acceptance (Al, serial, VASP through ASE).

    Triple-gated on PHOTOMATAGENT_RUN_LIVE_SCIENCE=1,
    PHOTOMATAGENT_ALLOW_HPC_SUBMIT=1 and
    PHOTOMATAGENT_RUN_LIVE_MAGUS_VASP=1; skipped by default. Requires the
    verified SCNet configuration (SCNET_MAGUS_ASE_VASP_COMMAND and
    SCNET_MAGUS_VASP_PP_PATH). Execution acceptance only -- not a
    scientific result; Al structure stability is NOT evaluated.

    Run::

        PHOTOMATAGENT_RUN_LIVE_SCIENCE=1 PHOTOMATAGENT_ALLOW_HPC_SUBMIT=1 \\
            PHOTOMATAGENT_RUN_LIVE_MAGUS_VASP=1 \\
            uv run pytest tests/test_magus_sprint.py -k live_magus_vasp -q -s
    """
    import os
    import time

    if (
        os.environ.get("PHOTOMATAGENT_RUN_LIVE_SCIENCE") != "1"
        or os.environ.get("PHOTOMATAGENT_ALLOW_HPC_SUBMIT") != "1"
        or os.environ.get("PHOTOMATAGENT_RUN_LIVE_MAGUS_VASP") != "1"
    ):
        pytest.skip(
            "set PHOTOMATAGENT_RUN_LIVE_SCIENCE=1, "
            "PHOTOMATAGENT_ALLOW_HPC_SUBMIT=1 and "
            "PHOTOMATAGENT_RUN_LIVE_MAGUS_VASP=1 for the MAGUS+VASP "
            "acceptance"
        )
    app = default_magus_application()
    assert app is not None
    request = MagusSearchRequest.from_composition(
        "Al",
        structure_type="bulk",
        calculator="vasp",
        execution_mode="serial",
        init_size=2,
        population_size=2,
        generations=1,
        save_good=1,
        min_atoms=4,
        max_atoms=4,
    )
    job = tmp_path / "magus-vasp-acceptance"
    app.prepare_search(request, job)
    partition = os.environ.get("SCNET_PARTITION", "kshcnormal")
    ref = asyncio.run(
        app.submit(
            job_name="acceptance-vasp",
            prepared_dir=job,
            resource=ResourceRequest(
                partition=partition,
                nodes=1,
                tasks_per_node=2,
                walltime_minutes=10,
            ),
        )
    )

    async def wait_terminal() -> HPCJobState:
        state = HPCJobState.SUBMITTED
        deadline = time.monotonic() + 25 * 60
        while not state.terminal and time.monotonic() < deadline:
            await asyncio.sleep(20)
            state = await app.status(ref.job_id)
        return state

    state = asyncio.run(wait_terminal())
    assert state == HPCJobState.COMPLETED, f"job {ref.job_id} ended {state.value}"
    results_dir = tmp_path / "results"
    report = asyncio.run(app.collect(job_ref=ref, local_dir=results_dir))
    artifact_names = {Path(a["name"]).name for a in report["artifacts"]}
    # VASP must actually have run: OUTCAR / vasprun.xml / OSZICAR exist in
    # the downloaded tree (Slurm COMPLETED alone is not evidence).
    local_vasp_evidence = {
        path.name
        for path in results_dir.rglob("*")
        if path.is_file() and path.name in {"OUTCAR", "vasprun.xml", "OSZICAR"}
    }
    assert (
        artifact_names & {"OUTCAR", "vasprun.xml", "OSZICAR"}
        or local_vasp_evidence
    ), "no VASP execution evidence (OUTCAR/vasprun.xml/OSZICAR)"
    assert (
        report["candidate_count"] is not None and report["candidate_count"] >= 1
    ), "MAGUS reported no candidate structure"
    print(
        "MAGUS+VASP acceptance: job",
        ref.job_id,
        "state",
        state.value,
        "candidates",
        report["candidate_count"],
        "vasp_evidence",
        sorted(local_vasp_evidence or (artifact_names & {"OUTCAR", "vasprun.xml", "OSZICAR"})),
    )
