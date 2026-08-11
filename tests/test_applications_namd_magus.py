"""Offline tests for Hefei-NAMD and MAGUS application adapters."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from photomatagent.scientific.applications.magus.application import (
    MagusApplication,
)
from photomatagent.scientific.applications.namd.application import (
    NamdApplication,
)


def make_namd_tree(root: Path, *, wavecar_sizes: list[int] | None = None) -> Path:
    """Build a VASP AIMD trajectory tree in Hefei-NAMD layout."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "POSCAR").write_text("reference\n", encoding="utf-8")
    (root / "XDATCAR").write_text("trajectory\n", encoding="utf-8")
    (root / "OUTCAR").write_text("metadata\n", encoding="utf-8")
    for index, size in enumerate(
        wavecar_sizes or [1024, 1024], start=1
    ):
        snapshot = root / f"{index:04d}"
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "POSCAR").write_text("frame\n", encoding="utf-8")
        (snapshot / "OUTCAR").write_text("meta\n", encoding="utf-8")
        (snapshot / "WAVECAR").write_bytes(b"x" * size)
    return root


# -- Hefei-NAMD ----------------------------------------------------------------


def test_namd_validate_inputs_valid_tree(tmp_path):
    root = make_namd_tree(tmp_path / "traj")
    app = NamdApplication()
    assert app.validate_inputs(root) == []


def test_namd_validate_inputs_missing_files(tmp_path):
    root = make_namd_tree(tmp_path / "traj")
    (root / "XDATCAR").unlink()
    problems = NamdApplication().validate_inputs(root)
    assert any("XDATCAR" in problem for problem in problems)


def test_namd_validate_inputs_rejects_wavecar_size_mismatch(tmp_path):
    root = make_namd_tree(tmp_path / "traj", wavecar_sizes=[1024, 2048])
    problems = NamdApplication().validate_inputs(root)
    assert any("WAVECAR sizes differ" in problem for problem in problems)


def test_namd_prepare_writes_manifest_without_fabricating_inp(tmp_path):
    root = make_namd_tree(tmp_path / "traj")
    out = tmp_path / "prepared"
    manifest = NamdApplication().prepare(trajectory_dir=root, output_dir=out)
    assert manifest["status"] == "PREPARED"
    assert manifest["runtime_inputs"]["inp"] == "NOT_GENERATED"
    assert "inicon" in manifest["runtime_inputs"]
    assert len(manifest["snapshots"]) == 2
    assert (out / "namd_manifest.json").is_file()


def test_namd_submit_requires_module(tmp_path):
    from photomatagent.scientific.remote.fake import FakeSCNetBackend

    root = make_namd_tree(tmp_path / "traj")
    out = tmp_path / "prepared"
    app = NamdApplication(backend=FakeSCNetBackend(), module_name="")
    app.prepare(trajectory_dir=root, output_dir=out)

    async def scenario():
        with pytest.raises(ValueError, match="module"):
            await app.submit(job_name="x", prepared_dir=out)

    asyncio.run(scenario())


def test_namd_render_slurm_with_module():
    app = NamdApplication(module_name="hefei-namd/1.0")
    script = app.render_slurm(
        job_name="na-md",
        resource=__import__(
            "photomatagent.scientific.remote.models", fromlist=["ResourceRequest"]
        ).ResourceRequest(),
    )
    assert "module load hefei-namd/1.0" in script
    assert "srun --mpi=pmi2 namd" in script


def test_namd_probe_unconfigured_without_backend():
    report = NamdApplication().probe_environment()
    assert report["status"] == "UNCONFIGURED"
    assert "required_vasp_artifacts" in report
    assert any("WAVECAR" in item for item in report["required_vasp_artifacts"])


def test_namd_tools_all_deferred():
    from photomatagent.scientific.applications.namd.tools import (
        NamdCapabilityPack,
    )

    pack = NamdCapabilityPack()
    names = [tool.name for tool in pack.tools()]
    for expected in (
        "namd.capabilities",
        "namd.validate_inputs",
        "namd.prepare",
        "namd.submit",
        "namd.status",
        "namd.collect",
        "namd.inspect_result",
    ):
        assert expected in names
    assert all(tool.exposure.value == "deferred" for tool in pack.tools())
    assert pack.probe().status.value == "UNCONFIGURED"


# -- MAGUS ---------------------------------------------------------------------


def test_magus_probe_unconfigured_when_not_installed():
    report = MagusApplication(executable="magus-definitely-not-installed").probe_environment()
    assert report["status"] == "UNCONFIGURED"
    assert report["installed"] is False
    assert "UNVALIDATED" in report["candidate_validity"]


def test_magus_prepare_manifest(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    out = tmp_path / "out"
    manifest = MagusApplication().prepare(
        search_type="bulk",
        composition="HgTe",
        target_dir=target,
        output_dir=out,
    )
    assert manifest["search_type"] == "bulk"
    assert manifest["composition"] == "HgTe"
    assert (out / "magus_manifest.json").is_file()


def test_magus_prepare_rejects_unsupported_search_type(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    app = MagusApplication(search_types=["bulk"])
    with pytest.raises(ValueError, match="not exposed"):
        app.prepare(
            search_type="surface",
            composition="HgTe",
            target_dir=target,
            output_dir=tmp_path / "out",
        )


def test_magus_tools_all_deferred():
    from photomatagent.scientific.applications.magus.tools import (
        MagusCapabilityPack,
    )

    pack = MagusCapabilityPack()
    names = [tool.name for tool in pack.tools()]
    assert "magus.capabilities" in names
    assert "magus.probe" in names
    assert "magus.search_bulk" in names
    assert "magus.search_cluster" in names
    assert "magus.search_surface" in names
    assert all(tool.exposure.value == "deferred" for tool in pack.tools())


def test_mcp_scnet_server_imports_and_doctor_runs():
    import json
    import subprocess
    import sys

    from photomatagent.mcp_servers.scnet import server

    assert server.SERVER_NAME == "scnet-science"
    # --doctor must run without any SCNet credentials
    result = subprocess.run(
        [sys.executable, "-m", "photomatagent.mcp_servers.scnet.server", "--doctor"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd="/home/shiqiany/AIagent/PhomatAgent",
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert "vasp" in report and "namd" in report and "magus" in report


def test_live_scnet_mcp_server_roundtrip():
    """Gated live test: real stdio MCP handshake with the scnet server."""
    import os

    if os.environ.get("PHOTOMATAGENT_RUN_LIVE_SCIENCE") != "1":
        import pytest

        pytest.skip("set PHOTOMATAGENT_RUN_LIVE_SCIENCE=1 for live MCP test")
    import asyncio
    import sys

    from mcp import ClientSession, StdioServerParameters, stdio_client

    async def scenario():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "photomatagent.mcp_servers.scnet.server"],
            env={"PYTHONPATH": "src"},
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = [tool.name for tool in tools.tools]
                assert "vasp_capabilities" in names
                assert "vasp_prepare" in names
                assert "namd_capabilities" in names
                result = await session.call_tool("vasp_capabilities", {})
                assert not result.isError

    asyncio.run(scenario())
