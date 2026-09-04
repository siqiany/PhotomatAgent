"""Offline tests for the pseudopotential layout resolver (Sprint 4, §78-79).

Uses fake tiny files only -- no real POTCAR content anywhere.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from photomatagent.scientific.applications.vasp.application import VaspApplication
from photomatagent.scientific.applications.vasp.psp import (
    is_safe_potcar_symbol,
    local_potcar_check,
    potcar_element_name,
    remote_potcar_check,
    resolve_local_psp_library,
    resolve_remote_psp_library,
)
from photomatagent.scientific.remote.fake import FakeSCNetBackend
from photomatagent.scientific.remote.models import ResourceRequest


def make_library(root: Path, layout: str, elements: list[str]) -> Path:
    """Build a fake psp tree; returns the library directory (children are
    ``<setup>/POTCAR``)."""
    if layout == "direct":
        library = root
    else:
        library = root / layout
    for element in elements:
        (library / element).mkdir(parents=True, exist_ok=True)
        (library / element / "POTCAR").write_text("FAKE POTCAR\n", encoding="utf-8")
    return library


def make_input_dir(root: Path, symbols: list[str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    lines = ["# POTCAR policy"] + [
        f"  {symbol}: resolved" for symbol in symbols
    ]
    (root / "POTCAR.policy").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def test_resolve_local_direct_layout(tmp_path):
    library = make_library(tmp_path / "potpaw_PBE", "direct", ["In", "As"])
    resolved = resolve_local_psp_library(library)
    assert resolved is not None
    resolved_library, layout = resolved
    assert layout == "direct"
    assert resolved_library == library.resolve()


def test_resolve_local_potpaw_pbe_layout(tmp_path):
    make_library(tmp_path / "psp", "potpaw_PBE", ["In", "As"])
    resolved = resolve_local_psp_library(tmp_path / "psp")
    assert resolved is not None
    resolved_library, layout = resolved
    assert layout == "potpaw_PBE"
    assert resolved_library == (tmp_path / "psp" / "potpaw_PBE").resolve()


def test_resolve_local_legacy_potpaw_pbe64_layout(tmp_path):
    make_library(tmp_path / "psp", "potpaw_PBE.64", ["In"])
    resolved = resolve_local_psp_library(tmp_path / "psp")
    assert resolved is not None
    resolved_library, layout = resolved
    assert layout == "potpaw_PBE.64"
    assert resolved_library == (tmp_path / "psp" / "potpaw_PBE.64").resolve()


def test_resolve_local_none_when_missing(tmp_path):
    assert resolve_local_psp_library(tmp_path / "nothing") is None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert resolve_local_psp_library(empty) is None


def test_local_potcar_check_missing(tmp_path):
    library = make_library(tmp_path / "psp", "potpaw_PBE", ["In"])
    assert local_potcar_check(library, [("In", ""), ("As", "")]) == ["As"]
    assert local_potcar_check(tmp_path / "absent", [("In", "")]) == ["In"]


def test_vasp_resolve_potcar_potpaw_pbe(tmp_path):
    make_library(tmp_path / "psp", "potpaw_PBE", ["In", "As"])
    inputs = make_input_dir(tmp_path / "inputs", ["In", "As"])
    app = VaspApplication(psp_dir=str(tmp_path / "psp"))
    potcar = app.resolve_potcar(inputs)
    assert potcar is not None
    assert potcar.read_text(encoding="utf-8").count("FAKE POTCAR") == 2


def test_vasp_resolve_potcar_legacy_pbe64(tmp_path):
    make_library(tmp_path / "psp", "potpaw_PBE.64", ["In"])
    inputs = make_input_dir(tmp_path / "inputs", ["In"])
    app = VaspApplication(psp_dir=str(tmp_path / "psp"))
    potcar = app.resolve_potcar(inputs)
    assert potcar is not None


def test_vasp_render_slurm_remote_psp_layout_detection():
    app = VaspApplication(remote_psp_dir="~/pbe")
    script = app.render_slurm(
        job_name="vasp-test",
        profile=__import__(
            "photomatagent.scientific.applications.vasp.profiles",
            fromlist=["get_profile"],
        ).get_profile("standard_semiconductor"),
        resource=ResourceRequest(),
        potcar_symbols=["In", "As"],
    )
    assert "potpaw_PBE" in script
    assert "potpaw_PBE.64" in script
    assert 'for cand in "$psp_base" "$psp_base/potpaw_PBE" ' in script
    assert 'cat "$psp_lib/$sym/POTCAR" >> POTCAR' in script
    assert "$psp_base/potpaw_PBE.64" in script


def test_vasp_probe_reports_psp_layout(tmp_path):
    make_library(tmp_path / "psp", "potpaw_PBE", ["In"])
    app = VaspApplication(psp_dir=str(tmp_path / "psp"))
    report = app.probe_environment()
    assert report["psp_layout_local"] == "potpaw_PBE"
    assert report["psp_dir_local"] == str((tmp_path / "psp" / "potpaw_PBE").resolve())


def test_resolve_remote_psp_layouts():
    backend = FakeSCNetBackend()
    backend.add_ssh_script("test -f /pbe/In/POTCAR", "", ok=False)
    backend.add_ssh_script("test -f /pbe/potpaw_PBE/In/POTCAR", "OK")
    backend.add_ssh_script("test -f /pbe/potpaw_PBE.64/In/POTCAR", "", ok=False)

    async def scenario():
        resolved = await resolve_remote_psp_library("/pbe", backend)
        assert resolved == ("/pbe/potpaw_PBE", "potpaw_PBE")

    asyncio.run(scenario())


def test_resolve_remote_direct_layout():
    backend = FakeSCNetBackend()
    backend.add_ssh_script("test -f /lib/In/POTCAR", "OK")

    async def scenario():
        resolved = await resolve_remote_psp_library("/lib", backend)
        assert resolved == ("/lib", "direct")

    asyncio.run(scenario())


def test_resolve_remote_legacy_pbe64_layout():
    backend = FakeSCNetBackend()
    backend.add_ssh_script("test -f /lib/In/POTCAR", "", ok=False)
    backend.add_ssh_script("test -f /lib/potpaw_PBE/In/POTCAR", "", ok=False)
    backend.add_ssh_script("test -f /lib/potpaw_PBE.64/In/POTCAR", "OK")

    async def scenario():
        resolved = await resolve_remote_psp_library("/lib", backend)
        assert resolved == ("/lib/potpaw_PBE.64", "potpaw_PBE.64")

    asyncio.run(scenario())


def test_resolve_remote_none():
    backend = FakeSCNetBackend()
    backend.add_ssh_script("test -f", "", ok=False)

    async def scenario():
        assert await resolve_remote_psp_library("/none", backend) is None
        assert await resolve_remote_psp_library("", backend) is None

    asyncio.run(scenario())


def test_remote_potcar_check_missing_list():
    backend = FakeSCNetBackend()
    backend.add_ssh_script("test -f /lib/potpaw_PBE/In/POTCAR", "OK")
    backend.add_ssh_script("test -f /lib/potpaw_PBE/As/POTCAR", "", ok=False)

    async def scenario():
        missing = await remote_potcar_check(
            backend, "/lib", [("In", ""), ("As", "")],
        )
        assert missing == ["As"]

    asyncio.run(scenario())


def test_potcar_element_name_maps_barium_to_semicore_setup():
    """VASP PBE libraries ship no plain ``Ba``; it maps to ``Ba_sv``."""
    assert potcar_element_name("Ba") == "Ba_sv"
    assert potcar_element_name("Ti") == "Ti"
    assert potcar_element_name("Te") == "Te"


def test_is_safe_potcar_symbol_accepts_setups_but_rejects_path_tricks():
    assert is_safe_potcar_symbol("Ba")
    assert is_safe_potcar_symbol("Ba_sv")
    assert is_safe_potcar_symbol("Ti_pv")
    for unsafe in (
        "Ba/../evil",
        "../Ba",
        "Ba sv",
        "Ba;rm",
        "Ba$X",
        "",
        ".",
    ):
        assert not is_safe_potcar_symbol(unsafe), unsafe


@pytest.mark.asyncio
async def test_preflight_reports_unresolvable_potcar(tmp_path):
    """A periodic workflow whose POTCAR cannot be supplied must fail
    preflight with a clear message instead of submitting a doomed job."""
    from pymatgen.core import Lattice, Structure

    from photomatagent.scientific.applications.vasp.unified.fingerprints import (
        scientific_fingerprint,
    )
    from photomatagent.scientific.applications.vasp.unified.models import (
        PeriodicScientificSpec,
        UnifiedStage,
        UnifiedVaspManifest,
        VaspWorkflowKind,
    )
    from photomatagent.scientific.applications.vasp.unified.periodic import (
        PeriodicVaspExecutor,
    )

    structure = Structure(
        Lattice.cubic(4.3),
        ["Ba", "Ti", "Te", "Te", "Te"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25],
         [0.75, 0.75, 0.75], [0, 0.5, 0.5]],
    )
    cif = tmp_path / "BaTiTe3.cif"
    structure.to(filename=str(cif), fmt="cif")
    (tmp_path / "source").mkdir(exist_ok=True)
    (tmp_path / "source" / "structure.cif").write_bytes(cif.read_bytes())

    app = VaspApplication(workspace=tmp_path)
    app.generator.psp_dir = tmp_path / "no_such_psp"  # local unresolvable
    app.remote_psp_dir = ""  # remote unconfigured
    executor = PeriodicVaspExecutor.__new__(PeriodicVaspExecutor)
    executor.application = app

    spec = PeriodicScientificSpec(
        structure_path="source/structure.cif",
        profile="narrow_gap_soc",
        scientific_overrides={},
    )
    manifest = UnifiedVaspManifest(
        workflow_id="vasp_0123456789abcdef",
        workflow_kind=VaspWorkflowKind.PERIODIC,
        scientific_spec=spec,
        scientific_fingerprint=scientific_fingerprint(spec),
        stages=[UnifiedStage(name="relax")],
    )
    result = await executor.preflight(manifest)
    assert not result.passed
    assert any("POTCAR cannot be resolved" in problem for problem in result.errors)

    # With a remote PSP directory configured the same workflow becomes ready.
    app.remote_psp_dir = "/public/home/scniv4a4go/potpaw_PBE"
    assert (await executor.preflight(manifest)).passed


def test_resolve_potcar_leaves_no_empty_file_when_unresolvable(tmp_path):
    """A missing setup must not leave a torn/empty POTCAR behind."""
    from photomatagent.scientific.applications.vasp.psp import (
        resolve_local_psp_library,
    )

    app = VaspApplication(workspace=tmp_path)
    stage = make_input_dir(tmp_path / "stage", ["Ba_sv", "Ti"])
    fake_library = make_library(
        tmp_path / "fake", "direct", ["In", "Ti"]  # probe element + Ti; Ba_sv absent
    )
    app.generator.psp_dir = tmp_path / "fake"
    assert resolve_local_psp_library(app.generator.psp_dir) is not None
    assert app.resolve_potcar(stage) is None
    assert not (stage / "POTCAR").exists()
