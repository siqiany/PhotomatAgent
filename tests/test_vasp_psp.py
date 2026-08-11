"""Offline tests for the pseudopotential layout resolver (Sprint 4, §78-79).

Uses fake tiny files only -- no real POTCAR content anywhere.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from photomatagent.scientific.applications.vasp.application import VaspApplication
from photomatagent.scientific.applications.vasp.psp import (
    local_potcar_check,
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
