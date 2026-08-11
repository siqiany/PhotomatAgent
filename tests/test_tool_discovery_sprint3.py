"""Sprint 3 tool discovery contract (spec section 90)."""

from __future__ import annotations

import pytest

from photomatagent.scientific.state import ScientificState
from photomatagent.tools.factory import create_default_registry
from photomatagent.tools.surface import ToolCatalog
from photomatagent.workspace import Workspace


@pytest.fixture(scope="module")
def catalog() -> ToolCatalog:
    registry = create_default_registry(
        ScientificState(), Workspace("/home/shiqiany/AIagent/PhomatAgent")
    )
    return ToolCatalog(registry)


def _top_names(catalog: ToolCatalog, query: str) -> list[str]:
    return [match.entry.name for match in catalog.search(query, limit=8)]


def test_tool_search_finds_vasp(catalog):
    names = _top_names(catalog, "VASP")
    assert any(name.startswith("vasp.") for name in names)


def test_tool_search_finds_dft_band_structure(catalog):
    names = _top_names(catalog, "DFT band structure")
    assert any(name.startswith("vasp.") for name in names)


def test_tool_search_finds_carrier_dynamics(catalog):
    names = _top_names(catalog, "carrier dynamics")
    assert any(name.startswith("namd.") for name in names)


def test_tool_search_finds_mattergen(catalog):
    names = _top_names(catalog, "MatterGen")
    assert "generation.mattergen" in names


def test_tool_search_finds_inverse_composition(catalog):
    names = _top_names(catalog, "inverse composition")
    assert "generation.vae_formula" in names


def test_tool_search_finds_thin_film_absorption(catalog):
    names = _top_names(catalog, "thin film absorption")
    assert "optics.meep_thinfilm" in names


def test_tool_search_finds_structure_search(catalog):
    names = _top_names(catalog, "structure search")
    assert any(name.startswith("magus.") for name in names)


def test_tool_search_never_returns_remote_shell(catalog):
    names = _top_names(catalog, "run shell command on scnet")
    assert not any("shell" in name for name in names)
    assert not any(name.startswith("scnet.") for name in names)
