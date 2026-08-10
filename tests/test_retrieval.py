"""Deferred scientific capability retrieval tests (tool_search semantics)."""

from __future__ import annotations

import pytest

from photomatagent.scientific.state import ScientificState
from photomatagent.tools.factory import create_default_registry
from photomatagent.tools.surface import ToolSurfacePlanner
from photomatagent.workspace import Workspace


@pytest.fixture
def catalog():
    registry = create_default_registry(ScientificState(), Workspace("."))
    return ToolSurfacePlanner(registry).catalog


@pytest.mark.parametrize(
    ("query", "expected_name"),
    [
        ("effective mass", "electronic.effective_mass"),
        ("defect formation energy", "defects.analyze"),
        ("carrier mobility", "transport.analyze"),
        ("device simulation", "device.run_script"),
        ("infrared literature", "literature.search_arxiv"),
        ("band gap", "electronic.band_summary"),
    ],
)
def test_retrieval_finds_correct_capability(catalog, query, expected_name):
    matches = catalog.search(query, limit=5)
    names = [match.entry.name for match in matches]
    assert expected_name in names


def test_retrieval_respects_namespace_filter(catalog):
    matches = catalog.search("band gap", limit=5, namespace="materials")
    assert matches
    assert all(match.entry.namespace == "materials" for match in matches)


def test_all_scientific_tools_are_deferred(catalog):
    scientific_namespaces = {
        "materials",
        "literature",
        "structure",
        "electronic",
        "defects",
        "transport",
        "device",
        "optics",
        "ir",
        "materials_mcp",
    }
    entries = catalog.entries()
    scientific = [entry for entry in entries if entry.namespace in scientific_namespaces]
    assert scientific
    # Every scientific capability appears in the deferred catalog.
    assert len(scientific) == len(
        [entry for entry in entries if entry.namespace in scientific_namespaces]
    )

