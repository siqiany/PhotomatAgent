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


def test_mock_tool_is_not_discoverable_via_search(catalog):
    # mock.run_calculation is a test-only placeholder; it must never rank
    # ahead of real capabilities (e.g. electronic.band_summary) in searches.
    for query in ("band gap calculation", "mock scientific calculation", "dos"):
        names = [match.entry.name for match in catalog.search(query, limit=10)]
        assert "mock.run_calculation" not in names, query


def test_mock_tool_still_describable_and_outside_manifest(catalog):
    # tool_describe by exact name must keep working (offline smoke flow),
    # but the manifest sent to the model must not advertise the mock.
    entry = catalog.get("mock.run_calculation")
    assert entry is not None
    assert entry.name == "mock.run_calculation"
    assert "mock.run_calculation" not in [e.name for e in catalog.entries()]


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
