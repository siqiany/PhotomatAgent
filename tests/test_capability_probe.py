"""Capability probes must never raise and must report accurately."""

from __future__ import annotations

from pathlib import Path

from photomatagent.scientific.capabilities.config import ScientificConfig
from photomatagent.scientific.capabilities.status import probe_all_capabilities
from photomatagent.workspace import Workspace


def test_all_probes_report_without_raising(tmp_path):
    infos = probe_all_capabilities(
        config=ScientificConfig.from_environment(workspace=tmp_path),
        workspace=Workspace(tmp_path),
    )
    names = {info.name for info in infos}
    assert {
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
    } <= names
    for info in infos:
        assert info.status.value in {
            "AVAILABLE",
            "MISSING_DEPENDENCY",
            "UNCONFIGURED",
            "ERROR",
        }


def test_ir_always_available(tmp_path):
    infos = probe_all_capabilities(workspace=Workspace(tmp_path))
    ir_info = next(info for info in infos if info.name == "ir")
    assert ir_info.status.value == "AVAILABLE"
    assert any(tool == "ir.compile_constraints" for tool in ir_info.tools)


def test_materials_unconfigured_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv("MATERIALS_API_KEY", raising=False)
    infos = probe_all_capabilities(workspace=Workspace(tmp_path))
    materials = next(info for info in infos if info.name == "materials")
    assert materials.status.value == "UNCONFIGURED"


def test_materials_key_read_from_workspace_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("MATERIALS_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        "MATERIALS_API_KEY=test-key-123\n", encoding="utf-8"
    )
    config = ScientificConfig.from_environment(workspace=tmp_path)
    assert config.materials_api_key() == "test-key-123"

    infos = probe_all_capabilities(config=config, workspace=Workspace(tmp_path))
    materials = next(info for info in infos if info.name == "materials")
    assert materials.status.value == "AVAILABLE"


def test_existing_env_wins_over_workspace_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("MATERIALS_API_KEY", "from-process-env")
    (tmp_path / ".env").write_text(
        "MATERIALS_API_KEY=from-dotenv\n", encoding="utf-8"
    )
    config = ScientificConfig.from_environment(workspace=tmp_path)
    assert config.materials_api_key() == "from-process-env"


def test_embedding_vector_dimension_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTOMATAGENT_EMBEDDING_VECTOR_DIM", "768")
    config = ScientificConfig.from_environment(workspace=tmp_path)
    assert config.embedding_vector_dim == 768


def test_optics_available_here(tmp_path):
    import importlib.util

    infos = probe_all_capabilities(workspace=Workspace(tmp_path))
    optics = next(info for info in infos if info.name == "optics")
    meep_installed = importlib.util.find_spec("meep") is not None
    pytaser_installed = importlib.util.find_spec("pytaser") is not None
    if meep_installed and pytaser_installed:
        assert optics.status.value == "AVAILABLE"
    else:
        # Probe reports MISSING_DEPENDENCY listing the missing backend(s).
        assert optics.status.value == "MISSING_DEPENDENCY"
        assert "meep" in optics.detail or "pytaser" in optics.detail


def test_structure_pack_tools_are_deferred(tmp_path):
    from photomatagent.scientific.capabilities.registry import build_scientific_tools
    from photomatagent.tools.exposure import ToolExposure

    tools = build_scientific_tools(
        config=ScientificConfig.from_environment(workspace=tmp_path),
        workspace=Workspace(tmp_path),
    )
    structure_tools = [tool for tool in tools if tool.namespace == "structure"]
    assert {tool.name for tool in structure_tools} == {
        "structure.summary",
        "structure.symmetry",
        "structure.density",
        "structure.neighbors",
        "structure.convert",
    }
    assert all(tool.exposure is ToolExposure.DEFERRED for tool in structure_tools)
