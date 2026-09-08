"""Capability probes must never raise and must report accurately."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from photomatagent.scientific.capabilities.config import ScientificConfig
from photomatagent.scientific.capabilities.literature import LiteratureProbe
from photomatagent.scientific.capabilities.literature.qdrant_store import (
    CollectionGeneration,
    QdrantStoreError,
    collection_fingerprint,
)
from photomatagent.scientific.capabilities.literature.providers.factory import (
    build_embedding_provider,
)
from photomatagent.scientific.capabilities.status import probe_all_capabilities
from photomatagent.workspace import Workspace


class _ProbeClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_collections(self):
        return SimpleNamespace(collections=[])

    def get_aliases(self):
        return SimpleNamespace(
            aliases=[
                SimpleNamespace(
                    alias_name="photomat_literature_documents_current",
                    collection_name="photomat_literature_documents_123456789abc",
                ),
                SimpleNamespace(
                    alias_name="photomat_literature_passages_current",
                    collection_name="photomat_literature_passages_123456789abc",
                ),
            ]
        )

    def get_collection_aliases(self, *args, **kwargs):
        del args, kwargs
        return self.get_aliases()

    def info(self):
        return SimpleNamespace(version="1.18.2")

    def close(self):
        return None


def _patch_probe_client(monkeypatch):
    import qdrant_client

    monkeypatch.setattr(qdrant_client, "QdrantClient", _ProbeClient)


def _ready_generation(config: ScientificConfig) -> CollectionGeneration:
    embedding = build_embedding_provider(config)
    fingerprint = collection_fingerprint(
        embedding.identity,
        1,
        prefix=config.qdrant_collection_prefix,
    )
    return CollectionGeneration(
        fingerprint=fingerprint,
        documents_physical=(
            f"{config.qdrant_collection_prefix}_documents_{fingerprint[:12]}"
        ),
        passages_physical=(
            f"{config.qdrant_collection_prefix}_passages_{fingerprint[:12]}"
        ),
        documents_alias=f"{config.qdrant_collection_prefix}_documents_current",
        passages_alias=f"{config.qdrant_collection_prefix}_passages_current",
    )


def _patch_probe_store(monkeypatch, config: ScientificConfig, *, error=None, missing=False):
    generation = _ready_generation(config)
    calls: list[str] = []

    class FakeStore:
        prefix = config.qdrant_collection_prefix
        sparse_model = "qdrant/bm25"

        async def resolve_current_generation(self):
            calls.append("resolve")
            if error is not None:
                raise error
            return None if missing else generation

        async def validate_current_generation(self, expected_fingerprint):
            calls.append(f"validate:{expected_fingerprint}")
            if error is not None:
                raise error
            assert expected_fingerprint == generation.fingerprint

    class FakeStoreFactory:
        @classmethod
        def from_config(cls, config):
            del config
            return FakeStore()

    qdrant_store_module = __import__(
        "photomatagent.scientific.capabilities.literature.qdrant_store",
        fromlist=["QdrantLiteratureStore"],
    )
    monkeypatch.setattr(qdrant_store_module, "QdrantLiteratureStore", FakeStoreFactory)
    return calls


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
    monkeypatch.setenv("PHOTOMATAGENT_RAG_EMBEDDING_VECTOR_DIM", "768")
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


def test_literature_tools_remain_deferred_when_qdrant_is_unavailable(tmp_path):
    from photomatagent.scientific.capabilities.literature import literature_pack
    from photomatagent.tools.exposure import ToolExposure

    pack = literature_pack(
        ScientificConfig.from_environment(workspace=tmp_path), Workspace(tmp_path)
    )
    names = {tool.name for tool in pack.tools()}
    assert {
        "literature.index_papers",
        "literature.search_passages",
        "literature.read_passage",
        "literature.extract_evidence",
    } <= names
    assert all(tool.exposure is ToolExposure.DEFERRED for tool in pack.tools())


def test_literature_probe_rejects_incomplete_external_provider_config(
    tmp_path, monkeypatch
):
    source_root = tmp_path / "dataset" / "paper"
    source_root.mkdir(parents=True)
    _patch_probe_client(monkeypatch)
    config = ScientificConfig(
        literature_root="dataset/paper",
        embedding_provider="openai_compatible",
        embedding_model="embedding-test",
        embedding_base_url="",
        embedding_api_key_env="MISSING_EMBEDDING_KEY",
        rag_allow_external=True,
    )

    result = LiteratureProbe(config, Workspace(tmp_path)).probe()

    assert result.status.value in {"UNCONFIGURED", "ERROR"}
    assert "available" not in result.detail.casefold()
    assert "missing" in result.detail.casefold() or "base" in result.detail.casefold()


def test_literature_probe_reports_server_version_and_validates_generation(
    tmp_path, monkeypatch
):
    source_root = tmp_path / "dataset" / "paper"
    source_root.mkdir(parents=True)
    _patch_probe_client(monkeypatch)
    config = ScientificConfig(literature_root="dataset/paper")
    calls = _patch_probe_store(monkeypatch, config)

    result = LiteratureProbe(config, Workspace(tmp_path)).probe()

    assert result.status.value == "AVAILABLE"
    assert "qdrant-server=1.18.2" in result.version
    assert "alias" in result.detail.casefold()
    assert "generation" in result.detail.casefold()
    assert calls[0] == "resolve"
    assert any(call.startswith("validate:") for call in calls)


def test_literature_probe_marks_missing_alias_generation_unconfigured(
    tmp_path, monkeypatch
):
    source_root = tmp_path / "dataset" / "paper"
    source_root.mkdir(parents=True)
    _patch_probe_client(monkeypatch)
    config = ScientificConfig(literature_root="dataset/paper")
    calls = _patch_probe_store(monkeypatch, config, missing=True)

    result = LiteratureProbe(config, Workspace(tmp_path)).probe()

    assert result.status.value == "UNCONFIGURED"
    assert "alias" in result.detail.casefold() or "generation" in result.detail.casefold()
    assert calls == ["resolve"]


@pytest.mark.parametrize("error_code", ["schema_mismatch", "model_fingerprint_mismatch"])
def test_literature_probe_rejects_schema_or_fingerprint_mismatch(
    tmp_path, monkeypatch, error_code
):
    source_root = tmp_path / "dataset" / "paper"
    source_root.mkdir(parents=True)
    _patch_probe_client(monkeypatch)
    config = ScientificConfig(literature_root="dataset/paper")
    _patch_probe_store(
        monkeypatch,
        config,
        error=QdrantStoreError(error_code, "invalid current generation"),
    )

    result = LiteratureProbe(config, Workspace(tmp_path)).probe()

    assert result.status.value == "ERROR"
    assert error_code in result.detail
