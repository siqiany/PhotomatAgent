"""Strict configuration tests for the Qdrant literature RAG surface."""

from __future__ import annotations

from dataclasses import fields

import pytest

from photomatagent.scientific.capabilities.config import ScientificConfig


def test_rag_defaults_are_local(tmp_path, monkeypatch):
    monkeypatch.delenv("PHOTOMATAGENT_RAG_ALLOW_EXTERNAL", raising=False)
    config = ScientificConfig.from_environment(workspace=tmp_path)
    assert config.qdrant_url == "http://127.0.0.1:6333"
    assert config.rag_allow_external is False
    assert config.embedding_provider == "local"
    assert config.embedding_model == "intfloat/multilingual-e5-small"
    assert config.embedding_vector_dim == 384
    assert config.reranker_provider == "local"
    assert config.rag_batch_size == 128


def test_invalid_rag_batch_size_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTOMATAGENT_RAG_BATCH_SIZE", "0")
    with pytest.raises(ValueError, match="PHOTOMATAGENT_RAG_BATCH_SIZE"):
        ScientificConfig.from_environment(workspace=tmp_path)


@pytest.mark.parametrize(
    ("env_name", "value", "message"),
    [
        ("PHOTOMATAGENT_QDRANT_TIMEOUT_SECONDS", "0", "between 1 and 300"),
        ("PHOTOMATAGENT_QDRANT_TIMEOUT_SECONDS", "301", "between 1 and 300"),
        ("PHOTOMATAGENT_RAG_EMBEDDING_VECTOR_DIM", "0", "between 1 and 8192"),
        ("PHOTOMATAGENT_RAG_EMBEDDING_VECTOR_DIM", "8193", "between 1 and 8192"),
        ("PHOTOMATAGENT_RAG_BATCH_SIZE", "15", "between 16 and 512"),
        ("PHOTOMATAGENT_RAG_BATCH_SIZE", "513", "between 16 and 512"),
        ("PHOTOMATAGENT_RAG_TOOL_MAX_DOCUMENTS", "0", "between 1 and 100"),
        ("PHOTOMATAGENT_RAG_TOOL_MAX_DOCUMENTS", "101", "between 1 and 100"),
        ("PHOTOMATAGENT_RAG_BATCH_SIZE", "not-an-int", "must be an integer"),
    ],
)
def test_rag_integer_bounds_are_strict(tmp_path, monkeypatch, env_name, value, message):
    monkeypatch.setenv(env_name, value)
    with pytest.raises(ValueError, match=env_name):
        ScientificConfig.from_environment(workspace=tmp_path)
    # Keep the assertion useful if an implementation includes the explanatory
    # detail after the required environment variable name.
    with pytest.raises(ValueError, match=message):
        ScientificConfig.from_environment(workspace=tmp_path)


def test_rag_environment_mapping_is_explicit(tmp_path, monkeypatch):
    values = {
        "PHOTOMATAGENT_QDRANT_URL": "http://qdrant.internal:6333",
        "PHOTOMATAGENT_QDRANT_API_KEY_ENV": "QDRANT_TEST_KEY",
        "PHOTOMATAGENT_QDRANT_COLLECTION_PREFIX": "papers_test",
        "PHOTOMATAGENT_QDRANT_TIMEOUT_SECONDS": "42",
        "PHOTOMATAGENT_RAG_ALLOW_EXTERNAL": "true",
        "PHOTOMATAGENT_RAG_EMBEDDING_PROVIDER": "openai_compatible",
        "PHOTOMATAGENT_RAG_EMBEDDING_MODEL": "text-embedding-test",
        "PHOTOMATAGENT_RAG_EMBEDDING_VECTOR_DIM": "768",
        "PHOTOMATAGENT_RAG_EMBEDDING_BASE_URL": "https://embed.test/v1",
        "PHOTOMATAGENT_RAG_EMBEDDING_API_KEY_ENV": "EMBED_TEST_KEY",
        "PHOTOMATAGENT_RAG_RERANKER_PROVIDER": "cohere_compatible",
        "PHOTOMATAGENT_RAG_RERANKER_MODEL": "rerank-test",
        "PHOTOMATAGENT_RAG_RERANKER_BASE_URL": "https://rerank.test/v1",
        "PHOTOMATAGENT_RAG_RERANKER_API_KEY_ENV": "RERANK_TEST_KEY",
        "PHOTOMATAGENT_RAG_BATCH_SIZE": "256",
        "PHOTOMATAGENT_RAG_TOOL_MAX_DOCUMENTS": "17",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    config = ScientificConfig.from_environment(workspace=tmp_path)

    assert config.qdrant_url == values["PHOTOMATAGENT_QDRANT_URL"]
    assert config.qdrant_api_key_env == values["PHOTOMATAGENT_QDRANT_API_KEY_ENV"]
    assert config.qdrant_collection_prefix == values["PHOTOMATAGENT_QDRANT_COLLECTION_PREFIX"]
    assert config.qdrant_timeout_seconds == 42
    assert config.rag_allow_external is True
    assert config.embedding_provider == values["PHOTOMATAGENT_RAG_EMBEDDING_PROVIDER"]
    assert config.embedding_model == values["PHOTOMATAGENT_RAG_EMBEDDING_MODEL"]
    assert config.embedding_vector_dim == 768
    assert config.embedding_base_url == values["PHOTOMATAGENT_RAG_EMBEDDING_BASE_URL"]
    assert config.embedding_api_key_env == values["PHOTOMATAGENT_RAG_EMBEDDING_API_KEY_ENV"]
    assert config.reranker_provider == values["PHOTOMATAGENT_RAG_RERANKER_PROVIDER"]
    assert config.reranker_model == values["PHOTOMATAGENT_RAG_RERANKER_MODEL"]
    assert config.reranker_base_url == values["PHOTOMATAGENT_RAG_RERANKER_BASE_URL"]
    assert config.reranker_api_key_env == values["PHOTOMATAGENT_RAG_RERANKER_API_KEY_ENV"]
    assert config.rag_batch_size == 256
    assert config.rag_tool_max_documents == 17


def test_literature_index_dir_is_not_a_rag_configuration_field(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTOMATAGENT_LITERATURE_INDEX_DIR", "legacy-index")
    config = ScientificConfig.from_environment(workspace=tmp_path)
    assert "literature_index_dir" not in {item.name for item in fields(config)}
    assert not hasattr(config, "literature_index_dir")


@pytest.mark.parametrize(
    ("env_name", "value"),
    [
        ("PHOTOMATAGENT_MATERIALS_MAX_RESULTS", "not-an-int"),
        ("PHOTOMATAGENT_MATERIALS_MAX_RESULTS", "0"),
        ("PHOTOMATAGENT_LITERATURE_MAX_PAPERS", "11"),
        ("PHOTOMATAGENT_LITERATURE_MAX_CHARS", "199"),
        ("PHOTOMATAGENT_LITERATURE_TOP_K", "0"),
        ("PHOTOMATAGENT_LITERATURE_PASSAGE_CHARS", "601"),
    ],
)
def test_model_visible_output_limits_are_strictly_bounded(
    tmp_path, monkeypatch, env_name, value
):
    monkeypatch.setenv(env_name, value)
    with pytest.raises(ValueError, match=env_name):
        ScientificConfig.from_environment(workspace=tmp_path)
