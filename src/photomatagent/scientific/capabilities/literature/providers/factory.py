"""Construct explicitly selected literature RAG providers."""

from __future__ import annotations

import os

from photomatagent.scientific.capabilities.config import ScientificConfig

from .base import EmbeddingProvider, RagProviderError, RerankerProvider
from .external import (
    CohereCompatibleRerankerProvider,
    OpenAICompatibleEmbeddingProvider,
)
from .local import (
    DisabledReranker,
    LocalCrossEncoderProvider,
    LocalSentenceTransformerProvider,
)


def _require_external_allowed(config: ScientificConfig) -> None:
    if not config.rag_allow_external:
        raise RagProviderError(
            "external_provider_not_allowed",
            "external RAG providers require explicit authorization",
        )


def _external_key(config: ScientificConfig, env_name: str) -> str:
    name = env_name.strip()
    if not name:
        raise RagProviderError(
            "external_api_key_env_missing",
            "external provider API key environment name is required",
        )
    value = os.environ.get(name, "").strip()
    if not value:
        raise RagProviderError(
            "external_api_key_missing",
            "external provider API key is not configured",
        )
    return value


def _require_base_url(base_url: str) -> str:
    value = base_url.strip()
    if not value:
        raise RagProviderError(
            "external_base_url_missing",
            "external provider base URL is not configured",
        )
    return value


def build_embedding_provider(config: ScientificConfig) -> EmbeddingProvider:
    """Build exactly the embedding provider named by ``config``."""
    if config.embedding_provider == "local":
        return LocalSentenceTransformerProvider(
            config.embedding_model,
            config.embedding_vector_dim,
            batch_size=config.rag_batch_size,
        )
    if config.embedding_provider == "openai_compatible":
        _require_external_allowed(config)
        api_key = _external_key(config, config.embedding_api_key_env)
        base_url = _require_base_url(config.embedding_base_url)
        return OpenAICompatibleEmbeddingProvider(
            config.embedding_model,
            config.embedding_vector_dim,
            base_url=base_url,
            api_key=api_key,
            batch_size=config.rag_batch_size,
            timeout_seconds=config.qdrant_timeout_seconds,
        )
    raise RagProviderError(
        "embedding_provider_unknown",
        "unknown embedding provider",
    )


def build_reranker_provider(config: ScientificConfig) -> RerankerProvider:
    """Build exactly the reranker provider named by ``config``."""
    if config.reranker_provider == "local":
        return LocalCrossEncoderProvider(
            config.reranker_model,
            batch_size=config.rag_batch_size,
        )
    if config.reranker_provider == "cohere_compatible":
        _require_external_allowed(config)
        api_key = _external_key(config, config.reranker_api_key_env)
        base_url = _require_base_url(config.reranker_base_url)
        return CohereCompatibleRerankerProvider(
            config.reranker_model,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=config.qdrant_timeout_seconds,
        )
    if config.reranker_provider == "disabled":
        return DisabledReranker(config.reranker_model)
    raise RagProviderError(
        "reranker_provider_unknown",
        "unknown reranker provider",
    )


__all__ = ["build_embedding_provider", "build_reranker_provider"]
