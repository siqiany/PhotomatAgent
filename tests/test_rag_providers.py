"""Contract tests for local and explicitly-gated external RAG providers."""

from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace

import pytest

from photomatagent.scientific.capabilities.config import ScientificConfig
from photomatagent.scientific.capabilities.literature.providers.base import (
    ModelIdentity,
    RagProviderError,
    RerankScore,
    validate_vectors,
)
from photomatagent.scientific.capabilities.literature.providers.external import (
    CohereCompatibleRerankerProvider,
    OpenAICompatibleEmbeddingProvider,
)
from photomatagent.scientific.capabilities.literature.providers.factory import (
    build_embedding_provider,
    build_reranker_provider,
)
from photomatagent.scientific.capabilities.literature.providers.local import (
    DisabledReranker,
    LocalCrossEncoderProvider,
    LocalSentenceTransformerProvider,
)


def test_model_identity_fingerprint_material_is_stable():
    identity = ModelIdentity(
        provider="local",
        model="model-id",
        dimension=3,
        document_prefix="passage: ",
        query_prefix="query: ",
        normalize=True,
    )
    assert identity.fingerprint_material() == {
        "provider": "local",
        "model": "model-id",
        "dimension": 3,
        "document_prefix": "passage: ",
        "query_prefix": "query: ",
        "normalize": True,
    }


@pytest.mark.parametrize(
    ("vectors", "expected_count", "expected_dimension", "code"),
    [
        ([[1.0, 2.0]], 2, 2, "vector_count_mismatch"),
        ([[1.0], [2.0, 3.0]], 2, 2, "vector_dimension_mismatch"),
        ([[math.nan]], 1, 1, "vector_non_finite"),
        ([[math.inf]], 1, 1, "vector_non_finite"),
    ],
)
def test_validate_vectors_rejects_invalid_shape_or_values(
    vectors, expected_count, expected_dimension, code
):
    with pytest.raises(RagProviderError) as exc:
        validate_vectors(vectors, expected_count, expected_dimension)
    assert exc.value.code == code


def test_validate_vectors_normalizes_numeric_rows():
    assert validate_vectors([[1, 2], (3, 4)], 2, 2) == [[1.0, 2.0], [3.0, 4.0]]


def test_external_embedding_requires_hard_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTOMATAGENT_RAG_EMBEDDING_PROVIDER", "openai_compatible")
    config = ScientificConfig.from_environment(workspace=tmp_path)
    with pytest.raises(RagProviderError) as exc:
        build_embedding_provider(config)
    assert exc.value.code == "external_provider_not_allowed"


def test_external_reranker_requires_hard_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTOMATAGENT_RAG_RERANKER_PROVIDER", "cohere_compatible")
    config = ScientificConfig.from_environment(workspace=tmp_path)
    with pytest.raises(RagProviderError) as exc:
        build_reranker_provider(config)
    assert exc.value.code == "external_provider_not_allowed"


def test_external_factory_rejects_missing_key_without_secret_in_error(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PHOTOMATAGENT_RAG_ALLOW_EXTERNAL", "true")
    monkeypatch.setenv("PHOTOMATAGENT_RAG_EMBEDDING_PROVIDER", "openai_compatible")
    monkeypatch.setenv("PHOTOMATAGENT_RAG_EMBEDDING_API_KEY_ENV", "SECRET_KEY_ENV")
    monkeypatch.delenv("SECRET_KEY_ENV", raising=False)
    config = ScientificConfig.from_environment(workspace=tmp_path)
    with pytest.raises(RagProviderError) as exc:
        build_embedding_provider(config)
    assert exc.value.code == "external_api_key_missing"
    assert "SECRET_KEY_ENV" not in str(exc.value)


def test_local_embedding_prefixes_and_normalizes_through_thread(monkeypatch):
    calls: list[dict[str, object]] = []

    class FakeModel:
        def encode(self, texts, **kwargs):
            calls.append({"texts": texts, **kwargs})
            return [[3, 4], [0, 5]]

    provider = LocalSentenceTransformerProvider("fake-model", 2, batch_size=7)
    monkeypatch.setattr(provider, "_model", lambda: FakeModel())
    vectors = asyncio.run(provider.embed_documents(["one", "two"]))
    assert vectors == [[3.0, 4.0], [0.0, 5.0]]
    assert calls == [
        {
            "texts": ["passage: one", "passage: two"],
            "batch_size": 7,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
        }
    ]

    query_calls: list[list[str]] = []
    monkeypatch.setattr(
        provider,
        "_model",
        lambda: SimpleNamespace(
            encode=lambda texts, **kwargs: query_calls.append(texts) or [[1, 0]]
        ),
    )
    assert asyncio.run(provider.embed_query("hello")) == [1.0, 0.0]
    assert query_calls == [["query: hello"]]


def test_local_failure_is_not_replaced_by_an_external_provider(monkeypatch):
    provider = LocalSentenceTransformerProvider("missing-model", 2)

    def fail_loader():
        raise RuntimeError("local model unavailable")

    monkeypatch.setattr(provider, "_model", fail_loader)
    with pytest.raises(RuntimeError, match="local model unavailable"):
        asyncio.run(provider.embed_query("query"))


def test_local_cross_encoder_caps_text_and_sorts_scores(monkeypatch):
    calls: list[object] = []

    class FakeCrossEncoder:
        def predict(self, pairs, **kwargs):
            calls.append((pairs, kwargs))
            return [0.1, 0.9, 0.4]

    provider = LocalCrossEncoderProvider("fake-reranker", max_text_chars=5)
    monkeypatch.setattr(provider, "_model", lambda: FakeCrossEncoder())
    scores = asyncio.run(
        provider.rerank("query-too-long", ["first passage", "second", "third"], top_n=2)
    )
    assert scores == [RerankScore(index=1, score=0.9), RerankScore(index=2, score=0.4)]
    assert calls == [
        (
            [("query", "first"), ("query", "secon"), ("query", "third")],
            {"batch_size": 128, "show_progress_bar": False},
        )
    ]


class _FakeEmbeddings:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _FakeOpenAIClient:
    def __init__(self, responses):
        self.embeddings = _FakeEmbeddings(responses)


def test_openai_embedding_restores_response_order_and_batches():
    client = _FakeOpenAIClient(
        [
            SimpleNamespace(
                data=[
                    SimpleNamespace(index=1, embedding=[0, 2]),
                    SimpleNamespace(index=0, embedding=[1, 1]),
                ]
            ),
            SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[3, 4])]),
        ]
    )
    provider = OpenAICompatibleEmbeddingProvider(
        "embedding-test", 2, client=client, batch_size=2
    )
    vectors = asyncio.run(provider.embed_documents(["one", "two", "three"]))
    assert vectors == [[1.0, 1.0], [0.0, 2.0], [3.0, 4.0]]
    assert [call["input"] for call in client.embeddings.calls] == [
        ["one", "two"],
        ["three"],
    ]


def test_openai_embedding_retries_only_retryable_failures():
    class TransportError(OSError):
        pass

    client = _FakeOpenAIClient(
        [
            TransportError("temporary transport failure"),
            SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1, 2])]),
        ]
    )
    provider = OpenAICompatibleEmbeddingProvider("embedding-test", 2, client=client)
    assert asyncio.run(provider.embed_documents(["one"])) == [[1.0, 2.0]]
    assert len(client.embeddings.calls) == 2


def test_openai_embedding_does_not_retry_non_retryable_status():
    error = RuntimeError("bad request")
    error.status_code = 400  # type: ignore[attr-defined]
    client = _FakeOpenAIClient([error])
    provider = OpenAICompatibleEmbeddingProvider("embedding-test", 2, client=client)
    with pytest.raises(RagProviderError) as exc:
        asyncio.run(provider.embed_documents(["one"]))
    assert exc.value.code == "embedding_request_failed"
    assert len(client.embeddings.calls) == 1


def test_openai_embedding_does_not_retry_plain_runtime_error():
    client = _FakeOpenAIClient([RuntimeError("application bug")])
    provider = OpenAICompatibleEmbeddingProvider("embedding-test", 2, client=client)
    with pytest.raises(RagProviderError) as exc:
        asyncio.run(provider.embed_documents(["one"]))
    assert exc.value.code == "embedding_request_failed"
    assert len(client.embeddings.calls) == 1


def test_openai_embedding_does_not_retry_name_only_transport_exception():
    class TransportNamedApplicationError(Exception):
        pass

    client = _FakeOpenAIClient(
        [
            TransportNamedApplicationError("application failure"),
            SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1, 2])]),
        ]
    )
    provider = OpenAICompatibleEmbeddingProvider("embedding-test", 2, client=client)
    with pytest.raises(RagProviderError) as exc:
        asyncio.run(provider.embed_documents(["one"]))
    assert exc.value.code == "embedding_request_failed"
    assert len(client.embeddings.calls) == 1


class _FakeHTTPResponse:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class _FakeHTTPTransport:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def post(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _MappingHTTPTransport:
    async def post(self, **kwargs):
        return {"results": [{"index": 0, "relevance_score": 0.5}]}


def test_cohere_reranker_posts_bounded_payload_and_validates_results():
    transport = _FakeHTTPTransport(
        _FakeHTTPResponse(
            {
                "results": [
                    {"index": 1, "relevance_score": 0.2},
                    {"index": 0, "relevance_score": 0.8},
                ]
            }
        )
    )
    provider = CohereCompatibleRerankerProvider(
        "rerank-test",
        "https://rerank.test/v1",
        "secret-value",
        transport=transport,
    )
    scores = asyncio.run(provider.rerank("query", ["one", "two"], top_n=10))
    assert scores == [RerankScore(index=0, score=0.8), RerankScore(index=1, score=0.2)]
    assert transport.calls[0]["json"] == {
        "model": "rerank-test",
        "query": "query",
        "documents": ["one", "two"],
        "top_n": 2,
    }
    assert "secret-value" not in str(transport.calls[0]["json"])


def test_cohere_reranker_clamps_top_n_and_response_size():
    transport = _FakeHTTPTransport(
        _FakeHTTPResponse(
            {
                "results": [
                    {"index": 0, "relevance_score": 0.8},
                    {"index": 1, "relevance_score": 0.2},
                ]
            }
        )
    )
    provider = CohereCompatibleRerankerProvider(
        "rerank-test",
        "https://rerank.test/v1",
        "secret-value",
        transport=transport,
    )
    scores = asyncio.run(provider.rerank("query", ["one", "two"], top_n=1))
    assert len(scores) == 1
    assert scores[0] == RerankScore(index=0, score=0.8)
    assert transport.calls[0]["json"]["top_n"] == 1


def test_cohere_reranker_accepts_mapping_transport_response():
    provider = CohereCompatibleRerankerProvider(
        "rerank-test",
        "https://rerank.test/v1",
        "secret",
        transport=_MappingHTTPTransport(),
    )
    assert asyncio.run(provider.rerank("query", ["one"], top_n=1)) == [
        RerankScore(index=0, score=0.5)
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"results": [{"index": 0, "relevance_score": 0.1}, {"index": 0, "relevance_score": 0.2}]},
        {"results": [{"index": 5, "relevance_score": 0.1}]},
        {"results": [{"index": 0, "relevance_score": "not-a-number"}]},
    ],
)
def test_cohere_reranker_rejects_malformed_response(payload):
    transport = _FakeHTTPTransport(_FakeHTTPResponse(payload))
    provider = CohereCompatibleRerankerProvider(
        "rerank-test", "https://rerank.test/v1", "secret", transport=transport
    )
    with pytest.raises(RagProviderError) as exc:
        asyncio.run(provider.rerank("query", ["one"], top_n=1))
    assert exc.value.code == "reranker_invalid_response"
    assert "secret" not in str(exc.value)


def test_disabled_reranker_has_no_external_side_effect(tmp_path):
    config = ScientificConfig.from_environment(workspace=tmp_path)
    config = config.__class__(**{**config.__dict__, "reranker_provider": "disabled"})
    provider = build_reranker_provider(config)
    assert isinstance(provider, DisabledReranker)
    assert asyncio.run(provider.rerank("query", ["passage"], top_n=1)) == []
