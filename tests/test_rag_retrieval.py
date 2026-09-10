"""Bounded Qdrant literature retrieval tests.

These tests deliberately use only in-process fakes.  They verify the
application boundary without starting Qdrant, loading a corpus, or invoking a
model implementation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from types import SimpleNamespace
from typing import Any

import pytest

from photomatagent.scientific.capabilities.literature.qdrant_store import (
    SearchCandidate,
    QdrantStoreError,
    collection_fingerprint,
)
from photomatagent.scientific.capabilities.literature.providers.base import ModelIdentity
from photomatagent.scientific.capabilities.literature.models import (
    IngestState,
    LiteratureSourceKind,
    PassagePoint,
)
from photomatagent.scientific.capabilities.literature.retrieval import (
    LiteratureRetriever,
    RagRetrievalError,
)


def _hash(text: str) -> str:
    normalized = " ".join(text.split()).casefold()
    return sha256(normalized.encode("utf-8")).hexdigest()


def candidate_fixture(
    count: int = 4,
    *,
    workspace_id: str = "ws",
    document_id: str = "doc",
    text_prefix: str = "passage",
    source_kind: LiteratureSourceKind | str = LiteratureSourceKind.FULLTEXT,
) -> list[SearchCandidate]:
    candidates: list[SearchCandidate] = []
    for index in range(count):
        text = f"{text_prefix} {index}"
        payload: dict[str, Any] = {
            "workspace_id": workspace_id,
            "record_type": "passage",
            "document_id": document_id,
            "document_revision": "a" * 64,
            "ingest_state": "ready",
            "passage_id": f"p-{index}",
            "chunk_index": index,
            "text": text,
            "title": "HgTe detector",
            "authors": ["Author"],
            "year": 2024,
            "section": "Results",
            "heading_path": "Results / Detector",
            "page_start": index + 1,
            "page_end": index + 1,
            "relative_source_path": "papers/hgte.pdf",
            "previous_passage_id": f"p-{index - 1}" if index else None,
            "next_passage_id": f"p-{index + 1}" if index < count - 1 else None,
            "normalized_text_sha256": _hash(text),
            "indexed_at": datetime(2024, 1, (index % 28) + 1, tzinfo=timezone.utc),
            "source_kind": source_kind,
        }
        candidates.append(
            SearchCandidate(passage_id=f"p-{index}", score=1.0 / (index + 1), payload=payload)
        )
    return candidates


class FakeEmbedder:
    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [0.1, 0.2]
        self.calls: list[str] = []

    async def embed_query(self, query: str) -> list[float]:
        self.calls.append(query)
        return list(self.vector)


class FailingLocalEmbedder:
    async def embed_query(self, query: str) -> list[float]:
        del query
        raise RuntimeError("local embedding unavailable")


class FakeReranker:
    def __init__(self, scores: list[Any] | None = None) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str], int]] = []

    async def rerank(self, query: str, passages: list[str], *, top_n: int) -> list[Any]:
        self.calls.append((query, passages, top_n))
        if self.scores is not None:
            return self.scores
        return []


class FailingReranker:
    def __init__(self, reason: str) -> None:
        self.reason = reason

    async def rerank(self, query: str, passages: list[str], *, top_n: int) -> list[Any]:
        del query, passages, top_n
        raise RuntimeError(self.reason)


class DisabledReranker:
    async def rerank(self, query: str, passages: list[str], *, top_n: int) -> list[Any]:
        del query, passages, top_n
        return []


class FakeStore:
    def __init__(self, candidates: list[SearchCandidate] | None = None) -> None:
        self.hybrid_results = candidates if candidates is not None else candidate_fixture()
        self.dense_results = list(self.hybrid_results)
        self.sparse_results = list(self.hybrid_results)
        self.hybrid_limit: int | None = None
        self.dense_limit: int | None = None
        self.sparse_limit: int | None = None
        self.hybrid_queries = 0
        self.dense_queries = 0
        self.sparse_queries = 0
        self.neighbor_calls: list[tuple[str, list[str]]] = []
        self.neighbor_source_kinds: list[LiteratureSourceKind] = []
        self.neighbors: dict[str, Any] = {}
        self.hybrid_error: Exception | None = None
        self.dense_error: Exception | None = None
        self.sparse_error: Exception | None = None

    async def hybrid_candidates(
        self,
        query: str,
        dense: list[float],
        *,
        workspace_id: str,
        source_kind: LiteratureSourceKind,
        limit: int,
    ) -> list[SearchCandidate]:
        del query, dense, workspace_id, source_kind
        self.hybrid_queries += 1
        self.hybrid_limit = limit
        if self.hybrid_error is not None:
            raise self.hybrid_error
        return list(self.hybrid_results)

    async def dense_candidates(
        self,
        dense: list[float],
        *,
        workspace_id: str,
        source_kind: LiteratureSourceKind,
        limit: int,
    ) -> list[SearchCandidate]:
        del dense, workspace_id, source_kind
        self.dense_queries += 1
        self.dense_limit = limit
        if self.dense_error is not None:
            raise self.dense_error
        return list(self.dense_results)

    async def sparse_candidates(
        self,
        query: str,
        *,
        workspace_id: str,
        source_kind: LiteratureSourceKind,
        limit: int,
    ) -> list[SearchCandidate]:
        del query, workspace_id, source_kind
        self.sparse_queries += 1
        self.sparse_limit = limit
        if self.sparse_error is not None:
            raise self.sparse_error
        return list(self.sparse_results)

    async def retrieve_passages(
        self,
        workspace_id: str,
        passage_ids: list[str],
        *,
        source_kind: LiteratureSourceKind,
    ) -> list[Any]:
        self.neighbor_calls.append((workspace_id, list(passage_ids)))
        self.neighbor_source_kinds.append(source_kind)
        return [self.neighbors[passage_id] for passage_id in passage_ids if passage_id in self.neighbors]


@pytest.mark.asyncio
async def test_retrieval_validates_provider_generation_before_query() -> None:
    store = FakeStore()
    store.prefix = "photomat_test_retrieval"
    store.sparse_model = "qdrant/bm25"
    identity = ModelIdentity(
        provider="fixture",
        model="embedding-v1",
        dimension=2,
        document_prefix="",
        query_prefix="",
        normalize=False,
    )
    embedder = FakeEmbedder()
    embedder.identity = identity
    calls: list[str] = []

    async def validate(expected: str) -> None:
        calls.append(expected)

    store.validate_current_generation = validate  # type: ignore[attr-defined]
    await LiteratureRetriever(store, embedder, DisabledReranker()).search(
        "query", workspace_id="ws", top_k=1
    )

    assert calls == [
        collection_fingerprint(
            identity, 2, prefix="photomat_test_retrieval", sparse_model="qdrant/bm25"
        )
    ]


@pytest.mark.asyncio
async def test_retrieval_propagates_generation_fingerprint_mismatch() -> None:
    store = FakeStore()
    store.prefix = "photomat_test_retrieval_mismatch"
    store.sparse_model = "qdrant/bm25"
    embedder = FakeEmbedder()
    embedder.identity = ModelIdentity(
        provider="fixture",
        model="embedding-v1",
        dimension=2,
        document_prefix="",
        query_prefix="",
        normalize=False,
    )

    async def reject(expected: str) -> None:
        del expected
        raise QdrantStoreError(
            "model_fingerprint_mismatch", "current generation is incompatible"
        )

    store.validate_current_generation = reject  # type: ignore[attr-defined]
    with pytest.raises(QdrantStoreError, match="incompatible"):
        await LiteratureRetriever(store, embedder, DisabledReranker()).search(
            "query", workspace_id="ws", top_k=1
        )
    assert store.hybrid_queries == 0


@pytest.mark.asyncio
async def test_search_never_loads_the_corpus() -> None:
    store = FakeStore(candidate_fixture(60))
    embedder = FakeEmbedder()
    result = await LiteratureRetriever(store, embedder, DisabledReranker()).search(
        "HgTe infrared detector", workspace_id="ws", top_k=5
    )

    assert len(result.passages) == 5
    assert store.hybrid_limit == 50
    assert not hasattr(store, "all_passages")
    assert embedder.calls == ["HgTe infrared detector"]


@pytest.mark.asyncio
async def test_reranker_failure_returns_rrf_with_diagnostic() -> None:
    store = FakeStore()
    result = await LiteratureRetriever(
        store, FakeEmbedder(), FailingReranker("timeout")
    ).search("query", workspace_id="ws", top_k=3)

    assert result.passages
    assert result.diagnostics.mode == "hybrid_rrf"
    assert "reranker_unavailable" in result.diagnostics.degraded_reasons
    assert result.passages[0].score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_dense_failure_uses_sparse_without_external_fallback() -> None:
    store = FakeStore()
    result = await LiteratureRetriever(
        store, FailingLocalEmbedder(), DisabledReranker()
    ).search("query", workspace_id="ws", top_k=3)

    assert result.diagnostics.mode == "sparse_only"
    assert "dense_unavailable" in result.diagnostics.degraded_reasons
    assert store.sparse_queries == 1
    assert store.sparse_limit == 50
    assert store.hybrid_queries == 0


@pytest.mark.asyncio
async def test_retriever_rejects_wrong_source_candidates() -> None:
    store = FakeStore(candidate_fixture(source_kind=LiteratureSourceKind.ABSTRACT))
    result = await LiteratureRetriever(
        store, FakeEmbedder(), DisabledReranker()
    ).search("HgTe", workspace_id="ws", source_kind="fulltext")
    assert result.passages == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_field", ["workspace_id", "ingest_state"])
async def test_retriever_rejects_candidates_missing_scope_or_readiness(
    missing_field: str,
) -> None:
    candidate = candidate_fixture(1)[0]
    payload = dict(candidate.payload)
    payload.pop(missing_field)
    store = FakeStore(
        [SearchCandidate(candidate.passage_id, candidate.score, payload)]
    )

    result = await LiteratureRetriever(store, FakeEmbedder(), DisabledReranker()).search(
        "query", workspace_id="ws", top_k=1
    )

    assert result.passages == ()


@pytest.mark.asyncio
async def test_abstract_retrieval_source_kind_marks_result_abstract_only() -> None:
    store = FakeStore(candidate_fixture(source_kind=LiteratureSourceKind.ABSTRACT))
    result = await LiteratureRetriever(
        store, FakeEmbedder(), DisabledReranker()
    ).search("HgTe", workspace_id="ws", source_kind=LiteratureSourceKind.ABSTRACT)

    assert result.passages
    assert result.passages[0].source_kind is LiteratureSourceKind.ABSTRACT
    assert "abstract_only" in result.passages[0].limitations


@pytest.mark.asyncio
async def test_sparse_failure_uses_dense_only() -> None:
    store = FakeStore()
    store.hybrid_error = RuntimeError("sparse route unavailable")
    result = await LiteratureRetriever(
        store, FakeEmbedder(), DisabledReranker()
    ).search("query", workspace_id="ws", top_k=3)

    assert result.diagnostics.mode == "dense_only"
    assert "sparse_unavailable" in result.diagnostics.degraded_reasons
    assert store.dense_queries == 1
    assert store.dense_limit == 50


@pytest.mark.asyncio
async def test_both_retrieval_routes_fail_with_typed_error() -> None:
    store = FakeStore()
    store.sparse_error = RuntimeError("sparse down")
    with pytest.raises(RagRetrievalError) as exc_info:
        await LiteratureRetriever(
            store, FailingLocalEmbedder(), DisabledReranker()
        ).search("query", workspace_id="ws", top_k=3)

    assert exc_info.value.code == "retrieval_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query, top_k",
    [("", 3), ("  \n", 3), ("query", 0), ("query", 11)],
)
async def test_search_validates_query_and_top_k(query: str, top_k: int) -> None:
    with pytest.raises(ValueError):
        await LiteratureRetriever(FakeStore(), FakeEmbedder(), DisabledReranker()).search(
            query, workspace_id="ws", top_k=top_k
        )


@pytest.mark.asyncio
async def test_dedupe_prefers_score_then_newest_indexed_revision() -> None:
    first = candidate_fixture(1)[0]
    lower_score = candidate_fixture(1, text_prefix="other")[0]
    lower_score = SearchCandidate(
        passage_id="p-new",
        score=0.2,
        payload={
            **lower_score.payload,
            "passage_id": "p-new",
            "normalized_text_sha256": first.payload["normalized_text_sha256"],
            "indexed_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
        },
    )
    newer_tie = SearchCandidate(
        passage_id="p-newest",
        score=first.score,
        payload={
            **first.payload,
            "passage_id": "p-newest",
            "indexed_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        },
    )
    store = FakeStore([first, lower_score, newer_tie])
    result = await LiteratureRetriever(store, FakeEmbedder(), DisabledReranker()).search(
        "query", workspace_id="ws", top_k=3
    )

    assert len(result.passages) == 1
    assert result.passages[0].passage_id == "p-newest"


def _passage_point_for_tie_break(
    *, passage_id: str, revision: str, indexed_at: datetime
) -> PassagePoint:
    return PassagePoint(
        schema_version=1,
        record_type="passage",
        workspace_id="ws",
        document_id="doc",
        document_revision=revision,
        ingest_state=IngestState.READY,
        passage_id=passage_id,
        chunk_index=0,
        text="same passage text",
        title="HgTe detector",
        authors=("Author",),
        year=2024,
        section="Results",
        heading_path="Results",
        page_start=1,
        page_end=1,
        previous_passage_id=None,
        next_passage_id=None,
        relative_source_path="papers/hgte.pdf",
        model_fingerprint="f" * 64,
        normalized_text_sha256=_hash("same passage text"),
        indexed_at=indexed_at,
    )


@pytest.mark.asyncio
async def test_equal_rrf_scores_prefer_newer_persisted_revision_timestamp() -> None:
    old_point = _passage_point_for_tie_break(
        passage_id="old",
        revision="a" * 64,
        indexed_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    new_point = _passage_point_for_tie_break(
        passage_id="new",
        revision="b" * 64,
        indexed_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    store = FakeStore(
        [
            SearchCandidate(old_point.passage_id, 0.5, old_point.to_payload()),
            SearchCandidate(new_point.passage_id, 0.5, new_point.to_payload()),
        ]
    )

    result = await LiteratureRetriever(store, FakeEmbedder(), DisabledReranker()).search(
        "query", workspace_id="ws", top_k=1
    )

    assert result.passages[0].passage_id == "new"
    assert result.passages[0].document_revision == "b" * 64


@pytest.mark.asyncio
async def test_reranker_is_bounded_and_invalid_indexes_degrade() -> None:
    store = FakeStore(candidate_fixture(60))
    reranker = FakeReranker([SimpleNamespace(index=60, score=99.0)])
    result = await LiteratureRetriever(store, FakeEmbedder(), reranker).search(
        "query", workspace_id="ws", top_k=5
    )

    assert len(reranker.calls) == 1
    assert len(reranker.calls[0][1]) == 50
    assert reranker.calls[0][2] == 50
    assert result.diagnostics.reranked is False
    assert "reranker_unavailable" in result.diagnostics.degraded_reasons


@pytest.mark.asyncio
async def test_neighbor_context_is_one_bounded_call_and_isolated() -> None:
    center = candidate_fixture(1)[0]
    center = SearchCandidate(
        passage_id="center",
        score=1.0,
        payload={
            **center.payload,
            "passage_id": "center",
            "previous_passage_id": "before",
            "next_passage_id": "after",
        },
    )
    store = FakeStore([center])
    store.neighbors = {
        "before": SimpleNamespace(
            passage_id="before",
            workspace_id="ws",
            document_id="doc",
            document_revision="a" * 64,
            ingest_state="ready",
            source_kind=LiteratureSourceKind.FULLTEXT,
            text="before context that is long",
            previous_passage_id=None,
            next_passage_id=None,
        ),
        "after": SimpleNamespace(
            passage_id="after",
            workspace_id="other-workspace",
            document_id="doc",
            document_revision="a" * 64,
            ingest_state="ready",
            source_kind=LiteratureSourceKind.FULLTEXT,
            text="must be rejected",
            previous_passage_id=None,
            next_passage_id=None,
        ),
    }

    result = await LiteratureRetriever(store, FakeEmbedder(), DisabledReranker()).search(
        "query", workspace_id="ws", top_k=1, context_chars=9
    )

    assert len(store.neighbor_calls) == 1
    assert store.neighbor_calls[0] == ("ws", ["before", "after"])
    assert store.neighbor_source_kinds == [LiteratureSourceKind.FULLTEXT]
    assert len(result.passages[0].context_before) == 9
    assert result.passages[0].context_before.endswith("long")
    assert result.passages[0].context_after == ""


@pytest.mark.asyncio
async def test_neighbor_context_is_source_isolated() -> None:
    center = candidate_fixture(1)[0]
    center = SearchCandidate(
        passage_id="center",
        score=1.0,
        payload={
            **center.payload,
            "passage_id": "center",
            "previous_passage_id": "abstract-before",
        },
    )
    store = FakeStore([center])
    store.neighbors = {
        "abstract-before": SimpleNamespace(
            passage_id="abstract-before",
            workspace_id="ws",
            document_id="doc",
            document_revision="a" * 64,
            ingest_state="ready",
            source_kind=LiteratureSourceKind.ABSTRACT,
            text="abstract context must not leak",
            previous_passage_id=None,
            next_passage_id=None,
        ),
    }

    result = await LiteratureRetriever(store, FakeEmbedder(), DisabledReranker()).search(
        "query", workspace_id="ws", top_k=1, source_kind=LiteratureSourceKind.FULLTEXT
    )

    assert store.neighbor_source_kinds == [LiteratureSourceKind.FULLTEXT]
    assert result.passages[0].context_before == ""


@pytest.mark.asyncio
async def test_expand_radius_above_one_is_rejected_before_neighbor_fetch() -> None:
    store = FakeStore()

    with pytest.raises(ValueError, match="expand_radius"):
        await LiteratureRetriever(store, FakeEmbedder(), DisabledReranker()).search(
            "query", workspace_id="ws", top_k=1, expand_radius=2
        )

    assert store.neighbor_calls == []


@pytest.mark.asyncio
async def test_expand_radius_zero_skips_neighbor_fetch() -> None:
    center = candidate_fixture(1)[0]
    center = SearchCandidate(
        passage_id="center",
        score=1.0,
        payload={
            **center.payload,
            "passage_id": "center",
            "previous_passage_id": "before",
            "next_passage_id": "after",
        },
    )
    store = FakeStore([center])

    result = await LiteratureRetriever(store, FakeEmbedder(), DisabledReranker()).search(
        "query", workspace_id="ws", top_k=1, expand_radius=0
    )

    assert store.neighbor_calls == []
    assert result.passages[0].context_before == ""
    assert result.passages[0].context_after == ""
