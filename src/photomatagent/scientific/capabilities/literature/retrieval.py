"""Hybrid retrieval: dense + BM25, RRF fusion, cross-encoder rerank.

Retrieval is fully local: embeddings come from the same sentence-transformers
model used at index time, keyword scoring is a compact BM25 over the index's
passage table, and reranking uses the small ``ms-marco-MiniLM`` cross-encoder.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any

from photomatagent.scientific.capabilities.literature.index import LiteratureIndex

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-.][A-Za-z0-9]+)*")
_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "for", "on", "with",
    "at", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "this", "that", "these", "those", "it", "its", "we", "our", "they",
    "their", "not", "no", "but", "than", "then", "into", "over", "under",
}

RRF_K = 60.0
BM25_K1 = 1.5
BM25_B = 0.75


def _tokens(text: str) -> list[str]:
    return [
        token.casefold()
        for token in _TOKEN_RE.findall(text)
        if len(token) > 1 and token.casefold() not in _STOPWORDS
    ]


class _Bm25:
    """In-memory BM25 over a passage corpus (loaded once per index state)."""

    def __init__(self, passages: list[dict[str, Any]]) -> None:
        self.documents: list[tuple[str, list[str]]] = [
            (row["passage_id"], _tokens(row["text"])) for row in passages
        ]
        self.doc_lengths = [len(tokens) for _, tokens in self.documents]
        self.average_length = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        )
        self.document_frequency: dict[str, int] = {}
        for _, tokens in self.documents:
            for token in set(tokens):
                self.document_frequency[token] = (
                    self.document_frequency.get(token, 0) + 1
                )
        self.total_documents = len(self.documents)

    def score(self, passage_id: str, query_tokens: list[str]) -> float:
        index = next(
            (i for i, (pid, _) in enumerate(self.documents) if pid == passage_id),
            None,
        )
        if index is None:
            return 0.0
        tokens = self.documents[index][1]
        length = self.doc_lengths[index]
        frequencies: dict[str, int] = {}
        for token in tokens:
            frequencies[token] = frequencies.get(token, 0) + 1
        total = 0.0
        for token in query_tokens:
            frequency = frequencies.get(token, 0)
            if not frequency:
                continue
            df = self.document_frequency.get(token, 0)
            idf = math.log((self.total_documents - df + 0.5) / (df + 0.5) + 1.0)
            denominator = frequency + BM25_K1 * (
                1 - BM25_B + BM25_B * length / self.average_length
            )
            total += idf * frequency * (BM25_K1 + 1) / denominator
        return total

    def rank(self, query_tokens: list[str], limit: int) -> list[tuple[str, float]]:
        if not self.documents or not query_tokens:
            return []
        scored = [(pid, self.score(pid, query_tokens)) for pid, _ in self.documents]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [(pid, score) for pid, score in scored[:limit] if score > 0.0]


@lru_cache(maxsize=8)
def _reranker(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


@lru_cache(maxsize=16)
def _query_embedding(query: str, model_name: str) -> tuple[float, ...]:
    from photomatagent.scientific.capabilities.literature.index import _embedder

    model = _embedder(model_name)
    vector = model.encode(
        [f"query: {query}"], normalize_embeddings=True, convert_to_numpy=True
    )[0]
    return tuple(map(float, vector))


def _rrf_fuse(rankings: list[list[str]], k: float = RRF_K) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, passage_id in enumerate(ranking, start=1):
            fused[passage_id] = fused.get(passage_id, 0.0) + 1.0 / (k + rank)
    return fused


class Retriever:
    """Hybrid search over a ``LiteratureIndex``."""

    def __init__(
        self,
        index: LiteratureIndex,
        *,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        candidates: int = 50,
    ) -> None:
        self.index = index
        self.reranker_model = reranker_model
        self.candidates = candidates
        self._corpus: list[dict[str, Any]] | None = None
        self._bm25: _Bm25 | None = None
        self._revision: object | None = None

    def _load_corpus(self) -> list[dict[str, Any]]:
        """Passages cached per content-derived index revision."""
        revision = self.index.revision()
        if self._corpus is None or self._revision != revision:
            self._corpus = self.index.all_passages()
            self._bm25 = _Bm25(self._corpus)
            self._revision = revision
        return self._corpus

    def _by_id(self) -> dict[str, dict[str, Any]]:
        return {row["passage_id"]: row for row in self._load_corpus()}

    def _dense_ranking(self, query: str, limit: int) -> list[str]:
        if self.index.count_passages() == 0:
            return []
        vector = _query_embedding(query, self.index.embedding_model)
        table = self.index.passage_table()
        rows = (
            table.search(list(vector))
            .limit(limit)
            .metric("cosine")
            .select(["passage_id"])
            .to_list()
        )
        return [row["passage_id"] for row in rows]

    def hybrid_search(
        self,
        query: str,
        *,
        top_k: int = 5,
        expand_radius: int = 1,
        context_chars: int = 300,
    ) -> list[dict[str, Any]]:
        """Run dense + BM25, fuse with RRF, rerank, then expand context."""
        corpus = self._load_corpus()
        if not corpus:
            return []
        query_tokens = _tokens(query)
        dense_ranking = self._dense_ranking(query, self.candidates)
        bm25_ranking = [
            pid
            for pid, _ in (self._bm25.rank(query_tokens, self.candidates) if self._bm25 else [])
        ]
        fused = _rrf_fuse([dense_ranking, bm25_ranking])
        candidates = sorted(
            fused.items(), key=lambda item: item[1], reverse=True
        )[: self.candidates]

        rows_by_id = self._by_id()
        candidate_rows = [
            rows_by_id[pid] for pid, _ in candidates if pid in rows_by_id
        ]
        if not candidate_rows:
            return []

        reranked = self._rerank(query, candidate_rows)
        results: list[dict[str, Any]] = []
        for row, score in reranked[:top_k]:
            result = {
                "passage_id": row["passage_id"],
                "paper_id": row["paper_id"],
                "title": row["title"],
                "passage": row["text"],
                "section": row["section"],
                "page": row["page"],
                "score": score,
                "source": row["file_name"],
                "heading_path": row["heading_path"],
            }
            if expand_radius > 0:
                context = self._expand_context(
                    row, radius=expand_radius, chars=context_chars
                )
                result["context_before"] = context[0]
                result["context_after"] = context[1]
            results.append(result)
        return results

    def _rerank(
        self, query: str, rows: list[dict[str, Any]]
    ) -> list[tuple[dict[str, Any], float]]:
        if len(rows) <= 1:
            return [(rows[0], 1.0)] if rows else []
        model = _reranker(self.reranker_model)
        pairs = [(query, row["text"][:512]) for row in rows]
        scores = model.predict(pairs, batch_size=16)
        scored = sorted(
            zip(rows, [float(s) for s in scores]), key=lambda x: x[1], reverse=True
        )
        return scored

    def _expand_context(
        self, row: dict[str, Any], *, radius: int, chars: int
    ) -> tuple[str, str]:
        rows_by_id = self._by_id()
        before_parts: list[str] = []
        cursor = row.get("previous_chunk_id")
        for _ in range(radius):
            if not cursor or cursor not in rows_by_id:
                break
            before_parts.append(rows_by_id[cursor]["text"])
            cursor = rows_by_id[cursor].get("previous_chunk_id")
        after_parts: list[str] = []
        cursor = row.get("next_chunk_id")
        for _ in range(radius):
            if not cursor or cursor not in rows_by_id:
                break
            after_parts.append(rows_by_id[cursor]["text"])
            cursor = rows_by_id[cursor].get("next_chunk_id")
        return (
            " ".join(reversed(before_parts))[-chars:],
            " ".join(after_parts)[:chars],
        )
