"""Local sentence-transformers providers with lazy optional imports."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Sequence
from functools import lru_cache
from math import isfinite
from typing import Any

from .base import ModelIdentity, RagProviderError, RerankScore, validate_vectors


async def _to_local_thread(function: Any, *args: Any) -> Any:
    """Run local model work in a managed worker thread.

    Some Python 3.12 builds do not reliably tear down the implicit asyncio
    executor at ``asyncio.run`` shutdown.  Installing a bounded executor on
    the active loop keeps the required ``asyncio.to_thread`` boundary while
    making CLI/test event loops terminate deterministically.
    """
    loop = asyncio.get_running_loop()
    executor: ThreadPoolExecutor | None = None
    if getattr(loop, "_default_executor", None) is None:
        executor = ThreadPoolExecutor(max_workers=4)
        loop.set_default_executor(executor)
    try:
        return await asyncio.to_thread(function, *args)
    finally:
        if executor is not None:
            # Python 3.12's asyncio.run shutdown can wait forever for an
            # implicit executor after to_thread.  The work item is complete;
            # enqueue the sentinel without blocking the event loop and detach
            # this executor before asyncio performs its own shutdown pass.
            executor.shutdown(wait=False)
            loop._default_executor = None  # type: ignore[attr-defined]


@lru_cache(maxsize=8)
def _load_sentence_transformer(model_name: str) -> Any:
    """Load a sentence transformer only when a local operation is requested."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RagProviderError(
            "local_embedding_dependency_missing",
            "sentence-transformers is required for local embeddings",
        ) from exc
    return SentenceTransformer(model_name)


@lru_cache(maxsize=8)
def _load_cross_encoder(model_name: str) -> Any:
    """Load a cross encoder only when a local rerank is requested."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RagProviderError(
            "local_reranker_dependency_missing",
            "sentence-transformers is required for local reranking",
        ) from exc
    return CrossEncoder(model_name)


class LocalSentenceTransformerProvider:
    """E5-style local embedding provider."""

    def __init__(
        self,
        model: str,
        dimension: int,
        *,
        batch_size: int = 128,
    ) -> None:
        self._model_name = model
        self._batch_size = batch_size
        self.identity = ModelIdentity(
            provider="local",
            model=model,
            dimension=dimension,
            document_prefix="passage: ",
            query_prefix="query: ",
            normalize=True,
        )

    def _model(self) -> Any:
        return _load_sentence_transformer(self._model_name)

    def _encode(self, prepared: Sequence[str]) -> Any:
        """Load the model and encode in the same worker invocation."""
        return self._model().encode(
            prepared,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        prepared = [f"passage: {text}" for text in texts]
        raw = await _to_local_thread(self._encode, prepared)
        return validate_vectors(raw, len(texts), self.identity.dimension)

    async def embed_query(self, text: str) -> list[float]:
        raw = await _to_local_thread(self._encode, [f"query: {text}"])
        vectors = validate_vectors(raw, 1, self.identity.dimension)
        return vectors[0]


class LocalCrossEncoderProvider:
    """Local cross-encoder reranker with bounded text and deterministic order."""

    def __init__(
        self,
        model: str,
        *,
        batch_size: int = 128,
        max_text_chars: int = 4096,
        max_length: int | None = None,
    ) -> None:
        if max_length is not None:
            max_text_chars = max_length
        if max_text_chars < 1:
            raise ValueError("max_text_chars must be positive")
        self._model_name = model
        self._batch_size = batch_size
        self._max_text_chars = max_text_chars
        self.identity = ModelIdentity(
            provider="local",
            model=model,
            dimension=None,
            normalize=False,
        )

    def _model(self) -> Any:
        return _load_cross_encoder(self._model_name)

    def _text_limit(self, model: Any) -> int:
        configured = self._max_text_chars
        supported = getattr(model, "max_length", None)
        if isinstance(supported, int) and supported > 0:
            return min(configured, supported)
        tokenizer = getattr(model, "tokenizer", None)
        token_limit = getattr(tokenizer, "model_max_length", None)
        if isinstance(token_limit, int) and 0 < token_limit < 1_000_000:
            return min(configured, token_limit)
        return configured

    def _predict(self, query: str, passages: Sequence[str]) -> Any:
        """Load the reranker and score pairs in the same worker invocation."""
        model = self._model()
        limit = self._text_limit(model)
        pairs = [(query[:limit], passage[:limit]) for passage in passages]
        return model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )

    async def rerank(
        self, query: str, passages: Sequence[str], *, top_n: int
    ) -> list[RerankScore]:
        if top_n < 0:
            raise RagProviderError(
                "reranker_invalid_request", "top_n must not be negative"
            )
        if not passages or top_n == 0:
            return []
        raw_scores = await _to_local_thread(self._predict, query, passages)
        try:
            count = len(raw_scores)
        except Exception as exc:
            raise RagProviderError(
                "reranker_invalid_response", "reranker returned a non-sequence"
            ) from exc
        if count != len(passages):
            raise RagProviderError(
                "reranker_invalid_response",
                f"expected {len(passages)} scores, received {count}",
            )
        scored: list[RerankScore] = []
        for index, raw_score in enumerate(raw_scores):
            try:
                score = float(raw_score)
            except (TypeError, ValueError) as exc:
                raise RagProviderError(
                    "reranker_invalid_response", "reranker score is not numeric"
                ) from exc
            if not isfinite(score):
                raise RagProviderError(
                    "reranker_invalid_response",
                    "reranker score is not finite",
                )
            scored.append(RerankScore(index=index, score=score))
        scored.sort(key=lambda item: (-item.score, item.index))
        return scored[: min(top_n, len(scored))]


class DisabledRerankerProvider:
    """Explicit no-op reranker used to retain the retrieval fusion order."""

    def __init__(self, model: str = "disabled") -> None:
        self.identity = ModelIdentity(
            provider="disabled",
            model=model,
            dimension=None,
            normalize=False,
        )

    async def rerank(
        self, query: str, passages: Sequence[str], *, top_n: int
    ) -> list[RerankScore]:
        del query, passages, top_n
        return []


class DisabledReranker(DisabledRerankerProvider):
    """Backward-compatible short name for the explicit disabled provider."""


__all__ = [
    "DisabledReranker",
    "DisabledRerankerProvider",
    "LocalCrossEncoderProvider",
    "LocalSentenceTransformerProvider",
]
