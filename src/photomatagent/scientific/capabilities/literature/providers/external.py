"""Explicitly-gated providers for OpenAI- and Cohere-compatible services."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from functools import lru_cache
from math import isfinite
from typing import Any

from .base import ModelIdentity, RagProviderError, RerankScore, validate_vectors


def _status_code(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


@lru_cache(maxsize=1)
def _retryable_exception_types() -> tuple[type[BaseException], ...]:
    """Return explicitly supported transport/timeout exception classes."""
    types: list[type[BaseException]] = [OSError, TimeoutError, ConnectionError]
    try:
        import httpx
    except ImportError:
        pass
    else:
        for name in ("TransportError", "TimeoutException"):
            candidate = getattr(httpx, name, None)
            if isinstance(candidate, type) and issubclass(candidate, BaseException):
                types.append(candidate)
    try:
        import openai
    except ImportError:
        pass
    else:
        for name in ("APIConnectionError", "APITimeoutError"):
            candidate = getattr(openai, name, None)
            if isinstance(candidate, type) and issubclass(candidate, BaseException):
                types.append(candidate)
    return tuple(dict.fromkeys(types))


def _is_retryable(exc: BaseException) -> bool:
    status = _status_code(exc)
    if status is not None:
        return status == 408 or status == 429 or 500 <= status <= 599
    return isinstance(exc, _retryable_exception_types())


def _response_data(response: Any) -> Any:
    if isinstance(response, dict):
        return response.get("data")
    return getattr(response, "data", None)


def _item_value(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


class OpenAICompatibleEmbeddingProvider:
    """OpenAI ``/embeddings`` compatible provider with bounded retries."""

    def __init__(
        self,
        model: str,
        dimension: int,
        base_url: str = "",
        api_key: str = "",
        *,
        client: Any | None = None,
        batch_size: int = 128,
        timeout_seconds: int = 20,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")
        self._model_name = model
        self._dimension = dimension
        self._base_url = base_url.strip()
        self._api_key = api_key
        self._client_instance = client
        self._batch_size = batch_size
        self._timeout_seconds = timeout_seconds
        self.identity = ModelIdentity(
            provider="openai_compatible",
            model=model,
            dimension=dimension,
            normalize=False,
        )

    def _client(self) -> Any:
        if self._client_instance is not None:
            return self._client_instance
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RagProviderError(
                "external_embedding_dependency_missing",
                "openai is required for the configured embedding provider",
            ) from exc
        options: dict[str, Any] = {
            "api_key": self._api_key,
            "timeout": self._timeout_seconds,
        }
        if self._base_url:
            options["base_url"] = self._base_url
        self._client_instance = AsyncOpenAI(**options)
        return self._client_instance

    async def _create(self, texts: Sequence[str]) -> Any:
        client = self._client()
        last_error: BaseException | None = None
        for attempt in range(3):
            try:
                return await client.embeddings.create(
                    model=self._model_name,
                    input=list(texts),
                    timeout=self._timeout_seconds,
                )
            except RagProviderError:
                raise
            except Exception as exc:
                last_error = exc
                if not _is_retryable(exc) or attempt == 2:
                    raise RagProviderError(
                        "embedding_request_failed",
                        "embedding request failed",
                    ) from exc
                # Yield without imposing an unbounded or test-hostile delay.
                await asyncio.sleep(0)
        raise RagProviderError("embedding_request_failed", "embedding request failed") from last_error

    async def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        response = await self._create(texts)
        data = _response_data(response)
        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise RagProviderError(
                "embedding_invalid_response", "embedding response data is invalid"
            )
        indexed: list[tuple[int, Any]] = []
        for item in data:
            index = _item_value(item, "index")
            vector = _item_value(item, "embedding")
            if type(index) is not int or index < 0 or index >= len(texts):
                raise RagProviderError(
                    "embedding_invalid_response", "embedding response index is invalid"
                )
            if any(existing == index for existing, _ in indexed):
                raise RagProviderError(
                    "embedding_invalid_response", "embedding response index is duplicated"
                )
            indexed.append((index, vector))
        if len(indexed) != len(texts):
            raise RagProviderError(
                "embedding_invalid_response", "embedding response count is invalid"
            )
        indexed.sort(key=lambda item: item[0])
        vectors = [vector for _, vector in indexed]
        return validate_vectors(vectors, len(texts), self.identity.dimension)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        result: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            result.extend(await self._embed_batch(texts[start : start + self._batch_size]))
        return result

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_documents([text])
        return vectors[0]


class CohereCompatibleRerankerProvider:
    """Cohere ``/rerank`` compatible provider with strict response parsing."""

    DEFAULT_BASE_URL = "https://api.cohere.com/v1/rerank"

    def __init__(
        self,
        model: str,
        base_url: str = "",
        api_key: str = "",
        *,
        transport: Any | None = None,
        timeout_seconds: int = 20,
    ) -> None:
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")
        self._model_name = model
        self._base_url = base_url.strip() or self.DEFAULT_BASE_URL
        self._api_key = api_key
        self._transport_instance = transport
        self._timeout_seconds = timeout_seconds
        self.identity = ModelIdentity(
            provider="cohere_compatible",
            model=model,
            dimension=None,
            normalize=False,
        )

    def _transport(self) -> Any:
        if self._transport_instance is not None:
            return self._transport_instance
        try:
            import httpx
        except ImportError as exc:
            raise RagProviderError(
                "external_reranker_dependency_missing",
                "httpx is required for the configured reranker provider",
            ) from exc
        self._transport_instance = httpx.AsyncClient()
        return self._transport_instance

    async def rerank(
        self, query: str, passages: Sequence[str], *, top_n: int
    ) -> list[RerankScore]:
        if top_n < 0:
            raise RagProviderError(
                "reranker_invalid_request", "top_n must not be negative"
            )
        if not passages or top_n == 0:
            return []
        requested_top_n = min(top_n, len(passages))
        payload = {
            "model": self._model_name,
            "query": query,
            "documents": list(passages),
            "top_n": requested_top_n,
        }
        try:
            response = await self._transport().post(
                url=self._base_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self._timeout_seconds,
            )
        except RagProviderError:
            raise
        except Exception as exc:
            raise RagProviderError(
                "reranker_request_failed", "reranker request failed"
            ) from exc
        status = (
            response.get("status_code", 200)
            if isinstance(response, dict)
            else getattr(response, "status_code", None)
        )
        if not isinstance(status, int) or not 200 <= status < 300:
            raise RagProviderError(
                "reranker_request_failed", "reranker request failed"
            )
        if isinstance(response, dict):
            body = response
        else:
            try:
                body = response.json()
            except Exception as exc:
                raise RagProviderError(
                    "reranker_invalid_response", "reranker response is not valid JSON"
                ) from exc
        if not isinstance(body, dict) or not isinstance(body.get("results"), list):
            raise RagProviderError(
                "reranker_invalid_response", "reranker response results are invalid"
            )
        results: list[RerankScore] = []
        seen: set[int] = set()
        for item in body["results"]:
            if not isinstance(item, dict):
                raise RagProviderError(
                    "reranker_invalid_response", "reranker result item is invalid"
                )
            index = item.get("index")
            score = item.get("relevance_score")
            if type(index) is not int or index < 0 or index >= len(passages):
                raise RagProviderError(
                    "reranker_invalid_response", "reranker result index is invalid"
                )
            if index in seen or isinstance(score, bool) or not isinstance(
                score, (int, float)
            ):
                raise RagProviderError(
                    "reranker_invalid_response", "reranker result score is invalid"
                )
            numeric_score = float(score)
            if not isfinite(numeric_score):
                raise RagProviderError(
                    "reranker_invalid_response", "reranker result score is invalid"
                )
            seen.add(index)
            results.append(RerankScore(index=index, score=numeric_score))
        results.sort(key=lambda item: (-item.score, item.index))
        return results[:requested_top_n]


__all__ = [
    "CohereCompatibleRerankerProvider",
    "OpenAICompatibleEmbeddingProvider",
]
