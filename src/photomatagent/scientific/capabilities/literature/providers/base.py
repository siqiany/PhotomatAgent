"""Contracts shared by literature RAG model providers.

The provider layer deliberately contains no model-specific imports.  Optional
backends are loaded by the concrete providers only when their first operation
needs them, keeping capability probing and the base runtime soft-failing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Protocol


@dataclass(frozen=True)
class ModelIdentity:
    """The model semantics that affect stored/query vectors."""

    provider: str
    model: str
    dimension: int | None
    document_prefix: str = ""
    query_prefix: str = ""
    normalize: bool = False

    def fingerprint_material(self) -> dict[str, object]:
        """Return canonical, secret-free material for collection fingerprints."""
        return {
            "provider": self.provider,
            "model": self.model,
            "dimension": self.dimension,
            "document_prefix": self.document_prefix,
            "query_prefix": self.query_prefix,
            "normalize": self.normalize,
        }


@dataclass(frozen=True)
class RerankScore:
    """A score associated with the input passage at ``index``."""

    index: int
    score: float


class RagProviderError(RuntimeError):
    """Typed, stable provider failure without secret-bearing diagnostics."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class EmbeddingProvider(Protocol):
    identity: ModelIdentity

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class RerankerProvider(Protocol):
    identity: ModelIdentity

    async def rerank(
        self, query: str, passages: Sequence[str], *, top_n: int
    ) -> list[RerankScore]: ...


def validate_vectors(
    vectors: Sequence[Sequence[float]],
    expected_count: int,
    expected_dimension: int | None,
) -> list[list[float]]:
    """Validate and normalize an embedding response into plain float rows.

    Provider responses are external or optional-library boundaries, so this
    helper intentionally rejects malformed shapes and non-finite values before
    anything can be persisted in a vector store.
    """
    try:
        count = len(vectors)
    except Exception as exc:
        raise RagProviderError(
            "vector_count_mismatch", "embedding response is not a sequence"
        ) from exc
    if count != expected_count:
        raise RagProviderError(
            "vector_count_mismatch",
            f"expected {expected_count} vectors, received {count}",
        )

    normalized: list[list[float]] = []
    for row_index, row in enumerate(vectors):
        try:
            row_length = len(row)
        except Exception as exc:
            raise RagProviderError(
                "vector_dimension_mismatch",
                f"vector {row_index} is not a sequence",
            ) from exc
        if expected_dimension is not None and row_length != expected_dimension:
            raise RagProviderError(
                "vector_dimension_mismatch",
                f"expected dimension {expected_dimension}, received {row_length}",
            )
        values: list[float] = []
        for value in row:
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise RagProviderError(
                    "vector_non_numeric", "embedding contains a non-numeric value"
                ) from exc
            if not isfinite(number):
                raise RagProviderError(
                    "vector_non_finite", "embedding contains a non-finite value"
                )
            values.append(number)
        normalized.append(values)
    return normalized


__all__ = [
    "EmbeddingProvider",
    "ModelIdentity",
    "RagProviderError",
    "RerankScore",
    "RerankerProvider",
    "validate_vectors",
]
