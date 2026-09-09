"""Bounded hybrid retrieval over the Qdrant literature store.

The retriever deliberately contains no corpus or lexical index.  Qdrant owns
dense retrieval, BM25, and RRF fusion; this module only orchestrates provider
degradation, bounded post-processing, and provenance-safe context expansion.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from photomatagent.scientific.capabilities.literature.qdrant_store import (
    SearchCandidate,
    collection_fingerprint,
)


MAX_CANDIDATES = 50
MIN_TOP_K = 1
MAX_TOP_K = 10
MAX_NEIGHBOR_IDS = 50
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RagRetrievalError(RuntimeError):
    """Stable, secret-free retrieval boundary failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class RetrievalDiagnostics:
    """Bounded diagnostics describing the route used for one query."""

    mode: str
    candidate_count: int
    reranked: bool
    degraded_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievedPassage:
    """One provenance-carrying passage returned by :class:`LiteratureRetriever`."""

    passage_id: str
    workspace_id: str
    document_id: str
    document_revision: str
    text: str
    score: float
    title: str = ""
    authors: tuple[str, ...] = ()
    year: int | None = None
    section: str = ""
    heading_path: str = ""
    page_start: int | None = None
    page_end: int | None = None
    relative_source_path: str = ""
    previous_passage_id: str | None = None
    next_passage_id: str | None = None
    limitations: tuple[str, ...] = ()
    indexed_at: datetime | str | None = None
    context_before: str = ""
    context_after: str = ""

    @property
    def passage(self) -> str:
        """Compatibility alias used by the model-facing literature contract."""
        return self.text

    @property
    def paper_id(self) -> str:
        """Compatibility alias for callers that still call documents papers."""
        return self.document_id

    @property
    def source(self) -> str:
        """Compatibility alias for the workspace-relative source path."""
        return self.relative_source_path

    @property
    def file_name(self) -> str:
        """Compatibility alias for callers rendering source file metadata."""
        return self.relative_source_path.rsplit("/", 1)[-1]

    @property
    def page(self) -> int | None:
        """Return the first page for legacy bounded result rendering."""
        return self.page_start

    def as_dict(self) -> dict[str, Any]:
        """Render the stable, bounded public result shape."""
        return {
            "passage_id": self.passage_id,
            "paper_id": self.paper_id,
            "document_id": self.document_id,
            "document_revision": self.document_revision,
            "workspace_id": self.workspace_id,
            "title": self.title,
            "passage": self.text,
            "section": self.section,
            "heading_path": self.heading_path,
            "page": self.page,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "score": self.score,
            "source": self.source,
            "relative_source_path": self.relative_source_path,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "authors": list(self.authors),
            "year": self.year,
            "previous_passage_id": self.previous_passage_id,
            "next_passage_id": self.next_passage_id,
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Passages plus route/reranking diagnostics for one query."""

    passages: tuple[RetrievedPassage, ...]
    diagnostics: RetrievalDiagnostics


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _candidate_parts(candidate: Any) -> tuple[str, float, dict[str, Any]] | None:
    """Normalize a store candidate without trusting mutable provider objects."""
    payload_value = _value(candidate, "payload", {})
    if not isinstance(payload_value, Mapping):
        return None
    payload = dict(payload_value)
    passage_id = str(
        _value(candidate, "passage_id", payload.get("passage_id", "")) or ""
    )
    if not passage_id:
        return None
    raw_score = _value(candidate, "score", 0.0)
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 0.0
    if not math.isfinite(score):
        score = 0.0
    payload.setdefault("passage_id", passage_id)
    return passage_id, score, payload


def _text(payload: Mapping[str, Any]) -> str:
    value = payload.get("text", payload.get("passage", ""))
    return str(value) if value is not None else ""


def _normalized_text_hash(payload: Mapping[str, Any]) -> str:
    value = payload.get("normalized_text_sha256", "")
    if isinstance(value, str) and _SHA256_RE.fullmatch(value):
        return value
    normalized = " ".join(_text(payload).split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _document_id(payload: Mapping[str, Any]) -> str:
    return str(payload.get("document_id", payload.get("paper_id", "")) or "")


def _document_revision(payload: Mapping[str, Any]) -> str:
    return str(payload.get("document_revision", payload.get("revision", "")) or "")


def _datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    return None


def _datetime_rank(value: Any) -> int:
    parsed = _datetime(value)
    if parsed is None:
        return 0
    parsed_utc = parsed.astimezone(timezone.utc)
    return (
        parsed_utc.toordinal() * 86_400_000_000
        + (parsed_utc.hour * 3_600 + parsed_utc.minute * 60 + parsed_utc.second)
        * 1_000_000
        + parsed_utc.microsecond
    )


def _candidate_sort_key(item: tuple[str, float, dict[str, Any]]) -> tuple[float, int, str]:
    passage_id, score, payload = item
    return (-score, -_datetime_rank(payload.get("indexed_at")), passage_id)


def _ready_neighbor_value(item: Any, name: str, default: Any = None) -> Any:
    value = _value(item, name, default)
    if name == "passage_id" and not value:
        value = _value(item, "id", default)
    return value


def _neighbor_is_compatible(
    item: Any,
    *,
    requested_id: str,
    workspace_id: str,
    document_id: str,
    document_revision: str,
) -> bool:
    return (
        str(_ready_neighbor_value(item, "passage_id", "")) == requested_id
        and str(_ready_neighbor_value(item, "workspace_id", "")) == workspace_id
        and str(_ready_neighbor_value(item, "document_id", "")) == document_id
        and str(_ready_neighbor_value(item, "document_revision", ""))
        == document_revision
        and _enum_value(_ready_neighbor_value(item, "ingest_state", "")) == "ready"
    )


class LiteratureRetriever:
    """Orchestrate bounded Qdrant retrieval and optional provider reranking."""

    def __init__(
        self,
        store: Any,
        embedder: Any,
        reranker: Any,
        *,
        chunk_schema_version: int = 1,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._chunk_schema_version = chunk_schema_version

    async def _validate_current_generation(self) -> None:
        """Fail closed before querying vectors from another semantic space."""
        validate = getattr(self._store, "validate_current_generation", None)
        identity = getattr(self._embedder, "identity", None)
        if not callable(validate) or identity is None:
            return
        expected = collection_fingerprint(
            identity,
            self._chunk_schema_version,
            prefix=str(getattr(self._store, "prefix", "photomat_literature")),
            sparse_model=str(getattr(self._store, "sparse_model", "qdrant/bm25")),
        )
        result = validate(expected)
        if hasattr(result, "__await__"):
            await result

    @staticmethod
    def _validate_request(
        query: str,
        workspace_id: str,
        top_k: int,
        expand_radius: int,
        context_chars: int,
    ) -> None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(workspace_id, str) or not workspace_id.strip():
            raise ValueError("workspace_id must be a non-empty string")
        if type(top_k) is not int or not MIN_TOP_K <= top_k <= MAX_TOP_K:
            raise ValueError("top_k must be between 1 and 10")
        if type(expand_radius) is not int or expand_radius not in (0, 1):
            raise ValueError("expand_radius must be 0 or 1")
        if type(context_chars) is not int or context_chars < 0:
            raise ValueError("context_chars must be non-negative")

    async def _retrieve_candidates(
        self, query: str, *, workspace_id: str
    ) -> tuple[list[tuple[str, float, dict[str, Any]]], str, tuple[str, ...]]:
        await self._validate_current_generation()
        dense: list[float] | None = None
        try:
            vector = await self._embedder.embed_query(query)
            if isinstance(vector, (str, bytes, bytearray)) or not isinstance(
                vector, Sequence
            ):
                raise ValueError("embedding response is not a vector")
            dense = [float(value) for value in vector]
            if not dense or any(not math.isfinite(value) for value in dense):
                raise ValueError("embedding response is invalid")
        except Exception:
            dense = None

        if dense is None:
            try:
                sparse = await self._store.sparse_candidates(
                    query, workspace_id=workspace_id, limit=MAX_CANDIDATES
                )
            except Exception as exc:
                raise RagRetrievalError(
                    "retrieval_unavailable", "dense and sparse retrieval are unavailable"
                ) from exc
            return self._normalize_candidates(sparse, workspace_id), "sparse_only", (
                "dense_unavailable",
            )

        try:
            hybrid = await self._store.hybrid_candidates(
                query,
                dense,
                workspace_id=workspace_id,
                limit=MAX_CANDIDATES,
            )
        except Exception:
            try:
                dense_only = await self._store.dense_candidates(
                    dense, workspace_id=workspace_id, limit=MAX_CANDIDATES
                )
            except Exception as exc:
                raise RagRetrievalError(
                    "retrieval_unavailable", "dense and sparse retrieval are unavailable"
                ) from exc
            return self._normalize_candidates(dense_only, workspace_id), "dense_only", (
                "sparse_unavailable",
            )
        return self._normalize_candidates(hybrid, workspace_id), "hybrid_rrf", ()

    @staticmethod
    def _normalize_candidates(
        candidates: Any, workspace_id: str
    ) -> list[tuple[str, float, dict[str, Any]]]:
        if not isinstance(candidates, Sequence) or isinstance(
            candidates, (str, bytes, bytearray)
        ):
            return []
        unique: dict[tuple[str, str], tuple[str, float, dict[str, Any]]] = {}
        for raw_candidate in list(candidates)[:MAX_CANDIDATES]:
            normalized = _candidate_parts(raw_candidate)
            if normalized is None:
                continue
            passage_id, score, payload = normalized
            payload_workspace = payload.get("workspace_id")
            if payload_workspace is not None and str(payload_workspace) != workspace_id:
                continue
            state = payload.get("ingest_state")
            if state is not None and _enum_value(state) != "ready":
                continue
            key = (_document_id(payload), _normalized_text_hash(payload))
            prior = unique.get(key)
            if prior is None or _candidate_sort_key(normalized) < _candidate_sort_key(prior):
                unique[key] = normalized
        return sorted(unique.values(), key=_candidate_sort_key)

    async def _rerank(
        self,
        query: str,
        candidates: list[tuple[str, float, dict[str, Any]]],
    ) -> tuple[list[tuple[str, float, dict[str, Any], float]], bool, tuple[str, ...]]:
        if not candidates:
            return [], False, ()
        bounded = candidates[:MAX_CANDIDATES]
        passages = [_text(payload) for _, _, payload in bounded]
        try:
            raw_scores = await self._reranker.rerank(
                query, passages, top_n=len(bounded)
            )
            if raw_scores is None:
                raise ValueError("reranker returned no score sequence")
            if not isinstance(raw_scores, Sequence) or isinstance(
                raw_scores, (str, bytes, bytearray)
            ):
                raise ValueError("reranker returned an invalid score sequence")
            parsed: list[tuple[int, float]] = []
            seen: set[int] = set()
            for raw_score in raw_scores:
                index = _value(raw_score, "index")
                score_value = _value(raw_score, "score")
                if type(index) is not int or not 0 <= index < len(bounded):
                    raise ValueError("reranker index is invalid")
                if index in seen:
                    raise ValueError("reranker index is duplicated")
                score = float(score_value)
                if not math.isfinite(score):
                    raise ValueError("reranker score is invalid")
                seen.add(index)
                parsed.append((index, score))
            if not parsed:
                # DisabledReranker intentionally returns no scores to preserve
                # Qdrant's fused order; that is not a provider failure.
                return [
                    (passage_id, fused_score, payload, fused_score)
                    for passage_id, fused_score, payload in bounded
                ], False, ()
            parsed.sort(key=lambda item: (-item[1], item[0]))
            ranked_indices = {index for index, _ in parsed}
            ranked = [
                (*bounded[index], score)
                for index, score in parsed
            ]
            ranked.extend(
                (*candidate, candidate[1])
                for index, candidate in enumerate(bounded)
                if index not in ranked_indices
            )
            return ranked, True, ()
        except Exception:
            return [
                (passage_id, fused_score, payload, fused_score)
                for passage_id, fused_score, payload in bounded
            ], False, ("reranker_unavailable",)

    @staticmethod
    def _pointer(payload_or_item: Any, name: str) -> str | None:
        value = _ready_neighbor_value(payload_or_item, name, None)
        if value is None or value == "":
            return None
        return str(value)

    async def _contexts(
        self,
        rows: Sequence[tuple[str, float, dict[str, Any], float]],
        *,
        workspace_id: str,
        expand_radius: int,
        context_chars: int,
    ) -> dict[str, tuple[str, str]]:
        contexts = {passage_id: ("", "") for passage_id, *_ in rows}
        if not rows or expand_radius <= 0 or context_chars == 0:
            return contexts

        radius = expand_radius
        requested_ids: list[str] = []
        requested_set: set[str] = set()
        # Candidate payloads are used only to discover already-known links.
        # The final retrieval call remains a single bounded store operation.
        known_payloads = {
            passage_id: payload for passage_id, _, payload, _ in rows
        }
        for passage_id, _, payload, _ in rows:
            del passage_id
            for direction in ("previous_passage_id", "next_passage_id"):
                cursor = self._pointer(payload, direction)
                seen: set[str] = set()
                for _ in range(radius):
                    if not cursor or cursor in seen or len(requested_ids) >= MAX_NEIGHBOR_IDS:
                        break
                    seen.add(cursor)
                    if cursor not in requested_set:
                        requested_set.add(cursor)
                        requested_ids.append(cursor)
                    known = known_payloads.get(cursor)
                    cursor = self._pointer(
                        known,
                        direction,
                    ) if known is not None else None
        if not requested_ids:
            return contexts

        try:
            neighbors = await self._store.retrieve_passages(
                workspace_id, requested_ids[:MAX_NEIGHBOR_IDS]
            )
        except Exception:
            return contexts

        neighbor_map: dict[str, Any] = {}
        for _, _, payload, _ in rows:
            document_id = _document_id(payload)
            revision = _document_revision(payload)
            for neighbor in neighbors:
                neighbor_id = str(_ready_neighbor_value(neighbor, "passage_id", ""))
                if neighbor_id in neighbor_map:
                    continue
                if _neighbor_is_compatible(
                    neighbor,
                    requested_id=neighbor_id,
                    workspace_id=workspace_id,
                    document_id=document_id,
                    document_revision=revision,
                ):
                    neighbor_map[neighbor_id] = neighbor

        for passage_id, _, payload, _ in rows:
            document_id = _document_id(payload)
            revision = _document_revision(payload)
            previous_parts: list[str] = []
            cursor = self._pointer(payload, "previous_passage_id")
            context_seen: set[str] = set()
            for _ in range(radius):
                if not cursor or cursor in context_seen:
                    break
                context_seen.add(cursor)
                neighbor = neighbor_map.get(cursor)
                if neighbor is None or not _neighbor_is_compatible(
                    neighbor,
                    requested_id=cursor,
                    workspace_id=workspace_id,
                    document_id=document_id,
                    document_revision=revision,
                ):
                    break
                previous_parts.append(str(_ready_neighbor_value(neighbor, "text", "")))
                cursor = self._pointer(neighbor, "previous_passage_id")

            next_parts: list[str] = []
            cursor = self._pointer(payload, "next_passage_id")
            context_seen.clear()
            for _ in range(radius):
                if not cursor or cursor in context_seen:
                    break
                context_seen.add(cursor)
                neighbor = neighbor_map.get(cursor)
                if neighbor is None or not _neighbor_is_compatible(
                    neighbor,
                    requested_id=cursor,
                    workspace_id=workspace_id,
                    document_id=document_id,
                    document_revision=revision,
                ):
                    break
                next_parts.append(str(_ready_neighbor_value(neighbor, "text", "")))
                cursor = self._pointer(neighbor, "next_passage_id")
            before = " ".join(reversed(previous_parts))[-context_chars:]
            after = " ".join(next_parts)[:context_chars]
            contexts[passage_id] = (before, after)
        return contexts

    @staticmethod
    def _to_passage(
        candidate: tuple[str, float, dict[str, Any], float],
        *,
        workspace_id: str,
        context: tuple[str, str],
    ) -> RetrievedPassage:
        passage_id, _, payload, output_score = candidate
        authors_value = payload.get("authors", ())
        authors: tuple[str, ...]
        if isinstance(authors_value, str):
            authors = (authors_value,)
        elif isinstance(authors_value, Sequence):
            authors = tuple(str(author) for author in authors_value)
        else:
            authors = ()
        limitations_value = payload.get("limitations", ())
        limitations: tuple[str, ...]
        if isinstance(limitations_value, str):
            limitations = (limitations_value,)
        elif isinstance(limitations_value, Sequence):
            limitations = tuple(str(item) for item in limitations_value)
        else:
            limitations = ()
        year_value = payload.get("year")
        try:
            year = int(year_value) if year_value is not None else None
        except (TypeError, ValueError):
            year = None
        page_value = payload.get("page", payload.get("page_start"))
        page_start_value = payload.get("page_start", page_value)
        page_end_value = payload.get("page_end", page_value)
        try:
            page_start = int(page_start_value) if page_start_value is not None else None
        except (TypeError, ValueError):
            page_start = None
        try:
            page_end = int(page_end_value) if page_end_value is not None else None
        except (TypeError, ValueError):
            page_end = None
        indexed_at = payload.get("indexed_at")
        if indexed_at is not None and not isinstance(indexed_at, (datetime, str)):
            indexed_at = str(indexed_at)
        return RetrievedPassage(
            passage_id=passage_id,
            workspace_id=workspace_id,
            document_id=_document_id(payload),
            document_revision=_document_revision(payload),
            text=_text(payload),
            score=float(output_score),
            title=str(payload.get("title", "") or ""),
            authors=authors,
            year=year,
            section=str(payload.get("section", "") or ""),
            heading_path=str(payload.get("heading_path", "") or ""),
            page_start=page_start,
            page_end=page_end,
            relative_source_path=str(
                payload.get("relative_source_path", payload.get("file_name", "")) or ""
            ),
            previous_passage_id=LiteratureRetriever._pointer(
                payload, "previous_passage_id"
            )
            or LiteratureRetriever._pointer(payload, "previous_chunk_id"),
            next_passage_id=LiteratureRetriever._pointer(payload, "next_passage_id")
            or LiteratureRetriever._pointer(payload, "next_chunk_id"),
            limitations=limitations,
            indexed_at=indexed_at,
            context_before=context[0],
            context_after=context[1],
        )

    async def search(
        self,
        query: str,
        *,
        workspace_id: str,
        top_k: int = 5,
        expand_radius: int = 1,
        context_chars: int = 300,
    ) -> RetrievalResult:
        """Search one workspace with bounded candidates and context."""
        self._validate_request(
            query, workspace_id, top_k, expand_radius, context_chars
        )
        candidates, mode, degraded = await self._retrieve_candidates(
            query, workspace_id=workspace_id
        )
        ranked, reranked, rerank_reasons = await self._rerank(query, candidates)
        final_rows = ranked[:top_k]
        contexts = await self._contexts(
            final_rows,
            workspace_id=workspace_id,
            expand_radius=expand_radius,
            context_chars=context_chars,
        )
        passages = tuple(
            self._to_passage(
                row,
                workspace_id=workspace_id,
                context=contexts.get(row[0], ("", "")),
            )
            for row in final_rows
        )
        diagnostics = RetrievalDiagnostics(
            mode=mode,
            candidate_count=len(candidates),
            reranked=reranked,
            degraded_reasons=tuple(dict.fromkeys((*degraded, *rerank_reasons))),
        )
        return RetrievalResult(passages=passages, diagnostics=diagnostics)


__all__ = [
    "LiteratureRetriever",
    "RagRetrievalError",
    "RetrievedPassage",
    "RetrievalDiagnostics",
    "RetrievalResult",
]
