"""The sole Qdrant persistence and query adapter for literature RAG.

The rest of the literature capability talks to this module through typed
contracts.  In particular, no caller can submit arbitrary Qdrant JSON,
filters, or collection names through this boundary.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from photomatagent.scientific.capabilities.literature.models import (
    DocumentManifest,
    DocumentStatus,
    IngestState,
    PassagePoint,
    validate_relative_source_path,
)
from photomatagent.scientific.capabilities.literature.providers.base import ModelIdentity

_AsyncQdrantClient: Any
try:  # qdrant is an optional capability until the final RAG task declares it.
    from qdrant_client import AsyncQdrantClient as _ImportedAsyncQdrantClient

    _AsyncQdrantClient = _ImportedAsyncQdrantClient
except ImportError:  # pragma: no cover - exercised by capability probing
    _AsyncQdrantClient = None

# Public module attribute retained for dependency injection in capability and
# unit tests; production callers still receive the single client built below.
AsyncQdrantClient: Any = _AsyncQdrantClient


DOCUMENT_NAMESPACE = uuid.UUID("9a6ec7b1-f1b6-4a2c-b9fd-8bb0b4df8f9b")
PASSAGE_NAMESPACE = uuid.UUID("bc81c8df-7614-4c79-8ad3-3d78535f3e64")
COLLECTION_SCHEMA_VERSION = 1
SPARSE_MODEL = "qdrant/bm25"
MAX_CANDIDATES = 50
GENERATION_POINT_NAMESPACE = uuid.UUID("b3a8b72d-b7d1-4e31-8589-14cb5132f0a0")
_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class CollectionGeneration:
    fingerprint: str
    documents_physical: str
    passages_physical: str
    documents_alias: str
    passages_alias: str


@dataclass(frozen=True)
class SearchCandidate:
    passage_id: str
    score: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class SnapshotFile:
    collection: str
    file_name: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class SnapshotManifest:
    generation: CollectionGeneration
    created_at: datetime
    files: tuple[SnapshotFile, ...]


class QdrantStoreError(RuntimeError):
    """Stable, secret-free adapter failures."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


def _qdrant_models() -> Any:
    try:
        from qdrant_client import models
    except ImportError as exc:  # pragma: no cover - optional capability path
        raise QdrantStoreError(
            "qdrant_dependency_missing", "qdrant-client is not installed"
        ) from exc
    return models


def workspace_id_for(root: Path) -> str:
    """Return the stable 16-hex workspace identity."""
    return hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()[:16]


def _validate_relative_path(relative_source_path: str) -> None:
    validate_relative_source_path(relative_source_path)


def document_id_for(workspace_id: str, relative_source_path: str) -> str:
    _validate_relative_path(relative_source_path)
    return str(uuid.uuid5(DOCUMENT_NAMESPACE, f"{workspace_id}:{relative_source_path}"))


def passage_id_for(document_id: str, revision: str, chunk_index: int) -> str:
    if _FINGERPRINT_RE.fullmatch(revision) is None:
        raise ValueError("revision must be 64 lowercase hexadecimal characters")
    if chunk_index < 0:
        raise ValueError("chunk_index must be non-negative")
    return str(uuid.uuid5(PASSAGE_NAMESPACE, f"{document_id}:{revision}:{chunk_index}"))


def collection_fingerprint(
    identity: ModelIdentity,
    chunk_schema_version: int,
    *,
    prefix: str = "photomat_literature",
    collection_schema_version: int = COLLECTION_SCHEMA_VERSION,
    sparse_model: str = SPARSE_MODEL,
) -> str:
    """Hash only collection-affecting, secret-free model/schema semantics."""
    material = {
        "collection_schema_version": collection_schema_version,
        "chunk_schema_version": chunk_schema_version,
        "embedding": identity.fingerprint_material(),
        "sparse_model": sparse_model,
    }
    encoded = json.dumps(
        material, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _record_attr(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _payload(record: Any) -> dict[str, Any]:
    value = _record_attr(record, "payload", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _datetime_value(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    return None


def _metadata(info: Any) -> dict[str, Any]:
    config = _record_attr(info, "config", None)
    metadata = _record_attr(config, "metadata", None)
    if isinstance(metadata, Mapping):
        return dict(metadata)
    metadata = _record_attr(info, "metadata", None)
    return dict(metadata) if isinstance(metadata, Mapping) else {}


class QdrantLiteratureStore:
    """Narrow asynchronous adapter for versioned literature collections."""

    def __init__(
        self,
        client: Any,
        *,
        prefix: str = "photomat_literature",
        dimension: int | None = None,
        timeout_seconds: int | None = None,
        sparse_model: str = SPARSE_MODEL,
        snapshot_base_url: str | None = None,
        snapshot_api_key: str | None = None,
        snapshot_transport: Any | None = None,
    ) -> None:
        if not prefix or any(char.isspace() for char in prefix):
            raise ValueError("Qdrant collection prefix must be non-empty and whitespace-free")
        self._client = client
        self.prefix = prefix
        self.dimension = dimension
        self.timeout_seconds = timeout_seconds
        self.sparse_model = sparse_model
        self.snapshot_base_url = snapshot_base_url.rstrip("/") if snapshot_base_url else None
        self._snapshot_api_key = snapshot_api_key
        self._snapshot_transport = snapshot_transport
        self._generations: dict[str, CollectionGeneration] = {}

    @classmethod
    def from_config(cls, config: Any) -> "QdrantLiteratureStore":
        """Build one client, resolving the optional key only at call time."""
        if AsyncQdrantClient is None:
            raise QdrantStoreError(
                "qdrant_dependency_missing", "qdrant-client is not installed"
            )
        api_key_name = str(getattr(config, "qdrant_api_key_env", "QDRANT_API_KEY"))
        api_key = os.environ.get(api_key_name, "").strip() or None
        url = str(getattr(config, "qdrant_url"))
        timeout = int(getattr(config, "qdrant_timeout_seconds", 20))
        client = AsyncQdrantClient(
            url=url,
            api_key=api_key,
            timeout=timeout,
            prefer_grpc=False,
        )
        store = cls(
            client,
            prefix=str(getattr(config, "qdrant_collection_prefix", "photomat_literature")),
            dimension=int(getattr(config, "embedding_vector_dim")),
            timeout_seconds=timeout,
            snapshot_base_url=url,
            snapshot_api_key=api_key,
        )
        return store

    def _documents_alias(self) -> str:
        return f"{self.prefix}_documents_current"

    def _passages_alias(self) -> str:
        return f"{self.prefix}_passages_current"

    def _generation_names(self, fingerprint: str) -> tuple[str, str]:
        suffix = fingerprint[:12]
        return (
            f"{self.prefix}_documents_{suffix}",
            f"{self.prefix}_passages_{suffix}",
        )

    async def _collection_exists(self, name: str) -> bool:
        method = getattr(self._client, "collection_exists", None)
        if method is not None:
            return bool(await method(name))
        try:
            await self._client.get_collection(name)
        except Exception:
            return False
        return True

    def _timeout_kwargs(self) -> dict[str, Any]:
        return {"timeout": self.timeout_seconds} if self.timeout_seconds is not None else {}

    def _metadata_for(
        self,
        fingerprint: str,
        *,
        identity: ModelIdentity,
        chunk_schema_version: int,
    ) -> dict[str, Any]:
        # Keep flat keys as well as the namespaced object: Qdrant versions and
        # fake clients expose collection metadata in slightly different shapes.
        dimension = identity.dimension if identity.dimension is not None else self.dimension
        return {
            "photomat_collection_schema_version": COLLECTION_SCHEMA_VERSION,
            "photomat_chunk_schema_version": chunk_schema_version,
            "photomat_model_fingerprint": fingerprint,
            "photomat_sparse_model": self.sparse_model,
            "photomat_dense_dimension": dimension,
            "collection_schema_version": COLLECTION_SCHEMA_VERSION,
            "chunk_schema_version": chunk_schema_version,
            "model_fingerprint": fingerprint,
            "sparse_model": self.sparse_model,
            "dense_dimension": dimension,
        }

    async def _create_collection(
        self,
        name: str,
        *,
        passages: bool,
        identity: ModelIdentity,
        fingerprint: str,
        chunk_schema_version: int,
    ) -> None:
        models = _qdrant_models()
        strict_mode = models.StrictModeConfig(
            enabled=True,
            unindexed_filtering_retrieve=False,
            unindexed_filtering_update=False,
        )
        kwargs: dict[str, Any] = {
            "collection_name": name,
            "shard_number": 1,
            "replication_factor": 1,
            "on_disk_payload": True,
            "strict_mode_config": strict_mode,
            "metadata": self._metadata_for(
                fingerprint,
                identity=identity,
                chunk_schema_version=chunk_schema_version,
            ),
            **self._timeout_kwargs(),
        }
        if passages:
            dimension = identity.dimension if identity.dimension is not None else self.dimension
            if dimension is None or dimension < 1:
                raise QdrantStoreError(
                    "schema_mismatch", "embedding dimension is required for passage collection"
                )
            kwargs["vectors_config"] = {
                "dense": models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                )
            }
            kwargs["sparse_vectors_config"] = {
                "sparse_bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            }
        else:
            kwargs["vectors_config"] = None
        await self._client.create_collection(**kwargs)

    def _payload_index_specs(self, *, passages: bool) -> list[tuple[str, Any]]:
        models = _qdrant_models()
        fields: list[tuple[str, Any]] = [
            ("schema_version", models.PayloadSchemaType.KEYWORD),
            ("record_type", models.PayloadSchemaType.KEYWORD),
            ("workspace_id", models.PayloadSchemaType.KEYWORD),
        ]
        if passages:
            fields.extend(
                [
                    ("document_id", models.PayloadSchemaType.KEYWORD),
                    ("document_revision", models.PayloadSchemaType.KEYWORD),
                    ("ingest_state", models.PayloadSchemaType.KEYWORD),
                    ("passage_id", models.PayloadSchemaType.KEYWORD),
                    ("relative_source_path", models.PayloadSchemaType.KEYWORD),
                    ("authors", models.PayloadSchemaType.KEYWORD),
                    ("year", models.PayloadSchemaType.INTEGER),
                    ("text", models.PayloadSchemaType.TEXT),
                    ("title", models.PayloadSchemaType.TEXT),
                    ("section", models.PayloadSchemaType.TEXT),
                    ("heading_path", models.PayloadSchemaType.TEXT),
                    ("page_start", models.PayloadSchemaType.INTEGER),
                    ("page_end", models.PayloadSchemaType.INTEGER),
                    ("previous_passage_id", models.PayloadSchemaType.KEYWORD),
                    ("next_passage_id", models.PayloadSchemaType.KEYWORD),
                    ("model_fingerprint", models.PayloadSchemaType.KEYWORD),
                    ("limitations", models.PayloadSchemaType.TEXT),
                ]
            )
        else:
            fields.extend(
                [
                    ("document_id", models.PayloadSchemaType.KEYWORD),
                    ("relative_source_path", models.PayloadSchemaType.KEYWORD),
                    ("file_name", models.PayloadSchemaType.KEYWORD),
                    ("content_sha256", models.PayloadSchemaType.KEYWORD),
                    ("status", models.PayloadSchemaType.KEYWORD),
                    ("authors", models.PayloadSchemaType.KEYWORD),
                    ("title", models.PayloadSchemaType.TEXT),
                    ("year", models.PayloadSchemaType.INTEGER),
                    ("num_pages", models.PayloadSchemaType.INTEGER),
                    ("chunk_count", models.PayloadSchemaType.INTEGER),
                    ("model_fingerprint", models.PayloadSchemaType.KEYWORD),
                    ("indexed_at", models.PayloadSchemaType.DATETIME),
                    ("last_error", models.PayloadSchemaType.TEXT),
                ]
            )
        return fields

    def _payload_index_type(self, info: Any, field_name: str) -> Any:
        payload_schema = _record_attr(info, "payload_schema", None)
        if not isinstance(payload_schema, Mapping):
            return None
        actual = payload_schema.get(field_name)
        if actual is None:
            return None
        actual_type = _record_attr(actual, "data_type", None)
        if actual_type is None:
            actual_type = _record_attr(_record_attr(actual, "params", None), "type", None)
        return actual_type

    def _payload_index_matches(self, info: Any, field_name: str, expected_type: Any) -> bool:
        actual_type = self._payload_index_type(info, field_name)
        return actual_type is not None and str(_enum_value(actual_type)).lower() == str(
            _enum_value(expected_type)
        ).lower()

    def _validate_payload_schema(self, name: str, info: Any, *, passages: bool) -> None:
        payload_schema = _record_attr(info, "payload_schema", None)
        if not isinstance(payload_schema, Mapping) or not payload_schema:
            raise QdrantStoreError(
                "schema_mismatch", f"collection {name!r} payload indexes cannot be inspected"
            )
        expected = dict(self._payload_index_specs(passages=passages))
        for field_name, expected_type in expected.items():
            if not self._payload_index_matches(info, field_name, expected_type):
                raise QdrantStoreError(
                    "schema_mismatch", f"payload index {field_name!r} is incompatible"
                )

    def _validate_collection_shape(
        self,
        name: str,
        info: Any,
        *,
        passages: bool,
        expected_dimension: int | None = None,
    ) -> None:
        metadata = _metadata(info)
        config = _record_attr(info, "config", None)
        params = _record_attr(config, "params", None)
        required_metadata = (
            "collection_schema_version",
            "chunk_schema_version",
            "model_fingerprint",
            "sparse_model",
        )
        for metadata_key in required_metadata:
            metadata_value = metadata.get(metadata_key)
            if metadata_value is None or metadata_value == "":
                metadata_value = metadata.get(f"photomat_{metadata_key}")
            if metadata_value is None or metadata_value == "":
                raise QdrantStoreError(
                    "schema_mismatch", f"collection {name!r} metadata cannot be inspected"
                )
        if passages and metadata.get("dense_dimension") is None and metadata.get(
            "photomat_dense_dimension"
        ) is None:
            raise QdrantStoreError(
                "schema_mismatch", f"collection {name!r} vector dimension metadata is missing"
            )
        if params is None:
            raise QdrantStoreError(
                "schema_mismatch", f"collection {name!r} schema cannot be inspected"
            )

        on_disk_payload = _record_attr(params, "on_disk_payload", None)
        if on_disk_payload is None:
            on_disk_payload = _record_attr(config, "on_disk_payload", None)
        if on_disk_payload is None:
            on_disk_payload = _record_attr(info, "on_disk_payload", None)
        if on_disk_payload is None or bool(on_disk_payload) is not True:
            raise QdrantStoreError(
                "schema_mismatch", f"collection {name!r} must keep payload on disk"
            )

        strict_mode = _record_attr(config, "strict_mode_config", None)
        if strict_mode is None:
            strict_mode = _record_attr(info, "strict_mode_config", None)
        if strict_mode is None:
            raise QdrantStoreError(
                "schema_mismatch", f"collection {name!r} strict mode cannot be inspected"
            )
        strict_values = (
            ("enabled", True),
            ("unindexed_filtering_retrieve", False),
            ("unindexed_filtering_update", False),
        )
        for field_name, expected_value in strict_values:
            actual_value = _record_attr(strict_mode, field_name, None)
            if actual_value is None or bool(actual_value) is not expected_value:
                raise QdrantStoreError(
                    "schema_mismatch", f"collection {name!r} has incompatible strict mode"
                )

        self._validate_payload_schema(name, info, passages=passages)
        if passages:
            metadata_dimension = metadata.get("dense_dimension") or metadata.get(
                "photomat_dense_dimension"
            )
            if metadata_dimension is None:
                raise QdrantStoreError(
                    "schema_mismatch", f"collection {name!r} vector dimension metadata is missing"
                )
            try:
                parsed_metadata_dimension = int(metadata_dimension)
            except (TypeError, ValueError) as exc:
                raise QdrantStoreError(
                    "schema_mismatch", f"collection {name!r} has an unreadable vector dimension"
                ) from exc
            if parsed_metadata_dimension < 1 or (
                expected_dimension is not None
                and parsed_metadata_dimension != expected_dimension
            ):
                raise QdrantStoreError(
                    "schema_mismatch", f"collection {name!r} has an incompatible vector dimension"
                )
        actual_vectors = _record_attr(params, "vectors", None)
        actual_sparse = _record_attr(params, "sparse_vectors", None)
        if passages:
            if not isinstance(actual_vectors, Mapping) or "dense" not in actual_vectors:
                raise QdrantStoreError(
                    "schema_mismatch", f"collection {name!r} is missing named dense vectors"
                )
            dense_config = actual_vectors["dense"]
            actual_size = _record_attr(dense_config, "size", None)
            if actual_size is None:
                raise QdrantStoreError(
                    "schema_mismatch", f"collection {name!r} dense vector size cannot be inspected"
                )
            try:
                parsed_size = int(actual_size)
            except (TypeError, ValueError) as exc:
                raise QdrantStoreError(
                    "schema_mismatch", f"collection {name!r} has an unreadable vector dimension"
                ) from exc
            if parsed_size < 1 or (
                expected_dimension is not None and parsed_size != expected_dimension
            ):
                raise QdrantStoreError(
                    "schema_mismatch", f"collection {name!r} has an incompatible vector dimension"
                )
            distance = _record_attr(dense_config, "distance", None)
            if distance is None:
                raise QdrantStoreError(
                    "schema_mismatch", f"collection {name!r} vector distance cannot be inspected"
                )
            if str(_enum_value(distance)).lower() != "cosine":
                raise QdrantStoreError(
                    "schema_mismatch", f"collection {name!r} has an incompatible vector distance"
                )
            dense_on_disk = _record_attr(dense_config, "on_disk", None)
            if dense_on_disk is None or bool(dense_on_disk) is not True:
                raise QdrantStoreError(
                    "schema_mismatch", f"collection {name!r} dense vectors must be on disk"
                )
            if not isinstance(actual_sparse, Mapping) or "sparse_bm25" not in actual_sparse:
                raise QdrantStoreError(
                    "schema_mismatch", f"collection {name!r} is missing sparse BM25 vectors"
                )
            modifier = _record_attr(actual_sparse["sparse_bm25"], "modifier", None)
            if modifier is None or str(_enum_value(modifier)).lower() != "idf":
                raise QdrantStoreError(
                    "schema_mismatch", f"collection {name!r} has a non-IDF sparse vector"
                )
        elif actual_vectors not in (None, {}):
            raise QdrantStoreError(
                "schema_mismatch", f"collection {name!r} must be vectorless"
            )
        for parameter_name in ("shard_number", "replication_factor"):
            actual_value = _record_attr(params, parameter_name, None)
            if actual_value is None or int(actual_value) != 1:
                raise QdrantStoreError(
                    "schema_mismatch", f"collection {name!r} has incompatible {parameter_name}"
                )

    async def _validate_existing_collection(
        self,
        name: str,
        *,
        passages: bool,
        identity: ModelIdentity,
        fingerprint: str,
        chunk_schema_version: int,
    ) -> None:
        try:
            info = await self._client.get_collection(name)
        except Exception as exc:
            raise QdrantStoreError(
                "collection_missing", f"collection {name!r} cannot be read"
            ) from exc
        metadata = _metadata(info)
        self._validate_collection_shape(
            name,
            info,
            passages=passages,
            expected_dimension=identity.dimension if identity.dimension is not None else self.dimension,
        )
        actual_fingerprint = (
            metadata.get("model_fingerprint")
            or metadata.get("photomat_model_fingerprint")
        )
        if actual_fingerprint and actual_fingerprint != fingerprint:
            raise QdrantStoreError(
                "model_fingerprint_mismatch",
                f"collection {name!r} has a different model fingerprint",
            )
        actual_schema = metadata.get("collection_schema_version") or metadata.get(
            "photomat_collection_schema_version"
        )
        actual_chunk = metadata.get("chunk_schema_version") or metadata.get(
            "photomat_chunk_schema_version"
        )
        if actual_schema is not None and int(actual_schema) != COLLECTION_SCHEMA_VERSION:
            raise QdrantStoreError(
                "schema_mismatch", f"collection {name!r} has an incompatible schema"
            )
        if actual_chunk is not None and int(actual_chunk) != chunk_schema_version:
            raise QdrantStoreError(
                "schema_mismatch", f"collection {name!r} has an incompatible chunk schema"
            )

    async def _ensure_payload_indexes(self, name: str, *, passages: bool) -> None:
        fields = self._payload_index_specs(passages=passages)
        existing_schema: Mapping[str, Any] = {}
        try:
            info = await self._client.get_collection(name)
            payload_schema = _record_attr(info, "payload_schema", None)
            if isinstance(payload_schema, Mapping):
                existing_schema = payload_schema
        except Exception as exc:
            raise QdrantStoreError(
                "schema_mismatch", f"collection {name!r} payload indexes cannot be inspected"
            ) from exc
        for field_name, field_schema in fields:
            if field_name in existing_schema:
                # _validate_existing_collection already checked the full
                # inspectable schema.  Skip exact existing indexes instead of
                # relying on an error string from create_payload_index.
                continue
            try:
                await self._client.create_payload_index(
                    collection_name=name,
                    field_name=field_name,
                    field_schema=field_schema,
                    wait=True,
                    **self._timeout_kwargs(),
                )
            except Exception as exc:
                status_code = _record_attr(exc, "status_code", None)
                error_name = type(exc).__name__
                if status_code != 409 and error_name not in {
                    "AlreadyExists",
                    "AlreadyExistsError",
                }:
                    raise QdrantStoreError(
                        "schema_mismatch", f"payload index {field_name!r} is incompatible"
                    ) from exc
                try:
                    info = await self._client.get_collection(name)
                except Exception as inspect_exc:
                    raise QdrantStoreError(
                        "schema_mismatch", f"payload index {field_name!r} cannot be inspected"
                    ) from inspect_exc
                if not self._payload_index_matches(info, field_name, field_schema):
                    raise QdrantStoreError(
                        "schema_mismatch", f"payload index {field_name!r} is incompatible"
                    ) from exc

    async def ensure_generation(
        self, *, identity: ModelIdentity, chunk_schema_version: int
    ) -> CollectionGeneration:
        fingerprint = collection_fingerprint(
            identity,
            chunk_schema_version,
            prefix=self.prefix,
            sparse_model=self.sparse_model,
        )
        documents, passages = self._generation_names(fingerprint)
        generation = CollectionGeneration(
            fingerprint=fingerprint,
            documents_physical=documents,
            passages_physical=passages,
            documents_alias=self._documents_alias(),
            passages_alias=self._passages_alias(),
        )
        self._generations[documents] = generation
        self._generations[passages] = generation
        if await self._collection_exists(documents):
            await self._validate_existing_collection(
                documents,
                passages=False,
                identity=identity,
                fingerprint=fingerprint,
                chunk_schema_version=chunk_schema_version,
            )
        else:
            await self._create_collection(
                documents,
                passages=False,
                identity=identity,
                fingerprint=fingerprint,
                chunk_schema_version=chunk_schema_version,
            )
        if await self._collection_exists(passages):
            await self._validate_existing_collection(
                passages,
                passages=True,
                identity=identity,
                fingerprint=fingerprint,
                chunk_schema_version=chunk_schema_version,
            )
        else:
            await self._create_collection(
                passages,
                passages=True,
                identity=identity,
                fingerprint=fingerprint,
                chunk_schema_version=chunk_schema_version,
            )
        await self._ensure_payload_indexes(documents, passages=False)
        await self._ensure_payload_indexes(passages, passages=True)
        models = _qdrant_models()
        generation_payload = {
            "schema_version": COLLECTION_SCHEMA_VERSION,
            "record_type": "generation",
            "model_fingerprint": fingerprint,
            "chunk_schema_version": chunk_schema_version,
            "sparse_model": self.sparse_model,
            "dense_dimension": identity.dimension if identity.dimension is not None else self.dimension,
            "documents_physical": documents,
            "passages_physical": passages,
        }
        await self._client.upsert(
            collection_name=documents,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid5(GENERATION_POINT_NAMESPACE, fingerprint)),
                    vector={},
                    payload=generation_payload,
                )
            ],
            wait=True,
            **self._timeout_kwargs(),
        )
        await self.switch_current_generation(generation)
        return generation

    async def _alias_map(self) -> dict[str, str]:
        method = getattr(self._client, "get_aliases", None)
        if method is None:
            return {}
        response = await method()
        aliases = _record_attr(response, "aliases", response)
        result: dict[str, str] = {}
        if isinstance(aliases, Mapping):
            return {str(key): str(value) for key, value in aliases.items()}
        if aliases is None:
            return result
        for alias in aliases:
            alias_name = _record_attr(alias, "alias_name", None)
            collection_name = _record_attr(alias, "collection_name", None)
            if alias_name is not None and collection_name is not None:
                result[str(alias_name)] = str(collection_name)
        return result

    async def switch_current_generation(self, generation: CollectionGeneration) -> None:
        if not await self._collection_exists(generation.documents_physical) or not await self._collection_exists(
            generation.passages_physical
        ):
            raise QdrantStoreError("collection_missing", "generation collections do not both exist")
        pair_metadata: dict[str, dict[str, Any]] = {}
        for physical_name in (generation.documents_physical, generation.passages_physical):
            try:
                info = await self._client.get_collection(physical_name)
            except Exception as exc:
                raise QdrantStoreError(
                    "collection_missing", f"collection {physical_name!r} cannot be read"
                ) from exc
            metadata = _metadata(info)
            pair_metadata[physical_name] = metadata
            self._validate_collection_shape(
                physical_name,
                info,
                passages=physical_name == generation.passages_physical,
                expected_dimension=self.dimension,
            )
            actual_fingerprint = metadata.get("model_fingerprint") or metadata.get(
                "photomat_model_fingerprint"
            )
            if actual_fingerprint and actual_fingerprint != generation.fingerprint:
                raise QdrantStoreError(
                    "model_fingerprint_mismatch",
                    f"collection {physical_name!r} has a different model fingerprint",
                )
            if not actual_fingerprint and not physical_name.endswith(generation.fingerprint[:12]):
                raise QdrantStoreError(
                    "model_fingerprint_mismatch",
                    f"collection {physical_name!r} does not match the requested generation",
                )
        await self._validate_generation_control(
            generation.documents_physical,
            generation.passages_physical,
            generation.fingerprint,
            pair_metadata[generation.documents_physical],
            pair_metadata[generation.passages_physical],
        )
        models = _qdrant_models()
        aliases = await self._alias_map()
        desired = (
            (generation.documents_alias, generation.documents_physical),
            (generation.passages_alias, generation.passages_physical),
        )
        actions: list[Any] = []
        for alias_name, collection_name in desired:
            current = aliases.get(alias_name)
            if current == collection_name:
                continue
            if current is not None:
                actions.append(
                    models.DeleteAliasOperation(
                        delete_alias=models.DeleteAlias(alias_name=alias_name)
                    )
                )
            actions.append(
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=collection_name,
                        alias_name=alias_name,
                    )
                )
            )
        if actions:
            await self._client.update_collection_aliases(actions, **self._timeout_kwargs())
        self._generations[generation.documents_physical] = generation
        self._generations[generation.passages_physical] = generation

    async def _validate_generation_control(
        self,
        documents_collection: str,
        passages_collection: str,
        fingerprint: str,
        documents_metadata: Mapping[str, Any],
        passages_metadata: Mapping[str, Any],
    ) -> None:
        if _FINGERPRINT_RE.fullmatch(fingerprint) is None:
            raise QdrantStoreError(
                "model_fingerprint_mismatch", "generation control requires a complete fingerprint"
            )
        retrieve = getattr(self._client, "retrieve", None)
        if retrieve is None:
            raise QdrantStoreError(
                "control_point_unavailable", "generation control point cannot be read"
            )
        control_id = str(uuid.uuid5(GENERATION_POINT_NAMESPACE, fingerprint))
        try:
            records = retrieve(
                collection_name=documents_collection,
                ids=[control_id],
                with_payload=True,
                with_vectors=False,
                **self._timeout_kwargs(),
            )
            if inspect.isawaitable(records):
                records = await records
        except Exception as exc:
            raise QdrantStoreError(
                "control_point_unavailable", "generation control point could not be read"
            ) from exc
        if not isinstance(records, Sequence) or not records:
            raise QdrantStoreError(
                "control_point_missing", "generation control point is missing"
            )
        control_payload = _payload(records[0])
        if control_payload.get("record_type") != "generation":
            raise QdrantStoreError(
                "schema_mismatch", "current documents collection has invalid generation control metadata"
            )
        control_documents = str(control_payload.get("documents_physical", ""))
        control_passages = str(control_payload.get("passages_physical", ""))
        if control_documents != documents_collection or control_passages != passages_collection:
            raise QdrantStoreError(
                "schema_mismatch", "generation control metadata does not bind the current pair"
            )
        required_control_fields = (
            "model_fingerprint",
            "chunk_schema_version",
            "sparse_model",
            "dense_dimension",
        )
        if any(control_payload.get(field_name) is None for field_name in required_control_fields):
            raise QdrantStoreError(
                "schema_mismatch", "generation control metadata is incomplete"
            )
        expected_fingerprint = str(
            documents_metadata.get("model_fingerprint")
            or documents_metadata.get("photomat_model_fingerprint")
            or ""
        )
        control_fingerprint = str(control_payload.get("model_fingerprint", ""))
        if _FINGERPRINT_RE.fullmatch(control_fingerprint) is None:
            raise QdrantStoreError(
                "model_fingerprint_mismatch", "generation control metadata has an incomplete fingerprint"
            )
        if control_fingerprint != fingerprint or (
            expected_fingerprint and control_fingerprint != expected_fingerprint
        ):
            raise QdrantStoreError(
                "model_fingerprint_mismatch", "generation control metadata has a different model fingerprint"
            )
        passage_fingerprint = str(
            passages_metadata.get("model_fingerprint")
            or passages_metadata.get("photomat_model_fingerprint")
            or ""
        )
        if passage_fingerprint and control_fingerprint != passage_fingerprint:
            raise QdrantStoreError(
                "model_fingerprint_mismatch", "generation control metadata does not match both collections"
            )
        for metadata_key in ("chunk_schema_version", "sparse_model", "dense_dimension"):
            control_value = control_payload.get(metadata_key)
            document_value = documents_metadata.get(metadata_key) or documents_metadata.get(
                f"photomat_{metadata_key}"
            )
            passage_value = passages_metadata.get(metadata_key) or passages_metadata.get(
                f"photomat_{metadata_key}"
            )
            for collection_value in (document_value, passage_value):
                if (
                    control_value is not None
                    and collection_value is not None
                    and str(control_value) != str(collection_value)
                ):
                    raise QdrantStoreError(
                        "schema_mismatch", "generation control metadata is incompatible with collection metadata"
                    )

    async def resolve_current_generation(self) -> CollectionGeneration | None:
        aliases = await self._alias_map()
        documents = aliases.get(self._documents_alias())
        passages = aliases.get(self._passages_alias())
        if documents is None and passages is None:
            return None
        if documents is None or passages is None:
            raise QdrantStoreError(
                "collection_missing", "documents/passages current aliases are incomplete"
            )
        if documents.rsplit("_", 1)[-1] != passages.rsplit("_", 1)[-1]:
            raise QdrantStoreError(
                "schema_mismatch", "current aliases point to different generations"
            )
        try:
            documents_info = await self._client.get_collection(documents)
            passages_info = await self._client.get_collection(passages)
        except Exception as exc:
            raise QdrantStoreError(
                "collection_missing", "current physical collections cannot be inspected"
            ) from exc
        self._validate_collection_shape(
            documents,
            documents_info,
            passages=False,
            expected_dimension=self.dimension,
        )
        self._validate_collection_shape(
            passages,
            passages_info,
            passages=True,
            expected_dimension=self.dimension,
        )
        documents_metadata = _metadata(documents_info)
        passages_metadata = _metadata(passages_info)
        document_fingerprint = str(
            documents_metadata.get("model_fingerprint")
            or documents_metadata.get("photomat_model_fingerprint")
            or ""
        )
        passage_fingerprint = str(
            passages_metadata.get("model_fingerprint")
            or passages_metadata.get("photomat_model_fingerprint")
            or ""
        )
        suffix = documents.rsplit("_", 1)[-1]
        for physical_name, physical_fingerprint in (
            (documents, document_fingerprint),
            (passages, passage_fingerprint),
        ):
            if physical_fingerprint and not self._fingerprint_matches(
                physical_fingerprint, suffix
            ):
                raise QdrantStoreError(
                    "model_fingerprint_mismatch",
                    f"collection {physical_name!r} has a different model fingerprint",
                )
        if document_fingerprint and passage_fingerprint and not self._fingerprint_matches(
            document_fingerprint, passage_fingerprint
        ):
            raise QdrantStoreError(
                "model_fingerprint_mismatch",
                "current aliases point to different model fingerprints",
            )
        for metadata_key in (
            "collection_schema_version",
            "photomat_collection_schema_version",
            "chunk_schema_version",
            "photomat_chunk_schema_version",
            "sparse_model",
            "photomat_sparse_model",
            "dense_dimension",
            "photomat_dense_dimension",
        ):
            document_value = documents_metadata.get(metadata_key)
            passage_value = passages_metadata.get(metadata_key)
            if (
                document_value is not None
                and passage_value is not None
                and str(document_value) != str(passage_value)
            ):
                raise QdrantStoreError(
                    "schema_mismatch",
                    "current aliases have incompatible generation metadata",
                )
        for physical_name, physical_metadata in (
            (documents, documents_metadata),
            (passages, passages_metadata),
        ):
            actual_schema = physical_metadata.get("collection_schema_version") or physical_metadata.get(
                "photomat_collection_schema_version"
            )
            if actual_schema is not None and int(actual_schema) != COLLECTION_SCHEMA_VERSION:
                raise QdrantStoreError(
                    "schema_mismatch", f"collection {physical_name!r} has an incompatible schema"
                )
            actual_sparse_model = physical_metadata.get("sparse_model") or physical_metadata.get(
                "photomat_sparse_model"
            )
            if actual_sparse_model is not None and str(actual_sparse_model) != self.sparse_model:
                raise QdrantStoreError(
                    "schema_mismatch", f"collection {physical_name!r} has an incompatible sparse model"
                )
        await self._validate_generation_control(
            documents,
            passages,
            document_fingerprint,
            documents_metadata,
            passages_metadata,
        )
        known_documents = self._generations.get(documents)
        known_passages = self._generations.get(passages)
        if (
            known_documents is not None
            and known_passages is not None
            and known_documents.fingerprint != known_passages.fingerprint
        ):
            raise QdrantStoreError(
                "model_fingerprint_mismatch",
                "current aliases point to different model fingerprints",
            )
        if documents not in self._generations or passages not in self._generations:
            fingerprint = document_fingerprint or passage_fingerprint or suffix
            generation = CollectionGeneration(
                fingerprint=fingerprint,
                documents_physical=documents,
                passages_physical=passages,
                documents_alias=self._documents_alias(),
                passages_alias=self._passages_alias(),
            )
            self._generations[documents] = generation
            self._generations[passages] = generation
        return self._generations[documents]

    async def validate_current_generation(self, expected_fingerprint: str) -> None:
        generation = await self.resolve_current_generation()
        if generation is None:
            raise QdrantStoreError("collection_missing", "current collection aliases are not configured")
        if generation.fingerprint != expected_fingerprint and not (
            len(generation.fingerprint) == 12 and expected_fingerprint.startswith(generation.fingerprint)
        ):
            raise QdrantStoreError(
                "model_fingerprint_mismatch", "current collection fingerprint does not match provider"
            )
        if not await self._collection_exists(generation.documents_physical) or not await self._collection_exists(
            generation.passages_physical
        ):
            raise QdrantStoreError("collection_missing", "current physical collections are missing")

    async def _current_or_error(self) -> CollectionGeneration:
        generation = await self.resolve_current_generation()
        if generation is None:
            raise QdrantStoreError("collection_missing", "current collection aliases are not configured")
        return generation

    @staticmethod
    def _fingerprint_matches(actual: str, expected: str) -> bool:
        return actual == expected or (
            len(actual) == 12 and expected.startswith(actual)
        ) or (len(expected) == 12 and actual.startswith(expected))

    @staticmethod
    def _limit(limit: int) -> int:
        if limit < 1:
            raise ValueError("limit must be positive")
        return min(int(limit), MAX_CANDIDATES)

    def _filter(
        self,
        *,
        workspace_id: str | None = None,
        record_type: str | None = None,
        document_id: str | None = None,
        document_revision: str | None = None,
        ingest_state: IngestState | None = None,
        extra: Sequence[Any] = (),
    ) -> Any:
        models = _qdrant_models()
        must: list[Any] = []
        for key, value in (
            ("workspace_id", workspace_id),
            ("record_type", record_type),
            ("document_id", document_id),
            ("document_revision", document_revision),
            ("ingest_state", _enum_value(ingest_state) if ingest_state is not None else None),
        ):
            if value is not None:
                must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
        must.extend(extra)
        return models.Filter(must=must)

    def _passage_filter(
        self,
        *,
        workspace_id: str,
        document_id: str | None = None,
        document_revision: str | None = None,
        ingest_state: IngestState | None = None,
        revision_except: str | None = None,
        extra: Sequence[Any] = (),
    ) -> Any:
        if not isinstance(workspace_id, str) or not workspace_id.strip():
            raise ValueError("workspace_id must be a non-empty string")
        models = _qdrant_models()
        revision_conditions: list[Any] = []
        if revision_except is not None:
            revision_conditions.append(
                models.FieldCondition(
                    key="document_revision",
                    match=models.MatchExcept(except_=[revision_except]),
                )
            )
        return self._filter(
            workspace_id=workspace_id,
            record_type="passage",
            document_id=document_id,
            document_revision=document_revision,
            ingest_state=ingest_state,
            extra=(*extra, *revision_conditions),
        )

    async def list_document_manifests(self, workspace_id: str) -> dict[str, DocumentManifest]:
        generation = await self._current_or_error()
        scroll_filter = self._filter(workspace_id=workspace_id, record_type="document")
        records: list[Any] = []
        offset: Any = None
        while True:
            result = await self._client.scroll(
                collection_name=generation.documents_alias,
                scroll_filter=scroll_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
                **self._timeout_kwargs(),
            )
            if isinstance(result, tuple):
                page, offset = result
            else:
                page, offset = _record_attr(result, "points", []), _record_attr(result, "next_page_offset", None)
            records.extend(page or [])
            if offset is None:
                break
        return {
            manifest.document_id: manifest
            for record in records
            if (manifest := self._manifest_from_record(record)) is not None
        }

    def _manifest_from_record(self, record: Any) -> DocumentManifest | None:
        payload = _payload(record)
        if payload.get("record_type") != "document":
            return None
        try:
            return DocumentManifest(
                schema_version=int(payload.get("schema_version", 1)),
                record_type="document",
                workspace_id=str(payload["workspace_id"]),
                document_id=str(payload.get("document_id", _record_attr(record, "id", ""))),
                relative_source_path=str(payload["relative_source_path"]),
                file_name=str(payload.get("file_name", "")),
                content_sha256=str(payload["content_sha256"]),
                status=DocumentStatus(str(payload.get("status", "pending"))),
                title=str(payload.get("title", "")),
                authors=tuple(str(author) for author in payload.get("authors", [])),
                year=payload.get("year"),
                num_pages=int(payload.get("num_pages", 0)),
                chunk_count=int(payload.get("chunk_count", 0)),
                model_fingerprint=str(payload.get("model_fingerprint", "")),
                indexed_at=_datetime_value(payload.get("indexed_at")),
                last_error=str(payload.get("last_error", "")),
            )
        except (KeyError, TypeError, ValueError):
            return None

    async def upsert_document(self, manifest: DocumentManifest, *, wait: bool = True) -> None:
        generation = await self._current_or_error()
        if not self._fingerprint_matches(manifest.model_fingerprint, generation.fingerprint):
            raise QdrantStoreError(
                "model_fingerprint_mismatch",
                "document manifest belongs to a different collection generation",
            )
        models = _qdrant_models()
        await self._client.upsert(
            collection_name=generation.documents_alias,
            points=[
                models.PointStruct(
                    id=manifest.document_id,
                    vector={},
                    payload=manifest.to_payload(),
                )
            ],
            wait=wait,
            **self._timeout_kwargs(),
        )

    async def upsert_passages(self, points: Sequence[PassagePoint], *, batch_size: int) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not points:
            return
        generation = await self._current_or_error()
        if any(
            not self._fingerprint_matches(point.model_fingerprint, generation.fingerprint)
            for point in points
        ):
            raise QdrantStoreError(
                "model_fingerprint_mismatch",
                "passage points belong to a different collection generation",
            )
        models = _qdrant_models()
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            qdrant_points = [
                models.PointStruct(
                    id=point.passage_id,
                    vector={
                        "dense": list(point.dense),
                        "sparse_bm25": models.Document(
                            text=point.text,
                            model=self.sparse_model,
                        ),
                    },
                    payload=point.to_payload(),
                )
                for point in batch
            ]
            await self._client.upsert(
                collection_name=generation.passages_alias,
                points=qdrant_points,
                wait=True,
                **self._timeout_kwargs(),
            )

    async def count_revision(
        self,
        document_id: str,
        revision: str,
        state: IngestState,
        *,
        workspace_id: str,
    ) -> int:
        generation = await self._current_or_error()
        result = await self._client.count(
            collection_name=generation.passages_alias,
            count_filter=self._passage_filter(
                workspace_id=workspace_id,
                document_id=document_id,
                document_revision=revision,
                ingest_state=state,
            ),
            exact=True,
            **self._timeout_kwargs(),
        )
        return int(result if isinstance(result, int) else _record_attr(result, "count", 0))

    async def set_revision_state(
        self,
        document_id: str,
        revision: str,
        state: IngestState,
        *,
        workspace_id: str,
    ) -> None:
        generation = await self._current_or_error()
        await self._client.set_payload(
            collection_name=generation.passages_alias,
            payload={"ingest_state": state.value},
            points=self._passage_filter(
                workspace_id=workspace_id,
                document_id=document_id,
                document_revision=revision,
            ),
            wait=True,
            **self._timeout_kwargs(),
        )

    async def delete_other_revisions(
        self,
        document_id: str,
        keep_revision: str,
        *,
        workspace_id: str,
    ) -> None:
        generation = await self._current_or_error()
        await self._client.delete(
            collection_name=generation.passages_alias,
            points_selector=self._passage_filter(
                workspace_id=workspace_id,
                document_id=document_id,
                revision_except=keep_revision,
            ),
            wait=True,
            **self._timeout_kwargs(),
        )

    async def delete_document_passages(
        self, document_id: str, *, workspace_id: str
    ) -> None:
        generation = await self._current_or_error()
        await self._client.delete(
            collection_name=generation.passages_alias,
            points_selector=self._passage_filter(
                workspace_id=workspace_id,
                document_id=document_id,
            ),
            wait=True,
            **self._timeout_kwargs(),
        )

    def _candidate_from_record(self, record: Any) -> SearchCandidate:
        payload = _payload(record)
        return SearchCandidate(
            passage_id=str(_record_attr(record, "id", payload.get("passage_id", ""))),
            score=float(_record_attr(record, "score", 0.0)),
            payload=payload,
        )

    @staticmethod
    def _query_points(response: Any) -> list[Any]:
        if isinstance(response, Sequence) and not isinstance(response, (str, bytes, bytearray)):
            return list(response)
        points = _record_attr(response, "points", None)
        if points is not None:
            return list(points)
        result = _record_attr(response, "result", [])
        return list(result or [])

    async def dense_candidates(
        self, dense: Sequence[float], *, workspace_id: str, limit: int
    ) -> list[SearchCandidate]:
        generation = await self._current_or_error()
        cap = self._limit(limit)
        query_filter = self._passage_filter(
            workspace_id=workspace_id,
            ingest_state=IngestState.READY,
        )
        response = await self._client.query_points(
            collection_name=generation.passages_alias,
            query=list(dense),
            using="dense",
            query_filter=query_filter,
            limit=cap,
            with_payload=True,
            with_vectors=False,
            **self._timeout_kwargs(),
        )
        return [self._candidate_from_record(record) for record in self._query_points(response)[:cap]]

    async def sparse_candidates(
        self, query: str, *, workspace_id: str, limit: int
    ) -> list[SearchCandidate]:
        generation = await self._current_or_error()
        cap = self._limit(limit)
        models = _qdrant_models()
        query_filter = self._passage_filter(
            workspace_id=workspace_id,
            ingest_state=IngestState.READY,
        )
        response = await self._client.query_points(
            collection_name=generation.passages_alias,
            query=models.Document(text=query, model=self.sparse_model),
            using="sparse_bm25",
            query_filter=query_filter,
            limit=cap,
            with_payload=True,
            with_vectors=False,
            **self._timeout_kwargs(),
        )
        return [self._candidate_from_record(record) for record in self._query_points(response)[:cap]]

    async def hybrid_candidates(
        self,
        query: str,
        dense: Sequence[float],
        *,
        workspace_id: str,
        limit: int,
    ) -> list[SearchCandidate]:
        generation = await self._current_or_error()
        cap = self._limit(limit)
        models = _qdrant_models()
        query_filter = self._passage_filter(
            workspace_id=workspace_id,
            ingest_state=IngestState.READY,
        )
        prefetch = [
            models.Prefetch(
                query=list(dense),
                using="dense",
                filter=query_filter,
                limit=cap,
            ),
            models.Prefetch(
                query=models.Document(text=query, model=self.sparse_model),
                using="sparse_bm25",
                filter=query_filter,
                limit=cap,
            ),
        ]
        response = await self._client.query_points(
            collection_name=generation.passages_alias,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
            limit=cap,
            with_payload=True,
            with_vectors=False,
            **self._timeout_kwargs(),
        )
        return [self._candidate_from_record(record) for record in self._query_points(response)[:cap]]

    def _passage_from_record(self, record: Any) -> PassagePoint | None:
        payload = _payload(record)
        if payload.get("record_type") != "passage":
            return None
        vector = _record_attr(record, "vector", {})
        dense: Sequence[float] = ()
        if isinstance(vector, Mapping):
            raw_dense = vector.get("dense", ())
            dense = raw_dense if isinstance(raw_dense, Sequence) else ()
        elif isinstance(vector, Sequence) and not isinstance(vector, (str, bytes, bytearray)):
            dense = vector
        try:
            return PassagePoint(
                schema_version=int(payload.get("schema_version", 1)),
                record_type="passage",
                workspace_id=str(payload["workspace_id"]),
                document_id=str(payload["document_id"]),
                document_revision=str(payload["document_revision"]),
                ingest_state=IngestState(str(payload.get("ingest_state", "staged"))),
                passage_id=str(payload.get("passage_id", _record_attr(record, "id", ""))),
                chunk_index=int(payload.get("chunk_index", 0)),
                text=str(payload.get("text", "")),
                title=str(payload.get("title", "")),
                authors=tuple(str(author) for author in payload.get("authors", [])),
                year=payload.get("year"),
                section=str(payload.get("section", "")),
                heading_path=str(payload.get("heading_path", "")),
                page_start=payload.get("page_start"),
                page_end=payload.get("page_end"),
                previous_passage_id=payload.get("previous_passage_id"),
                next_passage_id=payload.get("next_passage_id"),
                relative_source_path=str(payload["relative_source_path"]),
                model_fingerprint=str(payload.get("model_fingerprint", "")),
                limitations=tuple(str(item) for item in payload.get("limitations", [])),
                dense=tuple(float(value) for value in dense),
            )
        except (KeyError, TypeError, ValueError):
            return None

    async def retrieve_passages(
        self, workspace_id: str, passage_ids: Sequence[str]
    ) -> list[PassagePoint]:
        if not passage_ids:
            return []
        generation = await self._current_or_error()
        bounded_ids = list(dict.fromkeys(passage_ids))[:MAX_CANDIDATES]
        models = _qdrant_models()
        records_result = await self._client.scroll(
            collection_name=generation.passages_alias,
            scroll_filter=self._passage_filter(
                workspace_id=workspace_id,
                ingest_state=IngestState.READY,
                extra=(models.HasIdCondition(has_id=bounded_ids),),
            ),
            limit=len(bounded_ids),
            offset=None,
            with_payload=True,
            with_vectors=True,
            **self._timeout_kwargs(),
        )
        if isinstance(records_result, tuple):
            records = records_result[0]
        else:
            records = _record_attr(records_result, "points", [])
        result: list[PassagePoint] = []
        for record in records:
            payload = _payload(record)
            if (
                payload.get("workspace_id") != workspace_id
                or payload.get("record_type") != "passage"
                or payload.get("ingest_state") != IngestState.READY.value
            ):
                continue
            point = self._passage_from_record(record)
            if point is not None:
                result.append(point)
        return result

    async def _download_snapshot(
        self,
        collection_name: str,
        snapshot_name: str,
        destination: Path,
    ) -> None:
        method = getattr(self._client, "download_snapshot", None)
        if method is not None:
            try:
                result = method(collection_name, snapshot_name)
            except TypeError:
                result = method(collection_name, snapshot_name, destination)
            if inspect.isawaitable(result):
                result = await result
            if isinstance(result, (bytes, bytearray)):
                destination.write_bytes(bytes(result))
                return
            if isinstance(result, str):
                source = Path(result)
                if source.is_file():
                    destination.write_bytes(source.read_bytes())
                    return
            if isinstance(result, Path):
                if result.is_file():
                    destination.write_bytes(result.read_bytes())
                    return
            content = _record_attr(result, "content", None)
            if isinstance(content, (bytes, bytearray)):
                destination.write_bytes(bytes(content))
                return
            if destination.is_file():
                return
            raise QdrantStoreError(
                "snapshot_download_failed", f"snapshot {snapshot_name!r} was not downloaded"
            )

        if self.snapshot_base_url is None:
            raise QdrantStoreError(
                "snapshot_download_failed",
                "Qdrant REST snapshot base URL is not configured",
            )
        url = (
            f"{self.snapshot_base_url}/collections/{quote(collection_name, safe='')}"
            f"/snapshots/{quote(snapshot_name, safe='')}"
        )
        headers = (
            {"api-key": self._snapshot_api_key}
            if self._snapshot_api_key
            else {}
        )
        timeout = self.timeout_seconds if self.timeout_seconds is not None else 20
        transport = self._snapshot_transport
        if transport is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover - optional capability path
                raise QdrantStoreError(
                    "snapshot_download_failed", "httpx is required for REST snapshot download"
                ) from exc
            async with httpx.AsyncClient(timeout=timeout) as http:
                async with http.stream("GET", url, headers=headers) as response:
                    await self._write_snapshot_response(response, destination, snapshot_name)
            return
        get = getattr(transport, "get", None)
        if get is None:
            raise QdrantStoreError(
                "snapshot_download_failed", "snapshot transport does not expose GET"
            )
        response = get(url, headers=headers, timeout=timeout)
        if inspect.isawaitable(response):
            response = await response
        await self._write_snapshot_response(response, destination, snapshot_name)

    async def _write_snapshot_response(
        self, response: Any, destination: Path, snapshot_name: str
    ) -> None:
        status_code = _record_attr(response, "status_code", None)
        if status_code is None or not 200 <= int(status_code) < 300:
            raise QdrantStoreError(
                "snapshot_download_failed",
                f"snapshot {snapshot_name!r} returned an invalid HTTP status",
            )
        aiter_bytes = getattr(response, "aiter_bytes", None)
        if aiter_bytes is not None:
            with destination.open("wb") as handle:
                async for chunk in aiter_bytes(1024 * 1024):
                    if isinstance(chunk, (bytes, bytearray)):
                        handle.write(bytes(chunk))
            return
        content = _record_attr(response, "content", None)
        if isinstance(content, (bytes, bytearray)):
            destination.write_bytes(bytes(content))
            return
        raise QdrantStoreError(
            "snapshot_download_failed", f"snapshot {snapshot_name!r} was not downloaded"
        )

    async def create_current_snapshots(self, output_dir: Path) -> SnapshotManifest:
        generation = await self._current_or_error()
        output_dir.mkdir(parents=True, exist_ok=True)
        created: list[SnapshotFile] = []
        for collection_name in (
            generation.documents_physical,
            generation.passages_physical,
        ):
            snapshot = await self._client.create_snapshot(
                collection_name=collection_name,
                wait=True,
                **self._timeout_kwargs(),
            )
            snapshot_name = str(_record_attr(snapshot, "name", "snapshot"))
            safe_name = Path(snapshot_name).name
            final_name = f"{collection_name}--{safe_name}"
            final_path = output_dir / final_name
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{final_name}.", suffix=".tmp", dir=output_dir
            )
            os.close(fd)
            temporary_path = Path(temporary_name)
            try:
                await self._download_snapshot(collection_name, snapshot_name, temporary_path)
                digest = hashlib.sha256(temporary_path.read_bytes()).hexdigest()
                size = temporary_path.stat().st_size
                os.replace(temporary_path, final_path)
                created.append(
                    SnapshotFile(
                        collection=collection_name,
                        file_name=final_name,
                        sha256=digest,
                        size_bytes=size,
                    )
                )
            finally:
                temporary_path.unlink(missing_ok=True)
        manifest = SnapshotManifest(
            generation=generation,
            created_at=datetime.now(timezone.utc),
            files=tuple(created),
        )
        manifest_path = output_dir / "snapshot-manifest.json"
        manifest_data = asdict(manifest)
        manifest_data["created_at"] = manifest.created_at.isoformat()
        manifest_data["files"] = [asdict(item) for item in manifest.files]
        temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
        temporary_manifest.write_text(
            json.dumps(manifest_data, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)
        return manifest


__all__ = [
    "COLLECTION_SCHEMA_VERSION",
    "CollectionGeneration",
    "DocumentManifest",
    "DocumentStatus",
    "IngestState",
    "MAX_CANDIDATES",
    "PassagePoint",
    "QdrantLiteratureStore",
    "QdrantStoreError",
    "SearchCandidate",
    "SnapshotFile",
    "SnapshotManifest",
    "collection_fingerprint",
    "document_id_for",
    "passage_id_for",
    "workspace_id_for",
]
