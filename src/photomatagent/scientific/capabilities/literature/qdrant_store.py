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
import ipaddress
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

from photomatagent.scientific.capabilities.literature.models import (
    DocumentManifest,
    DocumentStatus,
    IngestState,
    LiteratureSourceKind,
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
INGESTION_RUN_POINT_NAMESPACE = uuid.UUID("0c8fc0cc-e4c1-45aa-b5d8-d4dbf5a4d19c")
_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")
_DIAGNOSTIC_SECRET_RE = re.compile(
    r"(?i)\b(api[_ -]?key|access[_ -]?token|authorization|token|password|secret)\b"
    r"\s*[:=]\s*(?:bearer\s+)?[^\s,;]+"
)
_DIAGNOSTIC_BEARER_RE = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_DIAGNOSTIC_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|/)[^\s,;]+")


def _qdrant_loopback(hostname: str) -> bool:
    host = hostname.strip().lower().strip("[]")
    if host in {"localhost", "localhost.localdomain"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def sanitize_qdrant_url(url: str) -> str:
    """Return a credential/query-free URL suitable for status output."""
    value = str(url or "").strip()
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname or ""
        if not hostname:
            return "[invalid-url]"
        host = f"[{hostname}]" if ":" in hostname else hostname
        netloc = host
        if parsed.port is not None:
            netloc = f"{netloc}:{parsed.port}"
        return urlunsplit((parsed.scheme.lower(), netloc, parsed.path, "", ""))
    except ValueError:
        return "[invalid-url]"


def validate_qdrant_url(url: str, *, api_key: str | None = None) -> str:
    """Validate Qdrant endpoint security and return a safe display URL."""
    value = str(url or "").strip()
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        parsed.port  # validates malformed/out-of-range ports
    except ValueError as exc:
        raise QdrantStoreError(
            "qdrant_url_invalid", "Qdrant URL must be an absolute HTTP(S) URL"
        ) from exc
    if (
        any(character.isspace() for character in value)
        or parsed.scheme.lower() not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise QdrantStoreError(
            "qdrant_url_invalid", "Qdrant URL must not contain credentials, query, or fragment"
        )
    loopback = _qdrant_loopback(hostname)
    if not loopback and parsed.scheme.lower() != "https":
        raise QdrantStoreError(
            "qdrant_tls_required", "non-loopback Qdrant endpoints require HTTPS"
        )
    if not loopback and not str(api_key or "").strip():
        raise QdrantStoreError(
            "qdrant_api_key_required", "non-loopback Qdrant endpoints require an API key"
        )
    return sanitize_qdrant_url(value)


def _sanitize_diagnostic(value: Any) -> str:
    """Apply store-boundary redaction independent of ingestion callers."""
    text = str(value).replace("\x00", " ")
    text = _DIAGNOSTIC_SECRET_RE.sub(
        lambda match: f"{match.group(1)}=[redacted]", text
    )
    text = _DIAGNOSTIC_BEARER_RE.sub("Bearer [redacted]", text)
    text = _DIAGNOSTIC_PATH_RE.sub("[path]", text)
    return text[:512]


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
    server_version: str = "unknown"
    schema_version: int = COLLECTION_SCHEMA_VERSION
    fingerprint: str = ""
    physical_collections: tuple[str, ...] = ()
    point_counts: dict[str, int | None] = field(default_factory=dict)
    indexed_vector_counts: dict[str, int | None] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    capacity: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["created_at"] = self.created_at.isoformat()
        value["files"] = [asdict(item) for item in self.files]
        return value


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


def _source_kind_value(value: LiteratureSourceKind | str | None) -> str | None:
    if value is None:
        return None
    try:
        return LiteratureSourceKind(value).value
    except (TypeError, ValueError) as exc:
        raise ValueError("source_kind must be 'fulltext' or 'abstract'") from exc


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


def _stream_file_digest(path: Path) -> tuple[str, int]:
    """Hash a snapshot incrementally so archive size does not bound memory."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _copy_file_stream(source: Path, destination: Path) -> None:
    """Copy a snapshot path in bounded blocks rather than reading it all."""
    with source.open("rb") as source_handle, destination.open("wb") as destination_handle:
        for block in iter(lambda: source_handle.read(1024 * 1024), b""):
            destination_handle.write(block)


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
        # A generation may be built and ingested before it is made visible via
        # the stable ``*_current`` aliases.  Keep that write target separate
        # from the read target so a model migration cannot mix vector spaces.
        self._staging_generation: CollectionGeneration | None = None

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
        validate_qdrant_url(url, api_key=api_key)
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

    def expected_generation(
        self, *, identity: ModelIdentity, chunk_schema_version: int
    ) -> CollectionGeneration:
        """Compute the generation identity without reading or mutating Qdrant."""
        fingerprint = collection_fingerprint(
            identity,
            chunk_schema_version,
            prefix=self.prefix,
            sparse_model=self.sparse_model,
        )
        documents, passages = self._generation_names(fingerprint)
        return CollectionGeneration(
            fingerprint=fingerprint,
            documents_physical=documents,
            passages_physical=passages,
            documents_alias=self._documents_alias(),
            passages_alias=self._passages_alias(),
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

    def _method_timeout_kwargs(self, method: Any) -> dict[str, Any]:
        """Return a timeout only when the target method explicitly supports it.

        qdrant-client 1.19 exposes snapshot operations with ``**kwargs`` for
        compatibility, but rejects unknown arguments at runtime.  Treating a
        catch-all ``**kwargs`` as timeout support would therefore still pass
        an invalid method-level timeout.  Signature inspection keeps this
        adaptation narrow and avoids retrying arbitrary ``TypeError`` values
        raised by the client implementation itself.
        """
        if self.timeout_seconds is None:
            return {}
        try:
            parameter = inspect.signature(method).parameters.get("timeout")
        except (TypeError, ValueError):
            return {}
        if parameter is None or parameter.kind not in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            return {}
        return {"timeout": self.timeout_seconds}

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
            ("source_kind", models.PayloadSchemaType.KEYWORD),
            ("source_record_id", models.PayloadSchemaType.KEYWORD),
            ("doi", models.PayloadSchemaType.KEYWORD),
            ("relevance_tier", models.PayloadSchemaType.KEYWORD),
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
                    ("run_id", models.PayloadSchemaType.KEYWORD),
                    ("generation_fingerprint", models.PayloadSchemaType.KEYWORD),
                    ("relative_root", models.PayloadSchemaType.KEYWORD),
                    ("cursor", models.PayloadSchemaType.KEYWORD),
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
                    ("errors", models.PayloadSchemaType.TEXT),
                ]
            )
        if passages:
            fields.append(("normalized_text_sha256", models.PayloadSchemaType.KEYWORD))
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
        generation = self.expected_generation(
            identity=identity, chunk_schema_version=chunk_schema_version
        )
        fingerprint = generation.fingerprint
        documents = generation.documents_physical
        passages = generation.passages_physical
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
        # Do not switch aliases here.  The caller must explicitly validate the
        # completed staging generation and call ``activate_generation``.
        self._staging_generation = generation
        return generation

    async def _validate_generation_ready(self, generation: CollectionGeneration) -> bool:
        """Reject a staging pair with unresolved document/run failures."""
        if not await self._collection_exists(generation.documents_physical) or not await self._collection_exists(
            generation.passages_physical
        ):
            raise QdrantStoreError(
                "collection_missing", "generation collections do not both exist"
            )
        scroll = getattr(self._client, "scroll", None)
        if scroll is None:
            raise QdrantStoreError(
                "generation_incomplete", "generation completeness cannot be inspected"
            )
        records: list[Any] = []
        offset: Any = None
        try:
            while True:
                response = scroll(
                    collection_name=generation.documents_physical,
                    scroll_filter=None,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                    **self._timeout_kwargs(),
                )
                if inspect.isawaitable(response):
                    response = await response
                if isinstance(response, tuple):
                    page, offset = response
                else:
                    page = _record_attr(response, "points", [])
                    offset = _record_attr(response, "next_page_offset", None)
                records.extend(page or [])
                if offset is None:
                    break
        except Exception as exc:
            raise QdrantStoreError(
                "generation_incomplete", "generation completeness cannot be inspected"
            ) from exc
        unresolved: list[str] = []
        has_ready_passage = False
        for record in records or ():
            payload = _payload(record)
            if payload.get("record_type") in {"document", "ingestion_run"}:
                point_fingerprint = str(
                    payload.get("model_fingerprint", payload.get("generation_fingerprint", ""))
                )
                if point_fingerprint and not self._fingerprint_matches(
                    point_fingerprint, generation.fingerprint
                ):
                    raise QdrantStoreError(
                        "model_fingerprint_mismatch",
                        "generation contains a record from another model fingerprint",
                    )
            if payload.get("record_type") == "document" and str(
                payload.get("status", "")
            ) in {DocumentStatus.PENDING.value, DocumentStatus.STAGED.value, DocumentStatus.FAILED.value}:
                unresolved.append(str(payload.get("relative_source_path", "document")))
            if payload.get("record_type") == "ingestion_run" and (
                payload.get("complete") is not True
                or payload.get("retryable") is True
            ):
                unresolved.append(str(payload.get("run_id", "ingestion_run")))
        if unresolved:
            raise QdrantStoreError(
                "generation_incomplete",
                "generation has unresolved ingestion failures",
            )
        passage_records: list[Any] = []
        passage_offset: Any = None
        try:
            while True:
                response = scroll(
                    collection_name=generation.passages_physical,
                    scroll_filter=None,
                    limit=256,
                    offset=passage_offset,
                    with_payload=True,
                    with_vectors=False,
                    **self._timeout_kwargs(),
                )
                if inspect.isawaitable(response):
                    response = await response
                if isinstance(response, tuple):
                    page, passage_offset = response
                else:
                    page = _record_attr(response, "points", [])
                    passage_offset = _record_attr(response, "next_page_offset", None)
                passage_records.extend(page or [])
                if passage_offset is None:
                    break
        except Exception as exc:
            raise QdrantStoreError(
                "generation_incomplete", "passage completeness cannot be inspected"
            ) from exc
        unresolved_passages: list[str] = []
        for record in passage_records:
            payload = _payload(record)
            if payload.get("record_type") != "passage":
                continue
            point_fingerprint = str(payload.get("model_fingerprint", ""))
            if point_fingerprint and not self._fingerprint_matches(
                point_fingerprint, generation.fingerprint
            ):
                raise QdrantStoreError(
                    "model_fingerprint_mismatch",
                    "generation contains a passage from another model fingerprint",
                )
            if payload.get("ingest_state") == IngestState.STAGED.value:
                unresolved_passages.append(str(payload.get("passage_id", "passage")))
            if payload.get("ingest_state") == IngestState.READY.value:
                has_ready_passage = True
        if unresolved_passages:
            raise QdrantStoreError(
                "generation_incomplete",
                "generation contains unresolved staged passages",
            )
        return has_ready_passage

    async def _activation_ready(
        self,
        generation: CollectionGeneration,
        *,
        allow_empty_bootstrap: bool,
    ) -> None:
        has_ready_passage = await self._validate_generation_ready(generation)
        if not has_ready_passage:
            if not allow_empty_bootstrap:
                raise QdrantStoreError(
                    "generation_incomplete",
                    "generation has no ready indexed passages; use explicit bootstrap for an empty corpus",
                )
            aliases = await self._alias_map()
            current_pair = (
                aliases.get(generation.documents_alias),
                aliases.get(generation.passages_alias),
            )
            target_pair = (
                generation.documents_physical,
                generation.passages_physical,
            )
            if any(value is not None for value in current_pair) and current_pair != target_pair:
                raise QdrantStoreError(
                    "generation_incomplete",
                    "empty bootstrap is allowed only before current aliases exist",
                )

    async def activate_generation(
        self,
        generation: CollectionGeneration,
        *,
        allow_empty_bootstrap: bool = False,
    ) -> None:
        """Validate and atomically expose a completed staging generation."""
        await self._activation_ready(
            generation, allow_empty_bootstrap=allow_empty_bootstrap
        )
        try:
            await self._switch_current_generation(generation)
        except QdrantStoreError as exc:
            if exc.code in {"control_point_missing", "control_point_unavailable"}:
                raise QdrantStoreError(
                    "generation_incomplete",
                    "generation control metadata is incomplete",
                ) from exc
            raise
        self._staging_generation = generation

    def select_staging_generation(self, generation: CollectionGeneration) -> None:
        """Select an already-created physical pair for ingestion writes."""
        if not isinstance(generation, CollectionGeneration):
            raise TypeError("generation must be a CollectionGeneration")
        self._staging_generation = generation

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

    async def switch_current_generation(
        self,
        generation: CollectionGeneration,
        *,
        allow_empty_bootstrap: bool = False,
    ) -> None:
        """Guarded public alias switch; never bypasses completeness checks."""
        await self._activation_ready(
            generation, allow_empty_bootstrap=allow_empty_bootstrap
        )
        await self._switch_current_generation(generation)

    async def _switch_current_generation(self, generation: CollectionGeneration) -> None:
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

    async def _ingestion_or_error(self) -> CollectionGeneration:
        """Return the explicit staging target, falling back to current reads."""
        if self._staging_generation is not None:
            return self._staging_generation
        return await self._current_or_error()

    def _ingestion_collection(
        self, generation: CollectionGeneration, *, passages: bool
    ) -> str:
        """Return a physical staging target or a stable current alias.

        Alias switching is the activation boundary.  All writes that happen
        before activation therefore use physical generation names so an
        incomplete migration can never mutate the active generation.
        """
        if self._staging_generation is generation:
            return (
                generation.passages_physical
                if passages
                else generation.documents_physical
            )
        return generation.passages_alias if passages else generation.documents_alias

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
        source_kind: LiteratureSourceKind | str | None = None,
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
            ("source_kind", _source_kind_value(source_kind)),
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
        source_kind: LiteratureSourceKind | str | None = None,
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
            source_kind=source_kind,
            extra=(*extra, *revision_conditions),
        )

    async def list_document_manifests(
        self,
        workspace_id: str,
        *,
        generation: CollectionGeneration | None = None,
    ) -> dict[str, DocumentManifest]:
        if not isinstance(workspace_id, str) or not workspace_id.strip():
            raise ValueError("workspace_id must be a non-empty string")
        explicit_generation = generation is not None
        generation = generation or self._staging_generation
        if generation is None:
            generation = await self._current_or_error()
        # Planning a brand-new generation is intentionally read-only.  A
        # missing staging collection means there are no prior manifests yet;
        # it must not be treated as a request to create or activate anything.
        if not await self._collection_exists(generation.documents_physical):
            return {}
        collection_name = (
            generation.documents_physical
            if explicit_generation or self._staging_generation is generation
            else generation.documents_alias
        )
        scroll_filter = self._filter(workspace_id=workspace_id, record_type="document")
        records: list[Any] = []
        offset: Any = None
        while True:
            result = await self._client.scroll(
                collection_name=collection_name,
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

    async def get_document_manifests(
        self,
        workspace_id: str,
        document_ids: Sequence[str],
        *,
        generation: CollectionGeneration | None = None,
    ) -> dict[str, DocumentManifest]:
        """Retrieve a bounded set of document manifests by their point IDs.

        Qdrant's point-ID lookup is intentionally used instead of a scroll so
        an abstract-ingestion batch can compare only the documents it is about
        to process.  The payload is still checked at this boundary because a
        fake/legacy client may return records outside the requested workspace
        or with a different record type.
        """
        if not isinstance(workspace_id, str) or not workspace_id.strip():
            raise ValueError("workspace_id must be a non-empty string")
        if isinstance(document_ids, (str, bytes, bytearray)):
            raise TypeError("document_ids must be a sequence of document IDs")
        requested = list(document_ids)
        if len(requested) > MAX_CANDIDATES:
            raise ValueError(
                f"document_ids cannot contain more than {MAX_CANDIDATES} IDs"
            )
        if not requested:
            return {}
        if any(not isinstance(document_id, str) or not document_id for document_id in requested):
            raise ValueError("document_ids must contain non-empty strings")
        requested_ids = list(dict.fromkeys(requested))

        explicit_generation = generation is not None
        generation = generation or self._staging_generation
        if generation is None:
            generation = await self._current_or_error()
        if not await self._collection_exists(generation.documents_physical):
            return {}
        collection_name = (
            generation.documents_physical
            if explicit_generation or self._staging_generation is generation
            else generation.documents_alias
        )
        retrieved = await self._client.retrieve(
            collection_name=collection_name,
            ids=requested_ids,
            with_payload=True,
            with_vectors=False,
            **self._timeout_kwargs(),
        )
        result: dict[str, DocumentManifest] = {}
        requested_set = set(requested_ids)
        for record in self._query_points(retrieved):
            payload = _payload(record)
            if (
                payload.get("workspace_id") != workspace_id
                or payload.get("record_type") != "document"
            ):
                continue
            payload_document_id = str(
                payload.get("document_id", _record_attr(record, "id", ""))
            )
            if payload_document_id not in requested_set:
                continue
            manifest = self._manifest_from_record(record)
            if manifest is not None and manifest.document_id in requested_set:
                result[manifest.document_id] = manifest
        return result

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
                source_kind=LiteratureSourceKind(
                    payload.get("source_kind", LiteratureSourceKind.FULLTEXT.value)
                ),
                source_record_id=str(payload.get("source_record_id", "")),
                doi=str(payload.get("doi", "")),
                pmid=str(payload.get("pmid", "")),
                pmcid=str(payload.get("pmcid", "")),
                journal=str(payload.get("journal", "")),
                relevance_tier=str(payload.get("relevance_tier", "")),
            )
        except (KeyError, TypeError, ValueError):
            return None

    async def upsert_document(self, manifest: DocumentManifest, *, wait: bool = True) -> None:
        if not isinstance(manifest.workspace_id, str) or not manifest.workspace_id.strip():
            raise ValueError("workspace_id must be a non-empty string")
        generation = await self._ingestion_or_error()
        if not self._fingerprint_matches(manifest.model_fingerprint, generation.fingerprint):
            raise QdrantStoreError(
                "model_fingerprint_mismatch",
                "document manifest belongs to a different collection generation",
            )
        models = _qdrant_models()
        await self._client.upsert(
            collection_name=self._ingestion_collection(generation, passages=False),
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

    async def upsert_passages(
        self,
        points: Sequence[PassagePoint],
        *,
        batch_size: int,
        workspace_id: str | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not points:
            return
        if workspace_id is None:
            # Older internal callers did not pass the scope explicitly.  Keep
            # that compatibility only when all points carry one unambiguous
            # workspace; new ingestion always supplies the keyword.
            point_workspaces = {point.workspace_id for point in points}
            if len(point_workspaces) != 1:
                raise ValueError("workspace_id must be explicit for mixed passage workspaces")
            workspace_id = next(iter(point_workspaces))
        if not isinstance(workspace_id, str) or not workspace_id.strip():
            raise ValueError("workspace_id must be a non-empty string")
        if any(point.workspace_id != workspace_id for point in points):
            raise ValueError("passage point workspace does not match workspace_id")
        generation = await self._ingestion_or_error()
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
                collection_name=self._ingestion_collection(generation, passages=True),
                points=qdrant_points,
                wait=True,
                **self._timeout_kwargs(),
            )

    async def upsert_ingestion_run(self, run: Any, *, wait: bool = True) -> None:
        """Persist bounded ingestion progress as a vectorless control point."""
        workspace_id = getattr(run, "workspace_id", None)
        if not isinstance(workspace_id, str) or not workspace_id.strip():
            raise ValueError("workspace_id must be a non-empty string")
        generation = await self._ingestion_or_error()
        generation_fingerprint = str(getattr(run, "generation_fingerprint", ""))
        if not _FINGERPRINT_RE.fullmatch(generation_fingerprint):
            raise QdrantStoreError(
                "model_fingerprint_mismatch", "ingestion run has an incomplete generation fingerprint"
            )
        if not self._fingerprint_matches(generation_fingerprint, generation.fingerprint):
            raise QdrantStoreError(
                "model_fingerprint_mismatch", "ingestion run belongs to another collection generation"
            )
        stats = getattr(run, "stats", None)
        if stats is None:
            raise ValueError("ingestion run stats are required")
        errors = [
            _sanitize_diagnostic(error)
            for error in tuple(getattr(stats, "errors", ()))[-20:]
        ]
        run_id = str(getattr(run, "run_id", ""))
        if not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        payload = {
            "schema_version": COLLECTION_SCHEMA_VERSION,
            "record_type": "ingestion_run",
            "workspace_id": workspace_id,
            "run_id": run_id,
            "generation": generation_fingerprint,
            "generation_fingerprint": generation_fingerprint,
            "relative_root": str(getattr(run, "relative_root", ".")),
            "cursor": getattr(run, "cursor", None),
            "status": str(getattr(run, "status", "running")),
            "discovered": int(getattr(stats, "discovered", 0)),
            "unchanged": int(getattr(stats, "unchanged", 0)),
            "indexed": int(getattr(stats, "indexed", 0)),
            "failed": int(getattr(stats, "failed", 0)),
            "deleted": int(getattr(stats, "deleted", 0)),
            "chunks": int(getattr(stats, "chunks", 0)),
            "staged_cleanup": int(getattr(stats, "staged_cleanup", 0)),
            "next_cursor": getattr(stats, "next_cursor", None),
            "complete": bool(getattr(stats, "complete", False)),
            "retryable": bool(getattr(stats, "retryable", False)),
            "errors": errors,
        }
        models = _qdrant_models()
        point_id = str(uuid.uuid5(INGESTION_RUN_POINT_NAMESPACE, f"{workspace_id}:{run_id}"))
        await self._client.upsert(
            collection_name=self._ingestion_collection(generation, passages=False),
            points=[models.PointStruct(id=point_id, vector={}, payload=payload)],
            wait=wait,
            **self._timeout_kwargs(),
        )

    async def get_ingestion_run(self, run_id: str, workspace_id: str) -> Any | None:
        """Read one workspace-scoped ingestion control point, if present."""
        if not isinstance(workspace_id, str) or not workspace_id.strip():
            raise ValueError("workspace_id must be a non-empty string")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        generation = await self._ingestion_or_error()
        point_id = str(uuid.uuid5(INGESTION_RUN_POINT_NAMESPACE, f"{workspace_id}:{run_id}"))
        records = await self._client.retrieve(
            collection_name=self._ingestion_collection(generation, passages=False),
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
            **self._timeout_kwargs(),
        )
        if not isinstance(records, Sequence) or not records:
            return None
        payload = _payload(records[0])
        if (
            payload.get("record_type") != "ingestion_run"
            or payload.get("workspace_id") != workspace_id
            or str(payload.get("run_id", "")) != run_id
            or str(payload.get("generation_fingerprint", payload.get("generation", "")))
            != generation.fingerprint
        ):
            return None
        from photomatagent.scientific.capabilities.literature.ingestion import (
            IngestionRunState,
            IngestionStats,
        )

        errors_value = payload.get("errors", ())
        errors: tuple[str, ...]
        if isinstance(errors_value, str):
            errors = (errors_value[:512],) if errors_value else ()
        elif isinstance(errors_value, Sequence):
            errors = tuple(str(item)[:512] for item in list(errors_value)[-20:])
        else:
            errors = ()
        stats = IngestionStats(
            run_id=run_id,
            discovered=int(payload.get("discovered", 0)),
            unchanged=int(payload.get("unchanged", 0)),
            indexed=int(payload.get("indexed", 0)),
            failed=int(payload.get("failed", 0)),
            deleted=int(payload.get("deleted", 0)),
            chunks=int(payload.get("chunks", 0)),
            staged_cleanup=int(payload.get("staged_cleanup", 0)),
            next_cursor=payload.get("next_cursor"),
            complete=bool(payload.get("complete", False)),
            errors=errors,
            retryable=bool(payload.get("retryable", False)),
        )
        return IngestionRunState(
            run_id=run_id,
            workspace_id=workspace_id,
            generation_fingerprint=str(
                payload.get("generation_fingerprint", payload.get("generation", ""))
            ),
            relative_root=str(payload.get("relative_root", ".")),
            cursor=payload.get("cursor"),
            status=str(payload.get("status", "running")),
            stats=stats,
        )

    async def delete_staged_revisions(
        self,
        document_id: str,
        *,
        keep_revision: str | None = None,
        workspace_id: str | None = None,
    ) -> int:
        """Delete only staged revisions, returning the exact pre-delete count."""
        if not isinstance(workspace_id, str) or not workspace_id.strip():
            raise ValueError("workspace_id must be a non-empty string")
        generation = await self._ingestion_or_error()
        extra: tuple[Any, ...] = ()
        if keep_revision is not None:
            if _FINGERPRINT_RE.fullmatch(keep_revision) is None:
                raise ValueError("keep_revision must be a complete SHA-256 fingerprint")
            models = _qdrant_models()
            extra = (
                models.FieldCondition(
                    key="document_revision",
                    match=models.MatchExcept(except_=[keep_revision]),
                ),
            )
        query_filter = self._passage_filter(
            workspace_id=workspace_id,
            document_id=document_id,
            ingest_state=IngestState.STAGED,
            extra=extra,
        )
        result = await self._client.count(
            collection_name=self._ingestion_collection(generation, passages=True),
            count_filter=query_filter,
            exact=True,
            **self._timeout_kwargs(),
        )
        count = int(result if isinstance(result, int) else _record_attr(result, "count", 0))
        if count:
            await self._client.delete(
                collection_name=self._ingestion_collection(generation, passages=True),
                points_selector=query_filter,
                wait=True,
                **self._timeout_kwargs(),
            )
        return count

    async def count_revision(
        self,
        document_id: str,
        revision: str,
        state: IngestState,
        *,
        workspace_id: str,
    ) -> int:
        generation = await self._ingestion_or_error()
        result = await self._client.count(
            collection_name=self._ingestion_collection(generation, passages=True),
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
        generation = await self._ingestion_or_error()
        await self._client.set_payload(
            collection_name=self._ingestion_collection(generation, passages=True),
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
        generation = await self._ingestion_or_error()
        await self._client.delete(
            collection_name=self._ingestion_collection(generation, passages=True),
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
        generation = await self._ingestion_or_error()
        await self._client.delete(
            collection_name=self._ingestion_collection(generation, passages=True),
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
        self,
        dense: Sequence[float],
        *,
        workspace_id: str,
        source_kind: LiteratureSourceKind | str = LiteratureSourceKind.FULLTEXT,
        limit: int,
    ) -> list[SearchCandidate]:
        generation = await self._current_or_error()
        cap = self._limit(limit)
        query_filter = self._passage_filter(
            workspace_id=workspace_id,
            ingest_state=IngestState.READY,
            source_kind=source_kind,
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
        self,
        query: str,
        *,
        workspace_id: str,
        source_kind: LiteratureSourceKind | str = LiteratureSourceKind.FULLTEXT,
        limit: int,
    ) -> list[SearchCandidate]:
        generation = await self._current_or_error()
        cap = self._limit(limit)
        models = _qdrant_models()
        query_filter = self._passage_filter(
            workspace_id=workspace_id,
            ingest_state=IngestState.READY,
            source_kind=source_kind,
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
        source_kind: LiteratureSourceKind | str = LiteratureSourceKind.FULLTEXT,
        limit: int,
    ) -> list[SearchCandidate]:
        generation = await self._current_or_error()
        cap = self._limit(limit)
        models = _qdrant_models()
        query_filter = self._passage_filter(
            workspace_id=workspace_id,
            ingest_state=IngestState.READY,
            source_kind=source_kind,
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
                normalized_text_sha256=str(payload.get("normalized_text_sha256", "")),
                indexed_at=_datetime_value(payload.get("indexed_at")),
                source_kind=LiteratureSourceKind(
                    payload.get("source_kind", LiteratureSourceKind.FULLTEXT.value)
                ),
                source_record_id=str(payload.get("source_record_id", "")),
                doi=str(payload.get("doi", "")),
                pmid=str(payload.get("pmid", "")),
                pmcid=str(payload.get("pmcid", "")),
                journal=str(payload.get("journal", "")),
                relevance_tier=str(payload.get("relevance_tier", "")),
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
                with destination.open("wb") as handle:
                    handle.write(result)
                return
            if isinstance(result, str):
                source = Path(result)
                if source.is_file():
                    _copy_file_stream(source, destination)
                    return
            if isinstance(result, Path):
                if result.is_file():
                    _copy_file_stream(result, destination)
                    return
            aiter_bytes = getattr(result, "aiter_bytes", None)
            if callable(aiter_bytes):
                chunks = aiter_bytes(1024 * 1024)
                if inspect.isawaitable(chunks):
                    chunks = await chunks
                with destination.open("wb") as handle:
                    async for chunk in chunks:
                        if isinstance(chunk, (bytes, bytearray)):
                            handle.write(chunk)
                return
            iter_bytes = getattr(result, "iter_bytes", None)
            if callable(iter_bytes):
                chunks = iter_bytes(1024 * 1024)
                if inspect.isawaitable(chunks):
                    chunks = await chunks
                with destination.open("wb") as handle:
                    for chunk in chunks:
                        if isinstance(chunk, (bytes, bytearray)):
                            handle.write(chunk)
                return
            if hasattr(result, "__aiter__"):
                with destination.open("wb") as handle:
                    async for chunk in result:
                        if isinstance(chunk, (bytes, bytearray)):
                            handle.write(chunk)
                return
            content = _record_attr(result, "content", None)
            if isinstance(content, (bytes, bytearray)):
                with destination.open("wb") as handle:
                    handle.write(content)
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

    async def _snapshot_collection_stats(
        self, collection_name: str
    ) -> tuple[int | None, int | None, dict[str, Any]]:
        points: int | None = None
        indexed_vectors: int | None = None
        capacity: dict[str, Any] = {}
        count = getattr(self._client, "count", None)
        if count is not None:
            try:
                result = count(
                    collection_name=collection_name,
                    exact=True,
                    **self._timeout_kwargs(),
                )
                if inspect.isawaitable(result):
                    result = await result
                raw_count = result if isinstance(result, int) else _record_attr(result, "count", None)
                if raw_count is not None:
                    points = int(raw_count)
            except Exception:
                points = None
        get_collection = getattr(self._client, "get_collection", None)
        if get_collection is not None:
            try:
                info = get_collection(collection_name)
                if inspect.isawaitable(info):
                    info = await info
                for source in (info, _record_attr(info, "result", None)):
                    raw_indexed = _record_attr(source, "indexed_vectors_count", None)
                    if raw_indexed is not None:
                        indexed_vectors = int(raw_indexed)
                        break
                for field_name in (
                    "disk_usage_bytes",
                    "payload_storage_size",
                    "vector_storage_size",
                    "segments_count",
                    "status",
                ):
                    raw_value = _record_attr(info, field_name, None)
                    if raw_value is not None:
                        capacity[field_name] = _enum_value(raw_value)
            except Exception:
                pass
        return points, indexed_vectors, capacity

    async def validate_restored_snapshot(
        self,
        manifest: SnapshotManifest,
        *,
        restored_collections: Mapping[str, str],
        sample_passage_id: str | None = None,
    ) -> dict[str, Any]:
        """Validate a restored, still-unaliased pair against a snapshot.

        Restore is intentionally explicit and non-activating.  The method
        checks aliases, metadata/control records, counts, and one sample
        passage (when available) before returning a bounded check report.
        """
        if not isinstance(manifest, SnapshotManifest):
            raise TypeError("manifest must be a SnapshotManifest")
        expected_aliases = dict(manifest.aliases)
        actual_aliases = await self._alias_map()
        # The manifest records the application aliases, but the Qdrant
        # service may also contain unrelated aliases owned by another
        # capability.  Compare only the recorded keys; an absent recorded
        # alias remains observable as ``None`` and is not treated as equal to
        # an unrelated current alias.
        expected_relevant = {
            alias: (collection or None)
            for alias, collection in expected_aliases.items()
        }
        actual_relevant = {
            alias: actual_aliases.get(alias) for alias in expected_relevant
        }
        if expected_relevant and actual_relevant != expected_relevant:
            raise QdrantStoreError(
                "snapshot_restore_alias_mismatch",
                "current aliases changed during snapshot restore",
            )
        if set(restored_collections) != set(manifest.physical_collections):
            raise QdrantStoreError(
                "snapshot_restore_collections_missing",
                "restored collection mapping does not cover the snapshot pair",
            )
        checks: dict[str, Any] = {
            "aliases_unchanged": True,
            "collections": {},
            "sample_retrieval": False,
        }
        restored_names = {
            source: str(target) for source, target in restored_collections.items()
        }
        for source_name, restored_name in restored_names.items():
            try:
                info = await self._client.get_collection(restored_name)
            except Exception as exc:
                raise QdrantStoreError(
                    "snapshot_restore_collection_missing",
                    "restored collection cannot be inspected",
                ) from exc
            self._validate_collection_shape(
                restored_name,
                info,
                passages=source_name == manifest.generation.passages_physical,
                expected_dimension=self.dimension,
            )
            metadata = _metadata(info)
            actual_schema = metadata.get("collection_schema_version") or metadata.get(
                "photomat_collection_schema_version"
            )
            actual_fingerprint = metadata.get("model_fingerprint") or metadata.get(
                "photomat_model_fingerprint"
            )
            schema_version = int(actual_schema) if actual_schema is not None else -1
            if schema_version != manifest.schema_version:
                raise QdrantStoreError(
                    "snapshot_restore_schema_mismatch",
                    "restored collection schema metadata does not match the manifest",
                )
            if not self._fingerprint_matches(
                str(actual_fingerprint or ""), manifest.fingerprint
            ):
                raise QdrantStoreError(
                    "snapshot_restore_fingerprint_mismatch",
                    "restored collection fingerprint does not match the manifest",
                )
            points, indexed_vectors, capacity = await self._snapshot_collection_stats(
                restored_name
            )
            expected_points = manifest.point_counts.get(source_name)
            expected_indexed = manifest.indexed_vector_counts.get(source_name)
            if expected_points is not None and points is None:
                raise QdrantStoreError(
                    "snapshot_restore_count_unavailable",
                    "restored collection point count cannot be verified",
                )
            if expected_points is not None and points != expected_points:
                raise QdrantStoreError(
                    "snapshot_restore_count_mismatch",
                    "restored collection point count does not match the manifest",
                )
            if expected_indexed is not None and indexed_vectors is None:
                raise QdrantStoreError(
                    "snapshot_restore_count_unavailable",
                    "restored indexed-vector count cannot be verified",
                )
            if expected_indexed is not None and indexed_vectors != expected_indexed:
                raise QdrantStoreError(
                    "snapshot_restore_count_mismatch",
                    "restored indexed-vector count does not match the manifest",
                )
            checks["collections"][source_name] = {
                "restored_name": restored_name,
                "schema_version": schema_version,
                "fingerprint": str(actual_fingerprint),
                "points": points,
                "indexed_vectors": indexed_vectors,
                "capacity": capacity,
            }

        documents_source = manifest.generation.documents_physical
        passages_source = manifest.generation.passages_physical
        documents_name = restored_names[documents_source]
        passages_name = restored_names[passages_source]
        control_id = str(uuid.uuid5(GENERATION_POINT_NAMESPACE, manifest.fingerprint))
        records = await self._client.retrieve(
            collection_name=documents_name,
            ids=[control_id],
            with_payload=True,
            with_vectors=False,
            **self._timeout_kwargs(),
        )
        if not isinstance(records, Sequence) or not records:
            raise QdrantStoreError(
                "snapshot_restore_control_missing",
                "restored generation control metadata is missing",
            )
        control = _payload(records[0])
        if control.get("record_type") != "generation" or str(
            control.get("model_fingerprint", "")
        ) != manifest.fingerprint:
            raise QdrantStoreError(
                "snapshot_restore_control_mismatch",
                "restored generation control metadata is incompatible",
            )
        if control.get("documents_physical") not in {
            documents_source,
            documents_name,
        } or control.get("passages_physical") not in {passages_source, passages_name}:
            raise QdrantStoreError(
                "snapshot_restore_control_mismatch",
                "restored generation control metadata does not bind the pair",
            )

        sample_id = sample_passage_id
        if sample_id is None:
            scroll = getattr(self._client, "scroll", None)
            if scroll is not None:
                response = scroll(
                    collection_name=passages_name,
                    limit=1,
                    offset=None,
                    with_payload=True,
                    with_vectors=False,
                    **self._timeout_kwargs(),
                )
                if inspect.isawaitable(response):
                    response = await response
                page = response[0] if isinstance(response, tuple) else _record_attr(response, "points", [])
                if page:
                    sample_id = str(_record_attr(page[0], "id", "")) or None
        if sample_id:
            sample = await self._client.retrieve(
                collection_name=passages_name,
                ids=[sample_id],
                with_payload=True,
                with_vectors=False,
                **self._timeout_kwargs(),
            )
            if not isinstance(sample, Sequence) or not sample:
                raise QdrantStoreError(
                    "snapshot_restore_sample_missing",
                    "restored sample passage cannot be retrieved",
                )
            sample_payload = _payload(sample[0])
            if (
                sample_payload.get("record_type") != "passage"
                or str(sample_payload.get("model_fingerprint", ""))
                != manifest.fingerprint
                or str(sample_payload.get("ingest_state", ""))
                != IngestState.READY.value
            ):
                raise QdrantStoreError(
                    "snapshot_restore_sample_mismatch",
                    "restored sample is not a ready passage from the manifest generation",
                )
            checks["sample_retrieval"] = True
        return checks

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
                **self._method_timeout_kwargs(self._client.create_snapshot),
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
                digest, size = _stream_file_digest(temporary_path)
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
        server_version = "unknown"
        info_method = getattr(self._client, "info", None)
        if info_method is not None:
            try:
                info = info_method()
                if inspect.isawaitable(info):
                    info = await info
                server_version = str(_record_attr(info, "version", "unknown") or "unknown")
            except Exception:
                pass
        point_counts: dict[str, int | None] = {}
        indexed_vector_counts: dict[str, int | None] = {}
        capacity: dict[str, Any] = {}
        for collection_name in (
            generation.documents_physical,
            generation.passages_physical,
        ):
            points, indexed_vectors, collection_capacity = await self._snapshot_collection_stats(
                collection_name
            )
            point_counts[collection_name] = points
            indexed_vector_counts[collection_name] = indexed_vectors
            if collection_capacity:
                capacity[collection_name] = collection_capacity
        aliases = await self._alias_map()
        manifest = SnapshotManifest(
            generation=generation,
            created_at=datetime.now(timezone.utc),
            files=tuple(created),
            server_version=server_version,
            schema_version=COLLECTION_SCHEMA_VERSION,
            fingerprint=generation.fingerprint,
            physical_collections=(
                generation.documents_physical,
                generation.passages_physical,
            ),
            point_counts=point_counts,
            indexed_vector_counts=indexed_vector_counts,
            aliases={
                generation.documents_alias: aliases.get(generation.documents_alias, ""),
                generation.passages_alias: aliases.get(generation.passages_alias, ""),
            },
            capacity=capacity,
        )
        manifest_path = output_dir / "snapshot-manifest.json"
        manifest_data = manifest.to_dict()
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
    "INGESTION_RUN_POINT_NAMESPACE",
    "IngestState",
    "LiteratureSourceKind",
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
    "sanitize_qdrant_url",
    "validate_qdrant_url",
    "workspace_id_for",
]
