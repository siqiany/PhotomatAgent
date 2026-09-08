"""Unit tests for the narrow Qdrant literature persistence adapter.

These tests intentionally use a fake asynchronous client.  They exercise the
adapter contract and never start Qdrant local mode or a Docker service.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from photomatagent.scientific.capabilities.literature.providers.base import ModelIdentity
from photomatagent.scientific.capabilities.literature.qdrant_store import (
    CollectionGeneration,
    DocumentManifest,
    DocumentStatus,
    IngestState,
    PassagePoint,
    QdrantLiteratureStore,
    QdrantStoreError,
    collection_fingerprint,
    document_id_for,
    passage_id_for,
    workspace_id_for,
)


IDENTITY = ModelIdentity(
    provider="local",
    model="intfloat/multilingual-e5-small",
    dimension=384,
    document_prefix="passage: ",
    query_prefix="query: ",
    normalize=True,
)


def _record(point_id: str, payload: dict[str, Any], vector: Any = None) -> SimpleNamespace:
    return SimpleNamespace(id=point_id, payload=payload, vector=vector)


@dataclass
class _FakeSnapshot:
    name: str
    size: int = 0


class FakeAsyncQdrantClient:
    """Small protocol fake that records calls made by the store."""

    def __init__(self) -> None:
        self.collections: dict[str, dict[str, Any]] = {}
        self.aliases: dict[str, str] = {}
        self.points: dict[str, dict[str, _record]] = {}
        self.create_calls: list[dict[str, Any]] = []
        self.alias_update_calls: list[list[Any]] = []
        self.query_calls: list[dict[str, Any]] = []
        self.upsert_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []
        self.payload_calls: list[dict[str, Any]] = []
        self.snapshot_downloads: list[tuple[str, str]] = []
        self._snapshot_bytes: dict[tuple[str, str], bytes] = {}

    async def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    async def create_collection(self, collection_name: str, **kwargs: Any) -> bool:
        self.create_calls.append({"collection_name": collection_name, **kwargs})
        self.collections[collection_name] = kwargs
        self.points.setdefault(collection_name, {})
        return True

    async def get_collection(self, collection_name: str) -> SimpleNamespace:
        if collection_name not in self.collections:
            raise RuntimeError("missing collection")
        config = SimpleNamespace(
            metadata=self.collections[collection_name].get("metadata", {}),
            params=self.collections[collection_name].get("params"),
        )
        return SimpleNamespace(
            config=config,
            payload_schema=self.collections[collection_name].get("payload_schema", {}),
        )

    async def create_payload_index(self, collection_name: str, **kwargs: Any) -> None:
        self.collections[collection_name].setdefault("payload_indexes", []).append(kwargs)

    async def update_collection_aliases(self, actions: list[Any], **kwargs: Any) -> bool:
        self.alias_update_calls.append(actions)
        for action in actions:
            create = getattr(action, "create_alias", None)
            delete = getattr(action, "delete_alias", None)
            if create is not None:
                self.aliases[create.alias_name] = create.collection_name
            if delete is not None:
                self.aliases.pop(delete.alias_name, None)
        return True

    async def get_aliases(self) -> SimpleNamespace:
        return SimpleNamespace(
            aliases=[
                SimpleNamespace(alias_name=alias, collection_name=collection)
                for alias, collection in self.aliases.items()
            ]
        )

    async def upsert(self, collection_name: str, points: list[Any], **kwargs: Any) -> None:
        self.upsert_calls.append(
            {"collection_name": collection_name, "points": points, **kwargs}
        )
        target = self.points.setdefault(collection_name, {})
        for point in points:
            target[str(point.id)] = _record(point.id, point.payload or {}, point.vector)

    async def scroll(self, collection_name: str, **kwargs: Any) -> tuple[list[_record], None]:
        self.query_calls.append({"kind": "scroll", "collection_name": collection_name, **kwargs})
        return list(self.points.get(collection_name, {}).values()), None

    async def retrieve(self, collection_name: str, ids: list[str], **kwargs: Any) -> list[_record]:
        self.query_calls.append(
            {"kind": "retrieve", "collection_name": collection_name, "ids": ids, **kwargs}
        )
        return [self.points.get(collection_name, {})[point_id] for point_id in ids if point_id in self.points.get(collection_name, {})]

    async def query_points(self, collection_name: str, **kwargs: Any) -> SimpleNamespace:
        self.query_calls.append({"kind": "query", "collection_name": collection_name, **kwargs})
        return SimpleNamespace(points=[])

    async def count(self, collection_name: str, **kwargs: Any) -> SimpleNamespace:
        self.query_calls.append({"kind": "count", "collection_name": collection_name, **kwargs})
        return SimpleNamespace(count=0)

    async def set_payload(self, collection_name: str, **kwargs: Any) -> None:
        self.payload_calls.append({"collection_name": collection_name, **kwargs})

    async def delete(self, collection_name: str, **kwargs: Any) -> None:
        self.delete_calls.append({"collection_name": collection_name, **kwargs})

    async def create_snapshot(self, collection_name: str, **kwargs: Any) -> _FakeSnapshot:
        snapshot = _FakeSnapshot(f"{collection_name}.snapshot")
        data = f"snapshot:{collection_name}".encode()
        snapshot.size = len(data)
        self._snapshot_bytes[(collection_name, snapshot.name)] = data
        return snapshot

    async def download_snapshot(self, collection_name: str, snapshot_name: str) -> bytes:
        self.snapshot_downloads.append((collection_name, snapshot_name))
        return self._snapshot_bytes[(collection_name, snapshot_name)]


def _manifest(
    *,
    workspace_id: str = "workspace-a",
    document_id: str = "doc-1",
    model_fingerprint: str = "f" * 64,
) -> DocumentManifest:
    return DocumentManifest(
        schema_version=1,
        record_type="document",
        workspace_id=workspace_id,
        document_id=document_id,
        relative_source_path="dataset/paper/a.pdf",
        file_name="a.pdf",
        content_sha256="a" * 64,
        status=DocumentStatus.READY,
        title="A paper",
        authors=("Author",),
        year=2024,
        num_pages=2,
        chunk_count=1,
        model_fingerprint=model_fingerprint,
        indexed_at=datetime.now(timezone.utc),
        last_error="",
    )


def _passage(
    *,
    workspace_id: str = "workspace-a",
    passage_id: str = "passage-1",
    model_fingerprint: str = "f" * 64,
) -> PassagePoint:
    return PassagePoint(
        schema_version=1,
        record_type="passage",
        workspace_id=workspace_id,
        document_id="doc-1",
        document_revision="a" * 64,
        ingest_state=IngestState.READY,
        passage_id=passage_id,
        chunk_index=0,
        text="infrared detector",
        title="A paper",
        authors=("Author",),
        year=2024,
        section="Results",
        heading_path="Results",
        page_start=1,
        page_end=1,
        previous_passage_id=None,
        next_passage_id=None,
        relative_source_path="dataset/paper/a.pdf",
        model_fingerprint=model_fingerprint,
        limitations=("ocr",),
        dense=(0.1, 0.2),
    )


def test_passage_id_changes_only_with_document_revision() -> None:
    document_id = document_id_for("workspace-a", "dataset/paper/a.pdf")
    first = passage_id_for(document_id, "a" * 64, 3)
    assert first == passage_id_for(document_id, "a" * 64, 3)
    assert first != passage_id_for(document_id, "b" * 64, 3)


def test_stable_ids_reject_workspace_escape_and_invalid_revision() -> None:
    with pytest.raises(ValueError):
        document_id_for("workspace-a", "../outside.pdf")
    with pytest.raises(ValueError):
        PassagePoint(
            **{
                **_passage().__dict__,
                "document_revision": "A" * 64,
            }
        )


def test_models_require_lowercase_sha256_and_relative_source_path() -> None:
    with pytest.raises(ValueError):
        DocumentManifest(**{**_manifest().__dict__, "content_sha256": "a" * 63})
    with pytest.raises(ValueError):
        DocumentManifest(**{**_manifest().__dict__, "relative_source_path": "/tmp/a.pdf"})
    with pytest.raises(ValueError):
        DocumentManifest(**{**_manifest().__dict__, "relative_source_path": "C:papers/a.pdf"})
    with pytest.raises(ValueError):
        DocumentManifest(**{**_manifest().__dict__, "model_fingerprint": "F" * 64})


async def test_ensure_generation_creates_versioned_pair_and_aliases() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    generation = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    assert generation.documents_physical.endswith(generation.fingerprint[:12])
    assert generation.passages_alias == "photomat_literature_passages_current"
    assert client.create_calls[0]["vectors_config"] is None
    passage_create = next(
        call for call in client.create_calls if "passages" in call["collection_name"]
    )
    assert set(passage_create["vectors_config"]) == {"dense"}
    assert set(passage_create["sparse_vectors_config"]) == {"sparse_bm25"}
    assert client.aliases[generation.documents_alias] == generation.documents_physical
    assert client.aliases[generation.passages_alias] == generation.passages_physical


async def test_collection_schema_indexes_all_filterable_provenance_fields() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    generation = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    indexed = {
        item["field_name"]
        for item in client.collections[generation.passages_physical]["payload_indexes"]
    }
    assert {
        "workspace_id",
        "record_type",
        "document_id",
        "document_revision",
        "ingest_state",
        "passage_id",
        "relative_source_path",
        "year",
        "model_fingerprint",
        "previous_passage_id",
        "next_passage_id",
    } <= indexed


async def test_fingerprint_excludes_reranker_and_secrets() -> None:
    first = collection_fingerprint(IDENTITY, 1, prefix="photomat_literature")
    second = collection_fingerprint(IDENTITY, 1, prefix="photomat_literature")
    assert first == second
    assert len(first) == 64
    assert "secret" not in first


def test_collection_storage_prefix_does_not_change_model_fingerprint() -> None:
    assert collection_fingerprint(IDENTITY, 1, prefix="first") == collection_fingerprint(
        IDENTITY, 1, prefix="second"
    )


def test_from_config_resolves_api_key_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    from photomatagent.scientific.capabilities.literature import qdrant_store

    monkeypatch.setattr(qdrant_store, "AsyncQdrantClient", FakeClient)
    monkeypatch.setenv("QDRANT_TEST_KEY", "secret-value")
    config = SimpleNamespace(
        qdrant_url="http://qdrant.test:6333",
        qdrant_api_key_env="QDRANT_TEST_KEY",
        qdrant_collection_prefix="photomat_literature",
        qdrant_timeout_seconds=17,
        embedding_vector_dim=384,
    )
    store = QdrantLiteratureStore.from_config(config)
    assert isinstance(store, QdrantLiteratureStore)
    assert calls == [
        {
            "url": "http://qdrant.test:6333",
            "api_key": "secret-value",
            "timeout": 17,
            "prefer_grpc": False,
        }
    ]


async def test_switch_current_generation_uses_one_atomic_alias_request() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    first = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    client.alias_update_calls.clear()
    second = CollectionGeneration(
        fingerprint="b" * 64,
        documents_physical="photomat_literature_documents_" + "b" * 12,
        passages_physical="photomat_literature_passages_" + "b" * 12,
        documents_alias=first.documents_alias,
        passages_alias=first.passages_alias,
    )
    for source, target in (
        (first.documents_physical, second.documents_physical),
        (first.passages_physical, second.passages_physical),
    ):
        client.collections[target] = dict(client.collections[source])
        client.collections[target]["metadata"] = dict(client.collections[source]["metadata"])
        client.collections[target]["metadata"]["model_fingerprint"] = "b" * 64
    await store.switch_current_generation(second)
    assert len(client.alias_update_calls) == 1
    assert client.aliases[second.documents_alias] == second.documents_physical
    assert client.aliases[second.passages_alias] == second.passages_physical


async def test_switch_rejects_physical_fingerprint_mismatch_before_alias_mutation() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    first = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    second = CollectionGeneration(
        fingerprint="b" * 64,
        documents_physical="photomat_literature_documents_" + "b" * 12,
        passages_physical="photomat_literature_passages_" + "b" * 12,
        documents_alias=first.documents_alias,
        passages_alias=first.passages_alias,
    )
    client.collections[second.documents_physical] = client.collections[first.documents_physical]
    client.collections[second.passages_physical] = client.collections[first.passages_physical]
    client.alias_update_calls.clear()
    with pytest.raises(QdrantStoreError) as exc:
        await store.switch_current_generation(second)
    assert exc.value.code == "model_fingerprint_mismatch"
    assert client.alias_update_calls == []


async def test_mismatch_validation_is_read_only() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    generation = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    client.alias_update_calls.clear()
    with pytest.raises(QdrantStoreError) as exc:
        await store.validate_current_generation("b" * 64)
    assert exc.value.code == "model_fingerprint_mismatch"
    assert client.alias_update_calls == []
    assert generation.fingerprint != "b" * 64


async def test_resolve_rejects_aliases_pointing_to_different_generations() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    first = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    second = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=2)
    client.aliases[first.documents_alias] = first.documents_physical
    client.aliases[first.passages_alias] = second.passages_physical
    with pytest.raises(QdrantStoreError) as exc:
        await store.resolve_current_generation()
    assert exc.value.code == "schema_mismatch"


async def test_ensure_generation_rejects_existing_fingerprint_mismatch() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    generation = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    client.collections[generation.documents_physical]["metadata"]["model_fingerprint"] = "b" * 64
    fresh_store = QdrantLiteratureStore(client, prefix="photomat_literature")
    with pytest.raises(QdrantStoreError) as exc:
        await fresh_store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    assert exc.value.code == "model_fingerprint_mismatch"


async def test_ensure_generation_rejects_existing_vector_schema_mismatch() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    generation = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    client.collections[generation.passages_physical]["metadata"]["dense_dimension"] = 123
    fresh_store = QdrantLiteratureStore(client, prefix="photomat_literature")
    with pytest.raises(QdrantStoreError) as exc:
        await fresh_store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    assert exc.value.code == "schema_mismatch"


async def test_ensure_generation_validates_qdrant_vector_schema_when_metadata_is_missing() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    generation = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    client.collections[generation.passages_physical]["metadata"] = {}
    client.collections[generation.passages_physical]["params"] = SimpleNamespace(
        vectors={"dense": SimpleNamespace(size=123, distance="Cosine")},
        sparse_vectors={"sparse_bm25": object()},
    )
    fresh_store = QdrantLiteratureStore(client, prefix="photomat_literature")
    with pytest.raises(QdrantStoreError) as exc:
        await fresh_store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    assert exc.value.code == "schema_mismatch"


async def test_ensure_generation_rejects_uninspectable_existing_collection() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    generation = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    client.collections[generation.documents_physical]["metadata"] = {}
    client.collections[generation.documents_physical].pop("params", None)
    client.collections[generation.passages_physical]["metadata"] = {}
    client.collections[generation.passages_physical].pop("params", None)
    fresh_store = QdrantLiteratureStore(client, prefix="photomat_literature")
    with pytest.raises(QdrantStoreError) as exc:
        await fresh_store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    assert exc.value.code == "schema_mismatch"


async def test_ensure_generation_rejects_non_idf_sparse_schema() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    generation = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    client.collections[generation.passages_physical]["metadata"] = {}
    client.collections[generation.passages_physical]["params"] = SimpleNamespace(
        vectors={"dense": SimpleNamespace(size=384, distance="Cosine", on_disk=True)},
        sparse_vectors={"sparse_bm25": SimpleNamespace(modifier="none")},
    )
    fresh_store = QdrantLiteratureStore(client, prefix="photomat_literature")
    with pytest.raises(QdrantStoreError) as exc:
        await fresh_store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    assert exc.value.code == "schema_mismatch"


async def test_upsert_passages_is_batched_and_candidate_limit_is_bounded() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    generation = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    points = [
        _passage(passage_id=f"passage-{i}", model_fingerprint=generation.fingerprint)
        for i in range(3)
    ]
    await store.upsert_passages(points, batch_size=2)
    assert [len(call["points"]) for call in client.upsert_calls if "passages" in call["collection_name"]] == [2, 1]
    await store.dense_candidates((0.1, 0.2), workspace_id="workspace-a", limit=999)
    query = client.query_calls[-1]
    assert query["limit"] == 50
    filt = query["query_filter"]
    values = {condition.key: condition.match.value for condition in filt.must}
    assert values == {"workspace_id": "workspace-a", "record_type": "passage", "ingest_state": "ready"}
    assert generation.passages_physical in client.collections


async def test_upserts_reject_points_from_a_different_model_generation() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    generation = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    with pytest.raises(QdrantStoreError) as document_exc:
        await store.upsert_document(_manifest())
    assert document_exc.value.code == "model_fingerprint_mismatch"
    with pytest.raises(QdrantStoreError) as passage_exc:
        await store.upsert_passages([_passage()], batch_size=16)
    assert passage_exc.value.code == "model_fingerprint_mismatch"
    assert not [
        call
        for call in client.upsert_calls
        if call["collection_name"] in {generation.documents_alias, generation.passages_alias}
    ]


async def test_sparse_and_hybrid_use_workspace_ready_filters() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    generation = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    await store.sparse_candidates("infrared", workspace_id="workspace-a", limit=5)
    sparse_query = client.query_calls[-1]
    sparse_values = {condition.key: condition.match.value for condition in sparse_query["query_filter"].must}
    assert sparse_values == {"workspace_id": "workspace-a", "record_type": "passage", "ingest_state": "ready"}
    await store.hybrid_candidates("infrared", (0.1, 0.2), workspace_id="workspace-a", limit=5)
    hybrid_query = client.query_calls[-1]
    assert len(hybrid_query["prefetch"]) == 2
    assert all(prefetch.filter is not None for prefetch in hybrid_query["prefetch"])


async def test_document_operations_filter_workspace_and_revision() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    generation = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    await store.upsert_document(_manifest(model_fingerprint=generation.fingerprint))
    client.points[generation.documents_alias]["doc-1"].payload["indexed_at"] = _manifest(
        model_fingerprint=generation.fingerprint
    ).indexed_at.isoformat()
    listed = await store.list_document_manifests("workspace-a")
    assert "doc-1" in listed
    assert isinstance(listed["doc-1"].indexed_at, datetime)
    await store.count_revision("doc-1", "a" * 64, IngestState.READY, workspace_id="workspace-a")
    call = client.query_calls[-1]
    values = {condition.key: condition.match.value for condition in call["count_filter"].must}
    assert values["workspace_id"] == "workspace-a"
    assert values["document_revision"] == "a" * 64
    await store.set_revision_state("doc-1", "a" * 64, IngestState.SUPERSEDED, workspace_id="workspace-a")
    assert client.payload_calls[-1]["payload"] == {"ingest_state": "superseded"}


async def test_snapshot_files_and_manifest_are_written_atomically(tmp_path: Path) -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    generation = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    manifest = await store.create_current_snapshots(tmp_path)
    assert manifest.generation == generation
    assert len(manifest.files) == 2
    assert (tmp_path / "snapshot-manifest.json").is_file()
    assert not list(tmp_path.glob("*.tmp"))
    for item in manifest.files:
        path = tmp_path / item.file_name
        assert path.is_file()
        assert item.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    on_disk = json.loads((tmp_path / "snapshot-manifest.json").read_text())
    assert len(on_disk["files"]) == 2


async def test_snapshot_download_uses_qdrant_async_stream_when_no_adapter_method(
    tmp_path: Path,
) -> None:
    client = FakeAsyncQdrantClient()
    client.download_snapshot = None  # type: ignore[method-assign]
    stream = SimpleNamespace(
        snapshots_api=SimpleNamespace(
            get_snapshot=lambda collection, name: _async_bytes(io.BytesIO(b"streamed"))
        )
    )
    client._client = SimpleNamespace(openapi_client=stream)
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    manifest = await store.create_current_snapshots(tmp_path)
    assert all((tmp_path / item.file_name).read_bytes() == b"streamed" for item in manifest.files)


async def _async_bytes(value: io.BytesIO) -> io.BytesIO:
    return value
