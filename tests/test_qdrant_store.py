"""Unit tests for the narrow Qdrant literature persistence adapter.

These tests intentionally use a fake asynchronous client.  They exercise the
adapter contract and never start Qdrant local mode or a Docker service.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, replace
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
    GENERATION_POINT_NAMESPACE,
    IngestState,
    PassagePoint,
    QdrantLiteratureStore,
    QdrantStoreError,
    SnapshotManifest,
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

    def _physical(self, collection_name: str) -> str:
        return self.aliases.get(collection_name, collection_name)

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
        stored = self.collections[collection_name]
        params = stored.get("params")
        if params is None and any(
            key in stored
            for key in (
                "vectors_config",
                "sparse_vectors_config",
                "shard_number",
                "replication_factor",
                "on_disk_payload",
            )
        ):
            params = SimpleNamespace(
                vectors=stored.get("vectors_config"),
                sparse_vectors=stored.get("sparse_vectors_config"),
                shard_number=stored.get("shard_number"),
                replication_factor=stored.get("replication_factor"),
                on_disk_payload=stored.get("on_disk_payload"),
            )
        config = SimpleNamespace(
            metadata=stored.get("metadata", {}),
            params=params,
            on_disk_payload=stored.get("on_disk_payload"),
            strict_mode_config=stored.get("strict_mode_config"),
        )
        return SimpleNamespace(
            config=config,
            payload_schema=stored.get("payload_schema", {}),
            on_disk_payload=stored.get("on_disk_payload"),
            strict_mode_config=stored.get("strict_mode_config"),
        )

    async def create_payload_index(self, collection_name: str, **kwargs: Any) -> None:
        self.collections[collection_name].setdefault("payload_indexes", []).append(kwargs)
        self.collections[collection_name].setdefault("payload_schema", {})[
            kwargs["field_name"]
        ] = SimpleNamespace(data_type=kwargs["field_schema"])

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
        target = self.points.setdefault(self._physical(collection_name), {})
        for point in points:
            target[str(point.id)] = _record(point.id, point.payload or {}, point.vector)

    async def scroll(self, collection_name: str, **kwargs: Any) -> tuple[list[_record], None]:
        self.query_calls.append({"kind": "scroll", "collection_name": collection_name, **kwargs})
        return list(self.points.get(self._physical(collection_name), {}).values()), None

    async def retrieve(self, collection_name: str, ids: list[str], **kwargs: Any) -> list[_record]:
        self.query_calls.append(
            {"kind": "retrieve", "collection_name": collection_name, "ids": ids, **kwargs}
        )
        physical = self._physical(collection_name)
        return [self.points.get(physical, {})[point_id] for point_id in ids if point_id in self.points.get(physical, {})]

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


async def _ensure_active(
    store: QdrantLiteratureStore,
    *,
    identity: ModelIdentity = IDENTITY,
    chunk_schema_version: int = 1,
) -> CollectionGeneration:
    generation = await store.ensure_generation(
        identity=identity, chunk_schema_version=chunk_schema_version
    )
    await store.activate_generation(generation, allow_empty_bootstrap=True)
    return generation


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
    # Building a generation is a staging operation.  Current aliases must not
    # move until the explicit post-build activation gate is called.
    assert client.aliases == {}
    await store.activate_generation(generation, allow_empty_bootstrap=True)
    assert client.create_calls[0]["vectors_config"] is None
    passage_create = next(
        call for call in client.create_calls if "passages" in call["collection_name"]
    )
    assert set(passage_create["vectors_config"]) == {"dense"}
    assert set(passage_create["sparse_vectors_config"]) == {"sparse_bm25"}
    assert client.aliases[generation.documents_alias] == generation.documents_physical
    assert client.aliases[generation.passages_alias] == generation.passages_physical


async def test_empty_generation_requires_explicit_bootstrap_before_activation() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_test_empty_bootstrap")
    generation = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)

    with pytest.raises(QdrantStoreError) as exc:
        await store.activate_generation(generation)

    assert exc.value.code == "generation_incomplete"
    assert client.aliases == {}
    await store.activate_generation(generation, allow_empty_bootstrap=True)
    assert client.aliases[generation.documents_alias] == generation.documents_physical


async def test_empty_bootstrap_cannot_replace_existing_current_generation() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_test_bootstrap_replace")
    current = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    await store.upsert_document(_manifest(model_fingerprint=current.fingerprint))
    await store.upsert_passages(
        [_passage(model_fingerprint=current.fingerprint)],
        batch_size=1,
        workspace_id="workspace-a",
    )
    await store.activate_generation(current)

    replacement_identity = replace(IDENTITY, model="local/replacement-model")
    replacement = await store.ensure_generation(
        identity=replacement_identity,
        chunk_schema_version=1,
    )

    with pytest.raises(QdrantStoreError) as exc:
        await store.activate_generation(replacement, allow_empty_bootstrap=True)

    assert exc.value.code == "generation_incomplete"
    assert client.aliases[current.documents_alias] == current.documents_physical
    assert client.aliases[current.passages_alias] == current.passages_physical


async def test_public_switch_cannot_bypass_empty_generation_guard() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_test_empty_switch")
    generation = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)

    with pytest.raises(QdrantStoreError) as exc:
        await store.switch_current_generation(generation)

    assert exc.value.code == "generation_incomplete"
    assert client.aliases == {}


async def test_expected_generation_is_read_only_and_bootstraps_without_aliases() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_test_bootstrap")

    expected = store.expected_generation(identity=IDENTITY, chunk_schema_version=1)
    assert expected.documents_physical.endswith(expected.fingerprint[:12])
    assert client.create_calls == []
    assert client.aliases == {}
    assert await store.resolve_current_generation() is None

    staged = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    assert staged == expected
    assert client.aliases == {}
    await store.activate_generation(staged, allow_empty_bootstrap=True)
    assert await store.resolve_current_generation() == staged


async def test_activation_rejects_incomplete_generation_before_alias_mutation() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_test_activation")
    generation = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    client.points[generation.documents_physical].clear()

    with pytest.raises(QdrantStoreError) as exc:
        await store.activate_generation(generation)

    assert exc.value.code == "generation_incomplete"
    assert client.aliases == {}


async def test_activation_rejects_retryable_ingestion_run_before_alias_mutation() -> None:
    from photomatagent.scientific.capabilities.literature.ingestion import (
        IngestionRunState,
        IngestionStats,
    )

    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_test_activation_run")
    generation = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    run = IngestionRunState(
        run_id="retryable-run",
        workspace_id="workspace-a",
        generation_fingerprint=generation.fingerprint,
        relative_root="dataset/paper",
        cursor=None,
        status="retryable",
        stats=IngestionStats(
            run_id="retryable-run",
            discovered=1,
            unchanged=0,
            indexed=0,
            failed=1,
            deleted=0,
            chunks=0,
            staged_cleanup=0,
            next_cursor=None,
            complete=False,
            errors=("failure",),
            retryable=True,
        ),
    )
    await store.upsert_ingestion_run(run)

    with pytest.raises(QdrantStoreError) as exc:
        await store.activate_generation(generation)

    assert exc.value.code == "generation_incomplete"
    assert client.aliases == {}


async def test_staging_writes_use_physical_collections_before_activation() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_test_staging")
    generation = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    await store.upsert_document(_manifest(model_fingerprint=generation.fingerprint))
    await store.upsert_passages(
        [_passage(model_fingerprint=generation.fingerprint)],
        batch_size=1,
        workspace_id="workspace-a",
    )

    assert client.aliases == {}
    assert all(
        call["collection_name"]
        in {generation.documents_physical, generation.passages_physical}
        for call in client.upsert_calls
    )
    assert generation.documents_alias not in client.points
    assert generation.passages_alias not in client.points


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
        qdrant_url="https://qdrant.test:6333",
        qdrant_api_key_env="QDRANT_TEST_KEY",
        qdrant_collection_prefix="photomat_literature",
        qdrant_timeout_seconds=17,
        embedding_vector_dim=384,
    )
    store = QdrantLiteratureStore.from_config(config)
    assert isinstance(store, QdrantLiteratureStore)
    assert calls == [
        {
            "url": "https://qdrant.test:6333",
            "api_key": "secret-value",
            "timeout": 17,
            "prefer_grpc": False,
        }
    ]


@pytest.mark.parametrize(
    ("url", "key", "code"),
    [
        ("http://qdrant.example:6333", "", "qdrant_tls_required"),
        ("https://qdrant.example:6333", "", "qdrant_api_key_required"),
        ("https://user:secret@qdrant.example:6333", "key", "qdrant_url_invalid"),
        ("https://qdrant.example:6333?token=secret", "key", "qdrant_url_invalid"),
    ],
)
def test_from_config_enforces_remote_qdrant_security(
    monkeypatch: pytest.MonkeyPatch, url: str, key: str, code: str
) -> None:
    from photomatagent.scientific.capabilities.literature import qdrant_store

    monkeypatch.delenv("QDRANT_REMOTE_TEST_KEY", raising=False)
    if key:
        monkeypatch.setenv("QDRANT_REMOTE_TEST_KEY", key)
    monkeypatch.setattr(qdrant_store, "AsyncQdrantClient", lambda **kwargs: kwargs)
    config = SimpleNamespace(
        qdrant_url=url,
        qdrant_api_key_env="QDRANT_REMOTE_TEST_KEY",
        qdrant_collection_prefix="photomat_test_remote",
        qdrant_timeout_seconds=17,
        embedding_vector_dim=384,
    )

    with pytest.raises(QdrantStoreError) as exc:
        QdrantLiteratureStore.from_config(config)

    assert exc.value.code == code


def test_remote_qdrant_security_allows_tls_with_configured_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from photomatagent.scientific.capabilities.literature import qdrant_store

    calls: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    monkeypatch.setenv("QDRANT_REMOTE_TEST_KEY", "secret-value")
    monkeypatch.setattr(qdrant_store, "AsyncQdrantClient", FakeClient)
    config = SimpleNamespace(
        qdrant_url="https://qdrant.example:6333",
        qdrant_api_key_env="QDRANT_REMOTE_TEST_KEY",
        qdrant_collection_prefix="photomat_test_remote",
        qdrant_timeout_seconds=17,
        embedding_vector_dim=384,
    )

    store = QdrantLiteratureStore.from_config(config)

    assert isinstance(store, QdrantLiteratureStore)
    assert calls[0]["url"] == "https://qdrant.example:6333"
    assert calls[0]["api_key"] == "secret-value"


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
    client.points[second.documents_physical] = {
        str(uuid.uuid5(GENERATION_POINT_NAMESPACE, second.fingerprint)): _record(
            "control",
            {
                "record_type": "generation",
                "model_fingerprint": second.fingerprint,
                "chunk_schema_version": 1,
                "sparse_model": "qdrant/bm25",
                "dense_dimension": 384,
                "documents_physical": second.documents_physical,
                "passages_physical": second.passages_physical,
            },
        )
    }
    await store.switch_current_generation(second, allow_empty_bootstrap=True)
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
        await store.switch_current_generation(second, allow_empty_bootstrap=True)
    assert exc.value.code == "model_fingerprint_mismatch"
    assert client.alias_update_calls == []


async def test_mismatch_validation_is_read_only() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    generation = await _ensure_active(store)
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


async def test_resolve_rejects_same_suffix_aliases_with_mixed_generation_metadata() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    generation = await _ensure_active(store)
    client.collections[generation.passages_physical]["metadata"]["model_fingerprint"] = "b" * 64
    fresh_store = QdrantLiteratureStore(client, prefix="photomat_literature")
    client.alias_update_calls.clear()
    with pytest.raises(QdrantStoreError) as exc:
        await fresh_store.validate_current_generation(generation.fingerprint)
    assert exc.value.code in {"model_fingerprint_mismatch", "schema_mismatch"}
    assert client.alias_update_calls == []


async def test_resolve_rejects_generation_control_fingerprint_mismatch() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    generation = await _ensure_active(store)
    control = next(
        record
        for record in client.points[generation.documents_physical].values()
        if record.payload.get("record_type") == "generation"
    )
    control.payload["model_fingerprint"] = "b" * 64
    fresh_store = QdrantLiteratureStore(client, prefix="photomat_literature")
    client.alias_update_calls.clear()
    with pytest.raises(QdrantStoreError) as exc:
        await fresh_store.validate_current_generation(generation.fingerprint)
    assert exc.value.code == "model_fingerprint_mismatch"
    assert client.alias_update_calls == []


async def test_resolve_fails_closed_when_generation_control_is_missing() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    generation = await _ensure_active(store)
    client.points[generation.documents_physical].clear()
    fresh_store = QdrantLiteratureStore(client, prefix="photomat_literature")
    with pytest.raises(QdrantStoreError) as exc:
        await fresh_store.validate_current_generation(generation.fingerprint)
    assert exc.value.code == "control_point_missing"


async def test_resolve_does_not_hide_generation_control_read_failure() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    generation = await _ensure_active(store)

    async def fail_retrieve(**kwargs: Any) -> list[Any]:
        raise RuntimeError("network unavailable")

    client.retrieve = fail_retrieve  # type: ignore[method-assign]
    fresh_store = QdrantLiteratureStore(client, prefix="photomat_literature")
    with pytest.raises(QdrantStoreError) as exc:
        await fresh_store.validate_current_generation(generation.fingerprint)
    assert exc.value.code == "control_point_unavailable"


async def test_generation_control_must_bind_both_physical_collection_names() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    generation = await _ensure_active(store)
    control = next(
        record
        for record in client.points[generation.documents_physical].values()
        if record.payload.get("record_type") == "generation"
    )
    control.payload["documents_physical"] = generation.documents_physical
    control.payload["passages_physical"] = "photomat_literature_passages_wrong"
    fresh_store = QdrantLiteratureStore(client, prefix="photomat_literature")
    with pytest.raises(QdrantStoreError) as exc:
        await fresh_store.validate_current_generation(generation.fingerprint)
    assert exc.value.code == "schema_mismatch"


async def test_switch_current_generation_uses_paired_control_validation() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    generation = await _ensure_active(store)
    control = next(
        record
        for record in client.points[generation.documents_physical].values()
        if record.payload.get("record_type") == "generation"
    )
    control.payload["documents_physical"] = generation.documents_physical
    control.payload["passages_physical"] = "photomat_literature_passages_wrong"
    client.alias_update_calls.clear()
    with pytest.raises(QdrantStoreError) as exc:
        await store.switch_current_generation(generation, allow_empty_bootstrap=True)
    assert exc.value.code == "schema_mismatch"
    assert client.alias_update_calls == []


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


async def test_ensure_generation_rejects_missing_dense_vector_size() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    generation = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    client.collections[generation.passages_physical]["params"] = SimpleNamespace(
        vectors={"dense": SimpleNamespace(distance="Cosine", on_disk=True)},
        sparse_vectors={"sparse_bm25": SimpleNamespace(modifier="idf")},
        shard_number=1,
        replication_factor=1,
        on_disk_payload=True,
    )
    fresh_store = QdrantLiteratureStore(client, prefix="photomat_literature")
    with pytest.raises(QdrantStoreError) as exc:
        await fresh_store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    assert exc.value.code == "schema_mismatch"


async def test_ensure_generation_rejects_missing_dense_vector_distance() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    generation = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    client.collections[generation.passages_physical]["params"] = SimpleNamespace(
        vectors={"dense": SimpleNamespace(size=384, on_disk=True)},
        sparse_vectors={"sparse_bm25": SimpleNamespace(modifier="idf")},
        shard_number=1,
        replication_factor=1,
        on_disk_payload=True,
    )
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
    for field in (
        "params",
        "vectors_config",
        "sparse_vectors_config",
        "shard_number",
        "replication_factor",
        "on_disk_payload",
        "strict_mode_config",
        "payload_schema",
    ):
        client.collections[generation.documents_physical].pop(field, None)
    client.collections[generation.passages_physical]["metadata"] = {}
    for field in (
        "params",
        "vectors_config",
        "sparse_vectors_config",
        "shard_number",
        "replication_factor",
        "on_disk_payload",
        "strict_mode_config",
        "payload_schema",
    ):
        client.collections[generation.passages_physical].pop(field, None)
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


async def test_ensure_generation_rejects_on_disk_and_strict_mode_mismatch() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    generation = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    client.collections[generation.passages_physical]["on_disk_payload"] = False
    client.collections[generation.passages_physical]["strict_mode_config"] = SimpleNamespace(
        enabled=False,
        unindexed_filtering_retrieve=True,
        unindexed_filtering_update=True,
    )
    fresh_store = QdrantLiteratureStore(client, prefix="photomat_literature")
    with pytest.raises(QdrantStoreError) as exc:
        await fresh_store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    assert exc.value.code == "schema_mismatch"


@pytest.mark.parametrize("missing_field", ["on_disk_payload", "strict_mode_config", "payload_schema"])
async def test_existing_collection_missing_contract_schema_fails_closed(missing_field: str) -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    generation = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    client.collections[generation.passages_physical].pop(missing_field, None)
    fresh_store = QdrantLiteratureStore(client, prefix="photomat_literature")
    with pytest.raises(QdrantStoreError) as exc:
        await fresh_store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    assert exc.value.code == "schema_mismatch"


async def test_existing_passage_collection_missing_dense_dimension_fails_closed() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    generation = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    client.collections[generation.passages_physical]["metadata"].pop("dense_dimension")
    client.collections[generation.passages_physical]["metadata"].pop("photomat_dense_dimension")
    fresh_store = QdrantLiteratureStore(client, prefix="photomat_literature")
    with pytest.raises(QdrantStoreError) as exc:
        await fresh_store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    assert exc.value.code == "schema_mismatch"


async def test_ensure_generation_rejects_existing_payload_index_type_mismatch() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    generation = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    client.collections[generation.passages_physical]["payload_schema"] = {
        "workspace_id": SimpleNamespace(data_type="integer")
    }
    fresh_store = QdrantLiteratureStore(client, prefix="photomat_literature")
    with pytest.raises(QdrantStoreError) as exc:
        await fresh_store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    assert exc.value.code == "schema_mismatch"


async def test_ensure_generation_does_not_swallow_untyped_existing_index_error() -> None:
    client = FakeAsyncQdrantClient()

    async def incompatible_index_error(**kwargs: Any) -> None:
        raise RuntimeError("already exists but has an incompatible type")

    client.create_payload_index = incompatible_index_error  # type: ignore[method-assign]
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    with pytest.raises(QdrantStoreError) as exc:
        await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    assert exc.value.code == "schema_mismatch"


async def test_ensure_generation_does_not_accept_unverified_typed_409_index_error() -> None:
    client = FakeAsyncQdrantClient()

    class ConflictError(RuntimeError):
        status_code = 409

    async def conflict(**kwargs: Any) -> None:
        raise ConflictError("conflict")

    client.create_payload_index = conflict  # type: ignore[method-assign]
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    with pytest.raises(QdrantStoreError) as exc:
        await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    assert exc.value.code == "schema_mismatch"


async def test_ensure_generation_rejects_typed_409_index_with_wrong_actual_type() -> None:
    client = FakeAsyncQdrantClient()

    class ConflictError(RuntimeError):
        status_code = 409

    async def conflict(**kwargs: Any) -> None:
        client.collections[kwargs["collection_name"]].setdefault("payload_schema", {})[
            kwargs["field_name"]
        ] = SimpleNamespace(data_type="integer")
        raise ConflictError("conflict")

    client.create_payload_index = conflict  # type: ignore[method-assign]
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    with pytest.raises(QdrantStoreError) as exc:
        await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    assert exc.value.code == "schema_mismatch"


async def test_upsert_passages_is_batched_and_candidate_limit_is_bounded() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    generation = await _ensure_active(store)
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
    generation = await _ensure_active(store)
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
    generation = await _ensure_active(store)
    await store.upsert_document(_manifest(model_fingerprint=generation.fingerprint))
    client.points[generation.documents_physical]["doc-1"].payload["indexed_at"] = _manifest(
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


async def test_retrieve_passages_uses_bounded_server_side_workspace_ready_filter() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    await _ensure_active(store)
    await store.retrieve_passages("workspace-a", [f"passage-{i}" for i in range(100)])
    call = client.query_calls[-1]
    assert call["kind"] == "scroll"
    assert call["limit"] == 50
    filt = call["scroll_filter"]
    field_conditions = [condition for condition in filt.must if hasattr(condition, "key")]
    values = {condition.key: condition.match.value for condition in field_conditions}
    assert values == {
        "workspace_id": "workspace-a",
        "record_type": "passage",
        "ingest_state": "ready",
    }
    id_conditions = [condition for condition in filt.must if hasattr(condition, "has_id")]
    assert len(id_conditions) == 1
    assert id_conditions[0].has_id == [f"passage-{i}" for i in range(50)]


async def test_retrieve_passages_parses_indexed_at_from_payload() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    generation = await _ensure_active(store)
    indexed_at = datetime(2025, 2, 3, 4, 5, 6, tzinfo=timezone.utc)
    base_point = _passage(model_fingerprint=generation.fingerprint)
    point = PassagePoint(**{**base_point.__dict__, "indexed_at": indexed_at})
    await store.upsert_passages([point], batch_size=1)
    client.points[generation.passages_physical][point.passage_id].payload[
        "indexed_at"
    ] = indexed_at.isoformat()

    retrieved = await store.retrieve_passages("workspace-a", [point.passage_id])

    assert len(retrieved) == 1
    assert retrieved[0].indexed_at == indexed_at


async def test_revision_mutations_require_workspace_scope() -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    with pytest.raises(TypeError):
        await store.count_revision("doc-1", "a" * 64, IngestState.READY)
    with pytest.raises(TypeError):
        await store.set_revision_state("doc-1", "a" * 64, IngestState.READY)
    with pytest.raises(TypeError):
        await store.delete_other_revisions("doc-1", "a" * 64)
    with pytest.raises(TypeError):
        await store.delete_document_passages("doc-1")
    with pytest.raises(ValueError):
        await store.count_revision(
            "doc-1", "a" * 64, IngestState.READY, workspace_id=None  # type: ignore[arg-type]
        )


async def test_snapshot_files_and_manifest_are_written_atomically(tmp_path: Path) -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_literature")
    generation = await _ensure_active(store)
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
    assert on_disk["server_version"] == "unknown"
    assert on_disk["schema_version"] == 1
    assert on_disk["fingerprint"] == generation.fingerprint
    assert on_disk["physical_collections"] == [
        generation.documents_physical,
        generation.passages_physical,
    ]
    assert set(on_disk["point_counts"]) == {
        generation.documents_physical,
        generation.passages_physical,
    }


async def test_snapshot_restore_validation_checks_pair_and_sample(tmp_path: Path) -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_test_restore")
    generation = await _ensure_active(store)
    point = _passage(model_fingerprint=generation.fingerprint)
    await store.upsert_passages([point], batch_size=1, workspace_id="workspace-a")
    manifest = await store.create_current_snapshots(tmp_path)
    # A restore must not fail merely because an unrelated managed alias exists
    # on the server.  It should compare only the aliases recorded in this
    # manifest, while still detecting changes to those aliases.
    client.aliases["unrelated_managed_alias"] = "unrelated_collection"

    checks = await store.validate_restored_snapshot(
        manifest,
        restored_collections={
            generation.documents_physical: generation.documents_physical,
            generation.passages_physical: generation.passages_physical,
        },
        sample_passage_id=point.passage_id,
    )

    assert checks["aliases_unchanged"] is True
    assert checks["sample_retrieval"] is True


class _FakeAdapterStream:
    async def aiter_bytes(self, chunk_size: int = 65536):
        del chunk_size
        yield b"adapter-"
        yield b"stream"


async def test_snapshot_download_adapter_stream_is_written_incrementally(
    tmp_path: Path,
) -> None:
    client = FakeAsyncQdrantClient()

    async def stream_snapshot(collection_name: str, snapshot_name: str) -> _FakeAdapterStream:
        del collection_name, snapshot_name
        return _FakeAdapterStream()

    client.download_snapshot = stream_snapshot  # type: ignore[method-assign]
    store = QdrantLiteratureStore(client, prefix="photomat_test_snapshot_stream")
    destination = tmp_path / "stream.snapshot"

    await store._download_snapshot("collection", "snapshot", destination)

    assert destination.read_bytes() == b"adapter-stream"


async def test_snapshot_download_adapter_path_is_copied_without_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeAsyncQdrantClient()
    source = tmp_path / "source.snapshot"
    source.write_bytes(b"path-stream")

    async def path_snapshot(collection_name: str, snapshot_name: str) -> Path:
        del collection_name, snapshot_name
        return source

    client.download_snapshot = path_snapshot  # type: ignore[method-assign]
    store = QdrantLiteratureStore(client, prefix="photomat_test_snapshot_path")
    destination = tmp_path / "destination.snapshot"

    def fail_read_bytes(self: Path) -> bytes:
        raise AssertionError("snapshot adapter must copy paths in bounded chunks")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)
    await store._download_snapshot("collection", "snapshot", destination)

    with destination.open("rb") as handle:
        assert handle.read() == b"path-stream"


async def test_snapshot_restore_validation_rejects_incompatible_collection_shape(
    tmp_path: Path,
) -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_test_restore_shape")
    generation = await _ensure_active(store)
    point = _passage(model_fingerprint=generation.fingerprint)
    await store.upsert_passages([point], batch_size=1, workspace_id="workspace-a")
    manifest = await store.create_current_snapshots(tmp_path)
    client.collections[generation.passages_physical]["strict_mode_config"] = None

    with pytest.raises(QdrantStoreError) as exc:
        await store.validate_restored_snapshot(
            manifest,
            restored_collections={
                generation.documents_physical: generation.documents_physical,
                generation.passages_physical: generation.passages_physical,
            },
            sample_passage_id=point.passage_id,
        )

    assert exc.value.code == "schema_mismatch"


async def test_snapshot_restore_validation_requires_a_ready_matching_sample(
    tmp_path: Path,
) -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_test_restore_sample")
    generation = await _ensure_active(store)
    point = _passage(model_fingerprint=generation.fingerprint)
    await store.upsert_passages([point], batch_size=1, workspace_id="workspace-a")
    manifest = await store.create_current_snapshots(tmp_path)
    client.points[generation.passages_physical][point.passage_id].payload["record_type"] = (
        "document"
    )

    with pytest.raises(QdrantStoreError) as exc:
        await store.validate_restored_snapshot(
            manifest,
            restored_collections={
                generation.documents_physical: generation.documents_physical,
                generation.passages_physical: generation.passages_physical,
            },
            sample_passage_id=point.passage_id,
        )

    assert exc.value.code == "snapshot_restore_sample_mismatch"


async def test_snapshot_restore_fails_when_manifest_counts_cannot_be_verified(
    tmp_path: Path,
) -> None:
    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_test_restore_counts")
    generation = await _ensure_active(store)
    point = _passage(model_fingerprint=generation.fingerprint)
    await store.upsert_passages([point], batch_size=1, workspace_id="workspace-a")
    manifest = await store.create_current_snapshots(tmp_path)

    async def unavailable_count(*args: Any, **kwargs: Any) -> SimpleNamespace:
        del args, kwargs
        return SimpleNamespace(count=None)

    client.count = unavailable_count  # type: ignore[method-assign]
    restored_manifest = replace(
        manifest,
        point_counts={
            generation.documents_physical: 1,
            generation.passages_physical: 1,
        },
    )

    with pytest.raises(QdrantStoreError) as exc:
        await store.validate_restored_snapshot(
            restored_manifest,
            restored_collections={
                generation.documents_physical: generation.documents_physical,
                generation.passages_physical: generation.passages_physical,
            },
            sample_passage_id=point.passage_id,
        )

    assert exc.value.code == "snapshot_restore_count_unavailable"


@dataclass
class _FakeSnapshotResponse:
    status_code: int
    chunks: tuple[bytes, ...]

    async def aiter_bytes(self, chunk_size: int = 65536):
        del chunk_size
        for chunk in self.chunks:
            yield chunk


class _FakeSnapshotTransport:
    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code
        self.calls: list[dict[str, Any]] = []

    async def get(self, url: str, *, headers: dict[str, str], timeout: int) -> _FakeSnapshotResponse:
        self.calls.append({"url": url, "headers": headers, "timeout": timeout})
        return _FakeSnapshotResponse(self.status_code, (b"stream", b"ed"))


async def test_snapshot_download_uses_documented_rest_stream_when_no_adapter_method(
    tmp_path: Path,
) -> None:
    client = FakeAsyncQdrantClient()
    client.download_snapshot = None  # type: ignore[method-assign]
    transport = _FakeSnapshotTransport()
    store = QdrantLiteratureStore(
        client,
        prefix="photomat_literature",
        timeout_seconds=17,
        snapshot_base_url="http://qdrant.test:6333/",
        snapshot_api_key="secret-value",
        snapshot_transport=transport,
    )
    await _ensure_active(store)
    manifest = await store.create_current_snapshots(tmp_path)
    assert all((tmp_path / item.file_name).read_bytes() == b"streamed" for item in manifest.files)
    assert len(transport.calls) == 2
    for call, item in zip(transport.calls, manifest.files):
        assert call["url"] == (
            "http://qdrant.test:6333/collections/"
            f"{item.collection}/snapshots/{item.collection}.snapshot"
        )
        assert call["headers"] == {"api-key": "secret-value"}
        assert call["timeout"] == 17


async def test_snapshot_download_rejects_non_success_rest_status(tmp_path: Path) -> None:
    client = FakeAsyncQdrantClient()
    client.download_snapshot = None  # type: ignore[method-assign]
    transport = _FakeSnapshotTransport(status_code=503)
    store = QdrantLiteratureStore(
        client,
        prefix="photomat_literature",
        snapshot_base_url="http://qdrant.test:6333",
        snapshot_transport=transport,
    )
    await _ensure_active(store)
    with pytest.raises(QdrantStoreError) as exc:
        await store.create_current_snapshots(tmp_path)
    assert exc.value.code == "snapshot_download_failed"


async def test_ingestion_run_store_boundary_redacts_paths_secrets_and_bounds_errors() -> None:
    from photomatagent.scientific.capabilities.literature.ingestion import (
        IngestionRunState,
        IngestionStats,
    )

    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_test_diag")
    generation = await _ensure_active(store)
    raw_errors = tuple(
        f"api_key=secret-{index} at /absolute/private/file-{index}.pdf " + "x" * 1000
        for index in range(40)
    )
    run = IngestionRunState(
        run_id="diagnostic-run",
        workspace_id="workspace-a",
        generation_fingerprint=generation.fingerprint,
        relative_root="dataset/paper",
        cursor=None,
        status="retryable",
        stats=IngestionStats(
            run_id="diagnostic-run",
            discovered=1,
            unchanged=0,
            indexed=0,
            failed=1,
            deleted=0,
            chunks=0,
            staged_cleanup=0,
            next_cursor=None,
            complete=False,
            errors=raw_errors,
            retryable=True,
        ),
    )

    await store.upsert_ingestion_run(run)

    payload = next(
        point.payload
        for point in client.points[generation.documents_physical].values()
        if point.payload.get("record_type") == "ingestion_run"
    )
    assert len(payload["errors"]) <= 20
    assert all(len(str(error)) <= 512 for error in payload["errors"])
    rendered = str(payload)
    assert "secret-" not in rendered
    assert "/absolute/private" not in rendered
    assert "[redacted]" in rendered
    assert "[path]" in rendered


@pytest.mark.parametrize(
    "diagnostic, secret",
    [
        ("Authorization: Bearer auth-secret", "auth-secret"),
        ("Bearer standalone-secret", "standalone-secret"),
        ("token=token-secret", "token-secret"),
    ],
)
def test_store_diagnostic_redacts_bearer_and_token_forms(
    diagnostic: str, secret: str
) -> None:
    from photomatagent.scientific.capabilities.literature.qdrant_store import (
        _sanitize_diagnostic,
    )

    rendered = _sanitize_diagnostic(diagnostic)

    assert secret not in rendered
    assert "[redacted]" in rendered
