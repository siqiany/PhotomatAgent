"""Opt-in contract tests against the pinned Qdrant Docker service.

The default test run never starts Qdrant, downloads models, indexes PDFs, or
contacts an embedding/reranking provider.  Set
``PHOTOMATAGENT_RUN_QDRANT_INTEGRATION=1`` only when the Compose service is
running and a disposable integration collection is acceptable.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest

from photomatagent.scientific.capabilities.literature.models import (
    DocumentManifest,
    DocumentStatus,
    IngestState,
    PassagePoint,
)
from photomatagent.scientific.capabilities.literature.providers.base import ModelIdentity
from photomatagent.scientific.capabilities.literature.qdrant_store import (
    QdrantLiteratureStore,
    document_id_for,
    passage_id_for,
)


INTEGRATION_ENV = "PHOTOMATAGENT_RUN_QDRANT_INTEGRATION"
QDRANT_URL_ENV = "PHOTOMATAGENT_QDRANT_TEST_URL"
TEST_WORKSPACE_ID = "photomat-qdrant-fixture-workspace"
IDENTITY = ModelIdentity(
    provider="fixture",
    model="synthetic-8d-v1",
    dimension=8,
    document_prefix="",
    query_prefix="",
    normalize=False,
)


def _safe_test_prefix(value: str) -> str:
    if not value.startswith("photomat_test_"):
        raise RuntimeError(
            "Qdrant integration refuses a collection prefix without the "
            "photomat_test_ safety marker"
        )
    return value


def test_integration_prefix_must_be_explicitly_safe() -> None:
    with pytest.raises(RuntimeError, match="photomat_test_"):
        _safe_test_prefix("photomat_literature")


def _vector(axis: int, magnitude: float = 1.0) -> tuple[float, ...]:
    values = [0.0] * 8
    values[axis % 8] = magnitude
    return tuple(values)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _manifest(
    generation: Any,
    *,
    relative_path: str,
    text: str,
    status: DocumentStatus = DocumentStatus.READY,
) -> DocumentManifest:
    document_id = document_id_for(TEST_WORKSPACE_ID, relative_path)
    return DocumentManifest(
        schema_version=1,
        record_type="document",
        workspace_id=TEST_WORKSPACE_ID,
        document_id=document_id,
        relative_source_path=relative_path,
        file_name=Path(relative_path).name,
        content_sha256=_sha(text),
        status=status,
        title=f"Synthetic {Path(relative_path).stem}",
        authors=("PhotomatAgent fixture authors",),
        year=2026,
        num_pages=1,
        chunk_count=1,
        model_fingerprint=generation.fingerprint,
        indexed_at=datetime.now(timezone.utc),
    )


def _point(
    generation: Any,
    *,
    relative_path: str,
    text: str,
    axis: int,
    ingest_state: IngestState = IngestState.READY,
    workspace_id: str = TEST_WORKSPACE_ID,
    chunk_index: int = 0,
) -> PassagePoint:
    document_id = document_id_for(workspace_id, relative_path)
    revision = _sha(text + relative_path)
    passage_id = passage_id_for(document_id, revision, chunk_index)
    return PassagePoint(
        schema_version=1,
        record_type="passage",
        workspace_id=workspace_id,
        document_id=document_id,
        document_revision=revision,
        ingest_state=ingest_state,
        passage_id=passage_id,
        chunk_index=chunk_index,
        text=text,
        title=f"Synthetic {Path(relative_path).stem}",
        authors=("PhotomatAgent fixture authors",),
        year=2026,
        section="Results",
        heading_path="Results / Synthetic fixture",
        page_start=1,
        page_end=1,
        previous_passage_id=None,
        next_passage_id=None,
        relative_source_path=relative_path,
        model_fingerprint=generation.fingerprint,
        limitations=("Synthetic authored text; not copied from a paper.",),
        dense=_vector(axis),
        normalized_text_sha256=_sha(" ".join(text.split()).casefold()),
        indexed_at=datetime.now(timezone.utc),
    )


@pytest.fixture
async def real_store() -> Any:
    """Yield one disposable store; skip unless the explicit Docker gate is set."""
    if os.environ.get(INTEGRATION_ENV) != "1":
        pytest.skip(f"set {INTEGRATION_ENV}=1 to run the Docker Qdrant contract")
    url = os.environ.get(QDRANT_URL_ENV, "http://127.0.0.1:6333").strip()
    prefix = _safe_test_prefix(f"photomat_test_{uuid.uuid4().hex}")
    try:
        from qdrant_client import AsyncQdrantClient
    except ImportError as exc:  # pragma: no cover - optional extra
        pytest.skip(f"qdrant-client is unavailable: {exc}")
    client = AsyncQdrantClient(url=url, timeout=20, prefer_grpc=False)
    try:
        await client.get_collections()
    except Exception as exc:  # pragma: no cover - depends on local Docker
        await client.close()
        pytest.skip(
            f"Qdrant Docker gate unavailable at {url}: "
            f"{type(exc).__name__}: {str(exc)[:240]}"
        )
    store = QdrantLiteratureStore(
        client,
        prefix=prefix,
        dimension=8,
        timeout_seconds=20,
        snapshot_base_url=url,
    )
    try:
        await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
        yield store
    finally:
        # Only remove collections created under this UUID-bearing safety
        # prefix.  Never inspect or delete the user's current aliases.
        try:
            collections = await client.get_collections()
            for descriptor in getattr(collections, "collections", ()):
                name = str(getattr(descriptor, "name", ""))
                if name.startswith(prefix + "_"):
                    await client.delete_collection(name)
        finally:
            await client.close()


@pytest.mark.asyncio
async def test_qdrant_generation_roundtrip(real_store: Any) -> None:
    generation = await real_store.ensure_generation(
        identity=IDENTITY,
        chunk_schema_version=1,
    )
    resolved = await real_store.resolve_current_generation()

    assert resolved == generation
    aliases = await real_store._client.get_aliases()
    alias_map = {
        item.alias_name: item.collection_name for item in aliases.aliases
    }
    assert alias_map[generation.documents_alias] == generation.documents_physical
    assert alias_map[generation.passages_alias] == generation.passages_physical
    assert await real_store._client.collection_exists(generation.documents_physical)
    assert await real_store._client.collection_exists(generation.passages_physical)


@pytest.mark.asyncio
async def test_vectorless_document_point(real_store: Any) -> None:
    generation = await real_store.resolve_current_generation()
    assert generation is not None
    manifest = _manifest(
        generation,
        relative_path="synthetic/vectorless.pdf",
        text="A vectorless document control point.",
    )
    await real_store.upsert_document(manifest)

    records = await real_store._client.retrieve(
        collection_name=generation.documents_alias,
        ids=[manifest.document_id],
        with_payload=True,
        with_vectors=True,
    )
    assert len(records) == 1
    assert records[0].payload["record_type"] == "document"
    assert records[0].vector in (None, {})
    manifests = await real_store.list_document_manifests(TEST_WORKSPACE_ID)
    assert manifests[manifest.document_id].status is DocumentStatus.READY


@pytest.mark.asyncio
async def test_dense_sparse_rrf_and_filters(real_store: Any) -> None:
    generation = await real_store.resolve_current_generation()
    assert generation is not None
    performance = _point(
        generation,
        relative_path="synthetic/performance.pdf",
        text="HgTe infrared detector responsivity is 0.42 A/W at 80 K and 3.5 micrometres.",
        axis=0,
    )
    unrelated = _point(
        generation,
        relative_path="synthetic/unrelated.pdf",
        text="A control sample reports dark current under a different condition.",
        axis=1,
    )
    staged = _point(
        generation,
        relative_path="synthetic/staged.pdf",
        text="HgTe responsivity staged before validation.",
        axis=0,
        ingest_state=IngestState.STAGED,
    )
    other_workspace = _point(
        generation,
        relative_path="synthetic/other-workspace.pdf",
        text="HgTe responsivity from another workspace.",
        axis=0,
        workspace_id="another-workspace",
    )
    await real_store.upsert_passages(
        [performance, unrelated, staged],
        batch_size=2,
        workspace_id=TEST_WORKSPACE_ID,
    )
    # Write the isolated point through its own workspace-scoped operation.
    await real_store.upsert_passages(
        [other_workspace], batch_size=1, workspace_id="another-workspace"
    )

    candidates = await real_store.hybrid_candidates(
        "responsivity at 80 K",
        _vector(0),
        workspace_id=TEST_WORKSPACE_ID,
        limit=10,
    )
    ids = [candidate.passage_id for candidate in candidates]
    assert performance.passage_id in ids
    assert staged.passage_id not in ids
    assert other_workspace.passage_id not in ids
    assert all(
        candidate.payload.get("workspace_id") == TEST_WORKSPACE_ID
        and candidate.payload.get("ingest_state") == IngestState.READY.value
        for candidate in candidates
    )


@pytest.mark.asyncio
async def test_staged_points_are_not_visible(real_store: Any) -> None:
    generation = await real_store.resolve_current_generation()
    assert generation is not None
    point = _point(
        generation,
        relative_path="synthetic/not-ready.pdf",
        text="A staged passage must remain invisible until validation.",
        axis=2,
        ingest_state=IngestState.STAGED,
    )
    await real_store.upsert_passages(
        [point], batch_size=1, workspace_id=TEST_WORKSPACE_ID
    )
    before = await real_store.dense_candidates(
        _vector(2), workspace_id=TEST_WORKSPACE_ID, limit=5
    )
    assert point.passage_id not in {candidate.passage_id for candidate in before}

    await real_store.set_revision_state(
        point.document_id,
        point.document_revision,
        IngestState.READY,
        workspace_id=TEST_WORKSPACE_ID,
    )
    after = await real_store.dense_candidates(
        _vector(2), workspace_id=TEST_WORKSPACE_ID, limit=5
    )
    assert point.passage_id in {candidate.passage_id for candidate in after}


@pytest.mark.asyncio
async def test_alias_pair_switches_to_validated_generation(real_store: Any) -> None:
    first = await real_store.resolve_current_generation()
    assert first is not None
    marker = _point(
        first,
        relative_path="synthetic/first-generation.pdf",
        text="First generation content remains in its physical collection.",
        axis=3,
    )
    await real_store.upsert_passages(
        [marker], batch_size=1, workspace_id=TEST_WORKSPACE_ID
    )

    second_identity = replace(IDENTITY, model="synthetic-8d-v2")
    second = await real_store.ensure_generation(
        identity=second_identity,
        chunk_schema_version=1,
    )
    resolved = await real_store.resolve_current_generation()
    assert resolved == second
    assert second.fingerprint != first.fingerprint
    assert await real_store._client.collection_exists(first.passages_physical)
    aliases = await real_store._client.get_aliases()
    alias_map = {
        item.alias_name: item.collection_name for item in aliases.aliases
    }
    assert alias_map[second.documents_alias] == second.documents_physical
    assert alias_map[second.passages_alias] == second.passages_physical
    assert alias_map[second.passages_alias] != first.passages_physical
    assert await real_store.dense_candidates(
        _vector(3), workspace_id=TEST_WORKSPACE_ID, limit=5
    ) == []


@pytest.mark.asyncio
async def test_snapshot_download_and_restore_noncurrent(
    real_store: Any, tmp_path: Path
) -> None:
    generation = await real_store.resolve_current_generation()
    assert generation is not None
    manifest = _manifest(
        generation,
        relative_path="synthetic/snapshot.pdf",
        text="Snapshot restore fixture content.",
    )
    await real_store.upsert_document(manifest)
    point = _point(
        generation,
        relative_path="synthetic/snapshot.pdf",
        text="Snapshot restore fixture passage.",
        axis=4,
    )
    await real_store.upsert_passages(
        [point], batch_size=1, workspace_id=TEST_WORKSPACE_ID
    )
    snapshot_manifest = await real_store.create_current_snapshots(tmp_path)
    assert len(snapshot_manifest.files) == 2
    for snapshot_file in snapshot_manifest.files:
        path = tmp_path / snapshot_file.file_name
        assert path.is_file()
        assert path.stat().st_size == snapshot_file.size_bytes
        assert hashlib.sha256(path.read_bytes()).hexdigest() == snapshot_file.sha256

    # Qdrant restores through a server-visible snapshot URL.  The URL points
    # back to the service's own snapshot endpoint, so the downloaded local
    # artifact is independently checked above while restore remains a real
    # server operation.  Restored collections deliberately stay unaliased.
    url = os.environ.get(QDRANT_URL_ENV, "http://127.0.0.1:6333").rstrip("/")
    for snapshot_file in snapshot_manifest.files:
        snapshots = await real_store._client.list_snapshots(snapshot_file.collection)
        suffix = snapshot_file.file_name.split("--", 1)[-1]
        snapshot = next(
            item for item in snapshots if str(item.name) == suffix
        )
        restored_name = (
            f"{real_store.prefix}_restored_"
            f"{('documents' if snapshot_file.collection == generation.documents_physical else 'passages')}"
        )
        location = (
            f"{url}/collections/{quote(snapshot_file.collection, safe='')}/snapshots/"
            f"{quote(str(snapshot.name), safe='')}"
        )
        await real_store._client.recover_snapshot(
            collection_name=restored_name,
            location=location,
            wait=True,
        )
        assert await real_store._client.collection_exists(restored_name)

    aliases = await real_store._client.get_aliases()
    alias_map = {
        item.alias_name: item.collection_name for item in aliases.aliases
    }
    assert alias_map[generation.documents_alias] == generation.documents_physical
    assert alias_map[generation.passages_alias] == generation.passages_physical


__all__ = ["real_store"]
