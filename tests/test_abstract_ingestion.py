"""Focused offline tests for streaming SQLite abstract ingestion."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from photomatagent.scientific.capabilities.literature.abstract_ingestion import (
    AbstractIngestionError,
    AbstractIngestionRunState,
    AbstractIngestionStats,
    AbstractIngestionService,
    AbstractSourceRecord,
    SQLiteAbstractReader,
    canonical_abstract_revision,
)
from photomatagent.scientific.capabilities.literature.models import (
    DocumentStatus,
    IngestState,
    LiteratureSourceKind,
)


@pytest.fixture()
def abstract_db(tmp_path: Path) -> Path:
    path = tmp_path / "abstracts.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE papers (
            paper_key TEXT PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            authors TEXT,
            publication_year INTEGER,
            doi TEXT,
            pmid TEXT,
            pmcid TEXT,
            journal TEXT,
            relevance_tier TEXT,
            retrieved_at TEXT
        )
        """
    )
    connection.executemany(
        "INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("key-a", "A", "abstract a", "Alice; Bob", 2022, "", "", "", "J", "core", "one"),
            ("key-b", "B", "abstract b", "Carol", 2023, "", "", "", "J", "related", "two"),
            ("key-c", "C", "abstract c", "Dan", 2024, "", "", "", "J", "broad", "three"),
        ],
    )
    connection.commit()
    connection.close()
    return path


def _record(**changes: object) -> AbstractSourceRecord:
    values: dict[str, object] = {
        "paper_key": "key-a",
        "title": "A title",
        "abstract": "knowledge",
        "authors": ("Author",),
        "publication_year": 2024,
        "doi": "10.1/example",
        "pmid": "1",
        "pmcid": "PMC1",
        "journal": "Journal",
        "relevance_tier": "core_title_match",
        "retrieved_at": "now",
    }
    values.update(changes)
    return AbstractSourceRecord(**values)


def test_reader_uses_keyset_pagination(abstract_db: Path) -> None:
    reader = SQLiteAbstractReader(abstract_db, workspace_root=abstract_db.parent)
    first = reader.fetch_after(None, limit=2)
    second = reader.fetch_after(first[-1].paper_key, limit=2)
    assert [row.paper_key for row in first] == ["key-a", "key-b"]
    assert [row.paper_key for row in second] == ["key-c"]
    reader.close()


def test_reader_preserves_raw_keyset_cursor_with_whitespace_key(abstract_db: Path) -> None:
    connection = sqlite3.connect(abstract_db)
    connection.execute(
        "INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (" a", "Leading", "leading abstract", "Author", 2020, "", "", "", "J", "", ""),
    )
    connection.execute(
        "INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("a", "Exact", "exact abstract", "Author", 2020, "", "", "", "J", "", ""),
    )
    connection.commit()
    connection.close()

    reader = SQLiteAbstractReader(abstract_db, workspace_root=abstract_db.parent)
    first = reader.fetch_after(None, limit=1)
    second = reader.fetch_after(first[-1].paper_key, limit=1)
    assert first[0].paper_key == " a"
    assert second[0].paper_key == "a"
    reader.close()


@pytest.mark.asyncio
async def test_invalid_keys_are_counted_once_and_valid_raw_keys_resume(tmp_path: Path) -> None:
    path = tmp_path / "invalid-keys.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE papers (paper_key TEXT, title TEXT, abstract TEXT)"
    )
    connection.executemany(
        "INSERT INTO papers VALUES (?, ?, ?)",
        [
            (None, "Null", "null abstract"),
            ("", "Empty", "empty abstract"),
            ("   ", "Whitespace", "whitespace abstract"),
            ("\u00a0", "No-break space", "nbsp abstract"),
            ("\u2003", "Em space", "em-space abstract"),
            (" a", "Leading", "leading abstract"),
            ("a", "Exact", "exact abstract"),
        ],
    )
    connection.commit()
    connection.close()

    reader = SQLiteAbstractReader(path, workspace_root=tmp_path)
    assert reader.count() == 7
    assert reader.count_invalid_keys() == 5
    first = reader.fetch_after(None, limit=1)
    second = reader.fetch_after(first[-1].paper_key, limit=1)
    assert [row.paper_key for row in first + second] == [" a", "a"]

    store = FakeAbstractStore()
    embedder = FakeEmbedder()
    service = AbstractIngestionService(
        reader,
        store,
        embedder,
        workspace_id="workspace-a",
        workspace_root=tmp_path,
        relative_source_path="invalid-keys.sqlite3",
        generation=store.generation,
    )
    first_progress = await service.index_batch(run_id="invalid-run", limit=1)
    assert first_progress.total == 7
    assert first_progress.processed == 6
    assert first_progress.skipped_invalid_key == 5
    assert first_progress.indexed == 1
    assert first_progress.cursor == " a"
    assert first_progress.complete is False

    resumed = await service.index_batch(
        run_id="invalid-run", cursor=first_progress.cursor, limit=1
    )
    assert resumed.complete is True
    assert resumed.processed == 7
    assert resumed.skipped_invalid_key == 5
    assert resumed.indexed == 2
    assert resumed.cursor is None
    assert embedder.embedded_document_count == 2
    reader.close()


def test_reader_requires_workspace_root_and_rejects_outside_database(
    abstract_db: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="workspace root"):
        SQLiteAbstractReader(abstract_db)

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    with pytest.raises(ValueError, match="outside workspace"):
        SQLiteAbstractReader(abstract_db, workspace_root=workspace_root)

    reader = SQLiteAbstractReader("abstracts.sqlite3", workspace_root=abstract_db.parent)
    assert reader.path == abstract_db.resolve()
    reader.close()


def test_revision_ignores_retrieval_timestamp() -> None:
    row = _record(abstract="knowledge")
    assert canonical_abstract_revision(row) == canonical_abstract_revision(
        replace(row, retrieved_at="later")
    )
    assert canonical_abstract_revision(row) != canonical_abstract_revision(
        replace(row, abstract="changed")
    )


class FakeAbstractStore:
    def __init__(self) -> None:
        self.generation = SimpleNamespace(fingerprint="f" * 64)
        self.manifests: dict[str, Any] = {}
        self.passages: list[Any] = []
        self.runs: dict[str, Any] = {}

    async def get_document_manifests(self, workspace_id: str, document_ids: list[str], **_: Any) -> dict[str, Any]:
        del workspace_id
        return {document_id: self.manifests[document_id] for document_id in document_ids if document_id in self.manifests}

    async def upsert_document(self, manifest: Any, **_: Any) -> None:
        self.manifests[manifest.document_id] = manifest

    async def upsert_passages(self, points: list[Any], **_: Any) -> None:
        self.passages.extend(points)

    async def count_revision(self, document_id: str, revision: str, state: IngestState, **_: Any) -> int:
        return sum(
            point.document_id == document_id
            and point.document_revision == revision
            and point.ingest_state is state
            for point in self.passages
        )

    async def set_revision_state(self, document_id: str, revision: str, state: IngestState, **_: Any) -> None:
        self.passages = [
            replace(point, ingest_state=state)
            if point.document_id == document_id and point.document_revision == revision
            else point
            for point in self.passages
        ]

    async def get_ingestion_run(self, run_id: str, workspace_id: str, **_: Any) -> Any:
        del workspace_id
        return self.runs.get(run_id)

    async def upsert_ingestion_run(self, run: Any, **_: Any) -> None:
        self.runs[run.run_id] = run


class FakeEmbedder:
    identity = SimpleNamespace(dimension=2)

    def __init__(self, *, fail: bool = False, fail_after: int | None = None) -> None:
        self.fail = fail
        self.fail_after = fail_after
        self.embedded_document_count = 0

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.fail or (
            self.fail_after is not None
            and self.embedded_document_count >= self.fail_after
        ):
            raise RuntimeError("embedding unavailable")
        self.embedded_document_count += len(texts)
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


def _service(abstract_db: Path, *, fail: bool = False) -> tuple[AbstractIngestionService, FakeAbstractStore, FakeEmbedder]:
    store = FakeAbstractStore()
    embedder = FakeEmbedder(fail=fail)
    service = AbstractIngestionService(
        SQLiteAbstractReader(abstract_db, workspace_root=abstract_db.parent),
        store,
        embedder,
        workspace_id="workspace-a",
        workspace_root=abstract_db.parent,
        relative_source_path="abstracts.sqlite3",
        generation=store.generation,
    )
    return service, store, embedder


def test_service_rejects_path_disguised_by_relative_source_path(
    abstract_db: Path, tmp_path: Path
) -> None:
    store = FakeAbstractStore()
    embedder = FakeEmbedder()
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    with pytest.raises(ValueError, match="outside workspace"):
        AbstractIngestionService(
            abstract_db,
            store,
            embedder,
            workspace_id="workspace-a",
            workspace_root=workspace_root,
            relative_source_path="abstracts.sqlite3",
            generation=store.generation,
        )

    reader = SQLiteAbstractReader(abstract_db, workspace_root=abstract_db.parent)
    with pytest.raises(ValueError, match="identify the SQLite database path"):
        AbstractIngestionService(
            reader,
            store,
            embedder,
            workspace_id="workspace-a",
            workspace_root=abstract_db.parent,
            relative_source_path="alias.sqlite3",
            generation=store.generation,
        )
    reader.close()


@pytest.mark.asyncio
async def test_batch_writes_one_document_and_passage_per_row(abstract_db: Path) -> None:
    service, store, _embedder = _service(abstract_db)
    progress = await service.index_batch(run_id="run-a", cursor=None, limit=2)
    assert (progress.indexed, progress.passages) == (2, 2)
    assert len(store.manifests) == 2
    assert all("abstract_only" in point.limitations for point in store.passages)
    assert all(point.source_kind is LiteratureSourceKind.ABSTRACT for point in store.passages)
    assert all(point.section == "Abstract" for point in store.passages)


@pytest.mark.asyncio
async def test_resume_does_not_reembed_committed_rows(abstract_db: Path) -> None:
    service, _store, embedder = _service(abstract_db)
    first = await service.index_batch(run_id="run-a", cursor=None, limit=1)
    second = await service.index_batch(run_id="run-a", cursor=first.cursor, limit=2)
    assert embedder.embedded_document_count == 3
    assert second.complete is True


@pytest.mark.asyncio
async def test_failure_keeps_pre_record_cursor(abstract_db: Path) -> None:
    service, _store, _embedder = _service(abstract_db, fail=True)
    progress = await service.index_batch(run_id="run-a", cursor=None, limit=1)
    assert progress.status == "retryable"
    assert progress.cursor is None


@pytest.mark.asyncio
async def test_partial_failure_persists_cursor_after_skipped_and_committed_rows(
    abstract_db: Path,
) -> None:
    connection = sqlite3.connect(abstract_db)
    connection.execute("UPDATE papers SET abstract = '' WHERE paper_key = 'key-b'")
    connection.commit()
    connection.close()

    store = FakeAbstractStore()
    embedder = FakeEmbedder(fail_after=1)
    service = AbstractIngestionService(
        SQLiteAbstractReader(abstract_db, workspace_root=abstract_db.parent),
        store,
        embedder,
        workspace_id="workspace-a",
        workspace_root=abstract_db.parent,
        relative_source_path="abstracts.sqlite3",
        generation=store.generation,
    )

    failed = await service.index_batch(run_id="partial-run", limit=3)
    assert failed.status == "retryable"
    assert failed.cursor == "key-b"
    assert failed.indexed == 1
    assert failed.skipped_empty == 1
    assert failed.failed == 1
    assert failed.processed == 3

    embedder.fail_after = None
    resumed = await service.index_batch(
        run_id="partial-run", cursor=failed.cursor, limit=1
    )
    assert resumed.complete is True
    assert resumed.indexed == 2
    assert resumed.skipped_empty == 1
    assert resumed.processed == 4
    assert embedder.embedded_document_count == 2


@pytest.mark.asyncio
async def test_empty_abstract_is_skipped_without_a_vector(abstract_db: Path) -> None:
    connection = sqlite3.connect(abstract_db)
    connection.execute("UPDATE papers SET abstract = '' WHERE paper_key = 'key-c'")
    connection.commit()
    connection.close()
    service, store, embedder = _service(abstract_db)
    progress = await service.index_batch(run_id="run-a", cursor=None, limit=3)
    assert progress.indexed == 2
    assert progress.skipped_empty == 1
    assert progress.passages == 2
    assert embedder.embedded_document_count == 2
    assert len(store.passages) == 2


@pytest.mark.asyncio
async def test_resume_rejects_changed_sqlite_source_identity(abstract_db: Path) -> None:
    service, _store, _embedder = _service(abstract_db)
    first = await service.index_batch(run_id="run-a", cursor=None, limit=1)
    with abstract_db.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(AbstractIngestionError, match="source"):
        await service.index_batch(run_id="run-a", cursor=first.cursor, limit=1)


@pytest.mark.asyncio
async def test_existing_abstract_manifest_is_unchanged(abstract_db: Path) -> None:
    service, _store, embedder = _service(abstract_db)
    first = await service.index_batch(run_id="run-a", cursor=None, limit=1)
    second = await service.index_batch(run_id="run-b", cursor=None, limit=1)
    assert first.indexed == 1
    assert second.unchanged == 1
    assert embedder.embedded_document_count == 1


@pytest.mark.asyncio
async def test_qdrant_run_roundtrip_preserves_abstract_source_context() -> None:
    from test_qdrant_store import FakeAsyncQdrantClient, IDENTITY
    from photomatagent.scientific.capabilities.literature.qdrant_store import (
        QdrantLiteratureStore,
    )

    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_abstract_run_test")
    generation = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=2)
    run = AbstractIngestionRunState(
        run_id="abstract-run",
        workspace_id="workspace-a",
        generation_fingerprint=generation.fingerprint,
        source_identity="a" * 64,
        source_path="dataset/abstracts.sqlite3",
        cursor="key-a",
        status="running",
        stats=AbstractIngestionStats(
            run_id="abstract-run",
            discovered=3,
            processed=1,
            indexed=1,
            skipped_invalid_key=2,
            passages=1,
            next_cursor="key-a",
        ),
    )
    await store.upsert_ingestion_run(run)
    loaded = await store.get_ingestion_run(
        "abstract-run", "workspace-a", source_kind=LiteratureSourceKind.ABSTRACT
    )
    assert loaded is not None
    assert loaded.source_identity == "a" * 64
    assert loaded.source_path == "dataset/abstracts.sqlite3"
    assert loaded.stats.passages == 1
    assert loaded.stats.skipped_invalid_key == 2
