"""Failure-safe, resumable Qdrant literature ingestion tests."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from photomatagent.scientific.capabilities.literature import ingestion
from photomatagent.scientific.capabilities.literature.ingestion import (
    IngestionPlan,
    IngestionPlanItem,
    IngestionRunState,
    IngestionStats,
    PlanKind,
    RagIngestionError,
    LiteratureIngestionService,
)
from photomatagent.scientific.capabilities.literature.models import (
    DocumentManifest,
    DocumentStatus,
    IngestState,
    PaperRecord,
    PassageRecord,
)
from photomatagent.scientific.capabilities.literature.parser import (
    passage_points_for,
)
from photomatagent.scientific.capabilities.literature.qdrant_store import (
    CollectionGeneration,
    document_id_for,
)
from photomatagent.workspace import Workspace


WORKSPACE = "workspace-test"
FINGERPRINT = "f" * 64
GENERATION = CollectionGeneration(
    fingerprint=FINGERPRINT,
    documents_physical="documents-f",
    passages_physical="passages-f",
    documents_alias="documents-current",
    passages_alias="passages-current",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest(
    relative_path: str,
    content_sha256: str,
    *,
    status: DocumentStatus = DocumentStatus.READY,
) -> DocumentManifest:
    return DocumentManifest(
        schema_version=1,
        record_type="document",
        workspace_id=WORKSPACE,
        document_id=document_id_for(WORKSPACE, relative_path),
        relative_source_path=relative_path,
        file_name=Path(relative_path).name,
        content_sha256=content_sha256,
        status=status,
        model_fingerprint=FINGERPRINT,
    )


class FakeEmbedder:
    identity = SimpleNamespace(dimension=2)

    def __init__(self, vectors: list[list[float]] | None = None) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.vectors is not None:
            return self.vectors
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


class FailingEmbedder(FakeEmbedder):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        raise RuntimeError("embedding failed")


class CancellingEmbedder(FakeEmbedder):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        del texts
        raise asyncio.CancelledError


class FakeStore:
    def __init__(self) -> None:
        self.generation = GENERATION
        self.manifests: dict[str, DocumentManifest] = {}
        self.passage_upserts: list[list[Any]] = []
        self.documents: dict[str, DocumentManifest] = {}
        self.staged: dict[tuple[str, str], int] = {}
        self.ready: set[tuple[str, str]] = set()
        self.delete_calls: list[tuple[str, str | None]] = []
        self.runs: dict[str, Any] = {}
        self.fail_cleanup = False
        self.fail_old_revision_cleanup = False

    async def resolve_current_generation(self) -> CollectionGeneration:
        return self.generation

    async def list_document_manifests(self, workspace_id: str) -> dict[str, DocumentManifest]:
        assert workspace_id
        return dict(self.manifests)

    async def upsert_passages(
        self,
        points: list[Any],
        *,
        batch_size: int,
        workspace_id: str,
    ) -> None:
        assert workspace_id == WORKSPACE
        self.passage_upserts.append(list(points))
        for point in points:
            self.staged[(point.document_id, point.document_revision)] = len(points)

    async def count_revision(
        self,
        document_id: str,
        revision: str,
        state: IngestState,
        *,
        workspace_id: str,
    ) -> int:
        assert workspace_id == WORKSPACE
        if state is IngestState.STAGED:
            return self.staged.get((document_id, revision), 0)
        return sum(
            1
            for doc, rev in self.ready
            if doc == document_id and rev == revision
        )

    async def set_revision_state(
        self,
        document_id: str,
        revision: str,
        state: IngestState,
        *,
        workspace_id: str,
    ) -> None:
        assert workspace_id == WORKSPACE
        if state is IngestState.READY:
            self.ready.add((document_id, revision))

    async def delete_other_revisions(
        self,
        document_id: str,
        keep_revision: str,
        *,
        workspace_id: str,
    ) -> None:
        assert workspace_id == WORKSPACE
        if self.fail_old_revision_cleanup:
            raise RuntimeError("old revision cleanup failed")
        self.ready = {
            item for item in self.ready if item[0] != document_id or item[1] == keep_revision
        }

    async def delete_document_passages(self, document_id: str, *, workspace_id: str) -> None:
        assert workspace_id == WORKSPACE
        self.ready = {item for item in self.ready if item[0] != document_id}

    async def upsert_document(self, manifest: DocumentManifest, *, wait: bool = True) -> None:
        del wait
        self.documents[manifest.document_id] = manifest

    async def delete_staged_revisions(
        self,
        document_id: str,
        *,
        keep_revision: str | None = None,
        workspace_id: str,
    ) -> int:
        assert workspace_id == WORKSPACE
        if self.fail_cleanup:
            raise RuntimeError("cleanup failed")
        removed = 0
        for key in list(self.staged):
            if key[0] == document_id and key[1] != keep_revision:
                removed += self.staged.pop(key)
        self.delete_calls.append((document_id, keep_revision))
        return removed

    async def upsert_ingestion_run(self, run: Any, *, wait: bool = True) -> None:
        del wait
        self.runs[run.run_id] = run

    async def get_ingestion_run(self, run_id: str, workspace_id: str) -> Any:
        run = self.runs.get(run_id)
        return run if run is not None and run.workspace_id == workspace_id else None


def _passages(*texts: str) -> tuple[PaperRecord, list[PassageRecord]]:
    paper = PaperRecord(
        paper_id="paper",
        file_name="paper.pdf",
        title="A paper",
        sha256=FINGERPRINT,
        num_chunks=len(texts),
    )
    return paper, [
        PassageRecord(
            passage_id=f"paper:{index}",
            paper_id="paper",
            file_name="paper.pdf",
            title="A paper",
            text=text,
            chunk_id=f"paper:{index}",
        )
        for index, text in enumerate(texts)
    ]


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    path = tmp_path / "papers" / "paper.pdf"
    path.parent.mkdir()
    path.write_bytes(b"paper")
    return path


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


async def test_plan_classifies_new_changed_unchanged_and_deleted(
    tmp_path: Path, store: FakeStore
) -> None:
    root = tmp_path / "papers"
    root.mkdir()
    (root / "same.pdf").write_bytes(b"same")
    (root / "changed.pdf").write_bytes(b"new")
    store.manifests = {
        _manifest("same.pdf", _sha(b"same")).document_id: _manifest(
            "same.pdf", _sha(b"same")
        ),
        _manifest("changed.pdf", _sha(b"old")).document_id: _manifest(
            "changed.pdf", _sha(b"old")
        ),
        _manifest("deleted.pdf", _sha(b"deleted")).document_id: _manifest(
            "deleted.pdf", _sha(b"deleted")
        ),
    }
    service = LiteratureIngestionService(store, FakeEmbedder())

    plan = await service.plan(root, WORKSPACE)

    assert [item.kind for item in plan.items] == [
        PlanKind.CHANGED,
        PlanKind.DELETED,
        PlanKind.UNCHANGED,
    ]


async def test_missing_root_never_deletes_existing_documents(
    tmp_path: Path, store: FakeStore
) -> None:
    service = LiteratureIngestionService(store, FakeEmbedder())

    with pytest.raises(RagIngestionError) as exc:
        await service.plan(tmp_path / "missing", WORKSPACE)

    assert exc.value.code == "source_root_missing"
    assert store.delete_calls == []


async def test_embedding_failure_writes_no_passages(
    pdf: Path, store: FakeStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    paper, chunks = _passages("one", "two")
    monkeypatch.setattr(ingestion, "parse_pdf", lambda path: (paper, chunks))
    relative = "paper.pdf"
    item = IngestionPlanItem(
        document_id=document_id_for(WORKSPACE, relative),
        relative_source_path=relative,
        content_sha256=_sha(pdf.read_bytes()),
        kind=PlanKind.NEW,
    )
    plan = IngestionPlan(WORKSPACE, GENERATION, (item,), source_root=pdf.parent)
    service = LiteratureIngestionService(store, FailingEmbedder())

    result = await service.index_batch(plan, max_documents=1)

    assert result.failed == 1
    assert store.passage_upserts == []
    assert store.documents[item.document_id].status is DocumentStatus.FAILED


async def test_resume_cleans_staged_revision_and_retries(
    pdf: Path, store: FakeStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    paper, chunks = _passages("one", "two")
    monkeypatch.setattr(ingestion, "parse_pdf", lambda path: (paper, chunks))
    relative = "paper.pdf"
    document_id = document_id_for(WORKSPACE, relative)
    revision = _sha(pdf.read_bytes())
    item = IngestionPlanItem(document_id, relative, revision, PlanKind.NEW)
    plan = IngestionPlan(WORKSPACE, GENERATION, (item,), source_root=pdf.parent)
    store.staged[(document_id, "a" * 64)] = 2

    result = await LiteratureIngestionService(store, FakeEmbedder()).index_batch(
        plan, max_documents=1
    )

    assert result.indexed == 1
    assert store.ready == {(document_id, revision)}
    assert (document_id, "a" * 64) not in store.staged


async def test_all_vectors_are_validated_before_first_passage_upsert(
    pdf: Path, store: FakeStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    paper, chunks = _passages("one", "two")
    monkeypatch.setattr(ingestion, "parse_pdf", lambda path: (paper, chunks))
    item = IngestionPlanItem(
        document_id=document_id_for(WORKSPACE, "paper.pdf"),
        relative_source_path="paper.pdf",
        content_sha256=_sha(pdf.read_bytes()),
        kind=PlanKind.NEW,
    )
    plan = IngestionPlan(WORKSPACE, GENERATION, (item,), source_root=pdf.parent)

    result = await LiteratureIngestionService(
        store, FakeEmbedder(vectors=[[1.0, 2.0], [1.0]])
    ).index_batch(plan, max_documents=1)

    assert result.failed == 1
    assert store.passage_upserts == []


async def test_failed_update_keeps_existing_ready_revision(
    pdf: Path, store: FakeStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    paper, chunks = _passages("one")
    monkeypatch.setattr(ingestion, "parse_pdf", lambda path: (paper, chunks))
    relative = "paper.pdf"
    document_id = document_id_for(WORKSPACE, relative)
    old_revision = "a" * 64
    store.ready.add((document_id, old_revision))
    item = IngestionPlanItem(document_id, relative, _sha(pdf.read_bytes()), PlanKind.CHANGED)
    plan = IngestionPlan(WORKSPACE, GENERATION, (item,), source_root=pdf.parent)
    store.fail_old_revision_cleanup = True

    result = await LiteratureIngestionService(store, FakeEmbedder()).index_batch(
        plan, max_documents=1
    )

    assert result.indexed == 1
    assert result.retryable is True
    assert (document_id, old_revision) in store.ready


async def test_incomplete_plan_never_processes_deletions(
    pdf: Path, store: FakeStore
) -> None:
    del pdf
    item = IngestionPlanItem(
        document_id=document_id_for(WORKSPACE, "deleted.pdf"),
        relative_source_path="deleted.pdf",
        content_sha256="a" * 64,
        kind=PlanKind.DELETED,
    )
    plan = IngestionPlan(WORKSPACE, GENERATION, (item,), complete=False)

    result = await LiteratureIngestionService(store, FakeEmbedder()).index_batch(
        plan, max_documents=1
    )

    assert store.delete_calls == []
    assert result.deleted == 0


async def test_cancellation_is_not_recorded_as_document_failure(
    pdf: Path, store: FakeStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    paper, chunks = _passages("one")
    monkeypatch.setattr(ingestion, "parse_pdf", lambda path: (paper, chunks))
    item = IngestionPlanItem(
        document_id=document_id_for(WORKSPACE, "paper.pdf"),
        relative_source_path="paper.pdf",
        content_sha256=_sha(pdf.read_bytes()),
        kind=PlanKind.NEW,
    )
    plan = IngestionPlan(WORKSPACE, GENERATION, (item,), source_root=pdf.parent)

    with pytest.raises(asyncio.CancelledError):
        await LiteratureIngestionService(store, CancellingEmbedder()).index_batch(
            plan, max_documents=1
        )
    assert store.documents == {}


async def test_workspace_plan_keeps_workspace_relative_source_path(
    tmp_path: Path, store: FakeStore
) -> None:
    workspace = Workspace(tmp_path)
    root = workspace.root / "papers"
    root.mkdir()
    (root / "paper.pdf").write_bytes(b"paper")

    plan = await LiteratureIngestionService(store, FakeEmbedder()).plan(root, workspace)

    assert plan.items[0].relative_source_path == "papers/paper.pdf"
    assert plan.relative_root == "papers"


async def test_batch_and_errors_are_bounded_and_cursor_is_persisted(
    tmp_path: Path, store: FakeStore
) -> None:
    items = tuple(
        IngestionPlanItem(
            document_id=document_id_for(WORKSPACE, f"paper-{index}.pdf"),
            relative_source_path=f"paper-{index}.pdf",
            content_sha256="a" * 64,
            kind=PlanKind.NEW,
        )
        for index in range(25)
    )
    plan = IngestionPlan(WORKSPACE, GENERATION, items, source_root=tmp_path)

    result = await LiteratureIngestionService(store, FakeEmbedder()).index_batch(
        plan, max_documents=100
    )

    assert result.failed == 20
    assert len(result.errors) == 20
    assert result.next_cursor == "paper-19.pdf"
    assert result.complete is False
    assert store.runs[result.run_id].cursor == "paper-19.pdf"


async def test_qdrant_ingestion_control_point_is_bounded_and_workspace_scoped() -> None:
    from test_qdrant_store import FakeAsyncQdrantClient, IDENTITY
    from photomatagent.scientific.capabilities.literature.qdrant_store import (
        QdrantLiteratureStore,
    )

    client = FakeAsyncQdrantClient()
    store = QdrantLiteratureStore(client, prefix="photomat_ingestion_test")
    generation = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    stats = IngestionStats(
        run_id="run-a",
        discovered=3,
        unchanged=1,
        indexed=1,
        failed=1,
        deleted=0,
        chunks=2,
        staged_cleanup=1,
        next_cursor="paper.pdf",
        complete=False,
        errors=tuple(f"error-{index}" for index in range(30)),
    )
    run = IngestionRunState(
        run_id="run-a",
        workspace_id=WORKSPACE,
        generation_fingerprint=generation.fingerprint,
        relative_root="papers",
        cursor="paper.pdf",
        status="retryable",
        stats=stats,
    )

    await store.upsert_ingestion_run(run)
    loaded = await store.get_ingestion_run("run-a", WORKSPACE)

    assert loaded is not None
    assert len(loaded.stats.errors) == 20
    payload = next(
        point.payload
        for point in client.points[generation.documents_alias].values()
        if point.payload.get("record_type") == "ingestion_run"
    )
    assert "text" not in payload
    assert "secret" not in str(payload).lower()
    with pytest.raises(ValueError):
        await store.get_ingestion_run("run-a", "")


def test_passage_points_use_revision_ids_and_uuid_neighbours() -> None:
    paper, chunks = _passages("first", "second")
    points = passage_points_for(
        paper,
        chunks,
        WORKSPACE,
        "papers/paper.pdf",
        FINGERPRINT,
        FINGERPRINT,
    )

    assert points[0].passage_id != chunks[0].passage_id
    assert points[0].next_passage_id == points[1].passage_id
    assert points[1].previous_passage_id == points[0].passage_id
    assert points[0].relative_source_path == "papers/paper.pdf"
