"""LanceDB-backed local index with incremental updates.

Layout (inside ``literature_index_dir``):

- ``passages`` table: one row per chunk, with a 384-dim float vector.
- ``documents`` table: one row per PDF with its content hash.

Incremental indexing skips PDFs whose sha256 is unchanged, replaces chunks of
changed PDFs, and removes passages of deleted PDFs.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

from photomatagent.scientific.capabilities.literature.models import (
    PaperRecord,
    PassageRecord,
)
from photomatagent.scientific.capabilities.literature.parser import (
    paper_id_for,
    parse_pdf,
)

logger = logging.getLogger(__name__)

PASSAGES_TABLE = "passages"
DOCUMENTS_TABLE = "documents"
DEFAULT_VECTOR_DIM = 384  # intfloat/multilingual-e5-small

_PASSAGE_SCHEMA = pa.schema(
    [
        pa.field("passage_id", pa.string()),
        pa.field("paper_id", pa.string()),
        pa.field("file_name", pa.string()),
        pa.field("title", pa.string()),
        pa.field("authors_json", pa.string()),
        pa.field("year", pa.int32()),
        pa.field("section", pa.string()),
        pa.field("page", pa.int32()),
        pa.field("chunk_id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("heading_path", pa.string()),
        pa.field("previous_chunk_id", pa.string()),
        pa.field("next_chunk_id", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), DEFAULT_VECTOR_DIM)),
    ]
)

_DOCUMENT_SCHEMA = pa.schema(
    [
        pa.field("paper_id", pa.string()),
        pa.field("file_name", pa.string()),
        pa.field("title", pa.string()),
        pa.field("authors_json", pa.string()),
        pa.field("year", pa.int32()),
        pa.field("num_pages", pa.int32()),
        pa.field("num_chunks", pa.int32()),
        pa.field("sha256", pa.string()),
        pa.field("indexed_at", pa.string()),
    ]
)


@lru_cache(maxsize=4)
def _embedder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _embed_texts(texts: list[str], model_name: str) -> list[list[float]]:
    """E5-style embedding: passages are prefixed with ``passage: ``."""
    prefixed = [f"passage: {text}" for text in texts]
    model = _embedder(model_name)
    vectors = model.encode(
        prefixed, batch_size=32, normalize_embeddings=True, convert_to_numpy=True
    )
    return [list(map(float, row)) for row in vectors]


def _json_authors(authors: list[str]) -> str:
    return json.dumps(authors, ensure_ascii=False)


class LiteratureIndex:
    """Thin wrapper around the LanceDB index directory."""

    def __init__(
        self,
        index_dir: Path,
        *,
        embedding_model: str = "intfloat/multilingual-e5-small",
        vector_dim: int = DEFAULT_VECTOR_DIM,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.embedding_model = embedding_model
        self.vector_dim = vector_dim
        self._db: Any | None = None

    @property
    def db(self) -> Any:
        if self._db is None:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self.index_dir))
            self._ensure_tables()
        return self._db

    def _ensure_tables(self) -> None:
        existing = set(self._db.list_tables().tables)
        if PASSAGES_TABLE not in existing:
            self._db.create_table(PASSAGES_TABLE, schema=_PASSAGE_SCHEMA)
        if DOCUMENTS_TABLE not in existing:
            self._db.create_table(DOCUMENTS_TABLE, schema=_DOCUMENT_SCHEMA)

    def passage_table(self) -> Any:
        return self.db.open_table(PASSAGES_TABLE)

    def document_table(self) -> Any:
        return self.db.open_table(DOCUMENTS_TABLE)

    def count_passages(self) -> int:
        try:
            return self.passage_table().count_rows()
        except Exception:
            return 0

    def existing_documents(self) -> dict[str, dict[str, Any]]:
        """Map paper_id -> row for all currently indexed documents."""
        rows = self.document_table().to_arrow().to_pylist()
        return {row["paper_id"]: row for row in rows}

    def _delete_paper_passages(self, paper_id: str) -> None:
        table = self.passage_table()
        if table.count_rows() == 0:
            return
        table.delete(f"paper_id = '{paper_id}'")

    def _upsert_document(self, record: PaperRecord) -> None:
        table = self.document_table()
        with contextlib.suppress(Exception):
            table.delete(f"paper_id = '{record.paper_id}'")
        table.add(
            [
                {
                    "paper_id": record.paper_id,
                    "file_name": record.file_name,
                    "title": record.title,
                    "authors_json": _json_authors(record.authors),
                    "year": record.year,
                    "num_pages": record.num_pages,
                    "num_chunks": record.num_chunks,
                    "sha256": record.sha256,
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                }
            ]
        )

    def index_directory(
        self,
        root: Path,
        *,
        on_progress: Any = None,
    ) -> dict[str, Any]:
        """Index (or update) every PDF under ``root`` recursively.

        ``on_progress`` (optional) is called after every PDF with the
        running stats dict, so long runs can report/resume in batches.
        """
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"literature root does not exist: {root}")
        pdfs = sorted(path for path in root.rglob("*.pdf") if path.is_file())
        known = self.existing_documents()
        stats = {
            "indexed": 0,
            "skipped": 0,
            "failed": 0,
            "removed": 0,
            "chunks": 0,
            "errors": [],
        }
        seen: set[str] = set()
        for pdf_path in pdfs:
            paper_id = paper_id_for(pdf_path)
            seen.add(paper_id)
            prior = known.get(paper_id)
            if prior and prior.get("file_name") == pdf_path.name:
                # Same path: skip only when the bytes are unchanged.
                import hashlib

                digest = hashlib.sha256()
                with open(pdf_path, "rb") as handle:
                    for block in iter(lambda: handle.read(65536), b""):
                        digest.update(block)
                if prior.get("sha256") == digest.hexdigest():
                    stats["skipped"] += 1
                    continue
            try:
                record, passages = parse_pdf(pdf_path)
            except Exception as exc:  # noqa: BLE001 - one bad PDF must not stop the run
                stats["failed"] += 1
                stats["errors"].append(f"{pdf_path.name}: {type(exc).__name__}: {exc}")
                logger.warning("failed to parse %s: %s", pdf_path, exc)
                continue
            if not passages:
                stats["failed"] += 1
                stats["errors"].append(f"{pdf_path.name}: no text chunks extracted")
                continue
            vectors = _embed_texts([p.text for p in passages], self.embedding_model)
            rows = []
            for passage, vector in zip(passages, vectors):
                row = passage.to_retrieval_row()
                row["vector"] = vector
                rows.append(row)
            self._delete_paper_passages(paper_id)
            self.passage_table().add(rows)
            record.num_chunks = len(rows)
            self._upsert_document(record)
            stats["indexed"] += 1
            stats["chunks"] += len(rows)
            logger.info("indexed %s (%d chunks)", pdf_path.name, len(rows))
            if on_progress is not None:
                on_progress(dict(stats))
        # Drop papers whose files disappeared from the literature root.
        for paper_id, row in known.items():
            if paper_id not in seen:
                self._delete_paper_passages(paper_id)
                with contextlib.suppress(Exception):
                    self.document_table().delete(f"paper_id = '{paper_id}'")
                stats["removed"] += 1
        stats["chunks"] = self.count_passages()
        stats["db_location"] = str(self.index_dir)
        return stats

    def get_passage(self, passage_id: str) -> dict[str, Any] | None:
        dataset = self.passage_table().to_lance()
        if dataset.count_rows() == 0:
            return None
        rows = dataset.to_table(
            filter=f"passage_id = '{passage_id}'", limit=1
        ).to_pylist()
        if not rows:
            return None
        row = dict(rows[0])
        row.pop("vector", None)
        with contextlib.suppress(Exception):
            row["authors"] = json.loads(row.get("authors_json") or "[]")
        row.pop("authors_json", None)
        return row

    def all_passages(self) -> list[dict[str, Any]]:
        """All passage rows without vectors (for lexical retrieval)."""
        dataset = self.passage_table().to_lance()
        if dataset.count_rows() == 0:
            return []
        columns = [
            "passage_id",
            "paper_id",
            "file_name",
            "title",
            "authors_json",
            "year",
            "section",
            "page",
            "chunk_id",
            "text",
            "heading_path",
            "previous_chunk_id",
            "next_chunk_id",
        ]
        rows = dataset.to_table(columns=columns).to_pylist()
        for row in rows:
            with contextlib.suppress(Exception):
                row["authors"] = json.loads(row.get("authors_json") or "[]")
            row.pop("authors_json", None)
        return rows

    @contextlib.contextmanager
    def locked(self):
        """Serialise index runs with a simple lock file."""
        lock_path = self.index_dir / ".index.lock"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        lock_file = open(lock_path, "w")
        try:
            if os.name == "posix":
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            if os.name == "posix":
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
