"""Data models for the local literature RAG index.

``PaperRecord`` describes one indexed PDF; ``PassageRecord`` is one retrievable
chunk. Every passage carries full provenance (paper, section, page, heading
path, neighbours) so results can always be traced back to a source file.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePosixPath, PureWindowsPath
import re
from typing import Any

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _validate_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be 64 lowercase hexadecimal characters")


def validate_relative_source_path(value: str) -> str:
    """Validate a workspace-relative POSIX source path.

    Qdrant payloads deliberately contain no absolute host paths.  Rejecting
    both POSIX and Windows absolute/parent paths here keeps the contract true
    even when a workspace is indexed on a different operating system.
    """
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("relative_source_path must be a non-empty path")
    windows_path = PureWindowsPath(value)
    if (
        PurePosixPath(value).is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
    ):
        raise ValueError("relative_source_path must be workspace-relative")
    parts = PurePosixPath(value.replace("\\", "/")).parts
    if not parts or parts == (".",) or ".." in parts:
        raise ValueError("relative_source_path must not escape the workspace")
    return value


class DocumentStatus(str, Enum):
    """Lifecycle state of one document control record."""

    PENDING = "pending"
    STAGED = "staged"
    READY = "ready"
    FAILED = "failed"
    DELETED = "deleted"


class IngestState(str, Enum):
    """Lifecycle state of a document revision in the passage collection."""

    STAGED = "staged"
    READY = "ready"
    SUPERSEDED = "superseded"


@dataclass(frozen=True)
class DocumentManifest:
    """Vectorless document control record persisted in Qdrant.

    The field names mirror the document payload contract in the Qdrant RAG
    design.  ``authors`` is normalised to a tuple so the frozen contract is
    not made mutable by a caller-provided list.
    """

    schema_version: int
    record_type: str
    workspace_id: str
    document_id: str
    relative_source_path: str
    file_name: str
    content_sha256: str
    status: DocumentStatus
    title: str = ""
    authors: tuple[str, ...] = ()
    year: int | None = None
    num_pages: int = 0
    chunk_count: int = 0
    model_fingerprint: str = ""
    indexed_at: datetime | None = None
    last_error: str = ""

    def __post_init__(self) -> None:
        validate_relative_source_path(self.relative_source_path)
        _validate_sha256(self.content_sha256, "content_sha256")
        _validate_sha256(self.model_fingerprint, "model_fingerprint")
        if self.record_type != "document":
            raise ValueError("DocumentManifest.record_type must be 'document'")
        object.__setattr__(self, "status", DocumentStatus(self.status))
        object.__setattr__(self, "authors", tuple(self.authors))

    def to_payload(self) -> dict[str, Any]:
        """Return the exact document payload shape used by the adapter."""
        return {
            "schema_version": self.schema_version,
            "record_type": self.record_type,
            "workspace_id": self.workspace_id,
            "document_id": self.document_id,
            "relative_source_path": self.relative_source_path,
            "file_name": self.file_name,
            "content_sha256": self.content_sha256,
            "status": self.status.value,
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "num_pages": self.num_pages,
            "chunk_count": self.chunk_count,
            "model_fingerprint": self.model_fingerprint,
            "indexed_at": self.indexed_at,
            "last_error": self.last_error,
        }


@dataclass(frozen=True)
class PassagePoint:
    """A dense-vector passage plus its provenance payload."""

    schema_version: int
    record_type: str
    workspace_id: str
    document_id: str
    document_revision: str
    ingest_state: IngestState
    passage_id: str
    chunk_index: int
    text: str
    title: str
    authors: tuple[str, ...]
    year: int | None
    section: str
    heading_path: str
    page_start: int | None
    page_end: int | None
    previous_passage_id: str | None
    next_passage_id: str | None
    relative_source_path: str
    model_fingerprint: str
    limitations: tuple[str, ...] = ()
    dense: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        validate_relative_source_path(self.relative_source_path)
        _validate_sha256(self.document_revision, "document_revision")
        _validate_sha256(self.model_fingerprint, "model_fingerprint")
        if self.record_type != "passage":
            raise ValueError("PassagePoint.record_type must be 'passage'")
        object.__setattr__(self, "ingest_state", IngestState(self.ingest_state))
        object.__setattr__(self, "authors", tuple(self.authors))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        object.__setattr__(self, "dense", tuple(float(value) for value in self.dense))

    def to_payload(self) -> dict[str, Any]:
        """Return the payload fields independent of the Qdrant vector."""
        return {
            "schema_version": self.schema_version,
            "record_type": self.record_type,
            "workspace_id": self.workspace_id,
            "document_id": self.document_id,
            "document_revision": self.document_revision,
            "ingest_state": self.ingest_state.value,
            "passage_id": self.passage_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "section": self.section,
            "heading_path": self.heading_path,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "previous_passage_id": self.previous_passage_id,
            "next_passage_id": self.next_passage_id,
            "relative_source_path": self.relative_source_path,
            "model_fingerprint": self.model_fingerprint,
            "limitations": list(self.limitations),
        }


class PaperRecord(BaseModel):
    """Metadata for one indexed PDF paper."""

    paper_id: str
    file_name: str
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    num_pages: int = 0
    num_chunks: int = 0
    sha256: str = ""
    indexed_at: datetime = Field(default_factory=_now)


class PassageRecord(BaseModel):
    """One chunk of one paper, with full traceability metadata."""

    passage_id: str
    paper_id: str
    file_name: str
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    section: str = ""
    page: int | None = None
    chunk_id: str = ""
    text: str = ""
    heading_path: str = ""
    previous_chunk_id: str = ""
    next_chunk_id: str = ""

    def to_retrieval_row(self) -> dict[str, Any]:
        """Flat dict for LanceDB inserts (lists serialised as JSON)."""
        import json

        return {
            "passage_id": self.passage_id,
            "paper_id": self.paper_id,
            "file_name": self.file_name,
            "title": self.title,
            "authors_json": json.dumps(self.authors, ensure_ascii=False),
            "year": self.year,
            "section": self.section,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "heading_path": self.heading_path,
            "previous_chunk_id": self.previous_chunk_id,
            "next_chunk_id": self.next_chunk_id,
        }
