"""Data models for the local literature RAG index.

``PaperRecord`` describes one indexed PDF; ``PassageRecord`` is one retrievable
chunk. Every passage carries full provenance (paper, section, page, heading
path, neighbours) so results can always be traced back to a source file.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


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
