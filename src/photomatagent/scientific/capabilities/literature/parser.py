"""PDF -> DoclingDocument -> structured chunks.

The parser is deliberately thin: it only converts one PDF into paper/chunk
records with provenance. Embedding and storage happen in ``index.py``.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from photomatagent.scientific.capabilities.literature.models import (
    PaperRecord,
    PassageRecord,
)

_YEAR_RE = re.compile(r"(?:^|[_\s])(\d{4})(?:[_\s]|$)")
_CLEAN_TITLE_RE = re.compile(r"[_-]+")


@lru_cache(maxsize=1)
def _converter():
    """Shared docling converter: layout models load once per process."""
    from docling.document_converter import DocumentConverter

    return DocumentConverter()


@lru_cache(maxsize=1)
def _chunker():
    """Shared HybridChunker: tokenizer loads once per process."""
    from docling.chunking import HybridChunker

    return HybridChunker()


def paper_id_for(pdf_path: Path) -> str:
    """Stable, content-agnostic paper id derived from the resolved path.

    ``resolve()`` normalises relative vs absolute paths and follows
    symlinks, so the same PDF gets the same id no matter how the indexer
    was invoked (and a symlinked paper aliases its real file).
    """
    return hashlib.sha1(str(pdf_path.resolve()).encode("utf-8")).hexdigest()[:12]


def _sha256(pdf_path: Path) -> str:
    digest = hashlib.sha256()
    with open(pdf_path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _year_from_name(pdf_path: Path, title: str) -> int | None:
    for candidate in (pdf_path.name, pdf_path.stem, title):
        match = _YEAR_RE.search(candidate)
        if match:
            try:
                year = int(match.group(1))
                if 1900 <= year <= 2100:
                    return year
            except ValueError:
                continue
    return None


def _chunk_meta(chunk: Any) -> dict[str, Any]:
    """Normalise docling chunk metadata across API versions."""
    meta = getattr(chunk, "meta", None)
    if meta is None:
        return {}
    headings = list(getattr(meta, "headings", None) or [])
    # Page numbers are not stored directly on the meta in docling-core 2.x;
    # derive them from the provenance of the chunk's doc items.
    pages: list[int] = []
    for item in getattr(meta, "doc_items", None) or []:
        for prov in getattr(item, "prov", None) or []:
            page_no = getattr(prov, "page_no", None)
            if isinstance(page_no, int) and page_no not in pages:
                pages.append(page_no)
    pages.sort()
    if not pages and hasattr(meta, "page_numbers"):
        pages = [int(p) for p in (meta.page_numbers or [])]
    return {"headings": headings, "page_numbers": pages}


def parse_pdf(pdf_path: Path) -> tuple[PaperRecord, list[PassageRecord]]:
    """Parse one PDF into a paper record plus chunk records (no embedding).

    Returns (paper_record, passages) where every passage has provenance
    (section, page, heading path, neighbours). Raises on unreadable PDFs so
    the indexer can record per-file failures.
    """
    result = _converter().convert(pdf_path)
    doc = result.document

    title = (getattr(doc, "name", "") or pdf_path.stem).strip()
    title = _CLEAN_TITLE_RE.sub(" ", title).strip()
    year = _year_from_name(pdf_path, title)
    paper_id = paper_id_for(pdf_path)

    chunks = list(_chunker().chunk(doc))

    passages: list[PassageRecord] = []
    for index, chunk in enumerate(chunks):
        text = (getattr(chunk, "text", "") or "").strip()
        if not text:
            continue
        meta = _chunk_meta(chunk)
        headings = [h for h in meta.get("headings", []) if h]
        pages = meta.get("page_numbers", [])
        heading_path = " > ".join(headings)
        passage_id = f"{paper_id}:{index:04d}"
        passage = PassageRecord(
            passage_id=passage_id,
            paper_id=paper_id,
            file_name=pdf_path.name,
            title=title,
            year=year,
            section=headings[-1] if headings else "",
            page=pages[0] if pages else None,
            chunk_id=passage_id,
            text=text,
            heading_path=heading_path,
        )
        if passages:
            passages[-1].next_chunk_id = passage_id
            passage.previous_chunk_id = passages[-1].chunk_id
        passages.append(passage)

    num_pages = getattr(doc, "num_pages", 0)
    if callable(num_pages):
        try:
            num_pages = num_pages()
        except Exception:
            num_pages = len(getattr(doc, "pages", None) or {})
    record = PaperRecord(
        paper_id=paper_id,
        file_name=pdf_path.name,
        title=title,
        year=year,
        num_pages=num_pages or 0,
        num_chunks=len(passages),
        sha256=_sha256(pdf_path),
    )
    return record, passages
