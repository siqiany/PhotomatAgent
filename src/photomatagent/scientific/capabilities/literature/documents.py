"""Workspace-owned structured documents, independent of the search index."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from threading import Lock
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, Field, PrivateAttr

from photomatagent.errors import ToolExecutionError
from photomatagent.models.image_assets import register_image_bytes
from photomatagent.models.types import ImageReference
from photomatagent.persistence.atomic import atomic_write
from photomatagent.scientific.capabilities.literature.models import (
    PaperRecord, PassageRecord, STRUCTURED_INDEX_PREFIX, StructuredPassageReference,
)
from photomatagent.workspace import Workspace

_PDF_RENDER_LOCK = Lock()
MAX_PDF_BYTES = 32 * 1024 * 1024
MAX_OBJECTS = 5000
MAX_RESPONSE_OBJECTS = 20
MAX_RESPONSE_OBJECT_METADATA_BYTES = 3_000
MAX_NATIVE_DOCUMENT_BYTES = 64 * 1024 * 1024
_PARTIAL_DIAGNOSTICS = {"DOCLING_PARTIAL_OR_FAILED", "DOCUMENT_OBJECT_LIMIT"}


class DocumentError(ValueError):
    """A bounded diagnostic suitable for returning to the caller."""


def _json_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


def _text_fragment(value: str, offset: int, budget: int) -> str:
    start, stop = offset, min(len(value), offset + budget)
    while start < stop:
        middle = (start + stop + 1) // 2
        if _json_bytes(value[offset:middle]) <= budget:
            start = middle
        else:
            stop = middle - 1
    return value[offset:start]


class DocumentObject(BaseModel):
    object_id: str
    kind: Literal["paragraph", "heading", "table", "figure", "equation", "caption", "reference", "footnote"]
    section_id: str | None = None
    text: str = ""
    original_text: str = ""
    table_cells: list[dict[str, Any]] = Field(default_factory=list)
    anchors: list[dict[str, Any]] = Field(default_factory=list)
    related: list[str] = Field(default_factory=list)
    parent_id: str | None = None
    relations: dict[str, list[str]] = Field(default_factory=dict)
    table_continuations: dict[str, Literal["explicit_caption", "matching_header_at_page_break"]] = Field(default_factory=dict)
    quality: Literal["anchored", "unknown"]


class ParsedDocument(BaseModel):
    schema_version: Literal[1] = 1
    title: str
    objects: list[DocumentObject]
    diagnostics: list[str] = Field(default_factory=list)
    source_page_count: int | None = None
    native_snapshot_available: bool = False
    native_snapshot_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    _native_document: Any = PrivateAttr(default=None)


class DocumentArtifact(BaseModel):
    schema_version: Literal[1] = 1
    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_url: str = ""
    media_type: Literal["application/pdf", "text/html"] = "application/pdf"


class PublicationContext(BaseModel):
    role: Literal["paper", "supplement"] = "paper"
    publication_id: str = Field(default="", max_length=250)
    work_id: str = Field(default="", max_length=250)
    version: Literal["preprint", "published", "supplement", "unknown"] = "unknown"
    supplement_urls: tuple[str, ...] = Field(default=(), max_length=12)
    parent_document_id: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    parent_source_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")


Parser = Callable[[Path], ParsedDocument]


@dataclass(frozen=True, slots=True)
class IndexedDocumentQuery:
    indexed: bool | None
    references: tuple[StructuredPassageReference, ...] = ()
    mode: str | None = None
    reranked: bool = False
    degraded_reasons: tuple[str, ...] = ()


def structured_index_path(document_id: str, revision: str) -> PurePosixPath:
    return STRUCTURED_INDEX_PREFIX / document_id / (revision + ".pdf")


def parse_document(path: Path) -> ParsedDocument:
    """Use the shared Docling converter with operator-provisioned local assets."""
    from photomatagent.scientific.capabilities.literature.parser import _converter, document_timeout_seconds

    artifacts = os.environ.get("PHOTOMATAGENT_DOCLING_ARTIFACTS_PATH")
    page_count = None
    if path.suffix == ".pdf":
        import pypdfium2

        if not artifacts or not Path(artifacts).is_dir():
            raise DocumentError("DOCLING_ASSETS_NOT_CONFIGURED")
        with _PDF_RENDER_LOCK, pypdfium2.PdfDocument(path) as pdf:
            page_count = len(pdf)
        result = _converter(artifacts, document_timeout=document_timeout_seconds()).convert(
            path, max_num_pages=150, raises_on_error=False,
        )
    else:
        result = _converter().convert(path, raises_on_error=False)
    doc = result.document
    diagnostics = [] if result.status.value == "success" else ["DOCLING_PARTIAL_OR_FAILED"]
    kinds = {
        "section_header": "heading", "title": "heading", "table": "table",
        "picture": "figure", "formula": "equation", "caption": "caption",
        "reference": "reference", "footnote": "footnote",
    }
    objects: list[DocumentObject] = []
    section_id = None
    for item, _level in doc.iterate_items():
        if len(objects) >= MAX_OBJECTS:
            diagnostics.append("DOCUMENT_OBJECT_LIMIT")
            break
        kind = kinds.get(item.label.value, "paragraph")
        object_id = item.self_ref
        if kind == "heading":
            section_id = object_id
        text = str(getattr(item, "text", "") or "")
        original_text = str(getattr(item, "orig", "") or text)
        cells = []
        if kind == "table":
            text = item.export_to_markdown(doc=doc)
            cells = [cell.model_dump(mode="json") for cell in item.data.table_cells]
        anchors = [
            {"page": anchor.page_no, "bbox": anchor.bbox.model_dump(mode="json"),
             "charspan": list(anchor.charspan)}
            for anchor in item.prov
        ]
        relations = {name: [ref.cref for ref in getattr(item, name, ())]
                     for name in ("captions", "references", "footnotes") if getattr(item, name, ())}
        suspect = "\ufffd" in text
        if suspect and "DOCUMENT_OBJECT_TEXT_CORRUPTED" not in diagnostics:
            diagnostics.append("DOCUMENT_OBJECT_TEXT_CORRUPTED")
        objects.append(DocumentObject.model_validate({
            "object_id": object_id, "kind": kind, "section_id": section_id,
            "text": text, "original_text": original_text, "table_cells": cells,
            "anchors": anchors, "related": [ref for refs in relations.values() for ref in refs],
            "relations": relations, "parent_id": getattr(getattr(item, "parent", None), "cref", None),
            "quality": "anchored" if anchors and text and not suspect else "unknown",
        }))
    if not objects:
        raise DocumentError("DOCUMENT_PARSE_EMPTY")
    if not any("method" in obj.text.casefold() for obj in objects if obj.kind == "heading"):
        diagnostics.append("METHODS_HEADING_NOT_FOUND")
    if any(obj.kind == "equation" and not obj.text.strip() for obj in objects):
        diagnostics.append("FORMULA_TEXT_UNAVAILABLE")
    if any(obj.kind == "table" and not obj.table_cells for obj in objects):
        diagnostics.append("TABLE_CELLS_UNAVAILABLE")
    objects, unresolved = link_table_continuations(objects, {
        int(page_no): item.size.height for page_no, item in doc.pages.items()
    })
    if unresolved:
        diagnostics.append("TABLE_CONTINUATION_UNRESOLVED")
    parsed = ParsedDocument(title=doc.name, objects=objects, diagnostics=diagnostics, source_page_count=page_count)
    parsed._native_document = doc
    return parsed


_TABLE_CAPTION = re.compile(r"^\s*(?:supplementary\s+)?table\s+([A-Za-z]?\d+)\b", re.IGNORECASE)
_CONTINUED = re.compile(r"\b(?:continued|cont\.)", re.IGNORECASE)


def _tables_at_page_break(
    previous: DocumentObject, current: DocumentObject, heights: dict[int, float],
) -> bool:
    first, second = previous.anchors[0], current.anchors[0]
    coordinates = []
    for anchor in (first, second):
        box = anchor.get("bbox", {})
        page = anchor.get("page")
        height = heights.get(page, 0) if type(page) is int else 0
        if not height or box.get("coord_origin") not in {"TOPLEFT", "BOTTOMLEFT"}:
            return False
        left, right, top, bottom = (box.get(name) for name in ("l", "r", "t", "b"))
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (left, right, top, bottom)):
            return False
        if box["coord_origin"] == "TOPLEFT":
            top, bottom = height - top, height - bottom
        coordinates.append((left, right, top / height, bottom / height))
    left, right, _top, bottom = coordinates[0]
    next_left, next_right, next_top, _bottom = coordinates[1]
    tolerance = 0.04 * (right - left)
    return (bottom <= 0.2 and next_top >= 0.78 and tolerance > 0
            and abs(left - next_left) <= tolerance and abs(right - next_right) <= tolerance)


def link_table_continuations(
    objects: list[DocumentObject], page_heights: dict[int, float] | None = None,
) -> tuple[list[DocumentObject], bool]:
    """Relate table fragments without merging their independently anchored cells."""
    by_id = {obj.object_id: obj for obj in objects}
    fragments: dict[int, list[tuple[DocumentObject, tuple[tuple[int, str], ...], str | None]]] = {}
    links: dict[str, dict[str, Literal["explicit_caption", "matching_header_at_page_break"]]] = {}
    unresolved = False
    for obj in objects:
        if obj.kind != "table":
            continue
        pages = {anchor["page"] for anchor in obj.anchors if type(anchor.get("page")) is int}
        if len(pages) != 1:
            continue
        page = next(iter(pages))
        captions = [by_id[reference].text for reference in obj.related
                    if reference in by_id and by_id[reference].kind == "caption"]
        labels = [(match[1].casefold(), bool(_CONTINUED.search(text)))
                  for text in captions if (match := _TABLE_CAPTION.match(text))]
        label, continued = labels[0] if len(labels) == 1 else (None, False)
        header = tuple(sorted(
            (int(cell["start_col_offset_idx"]), " ".join(str(cell.get("text", "")).casefold().split()))
            for cell in obj.table_cells if cell.get("column_header") is True
            and type(cell.get("start_col_offset_idx")) is int
        ))
        prior_rows = [(prior, prior_label) for prior, signature, prior_label in fragments.get(page - 1, ())
                      if header and signature == header]
        basis: Literal["explicit_caption", "matching_header_at_page_break"] = "explicit_caption"
        if continued:
            prior_rows = [(prior, prior_label) for prior, prior_label in prior_rows if prior_label == label]
            unresolved |= len(prior_rows) != 1
        elif not labels and len(header) >= 3 and page_heights:
            prior_rows = [(prior, prior_label) for prior, prior_label in prior_rows
                          if prior_label and _tables_at_page_break(prior, obj, page_heights)]
            basis = "matching_header_at_page_break"
        else:
            prior_rows = []
        if len(prior_rows) == 1:
            prior, prior_label = prior_rows[0]
            links.setdefault(prior.object_id, {})[obj.object_id] = basis
            links.setdefault(obj.object_id, {})[prior.object_id] = basis
            label = label or prior_label
        fragments.setdefault(page, []).append((obj, header, label))
    return [
        obj.model_copy(update={"related": list(dict.fromkeys([*obj.related, *links[obj.object_id]])),
                               "table_continuations": links[obj.object_id]})
        if obj.object_id in links else obj for obj in objects
    ], unresolved


def _hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class DocumentService:
    def __init__(self, workspace: Workspace, parser: Parser = parse_document, *, wait_seconds: float = 10) -> None:
        self.workspace = workspace
        self._parser = parser
        self._parse_lock = asyncio.Lock()
        self._converter_lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Task[ParsedDocument]] = {}
        self._wait_seconds = wait_seconds

    async def aclose(self) -> None:
        if self._pending:
            await asyncio.gather(*self._pending.values(), return_exceptions=True)
            self._pending.clear()

    async def _parse_serially(self, revision: str, content: bytes, is_pdf: bool) -> ParsedDocument:
        async with self._converter_lock:
            return await asyncio.to_thread(self._parse, revision, content, is_pdf)

    def _parse_finished(self, revision: str, worker: asyncio.Task[ParsedDocument]) -> None:
        self._pending.pop(revision, None)
        if not worker.cancelled():
            worker.exception()  # Observe a background failure even if the caller timed out.

    def _cache(self, path: str) -> Path:
        # The path components are fixed strings or validated digests.
        return self.workspace.resolve(f".photomatagent/literature/documents/{path}", must_exist=False)

    def local_path(self, filename: str) -> Path:
        if Path(filename).name != filename or not filename.lower().endswith(".pdf"):
            raise DocumentError("DOCUMENT_PATH_INVALID")
        try:
            root = self.workspace.resolve("papers", must_exist=False)
            path = self.workspace.resolve(f"papers/{filename}")
        except ToolExecutionError as exc:
            raise DocumentError("DOCUMENT_NOT_FOUND_OR_OUTSIDE_WORKSPACE") from exc
        if path.parent != root or not path.is_file():
            raise DocumentError("DOCUMENT_NOT_FOUND_OR_OUTSIDE_WORKSPACE")
        return path

    @staticmethod
    def _read_pdf(path: Path) -> bytes:
        with path.open("rb") as handle:
            content = handle.read(MAX_PDF_BYTES + 1)
        if len(content) > MAX_PDF_BYTES:
            raise DocumentError("DOCUMENT_TOO_LARGE")
        if not content.startswith(b"%PDF-"):
            raise DocumentError("DOCUMENT_NOT_PDF")
        return content

    def _artifact(
        self, path: Path, content: bytes, source_url: str = "",
        media_type: Literal["application/pdf", "text/html"] = "application/pdf",
    ) -> DocumentArtifact:
        relative = self.workspace.relative(path)
        digest = _hash_bytes(content)
        doc_id = _hash_bytes(json.dumps([relative, digest, source_url]).encode())
        return DocumentArtifact(document_id=doc_id, path=relative, sha256=digest, source_url=source_url, media_type=media_type)

    def register_pdf(self, content: bytes, source_url: str) -> DocumentArtifact:
        if len(content) > MAX_PDF_BYTES or not content.startswith(b"%PDF-"):
            raise DocumentError("DOCUMENT_NOT_PDF_OR_TOO_LARGE")
        return self._register(content, source_url, "application/pdf")

    def register_html(self, content: bytes, source_url: str) -> DocumentArtifact:
        if not content or len(content) > 1_000_000:
            raise DocumentError("DOCUMENT_HTML_EMPTY_OR_TOO_LARGE")
        return self._register(content, source_url, "text/html")

    def _register(
        self, content: bytes, source_url: str, media_type: Literal["application/pdf", "text/html"],
    ) -> DocumentArtifact:
        suffix = "pdf" if media_type == "application/pdf" else "html"
        path = self._cache(f"sources/{_hash_bytes(content)}.{suffix}")
        artifact = self._artifact(path, content, source_url, media_type)
        atomic_write(path, content)
        self._save(artifact)
        return artifact

    def _save(self, artifact: DocumentArtifact) -> None:
        atomic_write(
            self._cache(f"artifacts/{artifact.document_id}.json"),
            artifact.model_dump_json().encode(),
        )

    def publication_context(self, document_id: str) -> PublicationContext | None:
        if not re.fullmatch(r"[0-9a-f]{64}", document_id):
            raise DocumentError("DOCUMENT_ID_INVALID")
        path = self._cache(f"publications/{document_id}.json")
        return PublicationContext.model_validate_json(path.read_bytes()) if path.is_file() else None

    def bind_publication(self, document_id: str, context: PublicationContext) -> None:
        import fcntl

        self._resolve(None, document_id)
        path = self._cache(f"publications/{document_id}.json")
        lock = self._cache(f"publications/{document_id}.lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        with os.fdopen(os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600), "a+") as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            previous = self.publication_context(document_id)
            if previous is not None and previous != context:
                raise DocumentError("PUBLICATION_IDENTITY_CONFLICT")
            if previous is None:
                atomic_write(path, context.model_dump_json().encode())

    def supplement_parent(self, document_id: str, requested_url: str) -> tuple[DocumentArtifact, PublicationContext]:
        parent, _ = self._resolve(None, document_id)
        context = self.publication_context(document_id)
        if (context is None or context.role != "paper" or not context.publication_id
                or requested_url not in context.supplement_urls):
            raise DocumentError("SUPPLEMENT_NOT_LINKED_BY_PUBLICATION")
        return parent, context

    def record_publication_relation(self, source: str, relation: str, target: str) -> None:
        for key in {source, target}:
            row = {"source": source, "relation": relation, "target": target, "provider": "crossref"}
            payload = json.dumps(row, sort_keys=True).encode()
            path = self._cache(f"publication-relations/{_hash_bytes(key.encode())}/{_hash_bytes(payload)}.json")
            atomic_write(path, payload)

    def publication_relations(self, *identifiers: str) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for identifier in dict.fromkeys(filter(None, identifiers)):
            folder = self._cache(f"publication-relations/{_hash_bytes(identifier.encode())}")
            if folder.is_dir():
                for path in sorted(folder.glob("*.json"))[:8]:
                    if not path.is_symlink() and path.stat().st_size <= 1024:
                        raw = path.read_bytes()
                        if path.stem == _hash_bytes(raw):
                            row = json.loads(raw)
                            if isinstance(row, dict) and row not in rows:
                                rows.append(row)
        return rows[:8]

    def _resolve(self, filename: str | None, document_id: str | None) -> tuple[DocumentArtifact, bytes]:
        if bool(filename) == bool(document_id):
            raise DocumentError("DOCUMENT_INPUT_REQUIRES_FILE_OR_ID")
        if filename:
            path = self.local_path(filename)
            content = self._read_pdf(path)
            return self._artifact(path, content), content
        assert document_id is not None
        if len(document_id) != 64 or any(ch not in "0123456789abcdef" for ch in document_id):
            raise DocumentError("DOCUMENT_ID_INVALID")
        manifest = self._cache(f"artifacts/{document_id}.json")
        if not manifest.is_file():
            raise DocumentError("DOCUMENT_NOT_REGISTERED")
        artifact = DocumentArtifact.model_validate_json(manifest.read_bytes())
        if artifact.source_url:
            suffix = "pdf" if artifact.media_type == "application/pdf" else "html"
            path = self._cache(f"sources/{artifact.sha256}.{suffix}")
        else:
            path = self.local_path(Path(artifact.path).name)
        if artifact.media_type == "application/pdf":
            content = self._read_pdf(path)
        else:
            with path.open("rb") as handle:
                content = handle.read(1_000_001)
            if not content or len(content) > 1_000_000:
                raise DocumentError("DOCUMENT_HTML_EMPTY_OR_TOO_LARGE")
        if (artifact.document_id != document_id
                or self._artifact(path, content, artifact.source_url, artifact.media_type) != artifact):
            raise DocumentError("DOCUMENT_REVISION_CHANGED")
        return artifact, content

    @staticmethod
    def _revision(artifact: DocumentArtifact) -> str:
        from photomatagent.scientific.capabilities.literature.parser import document_timeout_seconds
        try:
            parser_version = version("docling")
        except PackageNotFoundError:
            parser_version = "unavailable"
        profile = (
            os.environ.get("PHOTOMATAGENT_DOCLING_ARTIFACTS_PATH", ""),
            os.environ.get("PHOTOMATAGENT_DOCLING_LOW_MEMORY", ""),
            document_timeout_seconds(),
        )
        return _hash_bytes(json.dumps([4, artifact.sha256, artifact.media_type, parser_version, profile]).encode())

    def cached(self, filename: str) -> tuple[DocumentArtifact, ParsedDocument | None]:
        artifact, _ = self._resolve(filename, None)
        return artifact, self._load(self._revision(artifact))

    def crop_objects(
        self, document_id: str, revision: str, object_ids: tuple[str, ...],
    ) -> tuple[list[dict[str, Any]], tuple[ImageReference, ...], list[str]]:
        import pypdfium2

        artifact, content = self._resolve(None, document_id)
        if revision != self._revision(artifact):
            raise DocumentError("DOCUMENT_REVISION_CHANGED")
        if artifact.media_type != "application/pdf":
            raise DocumentError("DOCUMENT_CROP_REQUIRES_PDF")
        document = self._load(revision)
        if document is None:
            raise DocumentError("DOCUMENT_NOT_PARSED")
        rows: list[dict[str, Any]] = []
        references: list[ImageReference] = []
        diagnostics: list[str] = []
        with _PDF_RENDER_LOCK, pypdfium2.PdfDocument(content) as pdf:
            for obj in document.objects:
                if obj.object_id not in object_ids:
                    continue
                for anchor in obj.anchors:
                    if len(rows) == 2:
                        return rows, tuple(references), [*diagnostics, "DOCUMENT_CROP_LIMIT"]
                    page_no = anchor.get("page")
                    bbox = anchor.get("bbox")
                    if type(page_no) is not int or not 1 <= page_no <= len(pdf) or not isinstance(bbox, dict):
                        continue
                    page = pdf[page_no - 1]
                    try:
                        image = self._crop_page(page, bbox)
                    except (ValueError, RuntimeError, ToolExecutionError):
                        diagnostics.append("DOCUMENT_CROP_ANCHOR_UNSUPPORTED")
                        continue
                    finally:
                        page.close()
                    rows.append({"object_id": obj.object_id, "page": page_no, "bbox": bbox,
                                 "source_sha256": artifact.sha256, "revision": revision,
                                 "image": image.model_dump(mode="json")})
                    references.append(image)
        if not rows and not diagnostics:
            diagnostics.append("DOCUMENT_CROP_ANCHOR_UNAVAILABLE")
        return rows, tuple(references), list(dict.fromkeys(diagnostics))

    def _crop_page(self, page: Any, bbox: dict[str, Any]) -> ImageReference:
        width, height = page.get_size()
        if page.get_rotation() or not (0 < width <= 10000 and 0 < height <= 10000):
            raise ValueError("DOCUMENT_CROP_PAGE_UNSUPPORTED")
        if page.get_bbox()[:2] != (0, 0):
            raise ValueError("DOCUMENT_CROP_PAGE_UNSUPPORTED")
        values: list[float] = []
        for name in ("l", "t", "r", "b"):
            value = bbox.get(name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError("DOCUMENT_CROP_ANCHOR_UNSUPPORTED")
            values.append(float(value))
        left, top, right, bottom = values
        if bbox.get("coord_origin") == "TOPLEFT":
            top, bottom = height - top, height - bottom
        elif bbox.get("coord_origin") != "BOTTOMLEFT":
            raise ValueError("DOCUMENT_CROP_ANCHOR_UNSUPPORTED")
        if not (0 <= left < right <= width and 0 <= bottom < top <= height):
            raise ValueError("DOCUMENT_CROP_ANCHOR_UNSUPPORTED")
        scale = min(2.0, 1200 / max(right - left, top - bottom))
        if min(right - left, top - bottom) * scale < 2:
            raise ValueError("DOCUMENT_CROP_ANCHOR_UNSUPPORTED")
        bitmap = page.render(scale=scale, crop=(left, bottom, width - right, height - top),
                             may_draw_forms=False, draw_annots=False)
        try:
            buffer = BytesIO()
            bitmap.to_pil().save(buffer, format="PNG")
            return register_image_bytes(self.workspace, buffer.getvalue())
        finally:
            bitmap.close()

    def index_projection(self, document_id: str) -> tuple[Path, PaperRecord, list[PassageRecord], str]:
        artifact, content = self._resolve(None, document_id)
        if artifact.media_type != "application/pdf":
            raise DocumentError("DOCUMENT_INDEX_COVERAGE_UNCONFIRMED")
        revision = self._revision(artifact)
        document = self._load(revision)
        if document is None:
            raise DocumentError("DOCUMENT_NOT_PARSED")
        if _PARTIAL_DIAGNOSTICS.intersection(document.diagnostics):
            raise DocumentError("DOCUMENT_INDEX_COVERAGE_UNCONFIRMED")
        headings = {obj.object_id: obj.text[:300] for obj in document.objects if obj.kind == "heading"}
        passages: list[PassageRecord] = []
        max_page = 0
        for obj in document.objects:
            if not obj.text.strip() or obj.kind == "heading":
                continue
            pages = [anchor["page"] for anchor in obj.anchors if type(anchor.get("page")) is int and anchor["page"] > 0]
            if pages:
                max_page = max(max_page, max(pages))
            for offset in range(0, len(obj.text), 2000):
                text = obj.text[offset:offset + 2000]
                if not text.strip():
                    continue
                section = headings.get(obj.section_id or "", "")
                passages.append(PassageRecord(
                    passage_id=f"{obj.object_id}:{offset}", paper_id=document_id,
                    file_name=Path(artifact.path).name, title=document.title,
                    page=min(pages) if pages else None, page_start=min(pages) if pages else None,
                    page_end=max(pages) if pages else None, section=section, heading_path=section,
                    text=text, structured_ref=StructuredPassageReference(
                        document_id=document_id, revision=revision, object_id=obj.object_id, text_offset=offset,
                    ),
                ))
                if len(passages) > 2048:
                    raise DocumentError("DOCUMENT_INDEX_PASSAGE_LIMIT")
        if not passages:
            raise DocumentError("DOCUMENT_INDEX_EMPTY")
        relative = structured_index_path(document_id, revision).as_posix()
        path = self.workspace.resolve(relative, must_exist=False)
        if self.workspace.relative(path) != relative:
            raise DocumentError("DOCUMENT_INDEX_PATH_INVALID")
        if not path.exists():
            atomic_write(path, content)
        record = PaperRecord(paper_id=document_id, file_name=Path(artifact.path).name, title=document.title,
                             num_pages=max_page, num_chunks=len(passages), sha256=artifact.sha256)
        return path, record, passages, revision

    def _load(self, revision: str) -> ParsedDocument | None:
        path = self._cache(f"parsed/{revision}.json")
        return ParsedDocument.model_validate_json(path.read_bytes()) if path.is_file() else None

    def cached_anchored_object(
        self, document_id: str, revision: str, source_sha256: str, object_id: str,
    ) -> tuple[DocumentArtifact, DocumentObject]:
        """Recheck the registered bytes and resolve one immutable local evidence anchor."""
        artifact, _ = self._resolve(None, document_id)
        if artifact.sha256 != source_sha256 or self._revision(artifact) != revision:
            raise DocumentError("DOCUMENT_REVISION_CHANGED")
        publication = self.publication_context(document_id)
        if publication is not None and publication.role == "supplement":
            parent, _ = self._resolve(None, publication.parent_document_id)
            if parent.sha256 != publication.parent_source_sha256:
                raise DocumentError("SUPPLEMENT_PARENT_CHANGED")
        parsed = self._load(revision)
        if parsed is None:
            raise DocumentError("DOCUMENT_NOT_PARSED")
        for obj in parsed.objects:
            if obj.object_id == object_id and obj.anchors and obj.text.strip() and obj.quality == "anchored":
                return artifact, obj
        raise DocumentError("DOCUMENT_ANCHOR_UNAVAILABLE")

    def _parse(self, revision: str, content: bytes, is_pdf: bool) -> ParsedDocument:
        import fcntl

        folder = self._cache("parse")
        folder.mkdir(parents=True, exist_ok=True)
        lock = self._cache(f"parse-jobs/{revision}.lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        with os.fdopen(os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600), "a+") as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if (cached := self._load(revision)) is not None:
                return cached
            job = self._cache(f"parse-jobs/{revision}.json")
            interrupted = job.is_file() and json.loads(job.read_bytes()).get("state") == "RUNNING"
            atomic_write(job, b'{"state":"RUNNING"}')
            try:
                with tempfile.TemporaryDirectory(dir=folder) as directory:
                    path = Path(directory) / ("source.pdf" if is_pdf else "source.html")
                    path.write_bytes(content)
                    parsed = self._parser(path)
                if interrupted:
                    parsed.diagnostics.append("DOCUMENT_PARSE_RESTARTED")
                if parsed._native_document is not None:
                    try:
                        native = json.dumps(parsed._native_document.export_to_dict(), ensure_ascii=False).encode()
                        if len(native) <= MAX_NATIVE_DOCUMENT_BYTES:
                            atomic_write(self._cache(f"native/{revision}.json"), native)
                            parsed.native_snapshot_available = True
                            parsed.native_snapshot_sha256 = _hash_bytes(native)
                        else:
                            parsed.diagnostics.append("DOCUMENT_NATIVE_SNAPSHOT_TOO_LARGE")
                    except (AttributeError, OSError, TypeError, ValueError):
                        parsed.diagnostics.append("DOCUMENT_NATIVE_SNAPSHOT_UNAVAILABLE")
                    parsed._native_document = None
                atomic_write(self._cache(f"parsed/{revision}.json"), parsed.model_dump_json().encode())
            except Exception:
                atomic_write(job, b'{"state":"FAILED"}')
                raise
            state = "PARTIAL" if _PARTIAL_DIAGNOSTICS.intersection(parsed.diagnostics) else "READY"
            atomic_write(job, json.dumps({"state": state, "restarted": interrupted}).encode())
            return parsed

    async def read(
        self, *, filename: str | None = None, document_id: str | None = None,
        section_id: str = "", object_ids: tuple[str, ...] = (), query: str = "",
        cursor: str = "", max_chars: int = 4000, cached_only: bool = False, expected_revision: str | None = None,
        index_query: Callable[[DocumentArtifact, str, str], Awaitable[IndexedDocumentQuery]] | None = None,
    ) -> dict[str, Any]:
        if sum(bool(selector) for selector in (section_id, object_ids, query)) > 1:
            raise DocumentError("DOCUMENT_SELECTORS_EXCLUSIVE")
        artifact, content = await asyncio.to_thread(self._resolve, filename, document_id)
        revision = self._revision(artifact)
        if expected_revision is not None and expected_revision != revision:
            raise DocumentError("DOCUMENT_REVISION_CHANGED")
        selection_scope: list[Any] = [2, artifact.document_id, revision, section_id, object_ids, query]
        selection = _hash_bytes(json.dumps(selection_scope).encode())
        if not 200 <= max_chars <= 20_000:
            raise DocumentError("DOCUMENT_BUDGET_INVALID")
        position, cell_index, offset = 0, 0, 0
        phase = "t"
        if cursor:
            parts = cursor.split(":")
            if (len(parts) != 5 or not re.fullmatch(r"[0-9a-f]{64}", parts[0])
                    or (not query and parts[0] != selection) or parts[2] not in {"t", "c"}
                    or any(not v.isascii() or not v.isdigit() or len(v) > 9
                           for v in (parts[1], parts[3], parts[4]))):
                raise DocumentError("DOCUMENT_CURSOR_INVALID_OR_STALE")
            position, phase, cell_index, offset = int(parts[1]), parts[2], int(parts[3]), int(parts[4])
        if not cached_only:
            await asyncio.to_thread(self._save, artifact)
        worker = None
        async with self._parse_lock:
            parsed = await asyncio.to_thread(self._load, revision)
            if parsed is None:
                if cached_only:
                    raise DocumentError("DOCUMENT_NOT_PARSED")
                worker = self._pending.get(revision)
                if worker is None:
                    if len(self._pending) >= 8:
                        raise DocumentError("DOCUMENT_PARSE_QUEUE_FULL")
                    worker = asyncio.create_task(self._parse_serially(
                        revision, content, artifact.media_type == "application/pdf",
                    ))
                    self._pending[revision] = worker
                    worker.add_done_callback(lambda completed: self._parse_finished(revision, completed))
        if worker is not None:
            try:
                parsed = await asyncio.wait_for(asyncio.shield(worker), timeout=self._wait_seconds)
            except TimeoutError:
                return {**artifact.model_dump(mode="json"), "revision": revision,
                        "parse_job_id": revision, "parse_state": "RUNNING",
                        "downloaded": bool(artifact.source_url), "parsed": False, "indexed": None,
                        "reranked": False, "coverage": "not_read", "objects": []}
        assert parsed is not None

        partial = bool(_PARTIAL_DIAGNOSTICS.intersection(parsed.diagnostics))
        publication = await asyncio.to_thread(self.publication_context, artifact.document_id)
        objects = parsed.objects
        route = "catalog"
        indexed: bool | None = None
        mode: str | None = None
        reranked = False
        query_diagnostics: tuple[str, ...] = ()
        matched: dict[str, StructuredPassageReference] = {}
        if section_id:
            objects = [obj for obj in objects if obj.section_id == section_id]
            route = "section"
        elif object_ids:
            objects = [obj for obj in objects if obj.object_id in object_ids]
            route = "objects"
        elif query:
            if index_query is not None:
                if partial or artifact.media_type != "application/pdf":
                    query_diagnostics = ("DOCUMENT_INDEX_COVERAGE_UNCONFIRMED",)
                else:
                    result = await index_query(artifact, revision, query)
                    indexed, query_diagnostics = result.indexed, result.degraded_reasons
                    by_id = {obj.object_id: obj for obj in objects}
                    for ref in result.references:
                        obj = by_id.get(ref.object_id)
                        if (ref.document_id == artifact.document_id and ref.revision == revision
                                and obj is not None and 0 <= ref.text_offset < len(obj.text)):
                            matched.setdefault(ref.object_id, ref)
                    if matched:
                        objects = [by_id[object_id] for object_id in matched]
                        route, mode, reranked = "document_index", result.mode, result.reranked
                    elif indexed and result.mode is not None:
                        query_diagnostics = (*query_diagnostics, "DOCUMENT_INDEX_NO_OBJECT_MATCH")
            if not matched:
                objects = [obj for obj in objects if query.casefold() in obj.text.casefold()]
                route = "document_text_match"
            selection_scope.extend([route, [(ref.object_id, ref.text_offset) for ref in matched.values()]])
            selection = _hash_bytes(json.dumps(selection_scope).encode())
        if cursor and (parts[0] != selection or position >= len(objects)):
            raise DocumentError("DOCUMENT_CURSOR_INVALID_OR_STALE")
        rows: list[dict[str, Any]] = []
        remaining = max_chars
        metadata_remaining = MAX_RESPONSE_OBJECT_METADATA_BYTES
        while position < len(objects) and len(rows) < MAX_RESPONSE_OBJECTS:
            obj = objects[position]
            row = obj.model_dump(mode="json", exclude={"text", "original_text", "table_cells", "anchors", "related", "relations"})
            if obj.parent_id is None:
                row.pop("parent_id", None)
            if obj.relations:
                row["relations"] = {kind: refs[:4] for kind, refs in obj.relations.items()}
            if not obj.table_continuations:
                row.pop("table_continuations", None)
            pages = [anchor["page"] for anchor in obj.anchors if isinstance(anchor.get("page"), int)]
            if pages:
                row.update(page_start=min(pages), page_end=max(pages))
            if matched_ref := matched.get(obj.object_id):
                row["matched_ref"] = matched_ref.model_dump(mode="json")
            for key, values in (("anchors", obj.anchors), ("related", obj.related)):
                visible: list[Any] = []
                for value in values[:8]:
                    if _json_bytes([*visible, value]) > 500:
                        break
                    visible.append(value)
                row[key] = visible
                if len(visible) < len(values):
                    row[f"{key}_omitted"] = len(values) - len(visible)
            metadata_size = _json_bytes(row)
            if rows and metadata_size > metadata_remaining:
                break
            metadata_remaining -= metadata_size
            include_text = route != "catalog" or obj.kind == "heading"
            cells = obj.table_cells if route != "catalog" and obj.kind == "table" else []
            if (phase == "t" and (cell_index or offset > len(obj.text) or (offset and not include_text))
                    or phase == "c" and (not cells or cell_index >= len(cells))):
                raise DocumentError("DOCUMENT_CURSOR_INVALID_OR_STALE")
            if phase == "t" and include_text and offset < len(obj.text):
                fragment = _text_fragment(obj.text, offset, remaining)
                if not fragment:
                    break
                row.update(text=fragment, text_start=offset, text_total_chars=len(obj.text))
                remaining -= _json_bytes(fragment)
                offset += len(fragment)
                if offset < len(obj.text):
                    rows.append(row)
                    break
            if phase == "t":
                phase, offset = "c", 0
            if cells:
                cell_start = cell_index
                complete_cells: list[dict[str, Any]] = []
                while cell_index < len(cells):
                    cell = cells[cell_index]
                    encoded = json.dumps(cell, ensure_ascii=False, separators=(",", ":"))
                    if offset >= len(encoded):
                        raise DocumentError("DOCUMENT_CURSOR_INVALID_OR_STALE")
                    size = len(encoded.encode()) + (1 if complete_cells else 2)
                    if not offset and size <= remaining:
                        complete_cells.append(cell)
                        remaining -= size
                        cell_index += 1
                        continue
                    if complete_cells or "text" in row or rows:
                        break
                    fragment = _text_fragment(encoded, offset, remaining)
                    if not fragment:
                        break
                    row["table_cell_fragment"] = {
                        "cell_index": cell_index, "start": offset, "total_chars": len(encoded), "json": fragment,
                    }
                    offset += len(fragment)
                    remaining -= _json_bytes(fragment)
                    if offset == len(encoded):
                        cell_index += 1
                        offset = 0
                    break
                if complete_cells:
                    row.update(table_cells=complete_cells, table_cell_start=cell_start)
                row["table_cell_count"] = len(cells)
                if cell_index < len(cells):
                    if "text" in row or complete_cells or "table_cell_fragment" in row:
                        rows.append(row)
                    break
            rows.append(row)
            position, phase, cell_index, offset = position + 1, "t", 0, 0
        return {
            **artifact.model_dump(mode="json"),
            **({"publication": publication.model_dump(mode="json", exclude={"supplement_urls"}),
                "publication_relations": await asyncio.to_thread(self.publication_relations,
                    publication.publication_id, publication.work_id)} if publication is not None else {}),
            "revision": revision, "parse_job_id": revision, "parse_state": "PARTIAL" if partial else "READY",
            "title": (parsed.title if parsed.title not in {"source", "source.pdf", "source.html"} else Path(artifact.path).stem)[:400],
            "downloaded": bool(artifact.source_url), "parsed": True, "indexed": indexed,
            "retrieval_route": route, "retrieval_mode": mode, "reranked": reranked,
            "coverage": "document_objects_partial" if partial else "document_objects",
            "source_page_count": parsed.source_page_count,
            "native_snapshot_available": parsed.native_snapshot_available and self._cache(f"native/{revision}.json").is_file(),
            "native_snapshot_sha256": parsed.native_snapshot_sha256,
            "degraded_reasons": [reason[:120] for reason in list(dict.fromkeys((*query_diagnostics, *parsed.diagnostics)))[:8]],
            "object_count": len(objects),
            "content_budget_utf8_bytes": max_chars,
            "truncated": position < len(objects),
            "cursor": f"{selection}:{position}:{phase}:{cell_index}:{offset}" if position < len(objects) else None,
            "objects": rows,
        }
