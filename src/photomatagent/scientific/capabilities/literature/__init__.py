"""Literature capability pack: arXiv search + local PDF search/read.

Namespace ``literature``, DEFERRED. Result counts and text lengths are hard
capped so a literature step can never flood model context.
"""

from __future__ import annotations

import importlib.metadata
import json
import re
from pathlib import Path
from typing import Any

from photomatagent.scientific.capabilities.base import (
    CapabilityPack,
    CapabilityStatus,
    ProbeResult,
)
from photomatagent.scientific.capabilities.config import ScientificConfig
from photomatagent.scientific.capabilities.contracts import (
    ScientificEvidence,
    ScientificToolResult,
)
from photomatagent.tools.base import Tool
from photomatagent.tools.exposure import ToolExposure
from photomatagent.workspace import Workspace


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return ""


class LiteratureProbe(CapabilityPack):
    name = "literature"
    description = (
        "Literature search and reading (arXiv + local PDFs) plus the local "
        "Literature RAG index (docling + LanceDB + hybrid retrieval)."
    )

    def probe(self) -> ProbeResult:
        missing = []
        try:
            import arxiv  # noqa: F401
        except ImportError:
            missing.append("arxiv")
        try:
            import pypdf  # noqa: F401
        except ImportError:
            missing.append("pypdf")
        try:
            import docling  # noqa: F401
        except ImportError:
            missing.append("docling")
        try:
            import lancedb  # noqa: F401
        except ImportError:
            missing.append("lancedb")
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            missing.append("sentence_transformers")
        if missing:
            return ProbeResult(
                status=CapabilityStatus.MISSING_DEPENDENCY,
                detail=(
                    f"missing: {', '.join(missing)} "
                    "(extra: photomatagent[literature])"
                ),
            )
        return ProbeResult(
            status=CapabilityStatus.AVAILABLE,
            detail="arxiv + pypdf + docling + lancedb + sentence-transformers available",
            version=(
                f"arxiv={_version('arxiv')}; pypdf={_version('pypdf')}; "
                f"docling={_version('docling')}; lancedb={_version('lancedb')}; "
                f"sentence-transformers={_version('sentence-transformers')}"
            ),
        )

    def tools(self) -> list[Tool]:
        return [
            LiteratureSearchArxivTool(self._config),
            LiteratureSearchLocalTool(self._config, self._workspace),
            LiteratureListPapersTool(self._workspace),
            LiteratureReadPaperTool(self._config, self._workspace),
            LiteratureIndexPapersTool(self._config, self._workspace),
            LiteratureSearchPassagesTool(self._config, self._workspace),
            LiteratureReadPassageTool(self._config, self._workspace),
            LiteratureExtractEvidenceTool(self._config, self._workspace),
        ]

    def __init__(self, config: ScientificConfig, workspace: Workspace) -> None:
        self._config = config
        self._workspace = workspace


def _papers_dir(workspace: Workspace) -> Path:
    candidates = [workspace.root / "papers", Path.cwd() / "papers"]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def _clean(text: str, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) > limit:
        cleaned = cleaned[:limit] + "...[truncated]"
    return cleaned


class LiteratureSearchArxivTool(Tool):
    name = "literature.search_arxiv"
    description = (
        "Search arXiv for recent papers; returns a strictly limited list of ids, "
        "titles, authors, dates, and short abstracts."
    )
    short_description = "Search arXiv papers by query (strictly limited results)."
    exposure = ToolExposure.DEFERRED
    namespace = "literature"
    source = "arxiv"
    tags = ("literature", "arxiv", "search", "infrared")
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "arXiv query, e.g. 'HgTe infrared photodetector'."},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
            "abstract_chars": {"type": "integer", "minimum": 0, "maximum": 1200},
        },
        "required": ["query"],
    }

    def __init__(self, config: ScientificConfig) -> None:
        self._config = config

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        import arxiv

        limit = min(
            int(arguments.get("max_results", 5)),
            self._config.literature_max_papers,
        )
        abstract_chars = int(arguments.get("abstract_chars", 400))
        client = arxiv.Client(page_size=min(limit, 20), delay_seconds=2, num_retries=1)
        try:
            search = arxiv.Search(
                query=str(arguments["query"]),
                max_results=limit,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            results = list(client.results(search))
        except Exception as exc:
            return ScientificToolResult(
                output=f"arxiv search failed: {type(exc).__name__}: {exc}",
                is_error=True,
                data={"error": type(exc).__name__},
            )
        cards = []
        evidence = []
        for paper in results[:limit]:
            authors = [author.name for author in paper.authors[:8]]
            card = {
                "arxiv_id": paper.get_short_id(),
                "title": paper.title,
                "authors": authors,
                "published": paper.published.date().isoformat() if paper.published else "",
                "abstract": _clean(paper.summary or "", abstract_chars),
            }
            cards.append(card)
            evidence.append(
                ScientificEvidence(
                    subject=paper.title,
                    property="literature_reference",
                    value=paper.get_short_id(),
                    unit="",
                    source="arXiv",
                    source_type="literature",
                    method="arxiv API relevance search",
                    summary=_clean(paper.summary or "", 200),
                    limitations="Title/abstract only; no full-text verification",
                    provenance={"arxiv_id": paper.get_short_id(), "tool": self.name},
                )
            )
        payload = {"count": len(cards), "results": cards}
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False),
            data={"results": cards},
            evidence=evidence,
        )


class LiteratureSearchLocalTool(Tool):
    name = "literature.search_local"
    description = (
        "Search text of PDF papers in the workspace papers/ directory; returns "
        "matching files with page-level snippets, strictly limited."
    )
    short_description = "Full-text search over local PDF papers (papers/)."
    exposure = ToolExposure.DEFERRED
    namespace = "literature"
    source = "pypdf"
    tags = ("literature", "pdf", "local search", "full text")
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_files": {"type": "integer", "minimum": 1, "maximum": 10},
            "snippet_chars": {"type": "integer", "minimum": 50, "maximum": 600},
        },
        "required": ["query"],
    }

    def __init__(self, config: ScientificConfig, workspace: Workspace) -> None:
        self._config = config
        self._workspace = workspace

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        from pypdf import PdfReader

        query = str(arguments["query"]).casefold()
        max_files = min(int(arguments.get("max_files", 3)), 10)
        snippet_chars = int(arguments.get("snippet_chars", 240))
        directory = _papers_dir(self._workspace)
        if not directory.is_dir():
            return ScientificToolResult(
                output=f"no papers/ directory found (looked at {directory})",
                is_error=True,
                data={"error": "no_papers_dir"},
            )
        matches = []
        for pdf_path in sorted(directory.glob("*.pdf"))[:max_files]:
            try:
                reader = PdfReader(str(pdf_path))
                pages = [page.extract_text() or "" for page in reader.pages]
            except Exception as exc:
                matches.append(
                    {"file": pdf_path.name, "error": f"{type(exc).__name__}: {exc}"}
                )
                continue
            snippets = []
            for index, page_text in enumerate(pages):
                lowered = page_text.casefold()
                position = lowered.find(query)
                if position < 0:
                    continue
                start = max(0, position - snippet_chars // 2)
                snippets.append(
                    {
                        "page": index + 1,
                        "snippet": _clean(page_text[start : start + snippet_chars], snippet_chars),
                    }
                )
                if len(snippets) >= 3:
                    break
            if snippets:
                matches.append({"file": pdf_path.name, "pages": snippets})
        payload = {"query": arguments["query"], "count": len(matches), "matches": matches}
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False),
            data={"matches": matches},
        )


class LiteratureListPapersTool(Tool):
    name = "literature.list_papers"
    description = "List PDF files available in the local papers/ directory."
    short_description = "List local PDF papers (papers/)."
    exposure = ToolExposure.DEFERRED
    namespace = "literature"
    source = "builtin"
    tags = ("literature", "pdf", "list")
    input_schema = {"type": "object", "properties": {}}

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        directory = _papers_dir(self._workspace)
        if not directory.is_dir():
            return ScientificToolResult(
                output=f"no papers/ directory found (looked at {directory})",
                is_error=True,
                data={"error": "no_papers_dir"},
            )
        files = [path.name for path in sorted(directory.glob("*.pdf"))]
        payload = {"count": len(files), "papers": files, "directory": str(directory)}
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False),
            data=payload,
        )


class LiteratureReadPaperTool(Tool):
    name = "literature.read_paper"
    description = (
        "Extract text from one local PDF (first N chars, capped) plus its page "
        "count and title metadata."
    )
    short_description = "Read a local PDF paper with a strict character cap."
    exposure = ToolExposure.DEFERRED
    namespace = "literature"
    source = "pypdf"
    tags = ("literature", "pdf", "read")
    input_schema = {
        "type": "object",
        "properties": {
            "file": {"type": "string", "description": "File name inside papers/."},
            "max_chars": {"type": "integer", "minimum": 200, "maximum": 20000},
        },
        "required": ["file"],
    }

    def __init__(self, config: ScientificConfig, workspace: Workspace) -> None:
        self._config = config
        self._workspace = workspace

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        from pypdf import PdfReader

        file_name = str(arguments["file"])
        max_chars = min(
            int(arguments.get("max_chars", 4000)),
            self._config.literature_max_chars,
        )
        directory = _papers_dir(self._workspace)
        pdf_path = (directory / file_name).resolve()
        if pdf_path.parent != directory.resolve() or not pdf_path.is_file():
            return ScientificToolResult(
                output=f"paper not found in {directory}: {file_name}",
                is_error=True,
                data={"error": "not_found"},
            )
        try:
            reader = PdfReader(str(pdf_path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            metadata = reader.metadata
        except Exception as exc:
            return ScientificToolResult(
                output=f"failed to read PDF: {type(exc).__name__}: {exc}",
                is_error=True,
                data={"error": type(exc).__name__},
            )
        title = str(metadata.get("/Title", "")).strip() if metadata else ""
        payload = {
            "file": file_name,
            "title": title,
            "pages": len(reader.pages),
            "chars": len(text),
            "text": _clean(text, max_chars),
        }
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False),
            data=payload,
            evidence=[
                ScientificEvidence(
                    subject=title or file_name,
                    property="literature_text",
                    value=_clean(text, 300),
                    unit="",
                    source=file_name,
                    source_type="literature",
                    method="pypdf text extraction",
                    summary=f"Read {len(reader.pages)} pages of {file_name}",
                    limitations="Extraction can mangle equations and figures",
                    provenance={"file": file_name, "tool": self.name},
                )
            ],
        )


def _resolve_literature_root(config: ScientificConfig, workspace: Workspace) -> Path:
    """Absolute literature PDF root: configured value or workspace-relative."""
    root = Path(config.literature_root)
    if not root.is_absolute():
        root = workspace.root / root
    return root


def _resolve_index_dir(config: ScientificConfig, workspace: Workspace) -> Path:
    index_dir = Path(config.literature_index_dir)
    if not index_dir.is_absolute():
        index_dir = workspace.root / index_dir
    return index_dir


class LiteratureIndexPapersTool(Tool):
    name = "literature.index_papers"
    description = (
        "Parse PDFs under a directory with docling, embed them, and build or "
        "incrementally update the local LanceDB index. Idempotent: unchanged "
        "PDFs are skipped by content hash. Returns indexed/skipped counts and "
        "the database location."
    )
    short_description = "Build/update the local literature RAG index from PDFs."
    exposure = ToolExposure.DEFERRED
    namespace = "literature"
    source = "lancedb"
    tags = ("literature", "rag", "index", "docling")
    input_schema = {
        "type": "object",
        "properties": {
            "directory": {
                "type": "string",
                "description": (
                    "PDF directory (searched recursively). Defaults to the "
                    "configured literature root (PHOTOMATAGENT_LITERATURE_DIR)."
                ),
            },
        },
    }

    def __init__(self, config: ScientificConfig, workspace: Workspace) -> None:
        self._config = config
        self._workspace = workspace

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        from photomatagent.scientific.capabilities.literature.index import (
            LiteratureIndex,
        )

        raw_directory = str(arguments.get("directory") or "")
        root = (
            Path(raw_directory)
            if raw_directory
            else _resolve_literature_root(self._config, self._workspace)
        )
        if not root.is_absolute():
            root = self._workspace.root / root
        index_dir = _resolve_index_dir(self._config, self._workspace)
        index = LiteratureIndex(
            index_dir, embedding_model=self._config.embedding_model
        )
        try:
            with index.locked():
                stats = index.index_directory(root)
        except FileNotFoundError as exc:
            return ScientificToolResult(
                output=str(exc),
                is_error=True,
                data={"error": "literature_root_not_found", "directory": str(root)},
            )
        payload = {
            "indexed": stats["indexed"],
            "skipped": stats["skipped"],
            "failed": stats["failed"],
            "removed": stats["removed"],
            "chunks": stats["chunks"],
            "db_location": stats["db_location"],
            "errors": stats["errors"][:5],
        }
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False),
            data=payload,
        )


class LiteratureSearchPassagesTool(Tool):
    name = "literature.search_passages"
    description = (
        "Hybrid (dense + keyword) search over the local literature index with "
        "reranking and context expansion. Returns strictly limited passages "
        "with provenance (paper, title, section, page, score, source file)."
    )
    short_description = "Hybrid RAG search for passages in local papers."
    exposure = ToolExposure.DEFERRED
    namespace = "literature"
    source = "lancedb"
    tags = ("literature", "rag", "search", "hybrid")
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Scientific query, e.g. 'HgTe quantum dot infrared detector responsivity'."},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
    }

    def __init__(self, config: ScientificConfig, workspace: Workspace) -> None:
        self._config = config
        self._workspace = workspace

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        from photomatagent.scientific.capabilities.literature.index import (
            LiteratureIndex,
        )
        from photomatagent.scientific.capabilities.literature.retrieval import (
            Retriever,
        )

        index = LiteratureIndex(
            _resolve_index_dir(self._config, self._workspace),
            embedding_model=self._config.embedding_model,
        )
        if index.count_passages() == 0:
            return ScientificToolResult(
                output=(
                    "literature index is empty; run literature.index_papers "
                    "first (or set PHOTOMATAGENT_LITERATURE_DIR)"
                ),
                is_error=True,
                data={"error": "empty_index", "db_location": str(index.index_dir)},
            )
        query = str(arguments["query"])
        top_k = min(
            int(arguments.get("top_k", self._config.literature_search_top_k)), 10
        )
        retriever = Retriever(index, reranker_model=self._config.reranker_model)
        try:
            results = retriever.hybrid_search(query, top_k=top_k)
        except Exception as exc:
            return ScientificToolResult(
                output=f"literature search failed: {type(exc).__name__}: {exc}",
                is_error=True,
                data={"error": type(exc).__name__},
            )
        rows = []
        for result in results:
            rows.append(
                {
                    "paper_id": result["paper_id"],
                    "title": result["title"],
                    "passage": _clean(
                        result["passage"], self._config.literature_passage_chars
                    ),
                    "section": result["section"],
                    "page": result["page"],
                    "score": round(float(result["score"]), 4),
                    "source": result["source"],
                    "passage_id": result["passage_id"],
                    "context_before": _clean(result.get("context_before", ""), 300),
                    "context_after": _clean(result.get("context_after", ""), 300),
                }
            )
        payload = {"query": query, "count": len(rows), "results": rows}
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False),
            data=payload,
        )


class LiteratureReadPassageTool(Tool):
    name = "literature.read_passage"
    description = (
        "Return one exact passage (full text + metadata) from the local "
        "literature index by passage_id."
    )
    short_description = "Read one indexed passage by passage_id."
    exposure = ToolExposure.DEFERRED
    namespace = "literature"
    source = "lancedb"
    tags = ("literature", "rag", "read")
    input_schema = {
        "type": "object",
        "properties": {
            "passage_id": {"type": "string", "description": "passage_id from literature.search_passages."},
        },
        "required": ["passage_id"],
    }

    def __init__(self, config: ScientificConfig, workspace: Workspace) -> None:
        self._config = config
        self._workspace = workspace

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        from photomatagent.scientific.capabilities.literature.index import (
            LiteratureIndex,
        )

        passage_id = str(arguments["passage_id"])
        index = LiteratureIndex(
            _resolve_index_dir(self._config, self._workspace),
            embedding_model=self._config.embedding_model,
        )
        row = index.get_passage(passage_id)
        if row is None:
            return ScientificToolResult(
                output=f"passage not found: {passage_id}",
                is_error=True,
                data={"error": "not_found", "passage_id": passage_id},
            )
        payload = {
            "passage_id": row["passage_id"],
            "paper_id": row["paper_id"],
            "title": row["title"],
            "authors": row.get("authors", []),
            "year": row.get("year"),
            "section": row.get("section", ""),
            "page": row.get("page"),
            "heading_path": row.get("heading_path", ""),
            "previous_chunk_id": row.get("previous_chunk_id", ""),
            "next_chunk_id": row.get("next_chunk_id", ""),
            "text": row.get("text", ""),
            "source": row.get("file_name", ""),
        }
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False),
            data=payload,
        )


class LiteratureExtractEvidenceTool(Tool):
    name = "literature.extract_evidence"
    description = (
        "Extract numerical scientific evidence (responsivity, detectivity, "
        "dark current, wavelength, temperature, bandgap, mobility, NETD) from "
        "passages. Each input item is either {'passage_id': ...} or "
        "{'text': ..., 'page': ...}. Never guesses: only explicit numbers "
        "with units are reported, as ScientificEvidence."
    )
    short_description = "Extract numbers + units as ScientificEvidence from passages."
    exposure = ToolExposure.DEFERRED
    namespace = "literature"
    source = "builtin"
    tags = ("literature", "evidence", "extraction")
    input_schema = {
        "type": "object",
        "properties": {
            "passages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "passage_id": {"type": "string"},
                        "text": {"type": "string"},
                        "page": {"type": "integer"},
                    },
                },
            },
        },
        "required": ["passages"],
    }

    def __init__(self, config: ScientificConfig, workspace: Workspace) -> None:
        self._config = config
        self._workspace = workspace

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        from photomatagent.scientific.capabilities.literature.evidence import (
            extract_evidence_from_passages,
        )
        from photomatagent.scientific.capabilities.literature.index import (
            LiteratureIndex,
        )

        passages = list(arguments.get("passages") or [])
        index = LiteratureIndex(
            _resolve_index_dir(self._config, self._workspace),
            embedding_model=self._config.embedding_model,
        )
        resolved: list[dict[str, Any]] = []
        for item in passages:
            if not isinstance(item, dict):
                continue
            passage_id = item.get("passage_id")
            if passage_id:
                row = index.get_passage(str(passage_id))
                if row is None:
                    resolved.append(
                        {
                            "text": "",
                            "error": f"passage not found: {passage_id}",
                        }
                    )
                    continue
                resolved.append(
                    {
                        "text": row.get("text", ""),
                        "page": row.get("page"),
                        "passage_id": passage_id,
                        "source": row.get("file_name", ""),
                    }
                )
            else:
                resolved.append(
                    {
                        "text": str(item.get("text") or ""),
                        "page": item.get("page"),
                        "source": str(item.get("source") or ""),
                    }
                )
        evidence = extract_evidence_from_passages(resolved)
        rows = [
            {
                "subject": item.subject,
                "property": item.property,
                "value": item.value,
                "unit": item.unit,
                "condition": item.provenance,
                "source": item.source,
                "method": item.method,
                "summary": item.summary,
            }
            for item in evidence
        ]
        payload = {"count": len(rows), "evidence": rows}
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False),
            data=payload,
            evidence=evidence,
        )


def literature_pack(config: ScientificConfig, workspace: Workspace) -> CapabilityPack:
    return LiteratureProbe(config, workspace)
