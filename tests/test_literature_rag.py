"""Public literature-tool contracts over injected Qdrant application services."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from photomatagent.scientific.capabilities.config import ScientificConfig
from photomatagent.scientific.capabilities import literature as literature_capability
from photomatagent.scientific.capabilities.literature import (
    LiteratureExtractEvidenceTool,
    LiteratureIndexPapersTool,
    LiteratureReadPassageTool,
    LiteratureSearchPassagesTool,
    build_literature_services,
)
from photomatagent.scientific.capabilities.literature.evidence import (
    extract_evidence_from_text,
)
from photomatagent.scientific.capabilities.literature.qdrant_store import (
    QdrantStoreError,
)
from photomatagent.scientific.capabilities.literature.retrieval import (
    RetrievedPassage,
    RetrievalDiagnostics,
    RetrievalResult,
)
from photomatagent.scientific.capabilities.literature import retrieval as retrieval_module
from photomatagent.tools.exposure import ToolExposure
from photomatagent.workspace import Workspace


CONFIG = ScientificConfig()


def test_literature_probe_uses_source_aware_generation_version() -> None:
    assert literature_capability._CHUNK_SCHEMA_VERSION == 2


def test_build_literature_services_passes_source_aware_generation_version(
    tmp_path, monkeypatch
) -> None:
    seen: list[int | None] = []

    class CapturingRetriever:
        def __init__(self, store, embedder, reranker, **kwargs: Any) -> None:
            del store, embedder, reranker
            seen.append(kwargs.get("chunk_schema_version"))

    monkeypatch.setattr(retrieval_module, "LiteratureRetriever", CapturingRetriever)
    build_literature_services(
        CONFIG,
        Workspace(tmp_path),
        store=object(),
        embedder=object(),
        reranker=object(),
    )

    assert seen == [2]


class FakeIngestion:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.plans: list[tuple[Any, Any]] = []
        self.batches: list[dict[str, Any]] = []

    async def plan(self, root: Any, workspace: Any) -> object:
        if self.error is not None:
            raise self.error
        self.plans.append((root, workspace))
        return "fake-plan"

    async def index_batch(self, plan: object, **kwargs: Any) -> SimpleNamespace:
        self.batches.append({"plan": plan, **kwargs})
        return SimpleNamespace(
            run_id=kwargs.get("run_id") or "run-fake",
            discovered=2,
            unchanged=0,
            indexed=2,
            failed=0,
            deleted=0,
            chunks=3,
            staged_cleanup=0,
            next_cursor=None,
            complete=True,
            retryable=False,
            errors=(),
        )


class FakeStore:
    async def retrieve_passages(self, workspace_id: str, passage_ids: list[str]) -> list[Any]:
        del workspace_id
        return [
            SimpleNamespace(
                passage_id=passage_id,
                document_id="paper-1",
                document_revision="a" * 64,
                workspace_id="workspace",
                text="The detector achieved a responsivity of 0.82 A/W at 80 K.",
                title="HgTe quantum dot infrared detector",
                authors=("A. Author",),
                year=2024,
                section="Results",
                heading_path="Results",
                page_start=3,
                page_end=3,
                relative_source_path="papers/hgte.pdf",
                previous_passage_id=None,
                next_passage_id=None,
                limitations=(),
            )
            for passage_id in passage_ids
        ]


class FakeRetriever:
    async def search(
        self,
        query: str,
        *,
        workspace_id: str,
        top_k: int,
    ) -> RetrievalResult:
        del workspace_id
        passage = RetrievedPassage(
            passage_id="passage-1",
            workspace_id="workspace",
            document_id="paper-1",
            document_revision="a" * 64,
            text=(
                "HgTe quantum dot infrared detector responsivity is 0.82 A/W "
                "at 3.5 um and 80 K. "
                + "long text " * 100
            ),
            score=0.987654,
            title="HgTe quantum dot infrared detector",
            section="Results",
            page_start=2,
            page_end=2,
            relative_source_path="papers/hgte.pdf",
        )
        return RetrievalResult(
            passages=(passage,) if top_k else (),
            diagnostics=RetrievalDiagnostics(
                mode="hybrid_rrf",
                candidate_count=1,
                reranked=False,
            ),
        )


@pytest.fixture
def services() -> SimpleNamespace:
    return SimpleNamespace(
        ingestion=FakeIngestion(),
        retriever=FakeRetriever(),
        store=FakeStore(),
        workspace_id="workspace",
    )


@pytest.mark.asyncio
async def test_tools_index_search_and_read_keep_public_contract(
    tmp_path, services: SimpleNamespace
) -> None:
    papers = tmp_path / "papers"
    papers.mkdir()
    config = ScientificConfig(literature_root="papers")
    workspace = Workspace(tmp_path)

    index_result = await LiteratureIndexPapersTool(config, workspace, services).execute(
        {"max_documents": 2}
    )
    assert index_result.data["run_id"]
    assert index_result.data["complete"] is True

    search_result = await LiteratureSearchPassagesTool(config, workspace, services).execute(
        {"query": "HgTe quantum dot infrared detector", "top_k": 3}
    )
    assert not search_result.is_error
    row = search_result.data["results"][0]
    assert {
        "passage_id",
        "paper_id",
        "title",
        "passage",
        "section",
        "page",
        "score",
        "source",
    } <= row.keys()
    assert len(row["passage"]) <= 600
    assert row["source"] == "papers/hgte.pdf"
    assert "diagnostics" in search_result.data
    assert "payload" not in row

    read_result = await LiteratureReadPassageTool(config, workspace, services).execute(
        {"passage_id": row["passage_id"]}
    )
    assert not read_result.is_error
    assert read_result.data["passage_id"] == row["passage_id"]
    assert read_result.data["text"]
    assert "payload" not in read_result.data


def test_qdrant_tools_are_deferred_and_index_is_expensive() -> None:
    workspace = Workspace(".")
    tools = [
        LiteratureIndexPapersTool(CONFIG, workspace),
        LiteratureSearchPassagesTool(CONFIG, workspace),
        LiteratureReadPassageTool(CONFIG, workspace),
        LiteratureExtractEvidenceTool(CONFIG, workspace),
    ]
    assert all(tool.exposure is ToolExposure.DEFERRED for tool in tools)
    assert tools[0].cost_class == "EXPENSIVE"
    assert all("qdrant" not in str(tool.input_schema).casefold() for tool in tools)


@pytest.mark.asyncio
async def test_missing_qdrant_keeps_typed_error_code(tmp_path) -> None:
    services = SimpleNamespace(
        ingestion=FakeIngestion(
            error=QdrantStoreError("qdrant_unreachable", "Qdrant is unavailable")
        ),
        retriever=FakeRetriever(),
        store=FakeStore(),
        workspace_id="workspace",
    )
    result = await LiteratureIndexPapersTool(
        ScientificConfig(), Workspace(tmp_path), services
    ).execute({})
    assert result.is_error
    assert result.data["error"] == "qdrant_unreachable"
    assert "qdrant_unreachable" in result.output


@pytest.mark.asyncio
async def test_extract_evidence_resolves_exact_passage(tmp_path, services) -> None:
    result = await LiteratureExtractEvidenceTool(
        ScientificConfig(), Workspace(tmp_path), services
    ).execute({"passages": [{"passage_id": "passage-1"}]})
    assert not result.is_error
    assert result.data["count"] == 1
    assert result.data["evidence"][0]["property"] == "responsivity"


def test_evidence_extraction_known_sentence() -> None:
    evidence = extract_evidence_from_text(
        "The detector achieved a responsivity of 0.82 A/W at 3.5 μm and 80 K.",
        source="paper_x",
    )
    responsivity = [item for item in evidence if item.property == "responsivity"]
    assert responsivity
    item = responsivity[0]
    assert item.value == pytest.approx(0.82)
    assert item.unit == "A/W"
    assert item.source == "paper_x"
    assert item.source_type == "literature"
    assert item.method == "reported experimental value"
    assert item.provenance["wavelength_um"] == pytest.approx(3.5)
    assert item.provenance["temperature_K"] == pytest.approx(80)


def test_evidence_never_guesses_missing_numbers() -> None:
    evidence = extract_evidence_from_text(
        "The device showed improved performance under illumination.",
        source="paper_z",
    )
    assert evidence == []


@pytest.mark.asyncio
async def test_extract_evidence_bounds_input_and_state_updates(tmp_path, services) -> None:
    tool = LiteratureExtractEvidenceTool(
        ScientificConfig(), Workspace(tmp_path), services
    )
    passages = [
        {
            "text": "The responsivity was 0.82 A/W at 80 K and 3.5 um. "
            * 20,
            "page": index + 1,
        }
        for index in range(150)
    ]

    result = await tool.execute({"passages": passages})

    assert not result.is_error
    assert len(result.data["evidence"]) <= 100
    assert len(result.evidence) <= 100
    assert len(result.state_updates) <= 100
    assert tool.input_schema["properties"]["passages"]["maxItems"] == 100
