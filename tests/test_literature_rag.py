"""Literature RAG V1 tests: index, hybrid search, provenance, evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from photomatagent.scientific.capabilities.config import ScientificConfig
from photomatagent.scientific.capabilities.literature.evidence import (
    extract_evidence_from_text,
)
from photomatagent.scientific.capabilities.literature.index import LiteratureIndex
from photomatagent.scientific.capabilities.literature.retrieval import Retriever
from photomatagent.workspace import Workspace


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(path: Path, pages: list[str]) -> None:
    """Write a minimal, valid single/multi-page PDF with plain text lines."""
    objects: list[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = b" ".join(f"{3 + 2 * i} 0 R".encode() for i in range(len(pages)))
    objects.append(
        b"<< /Type /Pages /Kids [" + kids + b"] /Count "
        + str(len(pages)).encode()
        + b" >>"
    )
    for index, text in enumerate(pages):
        content_id = 3 + 2 * index + 1
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents "
            + str(content_id).encode()
            + b" 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        )
        stream = (
            b"BT /F1 11 Tf 72 720 Td ("
            + _pdf_escape(text).encode()
            + b") Tj ET"
        )
        objects.append(
            b"<< /Length "
            + str(len(stream)).encode()
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    buffer = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(buffer))
        buffer += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_position = len(buffer)
    buffer += f"xref\n0 {len(objects) + 1}\n".encode()
    buffer += b"0000000000 65535 f \n"
    for offset in offsets:
        buffer += f"{offset:010d} 00000 n \n".encode()
    buffer += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_position}\n%%EOF\n"
    ).encode()
    path.write_bytes(buffer)


def _index_two_papers(tmp_path: Path) -> tuple[Path, Path]:
    papers = tmp_path / "papers"
    papers.mkdir()
    make_pdf(
        papers / "2020_HgTe-quantum-dot-infrared-detector_aaa.pdf",
        [
            "HgTe quantum dot infrared detector responsivity of 0.82 A/W "
            "at 3.5 um and 80 K.",
            "The specific detectivity reached 1.2e10 Jones at 80 K.",
        ],
    )
    make_pdf(
        papers / "2019_PbS-colloidal-quantum-dot-photodetector_bbb.pdf",
        [
            "PbS colloidal quantum dot photodetector with a band gap of "
            "1.3 eV and an electron mobility of 500 cm2/Vs.",
        ],
    )
    index_dir = tmp_path / "index"
    return papers, index_dir


def test_index_small_pdf_folder(tmp_path):
    papers, index_dir = _index_two_papers(tmp_path)
    index = LiteratureIndex(index_dir)
    stats = index.index_directory(papers)

    assert stats["indexed"] == 2
    assert stats["failed"] == 0
    assert stats["chunks"] >= 2
    assert index.count_passages() >= 2
    assert index_dir.is_dir()


def test_incremental_index_skips_unchanged_pdfs(tmp_path):
    papers, index_dir = _index_two_papers(tmp_path)
    index = LiteratureIndex(index_dir)
    first = index.index_directory(papers)
    second = index.index_directory(papers)
    assert second["indexed"] == 0
    assert second["skipped"] == first["indexed"]


def test_search_known_topic_returns_provenance(tmp_path):
    papers, index_dir = _index_two_papers(tmp_path)
    index = LiteratureIndex(index_dir)
    index.index_directory(papers)
    retriever = Retriever(index)
    results = retriever.hybrid_search(
        "HgTe quantum dot infrared detector", top_k=3
    )

    assert results
    first = results[0]
    for key in (
        "passage_id",
        "paper_id",
        "title",
        "passage",
        "section",
        "page",
        "score",
        "source",
        "context_before",
        "context_after",
    ):
        assert key in first, f"missing provenance key: {key}"
    assert first["source"].endswith(".pdf")
    assert first["title"]
    # The HgTe paper must rank above the PbS paper for this query.
    assert "HgTe" in first["title"] or "HgTe" in first["passage"]


def test_evidence_extraction_known_sentence():
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


def test_evidence_extraction_detectivity_scientific_notation():
    evidence = extract_evidence_from_text(
        "A specific detectivity of 1.2×10^10 Jones was measured.",
        source="paper_y",
    )
    detectivity = [item for item in evidence if item.property == "detectivity"]
    assert detectivity
    item = detectivity[0]
    assert item.value == pytest.approx(1.2e10)
    assert item.unit == "Jones"


def test_evidence_extraction_detectivity_latex_spaced_notation():
    evidence = extract_evidence_from_text(
        "The largest QDIP detectivity value obtained (~ 10 10 cm Hz 1/2 /W) "
        "orders of magnitude below the desired value.",
        source="paper_w",
    )
    detectivity = [item for item in evidence if item.property == "detectivity"]
    assert detectivity
    assert detectivity[0].value == pytest.approx(1e10)
    assert detectivity[0].unit == "Jones"


def test_evidence_never_guesses_missing_numbers():
    evidence = extract_evidence_from_text(
        "The device showed improved performance under illumination.",
        source="paper_z",
    )
    assert evidence == []


def test_empty_index_search_is_graceful(tmp_path):
    index = LiteratureIndex(tmp_path / "index")
    retriever = Retriever(index)
    assert retriever.hybrid_search("anything", top_k=3) == []


async def test_tools_index_search_and_read_roundtrip(tmp_path):
    from photomatagent.scientific.capabilities.literature import (
        LiteratureIndexPapersTool,
        LiteratureReadPassageTool,
        LiteratureSearchPassagesTool,
    )

    papers, index_dir = _index_two_papers(tmp_path)
    workspace = Workspace(tmp_path)
    config = ScientificConfig(
        literature_root=str(papers),
        literature_index_dir=str(index_dir),
    )

    index_result = await LiteratureIndexPapersTool(config, workspace).execute({})
    assert not index_result.is_error
    assert index_result.data["indexed"] == 2

    search_result = await LiteratureSearchPassagesTool(config, workspace).execute(
        {"query": "HgTe quantum dot infrared detector", "top_k": 3}
    )
    assert not search_result.is_error
    assert search_result.data["count"] >= 1
    row = search_result.data["results"][0]
    assert row["passage_id"]
    assert row["title"]
    assert row["source"].endswith(".pdf")
    assert len(row["passage"]) <= 600

    read_result = await LiteratureReadPassageTool(config, workspace).execute(
        {"passage_id": row["passage_id"]}
    )
    assert not read_result.is_error
    assert read_result.data["passage_id"] == row["passage_id"]
    assert read_result.data["text"]


async def test_search_without_index_returns_guidance(tmp_path):
    from photomatagent.scientific.capabilities.literature import (
        LiteratureSearchPassagesTool,
    )

    workspace = Workspace(tmp_path)
    config = ScientificConfig(
        literature_root=str(tmp_path / "papers"),
        literature_index_dir=str(tmp_path / "index"),
    )
    result = await LiteratureSearchPassagesTool(config, workspace).execute(
        {"query": "anything"}
    )
    assert result.is_error
    assert "index_papers" in result.output
