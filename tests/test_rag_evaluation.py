"""Deterministic, in-process checks for the frozen literature RAG evaluation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from photomatagent.cli.app import app
from photomatagent.cli import rag as rag_cli


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "literature_rag_eval.json"


def _passage(passage_id: str, *, duplicate_text: str | None = None) -> dict[str, Any]:
    return {
        "passage_id": passage_id,
        "paper_id": f"paper-{passage_id}",
        "document_id": f"paper-{passage_id}",
        "document_revision": "a" * 64,
        "passage": duplicate_text or f"Synthetic licensed passage for {passage_id}.",
        "text": duplicate_text or f"Synthetic licensed passage for {passage_id}.",
        "source": f"tests/fixtures/{passage_id}.pdf",
        "relative_source_path": f"tests/fixtures/{passage_id}.pdf",
        "page": 1,
        "page_start": 1,
        "page_end": 1,
        "title": f"Synthetic {passage_id}",
    }


class _FixtureRetriever:
    def __init__(self, rows_by_query: dict[str, list[dict[str, Any]]]) -> None:
        self.rows_by_query = rows_by_query
        self.queries: list[str] = []

    async def search(self, query: str, *, workspace_id: str, top_k: int) -> Any:
        assert workspace_id == "fixture-workspace"
        assert top_k == 5
        self.queries.append(query)
        return SimpleNamespace(
            passages=tuple(self.rows_by_query.get(query, ())),
            diagnostics=SimpleNamespace(
                mode="fixture",
                candidate_count=len(self.rows_by_query.get(query, ())),
                reranked=False,
                degraded_reasons=(),
            ),
        )


def _fixture_rows() -> list[dict[str, Any]]:
    rows = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return [item for item in rows if isinstance(item, dict)]


def test_fixture_has_twenty_authored_judgments_and_required_fields() -> None:
    rows = _fixture_rows()
    assert len(rows) >= 20
    assert all(
        isinstance(item.get("query"), str)
        and item["query"].strip()
        and isinstance(item.get("relevant_passage_ids"), list)
        and isinstance(item.get("category"), str)
        for item in rows
    )


def test_fixture_evaluation_reports_quality_metrics_and_label() -> None:
    from photomatagent.cli.rag import evaluate_retrieval_fixture

    judgments = [
        {"query": "first", "relevant_passage_ids": ["p-first"], "category": "numeric"},
        {"query": "second", "relevant_passage_ids": ["p-second"], "category": "synonym"},
    ]
    retriever = _FixtureRetriever(
        {
            "first": [_passage("p-first")],
            "second": [_passage("p-second")],
        }
    )

    report = asyncio.run(
        evaluate_retrieval_fixture(
            retriever,
            judgments,
            workspace_id="fixture-workspace",
        )
    )

    assert report["fixture_specific"] is True
    assert report["corpus_wide_claim"] is False
    assert report["label"] == "fixture-specific"
    assert report["recall_at_5"] == pytest.approx(1.0)
    assert report["mrr_at_10"] == pytest.approx(1.0)
    assert report["no_result_rate"] == pytest.approx(0.0)
    assert report["duplicate_rate"] == pytest.approx(0.0)
    assert report["provenance_completeness"] == pytest.approx(1.0)
    assert report["passed"] is True
    assert retriever.queries == ["first", "second"]


def test_fixture_evaluation_fails_thresholds_for_duplicate_and_missing_provenance() -> None:
    from photomatagent.cli.rag import evaluate_retrieval_fixture

    judgments = [
        {"query": "bad", "relevant_passage_ids": ["p-bad"], "category": "negative"}
    ]
    duplicate = _passage("p-bad", duplicate_text="same")
    duplicate["passage_id"] = "p-bad-duplicate"
    duplicate["document_id"] = "paper-p-bad"
    duplicate["paper_id"] = "paper-p-bad"
    incomplete = _passage("p-bad")
    incomplete["text"] = "same"
    incomplete["passage"] = "same"
    incomplete.pop("source")
    incomplete.pop("relative_source_path")
    incomplete.pop("page")
    incomplete.pop("page_start")
    incomplete.pop("page_end")
    retriever = _FixtureRetriever({"bad": [incomplete, duplicate]})

    report = asyncio.run(
        evaluate_retrieval_fixture(
            retriever,
            judgments,
            workspace_id="fixture-workspace",
        )
    )

    assert report["duplicate_rate"] > 0.0
    assert report["provenance_completeness"] < 1.0
    assert report["passed"] is False
    assert report["thresholds"]["recall_at_5"]["passed"] is True


def test_rag_evaluate_cli_returns_nonzero_when_fixture_quality_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = CliRunner()
    monkeypatch.setattr(rag_cli, "_evaluation_fixture_path", lambda: FIXTURE_PATH)

    class _Services:
        workspace_id = "fixture-workspace"
        retriever = _FixtureRetriever({})

    monkeypatch.setattr(
        rag_cli,
        "build_literature_services",
        lambda config, workspace: _Services(),
    )
    result = runner.invoke(app, ["rag", "evaluate", "--workspace", str(tmp_path)])

    assert result.exit_code != 0
    assert "fixture-specific" in result.stdout
    assert "recall_at_5" in result.stdout
