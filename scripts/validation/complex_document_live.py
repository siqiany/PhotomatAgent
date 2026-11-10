"""Opt-in acceptance run against a real PDF, local Docling and isolated Qdrant."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
from uuid import uuid4

from qdrant_client import AsyncQdrantClient

from photomatagent.scientific.capabilities.config import ScientificConfig
from photomatagent.scientific.capabilities.literature import (
    LiteratureIndexDocumentTool,
    LiteratureReadObjectTool,
    LiteratureReadPaperTool,
    LiteratureSearchPassagesTool,
    build_literature_services,
)
from photomatagent.scientific.capabilities.literature.documents import DocumentService
from photomatagent.scientific.capabilities.literature.models import LITERATURE_CHUNK_SCHEMA_VERSION
from photomatagent.scientific.capabilities.literature.source_fetch import LiteratureFetchSourceTool
from photomatagent.workspace import Workspace


async def verify(args: argparse.Namespace) -> None:
    args.workspace.mkdir(parents=True, exist_ok=True)
    workspace = Workspace(args.workspace)
    documents = DocumentService(workspace, wait_seconds=10)
    prefix = f"photomat_complex_acceptance_{uuid4().hex[:12]}"
    config = ScientificConfig(qdrant_url=args.qdrant, qdrant_collection_prefix=prefix,
                              reranker_provider=args.reranker_provider, reranker_model=args.reranker_model)
    generation = None
    services = None
    try:
        if args.document_id:
            document_id = args.document_id
        else:
            fetched = await LiteratureFetchSourceTool(documents=documents).execute(
                {"url": args.url, "content_role": "paper"},
            )
            if fetched.is_error or not fetched.data.get("downloaded"):
                raise RuntimeError(f"PDF download failed: {fetched.data.get('diagnostic')}")
            document_id = fetched.data["document_id"]
        print(json.dumps({"document_id": document_id, "qdrant_prefix": prefix}), flush=True)
        while True:
            page = await documents.read(document_id=document_id)
            if page["parse_state"] != "RUNNING":
                break
        revision = page["revision"]
        print(json.dumps({key: page[key] for key in (
            "revision", "sha256", "source_page_count", "parse_state", "degraded_reasons",
        )}), flush=True)
        if page["parse_state"] == "PARTIAL":
            if not args.expect_partial:
                raise RuntimeError("Document parsed partially")
            rejected = await LiteratureIndexDocumentTool(config, workspace, documents).execute({"document_id": document_id})
            if not rejected.is_error or "DOCUMENT_INDEX_COVERAGE_UNCONFIRMED" not in rejected.output:
                raise AssertionError("Partial PDF entered the full-text index")
            print(json.dumps({"partial_index_blocked": True}), flush=True)
            return
        if args.expect_partial:
            raise AssertionError("Expected a partial conversion")
        parsed = documents._load(revision)
        assert parsed is not None
        indexed_objects = {obj.object_id: obj for obj in parsed.objects}
        kinds = Counter(obj.kind for obj in parsed.objects)
        tables = [obj for obj in parsed.objects if obj.kind == "table" and obj.table_cells]
        continuations = [(obj.object_id, ref, basis) for obj in tables for ref, basis in obj.table_continuations.items()
                         if ref in indexed_objects and indexed_objects[ref].kind == "table"]
        assert tables and kinds["equation"] and kinds["caption"]
        if args.require_continuation:
            assert continuations, "Expected an explicitly linked cross-page table"
        reader = LiteratureReadObjectTool(config, workspace, documents)
        for equation in (obj for obj in parsed.objects if obj.kind == "equation" and obj.anchors):
            rendered = await reader.execute({"document_id": document_id, "revision": revision,
                                             "object_ids": [equation.object_id], "include_images": True})
            if rendered.images:
                assert rendered.data["crops"][0]["source_sha256"] == page["sha256"]
                break
        else:
            raise AssertionError("No anchored formula crop was readable")
        figure = next(obj for obj in parsed.objects if obj.kind == "figure" and any(
            ref in indexed_objects and indexed_objects[ref].kind == "caption" for ref in obj.related
        ))
        caption = next(indexed_objects[ref] for ref in figure.related
                       if ref in indexed_objects and indexed_objects[ref].kind == "caption")
        table = max(tables, key=lambda obj: len(obj.table_cells))
        selection = {"document_id": document_id, "revision": revision, "object_ids": [table.object_id], "max_chars": 1200}
        cursor = ""
        visited: set[str] = set()
        full_cells = 0
        while True:
            response = await reader.execute({**selection, "cursor": cursor})
            assert not response.is_error, response.output
            full_cells += sum(len(obj.get("table_cells", [])) for obj in response.data["objects"])
            cursor = response.data["cursor"]
            if not cursor:
                break
            assert cursor not in visited
            visited.add(cursor)
        assert full_cells == len(table.table_cells)
        print(json.dumps({"object_counts": kinds, "table_cells_read": full_cells,
                          "continuation_links": continuations, "equation_page": rendered.data["crops"][0]["page"],
                          "equation_text_available": bool(equation.text), "caption_object": caption.object_id}), flush=True)

        services = build_literature_services(config, workspace)
        identity = services.ingestion.embedder.identity
        generation = services.store.expected_generation(identity=identity, chunk_schema_version=LITERATURE_CHUNK_SCHEMA_VERSION)
        await services.store.ensure_generation(identity=identity, chunk_schema_version=LITERATURE_CHUNK_SCHEMA_VERSION)
        indexed = await LiteratureIndexDocumentTool(config, workspace, documents, services).execute({"document_id": document_id})
        assert not indexed.is_error and indexed.data["indexed"] == 1 and not indexed.data["search_visible"], indexed.data
        await services.store.activate_generation(generation)
        search = LiteratureSearchPassagesTool(config, workspace, services)
        searches = [(table, args.table_query or table.text.splitlines()[0]), (caption, caption.text[:130])]
        if args.continuation_query:
            target_id = next(iter(table.table_continuations))
            searches.append((indexed_objects[target_id], args.continuation_query))
        for wanted, query in searches:
            results = await search.execute({"query": query, "top_k": 10})
            assert not results.is_error, results.output
            diagnostics = results.data["diagnostics"]
            print(json.dumps({"reranker_provider": config.reranker_provider, "reranker_model": config.reranker_model,
                              "retrieval_mode": diagnostics["mode"], "reranked": diagnostics["reranked"],
                              "degraded_reasons": diagnostics["degraded_reasons"]}), flush=True)
            if config.reranker_provider == "local" and "reranker_unavailable" in diagnostics["degraded_reasons"]:
                raise RuntimeError("RERANK_ACCEPTANCE_BLOCKED: local reranker unavailable")
            assert diagnostics["mode"] == "hybrid_rrf" and not diagnostics["degraded_reasons"], diagnostics
            assert diagnostics["reranked"] is (config.reranker_provider == "local"), diagnostics
            row = next((row for row in results.data["results"] if row.get("structured_ref", {}).get("object_id") == wanted.object_id), None)
            assert row is not None, (wanted.object_id, [r.get("structured_ref") for r in results.data["results"]])
            ref = row["structured_ref"]
            assert (ref["document_id"], ref["revision"]) == (document_id, revision)
            reread = await reader.execute({key: selection[key] for key in ("document_id", "revision")} |
                                          {"object_ids": [ref["object_id"]]})
            assert not reread.is_error and reread.data["objects"][0]["object_id"] == ref["object_id"]
            print(json.dumps({"retrieval_mode": results.data["diagnostics"]["mode"], "matched_kind": wanted.kind,
                              "object_id": ref["object_id"], "offset": ref["text_offset"],
                              "page": reread.data["objects"][0].get("page_start")}), flush=True)
        paper = LiteratureReadPaperTool(config, workspace, documents, services)
        selected = await paper.execute({"document_id": document_id, "revision": revision, "query": searches[0][1]})
        assert not selected.is_error and selected.data["retrieval_route"] == "document_index", selected.data
        assert selected.data["indexed"] is True and selected.data["retrieval_mode"] == "hybrid_rrf"
        assert selected.data["reranked"] is (config.reranker_provider == "local")
        assert set(selected.data["degraded_reasons"]) <= set(page["degraded_reasons"]), selected.data
        refs = [obj["matched_ref"] for obj in selected.data["objects"]]
        assert refs and all(ref["document_id"] == document_id and ref["revision"] == revision for ref in refs)
        print(json.dumps({"route": selected.data["retrieval_route"], "mode": selected.data["retrieval_mode"],
                          "reranked": selected.data["reranked"], "degraded_reasons": selected.data["degraded_reasons"],
                          "object_references": refs}), flush=True)
    finally:
        await documents.aclose()
        if services is not None:
            await services.store.aclose()
        if generation is not None:
            client = AsyncQdrantClient(url=args.qdrant, timeout=30)
            try:
                for name in (generation.documents_physical, generation.passages_physical):
                    if name.startswith(prefix + "_") and await client.collection_exists(name):
                        await client.delete_collection(name)
            finally:
                await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url")
    source.add_argument("--document-id")
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--qdrant", default="http://127.0.0.1:6333")
    parser.add_argument("--table-query")
    parser.add_argument("--continuation-query")
    parser.add_argument("--require-continuation", action="store_true")
    parser.add_argument("--expect-partial", action="store_true")
    parser.add_argument("--reranker-provider", choices=("disabled", "local"), default="disabled")
    parser.add_argument("--reranker-model", default=ScientificConfig().reranker_model)
    asyncio.run(verify(parser.parse_args()))
