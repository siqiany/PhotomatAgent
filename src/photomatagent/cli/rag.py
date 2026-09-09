"""Direct, user-controlled operations for the literature Qdrant RAG services."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from dataclasses import replace as dataclass_replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import typer
from rich.console import Console
from rich.table import Table

from photomatagent.scientific.capabilities.config import ScientificConfig
from photomatagent.scientific.capabilities.literature import (
    LiteratureProbe,
    LiteratureReadPassageTool,
    LiteratureSearchPassagesTool,
    build_literature_services,
)
from photomatagent.scientific.capabilities.literature.models import (
    LITERATURE_CHUNK_SCHEMA_VERSION,
)
from photomatagent.workspace import Workspace
from photomatagent.scientific.capabilities.literature.qdrant_store import (
    sanitize_qdrant_url,
)


rag_app = typer.Typer(
    help="Plan, index, search, evaluate, and snapshot the literature RAG store.",
    no_args_is_help=False,
)
console = Console()


def _config(workspace: Path) -> ScientificConfig:
    return ScientificConfig.from_environment(workspace=workspace)


def _workspace(path: Path) -> Workspace:
    return Workspace(path)


def _error_code(exc: BaseException) -> str:
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        return code
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return "qdrant_unreachable"
    return type(exc).__name__.casefold() or "rag_error"


def _safe_message(exc: BaseException) -> str:
    message = str(getattr(exc, "message", "") or str(exc))
    message = re.sub(
        r"(?i)(api[_ -]?key|access[_ -]?token|authorization|bearer|password|secret)"
        r"\s*[:=]\s*[^\s,;]+",
        r"\1=[redacted]",
        message,
    )
    return message[:320].replace("\n", " ")


def _print_error(exc: BaseException) -> None:
    console.print(f"[red]{_error_code(exc)}: {_safe_message(exc)}[/]")


def _external_configured(config: ScientificConfig) -> bool:
    return config.embedding_provider != "local" or config.reranker_provider not in {
        "local",
        "disabled",
    }


def _confirm_external(config: ScientificConfig, directory: Path, *, yes: bool) -> None:
    if not _external_configured(config):
        return
    provider_parts = [
        f"embedding={config.embedding_provider}/{config.embedding_model}",
        f"reranker={config.reranker_provider}/{config.reranker_model}",
    ]
    console.print("外部 RAG provider 配置：" + ", ".join(provider_parts))
    console.print(f"数据范围：{directory}")
    console.print("警告：将发送全文片段到外部服务。")
    if yes:
        return
    # ``typer.confirm(..., abort=True)`` gives non-zero exit status on a
    # negative answer and keeps the confirmation explicit in non-interactive
    # invocations as well.
    typer.confirm("确认继续？", abort=True)


def _stats_dict(stats: Any) -> dict[str, Any]:
    def get(name: str, default: Any = None) -> Any:
        if isinstance(stats, dict):
            return stats.get(name, default)
        return getattr(stats, name, default)

    return {
        "run_id": str(get("run_id", "")),
        "discovered": int(get("discovered", 0) or 0),
        "unchanged": int(get("unchanged", 0) or 0),
        "indexed": int(get("indexed", 0) or 0),
        "failed": int(get("failed", 0) or 0),
        "deleted": int(get("deleted", 0) or 0),
        "chunks": int(get("chunks", 0) or 0),
        "staged_cleanup": int(get("staged_cleanup", 0) or 0),
        "next_cursor": get("next_cursor"),
        "complete": bool(get("complete", False)),
        "retryable": bool(get("retryable", False)),
        "errors": [str(item)[:320] for item in list(get("errors", ()) or ())[:20]],
    }


def _service_value(services: Any, name: str, default: Any = None) -> Any:
    if isinstance(services, dict):
        return services.get(name, default)
    return getattr(services, name, default)


def _evaluation_fixture_path() -> Path:
    """Locate the repository-owned, synthetic evaluation judgments.

    The fixture is deliberately kept outside the installed package: it is an
    authored quality gate, not application data and never contains copied
    paper text.  An environment override is useful for CI packaging checks,
    while the repository path remains the default for normal development.
    """
    configured = os.environ.get("PHOTOMATAGENT_RAG_EVAL_FIXTURE", "").strip()
    candidates = [
        Path(configured) if configured else None,
        Path("tests/fixtures/literature_rag_eval.json"),
        Path(__file__).resolve().parents[3]
        / "tests"
        / "fixtures"
        / "literature_rag_eval.json",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "frozen RAG evaluation fixture is unavailable; expected "
        "tests/fixtures/literature_rag_eval.json"
    )


def _load_evaluation_fixture(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("frozen RAG evaluation fixture is not valid JSON") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError("frozen RAG evaluation fixture must be a non-empty array")
    judgments: list[dict[str, Any]] = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"frozen RAG evaluation row {index} is not an object")
        query = raw.get("query")
        relevant = raw.get("relevant_passage_ids")
        category = raw.get("category")
        if (
            not isinstance(query, str)
            or not query.strip()
            or not isinstance(relevant, list)
            or not all(isinstance(item, str) and item for item in relevant)
            or not isinstance(category, str)
            or not category.strip()
            or not isinstance(raw.get("fixture_author"), str)
            or not raw["fixture_author"].strip()
            or not isinstance(raw.get("license"), str)
            or not raw["license"].strip()
        ):
            raise ValueError(f"frozen RAG evaluation row {index} is malformed")
        judgments.append(
            {
                "query": query,
                "relevant_passage_ids": list(dict.fromkeys(relevant)),
                "category": category,
                "fixture_author": raw["fixture_author"],
                "license": raw["license"],
            }
        )
    return judgments


def _evaluation_value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _evaluation_passage_id(item: Any) -> str:
    return str(_evaluation_value(item, "passage_id", "") or "")


def _evaluation_text(item: Any) -> str:
    value = _evaluation_value(item, "text", None)
    if value is None:
        value = _evaluation_value(item, "passage", "")
    return str(value or "")


def _evaluation_provenance_complete(item: Any) -> bool:
    passage_id = _evaluation_passage_id(item)
    document_id = _evaluation_value(item, "document_id", None) or _evaluation_value(
        item, "paper_id", None
    )
    source = _evaluation_value(item, "relative_source_path", None) or _evaluation_value(
        item, "source", None
    )
    page = _evaluation_value(item, "page", None)
    if page is None:
        page = _evaluation_value(item, "page_start", None)
    return bool(
        passage_id.strip()
        and str(document_id or "").strip()
        and _evaluation_text(item).strip()
        and str(source or "").strip()
        and page is not None
    )


def _evaluation_duplicate_key(item: Any) -> tuple[str, str] | None:
    document_id = _evaluation_value(item, "document_id", None) or _evaluation_value(
        item, "paper_id", None
    )
    text = " ".join(_evaluation_text(item).split()).casefold()
    if not str(document_id or "").strip() or not text:
        return None
    return str(document_id), text


def _fixture_texts(judgments: list[dict[str, Any]]) -> dict[str, str]:
    """Create deterministic synthetic passages without copying source prose."""
    query_anchors: dict[str, list[str]] = {}
    for judgment in judgments:
        query = str(judgment.get("query", "")).strip()
        for raw_id in judgment.get("relevant_passage_ids", ()):
            passage_id = str(raw_id).strip()
            if passage_id and query:
                query_anchors.setdefault(passage_id, []).append(query)
    return {
        passage_id: (
            f"Synthetic fixture passage {passage_id}. "
            "This authored CC0 test text is used only for retrieval wiring; "
            "the identifier is the stable semantic anchor. "
            f"Judgment anchors: {'; '.join(anchors)}"
        )
        for passage_id, anchors in sorted(query_anchors.items())
    }


def _fixture_point_ids(
    judgments: list[dict[str, Any]], workspace_id: str
) -> dict[str, str]:
    """Map authored fixture labels to stable UUID point IDs accepted by Qdrant.

    The fixture labels are intentionally readable (for example,
    ``fixture-hgte-performance``), while Qdrant's point-ID contract accepts
    integers or UUID strings.  Keep the authored labels in the judgment file
    and payload text, but use the same deterministic production ID derivation
    for the physical point IDs and the retrieval assertions.
    """
    from photomatagent.scientific.capabilities.literature.qdrant_store import (
        document_id_for,
        passage_id_for,
    )

    result: dict[str, str] = {}
    for index, (passage_id, text_value) in enumerate(_fixture_texts(judgments).items()):
        relative_path = f"synthetic/{passage_id}.txt"
        document_id = document_id_for(workspace_id, relative_path)
        revision = hashlib.sha256(text_value.encode("utf-8")).hexdigest()
        result[passage_id] = passage_id_for(document_id, revision, index)
    return result


async def _close_isolated_store(store: Any, prefix: str) -> None:
    client = getattr(store, "_client", None)
    if client is None:
        return
    try:
        get_collections = getattr(client, "get_collections", None)
        delete_collection = getattr(client, "delete_collection", None)
        if get_collections is not None and delete_collection is not None:
            result = get_collections()
            if hasattr(result, "__await__"):
                result = await result
            for descriptor in getattr(result, "collections", ()):
                name = str(getattr(descriptor, "name", ""))
                if name.startswith(prefix + "_"):
                    removed = delete_collection(name)
                    if hasattr(removed, "__await__"):
                        await removed
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result


async def evaluate_live_fixture(
    config: ScientificConfig,
    boundary: Workspace,
    judgments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Seed an isolated fixture and evaluate through the real retriever path."""
    from photomatagent.scientific.capabilities.literature.models import (
        DocumentManifest,
        DocumentStatus,
        IngestState,
        PassagePoint,
    )
    from photomatagent.scientific.capabilities.literature.providers.base import (
        validate_vectors,
    )
    from photomatagent.scientific.capabilities.literature.qdrant_store import (
        document_id_for,
    )
    from photomatagent.scientific.capabilities.literature.retrieval import LiteratureRetriever

    prefix = f"photomat_test_eval_{uuid4().hex}"
    isolated_config = dataclass_replace(
        config,
        qdrant_collection_prefix=prefix,
    )
    services: Any | None = None
    try:
        services = build_literature_services(isolated_config, boundary)
        store = _service_value(services, "store")
        ingestion_service = _service_value(services, "ingestion")
        retriever = _service_value(services, "retriever")
        embedder = _service_value(ingestion_service, "embedder")
        workspace_id = str(_service_value(services, "workspace_id", "") or "")
        if not isinstance(retriever, LiteratureRetriever):
            raise RuntimeError("live LiteratureRetriever service is unavailable")
        if store is None or embedder is None or not workspace_id:
            raise RuntimeError("live literature services are unavailable")

        ensure = getattr(store, "ensure_generation")
        generation = ensure(
            identity=embedder.identity,
            chunk_schema_version=LITERATURE_CHUNK_SCHEMA_VERSION,
        )
        if hasattr(generation, "__await__"):
            generation = await generation
        select_staging = getattr(store, "select_staging_generation", None)
        if callable(select_staging):
            select_staging(generation)

        texts = _fixture_texts(judgments)
        if not texts:
            raise RuntimeError("fixture has no relevant synthetic passages")
        fixture_point_ids = _fixture_point_ids(judgments, workspace_id)
        vectors = await embedder.embed_documents(list(texts.values()))
        vectors = validate_vectors(vectors, len(texts), embedder.identity.dimension)
        passages: list[PassagePoint] = []
        for index, (passage_id, text_value) in enumerate(texts.items()):
            relative_path = f"synthetic/{passage_id}.txt"
            document_id = document_id_for(workspace_id, relative_path)
            revision = hashlib.sha256(text_value.encode("utf-8")).hexdigest()
            manifest = DocumentManifest(
                schema_version=1,
                record_type="document",
                workspace_id=workspace_id,
                document_id=document_id,
                relative_source_path=relative_path,
                file_name=f"{passage_id}.txt",
                content_sha256=revision,
                status=DocumentStatus.READY,
                title=f"Synthetic {passage_id}",
                authors=("PhotomatAgent fixture authors",),
                year=2026,
                num_pages=1,
                chunk_count=1,
                model_fingerprint=generation.fingerprint,
                indexed_at=datetime.now(timezone.utc),
            )
            await store.upsert_document(manifest)
            passages.append(
                PassagePoint(
                    schema_version=1,
                    record_type="passage",
                    workspace_id=workspace_id,
                    document_id=document_id,
                    document_revision=revision,
                    ingest_state=IngestState.READY,
                    passage_id=fixture_point_ids[passage_id],
                    chunk_index=index,
                    text=text_value,
                    title=f"Synthetic {passage_id}",
                    authors=("PhotomatAgent fixture authors",),
                    year=2026,
                    section="Synthetic fixture",
                    heading_path="Synthetic fixture",
                    page_start=1,
                    page_end=1,
                    previous_passage_id=None,
                    next_passage_id=None,
                    relative_source_path=relative_path,
                    model_fingerprint=generation.fingerprint,
                    limitations=("Synthetic authored text; not copied from a paper.",),
                    dense=tuple(vectors[index]),
                    normalized_text_sha256=hashlib.sha256(
                        " ".join(text_value.split()).casefold().encode("utf-8")
                    ).hexdigest(),
                    indexed_at=datetime.now(timezone.utc),
                )
            )
        await store.upsert_passages(
            passages,
            batch_size=max(1, min(config.rag_batch_size, len(passages))),
            workspace_id=workspace_id,
        )
        activate = getattr(store, "activate_generation")
        result = activate(generation)
        if hasattr(result, "__await__"):
            await result
        live_judgments = [
            {
                **judgment,
                "relevant_passage_ids": [
                    fixture_point_ids[str(passage_id)]
                    for passage_id in judgment.get("relevant_passage_ids", ())
                ],
            }
            for judgment in judgments
        ]
        report = await evaluate_retrieval_fixture(
            retriever, live_judgments, workspace_id=workspace_id
        )
        report.update(
            {
                "live_evaluation": True,
                "live_collection_prefix": prefix,
                "live_isolated": True,
                "unavailable_reason": "",
            }
        )
        return report
    except Exception as exc:
        unavailable_reason = f"{_error_code(exc)}: {_safe_message(exc)}"
        # Keep the report schema stable for automation, but never turn an
        # unavailable live backend into a synthetic quality score.
        unavailable_metrics = {
            "recall_at_5": None,
            "mrr_at_10": None,
            "no_result_rate": None,
            "duplicate_rate": None,
            "provenance_completeness": None,
        }
        return {
            "label": "fixture-specific",
            "fixture_specific": True,
            "corpus_wide_claim": False,
            "live_evaluation": False,
            "live_collection_prefix": prefix,
            "live_isolated": True,
            "queries": len(judgments),
            "relevant_queries": None,
            "returned_results": None,
            "fixture_authors": sorted(
                {
                    str(item.get("fixture_author", "")).strip()
                    for item in judgments
                    if str(item.get("fixture_author", "")).strip()
                }
            ),
            "fixture_licenses": sorted(
                {
                    str(item.get("license", "")).strip()
                    for item in judgments
                    if str(item.get("license", "")).strip()
                }
            ),
            "metrics": unavailable_metrics,
            **unavailable_metrics,
            "thresholds": {
                "recall_at_5": {
                    "minimum": 0.90,
                    "actual": None,
                    "passed": False,
                },
                "provenance_completeness": {
                    "minimum": 1.0,
                    "actual": None,
                    "passed": False,
                },
                "duplicate_rate": {
                    "maximum": 0.0,
                    "actual": None,
                    "passed": False,
                },
            },
            "passed": False,
            "unavailable_reason": unavailable_reason,
            "backend_errors": [_error_code(exc)],
        }
    finally:
        if services is not None:
            await _close_isolated_store(_service_value(services, "store"), prefix)


async def evaluate_retrieval_fixture(
    retriever: Any,
    judgments: list[dict[str, Any]],
    *,
    workspace_id: str,
) -> dict[str, Any]:
    """Run the frozen synthetic judgments and return a bounded quality report.

    The report is intentionally fixture-specific.  It is a deterministic
    regression signal for retrieval wiring and provenance, not an estimate of
    quality over an arbitrary literature corpus.
    """
    if not judgments:
        raise ValueError("at least one evaluation judgment is required")
    recall_hits = 0
    reciprocal_rank_sum = 0.0
    relevant_queries = 0
    no_result_queries = 0
    returned_count = 0
    duplicate_count = 0
    complete_provenance_count = 0
    backend_errors: list[str] = []
    per_query: list[dict[str, Any]] = []

    for judgment in judgments:
        query = str(judgment["query"])
        relevant_ids = {
            str(item) for item in judgment.get("relevant_passage_ids", ()) if str(item)
        }
        try:
            result = await retriever.search(
                query,
                workspace_id=workspace_id,
                # Fetch ten once: Recall@5 is computed from the first five,
                # while MRR@10 must be able to observe a relevant hit at
                # ranks 6–10.
                top_k=10,
            )
        except Exception as exc:
            code = getattr(exc, "code", None) or type(exc).__name__.casefold()
            backend_errors.append(str(code)[:160])
            no_result_queries += 1
            per_query.append(
                {
                    "query": query,
                    "category": str(judgment.get("category", "")),
                    "result_count": 0,
                    "first_relevant_rank": None,
                    "error": str(code)[:160],
                }
            )
            continue
        passages = list(_evaluation_value(result, "passages", ()) or ())[:10]
        if not passages:
            no_result_queries += 1
        ids = [_evaluation_passage_id(item) for item in passages]
        first_relevant_rank: int | None = None
        if relevant_ids:
            relevant_queries += 1
            first_relevant_rank = next(
                (
                    rank
                    for rank, passage_id in enumerate(ids, start=1)
                    if passage_id in relevant_ids
                ),
                None,
            )
            if first_relevant_rank is not None and first_relevant_rank <= 5:
                recall_hits += 1
            if first_relevant_rank is not None and first_relevant_rank <= 10:
                reciprocal_rank_sum += 1.0 / first_relevant_rank

        seen_keys: set[tuple[str, str]] = set()
        for item in passages:
            returned_count += 1
            if _evaluation_provenance_complete(item):
                complete_provenance_count += 1
            duplicate_key = _evaluation_duplicate_key(item)
            if duplicate_key is not None:
                if duplicate_key in seen_keys:
                    duplicate_count += 1
                seen_keys.add(duplicate_key)
        per_query.append(
            {
                "query": query,
                "category": str(judgment.get("category", "")),
                "result_count": len(passages),
                "first_relevant_rank": first_relevant_rank,
            }
        )

    recall_at_5 = recall_hits / relevant_queries if relevant_queries else 1.0
    mrr_at_10 = reciprocal_rank_sum / relevant_queries if relevant_queries else 0.0
    no_result_rate = no_result_queries / len(judgments)
    duplicate_rate = duplicate_count / returned_count if returned_count else 0.0
    provenance_completeness = (
        complete_provenance_count / returned_count if returned_count else 1.0
    )
    thresholds = {
        "recall_at_5": {
            "minimum": 0.90,
            "actual": recall_at_5,
            "passed": recall_at_5 >= 0.90,
        },
        "provenance_completeness": {
            "minimum": 1.0,
            "actual": provenance_completeness,
            "passed": provenance_completeness >= 1.0,
        },
        "duplicate_rate": {
            "maximum": 0.0,
            "actual": duplicate_rate,
            "passed": duplicate_rate <= 0.0,
        },
    }
    passed = not backend_errors and all(
        bool(item["passed"]) for item in thresholds.values()
    )
    metrics = {
        "recall_at_5": recall_at_5,
        "mrr_at_10": mrr_at_10,
        "no_result_rate": no_result_rate,
        "duplicate_rate": duplicate_rate,
        "provenance_completeness": provenance_completeness,
    }
    fixture_authors = sorted(
        {
            str(judgment.get("fixture_author", "")).strip()
            for judgment in judgments
            if str(judgment.get("fixture_author", "")).strip()
        }
    )
    fixture_licenses = sorted(
        {
            str(judgment.get("license", "")).strip()
            for judgment in judgments
            if str(judgment.get("license", "")).strip()
        }
    )
    return {
        "label": "fixture-specific",
        "fixture_specific": True,
        "corpus_wide_claim": False,
        "fixture_authors": fixture_authors,
        "fixture_licenses": fixture_licenses,
        "queries": len(judgments),
        "relevant_queries": relevant_queries,
        "returned_results": returned_count,
        "metrics": metrics,
        "recall_at_5": recall_at_5,
        "mrr_at_10": mrr_at_10,
        "no_result_rate": no_result_rate,
        "duplicate_rate": duplicate_rate,
        "provenance_completeness": provenance_completeness,
        "backend_errors": list(dict.fromkeys(backend_errors))[:20],
        "thresholds": thresholds,
        "passed": passed,
        "per_query": per_query,
    }


def _directory(boundary: Workspace, directory: Path | None, config: ScientificConfig) -> Path:
    value = directory if directory is not None else Path(config.literature_root)
    return boundary.resolve(str(value), must_exist=False)


@rag_app.callback(invoke_without_command=True)
def rag_default(
    ctx: typer.Context,
    workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False),
) -> None:
    """Default to the read-only RAG status report."""
    if ctx.invoked_subcommand is None:
        rag_status(workspace=workspace)


@rag_app.command("status")
def rag_status(
    workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False),
) -> None:
    """Show Qdrant, alias, provider, and source-root readiness without secrets."""
    boundary = _workspace(workspace)
    config = _config(workspace)
    probe = LiteratureProbe(config, boundary)
    result = probe.probe()
    snapshot_method = getattr(probe, "status_snapshot", None)
    snapshot = snapshot_method() if callable(snapshot_method) else {}
    source_state = snapshot.get("source_root")
    if not source_state or source_state == "unknown":
        try:
            source_root = boundary.resolve(config.literature_root, must_exist=False)
        except Exception:
            source_state = "outside workspace"
        else:
            source_state = (
                f"{source_root} ({'ready' if source_root.is_dir() else 'missing'})"
            )
    legacy_path = boundary.root / "output" / "literature_index"
    legacy_state = (
        f"present; not imported or modified ({legacy_path})"
        if legacy_path.exists()
        else "absent; no legacy artifact to migrate"
    )
    table = Table("RAG status", "Value")
    rows = [
        ("Capability", result.status.value),
        ("Detail", result.detail or "—"),
        ("Version", result.version or "—"),
        ("Qdrant server", snapshot.get("server_version", "unknown")),
        ("Qdrant client", snapshot.get("client_version", "unknown")),
        ("Alias state", snapshot.get("alias_state", "unknown")),
        ("Generation", snapshot.get("generation_state", "unknown")),
        ("Schema", snapshot.get("schema", "unknown")),
        ("Fingerprint", snapshot.get("fingerprint", "unknown")),
        ("Documents", snapshot.get("documents", "unknown")),
        ("Passages", snapshot.get("passages", "unknown")),
        ("Indexed vectors", snapshot.get("indexed_vectors", "unknown")),
        ("Collection status", snapshot.get("collection_status", "unknown")),
        ("Capacity", snapshot.get("capacity", "unknown")),
        ("Capacity warning", snapshot.get("capacity_warning", "none") or "none"),
        ("Qdrant URL", sanitize_qdrant_url(config.qdrant_url)),
        ("TLS", snapshot.get("tls", "enabled" if config.qdrant_url.lower().startswith("https://") else "disabled")),
        ("Auth", snapshot.get("auth", "configured (value hidden)" if os.environ.get(config.qdrant_api_key_env) else "not configured")),
        ("Collection prefix", config.qdrant_collection_prefix),
        ("Embedding", snapshot.get("embedding_provider", f"{config.embedding_provider}/{config.embedding_model}")),
        ("Reranker", snapshot.get("reranker_provider", f"{config.reranker_provider}/{config.reranker_model}")),
        ("Source root", source_state),
        ("Legacy artifact", legacy_state),
    ]
    for label, value in rows:
        table.add_row(label, str(value))
    console.print(table)


@rag_app.command("plan")
def rag_plan(
    directory: Path | None = typer.Option(None, "--directory"),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False),
) -> None:
    """Create a read-only ingestion plan; no parsing, embedding, or writes."""
    boundary = _workspace(workspace)
    config = _config(workspace)
    root = _directory(boundary, directory, config)
    try:
        services = build_literature_services(config, boundary)
        plan = asyncio.run(services.ingestion.plan(root, boundary))
    except Exception as exc:
        _print_error(exc)
        raise typer.Exit(code=1) from exc
    items = list(getattr(plan, "items", ()))
    by_kind: dict[str, int] = {}
    for item in items:
        kind = str(getattr(getattr(item, "kind", "unknown"), "value", getattr(item, "kind", "unknown")))
        by_kind[kind] = by_kind.get(kind, 0) + 1
    print(
        json.dumps(
            {
                "directory": str(root),
                "workspace_id": str(getattr(plan, "workspace_id", "")),
                "complete": bool(getattr(plan, "complete", False)),
                "documents": len(items),
                "kinds": by_kind,
                "read_only": True,
            },
            ensure_ascii=False,
        )
    )


async def _index_until_complete(
    services: Any,
    boundary: Workspace,
    root: Path,
    *,
    config: ScientificConfig,
    run_id: str,
) -> dict[str, Any]:
    # Indexing owns the physical staging pair.  It intentionally does not
    # activate stable aliases; activation is a separate explicit command.
    store = _service_value(services, "store")
    embedder = _service_value(services, "ingestion").embedder
    ensure_generation = getattr(store, "ensure_generation", None)
    if callable(ensure_generation):
        generation = ensure_generation(
            identity=embedder.identity,
            chunk_schema_version=LITERATURE_CHUNK_SCHEMA_VERSION,
        )
        if hasattr(generation, "__await__"):
            generation = await generation
    plan = await services.ingestion.plan(root, boundary)
    cursor: str | None = None
    last: dict[str, Any] = {"run_id": run_id, "complete": False}
    while True:
        stats = await services.ingestion.index_batch(
            plan,
            run_id=run_id,
            resume_cursor=cursor,
            max_documents=config.rag_tool_max_documents,
        )
        last = _stats_dict(stats)
        last["run_id"] = run_id
        if bool(last.get("complete")):
            return last
        # A failed document is intentionally left at the retry cursor.  Do
        # not spin forever retrying a permanent failure in one CLI process;
        # the persisted run can be resumed explicitly after remediation.
        if bool(last.get("retryable")):
            return last
        next_cursor = last.get("next_cursor")
        if next_cursor == cursor and not bool(last.get("retryable")):
            return last
        cursor = str(next_cursor) if next_cursor is not None else None


@rag_app.command("index")
def rag_index(
    directory: Path | None = typer.Option(None, "--directory"),
    run_id: str | None = typer.Option(None, "--run-id"),
    yes: bool = typer.Option(False, "--yes", help="Confirm external data transfer."),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False),
) -> None:
    """Run bounded ingestion batches until the resumable plan is complete."""
    boundary = _workspace(workspace)
    config = _config(workspace)
    root = _directory(boundary, directory, config)
    effective_run_id = run_id or uuid4().hex
    try:
        # The disclosure is intentionally before service construction so a
        # declined confirmation cannot initialize an external client.
        _confirm_external(config, root, yes=yes)
        services = build_literature_services(config, boundary)
        stats = asyncio.run(
            _index_until_complete(
                services,
                boundary,
                root,
                config=config,
                run_id=effective_run_id,
            )
        )
    except KeyboardInterrupt:
        console.print(f"[yellow]索引已中断；run_id={effective_run_id}，可使用 --run-id 恢复。[/]")
        raise typer.Exit(code=1) from None
    except Exception as exc:
        _print_error(exc)
        raise typer.Exit(code=1) from exc
    print(json.dumps(stats, ensure_ascii=False))
    if not bool(stats.get("complete", False)):
        raise typer.Exit(code=1)


@rag_app.command("activate")
def rag_activate(
    yes: bool = typer.Option(False, "--yes", help="Confirm alias activation after validation."),
    bootstrap: bool = typer.Option(
        False,
        "--bootstrap",
        help="Explicitly allow activating an empty generation for a new corpus.",
    ),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False),
) -> None:
    """Explicitly activate the validated provider/schema generation."""
    if not yes:
        typer.confirm(
            "确认将当前 provider/schema generation 原子切换到 current aliases？",
            abort=True,
        )
    boundary = _workspace(workspace)
    config = _config(workspace)
    try:
        services = build_literature_services(config, boundary)
        store = _service_value(services, "store")
        embedder = _service_value(services, "ingestion").embedder

        async def activate() -> Any:
            ensure = getattr(store, "ensure_generation")
            generation = ensure(
                identity=embedder.identity,
                chunk_schema_version=LITERATURE_CHUNK_SCHEMA_VERSION,
            )
            if hasattr(generation, "__await__"):
                generation = await generation
            activate_method = getattr(store, "activate_generation")
            result = activate_method(
                generation,
                allow_empty_bootstrap=bootstrap,
            )
            if hasattr(result, "__await__"):
                await result
            return generation

        generation = asyncio.run(activate())
    except Exception as exc:
        _print_error(exc)
        raise typer.Exit(code=1) from exc
    print(
        json.dumps(
            {
                "activated": True,
                "fingerprint": str(getattr(generation, "fingerprint", "")),
                "documents": str(getattr(generation, "documents_physical", "")),
                "passages": str(getattr(generation, "passages_physical", "")),
            },
            ensure_ascii=False,
        )
    )


@rag_app.command("search")
def rag_search(
    query: str = typer.Argument(...),
    top_k: int = typer.Option(5, "--top-k", min=1, max=10),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False),
) -> None:
    """Search bounded provenance-carrying passages."""
    boundary = _workspace(workspace)
    config = _config(workspace)
    try:
        services = build_literature_services(config, boundary)
        result = asyncio.run(
            LiteratureSearchPassagesTool(config, boundary, services).execute(
                {"query": query, "top_k": top_k}
            )
        )
    except Exception as exc:
        _print_error(exc)
        raise typer.Exit(code=1) from exc
    if result.is_error:
        print(result.output)
        raise typer.Exit(code=1)
    print(json.dumps(result.data, ensure_ascii=False))


@rag_app.command("read")
def rag_read(
    passage_id: str = typer.Argument(...),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False),
) -> None:
    """Read one exact ready passage by opaque passage id."""
    boundary = _workspace(workspace)
    config = _config(workspace)
    try:
        services = build_literature_services(config, boundary)
        result = asyncio.run(
            LiteratureReadPassageTool(config, boundary, services).execute(
                {"passage_id": passage_id}
            )
        )
    except Exception as exc:
        _print_error(exc)
        raise typer.Exit(code=1) from exc
    if result.is_error:
        print(result.output)
        raise typer.Exit(code=1)
    print(json.dumps(result.data, ensure_ascii=False))


@rag_app.command("evaluate")
def rag_evaluate(
    fixture: Path | None = typer.Option(
        None,
        "--fixture",
        help="Optional path to a compatible synthetic judgment fixture.",
    ),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False),
) -> None:
    """Run the frozen, fixture-specific retrieval quality evaluation."""
    boundary = _workspace(workspace)
    config = _config(workspace)
    try:
        fixture_path = (
            boundary.resolve(str(fixture), must_exist=True)
            if fixture is not None
            else _evaluation_fixture_path()
        )
        judgments = _load_evaluation_fixture(fixture_path)
        report = asyncio.run(evaluate_live_fixture(config, boundary, judgments))
    except Exception as exc:
        _print_error(exc)
        raise typer.Exit(code=1) from exc
    print(json.dumps(report, ensure_ascii=False))
    if not bool(report.get("passed", False)):
        raise typer.Exit(code=1)


@rag_app.command("snapshot")
def rag_snapshot(
    output: Path | None = typer.Option(None, "--output"),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False),
) -> None:
    """Create current Qdrant collection snapshots in a workspace directory."""
    boundary = _workspace(workspace)
    config = _config(workspace)
    destination = output or Path(
        ".photomatagent/rag/backups/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    try:
        destination = boundary.resolve(str(destination), must_exist=False)
        services = build_literature_services(config, boundary)
        manifest = asyncio.run(services.store.create_current_snapshots(destination))
    except Exception as exc:
        _print_error(exc)
        raise typer.Exit(code=1) from exc
    if hasattr(manifest, "to_dict"):
        payload = manifest.to_dict()
    else:
        payload = {"directory": str(destination), "manifest": str(manifest)}
    print(json.dumps(payload, ensure_ascii=False, default=str))


__all__ = ["evaluate_live_fixture", "evaluate_retrieval_fixture", "rag_app"]
