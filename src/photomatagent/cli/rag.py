"""Direct, user-controlled operations for the literature Qdrant RAG services."""

from __future__ import annotations

import asyncio
import json
import os
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
from photomatagent.workspace import Workspace


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
    source_root = boundary.resolve(config.literature_root, must_exist=False)
    table = Table("RAG status", "Value")
    rows = [
        ("Capability", result.status.value),
        ("Detail", result.detail or "—"),
        ("Version", result.version or "—"),
        ("Qdrant server", snapshot.get("server_version", "unknown")),
        ("Alias state", snapshot.get("alias_state", "unknown")),
        ("Generation", snapshot.get("generation_state", "unknown")),
        ("Qdrant URL", config.qdrant_url),
        ("Qdrant API key", "configured (value hidden)" if os.environ.get(config.qdrant_api_key_env) else "not configured"),
        ("Collection prefix", config.qdrant_collection_prefix),
        ("Embedding", f"{config.embedding_provider}/{config.embedding_model}"),
        ("Reranker", f"{config.reranker_provider}/{config.reranker_model}"),
        ("Source root", f"{source_root} ({'ready' if source_root.is_dir() else 'missing'})"),
        ("Legacy artifact", "output/literature_index is not imported or modified"),
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
    workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False),
) -> None:
    """Run the frozen retrieval evaluation when its fixture is available."""
    del workspace
    print(json.dumps({"status": "not_configured", "message": "evaluation fixture is provided by the integration task"}, ensure_ascii=False))


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


__all__ = ["rag_app"]
