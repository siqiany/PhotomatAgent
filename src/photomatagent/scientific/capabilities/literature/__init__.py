"""Literature capability pack: arXiv search + local PDF search/read.

Namespace ``literature``, DEFERRED. Result counts and text lengths are hard
capped so a literature step can never flood model context.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import inspect
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
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


@dataclass(frozen=True)
class LiteratureServices:
    """Application services shared by model tools and the direct CLI.

    The object intentionally contains narrow service interfaces rather than a
    Qdrant client.  Qdrant stays an internal implementation detail of the
    ingestion/retrieval services and can be replaced with fakes in tests.
    """

    ingestion: Any
    retriever: Any
    store: Any
    workspace_id: str


def build_literature_services(
    config: ScientificConfig,
    workspace: Workspace,
    *,
    store: Any | None = None,
    embedder: Any | None = None,
    reranker: Any | None = None,
) -> LiteratureServices:
    """Build the one shared literature application-service graph lazily."""
    from photomatagent.scientific.capabilities.literature.ingestion import (
        LiteratureIngestionService,
    )
    from photomatagent.scientific.capabilities.literature.providers.factory import (
        build_embedding_provider,
        build_reranker_provider,
    )
    from photomatagent.scientific.capabilities.literature.qdrant_store import (
        QdrantLiteratureStore,
        workspace_id_for,
    )
    from photomatagent.scientific.capabilities.literature.retrieval import (
        LiteratureRetriever,
    )

    effective_store = (
        store if store is not None else QdrantLiteratureStore.from_config(config)
    )
    effective_embedder = (
        embedder if embedder is not None else build_embedding_provider(config)
    )
    effective_reranker = (
        reranker if reranker is not None else build_reranker_provider(config)
    )
    workspace_id = workspace_id_for(workspace.root.resolve())
    return LiteratureServices(
        ingestion=LiteratureIngestionService(
            effective_store,
            effective_embedder,
            batch_size=config.rag_batch_size,
        ),
        retriever=LiteratureRetriever(
            effective_store,
            effective_embedder,
            effective_reranker,
        ),
        store=effective_store,
        workspace_id=workspace_id,
    )


def _service_value(services: Any, name: str, default: Any = None) -> Any:
    if isinstance(services, dict):
        return services.get(name, default)
    return getattr(services, name, default)


def _error_code(exc: BaseException) -> str:
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        return code
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return "qdrant_unreachable"
    return type(exc).__name__.casefold() or "literature_error"


def _error_result(exc: BaseException, *, operation: str) -> ScientificToolResult:
    """Convert typed service failures to a bounded, secret-free tool result."""
    code = _error_code(exc)
    message = str(getattr(exc, "message", "") or "")
    if not message:
        message = f"{operation} failed"
    # Do not echo URLs, credentials, tracebacks, or unbounded provider output.
    message = re.sub(r"(?i)(api[_ -]?key|token|secret|password)\s*[:=]\s*[^\s,;]+", r"\1=[redacted]", message)
    message = _clean(message, 320)
    return ScientificToolResult(
        output=f"{code}: {message}",
        is_error=True,
        data={"error": code},
    )


_PROBE_TIMEOUT_SECONDS = 2
_CHUNK_SCHEMA_VERSION = 1
_PROVIDER_UNCONFIGURED_CODES = {
    "external_provider_not_allowed",
    "external_api_key_env_missing",
    "external_api_key_missing",
    "external_base_url_missing",
    "external_base_url_invalid",
}


def _probe_record_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _probe_capacity_data(value: Any) -> dict[str, int | str]:
    """Extract bounded server/collection capacity observations when exposed."""
    data: dict[str, int | str] = {}
    fields = (
        "disk_total_bytes",
        "disk_free_bytes",
        "disk_available_bytes",
        "disk_usage_bytes",
        "storage_limit_bytes",
        "capacity_bytes",
    )
    nested = _probe_record_value(value, "disk", None)
    for name in fields:
        raw = _probe_record_value(value, name, _probe_record_value(nested, name, None))
        if isinstance(raw, bool) or raw is None:
            continue
        try:
            parsed = int(raw)
        except (TypeError, ValueError):
            continue
        if parsed >= 0:
            data[name] = parsed
    return data


def _probe_capacity_warning(data: Mapping[str, int | str]) -> str:
    """Return a conservative warning when a service publishes usable limits."""
    total = next(
        (
            int(data[name])
            for name in ("disk_total_bytes", "storage_limit_bytes", "capacity_bytes")
            if name in data and int(data[name]) > 0
        ),
        None,
    )
    free = next(
        (
            int(data[name])
            for name in ("disk_free_bytes", "disk_available_bytes")
            if name in data
        ),
        None,
    )
    usage = int(data["disk_usage_bytes"]) if "disk_usage_bytes" in data else None
    if total is not None and free is not None and free / total <= 0.10:
        return "disk_free_below_10_percent"
    if total is not None and usage is not None and usage / total >= 0.90:
        return "disk_usage_above_90_percent"
    return ""


def _probe_alias_map(response: Any) -> dict[str, str]:
    aliases = _probe_record_value(response, "aliases", response)
    if isinstance(aliases, Mapping):
        return {str(key): str(value) for key, value in aliases.items()}
    result: dict[str, str] = {}
    for alias in list(aliases or ()):
        alias_name = _probe_record_value(alias, "alias_name")
        collection_name = _probe_record_value(alias, "collection_name")
        if alias_name is not None and collection_name is not None:
            result[str(alias_name)] = str(collection_name)
    return result


def _probe_version(server_version: str = "") -> str:
    versions = []
    if server_version:
        versions.append(f"qdrant-server={server_version}")
    versions.extend(
        [
            f"arxiv={_version('arxiv')}",
            f"pypdf={_version('pypdf')}",
            f"docling={_version('docling')}",
            f"qdrant-client={_version('qdrant-client')}",
            f"sentence-transformers={_version('sentence-transformers')}",
        ]
    )
    return "; ".join(versions)


def _probe_fingerprint_compatible(actual: str, expected: str) -> bool:
    return actual == expected or (
        len(actual) == 12 and expected.startswith(actual)
    ) or (len(expected) == 12 and actual.startswith(expected))


def _probe_error_code(exc: BaseException) -> str:
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        return code
    return type(exc).__name__.casefold() or "literature_probe_error"


async def _await_probe_close(value: Any) -> None:
    await value


class LiteratureProbe(CapabilityPack):
    name = "literature"
    description = (
        "Literature search and reading (arXiv + local PDFs) plus the "
        "Qdrant-backed literature RAG services."
    )

    def probe(self) -> ProbeResult:
        """Run bounded, read-only checks without loading local models."""
        self._status = {
            "server_version": "unknown",
            "client_version": _version("qdrant-client") or "unknown",
            "alias_state": "not checked",
            "generation_state": "not checked",
            "schema": "not checked",
            "fingerprint": "unknown",
            "documents": "unknown",
            "passages": "unknown",
            "indexed_vectors": "unknown",
            "collection_status": "unknown",
            "capacity": "unknown",
            "capacity_warning": "",
            "tls": "unknown",
            "auth": "not configured",
            "qdrant_url": "",
            "embedding_provider": "unknown",
            "reranker_provider": "unknown",
            "source_root": "unknown",
            "legacy_artifact": "unknown",
        }
        legacy_path = self._workspace.root / "output" / "literature_index"
        self._status["legacy_artifact"] = (
            "present; not imported or modified"
            if legacy_path.exists()
            else "absent; no legacy artifact to migrate"
        )
        missing: list[str] = []
        for module_name in ("arxiv", "pypdf", "docling", "qdrant_client", "sentence_transformers"):
            try:
                __import__(module_name)
            except ImportError:
                missing.append(module_name)
            except Exception:
                return ProbeResult(
                    status=CapabilityStatus.ERROR,
                    detail=f"dependency import failed: {module_name}",
                )
        if not self._config.qdrant_url.strip():
            return ProbeResult(
                status=CapabilityStatus.UNCONFIGURED,
                detail="qdrant_url is not configured",
            )
        if not self._config.embedding_model.strip():
            return ProbeResult(
                status=CapabilityStatus.UNCONFIGURED,
                detail="embedding_model_missing: embedding model is not configured",
            )
        if (
            self._config.reranker_provider != "disabled"
            and not self._config.reranker_model.strip()
        ):
            return ProbeResult(
                status=CapabilityStatus.UNCONFIGURED,
                detail="reranker_model_missing: reranker model is not configured",
            )

        try:
            from photomatagent.scientific.capabilities.literature.qdrant_store import (
                sanitize_qdrant_url,
                validate_qdrant_url,
            )

            api_key = os.environ.get(self._config.qdrant_api_key_env, "").strip()
            self._status["qdrant_url"] = sanitize_qdrant_url(self._config.qdrant_url)
            self._status["tls"] = (
                "enabled"
                if self._config.qdrant_url.lower().startswith("https://")
                else "disabled"
            )
            self._status["auth"] = "configured (value hidden)" if api_key else "not configured"
            validate_qdrant_url(self._config.qdrant_url, api_key=api_key or None)
        except Exception as exc:
            qdrant_url_error = _probe_error_code(exc)
        else:
            qdrant_url_error = ""

        try:
            from photomatagent.scientific.capabilities.literature.providers.factory import (
                build_embedding_provider,
                build_reranker_provider,
            )

            embedding = build_embedding_provider(self._config)
            reranker = build_reranker_provider(self._config)
            self._status["embedding_provider"] = (
                f"{embedding.identity.provider}/{embedding.identity.model}"
            )
            self._status["reranker_provider"] = (
                f"{reranker.identity.provider}/{reranker.identity.model}"
            )
            provider_error = ""
        except Exception as exc:
            code = _probe_error_code(exc)
            provider_error = code
            embedding = None
            self._status["embedding_provider"] = f"error:{code}"
            self._status["reranker_provider"] = "not checked"

        source_root_missing = False
        try:
            source_root = self._workspace.resolve(
                self._config.literature_root, must_exist=False
            )
        except Exception:
            source_root = None
            source_root_missing = True
            self._status["source_root"] = "outside workspace"
        else:
            assert source_root is not None
            source_root_missing = not source_root.is_dir()
            self._status["source_root"] = (
                f"{source_root} ({'ready' if not source_root_missing else 'missing'})"
            )

        client = None
        aliases: dict[str, str] = {}
        server_version = ""
        qdrant_error = qdrant_url_error
        if not qdrant_error:
            try:
                # The sync client is used only for a short, read-only health
                # check.  It never creates collections or loads model weights.
                from qdrant_client import QdrantClient

                api_key = os.environ.get(self._config.qdrant_api_key_env, "").strip()
                client = QdrantClient(
                    url=self._config.qdrant_url,
                    api_key=api_key or None,
                    timeout=min(
                        self._config.qdrant_timeout_seconds, _PROBE_TIMEOUT_SECONDS
                    ),
                    prefer_grpc=False,
                )
                # Keep each bounded read independent.  A version endpoint or
                # collection-health failure must not suppress alias/schema
                # inspection that can still explain the operator state.
                try:
                    client.get_collections()
                except Exception as exc:
                    qdrant_error = qdrant_error or _probe_error_code(exc)

                info_method = getattr(client, "info", None)
                if info_method is None:
                    qdrant_error = qdrant_error or "qdrant_server_version_missing"
                else:
                    try:
                        info = info_method()
                        raw_server_version = _probe_record_value(info, "version", "")
                        server_version = str(raw_server_version or "").strip()
                        if not server_version:
                            qdrant_error = qdrant_error or "qdrant_server_version_missing"
                        else:
                            self._status["server_version"] = server_version
                        capacity_data = _probe_capacity_data(info)
                        if capacity_data:
                            self._status["capacity"] = ", ".join(
                                f"{key}={value}" for key, value in sorted(capacity_data.items())
                            )
                            capacity_warning = _probe_capacity_warning(capacity_data)
                            if capacity_warning:
                                self._status["capacity_warning"] = capacity_warning
                    except Exception:
                        qdrant_error = qdrant_error or "qdrant_server_version_unavailable"

                aliases_method = getattr(client, "get_aliases", None)
                if aliases_method is None:
                    aliases_method = getattr(client, "get_collection_aliases", None)
                if aliases_method is None:
                    qdrant_error = qdrant_error or "qdrant_aliases_unavailable"
                else:
                    try:
                        aliases = _probe_alias_map(aliases_method())
                    except Exception as exc:
                        qdrant_error = qdrant_error or _probe_error_code(exc)
                    else:
                        expected_aliases = {
                            f"{self._config.qdrant_collection_prefix}_documents_current",
                            f"{self._config.qdrant_collection_prefix}_passages_current",
                        }
                        missing_aliases = sorted(expected_aliases - set(aliases))
                        if missing_aliases:
                            self._status["alias_state"] = "missing: " + ", ".join(
                                missing_aliases
                            )
                            qdrant_error = qdrant_error or "current_aliases_missing"
                        else:
                            self._status["alias_state"] = "ready"

                        # Collection metadata is observational only.  Different
                        # qdrant-client versions expose counts/configuration at
                        # different nesting levels, so leave unavailable values
                        # explicitly unknown rather than manufacturing readiness.
                        for alias_name, collection_name in aliases.items():
                            if not alias_name.endswith("_current"):
                                continue
                            try:
                                collection_info = client.get_collection(collection_name)
                            except Exception:
                                continue
                            metadata = _probe_record_value(collection_info, "metadata", {})
                            if not isinstance(metadata, Mapping) or not metadata:
                                config_info = _probe_record_value(collection_info, "config", None)
                                metadata = _probe_record_value(config_info, "metadata", {})
                            if isinstance(metadata, Mapping):
                                schema = metadata.get("collection_schema_version") or metadata.get(
                                    "photomat_collection_schema_version"
                                )
                                fingerprint = metadata.get("model_fingerprint") or metadata.get(
                                    "photomat_model_fingerprint"
                                )
                                if schema is not None:
                                    self._status["schema"] = str(schema)
                                if fingerprint:
                                    self._status["fingerprint"] = str(fingerprint)
                            for field_name, status_key in (
                                ("points_count", "documents" if "documents" in alias_name else "passages"),
                                ("indexed_vectors_count", "indexed_vectors"),
                            ):
                                value = _probe_record_value(collection_info, field_name, None)
                                if value is not None:
                                    self._status[status_key] = str(value)
                            collection_status = _probe_record_value(collection_info, "status", None)
                            if collection_status is not None:
                                self._status["collection_status"] = str(collection_status)
                                normalized_status = str(
                                    getattr(collection_status, "value", collection_status)
                                ).casefold()
                                if normalized_status not in {
                                    "green",
                                    "ok",
                                    "healthy",
                                    "active",
                                    "indexed",
                                }:
                                    self._status["capacity_warning"] = (
                                        f"collection_status:{normalized_status}"
                                    )
                            capacity = _probe_record_value(collection_info, "disk_usage_bytes", None)
                            if capacity is not None:
                                self._status["capacity"] = str(capacity)
            except ImportError:
                qdrant_error = "qdrant_dependency_missing"
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                qdrant_error = "qdrant_auth_failed" if status_code in {401, 403} else "qdrant_unreachable"
            finally:
                close = getattr(client, "close", None)
                if close is not None:
                    try:
                        closed = close()
                        if inspect.isawaitable(closed):
                            asyncio.run(_await_probe_close(closed))
                    except Exception:
                        pass

        generation_error = ""
        generation = None
        if embedding is None:
            generation_error = provider_error or "embedding_provider_unavailable"
            self._status["generation_state"] = "not checked (provider error)"
        else:
            try:
                from photomatagent.scientific.capabilities.literature.qdrant_store import (
                    QdrantLiteratureStore,
                    QdrantStoreError,
                    collection_fingerprint,
                )

                probe_config = dataclass_replace(
                    self._config,
                    qdrant_timeout_seconds=min(
                        self._config.qdrant_timeout_seconds, _PROBE_TIMEOUT_SECONDS
                    ),
                )
                store = QdrantLiteratureStore.from_config(probe_config)
                expected_fingerprint = collection_fingerprint(
                    embedding.identity,
                    _CHUNK_SCHEMA_VERSION,
                    prefix=self._config.qdrant_collection_prefix,
                    sparse_model=getattr(store, "sparse_model", "qdrant/bm25"),
                )

                async def validate_generation() -> Any:
                    try:
                        current = await store.resolve_current_generation()
                        if current is None:
                            return None
                        actual_fingerprint = str(
                            _probe_record_value(current, "fingerprint", "")
                        )
                        if not _probe_fingerprint_compatible(
                            actual_fingerprint, expected_fingerprint
                        ):
                            raise QdrantStoreError(
                                "model_fingerprint_mismatch",
                                "current collection fingerprint does not match provider",
                            )
                        validate = getattr(store, "validate_current_generation", None)
                        if validate is None:
                            raise QdrantStoreError(
                                "control_point_unavailable",
                                "current generation validation is unavailable",
                            )
                        await validate(expected_fingerprint)
                        return current
                    finally:
                        close_client = getattr(getattr(store, "_client", None), "close", None)
                        if close_client is not None:
                            closed_client = close_client()
                            if inspect.isawaitable(closed_client):
                                await closed_client

                generation = asyncio.run(validate_generation())
                if generation is None:
                    generation_error = "current_generation_missing"
                    self._status["generation_state"] = "missing"
                else:
                    generation_fingerprint = str(
                        _probe_record_value(generation, "fingerprint", "")
                    )
                    self._status["generation_state"] = (
                        f"ready:{generation_fingerprint[:12]}"
                    )
                    self._status["fingerprint"] = generation_fingerprint or "unknown"
            except Exception as exc:
                generation_error = _probe_error_code(exc)
                self._status["generation_state"] = f"error:{generation_error}"

        issues: list[str] = []
        if missing:
            issues.append(
                "missing_dependency: " + ", ".join(missing)
            )
        if qdrant_error:
            issues.append(qdrant_error)
        if provider_error:
            issues.append(provider_error)
        if source_root_missing:
            issues.append("source_root_missing")
        if generation_error:
            issues.append(generation_error)
        if not issues:
            result_status = CapabilityStatus.AVAILABLE
            detail = (
                "arxiv + pypdf + docling + qdrant-client + sentence-transformers "
                "available; source root and Qdrant ready"
            )
        else:
            unconfigured_codes = set(_PROVIDER_UNCONFIGURED_CODES) | {
                "current_aliases_missing",
                "current_generation_missing",
                "source_root_missing",
                "qdrant_api_key_required",
                "qdrant_tls_required",
                "qdrant_url_invalid",
                "qdrant_url_not_configured",
                "qdrant_dependency_missing",
            }
            if missing:
                result_status = CapabilityStatus.MISSING_DEPENDENCY
            else:
                result_status = (
                    CapabilityStatus.UNCONFIGURED
                    if any(issue in unconfigured_codes for issue in issues)
                    else CapabilityStatus.ERROR
                )
            detail = "; ".join(
                "source root is missing" if issue == "source_root_missing" else issue
                for issue in issues
            )
            if missing:
                detail = (
                    f"{detail} (extra: photomatagent[literature])"
                )
        detail = (
            f"{detail}; aliases={self._status['alias_state']}; "
            f"generation={self._status['generation_state']}"
        )
        return ProbeResult(
            status=result_status,
            detail=detail,
            version=_probe_version(server_version),
        )

    def status_snapshot(self) -> dict[str, str]:
        """Return the latest bounded status details for direct CLI rendering."""
        return dict(
            getattr(
                self,
                "_status",
                {
                    "server_version": "unknown",
                    "alias_state": "not checked",
                    "generation_state": "not checked",
                },
            )
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
        self._status = {
            "server_version": "unknown",
            "alias_state": "not checked",
            "generation_state": "not checked",
        }


def _papers_dir(workspace: Workspace) -> Path:
    candidates = [workspace.root / "papers", Path.cwd() / "papers"]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def _clean(text: str, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) > limit:
        marker = "...[truncated]"
        if limit <= len(marker):
            return marker[: max(0, limit)]
        cleaned = cleaned[: limit - len(marker)] + marker
    return cleaned


def _workspace_id(workspace: Workspace) -> str:
    from photomatagent.scientific.capabilities.literature.qdrant_store import (
        workspace_id_for,
    )

    return workspace_id_for(workspace.root.resolve())


def _record_value(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def _stats_payload(stats: Any) -> dict[str, Any]:
    """Render only the bounded ingestion progress contract."""
    get = lambda name, default=None: _record_value(stats, name, default)
    errors = [
        _clean(str(error), 320)
        for error in list(get("errors", ()) or ())[:20]
    ]
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
        "errors": errors,
    }


def _passage_payload(record: Any, *, text_limit: int | None) -> dict[str, Any]:
    """Convert a Qdrant passage point to the stable public read contract."""
    text = str(_record_value(record, "text", "") or "")
    if text_limit is not None:
        text = _clean(text, text_limit)
    authors = _record_value(record, "authors", ()) or ()
    if isinstance(authors, str):
        authors = [authors]
    limitations = _record_value(record, "limitations", ()) or ()
    if isinstance(limitations, str):
        limitations = [limitations]
    page = _record_value(record, "page", _record_value(record, "page_start"))
    return {
        "passage_id": str(_record_value(record, "passage_id", "") or ""),
        "paper_id": str(
            _record_value(record, "paper_id", _record_value(record, "document_id", ""))
            or ""
        ),
        "title": _clean(str(_record_value(record, "title", "") or ""), 300),
        "authors": [str(author) for author in list(authors)[:20]],
        "year": _record_value(record, "year"),
        "section": _clean(str(_record_value(record, "section", "") or ""), 160),
        "page": page,
        "heading_path": _clean(
            str(_record_value(record, "heading_path", "") or ""), 300
        ),
        "previous_chunk_id": str(
            _record_value(
                record,
                "previous_chunk_id",
                _record_value(record, "previous_passage_id", "") or "",
            )
            or ""
        ),
        "next_chunk_id": str(
            _record_value(
                record,
                "next_chunk_id",
                _record_value(record, "next_passage_id", "") or "",
            )
            or ""
        ),
        "text": text,
        "source": _clean(
            str(
                _record_value(
                    record,
                    "source",
                    _record_value(
                        record,
                        "relative_source_path",
                        _record_value(record, "file_name", ""),
                    ),
                )
                or ""
            ),
            300,
        ),
        "limitations": [
            _clean(str(limitation), 240) for limitation in list(limitations)[:8]
        ],
    }


MAX_EVIDENCE_ITEMS = 100


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
        payload: dict[str, Any] = {"count": len(cards), "results": cards}
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
        matches: list[dict[str, Any]] = []
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
        payload: dict[str, Any] = {
            "query": arguments["query"],
            "count": len(matches),
            "matches": matches,
        }
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
        payload: dict[str, Any] = {
            "count": len(files),
            "papers": files,
            "directory": str(directory),
        }
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
    return workspace.resolve(config.literature_root, must_exist=False)


class LiteratureIndexPapersTool(Tool):
    name = "literature.index_papers"
    description = (
        "Parse PDFs under a workspace directory, embed them, and incrementally "
        "update the Qdrant literature collection. Returns bounded progress "
        "statistics and a resumable cursor."
    )
    short_description = "Build/update the local literature RAG index from PDFs."
    exposure = ToolExposure.DEFERRED
    cost_class = "EXPENSIVE"
    namespace = "literature"
    source = "qdrant"
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
            "run_id": {"type": "string"},
            "resume_cursor": {"type": "string"},
            "max_documents": {"type": "integer", "minimum": 1, "maximum": 20},
        },
    }

    def __init__(
        self,
        config: ScientificConfig,
        workspace: Workspace,
        services: Any | None = None,
    ) -> None:
        self._config = config
        self._workspace = workspace
        self._services = services

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        raw_directory = str(arguments.get("directory") or "")
        try:
            root = (
                self._workspace.resolve(raw_directory, must_exist=False)
                if raw_directory
                else _resolve_literature_root(self._config, self._workspace)
            )
        except Exception as exc:
            return _error_result(exc, operation="literature plan")
        try:
            services = (
                self._services
                if self._services is not None
                else build_literature_services(self._config, self._workspace)
            )
            ingestion = _service_value(services, "ingestion")
            if ingestion is None:
                raise RuntimeError("literature ingestion service is unavailable")
            plan = await ingestion.plan(root, self._workspace)
            stats = await ingestion.index_batch(
                plan,
                run_id=str(arguments.get("run_id") or "") or None,
                resume_cursor=str(arguments.get("resume_cursor") or "") or None,
                max_documents=min(
                    int(arguments.get("max_documents", self._config.rag_tool_max_documents)),
                    20,
                ),
            )
        except Exception as exc:
            return _error_result(exc, operation="literature index")
        payload = _stats_payload(stats)
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
    source = "qdrant"
    tags = ("literature", "rag", "search", "hybrid")
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Scientific query, e.g. 'HgTe quantum dot infrared detector responsivity'."},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
    }

    def __init__(
        self,
        config: ScientificConfig,
        workspace: Workspace,
        services: Any | None = None,
    ) -> None:
        self._config = config
        self._workspace = workspace
        self._services = services

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        query = str(arguments["query"])
        top_k = min(
            int(arguments.get("top_k", self._config.literature_search_top_k)), 10
        )
        try:
            services = (
                self._services
                if self._services is not None
                else build_literature_services(self._config, self._workspace)
            )
            retriever = _service_value(services, "retriever")
            if retriever is None:
                raise RuntimeError("literature retriever service is unavailable")
            result = await retriever.search(
                query,
                workspace_id=str(
                    _service_value(
                        services,
                        "workspace_id",
                        _workspace_id(self._workspace),
                    )
                ),
                top_k=top_k,
            )
        except Exception as exc:
            return _error_result(exc, operation="literature search")
        rows: list[dict[str, Any]] = []
        for passage in list(_record_value(result, "passages", ()) or ())[:10]:
            row = {
                "passage_id": str(_record_value(passage, "passage_id", "")),
                "paper_id": str(
                    _record_value(
                        passage,
                        "paper_id",
                        _record_value(passage, "document_id", ""),
                    )
                ),
                "title": _clean(str(_record_value(passage, "title", "")), 300),
                "passage": _clean(
                    str(
                        _record_value(
                            passage,
                            "passage",
                            _record_value(passage, "text", ""),
                        )
                    ),
                    self._config.literature_passage_chars,
                ),
                "section": _clean(str(_record_value(passage, "section", "")), 160),
                "page": _record_value(
                    passage,
                    "page",
                    _record_value(passage, "page_start", None),
                ),
                "score": round(float(_record_value(passage, "score", 0.0)), 4),
                "source": _clean(
                    str(
                        _record_value(
                            passage,
                            "source",
                            _record_value(
                                passage,
                                "relative_source_path",
                                _record_value(passage, "file_name", ""),
                            ),
                        )
                    ),
                    300,
                ),
                "context_before": _clean(
                    str(_record_value(passage, "context_before", "")), 300
                ),
                "context_after": _clean(
                    str(_record_value(passage, "context_after", "")), 300
                ),
            }
            rows.append(row)
        diagnostics_obj = _record_value(result, "diagnostics", None)
        diagnostics = {
            "mode": str(_record_value(diagnostics_obj, "mode", "unknown")),
            "candidate_count": int(
                _record_value(diagnostics_obj, "candidate_count", 0)
            ),
            "reranked": bool(_record_value(diagnostics_obj, "reranked", False)),
            "degraded_reasons": [
                _clean(str(reason), 160)
                for reason in list(
                    _record_value(diagnostics_obj, "degraded_reasons", ()) or ()
                )[:8]
            ],
        }
        payload = {
            "query": _clean(query, 600),
            "count": len(rows),
            "results": rows,
            "diagnostics": diagnostics,
        }
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
    source = "qdrant"
    tags = ("literature", "rag", "read")
    input_schema = {
        "type": "object",
        "properties": {
            "passage_id": {"type": "string", "description": "passage_id from literature.search_passages."},
        },
        "required": ["passage_id"],
    }

    def __init__(
        self,
        config: ScientificConfig,
        workspace: Workspace,
        services: Any | None = None,
    ) -> None:
        self._config = config
        self._workspace = workspace
        self._services = services

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        passage_id = str(arguments["passage_id"])
        try:
            services = (
                self._services
                if self._services is not None
                else build_literature_services(self._config, self._workspace)
            )
            store = _service_value(services, "store")
            if store is None:
                raise RuntimeError("literature store service is unavailable")
            rows = await store.retrieve_passages(
                str(_service_value(services, "workspace_id", _workspace_id(self._workspace))),
                [passage_id],
            )
        except Exception as exc:
            return _error_result(exc, operation="literature read")
        if not rows:
            return ScientificToolResult(
                output=f"passage_not_found: passage not found: {_clean(passage_id, 160)}",
                is_error=True,
                data={"error": "passage_not_found", "passage_id": passage_id},
            )
        row = rows[0]
        payload = _passage_payload(
            row, text_limit=self._config.literature_max_chars
        )
        output_payload = dict(payload)
        return ScientificToolResult(
            output=json.dumps(output_payload, ensure_ascii=False),
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
                "maxItems": MAX_EVIDENCE_ITEMS,
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

    def __init__(
        self,
        config: ScientificConfig,
        workspace: Workspace,
        services: Any | None = None,
    ) -> None:
        self._config = config
        self._workspace = workspace
        self._services = services

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        from photomatagent.scientific.capabilities.literature.evidence import (
            extract_evidence_from_passages,
        )

        passages = list(arguments.get("passages") or [])
        passages = passages[:MAX_EVIDENCE_ITEMS]
        resolved: list[dict[str, Any]] = []
        try:
            services = (
                self._services
                if self._services is not None
                else build_literature_services(self._config, self._workspace)
            )
        except Exception as exc:
            return _error_result(exc, operation="literature evidence")
        for item in passages:
            if not isinstance(item, dict):
                continue
            passage_id = item.get("passage_id")
            if passage_id:
                try:
                    store = _service_value(services, "store")
                    if store is None:
                        raise RuntimeError("literature store service is unavailable")
                    rows = await store.retrieve_passages(
                        str(_service_value(services, "workspace_id", _workspace_id(self._workspace))),
                        [str(passage_id)],
                    )
                except Exception as exc:
                    return _error_result(exc, operation="literature evidence")
                if not rows:
                    resolved.append(
                        {
                            "text": "",
                            "error": f"passage_not_found: {passage_id}",
                        }
                    )
                    continue
                row = rows[0]
                row_data = _passage_payload(row, text_limit=None)
                resolved.append(
                    {
                        "text": _clean(
                            str(row_data.get("text", "")),
                            self._config.literature_max_chars,
                        ),
                        "page": row_data.get("page"),
                        "passage_id": passage_id,
                        "source": row_data.get("source", ""),
                    }
                )
            else:
                resolved.append(
                    {
                        "text": _clean(
                            str(item.get("text") or ""),
                            self._config.literature_max_chars,
                        ),
                        "page": item.get("page"),
                        "source": str(item.get("source") or ""),
                    }
                )
        evidence = extract_evidence_from_passages(resolved)
        bounded_evidence = evidence[:MAX_EVIDENCE_ITEMS]
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
            for item in bounded_evidence
        ]
        payload = {
            "count": len(rows),
            "evidence": rows,
        }
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False),
            data=payload,
            evidence=bounded_evidence,
        )


def literature_pack(config: ScientificConfig, workspace: Workspace) -> CapabilityPack:
    return LiteratureProbe(config, workspace)
