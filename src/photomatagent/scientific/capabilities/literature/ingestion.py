"""Resumable, failure-safe ingestion for the versioned literature store.

Planning is deliberately read-only.  Indexing then handles one document at a
time with a staged passage revision, so a parser/provider failure cannot erase
the last ready revision of an otherwise usable document.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import os
import re
import stat
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from photomatagent.scientific.capabilities.literature.models import (
    DocumentManifest,
    DocumentStatus,
    IngestState,
    PaperRecord,
    validate_relative_source_path,
)
from photomatagent.scientific.capabilities.literature.parser import (
    parse_pdf,
    passage_points_for,
)
from photomatagent.scientific.capabilities.literature.providers.base import (
    EmbeddingProvider,
    RagProviderError,
    validate_vectors,
)
from photomatagent.scientific.capabilities.literature.qdrant_store import (
    CollectionGeneration,
    document_id_for,
    workspace_id_for,
)


MAX_BATCH_DOCUMENTS = 20
MAX_ERRORS = 20
MAX_ERROR_CHARS = 512
_SECRET_RE = re.compile(
    r"(?i)(api[_ -]?key|access[_ -]?token|authorization|bearer|password|secret)"
    r"\s*[:=]\s*[^\s,;]+"
)


class RagIngestionError(RuntimeError):
    """Stable, secret-free ingestion boundary failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class PlanKind(str, Enum):
    NEW = "new"
    CHANGED = "changed"
    UNCHANGED = "unchanged"
    DELETED = "deleted"
    RETRY_FAILED = "retry_failed"


@dataclass(frozen=True)
class IngestionPlanItem:
    document_id: str
    relative_source_path: str
    content_sha256: str | None
    kind: PlanKind

    def __post_init__(self) -> None:
        validate_relative_source_path(self.relative_source_path)
        object.__setattr__(self, "kind", PlanKind(self.kind))


@dataclass(frozen=True)
class IngestionPlan:
    workspace_id: str
    generation: CollectionGeneration
    items: tuple[IngestionPlanItem, ...]
    # ``source_root`` is intentionally local-only and never persisted in a
    # Qdrant payload.  It lets a plan be handed to a later bounded batch.
    source_root: Path | None = None
    relative_root: str = "."
    complete: bool = True

    def __post_init__(self) -> None:
        _require_workspace_id(self.workspace_id)
        object.__setattr__(self, "items", tuple(self.items))


@dataclass(frozen=True)
class IngestionStats:
    run_id: str
    discovered: int
    unchanged: int
    indexed: int
    failed: int
    deleted: int
    chunks: int
    staged_cleanup: int
    next_cursor: str | None
    complete: bool
    errors: tuple[str, ...]
    retryable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "errors",
            _bounded_errors(self.errors),
        )


@dataclass(frozen=True)
class IngestionRunState:
    run_id: str
    workspace_id: str
    generation_fingerprint: str
    relative_root: str
    cursor: str | None
    status: str
    stats: IngestionStats


def _require_workspace_id(workspace_id: str) -> str:
    if not isinstance(workspace_id, str) or not workspace_id.strip():
        raise ValueError("workspace_id must be a non-empty string")
    return workspace_id


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _enumerate_pdf_files(root: Path) -> list[Path]:
    """Recursively enumerate regular PDFs without suppressing filesystem errors.

    ``Path.rglob`` intentionally suppresses some directory traversal errors,
    which is unsafe for deletion synchronization: an unreadable nested folder
    could look like a successfully empty folder.  ``scandir`` lets us fail the
    complete walk as soon as any child directory/stat operation is denied.
    """
    result: list[Path] = []

    def visit(directory: Path) -> None:
        with os.scandir(directory) as entries:
            children = sorted(entries, key=lambda entry: entry.name)
        for entry in children:
            path = Path(entry.path)
            if entry.is_symlink():
                continue
            entry_stat = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(entry_stat.st_mode):
                visit(path)
                continue
            if stat.S_ISREG(entry_stat.st_mode) and path.suffix == ".pdf":
                result.append(path)

    visit(root)
    return result


def _redact_error(exc: BaseException, *, relative_path: str = "") -> str:
    message = str(exc).replace("\x00", " ")
    message = _SECRET_RE.sub(lambda match: f"{match.group(1)}=[redacted]", message)
    # Absolute paths can contain workspace names or query fragments.  Keep
    # only the relative source path supplied by the plan in diagnostics.
    message = re.sub(r"(?:[A-Za-z]:[\\/]|/)[^\s,;]+", "[path]", message)
    prefix = f"{relative_path}: " if relative_path else ""
    result = f"{prefix}{type(exc).__name__}: {message}".strip()
    return result[:MAX_ERROR_CHARS]


def _bounded_errors(errors: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(item)[:MAX_ERROR_CHARS] for item in list(errors)[-MAX_ERRORS:])


def _generation_fingerprint(generation: Any) -> str:
    fingerprint = str(getattr(generation, "fingerprint", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise RagIngestionError(
            "generation_invalid", "collection generation fingerprint is unavailable"
        )
    return fingerprint


def _workspace_context(root: Path | str, workspace: Any) -> tuple[Path, str, Path | None, str]:
    """Resolve a source root and return (root, id, workspace_root, relative_root)."""
    workspace_root = getattr(workspace, "root", None)
    if workspace_root is not None:
        workspace_root = Path(workspace_root).expanduser().resolve()
        workspace_id = workspace_id_for(workspace_root)
        try:
            resolved = workspace.resolve(str(root), must_exist=False)
        except Exception as exc:
            raise RagIngestionError("source_root_invalid", "source root is outside workspace") from exc
        resolved = Path(resolved).resolve(strict=False)
        try:
            relative_root = resolved.relative_to(workspace_root).as_posix() or "."
        except ValueError as exc:
            raise RagIngestionError("source_root_invalid", "source root is outside workspace") from exc
        return resolved, workspace_id, workspace_root, relative_root

    if isinstance(workspace, Path):
        workspace_root = workspace.expanduser().resolve()
        workspace_id = workspace_id_for(workspace_root)
        candidate = Path(root).expanduser()
        if not candidate.is_absolute():
            candidate = workspace_root / candidate
        resolved = candidate.resolve(strict=False)
        try:
            relative_root = resolved.relative_to(workspace_root).as_posix() or "."
        except ValueError as exc:
            raise RagIngestionError(
                "source_root_invalid", "source root is outside workspace"
            ) from exc
        return resolved, workspace_id, workspace_root, relative_root
    else:
        workspace_id = _require_workspace_id(str(workspace))
    resolved = Path(root).expanduser().resolve(strict=False)
    return resolved, workspace_id, workspace_root, "."


def _relative_source(path: Path, root: Path, workspace_root: Path | None) -> str:
    base = workspace_root or root
    try:
        relative = path.relative_to(base)
    except ValueError:
        relative = path.relative_to(root)
    return relative.as_posix()


def _manifest_kind(
    manifest: DocumentManifest | None,
    content_sha256: str,
) -> PlanKind:
    if manifest is None:
        return PlanKind.NEW
    if manifest.status is DocumentStatus.FAILED and manifest.content_sha256 == content_sha256:
        return PlanKind.RETRY_FAILED
    if manifest.status is DocumentStatus.READY and manifest.content_sha256 == content_sha256:
        return PlanKind.UNCHANGED
    return PlanKind.CHANGED


def _stats_with(
    stats: IngestionStats,
    *,
    cursor: str | None,
    complete: bool,
    errors: Sequence[str] | None = None,
    retryable: bool | None = None,
    indexed_add: int = 0,
    failed_add: int = 0,
    deleted_add: int = 0,
    chunks_add: int = 0,
    cleanup_add: int = 0,
) -> IngestionStats:
    all_errors = list(stats.errors)
    if errors:
        all_errors.extend(errors)
    return replace(
        stats,
        indexed=stats.indexed + indexed_add,
        failed=stats.failed + failed_add,
        deleted=stats.deleted + deleted_add,
        chunks=stats.chunks + chunks_add,
        staged_cleanup=stats.staged_cleanup + cleanup_add,
        next_cursor=cursor,
        complete=complete,
        errors=_bounded_errors(all_errors),
        retryable=stats.retryable if retryable is None else retryable,
    )


class LiteratureIngestionService:
    """Compose parser, embedding provider, and the narrow literature store."""

    def __init__(
        self,
        store: Any,
        embedder: EmbeddingProvider,
        *,
        batch_size: int = 128,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.store = store
        self.embedder = embedder
        self.batch_size = batch_size

    async def _generation(self) -> CollectionGeneration:
        resolver = getattr(self.store, "resolve_current_generation", None)
        if resolver is not None:
            generation = resolver()
            if inspect.isawaitable(generation):
                generation = await generation
        else:
            generation = getattr(
                self.store,
                "generation",
                getattr(self.store, "current_generation", None),
            )
        if generation is None:
            raise RagIngestionError(
                "collection_missing", "current literature collection is unavailable"
            )
        _generation_fingerprint(generation)
        return generation

    async def plan(self, root: Path | str, workspace: Any) -> IngestionPlan:
        """Build a complete, read-only plan for the current source tree."""
        resolved_root, workspace_id, workspace_root, relative_root = _workspace_context(
            root, workspace
        )
        if not resolved_root.exists() or not resolved_root.is_dir():
            raise RagIngestionError("source_root_missing", "source root does not exist")
        generation = await self._generation()

        # Enumerate and hash every source file before consulting missing
        # manifests.  Deletion candidates are appended only after this phase,
        # so a failed/incomplete source walk can never trigger destructive sync.
        discovered: list[tuple[str, Path, str]] = []
        try:
            for path in _enumerate_pdf_files(resolved_root):
                relative_path = _relative_source(path, resolved_root, workspace_root)
                discovered.append((relative_path, path, _sha256(path)))
        except OSError as exc:
            raise RagIngestionError("source_enumeration_failed", "source enumeration failed") from exc
        discovered.sort(key=lambda item: item[0])

        try:
            manifests = await self.store.list_document_manifests(workspace_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise RagIngestionError("manifest_read_failed", _redact_error(exc)) from exc

        items: list[IngestionPlanItem] = []
        seen_ids: set[str] = set()
        for relative_path, _path, content_sha256 in discovered:
            document_id = document_id_for(workspace_id, relative_path)
            manifest = manifests.get(document_id)
            if manifest is None:
                manifest = next(
                    (
                        candidate
                        for candidate in manifests.values()
                        if candidate.workspace_id == workspace_id
                        and candidate.relative_source_path
                        in {relative_path, f"{resolved_root.name}/{relative_path}"}
                    ),
                    None,
                )
                if manifest is not None:
                    relative_path = manifest.relative_source_path
                    document_id = manifest.document_id
                    seen_ids.add(manifest.document_id)
            seen_ids.add(document_id)
            items.append(
                IngestionPlanItem(
                    document_id=document_id,
                    relative_source_path=relative_path,
                    content_sha256=content_sha256,
                    kind=_manifest_kind(manifest, content_sha256),
                )
            )
        # A successful full enumeration is the safety gate for deletions.
        for document_id, manifest in manifests.items():
            if manifest.workspace_id != workspace_id:
                continue
            if document_id in seen_ids or manifest.status is DocumentStatus.DELETED:
                continue
            items.append(
                IngestionPlanItem(
                    document_id=document_id,
                    relative_source_path=manifest.relative_source_path,
                    content_sha256=None,
                    kind=PlanKind.DELETED,
                )
            )
        items.sort(key=lambda item: item.relative_source_path)
        return IngestionPlan(
            workspace_id=workspace_id,
            generation=generation,
            items=tuple(items),
            source_root=resolved_root,
            relative_root=relative_root,
            complete=True,
        )

    def _source_path(self, plan: IngestionPlan, item: IngestionPlanItem) -> Path:
        if plan.source_root is None:
            return Path(plan.relative_root) / item.relative_source_path
        root = Path(plan.source_root).resolve()
        relative_path = item.relative_source_path
        if plan.relative_root not in {"", "."}:
            prefix = plan.relative_root.rstrip("/") + "/"
            if relative_path.startswith(prefix):
                relative_path = relative_path[len(prefix) :]
        elif relative_path.startswith(root.name + "/"):
            relative_path = relative_path[len(root.name) + 1 :]
        candidate = (root / relative_path).resolve()
        if root != candidate and root not in candidate.parents:
            raise RagIngestionError("source_path_invalid", "source path escapes source root")
        return candidate

    def _initial_stats(self, run_id: str, plan: IngestionPlan) -> IngestionStats:
        return IngestionStats(
            run_id=run_id,
            discovered=sum(item.kind is not PlanKind.DELETED for item in plan.items),
            unchanged=sum(item.kind is PlanKind.UNCHANGED for item in plan.items),
            indexed=0,
            failed=0,
            deleted=0,
            chunks=0,
            staged_cleanup=0,
            next_cursor=None,
            complete=False,
            errors=(),
        )

    async def _persist_run(self, run: IngestionRunState) -> None:
        await self.store.upsert_ingestion_run(run, wait=True)

    async def _recover_run(
        self,
        plan: IngestionPlan,
        run_id: str,
        resume_cursor: str | None,
    ) -> tuple[IngestionRunState, str | None]:
        existing = None
        getter = getattr(self.store, "get_ingestion_run", None)
        if getter is not None:
            existing = getter(run_id, plan.workspace_id)
            if inspect.isawaitable(existing):
                existing = await existing
        if existing is not None:
            if (
                existing.workspace_id != plan.workspace_id
                or existing.generation_fingerprint != plan.generation.fingerprint
                or existing.relative_root != plan.relative_root
            ):
                raise RagIngestionError(
                    "run_context_mismatch",
                    "ingestion run does not match workspace, generation, or source root",
                )
            stats = existing.stats
            cursor = resume_cursor if resume_cursor is not None else existing.cursor
            run = replace(existing, cursor=cursor, status="running", stats=replace(stats, next_cursor=cursor, complete=False))
            return run, cursor
        stats = self._initial_stats(run_id, plan)
        cursor = resume_cursor
        run = IngestionRunState(
            run_id=run_id,
            workspace_id=plan.workspace_id,
            generation_fingerprint=plan.generation.fingerprint,
            relative_root=plan.relative_root,
            cursor=cursor,
            status="running",
            stats=replace(stats, next_cursor=cursor),
        )
        return run, cursor

    async def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        all_vectors: list[list[float]] = []
        provider_batch = getattr(self.embedder, "batch_size", None)
        if not isinstance(provider_batch, int) or provider_batch < 1:
            provider_batch = getattr(self.embedder, "_batch_size", self.batch_size)
        if not isinstance(provider_batch, int) or provider_batch < 1:
            provider_batch = self.batch_size
        for start in range(0, len(texts), min(provider_batch, self.batch_size)):
            result = await self.embedder.embed_documents(texts[start : start + min(provider_batch, self.batch_size)])
            all_vectors.extend(result)
        expected_dimension = getattr(getattr(self.embedder, "identity", None), "dimension", None)
        try:
            return validate_vectors(all_vectors, len(texts), expected_dimension)
        except RagProviderError:
            raise
        except Exception as exc:
            raise RagProviderError("embedding_invalid_response", "embedding response is invalid") from exc

    @staticmethod
    def _manifest_for(
        item: IngestionPlanItem,
        workspace_id: str,
        model_fingerprint: str,
        *,
        status: DocumentStatus,
        record: PaperRecord | None = None,
        chunk_count: int = 0,
        last_error: str = "",
    ) -> DocumentManifest:
        return DocumentManifest(
            schema_version=1,
            record_type="document",
            workspace_id=workspace_id,
            document_id=item.document_id,
            relative_source_path=item.relative_source_path,
            file_name=Path(item.relative_source_path).name,
            content_sha256=item.content_sha256 or (record.sha256 if record else "0" * 64),
            status=status,
            title=record.title if record else "",
            authors=tuple(record.authors) if record else (),
            year=record.year if record else None,
            num_pages=record.num_pages if record else 0,
            chunk_count=chunk_count,
            model_fingerprint=model_fingerprint,
            indexed_at=datetime.now(timezone.utc) if status is DocumentStatus.READY else None,
            last_error=last_error,
        )

    async def _mark_failed(
        self,
        plan: IngestionPlan,
        item: IngestionPlanItem,
        exc: BaseException,
        stats: IngestionStats,
    ) -> IngestionStats:
        diagnostic = _redact_error(exc, relative_path=item.relative_source_path)
        try:
            await self.store.upsert_document(
                self._manifest_for(
                    item,
                    plan.workspace_id,
                    plan.generation.fingerprint,
                    status=DocumentStatus.FAILED,
                    last_error=diagnostic,
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception as manifest_exc:
            diagnostic = _bounded_errors((diagnostic, _redact_error(manifest_exc))) [-1]
        return _stats_with(
            stats,
            cursor=item.relative_source_path,
            complete=False,
            errors=(diagnostic,),
            failed_add=1,
            retryable=True,
        )

    async def _process_item(
        self,
        plan: IngestionPlan,
        item: IngestionPlanItem,
        stats: IngestionStats,
    ) -> tuple[IngestionStats, bool]:
        if item.kind is PlanKind.DELETED:
            await self.store.delete_staged_revisions(
                item.document_id,
                keep_revision=None,
                workspace_id=plan.workspace_id,
            )
            await self.store.delete_document_passages(
                item.document_id, workspace_id=plan.workspace_id
            )
            await self.store.upsert_document(
                self._manifest_for(
                    item,
                    plan.workspace_id,
                    plan.generation.fingerprint,
                    status=DocumentStatus.DELETED,
                )
            )
            return (
                _stats_with(
                    stats,
                    cursor=item.relative_source_path,
                    complete=False,
                    deleted_add=1,
                ),
                False,
            )

        path = self._source_path(plan, item)
        if not path.is_file() or path.is_symlink():
            raise RagIngestionError("source_missing", "source PDF is missing")
        revision = item.content_sha256 or _sha256(path)
        if _sha256(path) != revision:
            raise RagIngestionError("source_changed", "source PDF changed after planning")

        cleanup_count = await self.store.delete_staged_revisions(
            item.document_id,
            keep_revision=revision,
            workspace_id=plan.workspace_id,
        )
        record, passages = parse_pdf(path)
        if not passages:
            raise RagIngestionError("no_passages", "PDF produced no text passages")
        texts = [passage.text for passage in passages]
        vectors = await self._embed(texts)
        points = passage_points_for(
            record,
            passages,
            plan.workspace_id,
            item.relative_source_path,
            revision,
            plan.generation.fingerprint,
        )
        if len(points) != len(vectors):
            raise RagProviderError("vector_count_mismatch", "embedding count does not match passages")
        points = [replace(point, dense=tuple(vector)) for point, vector in zip(points, vectors)]
        await self.store.upsert_passages(
            points,
            batch_size=self.batch_size,
            workspace_id=plan.workspace_id,
        )
        staged_count = await self.store.count_revision(
            item.document_id,
            revision,
            IngestState.STAGED,
            workspace_id=plan.workspace_id,
        )
        if staged_count != len(points):
            raise RagIngestionError("staged_count_mismatch", "staged passage count is incomplete")
        # Ready first avoids a no-result window if old revision cleanup fails.
        await self.store.set_revision_state(
            item.document_id,
            revision,
            IngestState.READY,
            workspace_id=plan.workspace_id,
        )
        cleanup_error: str | None = None
        try:
            await self.store.delete_other_revisions(
                item.document_id,
                revision,
                workspace_id=plan.workspace_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            cleanup_error = _redact_error(exc, relative_path=item.relative_source_path)
        await self.store.upsert_document(
            self._manifest_for(
                item,
                plan.workspace_id,
                plan.generation.fingerprint,
                status=DocumentStatus.READY,
                record=record,
                chunk_count=len(points),
            )
        )
        errors = (cleanup_error,) if cleanup_error else ()
        return (
            _stats_with(
                stats,
                cursor=item.relative_source_path,
                complete=False,
                errors=errors,
                retryable=bool(cleanup_error) or stats.retryable,
                indexed_add=1,
                chunks_add=len(points),
                cleanup_add=cleanup_count,
            ),
            cleanup_error is not None,
        )

    async def index_batch(
        self,
        plan: IngestionPlan,
        *,
        run_id: str | None = None,
        resume_cursor: str | None = None,
        max_documents: int = MAX_BATCH_DOCUMENTS,
    ) -> IngestionStats:
        """Process a bounded number of non-unchanged plan items."""
        _require_workspace_id(plan.workspace_id)
        if max_documents < 1:
            raise ValueError("max_documents must be positive")
        limit = min(int(max_documents), MAX_BATCH_DOCUMENTS)
        if plan.generation.fingerprint != _generation_fingerprint(plan.generation):
            raise RagIngestionError("generation_invalid", "plan generation is invalid")
        run_id = run_id or uuid.uuid4().hex
        run, cursor = await self._recover_run(plan, run_id, resume_cursor)
        stats = run.stats
        await self._persist_run(run)

        candidates = [
            item
            for item in plan.items
            if item.kind is not PlanKind.UNCHANGED
            and (plan.complete or item.kind is not PlanKind.DELETED)
        ]
        if cursor is not None:
            candidates = [item for item in candidates if item.relative_source_path > cursor]
        selected = candidates[:limit]
        processed = 0
        progress_cursor = cursor
        retry_cursor: str | None = None
        retry_seen = False
        for item in selected:
            try:
                stats, retry_item = await self._process_item(plan, item, stats)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                stats = await self._mark_failed(plan, item, exc, stats)
                retry_item = False
            processed += 1
            if retry_item:
                retry_seen = True
                if retry_cursor is None:
                    retry_cursor = progress_cursor
            else:
                progress_cursor = item.relative_source_path
            next_candidates = [
                candidate
                for candidate in candidates
                if candidate.relative_source_path > item.relative_source_path
            ]
            persisted_cursor = retry_cursor if retry_seen else progress_cursor
            complete = plan.complete and not next_candidates and not retry_seen
            stats = replace(stats, next_cursor=None if complete else persisted_cursor)
            run = IngestionRunState(
                run_id=run_id,
                workspace_id=plan.workspace_id,
                generation_fingerprint=plan.generation.fingerprint,
                relative_root=plan.relative_root,
                cursor=stats.next_cursor,
                status="retryable" if stats.retryable else ("complete" if complete else "running"),
                stats=stats,
            )
            await self._persist_run(run)

        complete = plan.complete and not [
            item
            for item in candidates
            if processed == 0 or item.relative_source_path > (stats.next_cursor or "")
        ]
        # With no selected work, or after the final selected item, cursor is
        # cleared only when every non-unchanged item has been handled.
        if retry_seen:
            complete = False
        elif not candidates:
            complete = plan.complete
        elif selected and selected[-1].relative_source_path == candidates[-1].relative_source_path:
            complete = plan.complete
        elif not selected:
            complete = False
        if complete:
            stats = replace(stats, complete=True, next_cursor=None)
        status = "retryable" if stats.retryable else ("complete" if stats.complete else "running")
        final_run = IngestionRunState(
            run_id=run_id,
            workspace_id=plan.workspace_id,
            generation_fingerprint=plan.generation.fingerprint,
            relative_root=plan.relative_root,
            cursor=stats.next_cursor,
            status=status,
            stats=stats,
        )
        await self._persist_run(final_run)
        return stats


__all__ = [
    "IngestionPlan",
    "IngestionPlanItem",
    "IngestionRunState",
    "IngestionStats",
    "LiteratureIngestionService",
    "PlanKind",
    "RagIngestionError",
]
