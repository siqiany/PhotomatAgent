"""Streaming, resumable ingestion for the local abstract SQLite corpus.

The abstract corpus is intentionally independent from the PDF corpus.  A row
is represented by one document and one passage, and the SQLite reader keeps
only a bounded page of rows in memory.  All writes go through the same narrow
literature-store contracts used by PDF ingestion.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote

from photomatagent.scientific.capabilities.literature.models import (
    DocumentManifest,
    DocumentStatus,
    IngestState,
    LITERATURE_CHUNK_SCHEMA_VERSION,
    LiteratureSourceKind,
    PassagePoint,
    validate_relative_source_path,
)
from photomatagent.scientific.capabilities.literature.providers.base import (
    RagProviderError,
    validate_vectors,
)
from photomatagent.scientific.capabilities.literature.qdrant_store import (
    collection_fingerprint,
    document_id_for,
    passage_id_for,
    workspace_id_for,
)


MAX_ABSTRACT_BATCH = 20
MAX_ERRORS = 20
MAX_ERROR_CHARS = 512
_ABSTRACT_COLUMNS = (
    "paper_key",
    "title",
    "abstract",
    "authors",
    "publication_year",
    "doi",
    "pmid",
    "pmcid",
    "journal",
    "relevance_tier",
    "retrieved_at",
)
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "paper_key": ("paper_key",),
    "title": ("title", "paper_title"),
    "abstract": ("abstract", "abstract_text"),
    "authors": ("authors", "authors_json"),
    "publication_year": ("publication_year", "year"),
    "doi": ("doi",),
    "pmid": ("pmid",),
    "pmcid": ("pmcid",),
    "journal": ("journal",),
    "relevance_tier": ("relevance_tier",),
    "retrieved_at": ("retrieved_at",),
}
_REVISION_FIELDS = (
    "paper_key",
    "title",
    "abstract",
    "authors",
    "publication_year",
    "doi",
    "pmid",
    "pmcid",
    "journal",
    "relevance_tier",
)
_SQL_KEY_VALID = 'photomat_abstract_key_valid("paper_key") = 1'
_SQL_KEY_INVALID = 'photomat_abstract_key_valid("paper_key") = 0'


class AbstractIngestionError(RuntimeError):
    """Stable, bounded abstract-ingestion failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


def _file_signature(path: Path) -> tuple[int, int, int, int] | None:
    """Return a cheap identity for one immutable source file."""
    try:
        metadata = path.stat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise AbstractIngestionError(
            "source_snapshot_unavailable",
            "abstract SQLite source metadata cannot be inspected",
        ) from exc
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
    )


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _authors(value: Any) -> tuple[str, ...]:
    """Normalise the common SQLite author encodings into a tuple."""
    if value is None:
        return ()
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return tuple(item for item in (_text(item) for item in value) if item)
    value_text = _text(value)
    if not value_text:
        return ()
    if value_text.startswith("["):
        try:
            decoded = json.loads(value_text)
        except (TypeError, ValueError, json.JSONDecodeError):
            decoded = None
        if isinstance(decoded, list):
            return tuple(item for item in (_text(item) for item in decoded) if item)
    if ";" in value_text:
        return tuple(item for item in (_text(item) for item in value_text.split(";")) if item)
    return (value_text,)


def _year(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class AbstractSourceRecord:
    """One source row from the abstract corpus."""

    paper_key: str
    title: str
    abstract: str
    authors: tuple[str, ...]
    publication_year: int | None
    doi: str = ""
    pmid: str = ""
    pmcid: str = ""
    journal: str = ""
    relevance_tier: str = ""
    # Retrieval time describes collection provenance, not indexed content.
    retrieved_at: str = ""

    def __post_init__(self) -> None:
        # Keep the exact SQLite ordering key.  Stripping it before persisting
        # the cursor can make a later raw key unreachable through keyset SQL.
        object.__setattr__(self, "paper_key", _raw_key(self.paper_key))
        object.__setattr__(self, "title", _text(self.title))
        object.__setattr__(self, "abstract", _text(self.abstract))
        object.__setattr__(self, "authors", _authors(self.authors))
        object.__setattr__(self, "publication_year", _year(self.publication_year))
        for name in ("doi", "pmid", "pmcid", "journal", "relevance_tier", "retrieved_at"):
            object.__setattr__(self, name, _text(getattr(self, name)))


def canonical_abstract_revision(record: AbstractSourceRecord) -> str:
    """Hash fields that affect indexed text or source display.

    ``retrieved_at`` is deliberately excluded: refreshing metadata collection
    time must not force a new vector for an unchanged abstract.
    """
    values = {name: _value(record, name, "") for name in _REVISION_FIELDS}
    values["authors"] = list(_authors(values["authors"]))
    encoded = json.dumps(
        values,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def abstract_document_id_for(
    workspace_id: str, relative_source_path: str, paper_key: str
) -> str:
    """Return a stable ID scoped by workspace, source path, and source key."""
    validate_relative_source_path(relative_source_path)
    key = _raw_key(paper_key)
    if not key:
        raise ValueError("paper_key must be a non-empty string")
    # Reuse the established document namespace while keeping the SQLite key
    # distinct from the database document itself and from every PDF path.  A
    # digest prevents punctuation or slash characters in a DOI/key from being
    # interpreted as path components by the relative-path validator.
    key_digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return document_id_for(
        workspace_id, f"{relative_source_path}::abstract::{key_digest}"
    )


def _normalised_text_sha256(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).casefold().encode("utf-8")).hexdigest()


def _redact_error(exc: BaseException) -> str:
    value = str(exc).replace("\x00", " ")
    # Keep diagnostics bounded and avoid persisting obvious credentials or
    # absolute source paths in Qdrant control records.
    import re

    value = re.sub(
        r"(?i)(api[_ -]?key|access[_ -]?token|authorization|bearer|password|secret)"
        r"\s*[:=]\s*[^\s,;]+",
        lambda match: f"{match.group(1)}=[redacted]",
        value,
    )
    value = re.sub(r"(?:[A-Za-z]:[\\/]|/)[^\s,;]+", "[path]", value)
    return f"{type(exc).__name__}: {value}"[:MAX_ERROR_CHARS]


def _bounded_errors(errors: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(item)[:MAX_ERROR_CHARS] for item in list(errors)[-MAX_ERRORS:])


def _value(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _resolve_workspace_root(value: Any) -> Path:
    root = value if isinstance(value, (str, Path)) else getattr(value, "root", value)
    if root is None:
        raise ValueError("workspace root is required")
    resolved = Path(root).expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError("workspace root is not a directory")
    return resolved


def _path_inside(root: Path, path: Path) -> bool:
    return path == root or root in path.parents


def _raw_key(value: Any) -> str:
    """Convert a SQLite key to text without changing its ordering value."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _sqlite_key_valid(value: Any) -> int:
    """Apply Python Unicode whitespace semantics inside SQLite predicates."""
    return int(bool(_raw_key(value).strip()))


class SQLiteAbstractReader:
    """Read the ``papers`` table in SQLite read-only mode using keyset pages."""

    def __init__(
        self,
        path: Path | str,
        workspace_root: Path | str | Any | None = None,
        *,
        workspace: Any | None = None,
    ) -> None:
        if workspace_root is None:
            workspace_root = workspace
        self.workspace_root = _resolve_workspace_root(workspace_root)
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        self.path = candidate.resolve()
        if not _path_inside(self.workspace_root, self.path):
            raise ValueError("abstract SQLite database is outside workspace")
        if not self.path.is_file():
            raise ValueError("abstract SQLite database does not exist")
        self._connection: sqlite3.Connection | None = None
        self._snapshot_signature: tuple[
            tuple[int, int, int, int] | None,
            tuple[int, int, int, int] | None,
        ] | None = None
        try:
            self._check_wal()
            uri = "file:" + quote(str(self.path), safe="/:\\") + "?mode=ro"
            self._connection = sqlite3.connect(uri, uri=True)
            self._connection.row_factory = sqlite3.Row
            self._connection.create_function(
                "photomat_abstract_key_valid",
                1,
                _sqlite_key_valid,
                deterministic=True,
            )
            self._columns = self._validate_schema()
            # Hold one read transaction for the life of the reader.  A valid
            # checkpointed database therefore presents one logical snapshot
            # across all keyset pages; any source replacement is detected by
            # the cheap file-signature check before the next page.
            self._connection.execute("BEGIN")
            self._snapshot_signature = self.snapshot_signature()
            self._check_wal()
        except (OSError, sqlite3.Error, ValueError) as exc:
            try:
                if self._connection is not None:
                    self._connection.close()
            except (AttributeError, sqlite3.Error):
                pass
            if isinstance(exc, ValueError):
                raise
            raise ValueError("abstract SQLite database cannot be opened read-only") from exc
        except AbstractIngestionError:
            try:
                if self._connection is not None:
                    self._connection.close()
            except (AttributeError, sqlite3.Error):
                pass
            raise

    @property
    def wal_path(self) -> Path:
        """Return the sidecar WAL path associated with the source database."""
        return Path(f"{self.path}-wal")

    def _check_wal(self) -> None:
        try:
            wal_size = self.wal_path.stat().st_size
        except FileNotFoundError:
            return
        except OSError as exc:
            raise AbstractIngestionError(
                "source_snapshot_unavailable",
                "SQLite WAL metadata cannot be inspected; checkpoint or copy the database first",
            ) from exc
        if wal_size > 0:
            raise AbstractIngestionError(
                "source_wal_pending",
                "SQLite database has a non-empty WAL; checkpoint it or copy a checkpointed database before importing",
            )

    def _connection_or_error(self) -> sqlite3.Connection:
        connection = self._connection
        if connection is None:
            raise AbstractIngestionError(
                "source_snapshot_unavailable",
                "abstract SQLite reader is not open",
            )
        return connection

    def snapshot_signature(
        self,
    ) -> tuple[
        tuple[int, int, int, int] | None,
        tuple[int, int, int, int] | None,
    ]:
        """Return the cheap immutable main/WAL metadata snapshot."""
        return (_file_signature(self.path), _file_signature(self.wal_path))

    def validate_snapshot(
        self,
        expected: tuple[
            tuple[int, int, int, int] | None,
            tuple[int, int, int, int] | None,
        ] | None = None,
    ) -> None:
        """Fail closed when the checkpointed source changes between pages."""
        self._check_wal()
        current = self.snapshot_signature()
        expected_signature = expected if expected is not None else self._snapshot_signature
        if expected_signature is not None and current != expected_signature:
            raise AbstractIngestionError(
                "source_changed",
                "abstract SQLite source changed during import; start a new run from one checkpointed snapshot",
            )

    def _validate_schema(self) -> set[str]:
        connection = self._connection_or_error()
        try:
            table = connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name = 'papers'"
            ).fetchone()
            if table is None:
                raise ValueError("abstract SQLite database is missing the papers table")
            columns = {
                str(row[1]).casefold()
                for row in connection.execute('PRAGMA table_info("papers")')
            }
        except sqlite3.Error as exc:
            raise ValueError("abstract SQLite database schema cannot be inspected") from exc
        required = {
            canonical
            for canonical in ("paper_key", "title", "abstract")
            if not any(alias in columns for alias in _COLUMN_ALIASES[canonical])
        }
        missing = sorted(required)
        if missing:
            raise ValueError(
                "abstract SQLite papers table is missing required columns: "
                + ", ".join(missing)
            )
        return columns

    def _select_sql(self) -> str:
        expressions = []
        for name in _ABSTRACT_COLUMNS:
            selected = next(
                (alias for alias in _COLUMN_ALIASES[name] if alias.casefold() in self._columns),
                None,
            )
            if selected is None:
                expressions.append(f'NULL AS "{name}"')
            else:
                expressions.append(f'"{selected}" AS "{name}"')
        return ", ".join(expressions)

    @staticmethod
    def _record(row: sqlite3.Row | Mapping[str, Any]) -> AbstractSourceRecord:
        def value(name: str) -> Any:
            if isinstance(row, Mapping):
                return row.get(name)
            return row[name]

        return AbstractSourceRecord(
            paper_key=value("paper_key"),
            title=value("title"),
            abstract=value("abstract"),
            authors=_authors(value("authors")),
            publication_year=_year(value("publication_year")),
            doi=_text(value("doi")),
            pmid=_text(value("pmid")),
            pmcid=_text(value("pmcid")),
            journal=_text(value("journal")),
            relevance_tier=_text(value("relevance_tier")),
            retrieved_at=_text(value("retrieved_at")),
        )

    def count(self) -> int:
        self.validate_snapshot()
        row = self._connection_or_error().execute('SELECT COUNT(*) FROM "papers"').fetchone()
        return int(row[0]) if row is not None else 0

    def count_invalid_keys(self) -> int:
        self.validate_snapshot()
        row = self._connection_or_error().execute(
            f'SELECT COUNT(*) FROM "papers" WHERE {_SQL_KEY_INVALID}'
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def source_identity(self) -> str:
        """Hash database bytes incrementally for fail-closed resume checks."""
        self.validate_snapshot()
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        self.validate_snapshot()
        return digest.hexdigest()

    def fetch_after(
        self, cursor: str | None, *, limit: int
    ) -> tuple[AbstractSourceRecord, ...]:
        if limit < 1:
            raise ValueError("limit must be positive")
        self.validate_snapshot()
        statement = (
            f'SELECT {self._select_sql()} FROM "papers" '
            f'WHERE {_SQL_KEY_VALID} AND "paper_key" > ? '
            'ORDER BY "paper_key" LIMIT ?'
        )
        try:
            result = self._connection_or_error().execute(
                statement, (cursor or "", int(limit))
            )
            rows = result.fetchmany(int(limit))
        except sqlite3.Error as exc:
            raise AbstractIngestionError(
                "source_query_failed", "abstract SQLite query failed"
            ) from exc
        self.validate_snapshot()
        return tuple(self._record(row) for row in rows)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()

    def __enter__(self) -> "SQLiteAbstractReader":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class AbstractIngestionStats:
    """Durable counters carried by an abstract ingestion-run record."""

    run_id: str
    discovered: int
    processed: int = 0
    indexed: int = 0
    unchanged: int = 0
    failed: int = 0
    skipped_empty: int = 0
    skipped_invalid_key: int = 0
    passages: int = 0
    next_cursor: str | None = None
    complete: bool = False
    errors: tuple[str, ...] = ()
    retryable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "errors", _bounded_errors(self.errors))

    @property
    def deleted(self) -> int:
        return 0

    @property
    def chunks(self) -> int:
        return self.passages

    @property
    def staged_cleanup(self) -> int:
        return 0


@dataclass(frozen=True, slots=True)
class AbstractIngestionRunState:
    """Durable abstract-stage context used for resume validation."""

    run_id: str
    workspace_id: str
    generation_fingerprint: str
    source_identity: str
    source_path: str
    cursor: str | None
    status: str
    stats: AbstractIngestionStats
    source_kind: LiteratureSourceKind = LiteratureSourceKind.ABSTRACT
    # A changed SQLite source is represented by a new run.  These links keep
    # the old run auditable until the replacement has completed successfully.
    supersedes_run_id: str | None = None
    superseded_by_run_id: str | None = None

    @property
    def relative_root(self) -> str:
        return self.source_path

    @property
    def relative_source_path(self) -> str:
        return self.source_path


@dataclass(frozen=True, slots=True)
class AbstractIngestionProgress:
    """Bounded progress returned after each abstract batch."""

    run_id: str
    cursor: str | None
    total: int
    processed: int
    indexed: int
    unchanged: int
    failed: int
    skipped_empty: int
    passages: int
    status: str
    complete: bool
    errors: Sequence[str] = field(default_factory=tuple)
    skipped_invalid_key: int = 0


def _generation_fingerprint(generation: Any) -> str:
    fingerprint = str(getattr(generation, "fingerprint", generation or ""))
    if len(fingerprint) != 64 or any(character not in "0123456789abcdef" for character in fingerprint):
        raise AbstractIngestionError(
            "generation_invalid", "collection generation fingerprint is unavailable"
        )
    return fingerprint


class AbstractIngestionService:
    """Index bounded SQLite pages through the existing literature store."""

    def __init__(
        self,
        reader: SQLiteAbstractReader | Path | str | Any,
        store: Any | None = None,
        embedder: Any | None = None,
        *,
        workspace_id: str | None = None,
        workspace: Any | None = None,
        workspace_root: Path | str | Any | None = None,
        relative_source_path: str | None = None,
        source_path: str | None = None,
        generation: Any | None = None,
        generation_fingerprint: str | None = None,
        batch_size: int = MAX_ABSTRACT_BATCH,
    ) -> None:
        # Support the natural ``(reader, store, embedder)`` order and the two
        # common service-assembly permutations.  This only normalises the
        # constructor arguments; all execution still uses one service path.
        candidates = [reader, store, embedder]
        reader_candidate = next(
            (
                candidate
                for candidate in candidates
                if isinstance(candidate, (SQLiteAbstractReader, Path, str))
            ),
            None,
        )
        if reader_candidate is not None:
            remaining = [candidate for candidate in candidates if candidate is not reader_candidate]
            embedder_candidate = next(
                (
                    candidate
                    for candidate in remaining
                    if callable(getattr(candidate, "embed_documents", None))
                ),
                None,
            )
            if embedder_candidate is not None:
                reader = reader_candidate
                embedder = embedder_candidate
                store = next(
                    candidate
                    for candidate in remaining
                    if candidate is not embedder_candidate
                )
        boundary_root = workspace_root if workspace_root is not None else workspace
        if isinstance(reader, (Path, str)):
            reader = SQLiteAbstractReader(reader, workspace_root=boundary_root)
        if not isinstance(reader, SQLiteAbstractReader):
            raise TypeError("reader must be a SQLiteAbstractReader or database path")
        if store is None or embedder is None:
            raise TypeError("store and embedder are required")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.reader = reader
        self.store = store
        self.embedder = embedder
        self.batch_size = min(int(batch_size), MAX_ABSTRACT_BATCH)
        self.workspace_root = self._resolve_service_workspace_root(
            boundary_root, reader
        )
        self.workspace_id = self._resolve_workspace_id(workspace_id, workspace)
        self.relative_source_path = self._resolve_source_path(
            relative_source_path or source_path, workspace
        )
        if generation is None and generation_fingerprint is not None:
            generation = SimpleNamespace(fingerprint=generation_fingerprint)
        self.generation = generation
        self._source_identity: str | None = None
        self._source_total: int | None = None
        self._source_invalid_key_count: int | None = None
        self._source_signature: Any | None = None
        self._supersede_record: Any | None = None

    @staticmethod
    def _resolve_service_workspace_root(
        value: Any | None, reader: SQLiteAbstractReader
    ) -> Path:
        root = reader.workspace_root if value is None else _resolve_workspace_root(value)
        if not _path_inside(root, reader.path):
            raise ValueError("abstract SQLite database is outside workspace")
        return root

    def _resolve_workspace_id(self, workspace_id: str | None, workspace: Any | None) -> str:
        if workspace_id is not None:
            value = str(workspace_id).strip()
            if not value:
                raise ValueError("workspace_id must be a non-empty string")
            return value
        return workspace_id_for(self.workspace_root)

    def _resolve_source_path(self, value: str | None, workspace: Any | None) -> str:
        del workspace
        relative = self.reader.path.relative_to(self.workspace_root).as_posix()
        validate_relative_source_path(relative)
        if value is not None:
            validate_relative_source_path(value)
            if value != relative:
                raise ValueError(
                    "relative_source_path must identify the SQLite database path"
                )
            return value
        validate_relative_source_path(relative)
        return relative

    async def _generation(self) -> Any:
        if self.generation is not None:
            return self.generation
        resolver = getattr(self.store, "resolve_current_generation", None)
        if callable(resolver):
            value = resolver()
            if inspect.isawaitable(value):
                value = await value
            if value is not None:
                self.generation = value
                return value
        current = getattr(self.store, "generation", None)
        if current is not None:
            self.generation = current
            return current
        current = getattr(self.store, "current_generation", None)
        if current is not None:
            self.generation = current
            return current
        current_fingerprint = getattr(self.store, "generation_fingerprint", None)
        if current_fingerprint is not None:
            self.generation = SimpleNamespace(fingerprint=current_fingerprint)
            return self.generation
        identity = getattr(self.embedder, "identity", None)
        if identity is not None:
            try:
                fingerprint = collection_fingerprint(
                    identity, LITERATURE_CHUNK_SCHEMA_VERSION
                )
            except Exception:
                fingerprint = hashlib.sha256(repr(identity).encode("utf-8")).hexdigest()
        else:
            fingerprint = "0" * 64
        self.generation = SimpleNamespace(fingerprint=fingerprint)
        return self.generation

    async def _invalid_key_count(self) -> int:
        counter = getattr(self.reader, "count_invalid_keys", None)
        if not callable(counter):
            return 0
        value = counter()
        if inspect.isawaitable(value):
            value = await value
        return int(value)

    async def _source_snapshot(self) -> tuple[str, int, int]:
        """Prepare one source snapshot and cheaply validate it thereafter.

        A service instance corresponds to one CLI invocation.  The expensive
        database hash and row count are therefore computed once, while every
        later bounded page checks only the reader's immutable main/WAL file
        signature.  A newly-created service (including a resume invocation)
        recomputes and validates identity once against its persisted run.
        """
        validator = getattr(self.reader, "validate_snapshot", None)
        if self._source_identity is None:
            self._source_identity = str(self.reader.source_identity())
            self._source_total = int(self.reader.count())
            self._source_invalid_key_count = await self._invalid_key_count()
            signature_getter = getattr(self.reader, "snapshot_signature", None)
            if callable(signature_getter):
                signature = signature_getter()
                if inspect.isawaitable(signature):
                    signature = await signature
                self._source_signature = signature
        elif callable(validator):
            result = validator(self._source_signature)
            if inspect.isawaitable(result):
                await result
        if self._source_total is None or self._source_invalid_key_count is None:
            raise AbstractIngestionError(
                "source_snapshot_unavailable",
                "abstract SQLite source snapshot is unavailable",
            )
        return (
            self._source_identity,
            self._source_total,
            self._source_invalid_key_count,
        )

    async def _get_run(self, run_id: str) -> Any | None:
        getter = getattr(self.store, "get_ingestion_run", None)
        if not callable(getter):
            return None
        try:
            value = getter(run_id, self.workspace_id, source_kind=LiteratureSourceKind.ABSTRACT)
        except TypeError:
            value = getter(run_id, self.workspace_id)
        if inspect.isawaitable(value):
            value = await value
        return value

    async def _persist_run(self, run: AbstractIngestionRunState) -> None:
        writer = getattr(self.store, "upsert_ingestion_run", None)
        if not callable(writer):
            return
        try:
            value = writer(run, wait=True, source_kind=LiteratureSourceKind.ABSTRACT)
        except TypeError:
            try:
                value = writer(run, wait=True)
            except TypeError:
                value = writer(run)
        if inspect.isawaitable(value):
            await value

    @staticmethod
    def _stats_from(value: Any, run_id: str, total: int) -> AbstractIngestionStats:
        if isinstance(value, AbstractIngestionStats):
            return value
        if value is None:
            return AbstractIngestionStats(run_id=run_id, discovered=total)
        passages = int(getattr(value, "passages", getattr(value, "chunks", 0)))
        processed = int(
            getattr(
                value,
                "processed",
                int(getattr(value, "indexed", 0))
                + int(getattr(value, "unchanged", 0))
                + int(getattr(value, "failed", 0))
                + int(getattr(value, "skipped_empty", 0))
                + int(getattr(value, "skipped_invalid_key", 0)),
            )
        )
        return AbstractIngestionStats(
            run_id=run_id,
            discovered=int(getattr(value, "discovered", total)),
            processed=processed,
            indexed=int(getattr(value, "indexed", 0)),
            unchanged=int(getattr(value, "unchanged", 0)),
            failed=int(getattr(value, "failed", 0)),
            skipped_empty=int(getattr(value, "skipped_empty", 0)),
            skipped_invalid_key=int(getattr(value, "skipped_invalid_key", 0)),
            passages=passages,
            next_cursor=getattr(value, "next_cursor", None),
            complete=bool(getattr(value, "complete", False)),
            errors=tuple(getattr(value, "errors", ())),
            retryable=bool(getattr(value, "retryable", False)),
        )

    def _validate_existing_run(
        self,
        run: Any,
        *,
        run_id: str,
        generation_fingerprint: str,
        source_identity: str,
    ) -> None:
        raw_source_kind = _value(run, "source_kind", None)
        try:
            source_kind = LiteratureSourceKind(raw_source_kind)
        except (TypeError, ValueError) as exc:
            raise AbstractIngestionError(
                "run_context_mismatch",
                "abstract ingestion run has an invalid source kind",
            ) from exc
        if str(_value(run, "run_id", "")) != run_id:
            raise AbstractIngestionError(
                "run_context_mismatch",
                "abstract ingestion run ID does not match the requested run",
            )
        if str(_value(run, "workspace_id", "")) != self.workspace_id:
            raise AbstractIngestionError(
                "workspace_mismatch", "abstract ingestion run workspace does not match"
            )
        if str(_value(run, "generation_fingerprint", "")) != generation_fingerprint:
            raise AbstractIngestionError(
                "generation_mismatch", "abstract ingestion run generation does not match"
            )
        if str(_value(run, "source_identity", "")) != source_identity:
            raise AbstractIngestionError(
                "source_identity_mismatch",
                "abstract SQLite source identity does not match the persisted run",
            )
        if str(
            _value(run, "source_path", _value(run, "relative_source_path", ""))
        ) != self.relative_source_path:
            raise AbstractIngestionError(
                "source_path_mismatch", "abstract ingestion source path does not match"
            )
        if source_kind is not LiteratureSourceKind.ABSTRACT:
            raise AbstractIngestionError(
                "source_kind_mismatch", "abstract ingestion run source kind does not match"
            )

    async def _manifest_lookup(
        self, document_ids: Sequence[str], generation: Any
    ) -> dict[str, Any]:
        getter = getattr(self.store, "get_document_manifests", None)
        if not callable(getter) or not document_ids:
            return {}
        try:
            value = getter(
                self.workspace_id,
                document_ids,
                generation=generation,
                source_kind=LiteratureSourceKind.ABSTRACT,
            )
        except TypeError:
            try:
                value = getter(
                    self.workspace_id,
                    document_ids,
                    generation=generation,
                )
            except TypeError:
                value = getter(self.workspace_id, document_ids)
        if inspect.isawaitable(value):
            value = await value
        return dict(value or {})

    @staticmethod
    def _manifest_unchanged(
        manifest: Any,
        record: AbstractSourceRecord,
        revision: str,
        *,
        workspace_id: str,
        document_id: str,
        relative_source_path: str,
        generation_fingerprint: str,
    ) -> bool:
        if manifest is None:
            return False
        try:
            return (
                str(_value(manifest, "document_id", "")) == document_id
                and str(_value(manifest, "workspace_id", "")) == workspace_id
                and LiteratureSourceKind(
                    _value(manifest, "source_kind", "")
                )
                is LiteratureSourceKind.ABSTRACT
                and str(_value(manifest, "source_record_id", "")) == record.paper_key
                and str(_value(manifest, "relative_source_path", ""))
                == relative_source_path
                and str(_value(manifest, "content_sha256", "")) == revision
                and str(_value(manifest, "model_fingerprint", ""))
                == generation_fingerprint
                and DocumentStatus(_value(manifest, "status", ""))
                is DocumentStatus.READY
            )
        except (TypeError, ValueError):
            return False

    def _manifest(
        self,
        record: AbstractSourceRecord,
        revision: str,
        generation_fingerprint: str,
        *,
        status: DocumentStatus,
        indexed_at: datetime | None = None,
        last_error: str = "",
    ) -> DocumentManifest:
        return DocumentManifest(
            schema_version=LITERATURE_CHUNK_SCHEMA_VERSION,
            record_type="document",
            workspace_id=self.workspace_id,
            document_id=abstract_document_id_for(
                self.workspace_id, self.relative_source_path, record.paper_key
            ),
            relative_source_path=self.relative_source_path,
            file_name=self.reader.path.name,
            content_sha256=revision,
            status=status,
            title=record.title,
            authors=record.authors,
            year=record.publication_year,
            num_pages=0,
            chunk_count=1 if status is DocumentStatus.READY else 0,
            model_fingerprint=generation_fingerprint,
            indexed_at=indexed_at,
            last_error=last_error,
            source_kind=LiteratureSourceKind.ABSTRACT,
            source_record_id=record.paper_key,
            doi=record.doi,
            pmid=record.pmid,
            pmcid=record.pmcid,
            journal=record.journal,
            relevance_tier=record.relevance_tier,
        )

    async def _embed(self, text: str) -> tuple[float, ...]:
        try:
            response = await self.embedder.embed_documents([text])
            identity = getattr(self.embedder, "identity", None)
            dimension = getattr(identity, "dimension", None)
            vectors = validate_vectors(response, 1, dimension)
        except RagProviderError:
            raise
        except Exception as exc:
            raise RagProviderError(
                "embedding_failed", "abstract embedding failed"
            ) from exc
        return tuple(vectors[0])

    async def _index_record(
        self,
        record: AbstractSourceRecord,
        revision: str,
        generation_fingerprint: str,
    ) -> tuple[PassagePoint, str | None]:
        document_id = abstract_document_id_for(
            self.workspace_id, self.relative_source_path, record.paper_key
        )
        text = f"Title: {record.title}\nAbstract: {record.abstract}"
        vector = await self._embed(text)
        indexed_at = datetime.now(timezone.utc)
        point = PassagePoint(
            schema_version=LITERATURE_CHUNK_SCHEMA_VERSION,
            record_type="passage",
            workspace_id=self.workspace_id,
            document_id=document_id,
            document_revision=revision,
            ingest_state=IngestState.STAGED,
            passage_id=passage_id_for(document_id, revision, 0),
            chunk_index=0,
            text=text,
            title=record.title,
            authors=record.authors,
            year=record.publication_year,
            section="Abstract",
            heading_path="Abstract",
            page_start=None,
            page_end=None,
            previous_passage_id=None,
            next_passage_id=None,
            relative_source_path=self.relative_source_path,
            model_fingerprint=generation_fingerprint,
            limitations=("abstract_only", "fulltext_not_checked"),
            dense=vector,
            normalized_text_sha256=_normalised_text_sha256(text),
            indexed_at=indexed_at,
            source_kind=LiteratureSourceKind.ABSTRACT,
            source_record_id=record.paper_key,
            doi=record.doi,
            pmid=record.pmid,
            pmcid=record.pmcid,
            journal=record.journal,
            relevance_tier=record.relevance_tier,
        )
        await self.store.upsert_passages(
            [point], batch_size=1, workspace_id=self.workspace_id
        )
        staged_count = await self.store.count_revision(
            document_id,
            revision,
            IngestState.STAGED,
            workspace_id=self.workspace_id,
        )
        staged_count_value = getattr(staged_count, "count", staged_count)
        if int(staged_count_value) != 1:
            raise AbstractIngestionError(
                "staged_count_mismatch", "staged abstract passage count is incomplete"
            )
        await self.store.set_revision_state(
            document_id,
            revision,
            IngestState.READY,
            workspace_id=self.workspace_id,
        )
        await self.store.upsert_document(
            self._manifest(
                record,
                revision,
                generation_fingerprint,
                status=DocumentStatus.READY,
                indexed_at=indexed_at,
            )
        )
        cleanup_error = await self._cleanup_other_revisions(document_id, revision)
        return point, cleanup_error

    async def _record_failure(
        self,
        record: AbstractSourceRecord,
        revision: str,
        generation_fingerprint: str,
        exc: BaseException,
    ) -> str:
        error = _redact_error(exc)
        try:
            await self.store.upsert_document(
                self._manifest(
                    record,
                    revision,
                    generation_fingerprint,
                    status=DocumentStatus.FAILED,
                    last_error=error,
                )
            )
        except Exception as manifest_exc:
            error = _bounded_errors((error, _redact_error(manifest_exc)))[-1]
        return error

    @staticmethod
    def _run_complete(run: Any) -> bool:
        stats = _value(run, "stats", None)
        return bool(
            _value(stats, "complete", _value(run, "complete", False))
        )

    def _validate_supersede_record(
        self, run: Any, *, requested_run_id: str, generation_fingerprint: str
    ) -> None:
        try:
            source_kind = LiteratureSourceKind(_value(run, "source_kind", ""))
        except (TypeError, ValueError) as exc:
            raise AbstractIngestionError(
                "supersede_run_context_mismatch",
                "the previous abstract run has no valid source kind",
            ) from exc
        if source_kind is not LiteratureSourceKind.ABSTRACT:
            raise AbstractIngestionError(
                "supersede_run_context_mismatch",
                "the previous run is not an abstract ingestion run",
            )
        if str(_value(run, "run_id", "")) != requested_run_id:
            raise AbstractIngestionError(
                "supersede_run_context_mismatch",
                "the previous abstract run ID does not match",
            )
        if _value(run, "workspace_id", "") != self.workspace_id:
            raise AbstractIngestionError(
                "supersede_run_context_mismatch",
                "the previous abstract run belongs to another workspace",
            )
        if _value(
            run, "generation_fingerprint", _value(run, "generation", "")
        ) != generation_fingerprint:
            raise AbstractIngestionError(
                "supersede_run_context_mismatch",
                "the previous abstract run belongs to another generation",
            )
        source_path = _value(
            run, "source_path", _value(run, "relative_source_path", "")
        )
        if not isinstance(source_path, str):
            raise AbstractIngestionError(
                "supersede_run_context_mismatch",
                "the previous abstract run has an invalid SQLite source path",
            )
        try:
            validate_relative_source_path(source_path)
        except ValueError as exc:
            raise AbstractIngestionError(
                "supersede_run_context_mismatch",
                "the previous abstract run has an invalid SQLite source path",
            ) from exc

    async def _resolve_supersede_record(
        self,
        run_id: str,
        generation_fingerprint: str,
    ) -> Any:
        if run_id == "":
            raise AbstractIngestionError(
                "supersede_run_invalid", "previous abstract run ID must not be empty"
            )
        record = self._supersede_record
        if record is None:
            record = await self._get_run(run_id)
            if record is None:
                raise AbstractIngestionError(
                    "supersede_run_missing",
                    "the previous abstract run could not be found; provide its saved run ID",
                )
            self._validate_supersede_record(
                record,
                requested_run_id=run_id,
                generation_fingerprint=generation_fingerprint,
            )
            self._supersede_record = record
        return record

    async def _cleanup_nonready_source(
        self, relative_source_path: str | None = None
    ) -> str | None:
        """Remove only non-ready artifacts for one abstract source.

        The Qdrant adapter owns the filtered deletion.  Narrow fakes or older
        adapters may not expose it; in that case the safe fallback is to leave
        artifacts untouched rather than guessing which ready knowledge is
        absent from the replacement database.
        """
        source_path = (
            self.relative_source_path
            if relative_source_path is None
            else relative_source_path
        )
        cleanup = getattr(self.store, "delete_nonready_abstract_artifacts", None)
        if not callable(cleanup):
            return None
        try:
            result = cleanup(
                workspace_id=self.workspace_id,
                relative_source_path=source_path,
            )
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return _redact_error(exc)
        return None

    async def _cleanup_other_revisions(
        self, document_id: str, revision: str
    ) -> str | None:
        """Remove old staged/ready revisions only after ``revision`` is ready."""
        cleanup = getattr(self.store, "delete_other_revisions", None)
        if not callable(cleanup):
            # Narrow compatibility stores may not have revision cleanup yet;
            # they are allowed to retain old knowledge rather than risk an
            # unscoped deletion.
            return None
        try:
            result = cleanup(
                document_id,
                revision,
                workspace_id=self.workspace_id,
                source_kind=LiteratureSourceKind.ABSTRACT,
            )
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return _redact_error(exc)
        return None

    async def _mark_superseded(self, run: Any, *, new_run_id: str) -> None:
        if str(_value(run, "status", "")) == "superseded":
            return
        stats = self._stats_from(
            _value(run, "stats", None),
            str(_value(run, "run_id", "")),
            int(_value(_value(run, "stats", None), "discovered", 0) or 0),
        )
        stats = replace(
            stats,
            complete=True,
            next_cursor=None,
            retryable=False,
        )
        try:
            superseded = replace(
                run,
                cursor=None,
                status="superseded",
                stats=stats,
                superseded_by_run_id=new_run_id,
            )
        except TypeError:
            # A legacy run object can still be updated at the store boundary;
            # retain its original shape while preserving the audit link when
            # the object permits attribute assignment.
            setattr(run, "cursor", None)
            setattr(run, "status", "superseded")
            setattr(run, "stats", stats)
            try:
                setattr(run, "superseded_by_run_id", new_run_id)
            except Exception:
                pass
            superseded = run
        await self._persist_run(superseded)

    @staticmethod
    def _progress(
        run_id: str, total: int, cursor: str | None, status: str, stats: AbstractIngestionStats
    ) -> AbstractIngestionProgress:
        return AbstractIngestionProgress(
            run_id=run_id,
            cursor=cursor,
            total=total,
            processed=stats.processed,
            indexed=stats.indexed,
            unchanged=stats.unchanged,
            failed=stats.failed,
            skipped_empty=stats.skipped_empty,
            passages=stats.passages,
            status=status,
            complete=stats.complete,
            errors=stats.errors,
            skipped_invalid_key=stats.skipped_invalid_key,
        )

    async def index_batch(
        self,
        *,
        run_id: str,
        cursor: str | None = None,
        limit: int = MAX_ABSTRACT_BATCH,
        resume: bool = False,
        supersede_run_id: str | None = None,
    ) -> AbstractIngestionProgress:
        """Index one bounded keyset page and persist its durable cursor.

        ``resume=True`` is intentionally strict: an unknown ID is an error,
        never an implicit new run.  A changed source is replaced only through
        ``supersede_run_id``; the previous run remains unresolved until this
        replacement reaches a successful completion.
        """
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        if cursor is not None and not isinstance(cursor, str):
            raise ValueError("cursor must be a string or None")
        if limit < 1:
            raise ValueError("limit must be positive")
        if resume and supersede_run_id is not None:
            raise AbstractIngestionError(
                "resume_supersede_conflict",
                "a resumed run cannot also supersede another run",
            )
        if supersede_run_id is not None:
            if not isinstance(supersede_run_id, str) or not supersede_run_id.strip():
                raise ValueError("supersede_run_id must be a non-empty string")
            supersede_run_id = supersede_run_id.strip()
        requested_limit = min(int(limit), self.batch_size, MAX_ABSTRACT_BATCH)
        generation = await self._generation()
        generation_fingerprint = _generation_fingerprint(generation)
        source_identity, total, skipped_invalid_key = await self._source_snapshot()
        existing = await self._get_run(run_id)

        if existing is None and resume:
            raise AbstractIngestionError(
                "resume_run_missing",
                "the requested abstract resume run does not exist",
            )

        stored_supersede_id = (
            str(getattr(existing, "supersedes_run_id", "") or "").strip()
            if existing is not None
            else ""
        )
        if stored_supersede_id and supersede_run_id and stored_supersede_id != supersede_run_id:
            raise AbstractIngestionError(
                "supersede_run_mismatch",
                "the replacement run is already linked to another previous run",
            )
        effective_supersede_id = supersede_run_id or stored_supersede_id or None
        previous_run: Any | None = None
        was_complete = False

        if existing is not None:
            self._validate_existing_run(
                existing,
                run_id=run_id,
                generation_fingerprint=generation_fingerprint,
                source_identity=source_identity,
            )
            existing_cursor = getattr(existing, "cursor", None)
            if cursor is not None and cursor != existing_cursor:
                raise AbstractIngestionError(
                    "cursor_mismatch", "requested cursor does not match the persisted run"
                )
            effective_cursor = cursor if cursor is not None else existing_cursor
            stats = self._stats_from(getattr(existing, "stats", None), run_id, total)
            was_complete = self._run_complete(existing)
            stats = replace(
                stats,
                discovered=total,
                next_cursor=effective_cursor,
                complete=False,
                retryable=False,
            )
        else:
            effective_cursor = cursor
            stats = AbstractIngestionStats(
                run_id=run_id,
                discovered=total,
                processed=skipped_invalid_key,
                skipped_invalid_key=skipped_invalid_key,
                next_cursor=effective_cursor,
            )

        if effective_supersede_id is not None:
            previous_run = await self._resolve_supersede_record(
                effective_supersede_id,
                generation_fingerprint,
            )
        superseded_source_path: str | None = None
        if previous_run is not None:
            superseded_source_path = _value(
                previous_run,
                "source_path",
                _value(previous_run, "relative_source_path", ""),
            )
            # _validate_supersede_record has already checked this path.  Keep
            # the type guard here so compatibility run objects cannot widen
            # the cleanup call after validation.
            if not isinstance(superseded_source_path, str):
                raise AbstractIngestionError(
                    "supersede_run_context_mismatch",
                    "the previous abstract run has an invalid SQLite source path",
                )

        if was_complete and previous_run is None:
            return self._progress(
                run_id,
                total,
                None,
                "complete",
                replace(stats, complete=True, next_cursor=None),
            )

        select_staging = getattr(self.store, "select_staging_generation", None)
        if callable(select_staging):
            value = select_staging(generation)
            if inspect.isawaitable(value):
                await value

        def run_state(current_stats: AbstractIngestionStats, status: str) -> AbstractIngestionRunState:
            return AbstractIngestionRunState(
                run_id=run_id,
                workspace_id=self.workspace_id,
                generation_fingerprint=generation_fingerprint,
                source_identity=source_identity,
                source_path=self.relative_source_path,
                cursor=current_stats.next_cursor,
                status=status,
                stats=current_stats,
                supersedes_run_id=effective_supersede_id,
            )

        cleanup_sources: list[str] = []
        if superseded_source_path is not None:
            cleanup_sources.append(superseded_source_path)
        if self.relative_source_path not in cleanup_sources:
            cleanup_sources.append(self.relative_source_path)

        if was_complete:
            cleanup_error: str | None = None
            if previous_run is not None:
                for source_path in cleanup_sources:
                    cleanup_error = await self._cleanup_nonready_source(source_path)
                    if cleanup_error is not None:
                        break
            if cleanup_error is not None:
                stats = replace(
                    stats,
                    complete=False,
                    retryable=True,
                    errors=_bounded_errors((*stats.errors, cleanup_error)),
                )
                await self._persist_run(run_state(stats, "retryable"))
                return self._progress(
                    run_id, total, stats.next_cursor, "retryable", stats
                )
            try:
                if previous_run is not None:
                    await self._mark_superseded(previous_run, new_run_id=run_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                diagnostic = _redact_error(exc)
                stats = replace(
                    stats,
                    complete=False,
                    retryable=True,
                    errors=_bounded_errors((*stats.errors, diagnostic)),
                )
                await self._persist_run(run_state(stats, "retryable"))
                return self._progress(
                    run_id, total, stats.next_cursor, "retryable", stats
                )
            return self._progress(
                run_id,
                total,
                None,
                "complete",
                replace(stats, complete=True, next_cursor=None, retryable=False),
            )

        # This is deliberately before source fetch, embedding, or any document
        # and passage write.  A cancellation during the first batch leaves a
        # durable run ID for the driver to resume.
        await self._persist_run(run_state(stats, "running"))

        if previous_run is not None:
            batch_cleanup_error: str | None = None
            for source_path in cleanup_sources:
                batch_cleanup_error = await self._cleanup_nonready_source(source_path)
                if batch_cleanup_error is not None:
                    break
            if batch_cleanup_error is not None:
                stats = replace(
                    stats,
                    complete=False,
                    retryable=True,
                    errors=_bounded_errors((*stats.errors, batch_cleanup_error)),
                )
                retryable_run = run_state(stats, "retryable")
                await self._persist_run(retryable_run)
                return self._progress(
                    run_id, total, stats.next_cursor, "retryable", stats
                )

        rows: tuple[AbstractSourceRecord, ...]
        try:
            rows = self.reader.fetch_after(effective_cursor, limit=requested_limit)
            if inspect.isawaitable(rows):
                rows = await rows
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            diagnostic = _redact_error(exc)
            stats = replace(
                stats,
                next_cursor=effective_cursor,
                complete=False,
                retryable=True,
                errors=_bounded_errors((*stats.errors, diagnostic)),
            )
            await self._persist_run(run_state(stats, "retryable"))
            return self._progress(run_id, total, effective_cursor, "retryable", stats)

        ids_by_key: dict[str, str] = {}
        revisions_by_key: dict[str, str] = {}
        for row in rows:
            if not row.paper_key:
                continue
            ids_by_key[row.paper_key] = abstract_document_id_for(
                self.workspace_id, self.relative_source_path, row.paper_key
            )
            revisions_by_key[row.paper_key] = canonical_abstract_revision(row)
        try:
            manifests = await self._manifest_lookup(
                tuple(ids_by_key.values()), generation
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            diagnostic = _redact_error(exc)
            stats = replace(
                stats,
                next_cursor=effective_cursor,
                complete=False,
                retryable=True,
                errors=_bounded_errors((*stats.errors, diagnostic)),
            )
            await self._persist_run(run_state(stats, "retryable"))
            return self._progress(run_id, total, effective_cursor, "retryable", stats)

        progress_cursor = effective_cursor
        for row in rows:
            # A keyless row cannot safely advance a keyset cursor.  The reader
            # normally excludes it through ``paper_key > ?``; fail closed if a
            # custom reader nevertheless returns one.
            if not row.paper_key:
                diagnostic = "ValueError: abstract row has no stable paper_key"
                stats = replace(
                    stats,
                    failed=stats.failed + 1,
                    processed=stats.processed + 1,
                    next_cursor=progress_cursor,
                    complete=False,
                    retryable=True,
                    errors=_bounded_errors((*stats.errors, diagnostic)),
                )
                break
            revision = revisions_by_key[row.paper_key]
            document_id = ids_by_key[row.paper_key]
            if not row.abstract.strip():
                stats = replace(
                    stats,
                    processed=stats.processed + 1,
                    skipped_empty=stats.skipped_empty + 1,
                )
                progress_cursor = row.paper_key
                continue
            manifest = manifests.get(document_id)
            if self._manifest_unchanged(
                manifest,
                row,
                revision,
                workspace_id=self.workspace_id,
                document_id=document_id,
                relative_source_path=self.relative_source_path,
                generation_fingerprint=generation_fingerprint,
            ):
                cleanup_error = await self._cleanup_other_revisions(document_id, revision)
                stats = replace(
                    stats,
                    processed=stats.processed + 1,
                    unchanged=stats.unchanged + 1,
                    next_cursor=progress_cursor if cleanup_error else row.paper_key,
                    complete=False if cleanup_error else stats.complete,
                    retryable=bool(cleanup_error) or stats.retryable,
                    errors=(
                        _bounded_errors((*stats.errors, cleanup_error))
                        if cleanup_error
                        else stats.errors
                    ),
                )
                if cleanup_error:
                    break
                progress_cursor = row.paper_key
                continue
            try:
                _point, cleanup_error = await self._index_record(
                    row, revision, generation_fingerprint
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                diagnostic = await self._record_failure(
                    row, revision, generation_fingerprint, exc
                )
                stats = replace(
                    stats,
                    processed=stats.processed + 1,
                    failed=stats.failed + 1,
                    next_cursor=progress_cursor,
                    complete=False,
                    retryable=True,
                    errors=_bounded_errors((*stats.errors, diagnostic)),
                )
                break
            stats = replace(
                stats,
                processed=stats.processed + 1,
                indexed=stats.indexed + 1,
                passages=stats.passages + 1,
            )
            if cleanup_error:
                stats = replace(
                    stats,
                    next_cursor=progress_cursor,
                    complete=False,
                    retryable=True,
                    errors=_bounded_errors((*stats.errors, cleanup_error)),
                )
                break
            progress_cursor = row.paper_key

        failed = stats.retryable
        if failed:
            # Keep all rows that committed before the failure out of the next
            # page.  The batch-start cursor would reprocess them on retry.
            next_cursor = progress_cursor
            status = "retryable"
            complete = False
        elif not rows:
            next_cursor = None
            status = "complete"
            complete = True
        else:
            # One-row lookahead determines completion without loading another
            # page and keeps pagination strictly keyset based.
            try:
                has_more = bool(
                    self.reader.fetch_after(progress_cursor, limit=1)
                ) if progress_cursor is not None else True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                diagnostic = _redact_error(exc)
                stats = replace(
                    stats,
                    next_cursor=progress_cursor,
                    complete=False,
                    retryable=True,
                    errors=_bounded_errors((*stats.errors, diagnostic)),
                )
                next_cursor = progress_cursor
                status = "retryable"
                complete = False
            else:
                next_cursor = progress_cursor
                complete = not has_more
                status = "complete" if complete else "running"
        stats = replace(
            stats,
            next_cursor=None if complete else next_cursor,
            complete=complete,
            retryable=failed or stats.retryable,
        )
        final_run = run_state(stats, status)
        await self._persist_run(final_run)

        if complete and previous_run is not None:
            try:
                await self._mark_superseded(previous_run, new_run_id=run_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                diagnostic = _redact_error(exc)
                stats = replace(
                    stats,
                    complete=False,
                    retryable=True,
                    errors=_bounded_errors((*stats.errors, diagnostic)),
                )
                status = "retryable"
                await self._persist_run(run_state(stats, status))

        return self._progress(run_id, total, stats.next_cursor, status, stats)


__all__ = [
    "AbstractIngestionError",
    "AbstractIngestionProgress",
    "AbstractIngestionRunState",
    "AbstractIngestionService",
    "AbstractIngestionStats",
    "AbstractSourceRecord",
    "SQLiteAbstractReader",
    "abstract_document_id_for",
    "canonical_abstract_revision",
]
