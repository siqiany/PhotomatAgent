# Tiered Literature RAG Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Add resumable two-stage PDF/abstract ingestion and enforce the local full-text -> local abstract -> arXiv retrieval policy.

**Architecture:** Keep one generation-scoped Qdrant documents/passages pair, tag records with source_kind, and filter each retrieval tier in Qdrant. Reuse PDF ingestion, add a streaming SQLite abstract importer, expose bounded progress/resume CLI operations, and wrap them with one thin WSL Bash driver.

**Tech Stack:** Python 3.12, asyncio, dataclasses/Pydantic, sqlite3, Typer, qdrant-client 1.x, pytest, mypy, Bash.

**Spec:** docs/superpowers/specs/2026-09-10-tiered-literature-rag-ingestion-design.md

## Global Constraints

- PDF and abstract records are independent; never infer DOI/title links between them.
- Use source_kind=fulltext|abstract in one Qdrant generation and filter before candidate retrieval.
- Preserve AgentRuntime -> ToolRegistry -> Tool.execute() for all model-requested calls.
- arXiv remains a separate permission-controlled, session-only search and never writes to Qdrant.
- Resolve sources through Workspace.resolve; the intended workspace is /home/shiqiany/AIagent.
- External providers keep explicit disclosure and confirmation; never silently fall back externally.
- Do not implement structured device/material/process CSV ingestion.
- Do not import the real corpus, download models, access arXiv, call external APIs, or run the repository-wide test suite.
- Run only the focused tests and checks named in this plan.

## File Map

- Create src/photomatagent/scientific/capabilities/literature/abstract_ingestion.py for SQLite streaming and abstract batches.
- Modify models.py, qdrant_store.py, ingestion.py, and retrieval.py for source-aware contracts and filtering.
- Modify literature/__init__.py and cli/rag.py for assembly, tool schema, progress, resume, and activation gates.
- Create scripts/import_literature_qdrant.sh as the user-facing WSL driver.
- Update docs/qdrant_rag_operations.md.
- Add or modify only the focused literature/Qdrant tests named below.

---

### Task 1: Source-Aware Models and Qdrant Queries

**Files:**
- Modify: src/photomatagent/scientific/capabilities/literature/models.py
- Modify: src/photomatagent/scientific/capabilities/literature/qdrant_store.py
- Modify: src/photomatagent/scientific/capabilities/literature/retrieval.py
- Test: tests/test_qdrant_store.py
- Test: tests/test_rag_retrieval.py

**Interfaces:**
- Consumes: current document, passage, store, and retriever contracts.
- Produces: LiteratureSourceKind, additive provenance fields, bounded manifest lookup, and source-filtered retrieval.

- [ ] **Step 1: Write failing contract tests**

~~~python
def test_abstract_payload_preserves_source_fields() -> None:
    point = _passage(source_kind="abstract", source_record_id="key-a", doi="10.1/a")
    payload = point.to_payload()
    assert payload["source_kind"] == "abstract"
    assert payload["source_record_id"] == "key-a"
    assert payload["doi"] == "10.1/a"


@pytest.mark.asyncio
async def test_hybrid_candidates_filter_source_kind_in_qdrant() -> None:
    store, client = _store()
    await store.hybrid_candidates(
        "HgTe", [0.1, 0.2], workspace_id="workspace-a",
        source_kind="fulltext", limit=10,
    )
    assert _filter_value(client.query_calls[-1]["query_filter"], "source_kind") == "fulltext"


@pytest.mark.asyncio
async def test_retriever_rejects_wrong_source_candidates() -> None:
    retriever = _retriever_with_candidates(source_kind="abstract")
    result = await retriever.search(
        "HgTe", workspace_id="workspace-a", source_kind="fulltext"
    )
    assert result.passages == ()
~~~

- [ ] **Step 2: Run RED tests**

~~~bash
uv run pytest -q \
  tests/test_qdrant_store.py::test_abstract_payload_preserves_source_fields \
  tests/test_qdrant_store.py::test_hybrid_candidates_filter_source_kind_in_qdrant \
  tests/test_rag_retrieval.py::test_retriever_rejects_wrong_source_candidates
~~~

Expected: fail because source contracts and parameters do not exist.

- [ ] **Step 3: Implement minimal source-aware contracts**

~~~python
class LiteratureSourceKind(str, Enum):
    FULLTEXT = "fulltext"
    ABSTRACT = "abstract"

# Add default-compatible fields to DocumentManifest and PassagePoint:
source_kind: LiteratureSourceKind = LiteratureSourceKind.FULLTEXT
source_record_id: str = ""
doi: str = ""
pmid: str = ""
pmcid: str = ""
journal: str = ""
relevance_tier: str = ""
~~~

Serialize these fields and expose them through RetrievedPassage.as_dict(). Create payload indexes for source_kind, source_record_id, doi, and relevance_tier. Add:

The exact interface is an asynchronous method named get_document_manifests. It
accepts workspace_id, a bounded Sequence of document IDs, and an optional
generation, and returns a dictionary keyed by document ID. Its implementation
must reject oversized input, retrieve only the requested point IDs, and discard
payloads whose workspace ID or record type does not match.

Thread source_kind through dense_candidates, sparse_candidates, hybrid_candidates, and LiteratureRetriever.search. Apply the filter in Qdrant and revalidate candidate payloads in Python to fail closed with fake or legacy stores.

- [ ] **Step 4: Run GREEN tests**

~~~bash
uv run pytest -q tests/test_qdrant_store.py tests/test_rag_retrieval.py
~~~

- [ ] **Step 5: Commit**

~~~bash
git add src/photomatagent/scientific/capabilities/literature/models.py \
  src/photomatagent/scientific/capabilities/literature/qdrant_store.py \
  src/photomatagent/scientific/capabilities/literature/retrieval.py \
  tests/test_qdrant_store.py tests/test_rag_retrieval.py
git commit -m "feat: isolate literature retrieval by source kind"
~~~

---

### Task 2: Streaming SQLite Abstract Ingestion

**Files:**
- Create: src/photomatagent/scientific/capabilities/literature/abstract_ingestion.py
- Modify: src/photomatagent/scientific/capabilities/literature/qdrant_store.py
- Test: tests/test_abstract_ingestion.py

**Interfaces:**
- Consumes: embedding provider, Task 1 manifest lookup, Qdrant staged/ready methods, and ingestion-run storage.
- Produces: AbstractSourceRecord, SQLiteAbstractReader, AbstractIngestionProgress, and AbstractIngestionService.index_batch.

- [ ] **Step 1: Write failing reader tests with a three-row temporary SQLite database**

~~~python
def test_reader_uses_keyset_pagination(abstract_db: Path) -> None:
    reader = SQLiteAbstractReader(abstract_db)
    first = reader.fetch_after(None, limit=2)
    second = reader.fetch_after(first[-1].paper_key, limit=2)
    assert [row.paper_key for row in first] == ["key-a", "key-b"]
    assert [row.paper_key for row in second] == ["key-c"]


def test_revision_ignores_retrieval_timestamp() -> None:
    row = _record(abstract="knowledge")
    assert canonical_abstract_revision(row) == canonical_abstract_revision(
        replace(row, retrieved_at="later")
    )
    assert canonical_abstract_revision(row) != canonical_abstract_revision(
        replace(row, abstract="changed")
    )
~~~

- [ ] **Step 2: Run RED reader tests**

~~~bash
uv run pytest -q tests/test_abstract_ingestion.py
~~~

Expected: import failure because the module is absent.

- [ ] **Step 3: Implement the read-only reader**

~~~python
@dataclass(frozen=True, slots=True)
class AbstractSourceRecord:
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


class SQLiteAbstractReader:
    def count(self) -> int:
        return int(self._connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0])

    def source_identity(self) -> str:
        stat = self.path.stat()
        return hashlib.sha256(
            f"{self.relative_path}:{stat.st_size}:{stat.st_mtime_ns}".encode()
        ).hexdigest()

    def fetch_after(
        self, cursor: str | None, *, limit: int
    ) -> Sequence[AbstractSourceRecord]:
        rows = self._connection.execute(
            "SELECT paper_key, title, abstract FROM papers "
            "WHERE paper_key > ? ORDER BY paper_key LIMIT ?",
            (cursor or "", limit),
        ).fetchall()
        return tuple(self._record(row) for row in rows)
~~~

Open SQLite with URI mode=ro, validate the papers table and required columns, and query using WHERE paper_key > ? ORDER BY paper_key LIMIT ?. Never use OFFSET or load all rows.

- [ ] **Step 4: Write failing batch, idempotency, and resume tests**

~~~python
@pytest.mark.asyncio
async def test_batch_writes_one_document_and_passage_per_row(abstract_db: Path) -> None:
    service, store, embedder = _service(abstract_db)
    progress = await service.index_batch(run_id="run-a", cursor=None, limit=2)
    assert (progress.indexed, progress.passages) == (2, 2)
    assert all("abstract_only" in point.limitations for point in store.passages)


@pytest.mark.asyncio
async def test_resume_does_not_reembed_committed_rows(abstract_db: Path) -> None:
    service, store, embedder = _service(abstract_db)
    first = await service.index_batch(run_id="run-a", cursor=None, limit=1)
    second = await service.index_batch(run_id="run-a", cursor=first.cursor, limit=2)
    assert embedder.embedded_document_count == 3
    assert second.complete is True


@pytest.mark.asyncio
async def test_failure_keeps_pre_record_cursor(abstract_db: Path) -> None:
    service = _service_with_failing_embedder(abstract_db)
    progress = await service.index_batch(run_id="run-a", cursor=None, limit=1)
    assert progress.status == "retryable"
    assert progress.cursor is None
~~~

- [ ] **Step 5: Verify RED, then implement bounded batches**

~~~python
@dataclass(frozen=True, slots=True)
class AbstractIngestionProgress:
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
    errors: Sequence[str] = ()
~~~

Add AbstractIngestionService.index_batch as an async method with keyword-only
run_id, cursor defaulting to None, and limit defaulting to 20, returning
AbstractIngestionProgress. Use stable IDs from workspace + SQLite relative path +
paper_key. Hash canonical row fields, retrieve only current-batch manifests, skip
unchanged rows, embed Title plus Abstract, stage/upsert/count/ready, then persist
the cursor. Empty abstracts increment skipped_empty. Retryable errors retain the
pre-record cursor. Persist and validate source kind, source identity, workspace,
and generation in run records.

- [ ] **Step 6: Run GREEN tests and commit**

~~~bash
uv run pytest -q tests/test_abstract_ingestion.py
git add src/photomatagent/scientific/capabilities/literature/abstract_ingestion.py \
  src/photomatagent/scientific/capabilities/literature/qdrant_store.py \
  tests/test_abstract_ingestion.py
git commit -m "feat: add resumable abstract ingestion"
~~~

---

### Task 3: CLI Progress, Pausing, Resume, and Activation Gates

**Files:**
- Modify: src/photomatagent/cli/rag.py
- Modify: src/photomatagent/scientific/capabilities/literature/ingestion.py
- Modify: src/photomatagent/scientific/capabilities/literature/__init__.py
- Test: tests/test_rag_cli.py
- Test: tests/test_rag_ingestion.py

**Interfaces:**
- Consumes: existing PDF batches and Task 2 abstract batches.
- Produces: stop-after, resume, progress rows, rag index-abstracts, stage status, and guarded activation.

- [ ] **Step 1: Write failing CLI tests**

~~~python
@pytest.mark.asyncio
async def test_index_pauses_after_requested_budget() -> None:
    rows: list[dict[str, object]] = []
    result = await rag_cli._index_until_complete(
        _services(), _workspace(), _root(), config=_config(), run_id="pdf-run",
        stop_after=25, progress=rows.append,
    )
    assert result["paused"] is True
    assert result["processed_this_invocation"] <= 25
    assert rows


def test_abstract_resume_uses_saved_run(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(app, ["rag", "index-abstracts", "--resume"])
    assert result.exit_code == 0


def test_activate_rejects_incomplete_required_stage(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(
        app, ["rag", "activate", "--require-stage", "pdf",
              "--require-stage", "abstracts", "--yes"]
    )
    assert result.exit_code == 1
    assert "stage_incomplete" in result.stdout
~~~

- [ ] **Step 2: Run RED tests**

~~~bash
uv run pytest -q \
  tests/test_rag_cli.py::test_index_pauses_after_requested_budget \
  tests/test_rag_cli.py::test_abstract_resume_uses_saved_run \
  tests/test_rag_cli.py::test_activate_rejects_incomplete_required_stage
~~~

- [ ] **Step 3: Implement backward-compatible orchestration**

Extend _index_until_complete with keyword-only stop_after defaulting to None and
progress defaulting to None. The progress callback accepts one bounded dictionary.
Keep every existing positional and keyword argument unchanged so current callers
remain compatible.

Emit a bounded progress mapping after every committed batch with total, processed, indexed, unchanged, failed, skipped, passages, rate, ETA, cursor, run ID, and status. A budget stop is paused with exit code zero. Preserve exact resume guidance on KeyboardInterrupt.

Add rag index-abstracts with --database, --run-id, --resume, --stop-after, --yes, and --workspace. Resolve the database via Workspace.resolve. Add stage-aware status and activation preflight. Explicitly tag PDF manifests/passages as fulltext without changing PDF parsing, hashing, deletion, or chunking.

- [ ] **Step 4: Run GREEN tests and commit**

~~~bash
uv run pytest -q tests/test_rag_cli.py tests/test_rag_ingestion.py
git add src/photomatagent/cli/rag.py \
  src/photomatagent/scientific/capabilities/literature/ingestion.py \
  src/photomatagent/scientific/capabilities/literature/__init__.py \
  tests/test_rag_cli.py tests/test_rag_ingestion.py
git commit -m "feat: expose staged literature import controls"
~~~

---

### Task 4: Model-Visible Tiered Retrieval Policy

**Files:**
- Modify: src/photomatagent/scientific/capabilities/literature/__init__.py
- Modify: src/photomatagent/scientific/capabilities/literature/retrieval.py
- Test: tests/test_literature_rag.py
- Test: tests/test_rag_retrieval.py

**Interfaces:**
- Consumes: Task 1 source-aware retrieval and the existing LiteratureSearchArxivTool.
- Produces: source-selecting tool schema, public provenance, and fallback guidance without hidden network access.

- [ ] **Step 1: Write failing public-contract tests**

~~~python
def test_search_schema_defaults_to_fulltext() -> None:
    prop = LiteratureSearchPassagesTool.input_schema["properties"]["source_kind"]
    assert prop["enum"] == ["fulltext", "abstract"]
    assert prop["default"] == "fulltext"


@pytest.mark.asyncio
async def test_abstract_result_identifies_source_and_limit() -> None:
    result = await _abstract_search_tool().execute(
        {"query": "photodetector", "top_k": 1, "source_kind": "abstract"}
    )
    assert result.data["results"][0]["source_kind"] == "abstract"
    assert "abstract_only" in result.data["results"][0]["limitations"]


def test_arxiv_description_forbids_persistence() -> None:
    assert "not persisted" in LiteratureSearchArxivTool.description.lower()
~~~

- [ ] **Step 2: Run RED tests**

~~~bash
uv run pytest -q tests/test_literature_rag.py tests/test_rag_retrieval.py
~~~

- [ ] **Step 3: Implement the model-visible policy**

Add source_kind to LiteratureSearchPassagesTool, default to fulltext, validate it, pass it to retrieval, and return it. Add this stable guidance to capability and tool descriptions:

~~~text
Search local full-text passages first. If they do not directly support the answer
or leave an evidence gap, search local abstract passages. Only then, or when the
user explicitly asks for recent work, call literature.search_arxiv. Abstract and
arXiv results do not mean full text was inspected. arXiv results are not persisted.
~~~

Do not create a wrapper that calls arXiv internally; the network action must remain a separate runtime-visible call.

- [ ] **Step 4: Run GREEN tests and commit**

~~~bash
uv run pytest -q tests/test_literature_rag.py tests/test_rag_retrieval.py
git add src/photomatagent/scientific/capabilities/literature/__init__.py \
  src/photomatagent/scientific/capabilities/literature/retrieval.py \
  tests/test_literature_rag.py tests/test_rag_retrieval.py
git commit -m "feat: enforce tiered literature retrieval guidance"
~~~

---

### Task 5: WSL Driver, Documentation, and Minimal Verification

**Files:**
- Create: scripts/import_literature_qdrant.sh
- Create: tests/test_import_literature_script.py
- Modify: docs/qdrant_rag_operations.md

**Interfaces:**
- Consumes: Task 3 CLI operations.
- Produces: pdf, abstracts, status, and activate workflows with durable local run IDs.

- [ ] **Step 1: Write failing script tests**

~~~python
def test_script_exposes_required_stages() -> None:
    script = Path("scripts/import_literature_qdrant.sh").read_text(encoding="utf-8")
    assert "pdf|abstracts|status|activate" in script
    assert "Photoelectric detection/dataset/paper/pdf" in script
    assert "Photoelectric detection/dataset/paper/abstract/abstracts.sqlite3" in script


def test_dry_run_preserves_paths_with_spaces() -> None:
    result = subprocess.run(
        ["bash", "scripts/import_literature_qdrant.sh", "pdf",
         "--dry-run", "--stop-after", "20"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert "Photoelectric detection/dataset/paper/pdf" in result.stdout
    assert "--stop-after 20" in result.stdout
~~~

- [ ] **Step 2: Run RED tests**

~~~bash
uv run pytest -q tests/test_import_literature_script.py
~~~

- [ ] **Step 3: Implement the thin Bash driver and documentation**

Use set -euo pipefail, Bash argument arrays, repository-relative execution, and default workspace /home/shiqiany/AIagent. Save generated stage run IDs under user_output/rag-import/run-state before starting so interruption is resumable. Dry-run prints shell-escaped arguments and performs no provider or Qdrant work. Forward --yes only when supplied. Activate passes both required stages to the CLI and never calls Qdrant HTTP directly.

Document exactly:

~~~bash
bash scripts/import_literature_qdrant.sh pdf --stop-after 100
bash scripts/import_literature_qdrant.sh pdf --resume --stop-after 100
bash scripts/import_literature_qdrant.sh abstracts --stop-after 5000
bash scripts/import_literature_qdrant.sh abstracts --resume --stop-after 5000
bash scripts/import_literature_qdrant.sh status
bash scripts/import_literature_qdrant.sh activate
~~~

Explain progress fields, external-provider confirmation, source priority, resume state, database-change recovery, and session-only arXiv results.

- [ ] **Step 4: Run the agreed minimal verification**

~~~bash
uv run pytest -q \
  tests/test_abstract_ingestion.py \
  tests/test_qdrant_store.py \
  tests/test_rag_ingestion.py \
  tests/test_rag_retrieval.py \
  tests/test_literature_rag.py \
  tests/test_rag_cli.py \
  tests/test_import_literature_script.py
uv run mypy src/photomatagent/scientific/capabilities/literature \
  src/photomatagent/cli/rag.py
bash -n scripts/import_literature_qdrant.sh
bash scripts/import_literature_qdrant.sh --help
bash scripts/import_literature_qdrant.sh pdf --dry-run --stop-after 20
git diff --check
git diff --stat
git status --short
~~~

Expected: selected tests and targeted checks pass. Do not expand to the full suite. Report exact test counts and explicitly list the full suite, real corpus, model downloads, external APIs, and live arXiv as not run.

- [ ] **Step 5: Inspect invariants and commit**

Confirm server-side source filtering, no absolute paths in Qdrant payloads, abstract_only in public results, no arXiv write path, both-stage activation checks, and no structured CSV ingestion.

~~~bash
git add scripts/import_literature_qdrant.sh tests/test_import_literature_script.py \
  docs/qdrant_rag_operations.md
git commit -m "feat: add resumable WSL literature import driver"
~~~
