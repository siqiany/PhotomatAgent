# Qdrant Literature RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the embedded LanceDB literature index with a Docker-hosted Qdrant RAG backend that handles roughly 10,000 papers, supports local or explicitly authorized external embedding/reranking providers, and preserves PhotomatAgent's tool, permission, evidence, and provenance boundaries.

**Architecture:** Keep `literature.*` as thin deferred tools owned by the authoritative runtime. Move parsing, provider selection, resumable ingestion, Qdrant persistence, hybrid retrieval, and operational CLI behavior into focused application modules. Store document control records and passage vectors in a versioned Qdrant collection pair behind aliases; perform dense+sparse RRF in Qdrant and rerank only a bounded candidate set.

**Tech Stack:** Python 3.12, Pydantic/dataclasses, async protocols, Docling, sentence-transformers, OpenAI-compatible embeddings, Cohere-compatible reranking, qdrant-client 1.19.x, Qdrant server 1.18.2, Typer, Docker Compose, pytest, mypy.

**Spec:** `docs/superpowers/specs/2026-09-08-qdrant-literature-rag-design.md`

## Global Constraints

- `AgentRuntime` remains the sole authority for model-requested tool execution; no provider, CLI helper, Qdrant adapter, or application service may create a second model tool path.
- Keep `literature.index_papers`, `literature.search_passages`, `literature.read_passage`, and `literature.extract_evidence` deferred and preserve their bounded model-visible outputs.
- Qdrant is an internal application backend and must never be exposed as a generic model tool.
- Use `Workspace.resolve`; store only workspace-relative source paths in Qdrant.
- Default to local `intfloat/multilingual-e5-small` embeddings (384 dimensions) and local `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking.
- External providers require `PHOTOMATAGENT_RAG_ALLOW_EXTERNAL=1`; local failure must never trigger external fallback.
- Pin the server image to `qdrant/qdrant:v1.18.2`; use `qdrant-client>=1.19,<2` and verify the pair in Docker integration tests.
- Remove `lancedb`, `pylance`, `PHOTOMATAGENT_LITERATURE_INDEX_DIR`, and all runtime LanceDB branches.
- Do not delete, rewrite, or auto-import `output/literature_index`; report it only as a legacy artifact.
- Do not modify `.env`, real PDFs, `user_input/`, existing user outputs, or unrelated scientific capabilities.
- Use fake/local test doubles for unit tests; never send real papers to external APIs.
- Run narrow tests after every task. Before completion run the entire pytest suite, `uv run mypy src`, and `git diff --check`.

---

## Planned File Structure

```text
compose.qdrant.yaml
src/photomatagent/scientific/capabilities/literature/
  __init__.py                 # thin capability/tool adapters
  models.py                   # paper, passage, ingestion and retrieval contracts
  parser.py                   # Docling-only parsing
  evidence.py                 # deterministic evidence extraction
  providers/
    __init__.py               # public provider exports
    base.py                   # provider protocols, identities and errors
    local.py                  # local embedding and reranking
    external.py               # external embedding and reranking clients
    factory.py                # validated provider construction
  qdrant_store.py             # Qdrant-only persistence/query/snapshot adapter
  ingestion.py                # planning, batching, state transitions and resume
  retrieval.py                # hybrid retrieval, degradation, dedupe, context
src/photomatagent/cli/rag.py   # direct user operations over shared services
tests/
  test_rag_config.py
  test_rag_providers.py
  test_qdrant_store.py
  test_rag_ingestion.py
  test_rag_retrieval.py
  test_rag_cli.py
  test_qdrant_rag_integration.py
  fixtures/literature_rag_eval.json
scripts/benchmark_qdrant_rag.py
docs/qdrant_rag_operations.md
```

---

### Task 1: Strict RAG Configuration and Provider Contracts

**Files:**
- Create: `tests/test_rag_config.py`
- Create: `tests/test_rag_providers.py`
- Create: `src/photomatagent/scientific/capabilities/literature/providers/__init__.py`
- Create: `src/photomatagent/scientific/capabilities/literature/providers/base.py`
- Create: `src/photomatagent/scientific/capabilities/literature/providers/local.py`
- Create: `src/photomatagent/scientific/capabilities/literature/providers/external.py`
- Create: `src/photomatagent/scientific/capabilities/literature/providers/factory.py`
- Modify: `src/photomatagent/scientific/capabilities/config.py`
- Modify: `tests/test_capability_probe.py`

**Interfaces:**
- Produces: `ModelIdentity`, `EmbeddingProvider`, `RerankerProvider`, `RerankScore`, `RagProviderError`, `build_embedding_provider(config)`, and `build_reranker_provider(config)`.
- Produces: strict `ScientificConfig` fields named in Step 3.
- Consumes: existing `ScientificConfig.from_environment()` and installed OpenAI/sentence-transformers packages.

- [ ] **Step 1: Write failing strict configuration tests**

```python
def test_rag_defaults_are_local(tmp_path, monkeypatch):
    monkeypatch.delenv("PHOTOMATAGENT_RAG_ALLOW_EXTERNAL", raising=False)
    config = ScientificConfig.from_environment(workspace=tmp_path)
    assert config.qdrant_url == "http://127.0.0.1:6333"
    assert config.rag_allow_external is False
    assert config.embedding_provider == "local"
    assert config.embedding_model == "intfloat/multilingual-e5-small"
    assert config.embedding_vector_dim == 384
    assert config.reranker_provider == "local"
    assert config.rag_batch_size == 128


def test_invalid_rag_batch_size_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTOMATAGENT_RAG_BATCH_SIZE", "0")
    with pytest.raises(ValueError, match="PHOTOMATAGENT_RAG_BATCH_SIZE"):
        ScientificConfig.from_environment(workspace=tmp_path)


def test_external_provider_requires_explicit_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTOMATAGENT_RAG_EMBEDDING_PROVIDER", "openai_compatible")
    config = ScientificConfig.from_environment(workspace=tmp_path)
    with pytest.raises(RagProviderError) as exc:
        build_embedding_provider(config)
    assert exc.value.code == "external_provider_not_allowed"
```

- [ ] **Step 2: Run the configuration tests and confirm they fail**

Run: `uv run pytest -q tests/test_rag_config.py tests/test_rag_providers.py`

Expected: FAIL because the RAG config fields and provider package do not exist.

- [ ] **Step 3: Add exact strict configuration fields and parsers**

Add these frozen dataclass fields to `ScientificConfig`:

```python
qdrant_url: str = "http://127.0.0.1:6333"
qdrant_api_key_env: str = "QDRANT_API_KEY"
qdrant_collection_prefix: str = "photomat_literature"
qdrant_timeout_seconds: int = 20
rag_allow_external: bool = False
embedding_provider: str = "local"
embedding_model: str = "intfloat/multilingual-e5-small"
embedding_vector_dim: int = 384
embedding_base_url: str = ""
embedding_api_key_env: str = "RAG_EMBEDDING_API_KEY"
reranker_provider: str = "local"
reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker_base_url: str = ""
reranker_api_key_env: str = "RAG_RERANK_API_KEY"
rag_batch_size: int = 128
rag_tool_max_documents: int = 20
```

Replace silent integer fallback for new RAG values with:

```python
def _bounded_int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value
```

Use ranges: timeout 1–300, vector dimension 1–8192, batch 16–512, tool documents 1–100. Remove `literature_index_dir` from the dataclass and environment loader.

- [ ] **Step 4: Define provider contracts and validation helpers**

Implement in `providers/base.py`:

```python
@dataclass(frozen=True)
class ModelIdentity:
    provider: str
    model: str
    dimension: int | None
    document_prefix: str = ""
    query_prefix: str = ""
    normalize: bool = False

    def fingerprint_material(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class RerankScore:
    index: int
    score: float


class RagProviderError(RuntimeError):
    def __init__(self, code: str, message: str) -> None: ...


class EmbeddingProvider(Protocol):
    identity: ModelIdentity
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...


class RerankerProvider(Protocol):
    identity: ModelIdentity
    async def rerank(
        self, query: str, passages: Sequence[str], *, top_n: int
    ) -> list[RerankScore]: ...
```

Add `validate_vectors(vectors, expected_count, expected_dimension)` that rejects count mismatch, dimension mismatch, NaN, and infinity with stable codes.

- [ ] **Step 5: Implement local providers behind lazy imports**

`LocalSentenceTransformerProvider` must prefix documents with `passage: ` and queries with `query: `, normalize vectors, and run `model.encode` through `asyncio.to_thread`. `LocalCrossEncoderProvider` must cap text to the model's supported input contract, run `predict` through `asyncio.to_thread`, and return sorted `RerankScore` values. Cache loaded models by model ID with `lru_cache`; imports remain inside loader functions so missing optional dependencies fail soft.

```python
async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
    prepared = [f"passage: {text}" for text in texts]
    raw = await asyncio.to_thread(
        self._model().encode,
        prepared,
        batch_size=self._batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return validate_vectors(raw, len(texts), self.identity.dimension)
```

- [ ] **Step 6: Implement bounded external providers and tests with fake clients**

`OpenAICompatibleEmbeddingProvider` accepts an injected async OpenAI client for tests. It sends batches to `embeddings.create`, restores results by response index, validates count/dimension/finite values, uses a 20-second configured timeout, and retries only 408/429/5xx or transport failures at most three attempts.

`CohereCompatibleRerankerProvider` accepts an injected async HTTP transport and posts:

```json
{"model":"configured-model","query":"...","documents":["..."],"top_n":10}
```

It accepts only `results=[{"index": integer, "relevance_score": number}]`, validates unique in-range indexes, and raises `RagProviderError("reranker_invalid_response", ...)` on malformed output. Do not log response bodies or keys.

Test that local failure never invokes an external fake, external factories require the hard gate, missing key env names are rejected, and secrets are absent from `str(exc)`.

- [ ] **Step 7: Implement factories without automatic fallback**

```python
def build_embedding_provider(config: ScientificConfig) -> EmbeddingProvider:
    if config.embedding_provider == "local":
        return LocalSentenceTransformerProvider(...)
    if config.embedding_provider == "openai_compatible":
        _require_external_allowed(config)
        return OpenAICompatibleEmbeddingProvider(...)
    raise RagProviderError("embedding_provider_unknown", ...)
```

Use the same exact-match pattern for `local`, `cohere_compatible`, and `disabled` rerank providers. Never catch a local provider construction/runtime error and substitute an external provider.

- [ ] **Step 8: Run narrow tests and type checking**

Run: `uv run pytest -q tests/test_rag_config.py tests/test_rag_providers.py tests/test_capability_probe.py`

Run: `uv run mypy src/photomatagent/scientific/capabilities/config.py src/photomatagent/scientific/capabilities/literature/providers`

Expected: all pass.

- [ ] **Step 9: Commit Task 1**

```bash
git add src/photomatagent/scientific/capabilities/config.py src/photomatagent/scientific/capabilities/literature/providers tests/test_rag_config.py tests/test_rag_providers.py tests/test_capability_probe.py
git commit -m "feat: add configurable RAG model providers"
```

---

### Task 2: Qdrant Collection, Alias, Point, and Snapshot Adapter

**Files:**
- Create: `src/photomatagent/scientific/capabilities/literature/qdrant_store.py`
- Create: `tests/test_qdrant_store.py`
- Modify: `src/photomatagent/scientific/capabilities/literature/models.py`

**Interfaces:**
- Consumes: Task 1 `ModelIdentity`; existing `PaperRecord` and `PassageRecord`.
- Produces: `CollectionGeneration`, `DocumentManifest`, `PassagePoint`, `SearchCandidate`, `SnapshotManifest`, `QdrantLiteratureStore`, `collection_fingerprint()`, and stable ID helpers.
- Later tasks call store methods; they do not access the raw Qdrant client.

- [ ] **Step 1: Write failing identity, schema, and alias tests**

```python
def test_passage_id_changes_only_with_document_revision():
    document_id = document_id_for("workspace-a", "dataset/paper/a.pdf")
    first = passage_id_for(document_id, "a" * 64, 3)
    assert first == passage_id_for(document_id, "a" * 64, 3)
    assert first != passage_id_for(document_id, "b" * 64, 3)


async def test_ensure_generation_creates_versioned_pair_and_aliases(fake_client):
    store = QdrantLiteratureStore(fake_client, prefix="photomat_literature")
    generation = await store.ensure_generation(identity=IDENTITY, chunk_schema_version=1)
    assert generation.documents_physical.endswith(generation.fingerprint[:12])
    assert generation.passages_alias == "photomat_literature_passages_current"
    assert fake_client.created_documents_has_no_vectors
    assert fake_client.created_passages_has_vectors == {"dense", "sparse_bm25"}
```

- [ ] **Step 2: Run tests and confirm missing store failures**

Run: `uv run pytest -q tests/test_qdrant_store.py`

Expected: FAIL on missing module/types.

- [ ] **Step 3: Add exact storage contracts**

Add frozen models:

```python
class DocumentStatus(str, Enum):
    PENDING = "pending"
    STAGED = "staged"
    READY = "ready"
    FAILED = "failed"
    DELETED = "deleted"


class IngestState(str, Enum):
    STAGED = "staged"
    READY = "ready"
    SUPERSEDED = "superseded"


@dataclass(frozen=True)
class CollectionGeneration:
    fingerprint: str
    documents_physical: str
    passages_physical: str
    documents_alias: str
    passages_alias: str


@dataclass(frozen=True)
class SearchCandidate:
    passage_id: str
    score: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class SnapshotFile:
    collection: str
    file_name: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class SnapshotManifest:
    generation: CollectionGeneration
    created_at: datetime
    files: tuple[SnapshotFile, ...]
```

Define `DocumentManifest` and `PassagePoint` with every field in design sections 7.2 and 7.3. Validate SHA-256 as 64 lowercase hex characters and require workspace-relative source paths.

- [ ] **Step 4: Implement deterministic workspace, document, passage, and collection IDs**

```python
def workspace_id_for(root: Path) -> str:
    return hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()[:16]


def document_id_for(workspace_id: str, relative_source_path: str) -> str:
    _validate_relative_path(relative_source_path)
    return str(uuid.uuid5(DOCUMENT_NAMESPACE, f"{workspace_id}:{relative_source_path}"))


def passage_id_for(document_id: str, revision: str, chunk_index: int) -> str:
    return str(uuid.uuid5(PASSAGE_NAMESPACE, f"{document_id}:{revision}:{chunk_index}"))
```

Compute the collection fingerprint from canonical JSON containing collection schema version, chunk schema version, provider/model/dimension, prefixes, normalization, and sparse model. Exclude reranker identity and secrets.

- [ ] **Step 5: Implement Qdrant client construction and collection schema**

`QdrantLiteratureStore.from_config(config)` constructs one `AsyncQdrantClient` with URL, optional API key resolved at call time, timeout, and `prefer_grpc=False` for the first implementation. Create:

- vectorless documents collection with on-disk payload;
- passages collection with named `dense` cosine vector on disk and `sparse_bm25` with IDF modifier;
- single shard and replication factor one;
- required keyword/integer/datetime/text payload indexes;
- strict mode for unindexed filtering;
- generation metadata as a vectorless control point in the documents collection.

If physical collections exist, validate schema and fingerprint. Never drop or recreate an incompatible collection automatically.

- [ ] **Step 6: Implement alias switching and mismatch protection**

Add:

```python
async def switch_current_generation(self, generation: CollectionGeneration) -> None: ...
async def resolve_current_generation(self) -> CollectionGeneration | None: ...
async def validate_current_generation(self, expected_fingerprint: str) -> None: ...
```

Switch both documents and passages aliases in one `update_collection_aliases` request. Retain previous physical collections. Raise stable errors `collection_missing`, `schema_mismatch`, and `model_fingerprint_mismatch`; never mutate aliases during read-only validation.

- [ ] **Step 7: Implement bounded document/passage operations**

Provide async methods:

```python
list_document_manifests(workspace_id: str) -> dict[str, DocumentManifest]
upsert_document(manifest: DocumentManifest, *, wait: bool = True) -> None
upsert_passages(points: Sequence[PassagePoint], *, batch_size: int) -> None
count_revision(document_id: str, revision: str, state: IngestState) -> int
set_revision_state(document_id: str, revision: str, state: IngestState) -> None
delete_other_revisions(document_id: str, keep_revision: str) -> None
delete_document_passages(document_id: str) -> None
hybrid_candidates(query: str, dense: Sequence[float], *, workspace_id: str, limit: int) -> list[SearchCandidate]
retrieve_passages(workspace_id: str, passage_ids: Sequence[str]) -> list[PassagePoint]
dense_candidates(dense: Sequence[float], *, workspace_id: str, limit: int) -> list[SearchCandidate]
sparse_candidates(query: str, *, workspace_id: str, limit: int) -> list[SearchCandidate]
```

All filters include `workspace_id`. Hybrid search creates dense and `qdrant/bm25` sparse prefetches and `FusionQuery(Fusion.RRF)`, filtering `record_type=passage` and `ingest_state=ready`. Never expose a raw query/filter passthrough.

`dense_candidates` and `sparse_candidates` reuse the identical ready/workspace filters and candidate cap so Task 4 can degrade one failed retrieval route without loading the corpus.

- [ ] **Step 8: Implement snapshot download contracts**

Add `create_current_snapshots(output_dir: Path) -> SnapshotManifest`. Resolve aliases first, request one snapshot per physical collection, download to temporary files, calculate SHA-256, atomically rename, and write a manifest last. Reject paths outside the workspace at the CLI/service boundary; the store receives an already-resolved path.

- [ ] **Step 9: Run store tests and type checking**

Run: `uv run pytest -q tests/test_qdrant_store.py`

Run: `uv run mypy src/photomatagent/scientific/capabilities/literature/qdrant_store.py src/photomatagent/scientific/capabilities/literature/models.py`

Expected: all pass with an injected fake async client and no Docker/network.

- [ ] **Step 10: Commit Task 2**

```bash
git add src/photomatagent/scientific/capabilities/literature/qdrant_store.py src/photomatagent/scientific/capabilities/literature/models.py tests/test_qdrant_store.py
git commit -m "feat: add Qdrant literature store"
```

---

### Task 3: Resumable and Failure-Safe Literature Ingestion

**Files:**
- Create: `src/photomatagent/scientific/capabilities/literature/ingestion.py`
- Create: `tests/test_rag_ingestion.py`
- Modify: `src/photomatagent/scientific/capabilities/literature/parser.py`
- Modify: `src/photomatagent/scientific/capabilities/literature/models.py`
- Modify: `src/photomatagent/scientific/capabilities/literature/qdrant_store.py`

**Interfaces:**
- Consumes: Task 1 `EmbeddingProvider`; Task 2 store/ID/contracts; `parse_pdf(path)`.
- Produces: `IngestionPlan`, `IngestionRunState`, `IngestionStats`, `LiteratureIngestionService.plan()`, and `LiteratureIngestionService.index_batch()`.

- [ ] **Step 1: Write failing plan classification and deletion safety tests**

```python
async def test_plan_classifies_new_changed_unchanged_and_deleted(tmp_path, store):
    root = tmp_path / "papers"
    root.mkdir()
    (root / "same.pdf").write_bytes(b"same")
    (root / "changed.pdf").write_bytes(b"new")
    store.manifests = manifests_for_same_changed_and_missing(root)
    plan = await LiteratureIngestionService(store, FAKE_EMBEDDER).plan(root, WORKSPACE)
    assert [item.kind for item in plan.items] == ["changed", "deleted", "unchanged"]


async def test_missing_root_never_deletes_existing_documents(tmp_path, store):
    service = LiteratureIngestionService(store, FAKE_EMBEDDER)
    with pytest.raises(RagIngestionError) as exc:
        await service.plan(tmp_path / "missing", WORKSPACE)
    assert exc.value.code == "source_root_missing"
    assert store.delete_calls == []
```

- [ ] **Step 2: Write failing staged-state recovery tests**

```python
async def test_embedding_failure_writes_no_passages(service, store, failing_embedder, pdf):
    result = await service.index_batch(plan_for(pdf), max_documents=1)
    assert result.failed == 1
    assert store.passage_upserts == []
    assert store.documents[pdf_id].status is DocumentStatus.FAILED


async def test_resume_cleans_staged_revision_and_retries(service, store, pdf):
    store.seed_staged(pdf, revision="a" * 64, count=2)
    result = await service.index_batch(plan_for(pdf), max_documents=1)
    assert result.indexed == 1
    assert store.visible_revisions(pdf.document_id) == {pdf.sha256}
```

- [ ] **Step 3: Run ingestion tests and confirm failures**

Run: `uv run pytest -q tests/test_rag_ingestion.py`

Expected: FAIL because ingestion contracts/service do not exist.

- [ ] **Step 4: Add immutable planning and progress models**

Implement:

```python
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


@dataclass(frozen=True)
class IngestionPlan:
    workspace_id: str
    generation: CollectionGeneration
    items: tuple[IngestionPlanItem, ...]


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


@dataclass(frozen=True)
class IngestionRunState:
    run_id: str
    workspace_id: str
    generation_fingerprint: str
    relative_root: str
    cursor: str | None
    status: str
    stats: IngestionStats
```

Cap persisted/returned errors at 20, and cap each message after redaction.

- [ ] **Step 5: Implement read-only planning**

`plan(root, workspace)` must resolve/validate the root before listing, recursively enumerate only regular `*.pdf` files, sort normalized relative paths, calculate streaming SHA-256, compare current manifests, and append deletions only after enumeration completes. It must not parse PDFs, generate embeddings, create collections, or write Qdrant data.

- [ ] **Step 6: Convert parser output to revision-specific passage points**

Keep Docling parsing thin. Add a pure converter that takes `(PaperRecord, list[PassageRecord], workspace_id, relative_path, revision, model_fingerprint)` and emits `PassagePoint` values with stable UUIDs and corrected previous/next UUIDs. Do not use absolute paths in point payloads.

- [ ] **Step 7: Implement bounded index batches**

Extend `QdrantLiteratureStore` with the exact control-plane methods required by the service:

```python
upsert_ingestion_run(run: IngestionRunState, *, wait: bool = True) -> None
get_ingestion_run(run_id: str, workspace_id: str) -> IngestionRunState | None
delete_staged_revisions(document_id: str, *, keep_revision: str | None = None) -> int
```

These methods operate only on `record_type=ingestion_run` control points in the documents collection and staged passage revisions; they never store PDF text or secrets.

`index_batch(plan, *, run_id=None, resume_cursor=None, max_documents=20)` must:

1. recover/create an `ingestion_run` control point;
2. process at most `max_documents` non-unchanged items after the cursor;
3. parse a single PDF and hold only that document's chunks in memory;
4. call `embed_documents` in configured batches;
5. validate every vector before the first passage upsert;
6. upsert all points as staged and verify exact revision count;
7. promote the new revision to ready;
8. supersede/delete other revisions;
9. update the document manifest to ready;
10. persist the next cursor after every document;
11. isolate and record per-document failures;
12. process deletions only from a successfully constructed complete plan.

Use `try/except Exception` only at the per-document boundary, convert to a typed/redacted diagnostic, and never catch cancellation as an ordinary failure.

- [ ] **Step 8: Implement crash-window dedupe support and cleanup**

Store `normalized_text_sha256` in passage payload. On batch start, remove abandoned staged revisions older than the run or retry the same revision idempotently. Promotion is ready-first; if cleanup fails after promotion, retain both ready revisions and mark the run retryable so retrieval can dedupe without a no-result window.

- [ ] **Step 9: Run ingestion, parser, and existing literature tests**

Run: `uv run pytest -q tests/test_rag_ingestion.py tests/test_literature_rag.py`

Run: `uv run mypy src/photomatagent/scientific/capabilities/literature/ingestion.py src/photomatagent/scientific/capabilities/literature/parser.py`

Expected: new ingestion tests pass; existing tests may still exercise the old adapter until Task 5 but parser/evidence tests remain green.

- [ ] **Step 10: Commit Task 3**

```bash
git add src/photomatagent/scientific/capabilities/literature/ingestion.py src/photomatagent/scientific/capabilities/literature/parser.py src/photomatagent/scientific/capabilities/literature/models.py src/photomatagent/scientific/capabilities/literature/qdrant_store.py tests/test_rag_ingestion.py
git commit -m "feat: add resumable literature ingestion"
```

---

### Task 4: Bounded Hybrid Retrieval and Provider Degradation

**Files:**
- Replace: `src/photomatagent/scientific/capabilities/literature/retrieval.py`
- Create: `tests/test_rag_retrieval.py`

**Interfaces:**
- Consumes: Task 1 providers; Task 2 `QdrantLiteratureStore` and `SearchCandidate`.
- Produces: `RetrievalDiagnostics`, `RetrievedPassage`, and `LiteratureRetriever.search()`.

- [ ] **Step 1: Write failing hybrid, dedupe, context, and degradation tests**

```python
async def test_search_never_loads_the_corpus(store, embedder, reranker):
    store.hybrid_results = candidate_fixture(60)
    results = await LiteratureRetriever(store, embedder, reranker).search(
        "HgTe infrared detector", workspace_id="ws", top_k=5
    )
    assert len(results.passages) == 5
    assert store.hybrid_limit == 50
    assert not hasattr(store, "all_passages")


async def test_reranker_failure_returns_rrf_with_diagnostic(store, embedder):
    reranker = FailingReranker("timeout")
    result = await LiteratureRetriever(store, embedder, reranker).search(
        "query", workspace_id="ws", top_k=3
    )
    assert result.passages
    assert result.diagnostics.mode == "hybrid_rrf"
    assert "reranker_unavailable" in result.diagnostics.degraded_reasons


async def test_dense_failure_uses_sparse_without_external_fallback(store):
    result = await LiteratureRetriever(store, FailingLocalEmbedder(), DisabledReranker()).search(
        "query", workspace_id="ws", top_k=3
    )
    assert result.diagnostics.mode == "sparse_only"
    assert store.sparse_queries == 1
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `uv run pytest -q tests/test_rag_retrieval.py`

Expected: FAIL because the old retriever depends on LanceDB and in-memory BM25.

- [ ] **Step 3: Replace the retriever contracts**

```python
@dataclass(frozen=True)
class RetrievalDiagnostics:
    mode: str
    candidate_count: int
    reranked: bool
    degraded_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalResult:
    passages: tuple[RetrievedPassage, ...]
    diagnostics: RetrievalDiagnostics


class LiteratureRetriever:
    async def search(
        self, query: str, *, workspace_id: str, top_k: int = 5,
        expand_radius: int = 1, context_chars: int = 300
    ) -> RetrievalResult: ...
```

Reject empty queries and enforce top_k 1–10 in the application layer as defense in depth.

- [ ] **Step 4: Implement Qdrant hybrid retrieval**

Embed the query once, call the store with dense and raw query for Qdrant BM25, request at most 50 candidates per route, fuse in Qdrant with RRF, and never request the full collection. If dense generation fails, call a separate store sparse-only method; if sparse fails after dense succeeds, call dense-only. If both fail, raise `RagRetrievalError("retrieval_unavailable", ...)`.

- [ ] **Step 5: Implement deterministic dedupe and reranking**

Deduplicate candidates by `(document_id, normalized_text_sha256)`, preferring the highest fused score and newest indexed timestamp. Rerank no more than 50 texts. Validate reranker indexes before applying them. On reranker error, preserve fused ordering and append `reranker_unavailable`; do not return a fake score of 1.0.

- [ ] **Step 6: Implement bounded neighbor expansion**

Collect previous/next passage IDs only for final top_k rows, retrieve them in one bounded store call, ensure every neighbor shares workspace/document/revision and ready state, and truncate before/after context independently to `context_chars`. Missing neighbors produce empty context, not an error.

- [ ] **Step 7: Run retrieval tests and type checking**

Run: `uv run pytest -q tests/test_rag_retrieval.py`

Run: `uv run mypy src/photomatagent/scientific/capabilities/literature/retrieval.py`

Expected: all pass without importing LanceDB or loading all passages.

- [ ] **Step 8: Commit Task 4**

```bash
git add src/photomatagent/scientific/capabilities/literature/retrieval.py tests/test_rag_retrieval.py
git commit -m "feat: retrieve literature through Qdrant hybrid search"
```

---

### Task 5: Literature Tools, Probe, CLI, Docker, and LanceDB Removal

**Files:**
- Create: `src/photomatagent/cli/rag.py`
- Create: `tests/test_rag_cli.py`
- Create: `compose.qdrant.yaml`
- Modify: `src/photomatagent/scientific/capabilities/literature/__init__.py`
- Modify: `src/photomatagent/cli/app.py`
- Modify: `src/photomatagent/cli/commands.py`
- Modify: `tests/test_literature_rag.py`
- Modify: `tests/test_capability_probe.py`
- Modify: `.env.example`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Delete: `src/photomatagent/scientific/capabilities/literature/index.py`

**Interfaces:**
- Consumes: Tasks 1–4 application services.
- Produces: final model tool adapters, `rag` Typer commands, `/rag` slash routing, fixed Docker service, and dependency cleanup.

- [ ] **Step 1: Rewrite failing tool round-trip tests against fake services**

Replace LanceDB-specific construction in `tests/test_literature_rag.py` with injected fake store/providers and assert the public contract:

```python
async def test_tools_index_search_and_read_keep_public_contract(tmp_path, services):
    index_result = await LiteratureIndexPapersTool(CONFIG, WORKSPACE, services).execute(
        {"max_documents": 2}
    )
    assert index_result.data["run_id"]
    assert index_result.data["complete"] is True

    search_result = await LiteratureSearchPassagesTool(CONFIG, WORKSPACE, services).execute(
        {"query": "HgTe quantum dot infrared detector", "top_k": 3}
    )
    row = search_result.data["results"][0]
    assert set(("passage_id", "paper_id", "title", "passage", "section", "page", "score", "source")) <= row.keys()
    assert len(row["passage"]) <= 600
```

Add tests that tool `cost_class` is `EXPENSIVE`, exposure remains `DEFERRED`, missing Qdrant returns `qdrant_unreachable`, and no adapter exposes raw Qdrant JSON.

- [ ] **Step 2: Write failing CLI and slash-router tests**

Use `typer.testing.CliRunner` and injected/fake service factories to test:

```python
def test_rag_status_hides_api_key(cli_runner, monkeypatch):
    monkeypatch.setenv("QDRANT_API_KEY", "secret-value")
    result = cli_runner.invoke(app, ["rag", "status"])
    assert result.exit_code == 0
    assert "secret-value" not in result.stdout


def test_rag_index_external_requires_confirmation(cli_runner, external_config):
    result = cli_runner.invoke(app, ["rag", "index"], input="n\n")
    assert result.exit_code != 0
    assert "发送全文片段" in result.stdout
```

Assert `/rag status` routes to the same Typer group through `ChatCommandRouter`.

- [ ] **Step 3: Run tool and CLI tests and confirm failure**

Run: `uv run pytest -q tests/test_literature_rag.py tests/test_rag_cli.py tests/test_capability_probe.py`

Expected: FAIL until tools/probe/CLI are rewired.

- [ ] **Step 4: Make tools thin adapters over shared services**

Create one service factory from config/workspace that builds providers lazily, creates the Qdrant store, and returns ingestion/retrieval services. Allow dependency injection in tests. Tool behavior:

- `index_papers`: resolve optional directory, pass `run_id/resume_cursor/max_documents`, return capped stats/errors;
- `search_passages`: preserve legacy result keys, add bounded `diagnostics`;
- `read_passage`: exact ready passage lookup;
- `extract_evidence`: exact passage lookup followed by existing deterministic extraction.

Do not construct collections in search/read. Do not catch and erase typed error codes.

- [ ] **Step 5: Rewrite `LiteratureProbe` as a non-mutating layered probe**

Probe imports without downloading models. Use a short bounded Qdrant health/version request, resolve source root read-only, resolve aliases read-only, and map failures to `MISSING_DEPENDENCY`, `UNCONFIGURED`, or `ERROR`. `tools()` remains available as deferred adapters even when the backend is unavailable so the base registry can start and return guidance.

- [ ] **Step 6: Implement the `rag` Typer group and slash routing**

Implement exact commands from the spec:

```text
rag status
rag plan [--directory PATH]
rag index [--directory PATH] [--run-id ID] [--yes]
rag search QUERY [--top-k N]
rag read PASSAGE_ID
rag evaluate
rag snapshot [--output PATH]
```

All commands build the same application services. `plan` stays read-only. `index` loops bounded service batches and persists cursor after each document; Ctrl-C prints the run ID and exits nonzero without deleting ready data. When external embedding/reranking is configured, print provider/model/data scope and require confirmation unless `--yes` is present. Add `rag` to `_CLI_GROUPS`, default it to `status`, and update `/help`.

- [ ] **Step 7: Add the safe Docker Compose service**

Create:

```yaml
services:
  qdrant:
    image: qdrant/qdrant:v1.18.2
    restart: unless-stopped
    ports:
      - "127.0.0.1:6333:6333"
      - "127.0.0.1:6334:6334"
    volumes:
      - photomat_qdrant_data:/qdrant/storage

volumes:
  photomat_qdrant_data:
```

Do not mount the repository, PDF directory, or `.env` into the container.

- [ ] **Step 8: Replace dependencies and environment documentation**

In `pyproject.toml`, replace `lancedb` and `pylance` with `qdrant-client>=1.19,<2`; retain parsing and local model dependencies. Update `.env.example` with the exact variables from spec section 9, comments explaining external data transfer, and empty key values. Run `uv lock` to update `uv.lock`; inspect the lock diff to ensure LanceDB/pylance are absent unless required transitively by an unrelated dependency.

- [ ] **Step 9: Delete LanceDB runtime code and assert no runtime references**

Delete `literature/index.py`, remove custom `_Bm25`, and update imports/descriptions. Run:

```bash
rg -n "lancedb|pylance|LiteratureIndex|LITERATURE_INDEX_DIR|all_passages|class _Bm25" src pyproject.toml .env.example tests
```

Expected: no runtime/dependency hits; tests may mention LanceDB only when asserting legacy artifact guidance. Do not remove `output/literature_index`.

- [ ] **Step 10: Run all affected tool, probe, CLI, permission, and surface tests**

Run: `uv run pytest -q tests/test_literature_rag.py tests/test_rag_cli.py tests/test_capability_probe.py tests/test_permissions.py tests/test_tool_surface.py`

Run: `uv run mypy src/photomatagent/cli/rag.py src/photomatagent/scientific/capabilities/literature`

Expected: all pass.

- [ ] **Step 11: Commit Task 5**

```bash
git add compose.qdrant.yaml .env.example pyproject.toml uv.lock src/photomatagent/cli src/photomatagent/scientific/capabilities/literature tests/test_literature_rag.py tests/test_rag_cli.py tests/test_capability_probe.py
git commit -m "feat: replace LanceDB literature RAG with Qdrant"
```

---

### Task 6: Docker Integration, Frozen Retrieval Evaluation, and Capacity Benchmark

**Files:**
- Create: `tests/test_qdrant_rag_integration.py`
- Create: `tests/fixtures/literature_rag_eval.json`
- Create: `scripts/benchmark_qdrant_rag.py`
- Create: `docs/qdrant_rag_operations.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: completed Qdrant services and CLI.
- Produces: opt-in real-Docker verification, deterministic retrieval-quality report, scalable benchmark entry point, and operator documentation.

- [ ] **Step 1: Add an opt-in real Qdrant integration fixture**

Skip unless `PHOTOMATAGENT_RUN_QDRANT_INTEGRATION=1`. The fixture connects only to `PHOTOMATAGENT_QDRANT_TEST_URL` defaulting to `http://127.0.0.1:6333`, uses a unique prefix containing the test run UUID, and deletes only those exact test collections in teardown. It must refuse to run if the prefix does not start with `photomat_test_`.

- [ ] **Step 2: Write integration tests for the server/client contract**

Cover:

```python
async def test_qdrant_generation_roundtrip(real_store): ...
async def test_vectorless_document_point(real_store): ...
async def test_dense_sparse_rrf_and_filters(real_store): ...
async def test_staged_points_are_not_visible(real_store): ...
async def test_alias_pair_switches_to_validated_generation(real_store): ...
async def test_snapshot_download_and_restore_noncurrent(real_store, tmp_path): ...
```

Use synthetic texts and deterministic 8-dimensional fake vectors. Never call real embedding/rerank APIs.

- [ ] **Step 3: Run Docker integration tests**

Run: `docker compose -f compose.qdrant.yaml up -d`

Run: `PHOTOMATAGENT_RUN_QDRANT_INTEGRATION=1 uv run pytest -q tests/test_qdrant_rag_integration.py`

Expected: all pass against server 1.18.2. If Docker is unavailable, record the exact unverified gate; do not replace this with Qdrant local mode.

- [ ] **Step 4: Create a licensed deterministic retrieval fixture**

Write at least 20 authored query judgments in `tests/fixtures/literature_rag_eval.json`. Each row has:

```json
{
  "query": "HgTe 量子点探测器在 80 K 的响应率是多少？",
  "relevant_passage_ids": ["fixture-hgte-performance"],
  "category": "cross_lingual_numeric"
}
```

Generate fixture passages in tests rather than copying paper prose. Cover formulae, abbreviations, wavelength bands, units, Chinese-English semantics, metadata filters, synonyms, and irrelevant queries.

- [ ] **Step 5: Implement `rag evaluate` and its tests**

Calculate Recall@5, MRR@10, no-result rate, duplicate rate, and provenance completeness. Return nonzero if Recall@5 < 0.90, provenance completeness < 1.0, or ready-result duplicate rate > 0.0. Label the report as fixture-specific and never claim corpus-wide quality.

- [ ] **Step 6: Add the capacity benchmark script**

`scripts/benchmark_qdrant_rag.py` accepts `--points`, `--dimension`, `--queries`, `--batch-size`, `--url`, and a mandatory safe test prefix. Defaults are 100,000 points, 384 dimensions, 100 queries, and batch 256. It generates vectors deterministically by seeded NumPy, records cold/warm Qdrant candidate p50/p95, process RSS, server/collection config, and emits JSON under `user_output/qdrant-benchmark/`. A `--points 1000000` run is explicit and never automatic in pytest.

- [ ] **Step 7: Write operator documentation and README quickstart**

Document install, Compose start/stop, local defaults, external data warning, `rag status/plan/index/evaluate/snapshot`, restore-to-noncurrent procedure, model fingerprint rebuild rule, reranker-only changes, Docker volume location, source PDF backup distinction, typed error troubleshooting, and legacy LanceDB cleanup guidance. State that 10,000-paper performance is unverified until the 1M benchmark runs.

- [ ] **Step 8: Run documentation-adjacent and integration tests**

Run: `uv run pytest -q tests/test_rag_cli.py tests/test_qdrant_rag_integration.py`

Run the integration suite with its environment gate if Docker is available. Run: `uv run photomatagent rag --help` and `uv run photomatagent rag status`.

- [ ] **Step 9: Commit Task 6**

```bash
git add tests/test_qdrant_rag_integration.py tests/fixtures/literature_rag_eval.json scripts/benchmark_qdrant_rag.py docs/qdrant_rag_operations.md README.md
git commit -m "test: verify Qdrant RAG operations and quality"
```

---

### Task 7: Repository-Wide Verification and Handoff

**Files:**
- Modify only files required to correct failures introduced by Tasks 1–6.

**Interfaces:**
- Consumes: all completed tasks.
- Produces: evidence-backed completion report with explicit unverified performance gates.

- [ ] **Step 1: Run focused architectural boundary tests**

Run:

```bash
uv run pytest -q \
  tests/test_rag_config.py \
  tests/test_rag_providers.py \
  tests/test_qdrant_store.py \
  tests/test_rag_ingestion.py \
  tests/test_rag_retrieval.py \
  tests/test_literature_rag.py \
  tests/test_rag_cli.py \
  tests/test_capability_probe.py \
  tests/test_permissions.py \
  tests/test_tool_surface.py
```

Expected: all pass.

- [ ] **Step 2: Run the complete test suite**

Run: `uv run pytest -q`

Expected: all pass with Docker integration tests skipped only when their explicit environment gate is absent.

- [ ] **Step 3: Run full static typing**

Run: `uv run mypy src`

Expected: no errors.

- [ ] **Step 4: Run live local Qdrant verification when Docker is available**

Run:

```bash
docker compose -f compose.qdrant.yaml up -d
PHOTOMATAGENT_RUN_QDRANT_INTEGRATION=1 uv run pytest -q tests/test_qdrant_rag_integration.py
uv run photomatagent rag status
```

Expected: integration tests pass and status shows the pinned server, current aliases/provider fingerprint, and no secrets. Do not index real papers during verification.

- [ ] **Step 5: Verify LanceDB removal and user-data preservation**

Run:

```bash
rg -n "lancedb|pylance|LiteratureIndex|LITERATURE_INDEX_DIR|all_passages|class _Bm25" src pyproject.toml .env.example tests
test -d output/literature_index && echo "legacy index preserved"
git status --short
```

Expected: no runtime LanceDB hits; the legacy directory still exists if it existed before implementation; only intended source/test/doc changes appear.

- [ ] **Step 6: Inspect final diff integrity**

Run:

```bash
git diff --check
git diff --stat b18f355..HEAD
git status --short
```

Inspect every unexpected or unrelated path before proceeding. Never reset user changes.

- [ ] **Step 7: Commit any verification-only corrections**

If Steps 1–6 required corrections, commit only those corrections:

```bash
git add path/to/each/corrected/file
git commit -m "fix: complete Qdrant RAG verification"
```

Replace the illustrative paths with the exact files inspected and corrected. If no corrections were required, do not create an empty commit.

- [ ] **Step 8: Produce the final evidence report**

Report exact pytest totals, skipped Docker gates, mypy result, integration result, `git diff --check`, Qdrant server/client versions, legacy artifact preservation, and whether the optional 1M benchmark ran. Never claim 10,000-paper performance validation unless that benchmark completed on the target workstation.
