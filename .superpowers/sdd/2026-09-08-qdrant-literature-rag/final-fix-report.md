# Qdrant Literature RAG final-fix report

Date: 2026-09-09
Review baseline: `f98bb76`
Branch: `codex/qdrant-rag`

This report records the final-fix wave on the existing working-tree diff. No
`.env`, credentials, `user_input/`, `output/literature_index/`, PDFs, or other
user-generated output was modified.

## Finding closure

1. **Critical generation lifecycle and provider fingerprint**

   `ensure_generation` provisions a fingerprinted physical staging pair without
   changing current aliases. Ingestion selects that staging pair and rejects
   mismatched document/passage/run fingerprints. `activate_generation` and the
   public `switch_current_generation` both pass through the same completeness
   and fingerprint guard before one atomic paired-alias update. Empty
   activation is rejected by default; `rag activate --yes --bootstrap` is the
   explicit, documented first-empty-corpus path. Retrieval validates the
   configured provider/schema fingerprint before querying.

   Evidence: lifecycle, fingerprint, bootstrap, and public-switch bypass tests
   in `tests/test_qdrant_store.py`; staging and resume tests in
   `tests/test_rag_ingestion.py`; fingerprint tests in
   `tests/test_rag_retrieval.py`; CLI activation coverage in
   `tests/test_rag_cli.py`. The empty-generation tests were first run RED
   against the public alias bypass and then GREEN after the guarded path was
   introduced.

2. **Remote Qdrant security and diagnostic redaction**

   Qdrant endpoints reject credentials, query strings, fragments, non-HTTPS
   non-loopback endpoints, and missing non-loopback API keys. Status and errors
   use credential-free URLs and never print key values. Store-boundary
   diagnostics redact key/value secrets, `Authorization: Bearer ...`,
   standalone `Bearer ...`, and `token=...`, remove absolute paths, and cap
   individual messages and persisted error counts.

   Evidence: URL/security tests in `tests/test_qdrant_store.py`, status tests
   in `tests/test_rag_cli.py` and `tests/test_capability_probe.py`, and boundary
   sanitation tests for the bearer/token forms in `tests/test_qdrant_store.py`.

3. **Ingestion failure, retry, and completion semantics**

   A failed or retryable item retains the resume cursor and prevents completion.
   A successful retry clears persisted retry state; a run is complete only
   after unresolved failures are absent. CLI indexing stops on a retryable
   result rather than spinning indefinitely and returns non-zero for incomplete
   work.

   Evidence: final-item failure, retry recovery, cursor, and CLI incomplete-run
   tests in `tests/test_rag_ingestion.py` and `tests/test_rag_cli.py`.

4. **Local model event-loop safety**

   Sentence-transformer and cross-encoder loading plus inference occur in the
   same `asyncio.to_thread` worker boundary. The worker lifecycle uses a bounded
   executor so short provider tests terminate deterministically.

   Evidence: loader/inference thread-identity tests in
   `tests/test_rag_providers.py`; `mypy src` passes.

5. **Complete, order-independent status**

   `LiteratureProbe` performs bounded source, Qdrant connectivity, server
   version, aliases, collection metadata/counts/status, capacity, provider,
   generation, TLS/auth, and legacy-artifact checks independently. Missing
   source roots, optional packages, or a failed version endpoint no longer
   suppress independent Qdrant observations. Capacity fields populate
   `capacity_warning` for low free space/high usage and unhealthy collection
   status.

   Evidence: order/failure-soft status tests and
   `test_literature_probe_reports_capacity_warning_from_server_health` in
   `tests/test_capability_probe.py`, plus CLI rendering tests.

6. **Real isolated evaluation**

   `evaluate_live_fixture` builds a real `LiteratureRetriever`, embeds and
   writes a UUID-safe synthetic fixture to a unique `photomat_test_eval_*`
   physical prefix, activates only that generation, runs every authored
   judgment, and deletes only that prefix. Readable fixture labels map to
   deterministic UUID point IDs accepted by Qdrant. Unavailable Qdrant or model
   dependencies produce `live_evaluation: false`, null quality metrics, and a
   bounded reason rather than fabricated retrieval scores.

   Evidence: unit metric tests in `tests/test_rag_evaluation.py`; opt-in Docker
   test `test_live_fixture_evaluation_uses_real_retriever_and_preserves_aliases`
   asserts all 22 judgments pass and production aliases remain unchanged.

7. **Actual hybrid/RRF and end-to-end benchmark**

   The benchmark creates an isolated dense+sparse collection, writes synthetic
   points, executes Qdrant dense+sparse `Prefetch` with `Fusion.RRF`, and
   records only executed paths. Local query embedding, retrieval, and
   reranking are a separate opt-in path. Warmup reports use
   `first_request_phase` and `first_request_*`/`subsequent_*` fields; they do
   not call a post-warmup local sample “cold”.

   Evidence: dry-run/report-schema tests in `tests/test_benchmark_qdrant_rag.py`;
   opt-in Docker gate `test_live_benchmark_executes_hybrid_rrf_in_isolated_collection`
   verifies real RRF writes/queries, alias preservation, and optional local
   warmup labels.

8. **Streamed snapshot and restore validation**

   Snapshot hashing is incremental. REST and adapter `aiter_bytes`/`iter_bytes`
   responses stream to disk; adapter `Path` results copy in bounded blocks. The
   manifest records server/schema/fingerprint, physical collections, aliases,
   counts, indexed-vector counts, capacity, file hashes/sizes, and creation
   time. Restore validation compares only manifest aliases, so unrelated
   service aliases do not cause false failures while changes to recorded
   aliases remain detectable. It reuses the full collection shape validator,
   including dense dimension, sparse IDF vector, payload indexes, strict mode,
   metadata, counts, generation control metadata, and a ready/fingerprint-
   matching sample passage.

   Evidence: snapshot stream/path, unrelated-alias, count, shape, control, and
   sample tests in `tests/test_qdrant_store.py`; the opt-in Docker restore test
   covers actual Qdrant recovery.

9. **Workspace-root path reconstruction and scan race**

   Ingestion plans carry an explicit plan-root-relative item path. The former
   root-name/basename heuristic is removed, and paths resolve only from the
   explicit plan root. The source is rehashed after parsing and before
   embedding/upsert, so a scan-to-ingest change is persisted as retryable
   `source_changed`.

   Evidence: nested workspace-basename/path-boundary and post-parse rehash
   tests in `tests/test_rag_ingestion.py`.

10. **Strict model-visible output limits**

    Environment-backed result counts, character limits, timeout, batch size,
    top-k, passage length, and vector-related limits use strict bounded integer
    parsing. Values outside the contract are rejected before service
    construction.

    Evidence: boundary tests in `tests/test_rag_config.py` and tool-surface
    limit tests in `tests/test_rag_cli.py`/`tests/test_literature_rag.py`.

## Documentation and removal audit

`README.md` and `docs/qdrant_rag_operations.md` document staging versus
activation, explicit empty bootstrap, live isolated evaluation, snapshot
restore checks, and hybrid/local benchmark options. The removal scan was:

```text
rg -n "lancedb|pylance|LiteratureIndex|LITERATURE_INDEX_DIR|all_passages|class _Bm25" src pyproject.toml .env.example tests
```

It found only intentional compatibility/tool names and a legacy-environment
regression assertion; no runtime LanceDB/LanceIndex implementation remains.

## Verification evidence

- Focused RAG/store/provider groups: `209 passed, 11 skipped, 18 warnings`.
- Final post-residual groups: `104 passed` (store/ingestion/retrieval),
  `72 passed, 8 warnings` (CLI/providers/config/evaluation), and
  `26 passed, 11 skipped, 10 warnings` (probe/benchmark/integration).
- `mypy src`: `Success: no issues found in 233 source files`.
- `uv lock --check --offline`: passed (`Resolved 262 packages`).
- `git diff --check`: passed.

The full suite was attempted with bounded commands. The stable
`--maxfail=1` run reached `608 passed, 1 skipped` before the known unrelated
baseline failure `tests/test_kp.py::test_run_tool_requires_args_or_config`
(kdotpy is unavailable and the result is `external_solver_unavailable`, while
the old assertion expects `requires`). The longer full run was stopped after
the sandbox/asyncio worker-exit hang before a complete summary; no full-suite
green claim is made here. The root agent is responsible for final independent
full-suite accounting and exact remaining baseline-failure identities.

Docker is not available in this WSL distribution (`docker: command not found`),
so the opt-in 22-judgment evaluation, live RRF benchmark, and live snapshot
restore gates remain explicitly unverified here. They are present as opt-in
tests and were not replaced by Qdrant local mode.

## Commit

Implementation/tests/docs commit: `d05b593` (`fix: harden qdrant literature
rag final review`). The report is committed separately because
`.superpowers/` is intentionally ignored by the repository defaults.

At the end of the subagent fix wave, no RAG final-review finding was believed
open. The subsequent scoped re-review and controller corrections are recorded
below.

## Scoped re-review addendum

The single scoped re-review of `f98bb76..8ce9072` returned `NOT READY` with two
valid residuals. Controller-level TDD corrections were applied without opening
another subagent loop:

- Empty bootstrap previously checked only passage readiness, so it could move
  existing current aliases to a new empty generation. The new regression test
  failed first, then passed after bootstrap was restricted to either a service
  with no current aliases or an idempotent revalidation of the same current
  pair. A different or partial current pair is rejected before alias mutation.
- `rag status` previously repeated workspace path resolution after the probe,
  which discarded otherwise useful Qdrant diagnostics for an outside-workspace
  source root. The new CLI regression failed first, then passed after status
  rendering began using the probe's bounded `source_root` state with a guarded
  compatibility fallback.

Post-correction focused verification: `233 passed, 11 skipped`; `mypy src`,
`uv lock --check --offline`, and `git diff --check` passed. The root agent reruns
the complete pytest suite on the final commit and records its exact result in
the user handoff.

Final root verification on commit `612fa88` completed with `1559 passed, 17
skipped, 9 failed, 57 warnings`. The nine failures are the same pre-existing,
out-of-scope baseline group recorded earlier: unavailable kdotpy behavior,
four MAGUS tests requiring ASE, the unrelated tool-catalog effective-mass
ranking, two missing JARVIS archives, and two VASP isosurface expectations. No
RAG-focused test failed. Docker remained unavailable in WSL, so the three
opt-in live Qdrant gates remain explicitly unexecuted.

## Docker live-validation follow-up

After Docker Desktop WSL integration was enabled, Qdrant `1.18.2` started from
`compose.qdrant.yaml` and passed its `/healthz` check. The first live run exposed
three issues that the skipped suite could not reveal:

- Qdrant BM25 may fill a multi-result sparse query with a zero/low-score point;
  the integration assertion now verifies the controlled token-bearing positive-
  score winner without treating a non-winning filler as lexical evidence.
- The generation-switch fixture attempted to activate an empty second
  generation. It now writes fingerprint-matching READY data before using the
  ordinary guarded activation path and verifies both alias movement and old
  physical-generation preservation.
- `qdrant-client 1.19.0` accepts `**kwargs` in its public snapshot signature but
  rejects method-level `timeout` internally. Snapshot creation now uses narrow
  signature capability detection rather than catching and retrying arbitrary
  `TypeError` exceptions. The regression was RED (`1 failed, 1 passed`) before
  the adapter change and GREEN (`2 passed`) afterward.

With all live gates enabled together, `tests/test_qdrant_rag_integration.py`
completed with `13 passed, 23 warnings` in 215.28 seconds. This includes server
version/schema checks, restart persistence, staged visibility, alias switching,
snapshot restore, the real 22-judgment `LiteratureRetriever` evaluation, and
the local-model benchmark gate.

Two disposable capacity runs completed and removed their collections:

- 1,000,000 points, 384 dimensions, 100 hybrid/RRF queries: first query
  1773.60 ms; subsequent p50 140.72 ms and p95 767.39 ms; server status green.
- 100,000 points with warmed local embedding and reranking: Qdrant candidate
  p50 13.98 ms and p95 173.46 ms; end-to-end local p50 75.04 ms and p95
  118.40 ms; first post-warmup request 231.30 ms; server status green.

The machine-specific JSON reports are stored under
`user_output/qdrant-benchmark/`. Qdrant was left running with an empty
collection list; no current production alias or legacy Lance artifact was
modified.
