# Task 6 report — Docker integration, frozen evaluation, and benchmark

## Scope

Implemented the opt-in Qdrant contract suite, synthetic licensed retrieval
judgment fixture, fixture-specific `rag evaluate` report and thresholds, safe
capacity benchmark, operations guide, and README quickstart.

## TDD evidence

RED was observed before the implementation:

```text
./.venv/bin/pytest -q tests/test_rag_evaluation.py
4 failed — fixture file and evaluate_retrieval_fixture/_evaluation_fixture_path were missing.
./.venv/bin/pytest -q tests/test_benchmark_qdrant_rag.py
3 failed — scripts/benchmark_qdrant_rag.py was missing.
```

GREEN after the focused implementation:

```text
./.venv/bin/pytest -q tests/test_rag_evaluation.py tests/test_benchmark_qdrant_rag.py tests/test_qdrant_rag_integration.py tests/test_rag_cli.py
12 passed, 6 skipped
```

The six skipped tests are the explicitly gated live-Docker tests; the prefix
safety unit test passes without Docker.

## Docker verification

Attempted the required Docker gate:

```text
docker --version
/bin/bash: docker: command not found
The command 'docker' was not found in this WSL 2 distro.
We recommend to activate the WSL integration with Docker Desktop.
```

The live command was therefore not run and no Qdrant local mode was used as a
replacement:

```text
docker compose -f compose.qdrant.yaml up -d   # unverified: Docker unavailable
PHOTOMATAGENT_RUN_QDRANT_INTEGRATION=1 ./.venv/bin/pytest -q tests/test_qdrant_rag_integration.py   # unverified
```

The integration fixture defaults to skip, connects only to
`PHOTOMATAGENT_QDRANT_TEST_URL` (default `http://127.0.0.1:6333`), uses a
UUID-bearing `photomat_test_` prefix, and tears down only that exact prefix.

## Benchmark verification

The required dry-run safety check passed:

```text
./.venv/bin/pytest -q tests/test_benchmark_qdrant_rag.py
3 passed
```

The live benchmark was not run because Docker is unavailable. No one-million-
point claim is made. The script defaults to 100,000 points, 384 dimensions,
100 queries, batch 256, dry-run mode, and an explicit safe prefix; live writes
require `--confirm-write`, use a UUID-suffixed non-current collection, and
record server/collection config, hardware/RSS, and cold/warm p50/p95 latency.

## Additional verification

```text
./.venv/bin/pytest -q tests/test_rag_cli.py tests/test_qdrant_store.py tests/test_rag_ingestion.py tests/test_rag_retrieval.py tests/test_literature_rag.py
89 passed

./.venv/bin/mypy src/photomatagent/cli/rag.py scripts/benchmark_qdrant_rag.py tests/test_qdrant_rag_integration.py tests/test_rag_evaluation.py tests/test_benchmark_qdrant_rag.py
Success: no issues found in 5 source files

git diff --check
passed

./.venv/bin/photomatagent rag --help
passed; status/plan/index/search/read/evaluate/snapshot are listed

./.venv/bin/photomatagent rag status
passed; reports UNCONFIGURED because the default dataset/paper source root is absent
```

`10,000-paper performance` remains explicitly unverified until the opt-in
one-million-point benchmark has run against the pinned server.

The repository-wide suite was also attempted with `./.venv/bin/pytest -q -x`:
603 passed, 1 skipped, then the pre-existing environment-dependent
`tests/test_kp.py::test_run_tool_requires_args_or_config` failed because
`kdotpy` is not installed (`external_solver_unavailable` is returned before
the test's legacy `requires` assertion). This failure is outside Task 6 and no
K·p files were changed.

## Task 6 fix round — evaluator and live-contract gaps

This round closes the review findings without broadening production authority:

- The fixture evaluator requests `top_k=10` once per judgment, computes
  Recall@5 from ranks 1–5, and computes MRR@10 from ranks 1–10.  A rank-8
  regression judgment is included and asserts Recall@5 = 0 while MRR@10 = 1/8.
- The frozen fixture contains 22 synthetic judgments, each with explicit
  `fixture_author` and `license` metadata.  The evaluation test routes all 22
  through the actual `LiteratureRetriever` with a deterministic fake embedder,
  reranker, and store.  Its asserted metrics are Recall@5 = 1.0, MRR@10 =
  1.0, no-result rate = 2/22, provenance completeness = 1.0, duplicate rate =
  0.0, and `passed = true`.
- Gated live coverage now exercises dense-only and sparse-only RRF candidates,
  indexed year/source-path filters, rejection of an unindexed expensive filter
  under strict mode, document update/delete, and persistence across an actual
  `docker compose -f compose.qdrant.yaml restart qdrant`.  Teardown and all
  points remain under one UUID-bearing `photomat_test_` prefix; no current
  production aliases are inspected or modified.
- Every gated startup (including `PHOTOMATAGENT_QDRANT_TEST_URL` overrides)
  checks `client.info().version == "1.18.2"` and fails clearly on mismatch.
  Restart health and reconnect polling are bounded to 45 seconds.

RED evidence for this round:

```text
./.venv/bin/pytest -q tests/test_rag_evaluation.py -k rank_boundary
failed before the evaluator fix: the fixture retriever was called with
top_k=5, so the regression assertion observed [5] instead of [10].
```

GREEN evidence after the fixes:

```text
./.venv/bin/pytest -q tests/test_rag_evaluation.py
6 passed

./.venv/bin/pytest -q tests/test_rag_evaluation.py tests/test_benchmark_qdrant_rag.py tests/test_qdrant_rag_integration.py tests/test_rag_cli.py tests/test_qdrant_store.py tests/test_rag_ingestion.py tests/test_rag_retrieval.py tests/test_literature_rag.py
100 passed, 9 skipped

./.venv/bin/mypy src/photomatagent/cli/rag.py scripts/benchmark_qdrant_rag.py tests/test_rag_evaluation.py tests/test_qdrant_rag_integration.py tests/test_benchmark_qdrant_rag.py
Success: no issues found in 5 source files

git diff --check
passed
```

Docker remains unavailable in this WSL environment (`docker --version`:
`/bin/bash: docker: command not found`).  Consequently the nine live tests,
including update/delete, restart persistence, strict-filter rejection, and the
exact server-version check against a real endpoint, remain explicitly skipped
or unverified locally.  No local-mode Qdrant replacement was used, and no
real papers, external providers, or production collections were touched.

## Task 6 fix round 2 — non-tautological RRF coverage

The live RRF/filter test now indexes seven ready points under the test-owned
workspace and requests a fused `limit=2`.  Dense vectors have seven distinct
cosine similarities, so dense-only ordering is deterministic.  Exactly one
point contains the unique sparse query token: the dense-strong/sparse-weak
candidate and the orthogonal sparse-strong/dense-weak candidate must both be
returned, while the five distractors must be excluded.  The assertions are
therefore sensitive to a single-route implementation rather than merely
checking membership in a result set as large as the corpus.  Existing
workspace/ready filtering and indexed year/source-path and strict-mode checks
remain in the same gated test.

Verification for this round:

```text
./.venv/bin/pytest -q tests/test_rag_evaluation.py tests/test_benchmark_qdrant_rag.py tests/test_qdrant_rag_integration.py tests/test_rag_cli.py tests/test_qdrant_store.py tests/test_rag_ingestion.py tests/test_rag_retrieval.py tests/test_literature_rag.py
100 passed, 9 skipped

./.venv/bin/mypy tests/test_qdrant_rag_integration.py
Success: no issues found in 1 source file

git diff --check
passed
```

The nine skips are still the explicit Docker-gated tests.  Docker remains
unavailable locally (`docker --version` reports `/bin/bash: docker: command
not found`), so the strengthened live RRF ordering and filter contract is
recorded as unverified rather than claimed green against a real Qdrant.
