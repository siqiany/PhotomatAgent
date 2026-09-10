# Task 5 report — resumable WSL literature import driver

## Implementation

- Added `scripts/import_literature_qdrant.sh`, a strict Bash driver with
  `set -euo pipefail`, argv arrays, repository-root execution, and the
  `pdf|abstracts|status|activate` stages.
- The driver defaults to workspace `/home/shiqiany/AIagent`, PDF path
  `Photoelectric detection/dataset/paper/pdf`, and SQLite path
  `Photoelectric detection/dataset/paper/abstract/abstracts.sqlite3`.
- Fresh stage invocations generate and atomically save an ID before invoking
  `uv run photomatagent ...`; `--resume` reuses that exact ID. Stage IDs are
  stored in the repository's `user_output/rag-import/run-state` directory.
- `status` loads both saved IDs. `activate` requires both IDs and forwards both
  explicit stage requirements to the existing CLI; the script has no direct
  Qdrant request, provider client, or API-key handling.
- Added the two minimal script tests from the task brief and documented the
  stage commands, bounded progress fields, confirmation gate, source priority,
  resume state, SQLite-change recovery, and session-only arXiv behavior.

## TDD evidence

### RED

Before creating the driver, the focused script tests were run with the
repository's uv cache redirected to the writable temporary area:

```text
UV_CACHE_DIR=/tmp/photomatagent-uv-cache uv run pytest -q tests/test_import_literature_script.py
```

Result: `2 failed` — the first test could not read the missing script and the
second test received Bash exit status 127 for the missing script.

### GREEN

After implementing the driver and tests:

```text
UV_CACHE_DIR=/tmp/photomatagent-uv-cache uv run pytest -q tests/test_import_literature_script.py
```

Result: `2 passed in 0.06s`.

The fake-`uv` audit also verified that a fresh stage writes its ID before the
fake CLI starts, resume reuses the same ID, paths with spaces remain single
argv elements, status forwards both IDs, and activation forwards both stage
requirements and IDs. No provider or Qdrant service was started.

## Agreed narrow verification

- `tests/test_import_literature_script.py`: `2 passed`.
- `tests/test_abstract_ingestion.py tests/test_import_literature_script.py`:
  `14 passed, 1 failed`; the existing Qdrant round-trip test fails because
  `qdrant-client` is not installed (`ModuleNotFoundError`), outside Task 5.
- Qdrant source-kind selector: `1 passed, 2 failed`; both failures stop at the
  same missing optional `qdrant-client` dependency.
- Ingestion/retrieval source-kind/resume/retry selector: `6 passed`.
- Literature/CLI tiered-policy and staged-CLI selector: `7 passed`.
- Targeted mypy over the literature package and RAG CLI: `Success: no issues
  found in 14 source files`.
- `bash -n scripts/import_literature_qdrant.sh`: passed.
- `git diff --check`: passed.
- `shellcheck` was unavailable in the environment.

The full test suite, real PDF/SQLite corpus, model downloads, live Qdrant,
external providers, external APIs, and live arXiv were not run.
