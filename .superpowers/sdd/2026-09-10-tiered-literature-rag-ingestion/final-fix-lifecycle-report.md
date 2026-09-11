# Final lifecycle-fix report

Date: 2026-09-10

## RED evidence

The focused regression tests were added before the lifecycle implementation
and initially failed for the review findings: PDF planning could receive a
mixed manifest set; abstract revision cleanup was not invoked; a cancelled
first batch left no run record; unknown explicit abstract resume IDs were not
rejected; SQLite source identity/count work repeated per batch; WAL content was
not fail-closed; and changed-database supersession had no explicit lifecycle.

## GREEN evidence

Using the repository virtual environment and no network, live Qdrant, model, or
corpus tests:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /home/shiqiany/AIagent/PhomatAgent/.venv/bin/python -m pytest -q \
  tests/test_abstract_ingestion.py tests/test_rag_ingestion.py \
  tests/test_qdrant_store.py tests/test_rag_cli.py \
  tests/test_import_literature_script.py
149 passed, 8 warnings
```

Targeted `mypy` over the four changed Python modules passed with no issues.
`bash -n scripts/import_literature_qdrant.sh` and `git diff --check` also pass.

## Design rationale

- PDF manifest reads request `source_kind=fulltext`; both the Qdrant adapter
  and the planner's defensive fallback reject abstract or unknown manifests
  before deletion planning.
- Abstract indexing makes a new revision READY, writes its READY manifest, and
  only then calls revision cleanup. Cleanup errors remain retryable, preserving
  the new revision and retrying on the next batch invocation.
- A fresh abstract run is persisted before source fetch, embedding, or writes.
  `resume=True` performs a strict existing-run lookup and rejects unknown IDs.
- Changed databases use a new run linked with `supersedes_run_id`. The new run
  is persisted first; source-scoped cleanup can remove only non-ready abstract
  artifacts. READY knowledge missing from the replacement source is retained.
  The old run becomes `superseded` and complete only after replacement
  completion, with both IDs retained in control metadata.
- The abstract reader rejects a non-empty SQLite WAL with an actionable typed
  error, holds a read transaction for the reader lifetime, computes identity
  and counts once per service invocation, and validates cheap main/WAL file
  signatures between bounded pages.
- The WSL driver archives an existing stage ID before replacing state. Fresh
  abstract runs forward that prior ID as `--supersede-run-id`; explicit IDs may
  override it, while `--resume` and supersession are mutually exclusive.
