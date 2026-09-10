# Final fix-last report

## Scope

- Retrieval candidates now require an explicitly present string `workspace_id`
  equal to the requested workspace and an explicitly present `ingest_state` of
  `ready`; missing scope/readiness metadata is rejected.
- The abstract import driver records its canonical source path, only
  auto-supersedes when the saved path matches exactly, and requires
  `--supersede-run-id` for changed or unknown sources.
- Fresh abstract runs are written to pending state first. The active run ID is
  archived and promoted only after the CLI accepts the run; failed validation
  leaves the previous active state intact. Pending runs can be resumed.

## Verification

- `uv run pytest -q tests/test_rag_retrieval.py tests/test_import_literature_script.py`
  — 35 passed
- `bash -n scripts/import_literature_qdrant.sh` — passed
- `uv run mypy src/photomatagent/scientific/capabilities/literature/retrieval.py`
  — no issues
- `git diff --check` — passed

No live Qdrant, corpus import, network, or model execution was used.
