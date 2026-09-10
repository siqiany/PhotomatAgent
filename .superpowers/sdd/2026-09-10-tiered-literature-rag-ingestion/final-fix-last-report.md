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
  leaves the previous active state intact. Pending runs can be resumed or
  explicitly discarded with `--discard-pending`, which removes only the local
  pending pointers.
- Explicit abstract supersession validates the old run's workspace,
  generation, source kind, and bounded source path while allowing a changed
  source path. It cleans only source-scoped non-ready artifacts, retains READY
  knowledge, preserves run lineage, and marks the old run superseded only
  after the replacement completes.
- A completed replacement run now retries source-scoped cleanup and its
  supersession mark on resume after an interruption; an already superseded old
  run is a safe no-op.
- Pending promotion recovers the current active abstract run when the caller
  has no previous ID, archives a different active ID before replacement, and
  avoids duplicating that archive when promotion is retried.

## Verification

- Targeted recovery and changed-source regressions — 2 passed
- Crash-window regressions — 2 passed
- Focused retrieval/import/service tests excluding the optional Qdrant
  roundtrip — 57 passed, 1 deselected
- The complete focused-file invocation reached 57 passed and 1 failure only
  because `qdrant-client` is unavailable in this environment
- `bash -n scripts/import_literature_qdrant.sh` — passed
- Targeted mypy on both modified Python modules — no issues
- `git diff --check` — passed

No live Qdrant, corpus import, network, or model execution was used.
