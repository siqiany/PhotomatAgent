# Final review fix report — literature tier continuity and opaque IDs

## Scope

This review fix is limited to the model-facing literature capability and its
focused tests:

- `src/photomatagent/scientific/capabilities/literature/__init__.py`
- `tests/test_literature_rag.py`

The abstract ingestion, Qdrant adapter, CLI, driver, and their tests remain
owned by the other workers.

## Findings addressed

- `literature.read_passage` and `literature.extract_evidence` now expose and
  validate `source_kind` (`fulltext|abstract`), defaulting to `fulltext` for
  compatibility, and pass the selected tier into `retrieve_passages`.
- Abstract passage reads fail closed when the returned source tag is missing or
  mismatched, preserving `source_kind=abstract`, `abstract_only`, and
  `fulltext_not_checked` in the read payload.
- Evidence extracted from passage IDs carries source tier, exact source record
  ID, relative source path, bibliographic fields, and source limitations into
  `ScientificEvidence.provenance` and `ScientificEvidence.limitations`. The
  generic “source PDF” wording is changed for abstract evidence so it does not
  imply full-text inspection.
- `source_record_id` is treated as opaque: leading, trailing, and internal
  whitespace are preserved in search, read, and evidence outputs. IDs longer
  than the 300-character public bound fail closed instead of being truncated.

## TDD evidence

### RED

After adding the focused schema and search → read/evidence regression, before
the production fix:

```text
PYTHONPATH=src /home/shiqiany/AIagent/PhomatAgent/.venv/bin/python -m pytest -q \
  tests/test_literature_rag.py \
  -k 'read_and_evidence_schemas_default_to_fulltext or abstract_search_id_round_trips_read_and_evidence_provenance'
```

Observed result:

```text
3 failed, 18 deselected
```

The failures were the missing read/evidence schema fields and the normalized
`source_record_id` (`'raw key with spaces'` instead of the raw whitespace-
preserving key).

### GREEN

The focused file after the implementation:

```text
PYTHONPATH=src /home/shiqiany/AIagent/PhomatAgent/.venv/bin/python -m pytest -q tests/test_literature_rag.py
# 21 passed in 2.27s
```

## Verification

```text
PYTHONPATH=src /home/shiqiany/AIagent/PhomatAgent/.venv/bin/python -m mypy \
  src/photomatagent/scientific/capabilities/literature/__init__.py
# Success: no issues found in 1 source file

PYTHONPATH=src /home/shiqiany/AIagent/PhomatAgent/.venv/bin/python -m compileall -q \
  src/photomatagent/scientific/capabilities/literature
# exit 0

git diff --check -- \
  src/photomatagent/scientific/capabilities/literature/__init__.py \
  tests/test_literature_rag.py
# exit 0
```

The nearby source-kind retrieval selector also passed (`1 passed, 19
deselected`). No full suite, live Qdrant, real corpus ingestion, model
download, external provider/API, or arXiv request was run.

## Self-review

- Omitted `source_kind` remains fulltext for existing callers; abstract calls
  must opt into the abstract tier and cannot silently fall back to fulltext.
- Returned tier tags are checked before serialization. Legacy records with no
  tag remain readable only through the compatibility-default fulltext path.
- Opaque IDs are never passed through `_clean`; over-bound IDs are rejected,
  so no ambiguous truncated identifier is exposed.
- Evidence provenance is enriched only for passage-ID inputs; direct text
  inputs retain their existing behavior and are not assigned invented source
  records.
- The unrelated uncommitted changes visible in the shared worktree were not
  staged.
