# Task 4 report — model-visible tiered literature retrieval policy

## Implementation

- Added `source_kind` to `literature.search_passages` with the explicit
  `fulltext|abstract` enum and a backward-compatible `fulltext` default.
- Validated the requested tier at the tool boundary, passed it to the
  retriever, and returned the selected tier plus bounded provenance fields in
  each public result (and the result envelope).
- Ensured abstract `RetrievedPassage` values and public passage payloads carry
  the `abstract_only` limitation. They do not claim that full text was
  inspected.
- Added stable local-full-text → local-abstract → separate-arXiv guidance to
  the literature capability and relevant tool descriptions. The arXiv tool
  explicitly states that results are session-only and not persisted.
- Kept arXiv as an independent runtime-visible tool; the local search tool
  does not call or persist arXiv results.

## Files changed

- `src/photomatagent/scientific/capabilities/literature/__init__.py`
- `src/photomatagent/scientific/capabilities/literature/retrieval.py`
- `tests/test_literature_rag.py`
- `tests/test_rag_retrieval.py`

## TDD evidence

### RED

After adding the public contract tests and before implementing Task 4, the
focused tests failed for the expected missing behaviors:

```text
5 failed, 9 deselected
```

The failures covered the missing `source_kind` schema, missing retriever
pass-through, unknown-tier acceptance, absent `not persisted` arXiv wording,
and absent tier-order guidance. The dedicated retrieval regression also
failed because abstract passages had no `abstract_only` limitation:

```text
1 failed, 19 deselected
```

### GREEN

The exact Task 4 selectors pass:

```text
uv run pytest -q tests/test_literature_rag.py \
  -k 'search_schema_defaults_to_fulltext or abstract_result_identifies_source or arxiv_description_forbids_persistence'
# 3 passed, 11 deselected

uv run pytest -q tests/test_rag_retrieval.py -k 'source_kind'
# 1 passed, 19 deselected
```

The additional guidance/validation/abstract-limit tests pass as well:

```text
5 passed, 9 deselected
1 passed, 19 deselected
```

The existing public search/read contract and wrong-source regression pass:

```text
3 passed
```

## Additional checks

- Targeted mypy over the two changed source modules: `Success: no issues found
  in 2 source files`.
- Targeted `compileall` exited `0`.
- `git diff --check` exited `0`.
- No full repository suite, live Qdrant, real corpus ingestion, external
  network/arXiv request, or model download was run.

## Self-review

- The schema default preserves existing callers while making the selected
  retrieval tier explicit to the model.
- Invalid tiers are rejected before service construction or retrieval; no
  hidden external fallback is possible.
- Candidate source filtering remains owned by the Task 1 retriever/store
  path. This task only exposes and propagates the model-selected tier.
- Abstract labeling is enforced at both the typed retrieval result and public
  tool serialization boundaries, while full-text results retain their
  existing limitations.
- Provenance fields are bounded and JSON-compatible; arXiv remains a separate
  network action with no Qdrant write path.

## Review fix round 1

### Findings addressed

- `_public_limitations` now deduplicates limitation keys, recognizes casing and
  separator variants, reserves one of the eight bounded slots, and always
  emits exactly one canonical `abstract_only` entry for abstract results.
- Search result serialization now reads `source_kind` strictly. Missing,
  invalid, or requested-tier-mismatched result tags return the bounded
  `source_kind_invalid` error instead of being relabeled from the request.
  The compatibility fallback remains limited to the non-search passage-read
  serializer.

### TDD evidence

The new overflow/variant and malformed-result tests failed before the fix:

```text
4 failed, 14 deselected
```

After implementation, the same focused review selector passed:

```text
4 passed, 14 deselected
```

The complete focused Task 4 review/contract selector passed:

```text
9 passed, 9 deselected
```

The source-kind retrieval selector passed:

```text
1 passed, 19 deselected
```

The existing public search/read and wrong-source regressions passed:

```text
3 passed
```

### Review verification

- Targeted mypy: `Success: no issues found in 2 source files`.
- Targeted `compileall` exited `0`.
- `git diff --check` exited `0`.
- No full suite, live Qdrant, network/arXiv call, real corpus, or model
  download was run.
