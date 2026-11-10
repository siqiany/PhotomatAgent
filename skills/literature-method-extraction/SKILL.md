---
name: literature-method-extraction
description: Locate a paper version and read anchored methods, tables and formulas to establish task-relevant scientific conditions.
license: MIT
---

# Extract a literature method

1. Identify the DOI or arXiv version. Use `literature.resolve_fulltext` for a public location and `literature.fetch_source` for a supported, accessible source. Record the version and coverage.
2. Use `literature.read_paper` with the registered ID or a PDF filename in `papers/`. Start with the catalog, then select a section or call `literature.read_object` for known object IDs. A `query` prefers the active full-text index for the current revision and reports document text matching when it falls back. Repeat with the same selection and cursor. If the query route or ranked object selection changes, restart from the first page.
3. Bind each reported value to its object, source hash, sample and operating conditions. Preserve ambiguous conditions and unread material as gaps. Use `literature.extract_evidence` with document and object IDs for a numerical extraction candidate.
4. Compare the paper's scale, specimen, equipment and controls with the current task. An abstract, a generic webpage and the model's paraphrase provide leads for further checking.

Long objects can span multiple pages of tool results. Keep the revision and
selector unchanged, and join text using `text_start`. Table entries keep their
row and column metadata in `table_cells`. When a single cell exceeds the byte
budget, join its `table_cell_fragment.json` values by cell index and offset.
Start the selection again if a cursor is stale.

Inspect each fragment of a cross-page table using its own object ID, page and
cells. `table_continuations` identifies either an explicit caption or repeated
headers and matching geometry at a page break. If formula text is unavailable,
use the anchored crop and keep any unverified transcription uncertain. A
partial conversion can be read by object. It is blocked from full-text
indexing. Operators can change the bounded Docling timeout, which produces a
new parse revision.

Local PDF parsing requires configured Docling assets. A pending parse can be
polled with the same document ID. Structured reading can proceed with the search
index offline. An accessible article page can return candidate supplementary links. Verify each
link and read any supporting file separately before citing it.

Use `literature.index_document` only when object text needs cross-document
search. It indexes one parsed PDF revision in the configured Qdrant generation.
If `search_visible` is false, the generation still requires the normal explicit
activation. Search results can return a structured reference with document ID,
revision and object ID. Pass those identifiers to `literature.read_object`.
For an anchored PDF figure or table, set `include_images=true` to request a
bounded crop. Use the accompanying page, object ID and source hash as anchors.

For detector papers, capture ligand chemistry, exchange protocol and transport
conditions together. For film processing, keep substrate, stack, deposition and
annealing conditions tied to the actual specimen.
