# Qdrant literature RAG operations

This document describes the supported single-host deployment of the
literature RAG capability. Qdrant is an internal application service; it is
not a model-facing tool and it must not be exposed on a public interface.

## Install and start

Install the literature extra in the PhotomatAgent environment:

```bash
uv sync --extra literature
```

Start the pinned local Qdrant server from the repository root:

```bash
docker compose -f compose.qdrant.yaml up -d
uv run photomatagent rag status
```

Stop it when it is no longer needed:

```bash
docker compose -f compose.qdrant.yaml down
```

The Compose file publishes HTTP on `127.0.0.1:6333` and gRPC on
`127.0.0.1:6334` only. It uses the named Docker volume
`photomat_qdrant_data`, mounted at `/qdrant/storage`; it does not mount the
repository, `.env`, or PDF directories into the container. A Qdrant snapshot
is a database backup, not a backup of the source PDFs.

## Local defaults and external data

The default configuration is local and deterministic:

| Setting | Default |
| --- | --- |
| Qdrant URL | `http://127.0.0.1:6333` |
| Collection prefix | `photomat_literature` |
| Embedding | local `intfloat/multilingual-e5-small`, 384 dimensions |
| Reranker | local `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Batch size | 128 passages |

An external embedding or reranking provider is never selected implicitly when
a local model fails. Set `PHOTOMATAGENT_RAG_ALLOW_EXTERNAL=1`, configure the
provider/base URL/model/key environment name, and explicitly confirm
`rag index --yes` after checking the reported directory. Confirmation means
that full-text passage chunks may leave the workspace. Keep secrets in `.env`
or the process environment; never put them in a payload, report, or log.

## Routine commands

All paths are resolved inside the selected workspace. `plan` is read-only;
`index` is resumable and writes bounded progress to Qdrant:

```bash
uv run photomatagent rag status
uv run photomatagent rag plan --directory dataset/paper
uv run photomatagent rag index --directory dataset/paper
uv run photomatagent rag activate --yes
uv run photomatagent rag search "HgTe detector responsivity at 80 K" --top-k 5
uv run photomatagent rag read <passage-id>
uv run photomatagent rag evaluate
uv run photomatagent rag snapshot --output .photomatagent/rag/backups/$(date -u +%Y%m%dT%H%M%SZ)
```

`rag plan` does not create collections or write progress. `rag index` builds
the provider/schema generation in physical staging collections and never
changes the current aliases. Run the explicit `rag activate --yes` gate only
after the complete run and retrieval checks pass. An empty first corpus must
use the additional explicit `rag activate --yes --bootstrap` path; ordinary
activation rejects empty or unresolved generations, and bootstrap cannot
replace an existing current generation. Retrieval validates the
configured model fingerprint before querying. `rag evaluate` creates a
unique, disposable synthetic Qdrant prefix and runs every authored judgment
through `LiteratureRetriever`; it never selects rows from the production
aliases. If Qdrant or the configured models are unavailable, it reports
`live_evaluation: false` and exits nonzero rather than presenting fake quality
metrics. A live report is labelled `fixture-specific` and makes no claim about
quality over a real corpus. The acceptance thresholds are Recall@5 ≥ 0.90,
provenance completeness = 1.0, and duplicate rate = 0.0.

## Generation and model changes

The two current aliases point to a validated documents/passages generation.
Changing the embedding model, provider, dimension, document/query prefixes,
normalization, sparse model, or chunk schema changes the model fingerprint and
requires a new generation and a full rebuild. The old physical collections
remain available until an operator's retention policy removes them. Never
delete a current alias target during a rebuild.

Changing only the reranker does not change the vector fingerprint and does not
require re-embedding. It can change result ordering, so rerun the frozen
evaluation and record the provider/model change in the deployment notes.

## Snapshot and restore to a non-current generation

Create a snapshot of both physical collections with `rag snapshot`. Verify the
`snapshot-manifest.json` SHA-256 and size entries before moving the files. The
manifest records server version, schema/fingerprint, physical collection
names, point and indexed-vector counts, aliases, capacity observations, file
hashes/sizes, and creation time. Hashing is streamed, so archive size is not
loaded wholly into memory. Do not restore a snapshot into a collection with a
different schema or model fingerprint.

Restore each collection to a new, non-current physical name using the Qdrant
snapshot recovery API (or an equivalent Qdrant client operation), for example:

```text
POST /collections/<new-documents-name>/snapshots/recover
{"location":"<Qdrant-server-visible-snapshot-URL>"}
```

Repeat for passages, then use the store restore-validation operation (or an
equivalent checked procedure) to verify counts, collection metadata, payload
indexes, vector dimension, sparse IDF configuration, generation control
metadata, and a sample retrieval. Keep the current aliases untouched while
validating the restored pair; the manifest aliases must remain unchanged.
Switch both aliases in one validated alias-update operation only after
retrieval and the frozen evaluation pass. A scheduler/server `COMPLETED`-style
status is not a scientific retrieval success signal; validate the actual
artifacts and provenance.

## Troubleshooting typed errors

`rag status` is read-only and reports the server, aliases, generation, source
root, and provider state without exposing secrets. Common typed errors include:

| Error code | Meaning | Action |
| --- | --- | --- |
| `qdrant_unreachable` | Server URL/port is unavailable or timed out | Start Compose, verify `127.0.0.1:6333`, then retry |
| `qdrant_auth_failed` | A non-loopback endpoint rejected the configured key | Check TLS/reverse-proxy and key environment name |
| `collection_missing` | Current aliases or physical collections are absent | Run a planned/indexed generation or inspect aliases |
| `model_fingerprint_mismatch` | Provider/schema does not match the current generation | Build and validate a new generation; do not mix vectors |
| `schema_mismatch` | Required vector, sparse, strict-mode, or payload index contract differs | Restore a compatible snapshot or rebuild the generation |
| `source_root_invalid` | A directory escapes the workspace | Use a workspace-relative directory |
| `external_provider_not_allowed` | External mode lacks the explicit hard gate | Set the gate and confirm data transfer intentionally |
| `retrieval_unavailable` | Both dense and sparse retrieval routes failed | Inspect the Qdrant/provider diagnostics; do not treat it as an empty result |

No-result output is not evidence that a scientific claim is false. Recheck the
generation, query, filters, and source coverage; provenance gaps remain gaps.

## Two-stage import driver

For the local corpus, use the thin WSL driver from the repository root (or
invoke `bash /absolute/path/to/PhomatAgent/scripts/import_literature_qdrant.sh`
from another working directory). The driver then changes to the repository
root and invokes the current project's `uv` environment; it does not contain
provider logic or call the Qdrant HTTP API.
The default workspace is `/home/shiqiany/AIagent`, with these workspace-relative
source paths:

| Stage | Default source |
| --- | --- |
| PDF full text | `Photoelectric detection/dataset/paper/pdf` |
| Abstracts | `Photoelectric detection/dataset/paper/abstract/abstracts.sqlite3` |

Run the stages separately and activate only after both are complete:

```bash
bash scripts/import_literature_qdrant.sh pdf --stop-after 100
bash scripts/import_literature_qdrant.sh pdf --resume --stop-after 100
bash scripts/import_literature_qdrant.sh abstracts --stop-after 5000
bash scripts/import_literature_qdrant.sh abstracts --resume --stop-after 5000
bash scripts/import_literature_qdrant.sh status
bash scripts/import_literature_qdrant.sh activate
```

The driver also accepts `--workspace PATH`, `--pdf-directory PATH`, and
`--database PATH` to override those defaults. All source paths are passed as
individual arguments, so spaces in a path are preserved. `--stop-after N`
requests a successful, bounded pause after at most `N` source records;
`--resume` reuses the saved ID for that stage. `--yes` is forwarded only when
explicitly supplied. `--dry-run` prints the shell-escaped `uv run` argv and
resolved paths, without writing run state or starting `uv`, a provider, or
Qdrant.

Before starting a fresh stage, the driver archives any previous ID under
`user_output/rag-import/run-state/archive/`, then atomically saves its new run
ID under `pdf.run_id` or `abstracts.run_id`. A failed process or WSL restart
therefore leaves an exact ID for the next `--resume`; a missing or empty state
file is an explicit error for `--resume` and `activate` (status reports an
unsupplied stage). The CLI's ingestion record remains the source of truth for
its cursor and progress. Each progress line is bounded JSON
with `total`, `processed`, `indexed`, `unchanged`, `failed`, `skipped`,
`passages`, `rate`, `eta`/`eta_seconds`, `cursor`, `run_id`, and `status` (plus
the per-invocation `processed_this_invocation` audit field).

Abstract runs record a streamed SHA-256 identity for the SQLite file and the
workspace-relative source path. They require one immutable, checkpointed
snapshot: a non-empty `abstracts.sqlite3-wal` fails closed with
`source_wal_pending` rather than silently omitting committed WAL content.
Stop writers and run SQLite's `wal_checkpoint(TRUNCATE)`, then verify the
`-wal` file is absent or zero length. For a valuable/live source, make a
consistent SQLite backup/copy after checkpointing and import that
workspace-contained copy; do not copy a WAL sidecar independently. A main
database replacement or newly appearing WAL is reported as `source_changed`
and requires a new run from the verified snapshot.

The service computes the source SHA-256 and row counts once per invocation.
Each later bounded batch performs only a cheap main/WAL metadata check while
the reader holds one read transaction. A resume invocation recomputes the
identity once and validates it against the saved run; an unknown `--resume`
ID is rejected rather than creating an unrelated run.

If the database changed while an old abstract run was incomplete, do not
resume that run. Start a fresh run and explicitly link the old run:

```bash
uv run photomatagent rag index-abstracts \
  --database dataset/paper/abstract/abstracts.sqlite3 \
  --run-id NEW_RUN_ID --supersede-run-id OLD_RUN_ID --yes
```

The new run is persisted before fetching, embedding, or writing documents.
Only after it completes successfully is `OLD_RUN_ID` marked
`superseded`/complete; failures leave the old run auditable and activation
blocked. Replacement cleanup is filtered to this workspace, abstract source
kind, and exact SQLite path, and removes only non-ready artifacts. READY
abstract knowledge absent from the changed database is retained until an
explicit, separately reviewed cleanup. Revision replacement makes the new
READY revision visible before removing older revisions; cleanup failures are
retryable and do not delete the new revision.

External embedding or reranking remains behind the existing configuration and
confirmation gate. When an external provider is configured, each indexing
stage prints the provider/model and the data scope, then asks for confirmation;
pass `--yes` only after confirming that full-text or abstract text may leave the
workspace. The driver never stores or prints API keys.

`status` loads both saved stage IDs and performs the CLI's bounded read-only
lookups. `activate` requires both IDs, passes both explicit
`--require-stage pdf --require-stage abstracts` checks and their corresponding
`--pdf-run-id`/`--abstract-run-id` values to `photomatagent rag activate`, and
uses the existing atomic alias switch. It does not perform a direct Qdrant
request. An incomplete or retryable stage is rejected before activation.

The WSL driver archives the prior stage ID under `run-state/archive/` before
replacing its current state file. A fresh `abstracts` invocation automatically
forwards that prior ID as `--supersede-run-id`; pass an explicit ID to
override it. Use `--resume` only for the saved run itself, and never combine
it with supersession.

Retrieval follows the evidence priority local full text → local abstracts →
arXiv metadata. Abstract results identify themselves as abstract-only and do
not imply that a PDF was inspected. arXiv remains a separate, permission-
controlled, session-only search: its results are not written to Qdrant, the
SQLite database, or the workspace.

## Legacy LanceDB artifacts

The Qdrant migration does not import, rewrite, or delete an existing
`output/literature_index` directory. Keep a separate backup of any source PDFs
and user-generated artifacts before cleanup. After the Qdrant generation has
been validated and the backup policy is satisfied, an operator may archive or
remove the legacy directory manually. Never use it as an implicit fallback and
never infer that a Qdrant snapshot replaces the original PDF backup.

## Capacity benchmark and claims

The benchmark is opt-in and dry-run by default:

```bash
uv run python scripts/benchmark_qdrant_rag.py \
  --test-prefix photomat_test_capacity
uv run python scripts/benchmark_qdrant_rag.py \
  --test-prefix photomat_test_capacity_1m \
  --points 1000000 --confirm-write
uv run python scripts/benchmark_qdrant_rag.py \
  --test-prefix photomat_test_capacity_local \
  --points 100000 --confirm-write \
  --measure-local-retrieval --warmup-local-model
```

The live command uses only its own UUID-suffixed collection, never the current
aliases, and records hardware, server/collection configuration, cold/warm
candidate p50/p95 latency for real Qdrant hybrid/RRF retrieval, and process RSS
under `user_output/qdrant-benchmark/`. `--measure-local-retrieval` separately
records end-to-end query embedding plus local reranking; model loading is
included unless `--warmup-local-model` is requested. The report labels the
first request as `cold` or `post_warmup` explicitly and calls later samples
`subsequent`; an unavailable model is reported as unavailable rather than
replaced with a fake timing. Delete the
isolated collection after inspection unless `--keep-collection` is explicitly
requested. The benchmark does not run automatically in pytest. Performance for
10,000 papers is unverified until the explicit one-million-point benchmark has
actually run; do not report a 10,000-paper performance claim from the fixture
evaluation or from a dry run.
