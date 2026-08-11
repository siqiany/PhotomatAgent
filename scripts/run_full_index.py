#!/usr/bin/env python3
"""Full-dataset Literature RAG indexer (resumable, incremental).

Warms up the docling layout model and both sentence-transformers models
(cached under ~/.cache/huggingface after first use), then indexes every PDF
under the literature root. Safe to interrupt: unchanged PDFs are skipped by
content hash on the next run, so re-running resumes where it stopped.

Usage:
    PHOTOMATAGENT_LITERATURE_DIR="/path/to/dataset/paper" \\
    PHOTOMATAGENT_LITERATURE_INDEX_DIR="output/literature_index" \\
    uv run python scripts/run_full_index.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--literature-root",
        default=os.environ.get("PHOTOMATAGENT_LITERATURE_DIR", ""),
        help="PDF directory (recursive); env PHOTOMATAGENT_LITERATURE_DIR",
    )
    parser.add_argument(
        "--index-dir",
        default=os.environ.get(
            "PHOTOMATAGENT_LITERATURE_INDEX_DIR", "output/literature_index"
        ),
        help="LanceDB index directory",
    )
    parser.add_argument(
        "--log-file", default="output/full_index.log", help="Log file path"
    )
    parser.add_argument(
        "--status-file",
        default="output/full_index_status.json",
        help="JSON progress/status file",
    )
    args = parser.parse_args()

    root = Path(args.literature_root)
    if not root.is_dir():
        print(f"ERROR: literature root not found: {root}", file=sys.stderr)
        return 2

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(str(log_path)),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("full_index")

    pdfs = sorted(path for path in root.rglob("*.pdf") if path.is_file())
    logger.info("literature root: %s (%d PDFs)", root, len(pdfs))
    logger.info("index dir: %s", args.index_dir)

    # Warm up models so the first paper does not pay the download cost.
    from photomatagent.scientific.capabilities.literature.index import _embedder
    from photomatagent.scientific.capabilities.literature.retrieval import _reranker

    logger.info("warming up embedding model %s ...", EMBEDDING_MODEL)
    _embedder(EMBEDDING_MODEL)
    logger.info("warming up reranker %s ...", RERANKER_MODEL)
    _reranker(RERANKER_MODEL)
    logger.info("models ready")

    from photomatagent.scientific.capabilities.literature.index import (
        LiteratureIndex,
    )

    index = LiteratureIndex(Path(args.index_dir), embedding_model=EMBEDDING_MODEL)
    started = time.time()

    def write_status(phase: str, stats: dict[str, Any] | None) -> None:
        payload = {
            "phase": phase,
            "updated_at": _now_iso(),
            "elapsed_seconds": round(time.time() - started, 1),
            "stats": stats or {},
            "index_dir": str(index.index_dir),
            "literature_root": str(root),
        }
        Path(args.status_file).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def on_progress(stats: dict[str, Any]) -> None:
        done = stats["indexed"] + stats["skipped"] + stats["failed"]
        logger.info(
            "progress %d/%d | indexed=%d skipped=%d failed=%d chunks=%d",
            done,
            len(pdfs),
            stats["indexed"],
            stats["skipped"],
            stats["failed"],
            stats["chunks"],
        )
        write_status("indexing", stats)

    write_status("starting", None)
    logger.info("beginning full indexing (interrupt-safe, resumable)")
    try:
        with index.locked():
            stats = index.index_directory(root, on_progress=on_progress)
    except KeyboardInterrupt:
        logger.warning("interrupted; resume by re-running this script")
        write_status("interrupted", None)
        return 130

    stats["pdfs_found"] = len(pdfs)
    stats["elapsed_seconds"] = round(time.time() - started, 1)
    logger.info("DONE: %s", json.dumps(stats, ensure_ascii=False)[:600])
    write_status("done", stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
