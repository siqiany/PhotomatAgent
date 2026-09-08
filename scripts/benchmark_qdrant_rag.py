#!/usr/bin/env python3
"""Measure bounded Qdrant candidate latency for an isolated benchmark collection.

The command is deliberately a dry run unless ``--confirm-write`` is supplied.
It never creates or updates the application's ``*_current`` aliases and only
uses a collection whose name starts with the required ``photomat_test_``
safety marker.  A one-million-point run is therefore explicit and opt-in; it
is never run by pytest or by the normal CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:  # ``resource`` is unavailable on native Windows.
    import resource
except ImportError:  # pragma: no cover - exercised on native Windows
    resource = None  # type: ignore[assignment]


DEFAULT_URL = "http://127.0.0.1:6333"
DEFAULT_SEED = 20260908
MAX_POINTS = 1_000_000


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Benchmark isolated Qdrant RAG candidate retrieval."
    )
    argument_parser.add_argument(
        "--points", type=int, default=100_000, help="Number of points (default: 100000)."
    )
    argument_parser.add_argument(
        "--dimension", type=int, default=384, help="Dense vector dimension (default: 384)."
    )
    argument_parser.add_argument(
        "--queries", type=int, default=100, help="Number of query vectors (default: 100)."
    )
    argument_parser.add_argument(
        "--batch-size", type=int, default=256, help="Upsert batch size (default: 256)."
    )
    argument_parser.add_argument(
        "--url", default=DEFAULT_URL, help=f"Qdrant URL (default: {DEFAULT_URL})."
    )
    argument_parser.add_argument(
        "--test-prefix",
        "--prefix",
        dest="test_prefix",
        required=True,
        help="Required isolated prefix beginning with photomat_test_.",
    )
    argument_parser.add_argument(
        "--confirm-write",
        action="store_true",
        help="Explicitly create/upsert/query the isolated benchmark collection.",
    )
    argument_parser.add_argument(
        "--keep-collection",
        action="store_true",
        help="Keep the isolated collection after a confirmed run.",
    )
    argument_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path (default: user_output/qdrant-benchmark/...).",
    )
    argument_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return argument_parser


@dataclass(frozen=True)
class BenchmarkConfig:
    points: int
    dimension: int
    queries: int
    batch_size: int
    url: str
    test_prefix: str
    confirm_write: bool = False
    keep_collection: bool = False
    output: Path | None = None
    seed: int = DEFAULT_SEED
    collection_name: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.test_prefix.startswith("photomat_test_"):
            raise ValueError("test_prefix must start with photomat_test_")
        if any(char.isspace() for char in self.test_prefix):
            raise ValueError("test_prefix must not contain whitespace")
        if not 1 <= self.points <= MAX_POINTS:
            raise ValueError(f"points must be between 1 and {MAX_POINTS}")
        if not 1 <= self.dimension <= 8192:
            raise ValueError("dimension must be between 1 and 8192")
        if not 1 <= self.queries <= 10_000:
            raise ValueError("queries must be between 1 and 10000")
        if not 1 <= self.batch_size <= 10_000:
            raise ValueError("batch_size must be between 1 and 10000")
        if not self.url.strip():
            raise ValueError("url must not be empty")
        run_id = uuid.uuid4().hex
        object.__setattr__(
            self,
            "collection_name",
            f"{self.test_prefix}_benchmark_{run_id}",
        )

    @property
    def dry_run(self) -> bool:
        return not self.confirm_write

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "BenchmarkConfig":
        return cls(
            points=int(args.points),
            dimension=int(args.dimension),
            queries=int(args.queries),
            batch_size=int(args.batch_size),
            url=str(args.url),
            test_prefix=str(args.test_prefix),
            confirm_write=bool(args.confirm_write),
            keep_collection=bool(args.keep_collection),
            output=args.output,
            seed=int(args.seed),
        )


def default_output_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("user_output/qdrant-benchmark") / f"benchmark-{timestamp}-{uuid.uuid4().hex[:8]}.json"


def _rss_bytes() -> int:
    if resource is None:
        return 0
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB while macOS reports bytes.
    return value * 1024 if sys.platform.startswith("linux") else value


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_safe(model_dump(mode="json"))
    return str(value)


def _dry_run_report(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "label": "qdrant-capacity-benchmark",
        "dry_run": True,
        "writes_performed": False,
        "points": config.points,
        "dimension": config.dimension,
        "queries": config.queries,
        "batch_size": config.batch_size,
        "url": config.url,
        "collection_name": config.collection_name,
        "seed": config.seed,
        "server_config": None,
        "hardware": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "rss_bytes": _rss_bytes(),
        },
        "candidate_latency_ms": {
            "cold_p50": None,
            "cold_p95": None,
            "warm_p50": None,
            "warm_p95": None,
        },
        "note": "Dry run; pass --confirm-write for an isolated live collection benchmark.",
    }


def _live_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    try:
        import numpy as np
        from qdrant_client import QdrantClient, models
    except ImportError as exc:  # pragma: no cover - optional benchmark extra
        raise RuntimeError("numpy and qdrant-client are required for a live benchmark") from exc

    api_key_name = os.environ.get("PHOTOMATAGENT_QDRANT_API_KEY_ENV", "QDRANT_API_KEY")
    api_key = os.environ.get(api_key_name, "").strip() or None
    client = QdrantClient(
        url=config.url,
        api_key=api_key,
        timeout=60,
        prefer_grpc=False,
    )
    created = False
    try:
        client.create_collection(
            collection_name=config.collection_name,
            vectors_config=models.VectorParams(
                size=config.dimension,
                distance=models.Distance.COSINE,
                on_disk=True,
            ),
            shard_number=1,
            replication_factor=1,
            on_disk_payload=True,
        )
        created = True
        vector_rng = np.random.default_rng(config.seed)
        for start in range(0, config.points, config.batch_size):
            count = min(config.batch_size, config.points - start)
            vectors = vector_rng.random((count, config.dimension), dtype=np.float32)
            points = [
                models.PointStruct(id=start + offset, vector=vector.tolist())
                for offset, vector in enumerate(vectors)
            ]
            client.upsert(
                collection_name=config.collection_name,
                points=points,
                wait=True,
            )

        query_rng = np.random.default_rng(config.seed + 1)
        queries = query_rng.random((config.queries, config.dimension), dtype=np.float32)

        def query_once(vector: Any) -> float:
            started = time.perf_counter()
            client.query_points(
                collection_name=config.collection_name,
                query=vector.tolist(),
                limit=10,
                with_payload=False,
                with_vectors=False,
            )
            return (time.perf_counter() - started) * 1000.0

        cold_ms = [query_once(queries[0])]
        # Warm timings deliberately exclude the first request and use the
        # configured query count, making cold/hot behavior explicit.
        warm_ms = [query_once(vector) for vector in queries[1:]]
        info = client.info()
        collection = client.get_collection(config.collection_name)
        return {
            "label": "qdrant-capacity-benchmark",
            "dry_run": False,
            "writes_performed": True,
            "points": config.points,
            "dimension": config.dimension,
            "queries": config.queries,
            "batch_size": config.batch_size,
            "url": config.url,
            "collection_name": config.collection_name,
            "seed": config.seed,
            "server_config": {
                "server_version": str(getattr(info, "version", "") or ""),
                "collection": _json_safe(collection),
            },
            "hardware": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "rss_bytes": _rss_bytes(),
            },
            "candidate_latency_ms": {
                "cold_p50": _percentile(cold_ms, 0.50),
                "cold_p95": _percentile(cold_ms, 0.95),
                "warm_p50": _percentile(warm_ms, 0.50),
                "warm_p95": _percentile(warm_ms, 0.95),
            },
        }
    finally:
        if created and not config.keep_collection:
            client.delete_collection(config.collection_name)
        client.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config = BenchmarkConfig.from_args(args)
        report = _dry_run_report(config) if config.dry_run else _live_benchmark(config)
        output_path = config.output or default_output_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"benchmark_error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - command-line entry point
    raise SystemExit(main())
