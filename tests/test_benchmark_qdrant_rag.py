"""Safety and deterministic dry-run checks for the Qdrant capacity benchmark."""

from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import pytest


def _benchmark_module():
    path = Path(__file__).parents[1] / "scripts" / "benchmark_qdrant_rag.py"
    spec = importlib.util.spec_from_file_location("benchmark_qdrant_rag_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_benchmark_defaults_are_dry_run_and_use_bounded_safe_prefix() -> None:
    benchmark_qdrant_rag = _benchmark_module()
    BenchmarkConfig = benchmark_qdrant_rag.BenchmarkConfig
    parser = benchmark_qdrant_rag.parser

    args = parser().parse_args(["--test-prefix", "photomat_test_fixture"])
    config = BenchmarkConfig.from_args(args)
    assert config.points == 100_000
    assert config.dimension == 384
    assert config.queries == 100
    assert config.batch_size == 256
    assert config.dry_run is True
    assert config.collection_name.startswith("photomat_test_fixture_")
    assert "current" not in config.collection_name


def test_benchmark_rejects_prefix_that_could_touch_current_collections() -> None:
    BenchmarkConfig = _benchmark_module().BenchmarkConfig

    with pytest.raises(ValueError, match="photomat_test_"):
        BenchmarkConfig(
            points=10,
            dimension=8,
            queries=1,
            batch_size=1,
            url="http://127.0.0.1:6333",
            test_prefix="photomat_literature",
            confirm_write=False,
        )


def test_benchmark_dry_run_writes_report_without_contacting_qdrant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark_qdrant_rag = _benchmark_module()

    output = tmp_path / "benchmark.json"
    monkeypatch.setattr(
        benchmark_qdrant_rag,
        "default_output_path",
        lambda: output,
    )
    exit_code = benchmark_qdrant_rag.main(
        ["--test-prefix", "photomat_test_fixture", "--points", "32"]
    )
    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["dry_run"] is True
    assert report["points"] == 32
    assert report["writes_performed"] is False
    assert report["collection_name"].startswith("photomat_test_fixture_")
