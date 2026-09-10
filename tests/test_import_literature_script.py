"""Minimal contracts for the resumable WSL literature import driver."""

from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPT = Path("scripts/import_literature_qdrant.sh")


def test_script_exposes_required_stages() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    assert "pdf|abstracts|status|activate" in script
    assert "Photoelectric detection/dataset/paper/pdf" in script
    assert "Photoelectric detection/dataset/paper/abstract/abstracts.sqlite3" in script


def test_dry_run_preserves_paths_with_spaces() -> None:
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "pdf",
            "--dry-run",
            "--stop-after",
            "20",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Photoelectric detection/dataset/paper/pdf" in result.stdout
    assert "--stop-after 20" in result.stdout
