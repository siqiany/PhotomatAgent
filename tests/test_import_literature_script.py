"""Minimal contracts for the resumable WSL literature import driver."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path


SCRIPT = Path("scripts/import_literature_qdrant.sh")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / SCRIPT


def _fake_uv(tmp_path: Path, monkeypatch) -> dict[str, Path]:
    """Install a no-op uv shim that records exact argv and invocation order."""
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    paths = {
        "args": tmp_path / "uv-args.bin",
        "count": tmp_path / "uv-count.txt",
        "invoked": tmp_path / "uv-invoked",
        "error": tmp_path / "uv-error.txt",
    }
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${FAKE_STATE_FILE:-}" && ! -f "$FAKE_STATE_FILE" ]]; then
  printf '%s\\n' 'state was not persisted before uv started' > "$FAKE_ERROR_FILE"
  exit 91
fi
printf '%s\\0' "$@" > "$FAKE_ARGS_FILE"
count=0
if [[ -f "$FAKE_COUNT_FILE" ]]; then
  count="$(< "$FAKE_COUNT_FILE")"
fi
printf '%s' "$((count + 1))" > "$FAKE_COUNT_FILE"
: > "$FAKE_INVOKED_FILE"
if [[ -n "${FAKE_EXIT_CODE:-}" ]]; then
  exit "$FAKE_EXIT_CODE"
fi
""",
        encoding="utf-8",
    )
    fake_uv.chmod(fake_uv.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join(
            [str(bin_dir), os.environ.get("PATH", "")]
        ),
    )
    for name, path in paths.items():
        monkeypatch.setenv(f"FAKE_{name.upper()}_FILE", str(path))
    return paths


def _run_driver(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT_PATH), *args],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _recorded_argv(path: Path) -> list[str]:
    return [item.decode("utf-8") for item in path.read_bytes().split(b"\0") if item]


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


def test_pdf_argv_keeps_spaced_paths_and_saves_id_before_uv(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fake_uv(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace with spaces"
    workspace.mkdir()
    state_dir = tmp_path / "state with spaces"
    state_file = state_dir / "pdf.run_id"
    monkeypatch.setenv("FAKE_STATE_FILE", str(state_file))

    result = _run_driver(
        "pdf",
        "--workspace",
        str(workspace),
        "--pdf-directory",
        "Photoelectric detection/paper dir",
        "--run-state-dir",
        str(state_dir),
        "--stop-after",
        "20",
    )

    assert result.returncode == 0, result.stderr
    assert state_file.is_file()
    assert paths["invoked"].is_file()
    assert paths["error"].exists() is False
    run_id = state_file.read_text(encoding="utf-8").strip()
    argv = _recorded_argv(paths["args"])
    assert argv == [
        "run",
        "photomatagent",
        "rag",
        "index",
        "--workspace",
        str(workspace),
        "--directory",
        "Photoelectric detection/paper dir",
        "--run-id",
        run_id,
        "--stop-after",
        "20",
    ]


def test_resume_reuses_exact_id_and_missing_state_fails_closed(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fake_uv(tmp_path, monkeypatch)
    state_dir = tmp_path / "run state"
    state_dir.mkdir()
    state_file = state_dir / "pdf.run_id"
    state_file.write_text("saved-pdf-id\n", encoding="utf-8")
    monkeypatch.setenv("FAKE_STATE_FILE", str(state_file))

    resumed = _run_driver(
        "pdf",
        "--workspace",
        str(tmp_path),
        "--run-state-dir",
        str(state_dir),
        "--resume",
    )
    assert resumed.returncode == 0, resumed.stderr
    assert "saved-pdf-id" in _recorded_argv(paths["args"])
    assert "--resume" in _recorded_argv(paths["args"])

    state_file.unlink()
    missing = _run_driver(
        "pdf",
        "--workspace",
        str(tmp_path),
        "--run-state-dir",
        str(state_dir),
        "--resume",
    )
    assert missing.returncode == 2
    assert "no saved pdf run ID" in missing.stderr
    assert paths["count"].read_text(encoding="utf-8") == "1"


def test_fresh_abstract_run_archives_previous_id_and_forwards_supersession(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fake_uv(tmp_path, monkeypatch)
    state_dir = tmp_path / "run-state"
    state_dir.mkdir()
    state_file = state_dir / "abstracts.run_id"
    state_file.write_text("old-abstract-run\n", encoding="utf-8")
    (state_dir / "abstracts.source_path").write_text(
        "database with spaces.sqlite3\n", encoding="utf-8"
    )
    monkeypatch.setenv("FAKE_STATE_FILE", str(state_file))

    result = _run_driver(
        "abstracts",
        "--workspace",
        str(tmp_path),
        "--database",
        "database with spaces.sqlite3",
        "--run-state-dir",
        str(state_dir),
    )

    assert result.returncode == 0, result.stderr
    new_run_id = state_file.read_text(encoding="utf-8").strip()
    assert new_run_id and new_run_id != "old-abstract-run"
    argv = _recorded_argv(paths["args"])
    assert "--supersede-run-id" in argv
    assert argv[argv.index("--supersede-run-id") + 1] == "old-abstract-run"
    archived = list((state_dir / "archive").glob("abstracts.*"))
    assert len(archived) == 1
    assert archived[0].read_text(encoding="utf-8").strip() == "old-abstract-run"


def test_fresh_abstract_changed_source_requires_explicit_supersession(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fake_uv(tmp_path, monkeypatch)
    state_dir = tmp_path / "run-state"
    state_dir.mkdir()
    state_file = state_dir / "abstracts.run_id"
    state_file.write_text("old-abstract-run\n", encoding="utf-8")
    (state_dir / "abstracts.source_path").write_text(
        "old-database.sqlite3\n", encoding="utf-8"
    )

    result = _run_driver(
        "abstracts",
        "--workspace",
        str(tmp_path),
        "--database",
        "new-database.sqlite3",
        "--run-state-dir",
        str(state_dir),
    )

    assert result.returncode == 2
    assert "--supersede-run-id" in result.stderr
    assert state_file.read_text(encoding="utf-8") == "old-abstract-run\n"
    assert not paths["invoked"].exists()
    assert not (state_dir / "archive").exists()


def test_changed_source_supersession_failure_keeps_active_state_and_audit(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fake_uv(tmp_path, monkeypatch)
    state_dir = tmp_path / "run-state"
    state_dir.mkdir()
    state_file = state_dir / "abstracts.run_id"
    state_file.write_text("old-abstract-run\n", encoding="utf-8")
    source_file = state_dir / "abstracts.source_path"
    source_file.write_text("old-database.sqlite3\n", encoding="utf-8")
    monkeypatch.setenv("FAKE_EXIT_CODE", "17")

    result = _run_driver(
        "abstracts",
        "--workspace",
        str(tmp_path),
        "--database",
        "new-database.sqlite3",
        "--run-state-dir",
        str(state_dir),
        "--supersede-run-id",
        "old-abstract-run",
    )

    assert result.returncode == 17
    assert paths["invoked"].is_file()
    assert state_file.read_text(encoding="utf-8") == "old-abstract-run\n"
    assert source_file.read_text(encoding="utf-8") == "old-database.sqlite3\n"
    assert not (state_dir / "archive").exists()


def test_changed_source_supersession_promotes_state_after_cli_acceptance(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fake_uv(tmp_path, monkeypatch)
    state_dir = tmp_path / "run-state"
    state_dir.mkdir()
    state_file = state_dir / "abstracts.run_id"
    state_file.write_text("old-abstract-run\n", encoding="utf-8")
    source_file = state_dir / "abstracts.source_path"
    source_file.write_text("old-database.sqlite3\n", encoding="utf-8")

    result = _run_driver(
        "abstracts",
        "--workspace",
        str(tmp_path),
        "--database",
        "new-database.sqlite3",
        "--run-state-dir",
        str(state_dir),
        "--supersede-run-id",
        "old-abstract-run",
    )

    assert result.returncode == 0, result.stderr
    new_run_id = state_file.read_text(encoding="utf-8").strip()
    assert new_run_id and new_run_id != "old-abstract-run"
    assert source_file.read_text(encoding="utf-8").strip() == "new-database.sqlite3"
    argv = _recorded_argv(paths["args"])
    assert argv[argv.index("--supersede-run-id") + 1] == "old-abstract-run"
    archived = list((state_dir / "archive").glob("abstracts.*"))
    assert len(archived) == 1
    assert archived[0].read_text(encoding="utf-8").strip() == "old-abstract-run"


def test_pending_abstract_run_can_resume_and_promote_after_interruption(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fake_uv(tmp_path, monkeypatch)
    state_dir = tmp_path / "run-state"
    state_dir.mkdir()
    state_file = state_dir / "abstracts.run_id"
    state_file.write_text("old-abstract-run\n", encoding="utf-8")
    source_file = state_dir / "abstracts.source_path"
    source_file.write_text("abstracts.sqlite3\n", encoding="utf-8")
    monkeypatch.setenv("FAKE_STATE_FILE", str(state_file))
    monkeypatch.setenv("FAKE_EXIT_CODE", "17")

    interrupted = _run_driver(
        "abstracts",
        "--workspace",
        str(tmp_path),
        "--database",
        "abstracts.sqlite3",
        "--run-state-dir",
        str(state_dir),
    )

    assert interrupted.returncode == 17
    pending_file = state_dir / "abstracts.pending.run_id"
    pending_id = pending_file.read_text(encoding="utf-8").strip()
    assert pending_id
    assert state_file.read_text(encoding="utf-8") == "old-abstract-run\n"

    monkeypatch.delenv("FAKE_EXIT_CODE")
    resumed = _run_driver(
        "abstracts",
        "--workspace",
        str(tmp_path),
        "--database",
        "abstracts.sqlite3",
        "--run-state-dir",
        str(state_dir),
        "--resume",
    )

    assert resumed.returncode == 0, resumed.stderr
    assert state_file.read_text(encoding="utf-8").strip() == pending_id
    assert source_file.read_text(encoding="utf-8").strip() == "abstracts.sqlite3"
    assert not pending_file.exists()
    argv = _recorded_argv(paths["args"])
    assert "--resume" in argv


def test_discard_pending_recovers_failed_supersession_without_touching_active(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fake_uv(tmp_path, monkeypatch)
    state_dir = tmp_path / "run-state"
    state_dir.mkdir()
    state_file = state_dir / "abstracts.run_id"
    state_file.write_text("old-abstract-run\n", encoding="utf-8")
    source_file = state_dir / "abstracts.source_path"
    source_file.write_text("abstracts.sqlite3\n", encoding="utf-8")
    archive_dir = state_dir / "archive"
    archive_dir.mkdir()
    archive_audit = archive_dir / "abstracts.previous.audit"
    archive_audit.write_text("keep\n", encoding="utf-8")
    monkeypatch.setenv("FAKE_STATE_FILE", str(state_file))
    monkeypatch.setenv("FAKE_EXIT_CODE", "17")

    failed = _run_driver(
        "abstracts",
        "--workspace",
        str(tmp_path),
        "--database",
        "abstracts.sqlite3",
        "--run-state-dir",
        str(state_dir),
        "--supersede-run-id",
        "unknown-old-run",
    )
    assert failed.returncode == 17
    pending_file = state_dir / "abstracts.pending.run_id"
    pending_source_file = state_dir / "abstracts.pending.source_path"
    pending_id = pending_file.read_text(encoding="utf-8")

    resume_failed = _run_driver(
        "abstracts",
        "--workspace",
        str(tmp_path),
        "--database",
        "abstracts.sqlite3",
        "--run-state-dir",
        str(state_dir),
        "--resume",
    )
    assert resume_failed.returncode == 17
    assert pending_file.read_text(encoding="utf-8") == pending_id
    assert pending_source_file.is_file()
    assert state_file.read_text(encoding="utf-8") == "old-abstract-run\n"
    assert source_file.read_text(encoding="utf-8") == "abstracts.sqlite3\n"

    discarded = _run_driver(
        "abstracts",
        "--workspace",
        str(tmp_path),
        "--run-state-dir",
        str(state_dir),
        "--discard-pending",
    )
    assert discarded.returncode == 0, discarded.stderr
    assert not pending_file.exists()
    assert not pending_source_file.exists()
    assert state_file.read_text(encoding="utf-8") == "old-abstract-run\n"
    assert source_file.read_text(encoding="utf-8") == "abstracts.sqlite3\n"
    assert archive_audit.read_text(encoding="utf-8") == "keep\n"

    monkeypatch.delenv("FAKE_EXIT_CODE")
    fresh = _run_driver(
        "abstracts",
        "--workspace",
        str(tmp_path),
        "--database",
        "abstracts.sqlite3",
        "--run-state-dir",
        str(state_dir),
    )
    assert fresh.returncode == 0, fresh.stderr
    assert state_file.read_text(encoding="utf-8").strip() != "old-abstract-run"


def test_pending_promotion_recovers_active_id_and_is_idempotent(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fake_uv(tmp_path, monkeypatch)
    state_dir = tmp_path / "run-state"
    state_dir.mkdir()
    state_file = state_dir / "abstracts.run_id"
    state_file.write_text("current-active-run\n", encoding="utf-8")
    source_file = state_dir / "abstracts.source_path"
    source_file.write_text("abstracts.sqlite3\n", encoding="utf-8")
    pending_file = state_dir / "abstracts.pending.run_id"
    pending_file.write_text("pending-run\n", encoding="utf-8")
    pending_source_file = state_dir / "abstracts.pending.source_path"
    pending_source_file.write_text("abstracts.sqlite3\n", encoding="utf-8")
    monkeypatch.setenv("FAKE_STATE_FILE", str(state_file))

    first_resume = _run_driver(
        "abstracts",
        "--workspace",
        str(tmp_path),
        "--database",
        "abstracts.sqlite3",
        "--run-state-dir",
        str(state_dir),
        "--resume",
    )

    assert first_resume.returncode == 0, first_resume.stderr
    assert state_file.read_text(encoding="utf-8") == "pending-run\n"
    archived = list((state_dir / "archive").glob("abstracts.*"))
    assert len(archived) == 1
    assert archived[0].read_text(encoding="utf-8").strip() == "current-active-run"
    assert not pending_file.exists()
    assert not pending_source_file.exists()

    # A retry after active promotion but before pending cleanup must not
    # archive the same active run a second time.
    pending_file.write_text("pending-run\n", encoding="utf-8")
    pending_source_file.write_text("abstracts.sqlite3\n", encoding="utf-8")
    second_resume = _run_driver(
        "abstracts",
        "--workspace",
        str(tmp_path),
        "--database",
        "abstracts.sqlite3",
        "--run-state-dir",
        str(state_dir),
        "--resume",
    )

    assert second_resume.returncode == 0, second_resume.stderr
    assert state_file.read_text(encoding="utf-8") == "pending-run\n"
    assert len(list((state_dir / "archive").glob("abstracts.*"))) == 1
    assert not pending_file.exists()
    assert not pending_source_file.exists()


def test_supersession_cannot_be_combined_with_resume(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fake_uv(tmp_path, monkeypatch)
    state_dir = tmp_path / "run-state"
    state_dir.mkdir()
    state_file = state_dir / "abstracts.run_id"
    state_file.write_text("saved-abstract-run\n", encoding="utf-8")

    result = _run_driver(
        "abstracts",
        "--workspace",
        str(tmp_path),
        "--run-state-dir",
        str(state_dir),
        "--resume",
        "--supersede-run-id",
        "old-abstract-run",
    )

    assert result.returncode == 2
    assert "cannot be combined" in result.stderr
    assert not paths["invoked"].exists()


def test_status_and_activate_forward_both_ids_and_activation_requires_both(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fake_uv(tmp_path, monkeypatch)
    state_dir = tmp_path / "run-state"
    state_dir.mkdir()
    (state_dir / "pdf.run_id").write_text("pdf-id\n", encoding="utf-8")
    (state_dir / "abstracts.run_id").write_text("abstract-id\n", encoding="utf-8")

    status = _run_driver(
        "status",
        "--workspace",
        str(tmp_path),
        "--run-state-dir",
        str(state_dir),
    )
    assert status.returncode == 0, status.stderr
    status_argv = _recorded_argv(paths["args"])
    assert status_argv == [
        "run",
        "photomatagent",
        "rag",
        "status",
        "--workspace",
        str(tmp_path),
        "--pdf-run-id",
        "pdf-id",
        "--abstract-run-id",
        "abstract-id",
    ]

    activate = _run_driver(
        "activate",
        "--workspace",
        str(tmp_path),
        "--run-state-dir",
        str(state_dir),
        "--yes",
    )
    assert activate.returncode == 0, activate.stderr
    activate_argv = _recorded_argv(paths["args"])
    assert activate_argv == [
        "run",
        "photomatagent",
        "rag",
        "activate",
        "--workspace",
        str(tmp_path),
        "--require-stage",
        "pdf",
        "--require-stage",
        "abstracts",
        "--pdf-run-id",
        "pdf-id",
        "--abstract-run-id",
        "abstract-id",
        "--yes",
    ]

    (state_dir / "abstracts.run_id").unlink()
    failed = _run_driver(
        "activate",
        "--workspace",
        str(tmp_path),
        "--run-state-dir",
        str(state_dir),
    )
    assert failed.returncode == 2
    assert "requires a saved abstracts run ID" in failed.stderr
    assert paths["count"].read_text(encoding="utf-8") == "2"


def test_yes_is_forwarded_only_when_explicitly_requested(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fake_uv(tmp_path, monkeypatch)
    state_dir = tmp_path / "run-state"
    monkeypatch.setenv("FAKE_STATE_FILE", str(state_dir / "pdf.run_id"))

    without_yes = _run_driver(
        "pdf",
        "--workspace",
        str(tmp_path),
        "--run-state-dir",
        str(state_dir),
    )
    assert without_yes.returncode == 0, without_yes.stderr
    assert "--yes" not in _recorded_argv(paths["args"])

    with_yes = _run_driver(
        "pdf",
        "--workspace",
        str(tmp_path),
        "--run-state-dir",
        str(state_dir),
        "--yes",
    )
    assert with_yes.returncode == 0, with_yes.stderr
    assert "--yes" in _recorded_argv(paths["args"])


def test_dry_run_does_not_write_state_or_invoke_uv(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fake_uv(tmp_path, monkeypatch)
    state_dir = tmp_path / "dry-run-state"
    result = _run_driver(
        "pdf",
        "--workspace",
        str(tmp_path),
        "--run-state-dir",
        str(state_dir),
        "--dry-run",
        "--stop-after",
        "20",
    )

    assert result.returncode == 0, result.stderr
    assert not state_dir.exists()
    assert not paths["invoked"].exists()
    assert not paths["count"].exists()
