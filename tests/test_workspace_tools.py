from __future__ import annotations

import shlex
import sys

import pytest

from photomatagent.tools.bash import BashTool
from photomatagent.tools.edit import EditTool
from photomatagent.tools.glob import GlobTool
from photomatagent.tools.grep import GrepTool
from photomatagent.tools.read import ReadTool
from photomatagent.tools.write import WriteTool
from photomatagent.workspace import Workspace


@pytest.mark.asyncio
async def test_read_rejects_parent_traversal(tmp_path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (tmp_path / "outside.txt").write_text("outside", encoding="utf-8")
    result = await ReadTool(Workspace(workspace_root)).execute({"path": "../outside.txt"})
    assert result.is_error
    assert "outside workspace" in result.output


@pytest.mark.asyncio
async def test_read_line_range_and_truncation(tmp_path):
    (tmp_path / "file.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    result = await ReadTool(Workspace(tmp_path), max_chars=12).execute(
        {"path": "file.txt", "start_line": 2, "end_line": 3}
    )
    assert "2: two" in result.output
    assert result.data["truncated"] is True


@pytest.mark.asyncio
async def test_glob_and_grep_are_bounded_to_workspace(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("needle\n", encoding="utf-8")
    (src / "b.txt").write_text("needle\n", encoding="utf-8")
    workspace = Workspace(tmp_path)
    glob_result = await GlobTool(workspace).execute({"pattern": "src/**/*.py"})
    grep_result = await GrepTool(workspace).execute(
        {"pattern": "needle", "path": "src", "glob": "*.py"}
    )
    assert glob_result.data["matches"] == ["src/a.py"]
    assert "src/a.py:1:needle" in grep_result.output
    assert "b.txt" not in grep_result.output


@pytest.mark.asyncio
async def test_glob_skips_symlinks_escaping_workspace(tmp_path):
    (tmp_path / "real.txt").write_text("hi", encoding="utf-8")
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    try:
        (tmp_path / "escape.txt").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks not supported")
    result = await GlobTool(Workspace(tmp_path)).execute({"pattern": "**/*"})
    assert not result.is_error
    assert result.data["matches"] == ["real.txt"]


@pytest.mark.asyncio
async def test_glob_includes_symlinks_inside_workspace(tmp_path):
    (tmp_path / "real.txt").write_text("hi", encoding="utf-8")
    try:
        (tmp_path / "link.txt").symlink_to(tmp_path / "real.txt")
    except OSError:
        pytest.skip("symlinks not supported")
    result = await GlobTool(Workspace(tmp_path)).execute({"pattern": "**/*"})
    assert not result.is_error
    assert "real.txt" in result.data["matches"]
    assert "link.txt" in result.data["matches"]


@pytest.mark.asyncio
async def test_write_refuses_overwrite_and_edit_requires_unique_match(tmp_path):
    workspace = Workspace(tmp_path)
    writer = WriteTool(workspace)
    assert not (await writer.execute({"path": "a.txt", "content": "old"})).is_error
    assert (await writer.execute({"path": "a.txt", "content": "new"})).is_error
    editor = EditTool(workspace)
    assert not (
        await editor.execute({"path": "a.txt", "old_text": "old", "new_text": "new"})
    ).is_error
    (tmp_path / "a.txt").write_text("x x", encoding="utf-8")
    ambiguous = await editor.execute({"path": "a.txt", "old_text": "x", "new_text": "y"})
    assert ambiguous.is_error
    assert "ambiguous" in ambiguous.output


@pytest.mark.asyncio
async def test_bash_success_stdout_stderr_and_nonzero(tmp_path):
    tool = BashTool(Workspace(tmp_path))
    python = shlex.quote(sys.executable)
    success = await tool.execute(
        {"command": f"{python} -c \"import sys; print('out'); print('err', file=sys.stderr)\""}
    )
    assert not success.is_error
    assert success.data["exit_code"] == 0
    assert "out" in success.data["stdout"]
    assert "err" in success.data["stderr"]

    failure = await tool.execute({"command": f"{python} -c \"raise SystemExit(7)\""})
    assert failure.is_error
    assert failure.data["exit_code"] == 7


@pytest.mark.asyncio
async def test_bash_timeout(tmp_path):
    tool = BashTool(Workspace(tmp_path), default_timeout=0.05)
    python = shlex.quote(sys.executable)
    result = await tool.execute(
        {"command": f"{python} -c \"import time; time.sleep(2)\"", "timeout_seconds": 0.05}
    )
    assert result.is_error
    assert result.data["timed_out"] is True
    assert "timed out" in result.output
