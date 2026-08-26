from __future__ import annotations

from photomatagent.workspace import TMP_DIRNAME, USER_OUTPUT_DIRNAME, Workspace


def test_workspace_creates_user_output_and_tmp_directories(tmp_path):
    workspace = Workspace(tmp_path)
    assert workspace.user_output_dir.is_dir()
    assert workspace.tmp_dir.is_dir()
    assert workspace.user_output_dir.name == USER_OUTPUT_DIRNAME
    assert workspace.tmp_dir.name == TMP_DIRNAME


def test_workspace_directories_stay_inside_boundary(tmp_path):
    workspace = Workspace(tmp_path)
    assert workspace.contains(workspace.user_output_dir)
    assert workspace.contains(workspace.tmp_dir)
