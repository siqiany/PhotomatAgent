from __future__ import annotations

import asyncio

import pytest

from photomatagent.scientific.capabilities.config import ScientificConfig
from photomatagent.scientific.capabilities.device import DeviceRunScriptTool
from photomatagent.workspace import Workspace


def _fake_devsim(tmp_path, monkeypatch):
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "devsim.py").write_text("VERSION = 'fake'\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(modules))


@pytest.mark.asyncio
async def test_device_script_can_import_devsim(tmp_path, monkeypatch):
    _fake_devsim(tmp_path, monkeypatch)
    script = tmp_path / "device_job.py"
    script.write_text("import devsim\nprint('devsim-ok')\n", encoding="utf-8")
    result = await asyncio.wait_for(
        DeviceRunScriptTool(ScientificConfig(), Workspace(tmp_path)).execute(
            {"path": script.name, "timeout_seconds": 10}
        ),
        timeout=5,
    )
    assert not result.is_error
    assert result.data["stdout"] == "devsim-ok"


@pytest.mark.asyncio
async def test_device_timeout_terminates_worker(tmp_path, monkeypatch):
    _fake_devsim(tmp_path, monkeypatch)
    script = tmp_path / "never_finishes.py"
    script.write_text("import devsim\nwhile True:\n    pass\n", encoding="utf-8")
    result = await asyncio.wait_for(
        DeviceRunScriptTool(ScientificConfig(), Workspace(tmp_path)).execute(
            {"path": script.name, "timeout_seconds": 1}
        ),
        timeout=5,
    )
    assert result.is_error
    assert result.data["error"] == "timeout"
    assert "terminated" in result.output
