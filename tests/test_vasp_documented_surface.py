"""Task 15: documented VASP surface and public-path safety checks."""

from __future__ import annotations

from pathlib import Path

from photomatagent.scientific.state import ScientificState
from photomatagent.tools.factory import create_default_registry
from photomatagent.tools.surface import ToolSurfaceConfig, ToolSurfacePlanner
from photomatagent.workspace import Workspace

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS = [
    REPO_ROOT / "skills" / "vasp-hpc-operator" / "SKILL.md",
    REPO_ROOT / "skills" / "molecular-vasp-study" / "SKILL.md",
]
LEGACY_SKILL_NAMES = (
    "vasp_molecule.",
    "vasp_study.",
    "vasp_molecule.*",
    "vasp_study.*",
    "vasp.inspect_result",
    "vasp.run_workflow",
)
PUBLIC = {
    "vasp.capabilities",
    "vasp.plan",
    "vasp.prepare",
    "vasp.preflight",
    "vasp.submit",
    "vasp.status",
    "vasp.wait",
    "vasp.resume",
    "vasp.collect",
    "vasp.report",
}


def test_active_skills_contain_no_legacy_vasp_instructions(tmp_path):
    for skill_path in SKILLS:
        text = skill_path.read_text(encoding="utf-8")
        for legacy in LEGACY_SKILL_NAMES:
            assert legacy not in text, f"{skill_path} still references {legacy}"


def test_public_vasp_unified_path_never_calls_backend_submit_script_directly():
    unified_dir = REPO_ROOT / "src" / "photomatagent" / "scientific" / "applications" / "vasp" / "unified"
    for path in unified_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "submit_script(" not in text, f"{path} calls submit_script directly"
    tool_pack = (
        REPO_ROOT
        / "src"
        / "photomatagent"
        / "scientific"
        / "applications"
        / "vasp"
        / "unified"
        / "tool_pack.py"
    )
    assert "submit_script(" not in tool_pack.read_text(encoding="utf-8")


def test_registry_public_vasp_names_equal_documented_surface(tmp_path):
    registry = create_default_registry(ScientificState(), Workspace(tmp_path))
    for mode in ("progressive", "eager"):
        plan = ToolSurfacePlanner(
            registry, ToolSurfaceConfig(mode=mode)
        ).plan()
        names = {item.name for item in plan.definitions}
        if mode == "progressive":
            # Progressive manifest carries deferred names as text.
            import re

            names.update(
                re.findall(
                    r"vasp\.[A-Za-z0-9_.]+",
                    plan.manifest.text,
                )
            )
        visible = {name for name in names if name.startswith("vasp")}
        assert visible == PUBLIC


def test_submit_once_is_the_only_public_submission_entry():
    # The unified periodic adapter routes through SubmitOnceSession, not
    # through the backend's submit_script method directly.
    periodic = (
        REPO_ROOT
        / "src"
        / "photomatagent"
        / "scientific"
        / "applications"
        / "vasp"
        / "unified"
        / "periodic.py"
    )
    text = periodic.read_text(encoding="utf-8")
    assert "submit_once(" in text
    assert "submit_script(" not in text
