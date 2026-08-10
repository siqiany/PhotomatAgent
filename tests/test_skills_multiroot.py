"""Multi-root skill loader tests (native + external sources)."""

from __future__ import annotations

from pathlib import Path

from photomatagent.skills.config import (
    SkillSourceConfig,
    load_skill_sources_config,
)
from photomatagent.skills.loader import SkillLoader


def _write_skill(root: Path, name: str, description: str = "desc", **extra: str) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["---", f"name: {name}", f"description: {description}"]
    for key, value in extra.items():
        lines.append(f"{key}: {value}")
    lines.append("---")
    lines.append("# body")
    (directory / "SKILL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory / "SKILL.md"


def test_loads_skills_from_multiple_roots(tmp_path):
    native = tmp_path / "native"
    external = tmp_path / "external"
    native.mkdir()
    external.mkdir()
    _write_skill(native, "ir-design", "Native IR design skill.")
    _write_skill(external, "vasp-sop", "VASP execution SOP.", license="MIT")
    loader = SkillLoader(
        native,
        sources=[
            SkillSourceConfig(name="native", path=native, priority=100),
            SkillSourceConfig(name="external", path=external, priority=50, license="MIT"),
        ],
    )
    index = loader.load_index()
    names = {entry.name for entry in index}
    assert names == {"ir-design", "vasp-sop"}
    by_name = {entry.name: entry for entry in index}
    assert by_name["ir-design"].source == "native"
    assert by_name["vasp-sop"].source == "external"
    assert by_name["vasp-sop"].license == "MIT"


def test_priority_resolves_name_conflicts(tmp_path):
    native = tmp_path / "native"
    external = tmp_path / "external"
    native.mkdir()
    external.mkdir()
    _write_skill(native, "shared", "native version")
    _write_skill(external, "shared", "external version")
    loader = SkillLoader(
        native,
        sources=[
            SkillSourceConfig(name="native", path=native, priority=100),
            SkillSourceConfig(name="external", path=external, priority=200),
        ],
    )
    index = loader.load_index()
    assert len(index) == 1
    assert index[0].description == "external version"
    assert index[0].source == "external"

    # Lower priority loses.
    loader_low = SkillLoader(
        native,
        sources=[
            SkillSourceConfig(name="native", path=native, priority=100),
            SkillSourceConfig(name="external", path=external, priority=10),
        ],
    )
    assert loader_low.load_index()[0].description == "native version"


def test_missing_external_root_skipped_with_diagnostic(tmp_path):
    native = tmp_path / "native"
    native.mkdir()
    _write_skill(native, "only", "native only")
    missing = tmp_path / "does-not-exist"
    loader = SkillLoader(
        native,
        sources=[
            SkillSourceConfig(name="native", path=native, priority=100),
            SkillSourceConfig(name="ghost", path=missing, priority=50),
        ],
    )
    assert {entry.name for entry in loader.load_index()} == {"only"}
    codes = [diagnostic.code for diagnostic in loader.diagnostics]
    assert "SKIPPED_MISSING_ROOT" in codes


def test_non_skill_directories_skipped_with_diagnostic(tmp_path):
    native = tmp_path / "native"
    (native / "not-a-skill").mkdir(parents=True)
    (native / "not-a-skill" / "README.md").write_text("no frontmatter", encoding="utf-8")
    _write_skill(native, "real-skill")
    loader = SkillLoader(native)
    assert {entry.name for entry in loader.load_index()} == {"real-skill"}
    assert any(
        diagnostic.code == "SKIPPED_NO_SKILL_MD" for diagnostic in loader.diagnostics
    )


def test_progressive_disclosure_keeps_bodies_out_of_index(tmp_path):
    native = tmp_path / "native"
    native.mkdir()
    skill_file = _write_skill(native, "body-skill", "only metadata here")
    loader = SkillLoader(native)
    entries = loader.load_index()
    assert entries[0].description == "only metadata here"
    body, resolved = loader.view("body-skill")
    assert "# body" in body
    assert resolved == "SKILL.md"


def test_yaml_config_sources_and_skip(tmp_path):
    repo = tmp_path / "repo"
    skills = repo / "skills"
    skills.mkdir(parents=True)
    _write_skill(skills, "native-a", "native a")
    config_dir = repo / ".photomatagent"
    config_dir.mkdir()
    (config_dir / "skills.yaml").write_text(
        "skill_sources:\n"
        "  - name: photomat\n"
        "    path: ./skills\n"
        "    priority: 100\n"
        "  - name: atomistic-skills\n"
        "    path: ../AtomisticSkills/.agents/skills\n"
        "    priority: 50\n",
        encoding="utf-8",
    )
    import os

    previous = os.getcwd()
    os.chdir(repo)
    try:
        config = load_skill_sources_config()
        assert len(config.sources) == 1
        assert config.sources[0].name == "photomat"
        assert config.sources[0].path == skills.resolve()
        assert any(d.code == "SKIPPED_MISSING_ROOT" for d in config.diagnostics)
    finally:
        os.chdir(previous)
