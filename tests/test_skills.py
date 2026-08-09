from __future__ import annotations

from photomatagent.skills.loader import SkillLoader


def test_loader_reads_skill(tmp_path):
    skill_dir = tmp_path / "electronic-structure-analysis"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: electronic-structure-analysis\n"
        "description: SOP for band structure analysis\n"
        "---\n"
        "# Procedure\n"
        "1. Do a thing.\n",
        encoding="utf-8",
    )
    loader = SkillLoader(tmp_path)
    skills = loader.load_all()
    assert len(skills) == 1
    skill = skills[0]
    assert skill.name == "electronic-structure-analysis"
    assert skill.description == "SOP for band structure analysis"
    assert "Do a thing" in skill.content
    assert loader.get("electronic-structure-analysis") == skill


def test_loader_ignores_non_skill_dirs(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "README.md").write_text("no frontmatter here", encoding="utf-8")
    assert SkillLoader(tmp_path).load_all() == []


def test_loader_missing_dir_returns_empty(tmp_path):
    assert SkillLoader(tmp_path / "does-not-exist").load_all() == []
