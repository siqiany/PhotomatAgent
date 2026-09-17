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


def test_loader_ignores_symlinked_skill_directory(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "SKILL.md").write_text(
        "---\nname: outside\ndescription: should not load\n---\nsecret",
        encoding="utf-8",
    )
    skills = tmp_path / "skills"
    skills.mkdir()
    try:
        (skills / "linked").symlink_to(outside, target_is_directory=True)
    except OSError:
        return

    loader = SkillLoader(skills)

    assert loader.load_index() == []
    assert loader.load_all() == []


def test_native_composition_generation_skill_routes_to_vae_tool():
    loader = SkillLoader()
    entry = next(
        item
        for item in loader.load_index()
        if item.name == "material-composition-generation"
    )
    assert "成分生成" in entry.description
    body, resolved = loader.view(entry.name)
    assert resolved == "SKILL.md"
    assert "generation.vae_formula" in body
    assert "generation.mattergen" in body
    assert "chgnet.screen" in body
    assert "chgnet.relax" in body
    assert "vasp" in body.lower()


def test_native_mechanism_skill_is_indexed_and_loadable():
    loader = SkillLoader()
    entry = next(
        item
        for item in loader.load_index()
        if item.name == "mechanism-guided-material-design"
    )
    assert entry.category == "materials-discovery"
    assert entry.description.startswith("Use when")
    assert "mechanism" in entry.description
    body, resolved = loader.view(entry.name)
    assert resolved == "SKILL.md"
    for marker in (
        "generation.register_hypothesis",
        "NO_EXTERNAL_BASIS",
        "UNVALIDATED_GENERATED_STRUCTURE",
        "Known-material route",
    ):
        assert marker in body


def test_native_infrared_skill_routes_through_mattergen_chgnet_and_vasp():
    loader = SkillLoader()
    body, _ = loader.view("infrared-material-screening")
    for marker in ("generation.mattergen", "chgnet.screen", "chgnet.relax", "VASP"):
        assert marker in body
    assert "UNVALIDATED_GENERATED_STRUCTURE" in body


def test_ml_potential_skill_has_evidence_hierarchy_and_funnel():
    loader = SkillLoader()
    body, resolved = loader.view("ml-potential-screening")
    assert resolved == "SKILL.md"
    for marker in (
        "generation.mattergen",
        "chgnet.screen",
        "chgnet.relax",
        "VASP",
        "ml_interatomic_potential",
        "ml_potential",
        "UNVALIDATED_GENERATED_STRUCTURE",
    ):
        assert marker in body
