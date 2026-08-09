"""SkillLoader: scan a skills directory and read SKILL.md files.

No skill selection / execution this phase; the loader only surfaces metadata
and content so future loops can decide when to apply a skill.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path
    content: str


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def _parse_frontmatter(content: str) -> dict[str, str]:
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return {}
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
    return fields


def default_skills_dir() -> Path:
    """Resolve the skills directory: env override, then cwd, then repo root."""
    env = os.environ.get("MATAGENT_SKILLS_DIR")
    if env:
        return Path(env)
    cwd_skills = Path.cwd() / "skills"
    if cwd_skills.is_dir():
        return cwd_skills
    repo_skills = Path(__file__).resolve().parents[3] / "skills"
    return repo_skills


class SkillLoader:
    def __init__(self, skills_dir: Path | str | None = None) -> None:
        self.skills_dir = Path(skills_dir) if skills_dir is not None else default_skills_dir()

    def load_all(self) -> list[Skill]:
        if not self.skills_dir.is_dir():
            return []
        skills: list[Skill] = []
        for entry in sorted(self.skills_dir.iterdir()):
            if not entry.is_dir():
                continue
            skill_file = entry / "SKILL.md"
            if not skill_file.is_file():
                continue
            content = skill_file.read_text(encoding="utf-8")
            metadata = _parse_frontmatter(content)
            name = metadata.get("name", entry.name)
            description = metadata.get("description", "")
            skills.append(
                Skill(name=name, description=description, path=skill_file, content=content)
            )
        return sorted(skills, key=lambda s: s.name)

    def get(self, name: str) -> Skill | None:
        return next((s for s in self.load_all() if s.name == name), None)
