"""Progressive skill index and on-demand SKILL/reference loading."""

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
    category: str = ""
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class SkillIndexEntry:
    name: str
    description: str
    category: str
    tags: tuple[str, ...]
    path: Path


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
    env = os.environ.get("PHOTOMATAGENT_SKILLS_DIR") or os.environ.get("MATAGENT_SKILLS_DIR")
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
        """Compatibility API: explicitly load every primary SKILL.md."""
        if not self.skills_dir.is_dir():
            return []
        skills: list[Skill] = []
        for entry in sorted(self.skills_dir.iterdir()):
            if not self._safe_skill_directory(entry):
                continue
            skill_file = entry / "SKILL.md"
            if not self._safe_skill_file(entry, skill_file):
                continue
            content = skill_file.read_text(encoding="utf-8")
            metadata = _parse_frontmatter(content)
            name = metadata.get("name", entry.name)
            description = metadata.get("description", "")
            category = metadata.get("category", "")
            tags = _parse_tags(metadata.get("tags", ""))
            skills.append(
                Skill(
                    name=name,
                    description=description,
                    path=skill_file,
                    content=content,
                    category=category,
                    tags=tags,
                )
            )
        return sorted(skills, key=lambda s: s.name)

    def get(self, name: str) -> Skill | None:
        return next((s for s in self.load_all() if s.name == name), None)

    def load_index(self) -> list[SkillIndexEntry]:
        """Read frontmatter only; full skill bodies are not loaded or returned."""
        if not self.skills_dir.is_dir():
            return []
        entries: list[SkillIndexEntry] = []
        for directory in sorted(self.skills_dir.iterdir()):
            skill_file = directory / "SKILL.md"
            if not self._safe_skill_directory(directory) or not self._safe_skill_file(
                directory, skill_file
            ):
                continue
            metadata = _read_frontmatter(skill_file)
            entries.append(
                SkillIndexEntry(
                    name=metadata.get("name", directory.name),
                    description=metadata.get("description", ""),
                    category=metadata.get("category", ""),
                    tags=_parse_tags(metadata.get("tags", "")),
                    path=skill_file,
                )
            )
        return sorted(entries, key=lambda entry: entry.name)

    def view(self, name: str, path: str | None = None) -> tuple[str, str]:
        """Load the primary SKILL.md or one safe path below its directory."""
        entry = next((item for item in self.load_index() if item.name == name), None)
        if entry is None:
            raise KeyError(f"unknown skill: {name}")
        root = entry.path.parent.resolve()
        target = entry.path if path is None else root / path
        resolved = target.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError("skill reference escapes its skill directory")
        if not resolved.is_file():
            raise OSError(f"skill reference is not a file: {path}")
        return resolved.read_text(encoding="utf-8"), resolved.relative_to(root).as_posix()

    def _safe_skill_directory(self, directory: Path) -> bool:
        if directory.is_symlink() or not directory.is_dir():
            return False
        skills_root = self.skills_dir.resolve()
        resolved = directory.resolve()
        return resolved != skills_root and skills_root in resolved.parents

    def _safe_skill_file(self, directory: Path, skill_file: Path) -> bool:
        if not skill_file.is_file():
            return False
        root = directory.resolve()
        resolved = skill_file.resolve()
        return resolved != root and root in resolved.parents


def _parse_tags(value: str) -> tuple[str, ...]:
    cleaned = value.strip().strip("[]")
    return tuple(
        part.strip().strip("'\"")
        for part in cleaned.split(",")
        if part.strip().strip("'\"")
    )


def _read_frontmatter(path: Path) -> dict[str, str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        if handle.readline().strip() != "---":
            return {}
        for line in handle:
            if line.strip() == "---":
                break
            lines.append(line.rstrip("\n"))
    return _parse_frontmatter("---\n" + "\n".join(lines) + "\n---\n")
