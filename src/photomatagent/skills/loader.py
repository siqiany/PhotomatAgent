"""Multi-root progressive skill index and on-demand SKILL/reference loading.

Skills come from one or more roots (native ``skills/`` plus external source
directories). Only ``name + description`` (plus provenance metadata) is ever
loaded eagerly; full SKILL.md bodies and references load on demand via
``view()``. Third-party roots that do not follow the SKILL.md convention are
skipped with a diagnostic instead of failing startup.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

from photomatagent.skills.config import (
    SkillSourceConfig,
    load_skill_sources_config,
)
from photomatagent.skills.descriptor import SkillDescriptor, SkillDiagnostic


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
    """Resolve the native skills directory (env override, cwd, then repo)."""
    env = os.environ.get("PHOTOMATAGENT_SKILLS_DIR") or os.environ.get("MATAGENT_SKILLS_DIR")
    if env:
        return Path(env)
    cwd_skills = Path.cwd() / "skills"
    if cwd_skills.is_dir():
        return cwd_skills
    repo_skills = Path(__file__).resolve().parents[3] / "skills"
    return repo_skills


class SkillLoader:
    """Progressive loader over multiple skill source roots.

    ``skills_dir`` is retained for backwards compatibility: when provided
    explicitly it becomes the only native root unless ``sources``/config
    resolution finds configured external roots. Diagnostics never raise.
    """

    def __init__(
        self,
        skills_dir: Path | str | None = None,
        *,
        sources: Iterable[SkillSourceConfig] | None = None,
        config_path: Path | str | None = None,
    ) -> None:
        self.skills_dir = Path(skills_dir) if skills_dir is not None else default_skills_dir()
        self._explicit_skills_dir = skills_dir is not None
        self.diagnostics: list[SkillDiagnostic] = []
        self._sources = self._resolve_sources(sources, config_path)

    @property
    def sources(self) -> list[SkillSourceConfig]:
        return list(self._sources)

    def _resolve_sources(
        self,
        sources: Iterable[SkillSourceConfig] | None,
        config_path: Path | str | None,
    ) -> list[SkillSourceConfig]:
        if sources is not None:
            resolved = [self._normalize_source(source) for source in sources]
            return self._dedupe(resolved)
        if self._explicit_skills_dir:
            # Backwards-compatible isolated root: an explicit skills_dir means
            # "only this directory", without config-file auto-discovery.
            native = SkillSourceConfig(
                name="native",
                path=self.skills_dir.resolve(),
                priority=100,
            )
            return self._dedupe([native])
        loaded = load_skill_sources_config(config_path)
        self.diagnostics.extend(loaded.diagnostics)
        resolved = [self._normalize_source(source) for source in loaded.sources]
        native = SkillSourceConfig(
            name="native",
            path=self.skills_dir.resolve(),
            priority=100,
        )
        resolved.append(native)
        return self._dedupe(resolved)

    def _normalize_source(self, source: SkillSourceConfig) -> SkillSourceConfig:
        path = source.path.expanduser().resolve()
        if path.is_dir():
            return SkillSourceConfig(
                name=source.name,
                path=path,
                priority=source.priority,
                license=source.license,
            )
        self.diagnostics.append(
            SkillDiagnostic(
                code="SKIPPED_MISSING_ROOT",
                message=f"skill source '{source.name}' directory does not exist",
                source=source.name,
                path=path,
            )
        )
        return SkillSourceConfig(name=source.name, path=path, priority=0)

    @staticmethod
    def _dedupe(sources: list[SkillSourceConfig]) -> list[SkillSourceConfig]:
        by_path: dict[Path, SkillSourceConfig] = {}
        for source in sources:
            if source.path in by_path:
                existing = by_path[source.path]
                if source.priority > existing.priority:
                    by_path[source.path] = source
            else:
                by_path[source.path] = source
        return sorted(by_path.values(), key=lambda item: (-item.priority, item.name))

    def load_all(self) -> list[Skill]:
        """Compatibility API: explicitly load every primary SKILL.md."""
        return sorted(
            [entry.to_skill() for entry in self.load_index()],
            key=lambda skill: skill.name,
        )

    def get(self, name: str) -> Skill | None:
        return next((s for s in self.load_all() if s.name == name), None)

    def load_index(self) -> list[SkillIndexEntry]:
        """Read frontmatter only; full skill bodies are not loaded or returned."""
        entries: list[SkillIndexEntry] = []
        for source in self._sources:
            root = source.path
            if not root.is_dir():
                continue
            for directory in sorted(root.iterdir()):
                if not self._safe_skill_directory(root, directory):
                    continue
                skill_file = directory / "SKILL.md"
                if not self._safe_skill_file(directory, skill_file):
                    self.diagnostics.append(
                        SkillDiagnostic(
                            code="SKIPPED_NO_SKILL_MD",
                            message=f"'{directory.name}' has no readable SKILL.md",
                            source=source.name,
                            path=directory,
                        )
                    )
                    continue
                metadata = _read_frontmatter(skill_file)
                if not metadata:
                    self.diagnostics.append(
                        SkillDiagnostic(
                            code="SKIPPED_UNPARSEABLE",
                            message=f"'{directory.name}' SKILL.md has no parseable frontmatter",
                            source=source.name,
                            path=skill_file,
                        )
                    )
                    continue
                entries.append(
                    SkillIndexEntry(
                        name=metadata.get("name", directory.name),
                        description=metadata.get("description", ""),
                        category=metadata.get("category", ""),
                        tags=_parse_tags(metadata.get("tags", "")),
                        path=skill_file,
                        source=source.name,
                        license=(
                            metadata.get("license", "")
                            or metadata.get("license_name", "")
                            or source.license
                        ),
                        priority=source.priority,
                    )
                )
        return self._merge(entries)

    def _merge(self, entries: list[SkillIndexEntry]) -> list[SkillIndexEntry]:
        merged: dict[str, SkillIndexEntry] = {}
        for entry in sorted(entries, key=lambda item: (-item.priority, item.name)):
            existing = merged.get(entry.name)
            if existing is None or entry.priority > existing.priority:
                merged[entry.name] = entry
        return sorted(merged.values(), key=lambda entry: entry.name)

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

    def _safe_skill_directory(self, root: Path, directory: Path) -> bool:
        if directory.is_symlink() or not directory.is_dir():
            return False
        skills_root = root.resolve()
        resolved = directory.resolve()
        return resolved != skills_root and skills_root in resolved.parents

    def _safe_skill_file(self, directory: Path, skill_file: Path) -> bool:
        if not skill_file.is_file():
            return False
        root = directory.resolve()
        resolved = skill_file.resolve()
        return resolved != root and root in resolved.parents


class Skill(SkillDescriptor):
    """A skill with its body loaded (progressive disclosure)."""

    content: str

    def __init__(
        self,
        *,
        name: str,
        description: str,
        path: Path,
        content: str,
        source: str = "native",
        license: str = "",
        category: str = "",
        tags: tuple[str, ...] = (),
        priority: int = 100,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            path=path,
            source=source,
            license=license,
            tags=tags,
            priority=priority,
            category=category,
        )
        self.content = content


class SkillIndexEntry(SkillDescriptor):
    """Frontmatter-only view of a skill (name + description + provenance)."""

    def to_skill(self) -> Skill:
        content = self.path.read_text(encoding="utf-8")
        return Skill(
            name=self.name,
            description=self.description,
            path=self.path,
            content=content,
            source=self.source,
            license=self.license,
            category=self.category,
            tags=self.tags,
            priority=self.priority,
        )


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
