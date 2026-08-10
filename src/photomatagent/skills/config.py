"""Skill source configuration (``skill_sources``) loading.

The configuration is a YAML file (JSON also accepted) shaped like:

.. code-block:: yaml

    skill_sources:
      - name: photomat
        path: ./skills
        priority: 100
      - name: atomistic-skills
        path: ../AtomisticSkills/.agents/skills
        priority: 50
      - name: computational-chemistry-skills
        path: ../computational-chemistry-agent-skills
        priority: 40

Search order for the config file:
``PHOTOMATAGENT_SKILL_SOURCES`` env var, then ``.photomatagent/skills.yaml``
in the workspace, then ``skills.yaml`` in the current directory. A missing or
unparsable config never breaks startup: it degrades to the native skills root.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from photomatagent.skills.descriptor import SkillDiagnostic


@dataclass(frozen=True)
class SkillSourceConfig:
    name: str
    path: Path
    priority: int = 100
    license: str = ""


@dataclass(frozen=True)
class SkillSourcesConfig:
    sources: list[SkillSourceConfig] = field(default_factory=list)
    diagnostics: list[SkillDiagnostic] = field(default_factory=list)


def default_skill_sources_config_path() -> Path | None:
    """Return the first existing skill source config path, if any."""
    env = os.environ.get("PHOTOMATAGENT_SKILL_SOURCES")
    if env:
        candidate = Path(env).expanduser()
        if candidate.is_file():
            return candidate
    dot = Path.cwd() / ".photomatagent" / "skills.yaml"
    if dot.is_file():
        return dot
    root = Path.cwd() / "skills.yaml"
    if root.is_file():
        return root
    return None


def load_skill_sources_config(
    path: Path | str | None = None,
) -> SkillSourcesConfig:
    """Load ``skill_sources`` from YAML/JSON, degrading gracefully on errors."""
    diagnostics: list[SkillDiagnostic] = []
    sources: list[SkillSourceConfig] = []
    selected = Path(path) if path is not None else default_skill_sources_config_path()
    if selected is None or not selected.is_file():
        return SkillSourcesConfig(sources=[], diagnostics=diagnostics)
    try:
        raw = _parse_config_file(selected)
    except Exception as exc:  # pragma: no cover - depends on parser internals
        diagnostics.append(
            SkillDiagnostic(
                code="CONFIG_UNPARSEABLE",
                message=f"skill source config could not be parsed: {exc}",
                path=selected,
            )
        )
        return SkillSourcesConfig(sources=[], diagnostics=diagnostics)
    entries = raw.get("skill_sources", []) if isinstance(raw, dict) else None
    if not isinstance(entries, list):
        diagnostics.append(
            SkillDiagnostic(
                code="CONFIG_NO_SOURCES",
                message="config has no valid 'skill_sources' list",
                path=selected,
            )
        )
        return SkillSourcesConfig(sources=[], diagnostics=diagnostics)
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            diagnostics.append(
                SkillDiagnostic(
                    code="CONFIG_BAD_ENTRY",
                    message=f"skill_sources[{index}] is not a mapping",
                    path=selected,
                )
            )
            continue
        name = str(entry.get("name", "")).strip()
        path_value = str(entry.get("path", "")).strip()
        if not name or not path_value:
            diagnostics.append(
                SkillDiagnostic(
                    code="CONFIG_BAD_ENTRY",
                    message=f"skill_sources[{index}] needs non-empty name and path",
                    path=selected,
                )
            )
            continue
        try:
            priority = int(entry.get("priority", 100))
        except (TypeError, ValueError):
            priority = 100
        resolved = _resolve_skill_root(path_value, selected)
        if not resolved.is_dir():
            diagnostics.append(
                SkillDiagnostic(
                    code="SKIPPED_MISSING_ROOT",
                    message=f"skill source '{name}' directory does not exist",
                    source=name,
                    path=resolved,
                )
            )
            continue
        sources.append(
            SkillSourceConfig(
                name=name,
                path=resolved,
                priority=priority,
                license=str(entry.get("license", "")).strip(),
            )
        )
    return SkillSourcesConfig(sources=sources, diagnostics=diagnostics)


def _resolve_skill_root(path_value: str, config_path: Path) -> Path:
    """Resolve a relative skill root against cwd, then config locations."""
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    bases = [
        Path.cwd(),
        config_path.parent,
        config_path.parent.parent,
    ]
    for base in bases:
        resolved = (base / candidate).resolve()
        if resolved.is_dir():
            return resolved
    return (Path.cwd() / candidate).resolve()


def _parse_config_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - yaml is a base dependency
        raise RuntimeError("PyYAML is required to read skill source YAML") from exc
    value = yaml.safe_load(text)
    return value if isinstance(value, dict) else {}
