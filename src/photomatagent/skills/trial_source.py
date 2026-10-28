"""Discovery adapter for validated local replay methods, separate from QD publication."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from photomatagent.application.skills import LocalSkillService


class LocalTrialSkillSource:
    def __init__(self, workspace: Path | str) -> None:
        self.service = LocalSkillService(workspace)

    def load_index(self) -> list[Any]:
        from photomatagent.skills.loader import SkillIndexEntry
        return [SkillIndexEntry(name=record["name"], description=record["description"],
                                path=self.service._directory(record["skill_id"]) / "SKILL.md",
                                source="local-trial", priority=40, category="validated-local-method",
                                tags=("local", "trialable", record["scope"]))
                for record in self.service.list_skills() if record["status"] in {"TRIALABLE", "REUSED"}]

    def view(self, name: str, path: str | None = None) -> tuple[str, str]:
        record = next((r for r in self.service.list_skills() if r.get("name") == name and r["status"] in {"TRIALABLE", "REUSED"}), None)
        if record is None:
            raise KeyError(name)
        relative = path or "SKILL.md"
        # Inputs are private source fixtures, never progressive model context.
        if relative not in {"SKILL.md", "script.py"}:
            raise ValueError("LOCAL_SKILL_REFERENCE_PRIVATE")
        self.service.read(record["skill_id"])
        return (self.service._directory(record["skill_id"]) / relative).read_text(), relative
