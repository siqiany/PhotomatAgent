"""Progressive source for approved, non-executable procedural guidance."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from photomatagent.application.procedural_skills import ProcedureService


class LocalProcedureSkillSource:
    source_name = "local-procedure"
    priority = 35

    def __init__(
        self, workspace: Path | str, *, task_id: str | None = None, run_id: str | None = None,
    ) -> None:
        self.service = ProcedureService(workspace)
        self.task_id = task_id
        self.run_id = run_id

    def load_index(self) -> list[Any]:
        from photomatagent.skills.loader import SkillIndexEntry

        entries = []
        for item in self.service.list_skills():
            if item.get("status") != "APPROVED_PROCEDURE":
                continue
            try:
                record = self.service.read(item["skill_id"])
            except (OSError, ValueError, TypeError, KeyError):
                continue
            entries.append(SkillIndexEntry(
                name=record["index_name"],
                description=f"Approved procedural guidance: {record['description']} (unvalidated science)",
                path=self.service.root / record["skill_id"] / "SKILL.md",
                source=self.source_name,
                priority=self.priority,
                category="procedural-guidance",
                tags=("local", "procedure", "approved-guidance"),
            ))
        return entries

    def view(self, name: str, path: str | None = None) -> tuple[str, str]:
        if path not in (None, "SKILL.md"):
            raise ValueError("PROCEDURE_REFERENCE_NOT_AVAILABLE")
        for entry in self.load_index():
            if entry.name == name:
                skill_id = entry.path.parent.name
                record = self.service.read(skill_id, task_id=self.task_id, run_id=self.run_id)
                if record["status"] != "APPROVED_PROCEDURE":
                    break
                return _model_view(record), "SKILL.md"
        raise KeyError(f"unknown approved procedure: {name}")


def _model_view(record: dict[str, Any]) -> str:
    """Neutralize legacy attribution claims in model-visible text without rewriting snapshots."""
    content = record["skill_content"]
    frontmatter, separator, body = content.partition("---\n")
    if content.startswith("---\n") and separator:
        metadata, closing, remainder = body.partition("\n---\n")
        if closing:
            def neutralize_tags(match: re.Match[str]) -> str:
                tags = [tag.strip() for tag in match.group(1).split(",") if tag.strip()]
                tags = [tag for tag in tags if tag.lower() != "human-reviewed"]
                if "approved-guidance" not in {tag.lower() for tag in tags}:
                    tags.append("approved-guidance")
                return "tags: " + ", ".join(tags)

            metadata = re.sub(r"(?im)^tags:\s*([^\n]*)$", neutralize_tags, metadata)
            content = frontmatter + separator + metadata + closing + remainder
    before_instructions, marker, instructions = content.partition("## Instructions")
    before_instructions = re.sub(
        r"(?m)^This is procedural guidance drafted from prior work and reviewed by a human\.",
        "Procedural guidance drafted from prior work; approval status is recorded separately.",
        before_instructions,
    )
    content = before_instructions + marker + instructions
    note = record.get("review_note")
    if isinstance(note, str) and note.strip():
        content = content.rstrip() + "\n\n## Review record\n\n" + note.strip() + "\n"
    return content
