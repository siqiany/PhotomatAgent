"""Discover executable, validated local skills separately from static procedures."""
from __future__ import annotations

import json
import re
from typing import Any

from photomatagent.application.skills import LocalSkillService
from photomatagent.tools.base import Tool, ToolResult
from photomatagent.tools.exposure import ToolExposure


class SkillSearchTool(Tool):
    effect_class = "READ_ONLY"
    name = "skill_search"
    description = (
        "Find executable local methods that passed independent replay checks. "
        "Returns the exact skill_id, scope, required input keys, output paths, and checker contract for skill_run. "
        "Use skill_view for static instructions only; it does not discover executable skill IDs. "
        "Pass paths to the current task's inputs in skill_run."
    )
    short_description = "Find validated local methods and their skill_run contracts."
    exposure = ToolExposure.DIRECT
    namespace = "skills"
    tags = ("skill", "local method", "validated", "discovery")
    input_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 200},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        "required": ["query"],
    }

    def __init__(self, service: LocalSkillService) -> None:
        self.service = service

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        query = str(arguments.get("query", "")).strip().casefold()
        if not query or len(query) > 200:
            return ToolResult(output="SKILL_SEARCH_QUERY_INVALID", is_error=True,
                              data={"code": "SKILL_SEARCH_QUERY_INVALID"})
        limit = min(max(int(arguments.get("limit", 10)), 1), 20)
        terms = tuple(re.findall(r"[a-z0-9]+|[\u3400-\u9fff]+", query))
        ranked: list[tuple[int, int, dict[str, Any]]] = []
        try:
            for record in self.service.list_skills():
                if record.get("status") not in {"TRIALABLE", "REUSED"}:
                    continue
                searchable = " ".join((str(record.get("skill_id", "")), str(record.get("name", "")), str(record.get("description", "")),
                                        str(record.get("scope", "")))).casefold()
                normalized = re.sub(r"[^a-z0-9\u3400-\u9fff]+", " ", searchable)
                haystack = set(normalized.split())
                matched_terms = sum(
                    term in haystack if re.fullmatch(r"[a-z0-9]+", term)
                    else any(term in word for word in haystack)
                    for term in terms
                )
                if not matched_terms:
                    continue
                source_execution = record.get("execution", {})
                checker = record.get("checker") or {}
                result = {
                    "skill_id": record["skill_id"],
                    "name": record["name"],
                    "description": record["description"],
                    "status": record["status"],
                    "scope": record["scope"],
                    "plugin_id": record.get("plugin_id"),
                    "version": record.get("version"),
                    "parent_skill_id": record.get("parent_skill_id"),
                    "contract_sha256": record["contract_sha256"],
                    "interface": record.get("interface"),
                    "input_keys": sorted(record.get("input_keys", record.get("request", {}).get("input_refs", {}))),
                    "output_paths": sorted(artifact["path"] for artifact in source_execution.get("artifacts", [])),
                    "checker": {key: checker[key] for key in (
                        "checker_id", "input_key", "output_path", "field", "units", "minimum", "maximum",
                    ) if key in checker},
                }
                # Recall partial keyword queries, while keeping exact and broader
                # matches ahead of incidental one-word matches.
                ranked.append((matched_terms, int(" ".join(terms) in normalized), result))
        except (OSError, ValueError, KeyError, TypeError):
            return ToolResult(output="SKILL_INDEX_UNAVAILABLE", is_error=True,
                              data={"code": "SKILL_INDEX_UNAVAILABLE"})
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        matches = [result for _, _, result in ranked[:limit]]
        payload = {"matches": matches, "count": len(matches), "executable": True}
        return ToolResult(output=json.dumps(payload, ensure_ascii=False, separators=(",", ":")), data=payload)
