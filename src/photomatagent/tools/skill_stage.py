"""Runtime-gated tool for staging a QD candidate, never for promoting it."""
from __future__ import annotations

import json
from typing import Any, Mapping

from photomatagent.scientific.evolution.qd.models import SkillPackage
from photomatagent.scientific.evolution.qd.skill_package import SkillPackageBuilder
from photomatagent.tools.base import Tool, ToolResult
from photomatagent.tools.exposure import ToolExposure


class SkillStageTool(Tool):
    name = "skill_stage"
    description = "Stage a candidate skill from the bound validated QD trial; this cannot replay, approve, or publish it."
    short_description = "Stage a candidate skill bundle."
    exposure = ToolExposure.DEFERRED
    namespace = "skills"
    tags = ("qd", "skill", "candidate", "stage")
    input_schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "trial_id": {"type": "string", "minLength": 1, "maxLength": 200},
            "description": {"type": "string", "minLength": 1, "maxLength": 500},
            "applicability": {
                "type": "array", "minItems": 1, "maxItems": 10,
                "items": {"type": "string", "minLength": 1, "maxLength": 100},
            },
        },
        "required": ["trial_id", "description", "applicability"],
    }

    def __init__(
        self, *, builder: SkillPackageBuilder, bound_trial_id: str,
        replay_input: Mapping[str, Any], replay_expected: Mapping[str, Any],
        experience_refs: tuple[str, ...] = (),
    ) -> None:
        self.builder = builder
        self.bound_trial_id = bound_trial_id
        self.replay_input = dict(replay_input)
        self.replay_expected = dict(replay_expected)
        self.experience_refs = experience_refs

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        if (set(arguments) != {"trial_id", "description", "applicability"}
                or arguments.get("trial_id") != self.bound_trial_id
                or not isinstance(arguments.get("applicability"), list)
                or any(not isinstance(item, str) for item in arguments["applicability"])):
            return ToolResult(output="INVALID_SCHEMA", is_error=True, data={"code": "INVALID_SCHEMA"})
        try:
            package: SkillPackage = self.builder.stage(
                trial_id=self.bound_trial_id,
                description=str(arguments["description"]),
                applicability=tuple(arguments["applicability"]),
                replay_input=self.replay_input,
                replay_expected=self.replay_expected,
                experience_refs=self.experience_refs,
            )
        except (ValueError, OSError, FileExistsError) as exc:
            code = str(exc).split(":", maxsplit=1)[0]
            if not code or len(code) > 100:
                code = "SKILL_STAGE_FAILED"
            return ToolResult(output=code, is_error=True, data={"code": code})
        payload = {
            "candidate_id": package.skill_id,
            "bundle_sha256": package.bundle_sha256,
            "status": package.status,
            "missing_conditions": [
                "independent replay", "five linked task groups with REUSABLE_SKILL maturity",
                "explicit approval for this exact bundle",
            ],
        }
        return ToolResult(
            output=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            data={"candidate_id": package.skill_id, "bundle_sha256": package.bundle_sha256,
                  "status": package.status},
        )
