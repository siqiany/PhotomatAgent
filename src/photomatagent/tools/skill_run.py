"""Deferred local replay tool; AgentRuntime owns permissions and invocation."""
from __future__ import annotations

import json
from typing import Any

from photomatagent.application.errors import ServiceError
from photomatagent.application.skills import LocalSkillService, SkillReplay
from photomatagent.tools.base import Tool, ToolResult


class SkillRunTool(Tool):
    name = "skill_run"
    namespace = "skills"
    description = "Execute a validated local method with current task inputs and its frozen trusted checker. Discover methods using skill_view."
    tags = ("skill", "method", "replay", "execution")
    input_schema = {"type": "object", "additionalProperties": False,
                    "properties": {"skill_id": {"type": "string"}, "scope": {"type": "string"},
                                   "input_refs": {"type": "object", "additionalProperties": {"type": "string"}},
                                   "argv": {"type": "array", "maxItems": 32, "items": {"type": "string", "maxLength": 1000}}},
                    "required": ["skill_id", "scope", "input_refs"]}

    def __init__(self, service: LocalSkillService, task_id: str, run_id: str | None = None) -> None:
        self.service = service
        self.task_id = task_id
        self.run_id = run_id

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            payload = SkillReplay(task_id=self.task_id, scope=arguments["scope"],
                                  input_refs=arguments["input_refs"], argv=arguments.get("argv"))
            use = await self.service.replay(str(arguments["skill_id"]), payload, model_requested=True, run_id=self.run_id)
            # Checker sees input fixtures; the model sees only evidence and current output paths.
            summary = {"skill_id": use["skill_id"], "skill_use": use["use_id"], "code": use["code"],
                       "checker": use["checker"], "artifacts": use["artifacts"],
                       "execution_id": use["execution"]["execution_id"] if use["execution"] else None}
            from photomatagent.scientific.capabilities.contracts import ScientificEvidence
            evidence = ScientificEvidence(subject=f"local method {use['skill_id']}", property="deterministic_method_check",
                                          value=use["checker"]["status"], source=use["use_id"], source_type="derived",
                                          method=str(use["checker"].get("method", "trusted local checker")),
                                          unit=str(use["checker"].get("units", "")),
                                          limitations="Local artifact contract validation only; no material hypothesis is established.",
                                          provenance={"contract_sha256": use["contract_sha256"], "execution_id": summary["execution_id"],
                                                      "input_sha256": use.get("input_sha256"), "checker": use["checker"]})
            return ToolResult(output=json.dumps(summary), data=summary, state_updates=[evidence],
                              is_error=use["code"] != "OK" or use["checker"]["status"] != "PASS")
        except (ServiceError, ValueError, KeyError) as exc:
            code = exc.code if isinstance(exc, ServiceError) else "SKILL_ARGUMENT_INVALID"
            return ToolResult(output=code, data={"code": code}, is_error=True)
