"""Deferred tool for proposing a typed hypothesis registration update."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from photomatagent.scientific.discovery.models import (
    HypothesisProposal,
    HypothesisRegistration,
)
from photomatagent.scientific.discovery.registration import (
    candidate_id,
    hypothesis_id,
    validate_proposal,
)
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.base import Tool, ToolResult
from photomatagent.tools.exposure import ToolExposure


class RegisterHypothesisTool(Tool):
    name = "generation.register_hypothesis"
    description = (
        "Validate and register an unvalidated mechanism-guided material hypothesis. "
        "This records a proposal and validation questions; it does not validate any "
        "scientific claim."
    )
    short_description = "Register an unvalidated material hypothesis with provenance."
    exposure = ToolExposure.DEFERRED
    namespace = "generation"
    source = "photomatagent mechanism discovery"
    tags = ("generation", "hypothesis", "mechanism", "candidate")
    cost_class = "CHEAP"
    input_schema: dict[str, Any] = HypothesisProposal.model_json_schema()

    def __init__(self, state: ScientificState | None) -> None:
        self.state = state

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        if self.state is None:
            payload = {
                "error": "STATE_UNAVAILABLE",
                "message": "hypothesis registration requires a live ScientificState",
            }
            return ToolResult(
                output=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                is_error=True,
                data={"error_type": "STATE_UNAVAILABLE"},
            )
        try:
            proposal = HypothesisProposal.model_validate(arguments)
            diagnostics = validate_proposal(proposal, self.state)
        except (ValidationError, ValueError) as exc:
            payload = {
                "error": "INVALID_HYPOTHESIS",
                "message": str(exc),
            }
            return ToolResult(
                output=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                is_error=True,
                data={"error_type": "INVALID_HYPOTHESIS", "message": str(exc)},
            )

        data = {
            "registration": "accepted",
            "hypothesis_id": hypothesis_id(proposal),
            "candidate_id": candidate_id(proposal),
            "diagnostics": diagnostics,
            "validation_status": "UNVALIDATED_HYPOTHESIS",
        }
        return ToolResult(
            output=json.dumps(
                {"registration": "accepted", "diagnostics": diagnostics},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            data=data,
            state_updates=[HypothesisRegistration(proposal=proposal)],
        )

