"""Isolated, structured, read-only LLM Scientific Judge.

Position in the pipeline (after the deterministic ScientificEvaluator):

    Maker -> candidate -> ScientificEvaluator (deterministic Checker)
        -> ScientificJudge (optional, advisory, read-only)
        -> EvaluationReport + JudgeReport -> FeedbackSignal -> ScientificLoopPolicy

Properties:

* ISOLATED -- uses its own ``ModelProvider`` (never the Maker's), and its
  ``ModelRequest`` carries ``tools=[]``, so it cannot call tools by
  construction. It never touches ToolRegistry, permissions, or backends.
* STRUCTURED -- the prompt demands a schema-validated ``JudgeReport`` JSON
  object; anything unparseable degrades to ``JudgeReport(status=UNAVAILABLE)``
  instead of raising.
* READ-ONLY -- ``assess()`` takes an immutable JSON snapshot of the target,
  candidate, evaluation and bounded evidence. It never mutates
  ``ScientificState``, ``ScientificLoopState`` or the conversation.
* NON-AUTHORITATIVE -- the judge is explicitly told it does NOT decide
  whether constraints pass or fail (that is the deterministic evaluator's
  job). Its report can only *raise concerns* that keep the loop investigating;
  it can never convert a deterministic FAIL/UNKNOWN into a PASS, and it never
  rescinds a hard-constraint violation.

Graceful degradation: a provider failure, non-JSON reply or schema mismatch
yields ``UNAVAILABLE``; the deterministic loop and the SUCCESS path continue
to work unless the loop is configured with ``require_judge=True``.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from photomatagent.models.types import (
    ModelCompleted,
    ModelRequest,
    SystemMessage,
    UserMessage,
)
from photomatagent.scientific.loop.candidate import CandidateState
from photomatagent.scientific.loop.evaluation import EvaluationReport
from photomatagent.scientific.loop.target import TargetSpec
from photomatagent.scientific.state import ScientificState

JudgeStatus = Literal["AVAILABLE", "UNAVAILABLE"]
JudgeSeverity = Literal["LOW", "MEDIUM", "HIGH"]
JudgeCategory = Literal[
    "physical_consistency",
    "evidence_quality",
    "evidence_gap",
    "realizability",
    "scope_mismatch",
    "methodology",
    "other",
]

JUDGE_SYSTEM_PROMPT = """You are the Scientific Judge of a materials-design feedback loop.

You assess the SCIENTIFIC QUALITY and RISK of one candidate proposal. You are
strictly ADVISORY:

- You do NOT decide whether target constraints pass or fail. Hard constraints
  are evaluated deterministically elsewhere and always take precedence.
- You must NOT claim a constraint is satisfied when the deterministic
  evaluation reports FAIL or UNKNOWN, and you must NOT rescind a reported
  violation.
- Missing evidence is UNKNOWN, never a pass. You cannot invent values.
- If deterministic evaluation reports PASS but you see a material scientific
  concern (physical inconsistency, unsupported realizability claim, absent
  critical context, methodology problem), raise it as an issue with severity
  HIGH or MEDIUM -- the loop will keep investigating.

Return STRICT JSON matching this schema (no markdown, no prose outside JSON):
{
  "scientific_quality": <float 0..1, overall soundness of the proposal as an
      evidence-backed scientific claim>,
  "issues": [
    {"category": "physical_consistency|evidence_quality|evidence_gap|
         realizability|scope_mismatch|methodology|other",
     "severity": "LOW|MEDIUM|HIGH",
     "property": "<optional property name>",
     "description": "<precise concern>"}
  ],
  "recommendations": ["<concise next action, e.g. a validation step>"],
  "rationale": "<2-3 sentence justification>"
}
If you have no concerns, return an empty "issues" list with a high
scientific_quality and a short rationale.
"""


class JudgeIssue(BaseModel):
    """One advisory scientific concern raised by the LLM Judge."""

    category: JudgeCategory = "other"
    severity: JudgeSeverity = "MEDIUM"
    property: str | None = None
    description: str = ""


class JudgeReport(BaseModel):
    """Structured, advisory output of the LLM Scientific Judge.

    ``status=UNAVAILABLE`` means the judge could not produce a valid report
    (provider failure, non-JSON output, schema mismatch). Such reports never
    block the deterministic decision unless ``require_judge`` is set.
    """

    candidate_id: str = ""
    status: JudgeStatus = "AVAILABLE"
    scientific_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    issues: list[JudgeIssue] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    rationale: str = ""
    provider: str = ""
    model: str = ""
    error: str = ""

    @property
    def available(self) -> bool:
        return self.status == "AVAILABLE"

    @property
    def significant_issues(self) -> list[JudgeIssue]:
        """Issues strong enough to influence feedback/stopping."""
        return [issue for issue in self.issues if issue.severity in {"MEDIUM", "HIGH"}]

    def summary_line(self) -> str:
        if not self.available:
            return f"judge unavailable ({self.error or 'no report'})"
        concerns = [
            f"[{issue.severity}] {issue.category}: {issue.description}"
            for issue in self.significant_issues
        ]
        base = f"judge quality {self.scientific_quality:.2f}"
        return base + (f"; concerns: {'; '.join(concerns)}" if concerns else "")


class ScientificJudge:
    """One small, isolated LLM critic (not a swarm of reflection agents).

    Invariant C extension: the Judge is NOT the generator and NOT the final
    validator -- it is an advisory reader. The deterministic
    ``ScientificEvaluator`` remains the only authority that can turn evidence
    into PASS/FAIL/UNKNOWN.
    """

    def __init__(
        self,
        model: Any,
        *,
        evidence_limit: int = 24,
    ) -> None:
        self.model = model
        self.evidence_limit = max(1, evidence_limit)

    async def assess(
        self,
        *,
        target: TargetSpec,
        candidate: CandidateState,
        scientific: ScientificState,
        evaluation: EvaluationReport,
        round_number: int = 0,
    ) -> JudgeReport:
        """Run the read-only judge on an immutable snapshot.

        Never mutates any state. Any failure degrades to an UNAVAILABLE
        report so the deterministic outer loop is never dependent on the
        judge's availability.
        """
        provider = getattr(self.model, "provider", "unknown")
        model_name = getattr(self.model, "model", "unknown")
        snapshot = self._snapshot(
            target=target,
            candidate=candidate,
            scientific=scientific,
            evaluation=evaluation,
            round_number=round_number,
        )
        request = ModelRequest(
            messages=[
                SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                UserMessage(
                    content=(
                        "Assess the following candidate. Respond with STRICT JSON "
                        "only, matching the schema in the system instructions.\n\n"
                        + json.dumps(snapshot, ensure_ascii=False, indent=2, default=str)
                    )
                ),
            ],
            tools=[],  # the judge can never call tools
        )
        text = ""
        try:
            async for event in self.model.stream(request):
                if isinstance(event, ModelCompleted):
                    text = event.response.text
        except Exception as exc:
            return JudgeReport(
                status="UNAVAILABLE",
                provider=provider,
                model=model_name,
                error=f"judge provider failed: {type(exc).__name__}: {exc}",
            )
        if not text.strip():
            return JudgeReport(
                status="UNAVAILABLE",
                provider=provider,
                model=model_name,
                error="judge returned no completed text",
            )
        try:
            payload = json.loads(_extract_json_object(text))
            report = JudgeReport.model_validate(payload)
        except Exception as exc:
            return JudgeReport(
                status="UNAVAILABLE",
                provider=provider,
                model=model_name,
                error=f"judge output did not match schema: {type(exc).__name__}: {exc}",
            )
        report.provider = provider
        report.model = model_name
        if candidate.candidate_id and not report.candidate_id:
            report.candidate_id = candidate.candidate_id
        return report

    def _snapshot(
        self,
        *,
        target: TargetSpec,
        candidate: CandidateState,
        scientific: ScientificState,
        evaluation: EvaluationReport,
        round_number: int,
    ) -> dict[str, Any]:
        """Immutable, JSON-safe view of everything the judge may read."""
        evidence = [
            item.model_dump(mode="json") for item in scientific.evidence
        ][-self.evidence_limit :]
        return {
            "round": round_number,
            "target": target.model_dump(mode="json"),
            "candidate": candidate.model_dump(mode="json"),
            "evaluation": evaluation.model_dump(mode="json"),
            "evidence": evidence,
            "note": (
                "The deterministic evaluation above is authoritative for "
                "constraint satisfaction. Raise concerns only."
            ),
        }


def _extract_json_object(text: str) -> str:
    """Extract the first balanced JSON object from model output."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1]
        stripped = stripped.rsplit("```", 1)[0].strip()
    start = stripped.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : index + 1]
    raise ValueError("unbalanced JSON object")