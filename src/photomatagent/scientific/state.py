"""ScientificState: the agent's structured model of the science."""

from __future__ import annotations

from pydantic import BaseModel, Field

from photomatagent.scientific.calculations import CalculationRecord
from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.claims import ScientificClaim
from photomatagent.scientific.evidence import Evidence
from photomatagent.scientific.tasks import ScientificTask


class ScientificState(BaseModel):
    """Deliberately richer than a message history.

    Claims are linked to supporting/contradicting evidence ids; calculations
    are logged as immutable records; open questions and contradictions are
    first-class so future stopping policies can reason about them.
    """

    goal: str = ""
    hypotheses: list[str] = Field(default_factory=list)
    claims: list[ScientificClaim] = Field(default_factory=list)
    evidence: list[Evidence | ScientificEvidence] = Field(default_factory=list)
    calculations: list[CalculationRecord] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    pending_tasks: list[ScientificTask] = Field(default_factory=list)

    def add_evidence(
        self, evidence: Evidence | ScientificEvidence
    ) -> Evidence | ScientificEvidence:
        self.evidence.append(evidence)
        return evidence

    def add_claim(self, claim: ScientificClaim) -> ScientificClaim:
        self.claims.append(claim)
        return claim

    def add_calculation(self, record: CalculationRecord) -> CalculationRecord:
        self.calculations.append(record)
        return record

    def add_task(self, task: ScientificTask) -> ScientificTask:
        self.pending_tasks.append(task)
        return task
