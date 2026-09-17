"""ScientificState: the agent's structured model of the science."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from photomatagent.scientific.calculations import CalculationRecord
from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.claims import ScientificClaim
from photomatagent.scientific.evidence import Evidence
from photomatagent.scientific.discovery.models import ScientificHypothesis
from photomatagent.scientific.tasks import ScientificTask


class EvidenceAttestation(BaseModel):
    """Host-owned authority record, stored separately from producer evidence."""

    evidence_id: str
    authority: Literal["observation", "synthetic", "background"] = "background"
    origin: Literal["trusted_builtin", "synthetic_test", "untrusted_tool"]
    tool_name: str
    tool_call_id: str


class ScientificState(BaseModel):
    """Deliberately richer than a message history.

    Claims are linked to supporting/contradicting evidence ids; calculations
    are logged as immutable records; open questions and contradictions are
    first-class so future stopping policies can reason about them.
    """

    goal: str = ""
    hypotheses: list[str] = Field(default_factory=list)
    material_hypotheses: list[ScientificHypothesis] = Field(default_factory=list)
    claims: list[ScientificClaim] = Field(default_factory=list)
    evidence: list[Evidence | ScientificEvidence] = Field(default_factory=list)
    evidence_attestations: dict[str, EvidenceAttestation] = Field(default_factory=dict)
    calculations: list[CalculationRecord] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    pending_tasks: list[ScientificTask] = Field(default_factory=list)

    def add_evidence(
        self, evidence: Evidence | ScientificEvidence
    ) -> Evidence | ScientificEvidence:
        self.evidence.append(evidence)
        return evidence

    def attest_evidence(self, attestation: EvidenceAttestation) -> EvidenceAttestation:
        if not any(item.id == attestation.evidence_id for item in self.evidence):
            raise ValueError("cannot attest evidence that is absent from scientific state")
        self.evidence_attestations[attestation.evidence_id] = attestation
        return attestation

    def add_claim(self, claim: ScientificClaim) -> ScientificClaim:
        self.claims.append(claim)
        return claim

    def add_calculation(self, record: CalculationRecord) -> CalculationRecord:
        self.calculations.append(record)
        return record

    def add_task(self, task: ScientificTask) -> ScientificTask:
        self.pending_tasks.append(task)
        return task

    def add_material_hypothesis(
        self, record: ScientificHypothesis
    ) -> ScientificHypothesis:
        existing = next(
            (
                item
                for item in self.material_hypotheses
                if item.proposal.request_id == record.proposal.request_id
            ),
            None,
        )
        if existing is None:
            self.material_hypotheses.append(record)
            return record
        if existing.request_payload_sha256 != record.request_payload_sha256:
            raise ValueError(
                f"request_id {record.proposal.request_id!r} conflicts with an existing payload"
            )
        return existing
