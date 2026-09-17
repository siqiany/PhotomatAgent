"""ScientificState: the agent's structured model of the science."""

from __future__ import annotations

import hashlib
import hmac
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from photomatagent.scientific.calculations import CalculationRecord
from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.claims import ScientificClaim
from photomatagent.scientific.evidence import Evidence
from photomatagent.scientific.discovery.models import ScientificHypothesis
from photomatagent.scientific.tasks import ScientificTask

DEFAULT_TRUSTED_EVIDENCE_TOOLS = frozenset(
    {
        "electronic.band_summary",
        "electronic.dos_summary",
        "electronic.effective_mass",
        "materials.get_summary",
        "materials.get_structure",
        "materials.search",
        "structure.density",
        "structure.neighbors",
        "structure.summary",
        "structure.symmetry",
        "vasp.inspect_result",
    }
)


class EvidenceAttestation(BaseModel):
    """Host-owned authority record, stored separately from producer evidence."""

    evidence_id: str
    authority: Literal["observation", "synthetic", "background"] = "background"
    origin: Literal["trusted_builtin", "synthetic_test", "untrusted_tool"]
    tool_name: str
    tool_call_id: str
    host_proof: str = Field(default="", repr=False)

    @classmethod
    def host_create(
        cls,
        *,
        evidence_id: str,
        authority: Literal["observation", "synthetic", "background"],
        origin: Literal["trusted_builtin", "synthetic_test", "untrusted_tool"],
        tool_name: str,
        tool_call_id: str,
    ) -> EvidenceAttestation:
        values = (evidence_id, authority, origin, tool_name, tool_call_id)
        return cls(
            evidence_id=evidence_id,
            authority=authority,
            origin=origin,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            host_proof=_attestation_proof(values),
        )

    def is_host_valid(self) -> bool:
        expected = _attestation_proof(
            (
                self.evidence_id,
                self.authority,
                self.origin,
                self.tool_name,
                self.tool_call_id,
            )
        )
        return hmac.compare_digest(self.host_proof, expected)

    def has_compatible_authority(self) -> bool:
        return self.authority == {
            "trusted_builtin": "observation",
            "synthetic_test": "synthetic",
            "untrusted_tool": "background",
        }[self.origin]


def _attestation_proof(values: tuple[str, ...]) -> str:
    payload = "\x1f".join(values).encode("utf-8")
    return hmac.new(
        b"photomatagent-host-attestation-v1",
        payload,
        hashlib.sha256,
    ).hexdigest()


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

    @model_validator(mode="after")
    def validate_evidence_attestations(self) -> ScientificState:
        evidence_counts: dict[str, int] = {}
        for item in self.evidence:
            evidence_counts[item.id] = evidence_counts.get(item.id, 0) + 1
        self.evidence_attestations = {
            key: attestation
            for key, attestation in self.evidence_attestations.items()
            if key == attestation.evidence_id
            and evidence_counts.get(key) == 1
            and attestation.has_compatible_authority()
            and attestation.is_host_valid()
        }
        return self

    def add_evidence(
        self, evidence: Evidence | ScientificEvidence
    ) -> Evidence | ScientificEvidence:
        self.evidence.append(evidence)
        return evidence

    def attest_evidence(self, attestation: EvidenceAttestation) -> EvidenceAttestation:
        matches = sum(item.id == attestation.evidence_id for item in self.evidence)
        if matches != 1:
            raise ValueError("attested evidence must exist exactly once in scientific state")
        if not attestation.has_compatible_authority() or not attestation.is_host_valid():
            raise ValueError("evidence attestation is not host-valid")
        self.evidence_attestations[attestation.evidence_id] = attestation
        return attestation

    def verified_attestation(self, evidence_id: str) -> EvidenceAttestation | None:
        if sum(item.id == evidence_id for item in self.evidence) != 1:
            return None
        attestation = self.evidence_attestations.get(evidence_id)
        if (
            attestation is None
            or attestation.evidence_id != evidence_id
            or not attestation.has_compatible_authority()
            or not attestation.is_host_valid()
        ):
            return None
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
