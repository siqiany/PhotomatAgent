"""ScientificState: the agent's structured model of the science."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

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

    model_config = ConfigDict(frozen=True)

    evidence_id: str
    authority: Literal["observation", "synthetic", "background"] = "background"
    origin: Literal["trusted_builtin", "synthetic_test", "untrusted_tool"]
    tool_name: str
    tool_call_id: str

    def has_compatible_authority(self) -> bool:
        return self.authority == {
            "trusted_builtin": "observation",
            "synthetic_test": "synthetic",
            "untrusted_tool": "background",
        }[self.origin]


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
    _runtime_authority_capability: object | None = PrivateAttr(default=None)
    _runtime_attestations: dict[str, EvidenceAttestation] = PrivateAttr(
        default_factory=dict
    )

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> ScientificState:
        """Copy durable state without inheriting runtime-only authority."""

        copied = super().model_copy(update=update, deep=deep)
        copied._runtime_authority_capability = None
        copied._runtime_attestations = {}
        return copied

    def __eq__(self, other: Any) -> bool:
        """Compare durable scientific content, excluding runtime capabilities."""

        if not isinstance(other, ScientificState):
            return False
        return self.model_dump(mode="python") == other.model_dump(mode="python")

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
        }
        return self

    def add_evidence(
        self, evidence: Evidence | ScientificEvidence
    ) -> Evidence | ScientificEvidence:
        self.evidence.append(evidence)
        return evidence

    def _bind_runtime_authority(
        self, capability: object, *, replace: bool = False
    ) -> None:
        """Bind host-internal authority; model/tool payloads must not call this."""
        if (
            self._runtime_authority_capability is not None
            and self._runtime_authority_capability is not capability
            and not replace
        ):
            raise ValueError("scientific state is bound to another runtime authority")
        self._runtime_authority_capability = capability
        if replace:
            self._runtime_attestations = {}

    def _attest_evidence(
        self, attestation: EvidenceAttestation, *, capability: object
    ) -> EvidenceAttestation:
        if self._runtime_authority_capability is not capability:
            raise ValueError("runtime evidence authority capability is required")
        matches = sum(item.id == attestation.evidence_id for item in self.evidence)
        if matches != 1:
            raise ValueError("attested evidence must exist exactly once in scientific state")
        if not attestation.has_compatible_authority():
            raise ValueError("evidence attestation authority is incompatible with origin")
        public_record = attestation.model_copy(deep=True)
        runtime_record = attestation.model_copy(deep=True)
        self.evidence_attestations[attestation.evidence_id] = public_record
        self._runtime_attestations[attestation.evidence_id] = runtime_record
        return public_record

    def _copy_runtime_attestations_from(
        self, source: ScientificState, *, capability: object
    ) -> None:
        if (
            self._runtime_authority_capability is not capability
            or source._runtime_authority_capability is not capability
        ):
            raise ValueError("runtime evidence authority capability is required")
        self._runtime_attestations = {
            evidence_id: attestation.model_copy(deep=True)
            for evidence_id, attestation in source._runtime_attestations.items()
        }

    def verified_attestation(self, evidence_id: str) -> EvidenceAttestation | None:
        if sum(item.id == evidence_id for item in self.evidence) != 1:
            return None
        attestation = self._runtime_attestations.get(evidence_id)
        if (
            attestation is None
            or attestation.evidence_id != evidence_id
            or not attestation.has_compatible_authority()
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
