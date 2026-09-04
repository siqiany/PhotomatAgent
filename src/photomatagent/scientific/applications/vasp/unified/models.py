"""Typed requests, manifests, states, and decisions for unified VASP workflows."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

from photomatagent.scientific.applications.vasp.molecular.models import WorkflowSpec
from photomatagent.scientific.applications.vasp.study.models import VaspStudyRequest
from photomatagent.scientific.remote.models import ResourceRequest


class VaspWorkflowKind(str, Enum):
    PERIODIC = "periodic"
    MOLECULAR = "molecular"
    STUDY = "study"


class PeriodicScientificSpec(BaseModel):
    kind: Literal["periodic"] = "periodic"
    structure_path: str
    profile: str
    scientific_overrides: dict[str, Any] = Field(default_factory=dict)
    potcar_policy: str = "configured"


class MolecularScientificSpec(BaseModel):
    kind: Literal["molecular"] = "molecular"
    workflow: WorkflowSpec


class StudyScientificSpec(BaseModel):
    kind: Literal["study"] = "study"
    request: VaspStudyRequest


ScientificSpec = Annotated[
    PeriodicScientificSpec | MolecularScientificSpec | StudyScientificSpec,
    Field(discriminator="kind"),
]


class WorkflowState(str, Enum):
    PLANNED = "PLANNED"
    PREPARED = "PREPARED"
    PREFLIGHTED = "PREFLIGHTED"
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    RECONCILING = "RECONCILING"
    AWAITING_RESOURCE_CONFIRMATION = "AWAITING_RESOURCE_CONFIRMATION"
    AWAITING_SCIENTIFIC_CONFIRMATION = "AWAITING_SCIENTIFIC_CONFIRMATION"
    SCHEDULER_COMPLETED = "SCHEDULER_COMPLETED"
    VALIDATED = "VALIDATED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    FAILED = "FAILED"


class ScientificChange(BaseModel):
    parameter: str
    old_value: Any
    new_value: Any
    reason: str


class UnifiedStage(BaseModel):
    name: str
    depends_on: list[str] = Field(default_factory=list)
    state: WorkflowState = WorkflowState.PLANNED
    resource_recommendation: ResourceRequest | None = None
    request_id: str | None = None
    execution_fingerprint: str | None = None
    decision_epoch: int = 0
    # Execution-only restart provenance; never included in scientific intent.
    attempt_inputs: dict[str, Any] = Field(default_factory=dict)


class WorkflowEvent(BaseModel):
    event_type: str
    timestamp: datetime
    stage: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ReportKind(str, Enum):
    SUMMARY = "summary"
    ORBITALS = "orbitals"
    ESP = "esp"
    BINDING_ENERGY = "binding_energy"
    STUDY = "study"


class ReportRequest(BaseModel):
    kind: ReportKind = ReportKind.SUMMARY
    related_workflow_ids: list[str] = Field(default_factory=list)


class UnifiedVaspRequest(BaseModel):
    workflow_kind: VaspWorkflowKind
    scientific_spec: ScientificSpec

    @model_validator(mode="after")
    def _kind_matches_spec(self) -> UnifiedVaspRequest:
        if self.scientific_spec.kind != self.workflow_kind.value:
            raise ValueError(
                f"workflow_kind {self.workflow_kind.value!r} does not match "
                f"scientific_spec.kind {self.scientific_spec.kind!r}"
            )
        return self


class UnifiedVaspManifest(BaseModel):
    schema_version: Literal["2.0"] = "2.0"
    workflow_id: str
    workflow_kind: VaspWorkflowKind
    revision: int = 0
    state: WorkflowState = WorkflowState.PLANNED
    scientific_spec: ScientificSpec
    scientific_fingerprint: str
    execution_fingerprint: str | None = None
    decision_epoch: int = 0
    stages: list[UnifiedStage] = Field(default_factory=list)
    events: list[WorkflowEvent] = Field(default_factory=list)
    # Execution metadata only.  Child IDs are assigned by the study adapter
    # after planning and are deliberately absent from scientific fingerprints.
    child_workflow_ids: dict[str, str] = Field(default_factory=dict)
    resource_budget_consumed_core_hours: float = 0.0
    resource_budget_submitted_jobs: int = 0
    # Typed study execution progress.  This lives in the authoritative parent
    # manifest; study_state.json is only a human-readable derived copy.
    study_task_states: dict[str, dict[str, Any]] = Field(default_factory=dict)
    study_stage_states: dict[str, dict[str, str]] = Field(default_factory=dict)
