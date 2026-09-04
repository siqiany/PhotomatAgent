"""Internal typed operation results and executor Protocol for VASP adapters."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from photomatagent.scientific.applications.vasp.unified.approvals import PendingDecision
from photomatagent.scientific.applications.vasp.unified.models import (
    ReportKind,
    ReportRequest,
    UnifiedStage,
    UnifiedVaspManifest,
    WorkflowState,
)
from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.remote.models import ResourceRequest


class OperationResult(BaseModel):
    ok: bool
    data: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)


class PreflightResult(OperationResult):
    passed: bool


class SubmissionResult(OperationResult):
    request_id: str
    job_id: str | None = None
    submitted: bool = False
    duplicate: bool = False
    needs_reconciliation: bool = False
    pending_decision: PendingDecision | None = None


class StatusResult(OperationResult):
    stage_states: dict[str, str] = Field(default_factory=dict)
    query_failed: bool = False


class RestartStructuralValidation(BaseModel):
    """Executor-owned, deterministic validation of a restart structure."""

    model_config = ConfigDict(extra="forbid")

    atom_count: int = Field(gt=0)
    validator: str = Field(min_length=1, max_length=128)


class RecoveryResult(OperationResult):
    action: str
    pending_decision: PendingDecision | None = None
    # Per-stage lifecycle state (value of JobLifecycleState) after reconcile,
    # keyed by stage name. Lets the service distinguish a confirmed terminal
    # failure (eligible for a fresh attempt) from ambiguity/active jobs.
    stage_states: dict[str, str] = Field(default_factory=dict)
    contcar_restart: bool = False
    scientific_fingerprint: str | None = None
    # The executor may identify an artifact, but only the service resolves and
    # hashes it after enforcing the workspace containment boundary.
    restart_artifact_path: str | None = None
    restart_artifact_sha256: str | None = None
    restart_structural_validation: RestartStructuralValidation | None = None


class CollectionResult(OperationResult):
    validated: bool = False
    evidence: list[ScientificEvidence] = Field(default_factory=list)
    stage_states: dict[str, str] = Field(default_factory=dict)


class ReportResult(OperationResult):
    report_kind: ReportKind


class ServiceResult(OperationResult):
    workflow_id: str
    state: WorkflowState
    evidence: list[ScientificEvidence] = Field(default_factory=list)
    pending_decision: PendingDecision | None = None


@runtime_checkable
class VaspWorkflowExecutor(Protocol):
    """Internal executor contract. Never accepts Tool instances or raw schemas."""

    async def prepare(self, manifest: UnifiedVaspManifest) -> OperationResult: ...
    async def preflight(self, manifest: UnifiedVaspManifest) -> PreflightResult: ...
    async def submit(
        self,
        manifest: UnifiedVaspManifest,
        stage: UnifiedStage,
        resource: ResourceRequest,
    ) -> SubmissionResult: ...
    async def status(self, manifest: UnifiedVaspManifest) -> StatusResult: ...
    async def reconcile(self, manifest: UnifiedVaspManifest) -> RecoveryResult: ...
    async def collect(self, manifest: UnifiedVaspManifest) -> CollectionResult: ...
    async def report(
        self, manifest: UnifiedVaspManifest, request: ReportRequest
    ) -> ReportResult: ...
