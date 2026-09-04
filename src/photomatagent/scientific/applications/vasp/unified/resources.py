"""Resource recommendation and authorization for unified VASP workflows."""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, cast

from pydantic import BaseModel, Field

from photomatagent.scientific.applications.vasp.unified.approvals import (
    ApprovalKind,
    ApprovalReceiptStore,
    PendingDecision,
    pending_decision,
)
from photomatagent.scientific.applications.vasp.unified.models import (
    MolecularScientificSpec,
    PeriodicScientificSpec,
    ScientificChange,
    UnifiedStage,
    UnifiedVaspManifest,
)
from photomatagent.scientific.remote.models import ResourcePolicy, ResourceRequest


class ResourceDecisionState(str, Enum):
    ALLOWED = "ALLOWED"
    NEEDS_CONFIRMATION = "NEEDS_CONFIRMATION"
    DENIED = "DENIED"


class ResourceDecision(BaseModel):
    state: ResourceDecisionState
    recommended: ResourceRequest
    effective: ResourceRequest | None
    decision_hash: str
    reasons: list[str] = Field(default_factory=list)
    pending_decision: PendingDecision | None = None


class AutomaticBudget(BaseModel):
    """Stricter, automatic-approval resource threshold.

    This is intentionally separate from ``ResourcePolicy``. Hard caps remain
    enforced by ``ResourcePolicy.violations()``; anything above this budget
    needs an application-level resource confirmation.
    """

    max_nodes: int = 1
    max_tasks_per_node: int = 32
    max_walltime_minutes: int = 120
    max_memory_gb: float | None = None

    @classmethod
    def from_environment(cls) -> "AutomaticBudget":
        return cls(
            max_nodes=_int_env("PHOTOMATAGENT_VASP_AUTO_MAX_NODES", 1),
            max_tasks_per_node=_int_env(
                "PHOTOMATAGENT_VASP_AUTO_MAX_TASKS_PER_NODE", 32
            ),
            max_walltime_minutes=_int_env(
                "PHOTOMATAGENT_VASP_AUTO_MAX_WALLTIME_MINUTES", 120
            ),
            max_memory_gb=_float_env("PHOTOMATAGENT_VASP_AUTO_MAX_MEMORY_GB", None),
        )

    def violations(self, request: ResourceRequest) -> list[str]:
        problems: list[str] = []
        if request.nodes > self.max_nodes:
            problems.append(
                f"nodes={request.nodes} exceeds automatic budget {self.max_nodes}"
            )
        if request.tasks_per_node > self.max_tasks_per_node:
            problems.append(
                f"tasks_per_node={request.tasks_per_node} exceeds automatic "
                f"budget {self.max_tasks_per_node}"
            )
        if request.walltime_minutes > self.max_walltime_minutes:
            problems.append(
                f"walltime {request.walltime_minutes} min exceeds automatic "
                f"budget {self.max_walltime_minutes}"
            )
        if (
            self.max_memory_gb is not None
            and request.memory_gb is not None
            and request.memory_gb > self.max_memory_gb
        ):
            problems.append(
                f"memory {request.memory_gb} GB exceeds automatic budget "
                f"{self.max_memory_gb} GB"
            )
        return problems


class VaspResourcePlanner:
    """Recommend a resource request from the manifest and stage."""

    def recommend(
        self, manifest: UnifiedVaspManifest, stage: UnifiedStage
    ) -> ResourceRequest:
        if stage.resource_recommendation is not None:
            return stage.resource_recommendation
        kind = manifest.workflow_kind.value
        spec = manifest.scientific_spec
        if kind == "periodic":
            from photomatagent.scientific.applications.vasp.profiles import get_profile

            return get_profile(cast(PeriodicScientificSpec, spec).profile).default_resource
        if kind == "molecular":
            workflow = cast(MolecularScientificSpec, spec).workflow
            return ResourceRequest(
                partition=workflow.resource_ceiling.partition,
                nodes=workflow.resource_ceiling.nodes,
                tasks_per_node=workflow.resource_plan.tasks_per_node,
                walltime_minutes=workflow.resource_plan.walltime_minutes,
            )
        if kind == "study":
            return ResourceRequest(
                partition="kshcnormal",
                nodes=1,
                tasks_per_node=8,
                walltime_minutes=120,
            )
        return ResourceRequest()


class ResourceAuthorizationService:
    """Single orchestration point for VASP resource authorization."""

    def __init__(
        self,
        approval_store: ApprovalReceiptStore | None = None,
        *,
        policy: ResourcePolicy | None = None,
        automatic_budget: AutomaticBudget | None = None,
    ) -> None:
        self.approval_store = approval_store
        self.policy = policy
        self.automatic_budget = automatic_budget

    def decide(
        self,
        manifest: UnifiedVaspManifest,
        stage: UnifiedStage,
        request: ResourceRequest,
    ) -> ResourceDecision:
        policy = self.policy or ResourcePolicy.from_environment()
        budget = self.automatic_budget or AutomaticBudget.from_environment()
        reasons: list[str] = []

        policy_violations = policy.violations(request)
        if policy_violations:
            reasons.extend(policy_violations)

        calibration_problems = self._calibration_problems(manifest)
        if calibration_problems:
            reasons.extend(calibration_problems)

        auto_violations = budget.violations(request)
        if not manifest.execution_fingerprint:
            return ResourceDecision(
                state=ResourceDecisionState.DENIED,
                recommended=request,
                effective=None,
                decision_hash="",
                reasons=["authoritative execution fingerprint is required before resource authorization"],
            )
        pending = self._pending_decision(
            manifest, stage, request, reasons or auto_violations
        )
        decision_hash = pending.decision_hash
        if reasons:
            # Hard policy/calibration failures cannot be overridden by a receipt.
            if self.approval_store is not None:
                self.approval_store.record_pending(pending)
            return ResourceDecision(
                state=ResourceDecisionState.DENIED,
                recommended=request,
                effective=None,
                decision_hash=decision_hash,
                reasons=reasons,
                pending_decision=pending,
            )

        if not auto_violations:
            return ResourceDecision(
                state=ResourceDecisionState.ALLOWED,
                recommended=request,
                effective=request,
                decision_hash=decision_hash,
                reasons=[],
            )

        if (
            self.approval_store is not None
            and self.approval_store.valid_receipt(pending, manifest) is not None
        ):
            return ResourceDecision(
                state=ResourceDecisionState.ALLOWED,
                recommended=request,
                effective=request,
                decision_hash=decision_hash,
                reasons=["resource confirmation receipt accepted"],
            )
        if self.approval_store is not None:
            self.approval_store.record_pending(pending)
        return ResourceDecision(
            state=ResourceDecisionState.NEEDS_CONFIRMATION,
            recommended=request,
            effective=None,
            decision_hash=decision_hash,
            reasons=auto_violations,
            pending_decision=pending,
        )

    @staticmethod
    def _calibration_problems(manifest: UnifiedVaspManifest) -> list[str]:
        if manifest.workflow_kind.value != "molecular":
            return []
        workflow = cast(MolecularScientificSpec, manifest.scientific_spec).workflow
        return workflow.resource_plan.violations(len(workflow.stages))

    @staticmethod
    def _pending_decision(
        manifest: UnifiedVaspManifest,
        stage: UnifiedStage,
        request: ResourceRequest,
        reasons: list[str],
    ) -> PendingDecision:
        return pending_decision(
            manifest=manifest,
            kind=ApprovalKind.RESOURCE,
            summary=f"Resource confirmation for stage {stage.name}",
            stage=stage.name,
            resource_proposal=request.model_dump(mode="json"),
            changes=[ScientificChange(
                parameter="resource",
                old_value=None,
                new_value=request.model_dump(mode="json"),
                reason="; ".join(reasons),
            )],
        )


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip()) if os.environ.get(name, "").strip() else default
    except ValueError:
        return default


def _float_env(name: str, default: float | None) -> float | None:
    try:
        value = os.environ.get(name, "").strip()
        return float(value) if value else default
    except ValueError:
        return default
