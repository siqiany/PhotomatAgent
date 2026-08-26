"""Typed study models: request, spec, matrix, tasks and persisted state."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


class PropertyRequest(str, Enum):
    """Scientific properties a study can be asked for."""

    HOMO_LUMO = "homo_lumo"
    BINDING_ENERGY = "binding_energy"
    ESP = "esp"


class StudyTaskState(str, Enum):
    """Persisted per-calculation state in study_state.json."""

    PLANNED = "PLANNED"
    STRUCTURE_PENDING = "STRUCTURE_PENDING"
    PREPARED = "PREPARED"
    PREFLIGHT_FAILED = "PREFLIGHT_FAILED"
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    COLLECTED = "COLLECTED"
    VALIDATED = "VALIDATED"
    FAILED = "FAILED"
    CONFORMER_RETRY = "CONFORMER_RETRY"
    SKIPPED_PROXY = "SKIPPED_PROXY"
    SKIPPED_BUDGET = "SKIPPED_BUDGET"
    BLOCKED_NO_AUTHORIZATION = "BLOCKED_NO_AUTHORIZATION"
    UNKNOWN = "UNKNOWN"


class StudySystem(BaseModel):
    """One chemical entity in the study request (typed, no guessing)."""

    system_id: str
    display_name: str = ""
    aliases: list[str] = Field(default_factory=list)
    smiles: str | None = None
    structure_path: Path | None = None
    total_charge: int | None = None
    spin_multiplicity: int = 1
    role: str = "molecule"
    properties: list[PropertyRequest] = Field(default_factory=list)


class MethodSpec(BaseModel):
    """Calculation-method preferences (all optional, defaults recorded)."""

    functional: str = "PBE-D3(BJ)"
    encut_ev: float | None = None  # None -> 400 (smoke) / 520 (production)
    box_ang: float = 20.0  # smoke baseline; production 30 A needs calibration
    potcar_set: str = "PAW-PBE"
    spin: int | None = None  # None -> 1 unless implied by electron parity
    resource_profile: str = "smoke"  # ResourceProfile value (smoke|production)
    calibration: dict[str, Any] | None = None  # CalibrationRecord (audited)

    def profile(self) -> Any:
        from photomatagent.scientific.applications.vasp.molecular.models import (
            ResourceProfile,
        )

        return ResourceProfile(self.resource_profile)

    def calibration_record(self) -> Any | None:
        if not self.calibration:
            return None
        from photomatagent.scientific.applications.vasp.molecular.calibration import (
            CalibrationRecord,
        )

        return CalibrationRecord.model_validate(self.calibration)


class StructurePolicy(BaseModel):
    """How missing structures are handled (never silently)."""

    allow_assumed_structures: bool = True
    max_candidates_per_system: int = 3
    seed: int = 20260825


class ExecutionPolicy(BaseModel):
    """Execution gating: authorization, budget, retries and parallelism."""

    user_requested_computation: bool = False
    wait: bool = True
    stop_on_failure: bool = False  # studies isolate failures per system
    max_conformer_retries: int = 2
    wait_timeout_seconds: float = 3600.0
    screen_conformers: bool = True  # B3 funnel: cheap E0 screens rank candidates
    max_screen_candidates: int = 6


class ResourceBudget(BaseModel):
    """Hard study-level cost ceiling."""

    max_core_hours: float = 32.0
    max_jobs: int | None = None
    notes: str = ""


class ReportOptions(BaseModel):
    """Final-report preferences."""

    report_language: str = "zh"
    include_figures: bool = True


class VaspStudyRequest(BaseModel):
    """The typed study request (outer agent maps natural language to this)."""

    study_id: str = ""
    original_request: str = ""
    systems: list[StudySystem] = Field(default_factory=list)
    property_requests: list[PropertyRequest] = Field(default_factory=list)
    comparisons: list[dict[str, Any]] = Field(default_factory=list)
    method: MethodSpec = Field(default_factory=MethodSpec)
    structure_policy: StructurePolicy = Field(default_factory=StructurePolicy)
    execution_policy: ExecutionPolicy = Field(default_factory=ExecutionPolicy)
    resource_budget: ResourceBudget = Field(default_factory=ResourceBudget)
    report_options: ReportOptions = Field(default_factory=ReportOptions)


class CalculationTask(BaseModel):
    """One unique chemical calculation in the matrix."""

    task_id: str  # canonical_key, dedup anchor
    system_id: str
    display_name: str
    role: str
    formula: str = ""
    total_charge: int
    spin_multiplicity: int = 1
    structure_path: Path | None = None
    structure_candidates: list[str] = Field(default_factory=list)
    reliability: str = "B"
    structure_status: str = ""
    assists: list[str] = Field(
        default_factory=list
    )  # properties this workflow serves
    depends_on: list[str] = Field(default_factory=list)
    estimated_core_hours: float = 0.0
    workflow_dir: str = ""
    request_id: str = ""
    state: str = StudyTaskState.PLANNED.value
    conformer_index: int = 0
    error: str = ""
    results_dir: str = ""

    def canonical_key(self) -> str:
        return self.task_id


class BindingGroup(BaseModel):
    """One E_binding = E_complex - sum(E_fragments) calculation."""

    complex_task_id: str
    fragment_task_ids: list[str]
    label: str
    total_charge: int
    state: str = StudyTaskState.PLANNED.value
    delta_e_ev: float | None = None
    delta_delta_e_ev: float | None = None
    error: str = ""
    uses_declared_reference_assumption: bool = False
    high_risk_absolute_binding_energy: bool = False


class CalculationMatrix(BaseModel):
    """Deduplicated matrix + dependency DAG + resource estimates."""

    tasks: list[CalculationTask] = Field(default_factory=list)
    binding_groups: list[BindingGroup] = Field(default_factory=list)
    total_core_hours: float = 0.0
    total_jobs: int = 0
    estimated_disk_gb: float = 0.0
    notes: list[str] = Field(default_factory=list)

    def task_map(self) -> dict[str, CalculationTask]:
        return {task.task_id: task for task in self.tasks}


class VaspStudySpec(BaseModel):
    """Full plan: request + resolved structures + matrix."""

    study_id: str
    request: VaspStudyRequest
    study_dir: Path
    calculation_matrix: CalculationMatrix = Field(
        default_factory=CalculationMatrix
    )
    structure_manifest_path: Path = Path("")

    @model_validator(mode="after")
    def _validate_matrix(self) -> VaspStudySpec:
        ids = [task.task_id for task in self.calculation_matrix.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("calculation matrix contains duplicate task ids")
        return self
