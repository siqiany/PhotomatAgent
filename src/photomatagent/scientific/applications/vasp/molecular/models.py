"""Typed data models for the isolated-molecule VASP offline workflow.

Design rule: molecule semantics are never carried by loose stage-name strings
or arbitrary dicts. ``MoleculeSpec`` / ``StageSpec`` / ``WorkflowSpec`` are
the single source of truth for charge, spin, box, functional, pseudopotential
set, stage dependencies, produced/required outputs, resource ceiling and the
declared electrostatic-correction policy.

``total_charge`` is a REQUIRED field: it is never inferred from file names,
comments or chemical names (DME_Li, TFSI, ...). A caller that cannot state
the charge has nothing to submit.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class StructureKind(str, Enum):
    """Input structure formats accepted by the molecular readers."""

    XYZ = "xyz"
    SDF = "sdf"
    MOL = "mol"
    POSCAR = "poscar"


class StageName(str, Enum):
    """Known molecular workflow stages.

    The periodic profiles (band/dos/optics/md/snapshot) are out of scope for
    isolated-molecule calculations and are rejected by the preflight.
    """

    RELAX = "relax"
    STATIC_PRECONVERGE = "static_preconverge"
    CORRECTED_STATIC = "corrected_static"
    STATIC = "static"
    STATIC_HSE = "static_hse"
    ORBITAL_HOMO = "orbital_homo"
    ORBITAL_LUMO = "orbital_lumo"
    ORBITAL = "orbital"
    ESP = "esp"
    RESTART = "restart"


class ResourceProfile(str, Enum):
    """Molecule resource envelope profiles.

    ``smoke`` encodes the verified TFPMA baseline (8 cores, 20 A box, 400 eV,
    <= 20 min per stage); ``production`` targets 30 A / 520 eV and is refused
    for submission until a memory/resource calibration is recorded.
    """

    SMOKE = "smoke"
    PRODUCTION = "production"


class ResourceClass(str, Enum):
    """Coarse resource class attached to every stage."""

    SMOKE = "smoke"
    SMALL = "small"
    STANDARD = "standard"
    LARGE = "large"


class PolymerKind(str, Enum):
    """Whether the molecule is a user-defined polymer repeat-unit system."""

    NONE = "none"
    VM = "vm"
    TVM = "tvm"


class MonopoleMethod(str, Enum):
    """Declared monopole (charged-cell) correction method."""

    NONE = "none"
    LMONO = "lmono"


class Polymerization(BaseModel):
    """Explicit polymer definition required for VM/TVM submissions.

    VM/TVM are never invented by the model from feed ratios: connectivity,
    polymerization sites, repeat units and end caps must all be supplied by
    the user before any input generation is attempted.
    """

    connectivity: str = ""
    polymerization_sites: list[str] = Field(default_factory=list)
    repeat_units: list[str] = Field(default_factory=list)
    end_caps: list[str] = Field(default_factory=list)


class MoleculeSpec(BaseModel):
    """One isolated molecule (or ion/complex) placed in a fixed vacuum box."""

    name: str = Field(min_length=1)
    structure_path: Path | None = None
    structure_kind: StructureKind | None = None
    structure_source: str = "user-provided structure file"
    # Required on purpose: the charge is NEVER guessed from the molecule name.
    total_charge: int
    spin_multiplicity: int = 1
    box_ang: float = 30.0
    functional: str = "PBE-D3(BJ)"
    potcar_set: str = "PAW-PBE"
    calculation_purpose: str = "unspecified"
    model_id: str | None = None
    conformer_id: str | None = None
    polymer_kind: PolymerKind = PolymerKind.NONE
    polymerization: Polymerization | None = None
    blocked_reason: str | None = None

    @field_validator("spin_multiplicity")
    @classmethod
    def _multiplicity_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("spin_multiplicity must be >= 1")
        return value

    @field_validator("box_ang")
    @classmethod
    def _box_positive(cls, value: float) -> float:
        if value <= 5.0:
            raise ValueError("box_ang must be greater than 5 Angstrom")
        return value

    @field_validator("structure_kind", mode="before")
    @classmethod
    def _default_kind(cls, value: object, info: Any) -> object:
        if value is not None:
            return value
        path = info.data.get("structure_path")
        if isinstance(path, (str, Path)):
            path_obj = Path(path)
            if path_obj.name in {"POSCAR", "CONTCAR"}:
                return StructureKind.POSCAR
            suffix = path_obj.suffix.lower().lstrip(".")
            if suffix in {kind.value for kind in StructureKind}:
                return StructureKind(suffix)
        return value


class StageSpec(BaseModel):
    """One typed workflow stage with explicit dependency contracts."""

    name: StageName
    depends_on: StageName | None = None
    description: str = ""
    required_inputs: list[str] = Field(
        default_factory=lambda: ["POSCAR", "INCAR", "KPOINTS"]
    )
    required_upstream_outputs: list[str] = Field(default_factory=list)
    produced_outputs: list[str] = Field(default_factory=list)
    incar: dict[str, Any] = Field(default_factory=dict)
    resource_class: ResourceClass = ResourceClass.STANDARD
    validator: str = "default"


class CorrectionPolicy(BaseModel):
    """Explicit electrostatic-correction policy for the whole workflow.

    Rules enforced by the preflight:
    * a charged molecule (q != 0) MUST declare a monopole method (LMONO);
      NELECT alone does not remove periodic-image artifacts;
    * LMONO must NOT be combined with the dipole correction (LDIPOL/IDIPOL);
    * the declared policy must match the actual stage INCAR flags.
    """

    monopole_method: MonopoleMethod = MonopoleMethod.NONE
    dipole: bool = False


class ResourceCeiling(BaseModel):
    """Hard resource ceiling; NCORE/NPAR must divide node-grid tasks."""

    partition: str = "kshcnormal"
    nodes: int = 1
    tasks_per_node: int = 32
    walltime_minutes: int = 480


class SmokeBaseline(BaseModel):
    """Verified observations from the real TFPMA smoke run (not a budget).

    Source: gel_electrolyte_dft/codex_run/results/tfpma_smoke_corrected_static_clean/
    These numbers exist so resource planning starts from measured behaviour,
    not from hard-coded production assumptions.
    """

    two_core_oom_observed: bool = True
    four_core_timeout_minutes: float = 10.0
    eight_core_preconverge_seconds: float = 8.0 * 60 + 58
    eight_core_corrected_static_seconds: float = 1.0 * 60 + 42
    smoke_cores: int = 8
    smoke_box_ang: float = 20.0
    smoke_encut_ev: float = 400.0
    smoke_max_walltime_minutes: int = 20


class ResourcePlan(BaseModel):
    """Configurable, capped molecule resource plan.

    ``max_total_core_hours`` is the hard ceiling for the whole workflow
    (stages x cores x walltime). Production profiles must pass
    ``resource_calibrated`` (live memory/timings calibration) before any
    submission is allowed.
    """

    profile: ResourceProfile = ResourceProfile.SMOKE
    tasks_per_node: int = 8  # NOT 32 by default: the smoke baseline says 8
    walltime_minutes: int = Field(default=20, ge=1, le=720)
    max_total_core_hours: float = Field(default=32.0, gt=0)
    resource_calibrated: bool = False
    calibration_note: str = ""

    def core_hours(self, stages: int) -> float:
        return stages * self.tasks_per_node * self.walltime_minutes / 60.0

    def violations(self, stages: int) -> list[str]:
        problems: list[str] = []
        hours = self.core_hours(stages)
        if hours > self.max_total_core_hours + 1e-9:
            problems.append(
                f"workflow needs {hours:.2f} core-hours but the plan caps at "
                f"{self.max_total_core_hours:g}"
            )
        if self.profile is ResourceProfile.PRODUCTION and not self.resource_calibrated:
            problems.append(
                "production resource plan requires resource_calibrated=True "
                "(memory/ENMAX calibration from a live smoke run) before "
                "any submission"
            )
        return problems


class ScreenDecision(BaseModel):
    """Recorded PBE screening outcome that gates HSE06 stages.

    HSE06 is only executed for the lowest-energy conformer identified by the
    completed PBE screening; the tool refuses HSE stages when this record is
    absent or does not include the submitted molecule.
    """

    conformer_id: str
    pbe_e0_ev: float
    basis: str = "Gamma-only PBE-D3(BJ) fixed-box screened conformers"
    all_pbe_e0_ev: dict[str, float] = Field(default_factory=dict)


class PreflightConfig(BaseModel):
    """Tunable science thresholds used by the deterministic preflight."""

    encut_floor_ev: float = 520.0
    encut_max_enmax_ratio: float = 1.3
    min_vacuum_per_side_ang: float = 10.0
    min_interatomic_distance_ang: float = 0.5
    default_ispin: Literal[1, 2] = 1


class WorkflowSpec(BaseModel):
    """Full ordered molecular workflow: molecule + typed stages + policy."""

    molecule: MoleculeSpec
    stages: list[StageSpec] = Field(min_length=1)
    scientific_method: str
    correction_policy: CorrectionPolicy = Field(default_factory=CorrectionPolicy)
    resource_ceiling: ResourceCeiling = Field(default_factory=ResourceCeiling)
    resource_plan: ResourcePlan = Field(default_factory=ResourcePlan)
    smoke_baseline: SmokeBaseline = Field(default_factory=SmokeBaseline)
    screen_decision: ScreenDecision | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    preflight_config: PreflightConfig = Field(default_factory=PreflightConfig)
    # Set True only when submitting to a local run directory that intentionally
    # materializes the concatenated POTCAR (never committed, never logged).
    potcar_materialized: bool = False

    @model_validator(mode="after")
    def _dag_is_ordered_and_acyclic(self) -> WorkflowSpec:
        names = [stage.name for stage in self.stages]
        if len(names) != len({name for name in names}):
            raise ValueError("stage names must be unique: " + ", ".join(sorted(names)))
        positions = {name: index for index, name in enumerate(names)}
        roots = 0
        for stage in self.stages:
            if stage.depends_on is None:
                roots += 1
                continue
            if stage.depends_on not in positions:
                raise ValueError(
                    f"stage {stage.name.value} depends on unknown stage "
                    f"{stage.depends_on.value}"
                )
            if positions[stage.depends_on] >= positions[stage.name]:
                raise ValueError(
                    f"stage {stage.name.value} must depend only on an earlier "
                    f"stage (found {stage.depends_on.value})"
                )
        if roots == 0:
            raise ValueError("workflow has no root stage (depends_on=None)")
        return self


class PreflightIssue(BaseModel):
    """One deterministic preflight finding."""

    code: str
    message: str
    path: str = ""


class PreflightSummary(BaseModel):
    """Compact science summary returned to the agent."""

    formula: str
    charge: int
    nelect: float
    elements: list[str]
    neutral_valence_electrons: float
    box_ang: float
    potcar_set: str


class PreflightReport(BaseModel):
    """Full structured preflight result; persisted as preflight.json."""

    passed: bool
    errors: list[PreflightIssue] = Field(default_factory=list)
    warnings: list[PreflightIssue] = Field(default_factory=list)
    summary: PreflightSummary | None = None
    checks: list[str] = Field(default_factory=list)
