"""Isolated-molecule VASP offline capability (data models + preflight).

This subpackage is deliberately isolated from the periodic-profile modules
(``profiles.py``/``inputs.py``/``tools.py``) so that molecule-specific
semantics never leak boolean fields into the generic profiles. Everything in
this subpackage is offline and deterministic: structure reading, input
generation, POTCAR metadata bookkeeping and the submission preflight. No SSH,
no Slurm and no VASP execution are ever performed here.
"""

from __future__ import annotations

from photomatagent.scientific.applications.vasp.molecular.models import (
    CorrectionPolicy,
    MoleculeSpec,
    MonopoleMethod,
    PolymerKind,
    Polymerization,
    PreflightConfig,
    PreflightIssue,
    PreflightReport,
    PreflightSummary,
    ResourceCeiling,
    ResourceClass,
    StageName,
    StageSpec,
    StructureKind,
    WorkflowSpec,
)
from photomatagent.scientific.applications.vasp.molecular.preflight import (
    preflight_gate,
    render_agent_text,
    run_molecular_preflight,
    save_preflight_report,
)
from photomatagent.scientific.applications.vasp.molecular.generator import (
    MolecularVaspGenerator,
)
from photomatagent.scientific.applications.vasp.molecular.binding import (
    BindingEnergyInput,
    BindingReference,
    compute_binding_energy,
)
from photomatagent.scientific.applications.vasp.molecular.workflow import (
    TaskState,
    build_molecule_workflow,
    load_task_state,
    run_molecule_workflow,
    save_task_state,
)
from photomatagent.scientific.applications.vasp.molecular.tools import (
    MolecularVaspTools,
    bounded_payload,
)

__all__ = [
    "CorrectionPolicy",
    "BindingEnergyInput",
    "BindingReference",
    "MoleculeSpec",
    "MonopoleMethod",
    "MolecularVaspGenerator",
    "MolecularVaspTools",
    "PolymerKind",
    "Polymerization",
    "PreflightConfig",
    "PreflightIssue",
    "PreflightReport",
    "PreflightSummary",
    "TaskState",
    "bounded_payload",
    "build_molecule_workflow",
    "compute_binding_energy",
    "load_task_state",
    "preflight_gate",
    "ResourceCeiling",
    "ResourceClass",
    "StageName",
    "StageSpec",
    "StructureKind",
    "WorkflowSpec",
    "render_agent_text",
    "run_molecular_preflight",
    "run_molecule_workflow",
    "save_preflight_report",
    "save_task_state",
]
