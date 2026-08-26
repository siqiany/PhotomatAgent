"""VASP study orchestration on top of the vasp_molecule.* executor.

The study layer translates a typed study request (assembled by the outer
agent from natural language, never by an embedded LLM) into a deduplicated
calculation matrix, generates/resolves structures through the generic
``chemistry`` package, and executes the matrix exclusively through the
existing molecular executor (prepare/preflight/submit-once/collect/resume).
No submission, POTCAR, Slurm or monitoring logic is re-implemented here.
"""

from __future__ import annotations

from photomatagent.scientific.applications.vasp.study.models import (
    CalculationMatrix,
    CalculationTask,
    PropertyRequest,
    StudyTaskState,
    VaspStudyRequest,
    VaspStudySpec,
)

__all__ = [
    "CalculationMatrix",
    "CalculationTask",
    "PropertyRequest",
    "StudyTaskState",
    "VaspStudyRequest",
    "VaspStudySpec",
]
