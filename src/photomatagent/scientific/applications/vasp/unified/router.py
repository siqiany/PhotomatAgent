"""Deterministic workflow-kind routing for unified VASP executors."""

from __future__ import annotations

from photomatagent.scientific.applications.vasp.unified.executors import (
    VaspWorkflowExecutor,
)
from photomatagent.scientific.applications.vasp.unified.models import VaspWorkflowKind


class UnifiedVaspRouter:
    """Maps persisted workflow_kind to an internal executor.

    It never guesses from names, files, formulas, or later-stage data.
    """

    def __init__(
        self,
        *,
        periodic: VaspWorkflowExecutor | None = None,
        molecular: VaspWorkflowExecutor | None = None,
        study: VaspWorkflowExecutor | None = None,
    ) -> None:
        self._executors = {
            VaspWorkflowKind.PERIODIC: periodic,
            VaspWorkflowKind.MOLECULAR: molecular,
            VaspWorkflowKind.STUDY: study,
        }

    def executor_for(
        self, kind: VaspWorkflowKind
    ) -> VaspWorkflowExecutor:
        executor = self._executors.get(kind)
        if executor is None:
            raise RuntimeError(
                f"no executor is available for {kind.value!r}"
            )
        return executor

    def register(
        self, kind: VaspWorkflowKind, executor: VaspWorkflowExecutor
    ) -> None:
        self._executors[kind] = executor
