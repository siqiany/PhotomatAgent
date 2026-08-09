"""ScientificCalculationBackend interface.

This is the integration boundary for real compute (VASP on Slurm, HPC, ...).
The runtime and tools only depend on this interface; nothing assumes a
calculation returns instantly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from photomatagent.scientific.tasks import ScientificTask


class ScientificCalculationBackend(ABC):
    """Future surface: prepare / validate / submit / status / parse / cancel / restart."""

    name: str = "abstract"

    def prepare(self, request: dict[str, Any]) -> dict[str, Any]:
        """Stage inputs (write INCAR/KPOINTS/POSCAR, upload files...)."""
        raise NotImplementedError

    def validate(self, request: dict[str, Any]) -> list[str]:
        """Return a list of validation problems (empty means valid)."""
        raise NotImplementedError

    @abstractmethod
    async def submit(self, request: dict[str, Any]) -> ScientificTask:
        """Submit a calculation and return a task handle immediately."""

    @abstractmethod
    async def status(self, task_id: str) -> ScientificTask:
        """Refresh a task's status (poll for RUNNING/COMPLETED/FAILED)."""

    def parse(self, output_reference: str) -> dict[str, Any]:
        """Parse raw output files into structured results."""
        raise NotImplementedError

    async def cancel(self, task_id: str) -> ScientificTask:
        raise NotImplementedError

    async def restart(self, task_id: str) -> ScientificTask:
        raise NotImplementedError
