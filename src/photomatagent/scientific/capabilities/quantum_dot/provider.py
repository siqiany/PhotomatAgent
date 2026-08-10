"""High-fidelity QD solver boundary (kdotpy / nextnano / tight-binding / DFT).

This module defines the *contract* only. Sprint 2 deliberately does not
implement a full kdotpy wrapper: the probe reports whether an external
solver is reachable, and the contract documents exactly what a future
adapter must return. No capability is claimed that is not actually wired up.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict

from photomatagent.scientific.capabilities.base import (
    CapabilityStatus,
    ProbeResult,
)

FidelityLabel = Literal["kp", "tight_binding", "dft", "atomistic"]


class QDCalculationRequest(TypedDict, total=False):
    """What a high-fidelity QD solver must accept (future adapters)."""

    material: str
    composition: float
    radius_nm: float
    temperature_k: float
    model: str  # kp8x8 / tb / ...


class QDCalculationResult(TypedDict, total=False):
    """What a high-fidelity QD solver must return."""

    state_energies_eV: list[float]
    transition_energy_eV: float
    transition_wavelength_um: float
    method: str
    fidelity: FidelityLabel
    assumptions: list[str]
    limitations: list[str]
    provenance: dict[str, Any]


class QDHighFidelityProvider(Protocol):
    """Adapter contract for nextnano / kdotpy / TB / DFT-based solvers."""

    name: str
    fidelity: FidelityLabel

    def compute_states(self, request: QDCalculationRequest) -> QDCalculationResult:
        """Compute confined electronic states for a QD request."""


def probe_kdotpy(workspace_root: Path | None = None) -> ProbeResult:
    """Probe kdotpy availability in the main env or a dedicated venv.

    Search order: ``PHOTOMATAGENT_KDOTPY_PYTHON`` env var, workspace
    ``.venvs/kdotpy/bin/python``, then the main interpreter.
    """
    candidates: list[str] = []
    env_python = os.environ.get("PHOTOMATAGENT_KDOTPY_PYTHON", "")
    if env_python:
        candidates.append(env_python)
    if workspace_root is not None:
        candidates.append(str(Path(workspace_root) / ".venvs" / "kdotpy" / "bin" / "python"))
    for python in candidates:
        if not Path(python).is_file():
            continue
        found = _kdotpy_version_for(python)
        if found:
            return ProbeResult(
                status=CapabilityStatus.AVAILABLE,
                detail=f"kdotpy found via isolated venv interpreter {python}",
                version=found,
            )
    if importlib.util.find_spec("kdotpy") is not None:
        try:
            import kdotpy  # noqa: F401

            return ProbeResult(
                status=CapabilityStatus.AVAILABLE,
                detail="kdotpy importable in the main environment",
                version=getattr(kdotpy, "__version__", "unknown"),
            )
        except Exception as exc:
            return ProbeResult(
                status=CapabilityStatus.MISSING_DEPENDENCY,
                detail=f"kdotpy present but import fails: {type(exc).__name__}: {exc}",
            )
    return ProbeResult(
        status=CapabilityStatus.MISSING_DEPENDENCY,
        detail=(
            "kdotpy not importable; install into an isolated venv and set "
            "PHOTOMATAGENT_KDOTPY_PYTHON or create .venvs/kdotpy"
        ),
    )


def _kdotpy_version_for(python: str) -> str:
    try:
        result = subprocess.run(
            [python, "-c", "import kdotpy, importlib.metadata; "
             "print(importlib.metadata.version('kdotpy'))"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def kdotpy_capability_note() -> str:
    """Honest scope statement for the kdotpy namespace (no invented features)."""
    return (
        "kdotpy (CMU) solves k.p band structures for bulk and layered "
        "(1D) heterostructures. It is NOT a 3D quantum-dot confinement "
        "solver: colloidal-QD diameter-to-level calculations are outside "
        "its supported geometries. Supported: bulk band structure along "
        "high-symmetry paths; layered structures with user-supplied "
        "effective masses and band offsets. Unsupported: 0D confinement, "
        "full QD eigenstates, absorption spectra of colloidal dots."
    )
