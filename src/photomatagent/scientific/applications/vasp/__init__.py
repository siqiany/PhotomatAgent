"""VASP application adapter (namespace ``vasp``).

Sprint 3 Phase D: the donor's ``VaspCloudAdapter`` was split into
``SCNetBackend`` (generic HPC, ``scientific/remote``) plus this
``VaspApplication`` (VASP-specific knowledge). Input generation is migrated
from the donor ``VaspInputGenerator``; result validation keeps the donor's
contracts (vasprun.xml exists / valid XML / converged electronic / ionic
for relax). POTCAR files are never generated or committed by this package:
they are resolved from ``PMG_VASP_PSP_DIR`` (local) or a remote
pseudopotential location at submit time.
"""

from __future__ import annotations

from photomatagent.scientific.applications.vasp.application import VaspApplication
from photomatagent.scientific.applications.vasp.profiles import (
    NAMD_PREPARATION,
    NARROW_GAP_SOC,
    OPTICS,
    STANDARD_SEMICONDUCTOR,
    VaspProfile,
    profiles,
)

__all__ = [
    "NAMD_PREPARATION",
    "NARROW_GAP_SOC",
    "OPTICS",
    "STANDARD_SEMICONDUCTOR",
    "VaspApplication",
    "VaspProfile",
    "profiles",
]
