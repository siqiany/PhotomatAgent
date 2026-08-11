"""Remote scientific compute: SCNet HPC backend, scheduler, artifacts.

This package is infrastructure, not a tool surface. The agent never sees
``scnet.run_shell``; application adapters (VASP, Hefei-NAMD, MAGUS) expose
narrow tools on top of :class:`SCNetBackend`.
"""

from __future__ import annotations

from photomatagent.scientific.remote.models import (
    HPCJobState,
    RemoteArtifactRef,
    RemoteExecutionResult,
    RemoteJobRef,
    RemoteJobSpec,
    RemoteServerConfig,
    ResourcePolicy,
    ResourceRequest,
    ScientificArtifactRef,
)
from photomatagent.scientific.remote.scheduler import (
    parse_sbatch_job_id,
    render_slurm_script,
    slurm_state_to_hpc_state,
)
from photomatagent.scientific.remote.scnet import SCNetBackend

__all__ = [
    "HPCJobState",
    "RemoteArtifactRef",
    "RemoteExecutionResult",
    "RemoteJobRef",
    "RemoteJobSpec",
    "RemoteServerConfig",
    "ResourcePolicy",
    "ResourceRequest",
    "SCNetBackend",
    "ScientificArtifactRef",
    "parse_sbatch_job_id",
    "render_slurm_script",
    "slurm_state_to_hpc_state",
]
