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
from photomatagent.scientific.remote.lifecycle import (
    ReconciliationResult,
    StatusRefresh,
    SubmissionGate,
    SubmitOnceResult,
    SubmitOnceSession,
)
from photomatagent.scientific.remote.monitor import JobMonitor, MonitoringHandle
from photomatagent.scientific.remote.registry import (
    JobLifecycleState,
    JobRecord,
    JobRegistry,
    canonical_input_hash,
    derive_request_id,
    lifecycle_from_hpc,
)
from photomatagent.scientific.remote.scheduler import (
    parse_sbatch_job_id,
    render_slurm_script,
    remote_jobs_by_name_command,
    slurm_state_to_hpc_state,
)
from photomatagent.scientific.remote.scnet import SCNetBackend

__all__ = [
    "HPCJobState",
    "JobLifecycleState",
    "JobRecord",
    "JobRegistry",
    "JobMonitor",
    "MonitoringHandle",
    "ReconciliationResult",
    "RemoteArtifactRef",
    "RemoteExecutionResult",
    "RemoteJobRef",
    "RemoteJobSpec",
    "RemoteServerConfig",
    "ResourcePolicy",
    "ResourceRequest",
    "SCNetBackend",
    "StatusRefresh",
    "ScientificArtifactRef",
    "SubmissionGate",
    "SubmitOnceResult",
    "SubmitOnceSession",
    "canonical_input_hash",
    "derive_request_id",
    "lifecycle_from_hpc",
    "parse_sbatch_job_id",
    "render_slurm_script",
    "remote_jobs_by_name_command",
    "slurm_state_to_hpc_state",
]
