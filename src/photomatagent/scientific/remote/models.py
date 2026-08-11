"""Typed schemas for remote HPC execution (SCNet).

No secrets live in these models. ``RemoteServerConfig`` keeps the private
key path only in its own field; every serialization path (``redacted``,
``public_dict``) drops it so the path never reaches the model context.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class HPCJobState(str, Enum):
    """Unified HPC job lifecycle state (Sprint 3 section 14)."""

    PREPARED = "PREPARED"
    SUBMITTED = "SUBMITTED"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    NODE_FAIL = "NODE_FAIL"
    UNKNOWN = "UNKNOWN"

    @property
    def terminal(self) -> bool:
        return self in {
            HPCJobState.COMPLETED,
            HPCJobState.FAILED,
            HPCJobState.CANCELLED,
            HPCJobState.TIMEOUT,
            HPCJobState.OUT_OF_MEMORY,
            HPCJobState.NODE_FAIL,
        }

    @property
    def succeeded(self) -> bool:
        """Slurm COMPLETED is NOT scientific validity (section 14)."""
        return self is HPCJobState.COMPLETED


class ResourceRequest(BaseModel):
    """Deterministic resource envelope for one job submission."""

    partition: str = "normal"
    nodes: int = Field(default=1, ge=1)
    tasks_per_node: int = Field(default=32, ge=1)
    walltime_minutes: int = Field(default=120, ge=1)
    memory_gb: float | None = Field(default=None, gt=0)
    extra_sbatch: list[str] = Field(default_factory=list)


class ResourcePolicy(BaseModel):
    """Hard resource caps; the LLM can never exceed them (section 15).

    The policy is evaluated deterministically on every submission attempt;
    a request outside the caps is rejected before any SSH/Slurm call.
    """

    allow_hpc_submit: bool = False
    max_nodes: int = 1
    max_tasks_per_node: int = 64
    max_walltime_minutes: int = 720
    allowed_partitions: list[str] = Field(default_factory=list)

    @classmethod
    def from_environment(cls) -> "ResourcePolicy":
        """Build from ``PHOTOMATAGENT_ALLOW_HPC_SUBMIT`` + hard limits."""
        allow = os.environ.get("PHOTOMATAGENT_ALLOW_HPC_SUBMIT", "0").strip()
        return cls(
            allow_hpc_submit=allow.lower() in {"1", "true", "yes", "on"},
            max_nodes=_int_env("PHOTOMATAGENT_HPC_MAX_NODES", 1),
            max_tasks_per_node=_int_env(
                "PHOTOMATAGENT_HPC_MAX_TASKS_PER_NODE", 64
            ),
            max_walltime_minutes=_int_env(
                "PHOTOMATAGENT_HPC_MAX_WALLTIME_MINUTES", 720
            ),
            allowed_partitions=[
                item.strip()
                for item in os.environ.get(
                    "PHOTOMATAGENT_HPC_ALLOWED_PARTITIONS", ""
                ).split(",")
                if item.strip()
            ],
        )

    def violations(self, request: ResourceRequest) -> list[str]:
        """Return a list of policy violations (empty means permitted)."""
        problems: list[str] = []
        if not self.allow_hpc_submit:
            problems.append(
                "HPC submission is disabled; set "
                "PHOTOMATAGENT_ALLOW_HPC_SUBMIT=1 to authorize real jobs"
            )
        if request.nodes > self.max_nodes:
            problems.append(
                f"nodes={request.nodes} exceeds policy max {self.max_nodes}"
            )
        if request.tasks_per_node > self.max_tasks_per_node:
            problems.append(
                f"tasks_per_node={request.tasks_per_node} exceeds policy max "
                f"{self.max_tasks_per_node}"
            )
        if request.walltime_minutes > self.max_walltime_minutes:
            problems.append(
                f"walltime {request.walltime_minutes} min exceeds policy max "
                f"{self.max_walltime_minutes} min"
            )
        if self.allowed_partitions and request.partition not in self.allowed_partitions:
            problems.append(
                f"partition {request.partition!r} not in allowed partitions "
                f"{self.allowed_partitions}"
            )
        return problems


class RemoteServerConfig(BaseModel):
    """SSH connection description for SCNet.

    ``private_key_path`` is kept out of every serialized form: use
    ``redacted()`` / ``public_dict()`` for anything that can reach the model
    context or logs.
    """

    host: str
    username: str = ""
    port: int = Field(default=22, ge=1, le=65535)
    private_key_path: str = ""
    remote_root: str = "~/photomatagent"
    connect_timeout_seconds: float = Field(default=20.0, gt=0)
    ssh_batch_mode: bool = True

    @property
    def destination(self) -> str:
        return f"{self.username}@{self.host}" if self.username else self.host

    def public_dict(self) -> dict[str, Any]:
        """Serializable form with all secrets removed (key path never shown)."""
        data = self.model_dump()
        data.pop("private_key_path", None)
        data["private_key_configured"] = bool(self.private_key_path)
        return data

    def redacted(self) -> "RemoteServerConfig":
        return self.model_copy(update={"private_key_path": ""})


class RemoteJobSpec(BaseModel):
    """Everything needed to stage and submit one remote job."""

    application: str
    job_name: str
    remote_directory: str
    script_name: str = "run.slurm"
    resource: ResourceRequest = Field(default_factory=ResourceRequest)
    module_load: str = ""
    executable: str = ""
    executable_args: list[str] = Field(default_factory=list)
    env_vars: dict[str, str] = Field(default_factory=dict)
    timeout_minutes: float | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class RemoteArtifactRef(BaseModel):
    """Reference to one remote (or mirrored local) output artifact."""

    name: str
    remote_path: str
    size_bytes: int | None = None
    sha256: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RemoteJobRef(BaseModel):
    """Handle returned by a successful submission (never contains secrets)."""

    backend: str
    application: str
    job_id: str
    remote_directory: str
    state: HPCJobState = HPCJobState.SUBMITTED
    submitted_at: datetime = Field(default_factory=_now)
    completed_at: datetime | None = None
    resource_request: ResourceRequest = Field(default_factory=ResourceRequest)
    stdout_ref: str = ""
    stderr_ref: str = ""
    artifacts: list[RemoteArtifactRef] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


class RemoteExecutionResult(BaseModel):
    """Bounded result of one SSH command (stdout/stderr are truncated)."""

    ok: bool
    returncode: int
    stdout: str = ""
    stderr: str = ""
    command: str = ""
    error: str = ""

    def public_dict(self, max_chars: int = 8000) -> dict[str, Any]:
        data = self.model_dump()
        data["stdout"] = _bounded(self.stdout, max_chars)
        data["stderr"] = _bounded(self.stderr, max_chars)
        return data


class ScientificArtifactRef(BaseModel):
    """Local artifact reference with hash and kind (Sprint 3 section 58).

    Large files (vasprun.xml, WAVECAR, CHGCAR, NAMD trajectories, ...) are
    never copied into the model context: the agent receives this reference
    plus a bounded summary.
    """

    path: str
    kind: str = "file"  # cif | poscar | vasprun | wavecar | chgcar | trajectory | spectrum | ...
    sha256: str = ""
    size_bytes: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def bounded_summary(self, max_chars: int = 2000) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "sha256": self.sha256[:16] + "..." if self.sha256 else "",
            "size_bytes": self.size_bytes,
            "metadata": self.metadata,
        }


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name, "")
    try:
        return int(value) if value.strip() else default
    except ValueError:
        return default


def _bounded(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated {len(text) - max_chars} chars]"
