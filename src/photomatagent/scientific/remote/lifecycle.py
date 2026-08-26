"""Idempotent, recoverable submission lifecycle for remote HPC jobs.

``SubmitOnceSession.submit_once`` implements the submit-once contract:

* a request may only be submitted after its scientific preflight gate passes;
* the same ``request_id`` never produces a second remote job (verified against
  the persistent SQLite registry before any upload or sbatch call);
* an sbatch client timeout is treated as ambiguous ("unknown whether the job
  was submitted") -- it is never retried blindly; ``reconcile`` first checks
  the local registry, the remote request marker, the unique job name and
  squeue/sacct, and only re-submits after confirming no job exists;
* every request gets its own unique remote directory, and the registry
  refuses two records sharing a remote directory;
* status-query failures return UNKNOWN with a structured error instead of
  being misjudged as a job failure.

This module performs no SSH and no sbatch itself: it drives a backend object
(``SCNetBackend`` or ``FakeSCNetBackend``) through its public methods.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from pydantic import BaseModel, Field

from photomatagent.scientific.remote.models import (
    HPCJobState,
    RemoteJobSpec,
    ResourceRequest,
)
from photomatagent.scientific.remote.registry import (
    JobLifecycleState,
    JobRecord,
    JobRegistry,
    assert_marker_safe,
    canonical_input_hash,
    derive_request_id,
    lifecycle_from_hpc,
    request_marker_payload,
)
from photomatagent.scientific.remote.scheduler import (
    sanitize_job_name,
    slurm_state_to_hpc_state,
)
from photomatagent.scientific.remote.scnet import validate_remote_path

MARKER_FILENAME = "photomatagent.request.json"


class SubmissionGate(BaseModel):
    """Scientific gate that must pass before upload/sbatch is allowed.

    For molecular VASP workflows this is produced from the deterministic
    preflight report (``passed`` must be true). Keeping it a plain value here
    avoids coupling the generic lifecycle to any one application.
    """

    passed: bool
    report_path: str | None = None
    errors: list[str] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


class SubmitOnceResult(BaseModel):
    """Outcome of one ``submit_once`` call (never contains secrets)."""

    request_id: str
    record: dict[str, Any]
    submitted: bool = False
    duplicate: bool = False
    needs_reconciliation: bool = False
    blocked: bool = False
    error: str = ""


class ReconciliationResult(BaseModel):
    """Result of a timeout/ambiguity reconciliation."""

    outcome: str
    request_id: str
    job_ids: list[str] = Field(default_factory=list)
    adopted_job_id: str | None = None
    scheduler_state: str | None = None
    state: str | None = None
    can_resubmit: bool = False
    marker_matched: bool = False
    error: str = ""


class StatusRefresh(BaseModel):
    """One scheduler status pull; failures never masquerade as job failure."""

    request_id: str
    job_id: str | None
    state: str | None = None
    scheduler_state: str | None = None
    ok: bool = True
    query_failed: bool = False
    error: str = ""


class JobBackend(Protocol):
    """Minimal backend surface the lifecycle relies on."""

    async def ensure_remote_directory(self, remote_directory: str) -> bool: ...

    async def upload_files(
        self, local_paths: list[Path], remote_directory: str
    ) -> list[str]: ...

    async def submit_script(self, spec: RemoteJobSpec) -> Any: ...

    async def job_status(self, job_id: str) -> HPCJobState: ...

    async def jobs_by_name(self, job_name: str) -> list[tuple[str, str]]: ...

    async def read_remote_text(
        self, remote_directory: str, filename: str, max_bytes: int
    ) -> str | None: ...

    async def copy_remote_artifact(
        self,
        source_remote_directory: str,
        destination_remote_directory: str,
        filename: str,
    ) -> bool: ...

    async def verify_remote_inputs(
        self, remote_directory: str, names: list[str]
    ) -> list[str]:
        """Return the requested names missing on the remote directory.

        Implementations that cannot verify remotely SHOULD return ``names``
        verbatim (treat everything as missing) so a submitter never skips an
        input-availability check; the lifecycle treats a verification error
        as a hard pre-sbatch failure.
        """
        ...


class RemoteArtifactCopy(BaseModel):
    """One allow-listed artifact staged into the new job directory."""

    source_remote_directory: str
    filename: str


class SubmitOnceSession:
    """Registry-backed, idempotent submission session for one backend."""

    def __init__(
        self,
        registry: JobRegistry,
        backend: JobBackend,
        *,
        remote_root: str = "~/photomatagent",
        marker_temp_dir: str | Path | None = None,
    ) -> None:
        self.registry = registry
        self.backend = backend
        self.remote_root = remote_root.rstrip("/")
        self.marker_temp_dir = (
            Path(marker_temp_dir).expanduser()
            if marker_temp_dir is not None
            else None
        )

    # -- public API ---------------------------------------------------------

    async def submit_once(
        self,
        *,
        application: str,
        workflow_stage: str,
        job_name: str,
        local_input_dir: str | Path,
        gate: SubmissionGate,
        resource: ResourceRequest,
        executable: str,
        script_name: str = "run.slurm",
        module_load: str = "",
        executable_args: list[str] | None = None,
        env_vars: dict[str, str] | None = None,
        timeout_minutes: float | None = None,
        request_id: str | None = None,
        provenance: dict[str, Any] | None = None,
        force_new_attempt: bool = False,
        remote_copies: list[RemoteArtifactCopy] | None = None,
        script_renderer: Callable[[str, ResourceRequest], str] | None = None,
        potcar_mode: str = "none",
        potcar_symbols: list[str] | None = None,
        remote_psp_dir: str = "",
    ) -> SubmitOnceResult:
        """Submit at most once per request_id; see module docstring."""
        input_dir = Path(local_input_dir).expanduser().resolve()
        hash_value = canonical_input_hash(input_dir)
        rid = request_id or derive_request_id(application, workflow_stage, hash_value)
        original_request_id: str | None = None

        record = self.registry.get(rid)
        if record is not None:
            original_request_id = record.request_id
            if force_new_attempt:
                rid = self._attempt_request_id(rid)
                while self.registry.get(rid) is not None:
                    rid = self._attempt_request_id(rid)
            else:
                existing = await self._handle_existing(record, rid, input_dir)
                if existing is not None:
                    return existing

        # -- scientific gate (checked before ANY upload/sbatch call) ----------
        if not gate.passed:
            self._store(
                rid,
                application=application,
                workflow_stage=workflow_stage,
                job_name=job_name,
                hash_value=hash_value,
                input_dir=input_dir,
                resource=resource,
                provenance=provenance,
                state=JobLifecycleState.PREPARED,
                scientific_validation_state="failed",
                last_error=(
                    "; ".join(gate.errors)
                    or "preflight gate failed; submission refused"
                ),
                parent_request_id=original_request_id,
            )
            current = self.registry.get(rid)
            assert current is not None
            return SubmitOnceResult(
                request_id=rid,
                record=current.public_dict(),
                blocked=True,
                error="preflight gate failed; upload and sbatch were not called",
            )

        record = self._store(
            rid,
            application=application,
            workflow_stage=workflow_stage,
            job_name=job_name,
            hash_value=hash_value,
            input_dir=input_dir,
            resource=resource,
            provenance=provenance,
            state=JobLifecycleState.PREFLIGHT_PASSED,
            scientific_validation_state="passed",
            last_error=None,
            parent_request_id=original_request_id,
        )
        return await self._submit_attempt(
            record=record,
            input_dir=input_dir,
            gate=gate,
            script_name=script_name,
            module_load=module_load,
            executable=executable,
            executable_args=executable_args,
            env_vars=env_vars,
            timeout_minutes=timeout_minutes,
            provenance=provenance,
            remote_copies=remote_copies,
            script_renderer=script_renderer,
            potcar_mode=potcar_mode,
            potcar_symbols=potcar_symbols or [],
            remote_psp_dir=remote_psp_dir,
        )

    def mark_result_state(
        self,
        request_id: str,
        *,
        collected: bool,
        validated: bool,
        evidence: int = 0,
        error: str = "",
    ) -> "JobRecord | None":
        """Advance the lifecycle once results are collected and validated.

        Scheduler COMPLETED has no scientific meaning on its own: a
        downloaded job becomes COLLECTED and only becomes VALIDATED after
        the result analysis passes. Failed validation keeps the COLLECTED
        state with a recorded error and never fabricates evidence.
        """
        record = self.registry.get(request_id)
        if record is None:
            return None
        if collected and record.state in {
            JobLifecycleState.SUBMITTED,
            JobLifecycleState.PENDING,
            JobLifecycleState.RUNNING,
            JobLifecycleState.COMPLETED,
        }:
            self.registry.update(
                request_id,
                state=JobLifecycleState.COLLECTED,
                scientific_validation_state="collected",
                completed_at=self._now(),
                last_error=None,
            )
        if validated:
            self.registry.update(
                request_id,
                state=JobLifecycleState.VALIDATED,
                scientific_validation_state="passed",
                completed_at=self._now(),
                last_error=None,
            )
        elif collected:
            self.registry.update(
                request_id,
                scientific_validation_state="failed",
                last_error=error or "result validation failed",
            )
        return self.registry.get(request_id)

    async def reconcile(self, request_id: str) -> ReconciliationResult:
        """Recover from ambiguous submission state (e.g. sbatch timeout).

        Order: local registry -> remote request marker -> unique job name ->
        squeue/sacct. Only ``can_resubmit=True`` permits a new submission.
        """
        record = self.registry.get(request_id)
        if record is None:
            return ReconciliationResult(
                outcome="NOT_FOUND",
                request_id=request_id,
                error="no registry record for this request_id",
            )
        if record.job_id is not None:
            status = await self._query_status(request_id, record.job_id)
            if status.ok and status.state is not None:
                return ReconciliationResult(
                    outcome="FOUND_UNIQUE",
                    request_id=request_id,
                    job_ids=[record.job_id or ""],
                    adopted_job_id=record.job_id,
                    scheduler_state=status.scheduler_state,
                    state=status.state,
                    marker_matched=await self._marker_matches(record),
                )
            # Job id recorded but not confirmed alive: fall through to the
            # by-name scan (covers lost job ids from sbatch timeouts too).

        marker_matched = await self._marker_matches(record)
        assert record.job_name is not None
        try:
            candidates = await self.backend.jobs_by_name(record.job_name)
        except Exception as exc:
            self.registry.update(
                request_id,
                last_error=f"reconciliation query failed: {type(exc).__name__}: {exc}",
            )
            return ReconciliationResult(
                outcome="QUERY_FAILED",
                request_id=request_id,
                marker_matched=marker_matched,
                error=f"reconciliation query failed: {type(exc).__name__}: {exc}",
            )
        job_ids = [job_id for job_id, _ in candidates]
        if len(job_ids) > 1:
            self.registry.update(
                request_id,
                state=JobLifecycleState.UNKNOWN_RECONCILIATION_REQUIRED,
                last_error=(
                    "multiple candidate jobs found during reconciliation: "
                    + ", ".join(job_ids)
                ),
            )
            return ReconciliationResult(
                outcome="FOUND_MULTIPLE",
                request_id=request_id,
                job_ids=job_ids,
                marker_matched=marker_matched,
                state=JobLifecycleState.UNKNOWN_RECONCILIATION_REQUIRED.value,
                error=(
                    "multiple candidate jobs exist; duplicate-busting requires "
                    "an explicit decision, no auto-resubmit"
                ),
            )
        if len(job_ids) == 1:
            job_id, slurm_state = candidates[0]
            scheduler_state = slurm_state_to_hpc_state(slurm_state)
            mapped = lifecycle_from_hpc(scheduler_state)
            if mapped is None:
                self.registry.update(
                    request_id,
                    state=JobLifecycleState.UNKNOWN_RECONCILIATION_REQUIRED,
                    job_id=job_id,
                    last_error=(
                        f"candidate job {job_id} has unknown state {slurm_state!r}"
                    ),
                )
                return ReconciliationResult(
                    outcome="FOUND_UNIQUE_UNKNOWN_STATE",
                    request_id=request_id,
                    job_ids=job_ids,
                    adopted_job_id=job_id,
                    marker_matched=marker_matched,
                    state=JobLifecycleState.UNKNOWN_RECONCILIATION_REQUIRED.value,
                    error=f"candidate job {job_id} exists with unknown state",
                )
            self.registry.update(
                request_id,
                state=mapped,
                scheduler_state=scheduler_state,
                job_id=job_id,
                completed_at=self._now() if mapped.terminal else None,
                last_error=None,
            )
            return ReconciliationResult(
                outcome="FOUND_UNIQUE",
                request_id=request_id,
                job_ids=job_ids,
                adopted_job_id=job_id,
                scheduler_state=slurm_state,
                state=mapped.value,
                marker_matched=marker_matched,
            )
        # No candidates anywhere: safe to resubmit (a fresh attempt).
        self.registry.update(
            request_id,
            last_error="reconciliation found no existing job; safe to resubmit",
        )
        return ReconciliationResult(
            outcome="NOT_FOUND",
            request_id=request_id,
            marker_matched=marker_matched,
            can_resubmit=True,
            state=record.state.value,
        )

    async def refresh_status(self, request_id: str) -> StatusRefresh:
        """One status pull. Query failures return UNKNOWN, never FAILED."""
        record = self.registry.get(request_id)
        if record is None:
            return StatusRefresh(
                request_id=request_id,
                job_id=None,
                ok=False,
                query_failed=True,
                error="no registry record for this request_id",
            )
        if record.job_id is None:
            return StatusRefresh(
                request_id=request_id,
                job_id=None,
                ok=False,
                query_failed=True,
                error="record has no job_id; run reconcile first",
            )
        return await self._query_status(request_id, record.job_id)

    # -- internals ----------------------------------------------------------

    async def _handle_existing(
        self, record: JobRecord, request_id: str, input_dir: Path
    ) -> SubmitOnceResult | None:
        """Return a result for an already-registered request_id, or None."""
        if record.state in {
            JobLifecycleState.SUBMITTING,
            JobLifecycleState.PREFLIGHT_PASSED,
            JobLifecycleState.PREPARED,
        }:
            # Ambiguous client state: reconcile instead of blind resubmit.
            rec = await self.reconcile(request_id)
            if rec.outcome == "QUERY_FAILED":
                return SubmitOnceResult(
                    request_id=request_id,
                    record=record.public_dict(),
                    needs_reconciliation=True,
                    error=rec.error,
                )
            if rec.outcome == "NOT_FOUND":
                return None  # confirmed absent: a new attempt is allowed
            current = self.registry.get(request_id) or record
            return SubmitOnceResult(
                request_id=request_id,
                record=current.public_dict(),
                duplicate=True,
                needs_reconciliation=rec.outcome
                in {"FOUND_MULTIPLE", "FOUND_UNIQUE_UNKNOWN_STATE"},
                error=rec.error,
            )
        if record.state in {
            JobLifecycleState.SUBMITTED,
            JobLifecycleState.PENDING,
            JobLifecycleState.RUNNING,
        }:
            return SubmitOnceResult(
                request_id=request_id,
                record=record.public_dict(),
                duplicate=True,
                error=(
                    f"request already active as job {record.job_id}; "
                    "no second job was created"
                ),
            )
        if (
            record.state in {JobLifecycleState.FAILED}
            and record.job_id is None
        ):
            # Failed before a job existed (e.g. upload/sbatch rejection):
            # verify remotely that nothing was created, then allow a retry.
            rec = await self.reconcile(request_id)
            if rec.outcome == "NOT_FOUND":
                return None
            current = self.registry.get(request_id) or record
            return SubmitOnceResult(
                request_id=request_id,
                record=current.public_dict(),
                duplicate=True,
                needs_reconciliation=True,
                error=rec.error,
            )
        if record.state is JobLifecycleState.UNKNOWN_RECONCILIATION_REQUIRED:
            return SubmitOnceResult(
                request_id=request_id,
                record=record.public_dict(),
                duplicate=True,
                needs_reconciliation=True,
                error=(
                    "reconciliation required: duplicate or unknown jobs exist; "
                    "no second job was created"
                ),
            )
        return SubmitOnceResult(
            request_id=request_id,
            record=record.public_dict(),
            duplicate=True,
            error=(
                f"request already terminal in state {record.state.value}; "
                "use force_new_attempt for an explicit new request"
            ),
        )

    async def _submit_attempt(
        self,
        *,
        record: JobRecord,
        input_dir: Path,
        gate: SubmissionGate,
        script_name: str,
        module_load: str,
        executable: str,
        executable_args: list[str] | None,
        env_vars: dict[str, str] | None,
        timeout_minutes: float | None,
        provenance: dict[str, Any] | None,
        remote_copies: list[RemoteArtifactCopy] | None,
        script_renderer: Callable[[str, ResourceRequest], str] | None,
        potcar_mode: str,
        potcar_symbols: list[str],
        remote_psp_dir: str,
    ) -> SubmitOnceResult:
        request_id = record.request_id
        remote_directory = self._unique_remote_directory(
            record.job_name
        )
        clash = self.registry.find_by_remote_directory(
            remote_directory, exclude_request_id=request_id
        )
        if clash is not None:
            self.registry.update(
                request_id,
                state=JobLifecycleState.UNKNOWN_RECONCILIATION_REQUIRED,
                last_error=(
                    f"remote directory {remote_directory} is already claimed "
                    f"by request {clash.request_id}"
                ),
            )
            current = self.registry.get(request_id)
            assert current is not None
            return SubmitOnceResult(
                request_id=request_id,
                record=current.public_dict(),
                blocked=True,
                error=(
                    "remote directory collision; no upload or sbatch performed"
                ),
            )
        self.registry.update(
            request_id,
            state=JobLifecycleState.SUBMITTING,
            remote_directory=remote_directory,
        )
        updated = self.registry.get(request_id)
        assert updated is not None
        record = updated

        names = [
            name
            for name in ("POSCAR", "INCAR", "KPOINTS", "POTCAR.meta")
            if (input_dir / name).is_file()
        ]
        script_rendered = False
        if script_renderer is not None:
            # The submission script is generated from the caller's single
            # renderer and staged locally only so it can be uploaded; it is
            # regenerated on every attempt (job name carries the request id).
            script_path = input_dir / script_name
            script_path.write_text(
                script_renderer(record.job_name, record.resource),
                encoding="utf-8",
            )
            script_rendered = True
            names.append(script_name)
        # POTCAR is only uploaded when the caller explicitly opts into the
        # local (materialized) mode; a real POTCAR sitting in the input dir
        # is never swept up by a generic submit (mode defaults to "none").
        local_potcar = potcar_mode == "local" and (input_dir / "POTCAR").is_file()
        if local_potcar:
            names.append("POTCAR")
        if not names:
            self.registry.update(
                request_id,
                state=JobLifecycleState.FAILED,
                last_error="no input files found in local input dir",
            )
            current = self.registry.get(request_id)
            assert current is not None
            return SubmitOnceResult(
                request_id=request_id,
                record=current.public_dict(),
                error="no input files found in local input dir",
            )
        # -- deterministic POTCAR policy -----------------------------------
        for symbol in potcar_symbols:
            if not symbol.isalpha():
                self.registry.update(
                    request_id,
                    state=JobLifecycleState.FAILED,
                    last_error=f"unsafe POTCAR symbol: {symbol!r}",
                )
                current = self.registry.get(request_id)
                assert current is not None
                return SubmitOnceResult(
                    request_id=request_id,
                    record=current.public_dict(),
                    blocked=True,
                    error="unsafe POTCAR symbol; upload and sbatch refused",
                )
        if potcar_mode == "remote":
            if not remote_psp_dir:
                self.registry.update(
                    request_id,
                    state=JobLifecycleState.FAILED,
                    last_error=(
                        "remote POTCAR assembly requested but no "
                        "SCNET_VASP_PSP_DIR is configured"
                    ),
                )
                current = self.registry.get(request_id)
                assert current is not None
                return SubmitOnceResult(
                    request_id=request_id,
                    record=current.public_dict(),
                    blocked=True,
                    error=(
                        "remote POTCAR assembly needs SCNET_VASP_PSP_DIR; "
                        "upload and sbatch refused"
                    ),
                )
            if not potcar_symbols:
                self.registry.update(
                    request_id,
                    state=JobLifecycleState.FAILED,
                    last_error=(
                        "remote POTCAR assembly requested but the element "
                        "sequence is unknown"
                    ),
                )
                current = self.registry.get(request_id)
                assert current is not None
                return SubmitOnceResult(
                    request_id=request_id,
                    record=current.public_dict(),
                    blocked=True,
                    error=(
                        "remote POTCAR assembly needs a known element "
                        "sequence; upload and sbatch refused"
                    ),
                )
        elif potcar_mode == "local" and not (input_dir / "POTCAR").is_file():
            self.registry.update(
                request_id,
                state=JobLifecycleState.FAILED,
                last_error="local POTCAR requested but not present",
            )
            current = self.registry.get(request_id)
            assert current is not None
            return SubmitOnceResult(
                request_id=request_id,
                record=current.public_dict(),
                blocked=True,
                error="local POTCAR requested but not present in the input dir",
            )
        elif potcar_mode not in {"none", "local", "remote"}:
            self.registry.update(
                request_id,
                state=JobLifecycleState.FAILED,
                last_error=f"invalid potcar_mode: {potcar_mode!r}",
            )
            current = self.registry.get(request_id)
            assert current is not None
            return SubmitOnceResult(
                request_id=request_id,
                record=current.public_dict(),
                blocked=True,
                error=f"invalid potcar_mode {potcar_mode!r}",
            )

        marker_path = self._write_marker(
            request_id=request_id,
            job_name=record.job_name,
            hash_value=record.canonical_input_hash,
            remote_directory=remote_directory,
            attempt=record.retry_count,
        )
        try:
            await self.backend.upload_files(
                [input_dir / name for name in names] + [marker_path],
                remote_directory,
            )
        except Exception as exc:
            self.registry.update(
                request_id,
                state=JobLifecycleState.FAILED,
                last_error=f"upload failed: {type(exc).__name__}: {exc}",
            )
            current = self.registry.get(request_id)
            assert current is not None
            return SubmitOnceResult(
                request_id=request_id,
                record=current.public_dict(),
                error=f"upload failed: {type(exc).__name__}: {exc}",
            )

        for copy in remote_copies or []:
            try:
                ok = await self.backend.copy_remote_artifact(
                    copy.source_remote_directory,
                    remote_directory,
                    copy.filename,
                )
            except Exception as exc:
                ok = False
                detail = f"{type(exc).__name__}: {exc}"
            else:
                detail = "artifact missing on the upstream directory"
            if not ok:
                self.registry.update(
                    request_id,
                    state=JobLifecycleState.FAILED,
                    last_error=(
                        f"required remote artifact {copy.filename} could not "
                        f"be staged from {copy.source_remote_directory} "
                        f"({detail})"
                    ),
                )
                current = self.registry.get(request_id)
                assert current is not None
                return SubmitOnceResult(
                    request_id=request_id,
                    record=current.public_dict(),
                    error=(
                        f"remote artifact staging failed for "
                        f"{copy.filename}; sbatch was not called"
                    ),
                )

        # -- verify the remote inputs before touching the scheduler ---------
        required = ["POSCAR", "INCAR", "KPOINTS"]
        if script_rendered:
            required.append(script_name)
        if local_potcar:
            required.append("POTCAR")
        verify = getattr(self.backend, "verify_remote_inputs", None)
        if verify is not None:
            try:
                missing = await verify(remote_directory, required)
            except Exception as exc:
                self.registry.update(
                    request_id,
                    state=JobLifecycleState.FAILED,
                    last_error=(
                        "remote input verification failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
                current = self.registry.get(request_id)
                assert current is not None
                return SubmitOnceResult(
                    request_id=request_id,
                    record=current.public_dict(),
                    error=(
                        "remote input verification failed; sbatch was not "
                        "called"
                    ),
                )
            if missing:
                self.registry.update(
                    request_id,
                    state=JobLifecycleState.FAILED,
                    last_error=(
                        "remote inputs missing before sbatch: "
                        + ", ".join(missing)
                    ),
                )
                current = self.registry.get(request_id)
                assert current is not None
                return SubmitOnceResult(
                    request_id=request_id,
                    record=current.public_dict(),
                    error=(
                        "remote inputs missing before sbatch: "
                        + ", ".join(missing)
                    ),
                )

        spec = RemoteJobSpec(
            application="vasp",
            job_name=record.job_name,
            remote_directory=remote_directory,
            script_name=script_name,
            resource=record.resource,
            module_load=module_load,
            executable=executable,
            executable_args=executable_args or [],
            env_vars=env_vars or {},
            timeout_minutes=timeout_minutes,
            provenance={
                "request_id": request_id,
                "workflow_stage": record.workflow_stage,
                "canonical_input_hash": record.canonical_input_hash,
                "preflight_report": gate.report_path,
                "scientific_validation": "passed",
                "remote_copies": [
                    copy.model_dump() for copy in (remote_copies or [])
                ],
                "potcar_mode": potcar_mode,
                "potcar_symbols": list(potcar_symbols),
                **(provenance or {}),
            },
        )
        try:
            ref = await self.backend.submit_script(spec)
        except TimeoutError:
            self.registry.update(
                request_id,
                state=JobLifecycleState.SUBMITTING,
                last_error=(
                    "sbatch client timed out; whether the job was submitted "
                    "is unknown -- reconciliation is required before retry"
                ),
            )
            current = self.registry.get(request_id)
            assert current is not None
            return SubmitOnceResult(
                request_id=request_id,
                record=current.public_dict(),
                needs_reconciliation=True,
                error=(
                    "sbatch client timed out after upload; the job may exist "
                    "remotely -- run reconciliation before any resubmit"
                ),
            )
        except Exception as exc:
            self.registry.update(
                request_id,
                state=JobLifecycleState.FAILED,
                last_error=f"sbatch failed: {type(exc).__name__}: {exc}",
            )
            current = self.registry.get(request_id)
            assert current is not None
            return SubmitOnceResult(
                request_id=request_id,
                record=current.public_dict(),
                error=f"sbatch failed: {type(exc).__name__}: {exc}",
            )
        job_id = str(ref.job_id)
        self.registry.update(
            request_id,
            state=JobLifecycleState.SUBMITTED,
            scheduler_state=HPCJobState.SUBMITTED,
            job_id=job_id,
            submitted_at=self._now(),
            last_error=None,
        )
        current = self.registry.get(request_id)
        assert current is not None
        return SubmitOnceResult(
            request_id=request_id,
            record=current.public_dict(),
            submitted=True,
        )

    async def _query_status(
        self, request_id: str, job_id: str
    ) -> StatusRefresh:
        try:
            scheduler_state = await self.backend.job_status(job_id)
        except Exception as exc:
            return StatusRefresh(
                request_id=request_id,
                job_id=job_id,
                ok=False,
                query_failed=True,
                error=f"status query failed: {type(exc).__name__}: {exc}",
            )
        if scheduler_state is HPCJobState.UNKNOWN:
            # Query "failed to determine state" is NOT a job failure.
            return StatusRefresh(
                request_id=request_id,
                job_id=job_id,
                scheduler_state=HPCJobState.UNKNOWN.value,
                ok=False,
                query_failed=True,
                error="backend could not determine scheduler state",
            )
        mapped = lifecycle_from_hpc(scheduler_state)
        self.registry.update(
            request_id,
            state=mapped,
            scheduler_state=scheduler_state,
            completed_at=self._now() if (mapped and mapped.terminal) else None,
            last_error=None if mapped else None,
        )
        return StatusRefresh(
            request_id=request_id,
            job_id=job_id,
            state=mapped.value if mapped else None,
            scheduler_state=scheduler_state.value,
            ok=True,
        )

    async def _marker_matches(self, record: JobRecord) -> bool:
        if record.remote_directory is None:
            return False
        try:
            text = await self.backend.read_remote_text(
                record.remote_directory, MARKER_FILENAME, 4096
            )
        except Exception:
            return False
        if not text:
            return False
        try:
            payload = json.loads(text)
        except ValueError:
            return False
        return payload.get("request_id") == record.request_id

    def _write_marker(
        self,
        *,
        request_id: str,
        job_name: str,
        hash_value: str,
        remote_directory: str,
        attempt: int,
    ) -> Path:
        payload = request_marker_payload(
            request_id=request_id,
            job_name=job_name,
            canonical_hash=hash_value,
            remote_directory=remote_directory,
            attempt=attempt,
        )
        assert_marker_safe(payload)
        directory = self.marker_temp_dir or Path(tempfile.gettempdir())
        # Per-request subdirectory: the uploaded remote basename must be the
        # canonical marker name, and concurrent sessions must not collide.
        directory = directory / request_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / MARKER_FILENAME
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    def _store(
        self,
        request_id: str,
        *,
        application: str,
        workflow_stage: str,
        job_name: str,
        hash_value: str,
        input_dir: Path,
        resource: ResourceRequest,
        provenance: dict[str, Any] | None,
        state: JobLifecycleState,
        scientific_validation_state: str,
        last_error: str | None,
        parent_request_id: str | None,
    ) -> JobRecord:
        previous = self.registry.get(request_id)
        safe_name = self._job_name_with_request(job_name, request_id)
        record = JobRecord(
            request_id=request_id,
            canonical_input_hash=hash_value,
            job_name=safe_name,
            workflow_stage=workflow_stage,
            resource=resource,
            state=state,
            local_input_dir=input_dir,
            scientific_validation_state=scientific_validation_state,
            retry_count=(previous.retry_count + 1) if previous else 0,
            last_error=last_error,
            provenance={
                "application": application,
                "job_name": job_name,
                **(provenance or {}),
            },
            parent_request_id=parent_request_id,
        )
        self.registry.put(record)
        return record

    @staticmethod
    def _job_name_with_request(job_name: str, request_id: str) -> str:
        """Slurm job name scoped to the request (uniqueness aid)."""
        base = sanitize_job_name(job_name, max_len=40)
        return f"{base}-{request_id[:8]}"

    def _unique_remote_directory(self, safe_job_name: str) -> str:
        """Per-request unique remote directory (never reused across jobs).

        ``safe_job_name`` already carries the request-id suffix; the timestamp
        (microsecond precision) keeps repeated attempts in separate dirs.
        """
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        path = f"{self.remote_root}/{safe_job_name}-{stamp}"
        validate_remote_path(path)
        return path

    @staticmethod
    def _attempt_request_id(request_id: str) -> str:
        return f"{request_id}-attempt-{_attempt_counter()}"

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _attempt_counter() -> int:
    import random

    return random.SystemRandom().randrange(1, 10**9)
