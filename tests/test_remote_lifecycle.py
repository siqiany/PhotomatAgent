"""Offline tests for the idempotent remote job lifecycle (no SSH/Slurm).

Covers: SQLite job registry, canonical hashing, request markers, submit-once
semantics, sbatch-timeout reconciliation, status refresh, capability caching,
detached monitoring and the molecular preflight gate adapter. All backend
interactions run against ``FakeSCNetBackend`` (extended with safety/lifecycle
helpers) -- no real SSH is ever attempted.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from photomatagent.scientific.applications.vasp.molecular.models import (
    PreflightReport,
)
from photomatagent.scientific.applications.vasp.molecular.preflight import (
    preflight_gate,
)
from photomatagent.scientific.remote.cache import CapabilityCache
from photomatagent.scientific.remote.fake import FakeSCNetBackend
from photomatagent.scientific.remote.lifecycle import (
    MARKER_FILENAME,
    SubmitOnceSession,
    SubmissionGate,
)
from photomatagent.scientific.remote.models import (
    HPCJobState,
    RemoteServerConfig,
    ResourcePolicy,
    ResourceRequest,
)
from photomatagent.scientific.remote.monitor import JobMonitor
from photomatagent.scientific.remote.registry import (
    JobLifecycleState,
    JobRecord,
    JobRegistry,
    canonical_input_hash,
    derive_request_id,
    lifecycle_from_hpc,
    request_marker_payload,
)
from photomatagent.scientific.remote.scheduler import (
    remote_jobs_by_name_command,
)
from photomatagent.scientific.remote.scnet import SCNetBackend

RESOURCE = ResourceRequest(partition="kshcnormal", nodes=1, tasks_per_node=32)


def _gate(passed: bool = True, **extra) -> SubmissionGate:
    return SubmissionGate(passed=passed, **extra)


def _input_dir(tmp_path: Path, name: str = "stage") -> Path:
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "POSCAR").write_text("C O H\n1 1 4\nDirect\n", encoding="utf-8")
    (directory / "INCAR").write_text("ENCUT = 520\n", encoding="utf-8")
    (directory / "KPOINTS").write_text(
        "Gamma-point\n0\nGamma\n1 1 1\n0 0 0\n", encoding="utf-8"
    )
    return directory


def _session(tmp_path: Path, backend: FakeSCNetBackend) -> SubmitOnceSession:
    return SubmitOnceSession(
        JobRegistry(tmp_path / "jobs.sqlite3"),
        backend,
        marker_temp_dir=tmp_path / "markers",
    )


async def _submit(
    session: SubmitOnceSession,
    input_dir: Path,
    *,
    job_name: str = "dme-li",
    request_id: str | None = None,
    gate: SubmissionGate | None = None,
    **kwargs,
):
    return await session.submit_once(
        application="vasp",
        workflow_stage="static",
        job_name=job_name,
        local_input_dir=input_dir,
        gate=gate or _gate(),
        resource=RESOURCE,
        executable="vasp_std",
        request_id=request_id,
        **kwargs,
    )


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


def test_lifecycle_state_vocabulary():
    states = {state.value for state in JobLifecycleState}
    assert states == {
        "PREPARED",
        "PREFLIGHT_PASSED",
        "SUBMITTING",
        "SUBMITTED",
        "PENDING",
        "RUNNING",
        "COMPLETED",
        "COLLECTED",
        "VALIDATED",
        "FAILED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "CANCELLED",
        "UNKNOWN_RECONCILIATION_REQUIRED",
    }
    assert JobLifecycleState.COMPLETED.terminal is True
    assert JobLifecycleState.SUBMITTED.terminal is False
    assert JobLifecycleState.UNKNOWN_RECONCILIATION_REQUIRED.requires_attention


def test_hpc_to_lifecycle_mapping():
    assert lifecycle_from_hpc(HPCJobState.PENDING) is JobLifecycleState.PENDING
    assert lifecycle_from_hpc(HPCJobState.RUNNING) is JobLifecycleState.RUNNING
    assert lifecycle_from_hpc(HPCJobState.COMPLETED) is JobLifecycleState.COMPLETED
    assert lifecycle_from_hpc(HPCJobState.FAILED) is JobLifecycleState.FAILED
    assert lifecycle_from_hpc(HPCJobState.TIMEOUT) is JobLifecycleState.TIMEOUT
    assert (
        lifecycle_from_hpc(HPCJobState.OUT_OF_MEMORY)
        is JobLifecycleState.OUT_OF_MEMORY
    )
    assert lifecycle_from_hpc(HPCJobState.CANCELLED) is JobLifecycleState.CANCELLED
    assert lifecycle_from_hpc(HPCJobState.NODE_FAIL) is JobLifecycleState.FAILED
    # UNKNOWN (query failure) must never be mistaken for a job failure.
    assert lifecycle_from_hpc(HPCJobState.UNKNOWN) is None


def test_registry_persists_across_reopen(tmp_path):
    db = tmp_path / "jobs.sqlite3"
    record = JobRecord(
        request_id="req-1",
        canonical_input_hash="abc123",
        job_name="dme-li-abcdef12",
        workflow_stage="static",
        resource=RESOURCE,
        state=JobLifecycleState.SUBMITTED,
        job_id="1001",
        remote_directory="~/photomatagent/dme-li-abcdef12-20260824_000000_000001",
    )
    with JobRegistry(db) as registry:
        registry.put(record)
        assert registry.get("req-1").job_id == "1001"  # type: ignore[union-attr]
    with JobRegistry(db) as registry:
        loaded = registry.get("req-1")
        assert loaded is not None
        assert loaded.state is JobLifecycleState.SUBMITTED
        assert loaded.resource == RESOURCE
        assert loaded.job_id == "1001"


def test_registry_refuses_job_id_regression(tmp_path):
    registry = JobRegistry(tmp_path / "jobs.sqlite3")
    registry.put(
        JobRecord(
            request_id="req-1",
            canonical_input_hash="h",
            job_name="n-abcdef12",
            workflow_stage="static",
            resource=RESOURCE,
            state=JobLifecycleState.SUBMITTED,
            job_id="1001",
            remote_directory="~/photomatagent/n-abcdef12-ts",
        )
    )
    with pytest.raises(ValueError, match="refusing to change job_id"):
        registry.update("req-1", job_id="1002")
    registry.close()


def test_registry_unique_remote_directory_invariant(tmp_path):
    registry = JobRegistry(tmp_path / "jobs.sqlite3")
    shared = "~/photomatagent/shared-dir"
    registry.put(
        JobRecord(
            request_id="req-1",
            canonical_input_hash="h",
            job_name="n-abcdef12",
            workflow_stage="static",
            resource=RESOURCE,
            state=JobLifecycleState.PREPARED,
            remote_directory=shared,
        )
    )
    with pytest.raises(Exception):
        registry.put(
            JobRecord(
                request_id="req-2",
                canonical_input_hash="h2",
                job_name="n2-abcdef12",
                workflow_stage="static",
                resource=RESOURCE,
                state=JobLifecycleState.PREPARED,
                remote_directory=shared,
            )
        )
    registry.close()


def test_canonical_input_hash_deterministic_and_sensitive(tmp_path):
    directory = _input_dir(tmp_path)
    first = canonical_input_hash(directory)
    assert canonical_input_hash(directory) == first
    (directory / "INCAR").write_text("ENCUT = 600\n", encoding="utf-8")
    assert canonical_input_hash(directory) != first


def test_derive_request_id_stable():
    assert (
        derive_request_id("vasp", "static", "hash-a")
        == derive_request_id("vasp", "static", "hash-a")
    )
    assert (
        derive_request_id("vasp", "static", "hash-a")
        != derive_request_id("vasp", "static", "hash-b")
    )


def test_marker_payload_is_safe():
    payload = request_marker_payload(
        request_id="req-1",
        job_name="dme-li-abcdef12",
        canonical_hash="deadbeef",
        remote_directory="~/photomatagent/dme-li-abcdef12-ts",
        attempt=2,
    )
    text = json.dumps(payload)
    assert "potcar" not in text.lower()
    assert "private_key" not in text.lower()
    assert "/.ssh/" not in text
    assert payload["request_id"] == "req-1"
    assert payload["canonical_input_hash"] == "deadbeef"


def test_remote_jobs_by_name_command_shape():
    command = remote_jobs_by_name_command("dme-li-abcdef12")
    assert "squeue -h --name=" in command
    assert "sacct -n -X --name=" in command
    assert command.count("dme-li-abcdef12") == 2  # both queries, same name


# --------------------------------------------------------------------------
# capability cache
# --------------------------------------------------------------------------


async def test_capability_cache_hits_misses_and_exception_policy():
    cache = CapabilityCache(ttl_seconds=60)
    calls = 0

    async def probe():
        nonlocal calls
        calls += 1
        return {"value": calls}

    assert await cache.get_or_call("a", probe) == {"value": 1}
    assert await cache.get_or_call("a", probe) == {"value": 1}
    assert await cache.get_or_call("b", probe) == {"value": 2}
    assert calls == 2
    assert cache.stats()["hits"] == 1

    async def boom():
        raise OSError("transient ssh failure")

    with pytest.raises(OSError):
        await cache.get_or_call("boom", boom)
    # Failed probes are not cached: a retry re-runs them.
    with pytest.raises(OSError):
        await cache.get_or_call("boom", boom)
    cache.invalidate("a")
    await cache.get_or_call("a", probe)  # recomputes
    assert calls == 3


async def test_scnet_backend_caches_capability_probes():
    backend = SCNetBackend(RemoteServerConfig(host="login.scnet.cn", username="u"))
    calls = {"count": 0}

    async def probe():
        calls["count"] += 1
        return {"connected": "true"}

    first = await backend.capability_cache.get_or_call("connection", probe)
    second = await backend.capability_cache.get_or_call("connection", probe)
    assert first == second == {"connected": "true"}
    assert calls["count"] == 1


def test_server_config_has_separate_timeouts():
    config = RemoteServerConfig(
        host="login.scnet.cn",
        connect_timeout_seconds=25,
        command_timeout_seconds=180,
        transfer_timeout_seconds=7200,
    )
    assert config.command_timeout_seconds == 180
    assert config.connect_timeout_seconds == 25
    assert config.transfer_timeout_seconds == 7200


async def test_scnet_backend_submit_timeout_raises_timeouterror_not_runtime():
    """A timed-out sbatch client is ambiguous: TimeoutError, never FAILED."""
    from photomatagent.scientific.remote.models import (
        RemoteExecutionResult,
        RemoteJobSpec,
    )

    class TimeoutBackend(SCNetBackend):
        async def _run_ssh(self, remote_command, *, timeout_seconds=None):
            return RemoteExecutionResult(
                ok=False,
                returncode=-1,
                stdout="",
                stderr="ssh timed out after 120s",
                command=remote_command,
                error="ssh timed out after 120s",
            )

        async def ensure_remote_directory(self, remote_directory):
            return True

    backend = TimeoutBackend(
        RemoteServerConfig(host="login.scnet.cn", username="u"),
        policy=ResourcePolicy(allow_hpc_submit=True),
    )
    spec = RemoteJobSpec(
        application="vasp",
        job_name="x-abcdef12",
        remote_directory="~/photomatagent/x-abcdef12-ts",
        executable="vasp_std",
    )
    with pytest.raises(TimeoutError, match="sbatch client timed out"):
        await backend.submit_script(spec)


# --------------------------------------------------------------------------
# submit-once lifecycle
# --------------------------------------------------------------------------


async def test_submit_once_normal(tmp_path):
    backend = FakeSCNetBackend()
    session = _session(tmp_path, backend)
    input_dir = _input_dir(tmp_path, "relax")
    result = await _submit(session, input_dir)
    assert result.submitted is True
    assert result.duplicate is False
    record = session.registry.get(result.request_id)
    assert record is not None
    assert record.state is JobLifecycleState.SUBMITTED
    assert record.job_id == "1001"
    assert record.remote_directory is not None
    assert record.remote_directory.startswith("~/photomatagent/dme-li-")
    assert record.scientific_validation_state == "passed"
    assert len(backend.submitted_scripts) == 1
    # marker was uploaded into the unique remote directory, inputs too
    remote = backend.remote_files[record.remote_directory]
    assert set(remote) == {"POSCAR", "INCAR", "KPOINTS", MARKER_FILENAME}
    assert json.loads(remote[MARKER_FILENAME])["request_id"] == result.request_id


async def test_submit_once_same_request_never_duplicates(tmp_path):
    backend = FakeSCNetBackend()
    session = _session(tmp_path, backend)
    input_dir = _input_dir(tmp_path)
    first = await _submit(session, input_dir)
    second = await _submit(session, input_dir)
    assert first.request_id == second.request_id
    assert second.duplicate is True
    assert second.submitted is False
    assert len(backend.submitted_scripts) == 1
    assert len(backend.uploaded) == 4  # exactly one upload set


async def test_submit_once_upload_ok_sbatch_failure(tmp_path):
    backend = FakeSCNetBackend(fail_submit_with="sbatch: error: Partition not found")
    session = _session(tmp_path, backend)
    input_dir = _input_dir(tmp_path)
    result = await _submit(session, input_dir)
    assert result.submitted is False
    assert "sbatch failed" in result.error
    record = session.registry.get(result.request_id)
    assert record is not None
    assert record.state is JobLifecycleState.FAILED
    assert record.job_id is None
    assert len(backend.submitted_scripts) == 0  # sbatch never accepted
    assert len(backend.uploaded) == 4  # uploads did happen before sbatch
    # A retry with the same request_id passes remote confirmation first and
    # then submits (no job ever existed).
    backend.fail_submit_with = ""
    retry = await _submit(session, input_dir)
    assert retry.submitted is True
    assert len(backend.submitted_scripts) == 1


async def test_sbatch_client_timeout_requires_reconciliation(tmp_path):
    backend = FakeSCNetBackend(submit_succeeds_but_times_out=True)
    session = _session(tmp_path, backend)
    input_dir = _input_dir(tmp_path)
    result = await _submit(session, input_dir)
    assert result.submitted is False
    assert result.needs_reconciliation is True
    assert "reconciliation" in result.error
    record = session.registry.get(result.request_id)
    assert record is not None
    assert record.state is JobLifecycleState.SUBMITTING
    # The job really exists on the (fake) cluster; a blind second submit must
    # NOT happen.
    second = await _submit(session, input_dir)
    assert second.duplicate is True
    assert len(backend.submitted_scripts) == 1
    assert backend.submitted_job_names  # the cluster-side job exists


async def test_reconcile_adopts_unique_job_after_timeout(tmp_path):
    backend = FakeSCNetBackend(submit_succeeds_but_times_out=True)
    session = _session(tmp_path, backend)
    input_dir = _input_dir(tmp_path)
    result = await _submit(session, input_dir)
    request_id = result.request_id
    assert backend.submitted_job_names  # job 1001 exists remotely
    reconciliation = await session.reconcile(request_id)
    assert reconciliation.outcome == "FOUND_UNIQUE"
    assert reconciliation.adopted_job_id == "1001"
    assert reconciliation.marker_matched is True
    record = session.registry.get(request_id)
    assert record is not None
    assert record.job_id == "1001"
    assert record.state is JobLifecycleState.RUNNING
    # Now a fresh submit_once finds the active job and refuses to duplicate.
    again = await _submit(session, input_dir)
    assert again.duplicate is True
    assert len(backend.submitted_scripts) == 1


async def test_reconcile_multiple_candidates_blocks(tmp_path):
    backend = FakeSCNetBackend(submit_succeeds_but_times_out=True)
    backend.add_ssh_script("squeue -h --name", "1001 RUNNING\n1011 RUNNING")
    session = _session(tmp_path, backend)
    input_dir = _input_dir(tmp_path)
    result = await _submit(session, input_dir)
    request_id = result.request_id
    reconciliation = await session.reconcile(request_id)
    assert reconciliation.outcome == "FOUND_MULTIPLE"
    assert reconciliation.job_ids == ["1001", "1011"]
    record = session.registry.get(request_id)
    assert record is not None
    assert record.state is JobLifecycleState.UNKNOWN_RECONCILIATION_REQUIRED
    assert "multiple candidate jobs" in (record.last_error or "")
    # submit_once respects the trap: no third job.
    again = await _submit(session, input_dir)
    assert again.duplicate is True
    assert again.needs_reconciliation is True
    assert len(backend.submitted_scripts) == 1


async def test_squeue_empty_sacct_has_result(tmp_path):
    backend = FakeSCNetBackend(submit_succeeds_but_times_out=True)
    # squeue: nothing (job finished); sacct: completed record.
    backend.add_ssh_script("squeue -h --name", "")
    backend.add_ssh_script("sacct -n -X --name", "1001 COMPLETED")
    session = _session(tmp_path, backend)
    input_dir = _input_dir(tmp_path)
    result = await _submit(session, input_dir)
    request_id = result.request_id
    reconciliation = await session.reconcile(request_id)
    assert reconciliation.outcome == "FOUND_UNIQUE"
    assert reconciliation.adopted_job_id == "1001"
    assert reconciliation.scheduler_state == "COMPLETED"
    record = session.registry.get(request_id)
    assert record is not None
    assert record.state is JobLifecycleState.COMPLETED


async def test_reconcile_finds_nothing_then_resubmits(tmp_path):
    backend = FakeSCNetBackend()
    session = _session(tmp_path, backend)
    input_dir = _input_dir(tmp_path)
    # Simulate: sbatch client timed out, but the cluster never created a job.
    # An ambiguous record exists with no job_id and no remote directory.
    session.registry.put(
        JobRecord(
            request_id="req-ambiguous",
            canonical_input_hash="h",
            job_name="dme-li-abcdef12",
            workflow_stage="static",
            resource=RESOURCE,
            state=JobLifecycleState.SUBMITTING,
            last_error="sbatch client timed out (artificial)",
        )
    )
    reconciliation = await session.reconcile("req-ambiguous")
    assert reconciliation.outcome == "NOT_FOUND"
    assert reconciliation.can_resubmit is True
    again = await _submit(session, input_dir, request_id="req-ambiguous")
    assert again.submitted is True
    assert len(backend.submitted_scripts) == 1
    assert again.record["retry_count"] == 1


async def test_status_refresh_oom_and_timeout(tmp_path):
    backend = FakeSCNetBackend(scripted_states=[HPCJobState.OUT_OF_MEMORY])
    session = _session(tmp_path, backend)
    result = await _submit(session, _input_dir(tmp_path))
    refresh = await session.refresh_status(result.request_id)
    assert refresh.ok is True
    assert refresh.state == "OUT_OF_MEMORY"
    assert session.registry.get(result.request_id).state is JobLifecycleState.OUT_OF_MEMORY  # type: ignore[union-attr]

    backend2 = FakeSCNetBackend(scripted_states=[HPCJobState.TIMEOUT])
    session2 = _session(tmp_path, backend2)
    result2 = await _submit(session2, _input_dir(tmp_path, "t2"))
    refresh2 = await session2.refresh_status(result2.request_id)
    assert refresh2.state == "TIMEOUT"


async def test_transient_ssh_failure_is_not_job_failure(tmp_path):
    class FlakyBackend(FakeSCNetBackend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.failures_left = 1

        async def job_status(self, job_id):
            if self.failures_left > 0:
                self.failures_left -= 1
                raise OSError("ssh connection reset")
            return await super().job_status(job_id)

    backend = FlakyBackend()
    session = _session(tmp_path, backend)
    result = await _submit(session, _input_dir(tmp_path))
    request_id = result.request_id
    refresh = await session.refresh_status(request_id)
    assert refresh.ok is False
    assert refresh.query_failed is True
    assert "ssh" in refresh.error
    record = session.registry.get(request_id)
    assert record is not None
    assert record.state is JobLifecycleState.SUBMITTED  # NOT misjudged FAILED
    # Once the transient failure clears, polling recovers normally.
    refresh2 = await session.refresh_status(request_id)
    assert refresh2.ok is True
    assert refresh2.state == "COMPLETED"


async def test_distinct_request_ids_get_distinct_remote_dirs(tmp_path):
    backend = FakeSCNetBackend()
    session = _session(tmp_path, backend)
    first = await _submit(session, _input_dir(tmp_path, "a"), job_name="dme-li")
    second = await _submit(
        session, _input_dir(tmp_path, "b"), job_name="tfsi", request_id="explicit-2"
    )
    assert first.request_id != second.request_id
    assert first.record["remote_directory"] != second.record["remote_directory"]
    assert len(backend.submitted_scripts) == 2


async def test_preflight_failure_blocks_upload_and_sbatch(tmp_path):
    backend = FakeSCNetBackend()
    session = _session(tmp_path, backend)
    input_dir = _input_dir(tmp_path)
    gate = _gate(
        passed=False,
        errors=["ELECTRON_PARITY_MISMATCH: odd electron count with ISPIN=1"],
    )
    result = await _submit(session, input_dir, gate=gate)
    assert result.blocked is True
    assert "preflight gate failed" in result.error
    assert backend.uploaded == []
    assert backend.submitted_scripts == {}
    assert backend.remote_files == {}
    record = session.registry.get(result.request_id)
    assert record is not None
    assert record.state is JobLifecycleState.PREPARED
    assert record.scientific_validation_state == "failed"
    # After the gate is fixed the same request may proceed (fresh reconcile).
    ok = await _submit(session, input_dir)
    assert ok.submitted is True


async def test_no_secret_or_potcar_leak(tmp_path):
    backend = FakeSCNetBackend()
    session = _session(tmp_path, backend)
    input_dir = _input_dir(tmp_path)
    # A real POTCAR sitting in the input dir must never be uploaded by the
    # lifecycle and never reach the marker or the registry.
    (input_dir / "POTCAR").write_bytes(
        b"PAW_PBE synthetic pseudopotential secret-content-marker"
    )
    result = await _submit(session, input_dir)
    record = session.registry.get(result.request_id)
    assert record is not None
    remote = backend.remote_files[record.remote_directory or ""]
    assert "POTCAR" not in remote  # POTCAR never leaves the machine
    marker = json.loads(remote[MARKER_FILENAME])
    assert "secret-content-marker" not in json.dumps(marker)
    assert "private_key" not in json.dumps(marker)
    public = json.dumps(record.public_dict())
    assert "secret-content-marker" not in public
    assert ".ssh" not in public
    assert "id_rsa" not in " ".join(backend.uploaded)


async def test_force_new_attempt_derives_a_new_request(tmp_path):
    backend = FakeSCNetBackend(scripted_states=[HPCJobState.COMPLETED])
    session = _session(tmp_path, backend)
    input_dir = _input_dir(tmp_path)
    first = await _submit(session, input_dir)
    await session.refresh_status(first.request_id)  # -> COMPLETED
    plain = await _submit(session, input_dir)
    assert plain.duplicate is True
    assert len(backend.submitted_scripts) == 1
    forced = await _submit(session, input_dir, force_new_attempt=True)
    assert forced.submitted is True
    assert forced.request_id != first.request_id
    assert len(backend.submitted_scripts) == 2
    record = session.registry.get(forced.request_id)
    assert record is not None
    assert record.parent_request_id == first.request_id


# --------------------------------------------------------------------------
# monitoring
# --------------------------------------------------------------------------


async def test_detached_monitor_tracks_terminal_progression(tmp_path):
    backend = FakeSCNetBackend(
        scripted_states=[
            HPCJobState.PENDING,
            HPCJobState.RUNNING,
            HPCJobState.COMPLETED,
        ]
    )
    session = _session(tmp_path, backend)
    result = await _submit(session, _input_dir(tmp_path))
    transitions: list[str] = []

    async def on_transition(request_id, state):
        transitions.append(state)

    monitor = JobMonitor(
        session, poll_interval_seconds=0.01, on_transition=on_transition
    )
    handle = monitor.start(result.request_id)
    try:
        final = await asyncio.wait_for(
            handle.wait_next(), timeout=10
        )
        assert final.state in {"PENDING", "RUNNING"}
        while True:
            snapshot = await asyncio.wait_for(handle.wait_next(), timeout=10)
            if snapshot.state == "COMPLETED":
                break
    finally:
        handle.stop()
    record = session.registry.get(result.request_id)
    assert record is not None
    assert record.state is JobLifecycleState.COMPLETED
    assert "COMPLETED" in transitions
    refresh = await monitor.poll_once(result.request_id)
    assert refresh.request_id == result.request_id


async def test_monitor_refuses_to_overwrite_on_query_errors(tmp_path):
    backend = FakeSCNetBackend()
    session = _session(tmp_path, backend)
    result = await _submit(session, _input_dir(tmp_path))

    async def failing_status(job_id):
        raise OSError("scheduler down")

    backend.job_status = failing_status  # type: ignore[method-assign]
    refresh = await session.refresh_status(result.request_id)
    assert refresh.query_failed is True
    record = session.registry.get(result.request_id)
    assert record is not None
    assert record.state is JobLifecycleState.SUBMITTED  # unchanged


# --------------------------------------------------------------------------
# molecular gate adapter
# --------------------------------------------------------------------------


def test_preflight_gate_adapter():
    failed_report = PreflightReport(
        passed=False,
        errors=[
            {
                "code": "ELECTRON_PARITY_MISMATCH",
                "message": "odd electrons with ISPIN=1",
            }
        ],
    )
    gate = preflight_gate(failed_report, report_path="/tmp/preflight.json")
    assert gate.passed is False
    assert gate.errors == ["odd electrons with ISPIN=1"]
    assert gate.report_path == "/tmp/preflight.json"
    ok_gate = preflight_gate(PreflightReport(passed=True, checks=["A", "B"]))
    assert ok_gate.passed is True


# --------------------------------------------------------------------------
# vasp application default (no dangerous shared directories)
# --------------------------------------------------------------------------


def test_submit_stage_defaults_to_unique_remote_directory():
    import inspect

    from photomatagent.scientific.applications.vasp.application import (
        VaspApplication,
    )

    signature = inspect.signature(VaspApplication.submit_stage)
    assert signature.parameters["unique_remote_directory"].default is True
