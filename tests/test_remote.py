"""Offline tests for the SCNet remote compute package (no SSH/network)."""

from __future__ import annotations

import asyncio

import pytest

from photomatagent.scientific.remote.fake import FakeSCNetBackend
from photomatagent.scientific.remote.models import (
    HPCJobState,
    RemoteJobSpec,
    RemoteServerConfig,
    ResourcePolicy,
    ResourceRequest,
)
from photomatagent.scientific.remote.scheduler import (
    parse_sbatch_job_id,
    render_slurm_script,
    sanitize_job_name,
    slurm_state_to_hpc_state,
    validate_job_id,
)
from photomatagent.scientific.remote.scnet import (
    RemoteSubmissionBlocked,
    validate_remote_path,
)


# -- scheduler ----------------------------------------------------------------


def test_parse_sbatch_job_id():
    assert parse_sbatch_job_id("Submitted batch job 12345") == "12345"
    assert parse_sbatch_job_id("Submitted batch job 12345\n") == "12345"
    assert parse_sbatch_job_id("nothing here") is None


def test_validate_job_id():
    validate_job_id("0")
    validate_job_id("1234567890")
    with pytest.raises(ValueError):
        validate_job_id("12; rm -rf /")
    with pytest.raises(ValueError):
        validate_job_id("-1")
    with pytest.raises(ValueError):
        validate_job_id("")


def test_slurm_state_mapping():
    assert slurm_state_to_hpc_state("PENDING") is HPCJobState.PENDING
    assert slurm_state_to_hpc_state("PD") is HPCJobState.PENDING
    assert slurm_state_to_hpc_state("RUNNING") is HPCJobState.RUNNING
    assert slurm_state_to_hpc_state("COMPLETED+") is HPCJobState.COMPLETED
    assert slurm_state_to_hpc_state("cd") is HPCJobState.COMPLETED
    assert slurm_state_to_hpc_state("FAILED") is HPCJobState.FAILED
    assert slurm_state_to_hpc_state("CANCELLED") is HPCJobState.CANCELLED
    assert slurm_state_to_hpc_state("TIMEOUT") is HPCJobState.TIMEOUT
    assert slurm_state_to_hpc_state("OUT_OF_MEMORY") is HPCJobState.OUT_OF_MEMORY
    assert slurm_state_to_hpc_state("NODE_FAIL") is HPCJobState.NODE_FAIL
    assert slurm_state_to_hpc_state("GARBAGE") is HPCJobState.UNKNOWN
    # Slurm COMPLETED is not scientific validity: only the scheduler state.
    assert HPCJobState.COMPLETED.succeeded is True
    assert HPCJobState.COMPLETED.terminal is True
    assert HPCJobState.RUNNING.terminal is False


def test_render_slurm_script_deterministic_and_safe():
    script = render_slurm_script(
        job_name="HgTe-test",
        resource=ResourceRequest(partition="kshcnormal", nodes=1, tasks_per_node=32),
        module_load="vasp-6.4.2",
        executable="vasp_std",
    )
    assert "#SBATCH -J HgTe-test" in script
    assert "#SBATCH -p kshcnormal" in script
    assert "module load vasp-6.4.2" in script
    assert "srun --mpi=pmi2 vasp_std" in script
    # deterministic
    assert script == render_slurm_script(
        job_name="HgTe-test",
        resource=ResourceRequest(partition="kshcnormal", nodes=1, tasks_per_node=32),
        module_load="vasp-6.4.2",
        executable="vasp_std",
    )


def test_render_slurm_script_rejects_injection():
    with pytest.raises(ValueError):
        render_slurm_script(
            job_name="safe",
            resource=ResourceRequest(),
            executable="vasp_std; rm -rf ~",
        )
    with pytest.raises(ValueError):
        render_slurm_script(
            job_name="safe",
            resource=ResourceRequest(),
            module_load="vasp\nmodule load evil",
            executable="vasp_std",
        )


def test_sanitize_job_name():
    assert sanitize_job_name("HgTe/2026-08-10") == "HgTe-2026-08-10"
    assert sanitize_job_name("") == "job"


# -- models -------------------------------------------------------------------


def test_resource_policy_blocks_by_default():
    policy = ResourcePolicy(allow_hpc_submit=False)
    violations = policy.violations(ResourceRequest(nodes=1))
    assert any("disabled" in item for item in violations)


def test_resource_policy_enforces_caps():
    policy = ResourcePolicy(
        allow_hpc_submit=True,
        max_nodes=1,
        max_tasks_per_node=32,
        max_walltime_minutes=60,
        allowed_partitions=["kshcnormal"],
    )
    problems = policy.violations(
        ResourceRequest(
            partition="bigmem", nodes=4, tasks_per_node=128, walltime_minutes=1440
        )
    )
    assert len(problems) >= 4
    assert policy.violations(
        ResourceRequest(
            partition="kshcnormal", nodes=1, tasks_per_node=16, walltime_minutes=30
        )
    ) == []


def test_server_config_never_exposes_private_key():
    config = RemoteServerConfig(
        host="login.scnet.cn",
        username="user",
        private_key_path="/home/user/.ssh/id_scnet",
    )
    public = config.public_dict()
    assert "private_key_path" not in public
    assert public["private_key_configured"] is True
    assert "id_scnet" not in str(public)
    redacted = config.redacted()
    assert redacted.private_key_path == ""


def test_validate_remote_path():
    assert validate_remote_path("~/jobs/abc") == "~/jobs/abc"
    assert validate_remote_path("/home/user/jobs") == "/home/user/jobs"
    with pytest.raises(ValueError):
        validate_remote_path("relative/path")
    with pytest.raises(ValueError):
        validate_remote_path("~/jobs/$(rm -rf /)")
    with pytest.raises(ValueError):
        validate_remote_path("~/a//b")


# -- fake backend -------------------------------------------------------------


def _spec(**kwargs) -> RemoteJobSpec:
    base = dict(
        application="vasp",
        job_name="test",
        remote_directory="~/jobs/test",
        executable="vasp_std",
    )
    base.update(kwargs)
    return RemoteJobSpec(**base)


def test_fake_backend_submit_status_complete():
    backend = FakeSCNetBackend()

    async def scenario():
        ref = await backend.submit_script(_spec())
        assert ref.job_id.isdigit()
        assert ref.state is HPCJobState.SUBMITTED
        assert await backend.job_status(ref.job_id) is HPCJobState.COMPLETED
        info = await backend.check_connection()
        assert info["slurm_ready"] == "true"

    asyncio.run(scenario())


def test_fake_backend_state_progression():
    backend = FakeSCNetBackend(
        scripted_states=[
            HPCJobState.PENDING,
            HPCJobState.RUNNING,
            HPCJobState.FAILED,
        ]
    )

    async def scenario():
        ref = await backend.submit_script(_spec())
        assert await backend.job_status(ref.job_id) is HPCJobState.PENDING
        assert await backend.job_status(ref.job_id) is HPCJobState.RUNNING
        assert await backend.job_status(ref.job_id) is HPCJobState.FAILED
        assert (await backend.job_status(ref.job_id)) is HPCJobState.FAILED  # terminal

    asyncio.run(scenario())


def test_fake_backend_cancel():
    backend = FakeSCNetBackend(scripted_states=[HPCJobState.RUNNING])

    async def scenario():
        ref = await backend.submit_script(_spec())
        await backend.cancel_job(ref.job_id)
        assert await backend.job_status(ref.job_id) is HPCJobState.CANCELLED

    asyncio.run(scenario())


def test_fake_backend_upload_download(tmp_path):
    backend = FakeSCNetBackend()
    local = tmp_path / "INCAR"
    local.write_text("ENCUT = 520\n", encoding="utf-8")

    async def scenario():
        await backend.upload_files([local], "~/jobs/test")
        downloaded = await backend.download_file(
            "~/jobs/test", "INCAR", tmp_path / "results"
        )
        assert downloaded is not None
        assert downloaded.read_text(encoding="utf-8") == "ENCUT = 520\n"
        missing = await backend.download_file("~/jobs/test", "WAVECAR", tmp_path)
        assert missing is None
        artifacts = await backend.list_remote_artifacts("~/jobs/test")
        assert [artifact.name for artifact in artifacts] == ["INCAR"]

    asyncio.run(scenario())


def test_fake_backend_submit_blocked_by_policy():
    backend = FakeSCNetBackend(
        policy=ResourcePolicy(allow_hpc_submit=False, max_nodes=1)
    )

    async def scenario():
        with pytest.raises(RemoteSubmissionBlocked):
            await backend.submit_script(_spec())

    asyncio.run(scenario())


def test_fake_backend_submit_failure():
    backend = FakeSCNetBackend(fail_submit_with="sbatch: no partition found")

    async def scenario():
        with pytest.raises(RuntimeError, match="no partition"):
            await backend.submit_script(_spec())

    asyncio.run(scenario())


def test_fake_backend_out_of_memory_state():
    backend = FakeSCNetBackend(scripted_states=[HPCJobState.OUT_OF_MEMORY])

    async def scenario():
        ref = await backend.submit_script(_spec())
        assert await backend.job_status(ref.job_id) is HPCJobState.OUT_OF_MEMORY

    asyncio.run(scenario())
