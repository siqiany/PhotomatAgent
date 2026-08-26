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
    remote_mkdir_command,
    remote_read_command,
    remote_submit_command,
    sanitize_job_name,
    slurm_state_to_hpc_state,
    validate_job_id,
)
from photomatagent.scientific.remote.scnet import (
    SCNetBackend,
    RemoteSubmissionBlocked,
    validate_remote_path,
)


def test_ssh_and_scp_disable_locale_forwarding_to_login_shell():
    """SCNet login shells hang on forwarded LC_ALL/LANG; never send them."""
    backend = SCNetBackend(RemoteServerConfig(host="login.scnet.cn", username="u"))
    for args in (backend._ssh_base(), backend._scp_base()):
        assert "SendEnv=-*" in args
        assert "SetEnv=LC_ALL=C LANG=C" in args


@pytest.mark.asyncio
async def test_check_connection_does_not_cache_failures():
    class FlakyBackend(SCNetBackend):
        def __init__(self) -> None:
            super().__init__(RemoteServerConfig(host="login.scnet.cn", username="u"))
            self.attempts = 0

        async def _check_connection_uncached(self) -> dict[str, str]:
            self.attempts += 1
            if self.attempts == 1:
                return {
                    "connected": "false",
                    "error": "ssh timed out after 40s",
                    "host": "",
                    "sbatch": "",
                    "squeue": "",
                }
            return {
                "connected": "true",
                "host": "login.scnet.cn",
                "sbatch": "/usr/bin/sbatch",
                "squeue": "/usr/bin/squeue",
            }

    backend = FlakyBackend()
    first = await backend.check_connection()
    assert first["connected"] == "false"
    second = await backend.check_connection()
    assert second["connected"] == "true"
    assert backend.attempts == 2


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


def test_scnet_backend_reuses_one_controlmaster_socket():
    config = RemoteServerConfig(
        host="gateway.example",
        username="user",
        port=2222,
        private_key_path="/keys/id",
    )
    first = SCNetBackend(config)
    second = SCNetBackend(config)
    assert first.control_path == second.control_path
    assert "ControlMaster=auto" in first._ssh_base()
    assert f"ControlPath={first.control_path}" in first._ssh_base()
    assert f"ControlPath={first.control_path}" in first._scp_base()


def test_validate_remote_path():
    assert validate_remote_path("~/jobs/abc") == "~/jobs/abc"
    assert validate_remote_path("/home/user/jobs") == "/home/user/jobs"
    with pytest.raises(ValueError):
        validate_remote_path("relative/path")
    with pytest.raises(ValueError):
        validate_remote_path("~/jobs/$(rm -rf /)")
    with pytest.raises(ValueError):
        validate_remote_path("~/a//b")


def test_remote_commands_expand_tilde_via_home():
    assert remote_mkdir_command("~/jobs/test") == 'mkdir -p "$HOME"/jobs/test'
    assert remote_submit_command("~/jobs/test", "run.slurm") == (
        'cd "$HOME"/jobs/test && sbatch run.slurm'
    )
    assert remote_read_command("~/jobs/test", "1.out", 100) == (
        'tail -c 100 "$HOME"/jobs/test/1.out'
    )


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


def test_fake_backend_upload_tree_preserves_snapshot_paths(tmp_path):
    backend = FakeSCNetBackend()
    tree = tmp_path / "prepared"
    (tree / "run" / "0001").mkdir(parents=True)
    (tree / "run" / "0002").mkdir(parents=True)
    (tree / "run" / "0001" / "WAVECAR").write_bytes(b"one")
    (tree / "run" / "0002" / "WAVECAR").write_bytes(b"two")

    async def scenario():
        names = await backend.upload_tree(tree, "~/jobs/namd")
        assert names == ["run/0001/WAVECAR", "run/0002/WAVECAR"]
        assert backend.remote_files["~/jobs/namd"]["run/0001/WAVECAR"] == b"one"
        assert backend.remote_files["~/jobs/namd"]["run/0002/WAVECAR"] == b"two"

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
