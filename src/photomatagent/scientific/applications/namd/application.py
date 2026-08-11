"""NamdApplication: Hefei-NAMD preparation and execution adapter.

The first version focuses on the VASP -> Hefei-NAMD artifact bridge
(section 37): validate that a VASP AIMD trajectory plus per-snapshot
WAVECARs satisfy the Hefei-NAMD input contract, prepare the remote job
tree, and (only with explicit authorization and a confirmed module) submit.
No carrier-dynamics evidence is produced without actual NAMD output
(section 36).
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
from pathlib import Path
from typing import Any

from photomatagent.scientific.remote.models import (
    RemoteJobRef,
    RemoteJobSpec,
    ResourcePolicy,
    ResourceRequest,
)

NAMD_REQUIRED_TRAJECTORY = ("POSCAR", "XDATCAR", "OUTCAR")


class NamdApplication:
    """Hefei-NAMD adapter: probe, validate, prepare, submit, collect."""

    name = "namd"

    def __init__(
        self,
        backend: Any | None = None,
        *,
        module_name: str = "",
        executable: str = "namd",
        env_script: str = "",
        remote_root: str = "~/photomatagent",
        policy: ResourcePolicy | None = None,
    ) -> None:
        self.backend = backend
        self.module_name = module_name
        self.executable = executable
        self.env_script = env_script
        self.remote_root = remote_root.rstrip("/")
        self.policy = policy or ResourcePolicy.from_environment()

    # -- environment --------------------------------------------------------

    def probe_environment(self) -> dict[str, Any]:
        """Probe the SCNet environment for Hefei-NAMD (read-only)."""
        report: dict[str, Any] = {
            "application": "hefei-namd",
            "backend": getattr(self.backend, "name", "none"),
            "status": "UNCONFIGURED",
            "module": self.module_name,
            "executable": self.executable,
            "env_script_configured": bool(self.env_script),
            "detail": (
                "Hefei-NAMD module not confirmed; set the SCNet module name "
                "and verify `module avail` on the login node"
            ),
            "supported_workflow": (
                "VASP AIMD trajectory + per-snapshot WAVECAR -> Hefei-NAMD "
                "preparation; production runs gated on module confirmation"
            ),
            "required_vasp_artifacts": [
                "reference POSCAR",
                "XDATCAR (MD trajectory, every frame: NBLOCK=1)",
                "OUTCAR (metadata)",
                "one WAVECAR per snapshot, identical size",
            ],
            "requirements": [
                "same cell / ENCUT / NBANDS / k-mesh / spin across all "
                "snapshots (WAVECAR size consistency)",
                "enough empty bands for the excited-state window",
                "Gamma-only k-points only if the installed Hefei-NAMD "
                "supports the reduced WAVECAR format",
            ],
        }
        if self.backend is not None:
            import asyncio

            try:
                asyncio.get_running_loop()
            except RuntimeError:
                try:
                    connection = asyncio.run(self.backend.check_connection())
                    report["connection"] = connection
                    if connection.get("connected") == "true":
                        software = asyncio.run(
                            self.backend.probe_module(
                                self.module_name, self.executable
                            )
                        )
                        report["software"] = software
                        if software.get("available") == "true":
                            report.update(
                                status="AVAILABLE",
                                detail=(
                                    "configured Hefei-NAMD module and "
                                    "executable are available"
                                ),
                            )
                except Exception as exc:
                    report["connection"] = {
                        "connected": "false",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            else:
                report["connection"] = {
                    "connected": "unknown",
                    "error": "use probe_environment_async inside an event loop",
                }
        return report

    async def probe_environment_async(self) -> dict[str, Any]:
        """Async probe for MCP/CLI callers already running an event loop."""
        report = self._base_probe_report()
        if self.backend is None:
            return report
        report["backend"] = getattr(self.backend, "name", "none")
        connection = await self.backend.check_connection()
        report["connection"] = connection
        if connection.get("connected") == "true":
            report["available_partitions"] = await self.backend.available_partitions()
            software = await self.backend.probe_module(
                self.module_name, self.executable
            )
            report["software"] = software
            if software.get("available") == "true":
                report.update(
                    status="AVAILABLE",
                    detail="configured Hefei-NAMD module and executable are available",
                )
        return report

    def _base_probe_report(self) -> dict[str, Any]:
        """Build the static portion without starting nested event loops."""
        backend = self.backend
        self.backend = None
        try:
            return self.probe_environment()
        finally:
            self.backend = backend

    async def _probe_module(self) -> dict[str, Any]:
        """Query `module avail` for the configured Hefei-NAMD module."""
        assert self.backend is not None
        result = await self.backend._run_ssh(
            "module avail 2>&1 | grep -i -E 'namd|hefei' || true"
        )
        found = result.stdout.strip() if result.ok else ""
        if found:
            return {
                "status": "AVAILABLE",
                "module": self.module_name or "auto-detected",
                "module_avail": found[:2000],
                "detail": (
                    "Hefei-NAMD module found on SCNet; confirm exact module "
                    "name before production submissions"
                ),
            }
        return {
            "status": "UNCONFIGURED",
            "module_avail": "",
            "detail": (
                "no Hefei-NAMD module found via `module avail`; register the "
                "module name in the SCNet config to enable preparation"
            ),
        }

    # -- validation ---------------------------------------------------------

    def validate_inputs(self, trajectory_dir: str | Path) -> list[str]:
        """Validate a VASP AIMD trajectory tree against the NAMD contract."""
        root = Path(trajectory_dir).expanduser().resolve()
        problems: list[str] = []
        missing = [
            name
            for name in NAMD_REQUIRED_TRAJECTORY
            if not (root / name).is_file()
        ]
        if missing:
            problems.append(
                "missing trajectory files: " + ", ".join(missing)
            )
            return problems
        snapshot_dirs = sorted(
            path
            for path in root.iterdir()
            if path.is_dir() and path.name.isdigit()
        )
        if not snapshot_dirs:
            problems.append(
                "no per-snapshot directories (0001/, 0002/, ...) found; "
                "Hefei-NAMD needs one WAVECAR per snapshot"
            )
            return problems
        wavecar_sizes: set[int] = set()
        for snapshot in snapshot_dirs:
            for name in ("POSCAR", "WAVECAR", "OUTCAR"):
                if not (snapshot / name).is_file():
                    problems.append(
                        f"snapshot {snapshot.name}: missing {name}"
                    )
            wavecar = snapshot / "WAVECAR"
            if wavecar.is_file():
                wavecar_sizes.add(wavecar.stat().st_size)
        if len(wavecar_sizes) > 1:
            problems.append(
                "WAVECAR sizes differ across snapshots "
                f"({sorted(wavecar_sizes)}); Hefei-NAMD requires identical "
                "wavefunction size (same cell/ENCUT/NBANDS/k-mesh/spin)"
            )
        return problems

    # -- preparation --------------------------------------------------------

    def prepare(
        self,
        *,
        trajectory_dir: str | Path,
        output_dir: str | Path,
        snapshot_pattern: str = "{n:04d}",
        inp_path: str | Path | None = None,
        inicon_path: str | Path | None = None,
        parameters: dict[str, Any] | None = None,
        initial_conditions: list[list[int]] | None = None,
    ) -> dict[str, Any]:
        """Prepare a runnable tree from supplied or explicitly parameterized inputs."""
        problems = self.validate_inputs(trajectory_dir)
        if problems:
            raise ValueError("; ".join(problems))
        root = Path(trajectory_dir).expanduser().resolve()
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        snapshot_dirs = sorted(
            path
            for path in root.iterdir()
            if path.is_dir() and path.name.isdigit()
        )
        run_dir = output / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        for name in NAMD_REQUIRED_TRAJECTORY:
            shutil.copy2(root / name, output / name)
        for snapshot in snapshot_dirs:
            target = run_dir / snapshot.name
            target.mkdir(parents=True, exist_ok=True)
            for name in ("POSCAR", "WAVECAR", "OUTCAR"):
                shutil.copy2(snapshot / name, target / name)

        source_inp = Path(inp_path).expanduser().resolve() if inp_path else root / "inp"
        source_inicon = (
            Path(inicon_path).expanduser().resolve()
            if inicon_path
            else root / "INICON"
        )
        if parameters is not None or initial_conditions is not None:
            if parameters is None or initial_conditions is None:
                raise ValueError(
                    "parameters and initial_conditions must be provided together"
                )
            self._write_runtime_inputs(
                output, parameters=parameters, initial_conditions=initial_conditions
            )
        else:
            if source_inp.is_file():
                shutil.copy2(source_inp, output / "inp")
            if source_inicon.is_file():
                shutil.copy2(source_inicon, output / "INICON")
        runnable = (output / "inp").is_file() and (output / "INICON").is_file()
        manifest: dict[str, Any] = {
            "application": "hefei-namd",
            "status": "PREPARED",
            "input_tree": str(output),
            "reference_poscar": str(root / "POSCAR"),
            "trajectory": str(root / "XDATCAR"),
            "outcar": str(root / "OUTCAR"),
            "snapshots": [
                {
                    "name": snapshot.name,
                    "poscar": str(run_dir / snapshot.name / "POSCAR"),
                    "wavecar": str(run_dir / snapshot.name / "WAVECAR"),
                    "wavecar_size_bytes": (
                        (snapshot / "WAVECAR").stat().st_size
                        if (snapshot / "WAVECAR").is_file()
                        else None
                    ),
                }
                for snapshot in snapshot_dirs
            ],
            "runtime_inputs": {
                "inp": str(output / "inp") if (output / "inp").is_file() else "NOT_GENERATED",
                "inicon": str(output / "INICON") if (output / "INICON").is_file() else "NOT_GENERATED",
                "reason": (
                    "provide version-matched inp/INICON files, or explicit "
                    "parameters + initial_conditions"
                    if not runnable
                    else "runtime inputs are present and will be uploaded"
                ),
            },
            "runnable": runnable,
            "evidence_scope": (
                "carrier relaxation / nonadiabatic transition / population "
                "dynamics / recombination / lifetime are reported ONLY from "
                "actual NAMD output; nothing is guessed from the software "
                "name"
            ),
        }
        (output / "namd_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        return manifest

    @staticmethod
    def _write_runtime_inputs(
        output: Path,
        *,
        parameters: dict[str, Any],
        initial_conditions: list[list[int]],
    ) -> None:
        required = (
            "BMIN", "BMAX", "NBANDS", "NSW", "POTIM", "TEMP", "NSAMPLE",
            "NAMDTIME", "NELM", "NTRAJ", "LHOLE",
        )
        normalized = {str(key).upper(): value for key, value in parameters.items()}
        missing = [key for key in required if key not in normalized]
        if missing:
            raise ValueError("missing Hefei-NAMD parameters: " + ", ".join(missing))
        bmin, bmax, nbands = (int(normalized[key]) for key in ("BMIN", "BMAX", "NBANDS"))
        nsw = int(normalized["NSW"])
        nsample = int(normalized["NSAMPLE"])
        namdtime = int(normalized["NAMDTIME"])
        if not (1 <= bmin <= bmax <= nbands):
            raise ValueError("require 1 <= BMIN <= BMAX <= NBANDS")
        if nsample != len(initial_conditions):
            raise ValueError("NSAMPLE must equal the number of initial_conditions")
        for condition in initial_conditions:
            if len(condition) != 2:
                raise ValueError("each initial condition must be [start_step, band]")
            start, band = int(condition[0]), int(condition[1])
            if start < 1 or start + namdtime > nsw:
                raise ValueError("initial start_step + NAMDTIME must not exceed NSW")
            if not bmin <= band <= bmax:
                raise ValueError("initial band must be within [BMIN, BMAX]")
        lhole = normalized["LHOLE"]
        lhole_text = ".TRUE." if str(lhole).strip().lower() in {"1", "true", ".true.", "yes"} else ".FALSE."
        lshp_text = (
            ".TRUE."
            if str(normalized.get("LSHP", True)).strip().lower()
            in {"1", "true", ".true.", "yes"}
            else ".FALSE."
        )
        lcpext_text = (
            ".TRUE."
            if str(normalized.get("LCPEXT", False)).strip().lower()
            in {"1", "true", ".true.", "yes"}
            else ".FALSE."
        )
        inp = (
            "&NAMDPARA\n"
            f"  BMIN       = {bmin}\n  BMAX       = {bmax}\n  NBANDS     = {nbands}\n\n"
            f"  NSW        = {nsw}\n  POTIM      = {float(normalized['POTIM']):g}\n"
            f"  TEMP       = {float(normalized['TEMP']):g}\n\n"
            f"  NSAMPLE    = {nsample}\n  NAMDTIME   = {namdtime}\n"
            f"  NELM       = {int(normalized['NELM'])}\n  NTRAJ      = {int(normalized['NTRAJ'])}\n"
            f"  LHOLE      = {lhole_text}\n  LSHP       = {lshp_text}\n"
            f"  LCPEXT     = {lcpext_text}\n\n  RUNDIR     = \"./run/\"\n"
            "  TBINIT     = \"INICON\"\n/\n"
        )
        (output / "inp").write_text(inp, encoding="utf-8")
        (output / "INICON").write_text(
            "".join(f"{int(row[0]):6d} {int(row[1]):6d}\n" for row in initial_conditions),
            encoding="utf-8",
        )

    def render_slurm(self, *, job_name: str, resource: ResourceRequest) -> str:
        """Render the NAMD submission script (module-gated)."""
        from photomatagent.scientific.remote.scheduler import render_slurm_script

        if not self.module_name and not self.env_script:
            raise ValueError(
                "Hefei-NAMD module name is not configured; run "
                "namd.capabilities first"
            )
        preamble = ""
        if self.env_script:
            from photomatagent.scientific.remote.scnet import validate_remote_path

            validate_remote_path(self.env_script, allow_tilde=False)
            preamble = (
                "set +e\n"
                f"source {shlex.quote(self.env_script)}\n"
                "set -e\n"
                f"command -v {shlex.quote(self.executable)} >/dev/null"
            )
        return render_slurm_script(
            job_name=job_name,
            resource=resource,
            module_load="" if self.env_script else self.module_name,
            executable=self.executable,
            preamble=preamble,
            launcher="",
        )

    # -- submit / status / collect ------------------------------------------

    async def submit(
        self,
        *,
        job_name: str,
        prepared_dir: str | Path,
        resource: ResourceRequest | None = None,
    ) -> RemoteJobRef:
        """Upload the prepared NAMD tree and submit (authorization-gated)."""
        if self.backend is None:
            raise RuntimeError("NAMD backend is not configured")
        from photomatagent.scientific.remote.scnet import validate_remote_path

        root = Path(prepared_dir).expanduser().resolve()
        manifest_path = root / "namd_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("namd_manifest.json missing; run namd.prepare")
        if not self.module_name:
            raise ValueError(
                "Hefei-NAMD module name is not configured; run "
                "namd.capabilities first"
            )
        if not (root / "inp").is_file() or not (root / "INICON").is_file():
            raise ValueError(
                "prepared tree is not runnable: inp and INICON are required; "
                "rerun namd.prepare with files or explicit parameters"
            )
        safe_name = job_name.replace("/", "-")[:64] or "namd"
        remote_directory = f"{self.remote_root}/namd/{safe_name}"
        validate_remote_path(remote_directory)
        await self.backend.upload_tree(root, remote_directory)
        request = resource or ResourceRequest(
            partition=os.environ.get("SCNET_PARTITION", "normal"),
            nodes=1,
            tasks_per_node=32,
            walltime_minutes=720,
        )
        script = self.render_slurm(
            job_name=job_name,
            resource=request,
        )
        script_path = root / "namd.slurm"
        script_path.write_text(script, encoding="utf-8")
        await self.backend.upload_files([script_path], remote_directory)
        return await self.backend.submit_script(
            RemoteJobSpec(
                application="hefei-namd",
                job_name=job_name,
                remote_directory=remote_directory,
                script_name="namd.slurm",
                resource=request,
                executable=self.executable,
                module_load=self.module_name,
                provenance={"prepared_dir": str(root)},
            )
        )

    async def status(self, job_id: str) -> Any:
        if self.backend is None:
            from photomatagent.scientific.remote.models import HPCJobState

            return HPCJobState.UNKNOWN
        return await self.backend.job_status(job_id)

    async def collect(
        self, *, job_ref: RemoteJobRef, local_dir: str | Path
    ) -> dict[str, Any]:
        """Download NAMD outputs; evidence only from real output files."""
        if self.backend is None:
            raise RuntimeError("NAMD backend is not configured")
        local = Path(local_dir).expanduser().resolve()
        local.mkdir(parents=True, exist_ok=True)
        artifacts = await self.backend.list_remote_artifacts(job_ref.remote_directory)
        names = [
            item.name.lstrip("./")
            for item in artifacts
            if Path(item.name).name in {"inp", "INICON", "NATXT", "EIGTXT", "COUPCAR"}
            or Path(item.name).name.startswith(("PSICT.", "SHPROP."))
        ][:500]
        downloaded = await self.backend.download_files(
            job_ref.remote_directory, names, local
        )
        if not downloaded:
            return {
                "job_id": job_ref.job_id,
                "downloaded": [],
                "evidence_available": False,
                "note": (
                    "no Hefei-NAMD output files found; no carrier-dynamics "
                    "evidence can be produced"
                ),
            }
        return {
            "job_id": job_ref.job_id,
            "downloaded": [path.name for path in downloaded],
            "evidence_available": True,
            "note": (
                "output files collected; interpretation (population "
                "dynamics, lifetimes) requires the NAMD analysis step and "
                "is not fabricated"
            ),
        }


def default_namd_application() -> NamdApplication | None:
    """Build the Hefei-NAMD adapter from the same SCNet environment as VASP."""
    from photomatagent.scientific.remote.models import RemoteServerConfig
    from photomatagent.scientific.remote.scnet import SCNetBackend

    host = os.environ.get("SCNET_HOST", "").strip()
    username = os.environ.get("SCNET_USERNAME", "").strip()
    if not host or not username:
        return None
    config = RemoteServerConfig(
        host=host,
        username=username,
        port=int(os.environ.get("SCNET_PORT", "22") or "22"),
        private_key_path=os.environ.get("SCNET_PRIVATE_KEY_PATH", "").strip(),
        remote_root=os.environ.get("SCNET_REMOTE_ROOT", "~/photomatagent").strip()
        or "~/photomatagent",
        connect_timeout_seconds=float(
            os.environ.get("SCNET_CONNECT_TIMEOUT_SECONDS", "20") or "20"
        ),
        transfer_timeout_seconds=float(
            os.environ.get("SCNET_TRANSFER_TIMEOUT_SECONDS", "3600") or "3600"
        ),
    )
    return NamdApplication(
        SCNetBackend(config),
        module_name=os.environ.get("SCNET_NAMD_MODULE", "").strip(),
        executable=os.environ.get("SCNET_NAMD_EXECUTABLE", "namd").strip()
        or "namd",
        env_script=os.environ.get("SCNET_NAMD_ENV_SCRIPT", "").strip(),
        remote_root=config.remote_root,
    )
