"""NamdApplication: Hefei-NAMD preparation and execution adapter.

The first version focuses on the VASP -> Hefei-NAMD artifact bridge
(section 37): validate that a VASP AIMD trajectory plus per-snapshot
WAVECARs satisfy the Hefei-NAMD input contract, prepare the remote job
tree, and (only with explicit authorization and a confirmed module) submit.
No carrier-dynamics evidence is produced without actual NAMD output
(section 36).
"""

from __future__ import annotations

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
        policy: ResourcePolicy | None = None,
    ) -> None:
        self.backend = backend
        self.module_name = module_name
        self.policy = policy or ResourcePolicy.from_environment()

    # -- environment --------------------------------------------------------

    def probe_environment(self) -> dict[str, Any]:
        """Probe the SCNet environment for Hefei-NAMD (read-only)."""
        report: dict[str, Any] = {
            "application": "hefei-namd",
            "backend": getattr(self.backend, "name", "none"),
            "status": "UNCONFIGURED",
            "module": self.module_name,
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
                connection = asyncio.run(self.backend.check_connection())
                report["connection"] = connection
                if connection.get("connected") == "true":
                    probe = asyncio.run(self._probe_module())
                    report.update(probe)
            except Exception as exc:
                report["connection"] = {
                    "connected": "false",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        return report

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
    ) -> dict[str, Any]:
        """Prepare the Hefei-NAMD remote job tree (never fabricates inp)."""
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
                    "poscar": str(snapshot / "POSCAR"),
                    "wavecar": str(snapshot / "WAVECAR"),
                    "wavecar_size_bytes": (
                        (snapshot / "WAVECAR").stat().st_size
                        if (snapshot / "WAVECAR").is_file()
                        else None
                    ),
                }
                for snapshot in snapshot_dirs
            ],
            "runtime_inputs": {
                "inp": "NOT_GENERATED",
                "inicon": "NOT_GENERATED",
                "reason": (
                    "the Hefei-NAMD `inp`/`INICON` format is version-"
                    "dependent; they are generated only after the SCNet "
                    "module has been confirmed (namd.capabilities)"
                ),
            },
            "evidence_scope": (
                "carrier relaxation / nonadiabatic transition / population "
                "dynamics / recombination / lifetime are reported ONLY from "
                "actual NAMD output; nothing is guessed from the software "
                "name"
            ),
        }
        (output / "namd_manifest.json").write_text(
            __import__("json").dumps(manifest, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        return manifest

    def render_slurm(self, *, job_name: str, resource: ResourceRequest) -> str:
        """Render the NAMD submission script (module-gated)."""
        from photomatagent.scientific.remote.scheduler import render_slurm_script

        if not self.module_name:
            raise ValueError(
                "Hefei-NAMD module name is not configured; run "
                "namd.capabilities first"
            )
        return render_slurm_script(
            job_name=job_name,
            resource=resource,
            module_load=self.module_name,
            executable="namd",
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
        remote_directory = f"~/photomatagent/namd/{job_name.replace('/', '-')[:64]}"
        validate_remote_path(remote_directory)
        await self.backend.ensure_remote_directory(remote_directory)
        files = [path for path in root.rglob("*") if path.is_file()]
        await self.backend.upload_files(files, remote_directory)
        script = self.render_slurm(
            job_name=job_name,
            resource=resource or ResourceRequest(
                partition="kshcnormal", nodes=1, tasks_per_node=32, walltime_minutes=1440
            ),
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
                resource=resource or ResourceRequest(),
                executable="namd",
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
        names = ["out.log", "eigenvalues.dat", "inp", "INICON", "populations.dat"]
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
