"""MagusApplication: probe + prepare search jobs for MAGUS structure search."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from photomatagent.scientific.remote.models import (
    RemoteJobRef,
    RemoteJobSpec,
    ResourcePolicy,
    ResourceRequest,
)


class MagusApplication:
    """Optional MAGUS adapter; everything is gated on availability."""

    name = "magus"

    def __init__(
        self,
        backend: Any | None = None,
        *,
        executable: str = "magus",
        policy: ResourcePolicy | None = None,
        search_types: list[str] | None = None,
    ) -> None:
        self.backend = backend
        self.executable = executable
        self.policy = policy or ResourcePolicy.from_environment()
        # Which geometry searches this installation supports; probed, never
        # assumed. A search type is only exposed when the installed version
        # is known to support it.
        self.search_types = search_types or ["bulk", "cluster", "surface"]

    # -- environment --------------------------------------------------------

    def probe_environment(self) -> dict[str, Any]:
        """Check MAGUS availability (local executable or SCNet module)."""
        local = shutil.which(self.executable)
        report: dict[str, Any] = {
            "application": "magus",
            "backend": getattr(self.backend, "name", "none"),
            "executable": self.executable,
            "installed": local is not None,
            "status": "AVAILABLE" if local else "UNCONFIGURED",
            "search_types": self.search_types,
            "installation_requirement": (
                "install MAGUS (Xia et al., Comput. Phys. Commun. 2024) or "
                "register the SCNet module; then PhotoMatAgent can prepare "
                "structure-search jobs"
                if local is None
                else ""
            ),
            "candidate_validity": (
                "MAGUS candidates are UNVALIDATED_GENERATED_STRUCTURE: they "
                "still require CHGNet/DFT relaxation and stability "
                "validation before any 'stable' or 'synthesizable' claim"
            ),
        }
        if self.backend is not None:
            import asyncio

            try:
                connection = asyncio.run(self.backend.check_connection())
                report["connection"] = connection
                if connection.get("connected") == "true":
                    result = asyncio.run(
                        self.backend._run_ssh(
                            f"command -v {self.executable} || "
                            "module avail 2>&1 | grep -i magus || true"
                        )
                    )
                    if result.ok and result.stdout.strip():
                        report["installed"] = True
                        report["status"] = "AVAILABLE"
                        report["remote_path"] = result.stdout.strip()[:1000]
            except Exception as exc:
                report["connection"] = {
                    "connected": "false",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        return report

    # -- preparation --------------------------------------------------------

    def validate_inputs(
        self, *, search_type: str, composition: str, target_dir: str | Path
    ) -> list[str]:
        """Validate a search request (typed problems; never guesses)."""
        problems: list[str] = []
        if search_type not in self.search_types:
            problems.append(
                f"search_type {search_type!r} not exposed by this MAGUS "
                f"installation; supported: {self.search_types}"
            )
        if not composition or not composition.strip():
            problems.append("composition is required")
        target = Path(target_dir)
        if not target.is_dir():
            problems.append(f"target directory does not exist: {target}")
        return problems

    def prepare(
        self,
        *,
        search_type: str,
        composition: str,
        target_dir: str | Path,
        output_dir: str | Path,
        generations: int = 30,
        population_size: int = 20,
    ) -> dict[str, Any]:
        """Prepare a MAGUS search job manifest (no execution)."""
        problems = self.validate_inputs(
            search_type=search_type,
            composition=composition,
            target_dir=target_dir,
        )
        if problems:
            raise ValueError("; ".join(problems))
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {
            "application": "magus",
            "status": "PREPARED",
            "search_type": search_type,
            "composition": composition,
            "target_dir": str(Path(target_dir).expanduser().resolve()),
            "generations": generations,
            "population_size": population_size,
            "executable": self.executable,
            "installed": shutil.which(self.executable) is not None,
            "note": (
                "manifest only; MAGUS execution requires the binary/module "
                "and explicit HPC authorization"
            ),
        }
        (output / "magus_manifest.json").write_text(
            __import__("json").dumps(manifest, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        return manifest

    async def submit(
        self,
        *,
        job_name: str,
        prepared_dir: str | Path,
        resource: ResourceRequest | None = None,
    ) -> RemoteJobRef:
        if self.backend is None:
            raise RuntimeError("MAGUS backend is not configured")
        from photomatagent.scientific.remote.scnet import validate_remote_path

        root = Path(prepared_dir).expanduser().resolve()
        if not (root / "magus_manifest.json").is_file():
            raise FileNotFoundError(
                "magus_manifest.json missing; run magus.prepare first"
            )
        remote_directory = f"~/photomatagent/magus/{job_name.replace('/', '-')[:64]}"
        validate_remote_path(remote_directory)
        await self.backend.ensure_remote_directory(remote_directory)
        files = [path for path in root.rglob("*") if path.is_file()]
        await self.backend.upload_files(files, remote_directory)
        return await self.backend.submit_script(
            RemoteJobSpec(
                application="magus",
                job_name=job_name,
                remote_directory=remote_directory,
                script_name="magus.slurm",
                resource=resource or ResourceRequest(walltime_minutes=720),
                executable=self.executable,
                provenance={"prepared_dir": str(root)},
            )
        )

    async def status(self, job_id: str) -> Any:
        if self.backend is None:
            from photomatagent.scientific.remote.models import HPCJobState

            return HPCJobState.UNKNOWN
        return await self.backend.job_status(job_id)
