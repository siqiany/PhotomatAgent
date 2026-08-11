"""MagusApplication: SCNet-backed MAGUS structure search adapter.

Sprint 4 lifecycle: probe -> prepare_generate/prepare_search -> submit ->
status -> collect -> inspect_results. The application never fabricates
scientific results: candidates are UNVALIDATED_GENERATED_STRUCTURE until
an internal calculator (e.g. VASP) has actually evaluated them, and even
then the result is bounded to what the artifact contains.

No I/O happens in the constructor: SSH/Slurm are only touched by
``probe_environment_async`` / ``submit`` / ``status`` / ``collect``.
"""

from __future__ import annotations

import json
import os
import shlex
import uuid
from pathlib import Path
from typing import Any

from photomatagent.scientific.applications.magus.models import (
    MagusExecutionConfig,
    MagusGenerateRequest,
    MagusPseudopotentialRequirement,
    MagusSearchRequest,
    SUPPORTED_STRUCTURE_TYPES,
)
from photomatagent.scientific.applications.magus.probe import (
    parse_checkpack_calculators,
    parse_example_dirs,
    parse_example_structure_types,
    parse_magus_help_commands,
    parse_magus_version,
)
from photomatagent.scientific.applications.magus.render import (
    MAGUS_VASP_INCAR,
    magus_arguments,
    manifest_input_hash,
    render_generate_input,
    render_magus_slurm,
    render_search_input,
)
from photomatagent.scientific.applications.vasp.psp import (
    remote_potcar_check,
    resolve_remote_psp_library,
)
from photomatagent.scientific.remote.models import (
    HPCJobState,
    RemoteJobRef,
    RemoteJobSpec,
    ResourcePolicy,
    ResourceRequest,
)
from photomatagent.scientific.remote.scnet import (
    SCNetBackend,
    validate_remote_path,
)

MAGUS_JOB_FILES = ("input.yaml", "magus.slurm")
_TEXT_ARTIFACTS = {
    "input.yaml",
    "magus.slurm",
    "photomat_manifest.json",
    "magus_manifest.json",
    "log.txt",
    "summary",
    "DONE",
    "ERROR",
    "out",
}
_MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024
_MAX_DOWNLOAD_FILES = 200


class MagusUnconfiguredError(RuntimeError):
    """No SCNet backend configured (UNCONFIGURED)."""


class MagusDependencyError(RuntimeError):
    """MAGUS root or executable missing on the remote side (MISSING_DEPENDENCY)."""


class MagusPrerequisiteError(RuntimeError):
    """A required precondition is missing (MISSING_PREREQUISITE)."""


class MagusPseudopotentialMissingError(RuntimeError):
    """Required POTCAR setups are missing remotely (MISSING_PSEUDOPOTENTIALS)."""

    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        super().__init__("missing pseudopotentials: " + ", ".join(missing))


class MagusSubmissionBlockedError(RuntimeError):
    """HPC submission refused by the resource policy (SUBMISSION_BLOCKED)."""


class MagusExecutionError(RuntimeError):
    """MAGUS run failed (EXECUTION_FAILED)."""


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def default_magus_application() -> "MagusApplication | None":
    """Build the MAGUS adapter from the SCNet environment; None when SCNet
    itself is unconfigured (mirrors default_vasp_application)."""
    from photomatagent.scientific.remote.models import RemoteServerConfig

    host = _env("SUPERCOMPUTING_HOST") or _env("SCNET_HOST")
    username = _env("SUPERCOMPUTING_USERNAME") or _env("SCNET_USERNAME")
    if not host or not username:
        return None
    config = RemoteServerConfig(
        host=host,
        username=username,
        port=int(_env("SUPERCOMPUTING_PORT") or _env("SCNET_PORT") or "22"),
        private_key_path=(
            _env("SUPERCOMPUTING_PRIVATE_KEY_PATH")
            or _env("SCNET_PRIVATE_KEY_PATH")
            or ""
        ),
        remote_root=_env("SCNET_REMOTE_ROOT") or "~/photomatagent",
        connect_timeout_seconds=float(_env("SCNET_CONNECT_TIMEOUT_SECONDS") or "20"),
        transfer_timeout_seconds=float(
            _env("SCNET_TRANSFER_TIMEOUT_SECONDS") or "3600"
        ),
    )
    backend = SCNetBackend(config)
    return MagusApplication(
        backend,
        magus_root=_env("SCNET_MAGUS_ROOT") or "~/magus",
        executable=_env("SCNET_MAGUS_EXECUTABLE"),
        env_script=_env("SCNET_MAGUS_ENV_SCRIPT"),
        # MAGUS+VASP reuses the native VASP env script unless a dedicated
        # MAGUS VASP script is configured (both source into the job).
        vasp_script=_env("SCNET_MAGUS_VASP_SCRIPT") or _env("SCNET_VASP_ENV_SCRIPT"),
        vasp_pp_path=_env("SCNET_MAGUS_VASP_PP_PATH"),
        remote_root=config.remote_root,
    )


class MagusApplication:
    """MAGUS adapter on top of SCNetBackend (SSH + Slurm)."""

    name = "magus"

    def __init__(
        self,
        backend: SCNetBackend | Any | None = None,
        *,
        magus_root: str = "~/magus",
        executable: str = "",
        env_script: str = "",
        vasp_script: str = "",
        vasp_pp_path: str = "",
        remote_root: str = "~/photomatagent",
        policy: ResourcePolicy | None = None,
        search_types: list[str] | None = None,
    ) -> None:
        """Configuration only; no network access happens here."""
        self.backend = backend
        self.magus_root = magus_root.strip() or "~/magus"
        self.executable = executable.strip()
        self.env_script = env_script.strip()
        self.vasp_script = vasp_script.strip()
        self.vasp_pp_path = vasp_pp_path.strip()
        self.remote_root = remote_root.rstrip("/")
        self.policy = policy or ResourcePolicy.from_environment()
        self.search_types = list(search_types or SUPPORTED_STRUCTURE_TYPES)

    # ------------------------------------------------------------------
    # probe (read-only)
    # ------------------------------------------------------------------

    def probe_environment(self) -> dict[str, Any]:
        """Static probe; performs remote checks only outside an event loop."""
        report: dict[str, Any] = {
            "application": "magus",
            "backend": getattr(self.backend, "name", "none"),
            "status": "UNCONFIGURED" if self.backend is None else "UNKNOWN",
            "installed": False,  # compatibility: not verified without a backend
            "magus_root": self.magus_root,
            "executable": self.executable or "auto-discover",
            "search_types": self.search_types,
            "candidate_validity": (
                "MAGUS candidates are UNVALIDATED_GENERATED_STRUCTURE until "
                "an internal calculator has actually evaluated them; "
                "structure generation is NOT energy validation"
            ),
            "detail": (
                "SCNet backend not configured: set SCNET_HOST / "
                "SCNET_USERNAME"
                if self.backend is None
                else "run probe_environment_async inside an event loop"
            ),
        }
        if self.backend is None:
            return report
        try:
            import asyncio

            asyncio.get_running_loop()
        except RuntimeError:
            import asyncio

            try:
                remote = asyncio.run(self.probe_environment_async())
                report.update(remote)
            except Exception as exc:
                report["status"] = "ERROR"
                report["error"] = f"{type(exc).__name__}: {exc}"
        return report

    async def probe_environment_async(self) -> dict[str, Any]:
        """Full remote probe: root, executable, version, commands,
        calculators, examples, structure types, VASP/pseudopotential
        readiness. Read-only; never submits."""
        if self.backend is None:
            return {
                "application": "magus",
                "backend": "none",
                "status": "UNCONFIGURED",
                "error_type": "UNCONFIGURED",
                "detail": (
                    "no SCNet backend: set SCNET_HOST / SCNET_USERNAME / "
                    "SCNET_PRIVATE_KEY_PATH (see docs/scnet_scientific_compute.md)"
                ),
            }
        root = self.magus_root
        root_check = await self.backend._run_ssh(f"test -d {root} && echo EXISTS")
        if not root_check.ok:
            return {
                "application": "magus",
                "backend": "scnet",
                "status": "MISSING_DEPENDENCY",
                "error_type": "MISSING_DEPENDENCY",
                "magus_root": root,
                "detail": (
                    f"MAGUS root does not exist on SCNet: {root}; set "
                    "SCNET_MAGUS_ROOT to the real installation root"
                ),
            }
        executable = await self._discover_executable()
        if not executable:
            return {
                "application": "magus",
                "backend": "scnet",
                "status": "MISSING_DEPENDENCY",
                "error_type": "MISSING_DEPENDENCY",
                "magus_root": root,
                "detail": (
                    "no MAGUS executable found under the configured root; "
                    "set SCNET_MAGUS_EXECUTABLE to its absolute path"
                ),
            }
        report: dict[str, Any] = {
            "application": "magus",
            "backend": "scnet",
            "status": "AVAILABLE",
            "magus_root": root,
            "executable": executable,
            "version": "",
            "commands": [],
            "calculators": [],
            "failed_calculators": [],
            "structure_types": [],
            "examples": [],
            "job_system": "SLURM",
            "bin_exists": False,
            "condalib_exists": False,
            "vasp_readiness": {},
            "pseudopotential_readiness": {},
            "warnings": [],
            "candidate_validity": (
                "MAGUS candidates are UNVALIDATED_GENERATED_STRUCTURE until "
                "an internal calculator has actually evaluated them"
            ),
        }
        layout = await self.backend._run_ssh(
            f"test -d {root}/bin && echo bin-EXISTS; "
            f"test -d {root}/condalib && echo condalib-EXISTS"
        )
        report["bin_exists"] = "bin-EXISTS" in layout.stdout
        report["condalib_exists"] = "condalib-EXISTS" in layout.stdout
        version = await self.backend._run_ssh(f"{executable} -v 2>&1 | head -n 1")
        if version.ok:
            report["version"] = parse_magus_version(version.stdout)
        help_result = await self.backend._run_ssh(
            f"{executable} -h 2>&1 | head -n 120"
        )
        if help_result.ok:
            report["commands"] = parse_magus_help_commands(help_result.stdout)
        checkpack = await self.backend._run_ssh(
            f"{executable} checkpack calculators 2>&1 | tail -n 200"
        )
        if checkpack.ok:
            parsed = parse_checkpack_calculators(checkpack.stdout, checkpack.stderr)
            report["calculators"] = parsed["available"]
            report["failed_calculators"] = parsed["failed"]
            if parsed["failed"]:
                report["warnings"].append(
                    "optional calculators unavailable: "
                    + ", ".join(parsed["failed"])
                )
        examples = await self._discover_examples()
        report["examples"] = examples
        if examples:
            types = await self._probe_structure_types(examples)
            report["structure_types"] = types
            if types:
                report["search_types"] = [
                    t for t in self.search_types if t in types
                ]
        else:
            report["search_types"] = self.search_types
        report["vasp_readiness"] = await self._probe_vasp_readiness(
            executable, report["calculators"]
        )
        report["pseudopotential_readiness"] = await self._probe_psp_readiness()
        return report

    async def _discover_executable(self) -> str:
        """Priority: explicit env -> <root>/bin/magus -> <root>/magus ->
        bounded find inside the root only."""
        backend = self.backend
        if backend is None:
            return ""
        if self.executable:
            check = await backend._run_ssh(f"test -x {self.executable} && echo OK")
            if check.ok:
                return self.executable
            return ""
        root = self.magus_root
        for candidate in (f"{root}/bin/magus", f"{root}/magus"):
            check = await backend._run_ssh(f"test -x {candidate} && echo OK")
            if check.ok:
                return candidate
        found = await backend._run_ssh(
            f"find {root} -maxdepth 4 -type f -name magus -perm -u+x "
            "2>/dev/null | head -n 5"
        )
        if found.ok:
            first = found.stdout.strip().splitlines()
            if first:
                return first[0].strip()
        return ""

    async def _discover_examples(self) -> list[str]:
        backend = self.backend
        if backend is None:
            return []
        root = self.magus_root
        zip_check = await backend._run_ssh(f"test -f {root}/examples.zip && echo ZIP")
        if zip_check.ok:
            listing = await backend._run_ssh(
                f"unzip -l {root}/examples.zip 2>/dev/null | head -n 500"
            )
            if listing.ok:
                dirs = parse_example_dirs(listing.stdout)
                if dirs:
                    return [f"{root}/examples.zip#{name}" for name in dirs]
        found = await backend._run_ssh(
            f"find {root} -maxdepth 3 -type d \\( -iname '*example*' -o "
            "-iname '*demo*' \\) 2>/dev/null | head -n 10"
        )
        if found.ok:
            return [line.strip() for line in found.stdout.splitlines() if line.strip()]
        return []

    async def _probe_structure_types(self, examples: list[str]) -> list[str]:
        """Extract ``structureType`` values from installed example inputs."""
        backend = self.backend
        if backend is None:
            return []
        root = self.magus_root
        zip_entries = [
            entry.split("#", 1)[1] for entry in examples if "#" in entry
        ][:24]
        if not zip_entries:
            return []
        command = (
            "for d in "
            + " ".join(shlex.quote(entry) for entry in zip_entries)
            + "; do "
            # entries end with "/" so append the file name without a slash
            f"unzip -p {root}/examples.zip \"${{d}}input.yaml\" 2>/dev/null | "
            "grep -m1 '^structureType:' ; done | sort -u"
        )
        result = await backend._run_ssh(command)
        if not result.ok:
            return []
        return parse_example_structure_types(result.stdout)

    async def _probe_vasp_readiness(
        self, executable: str, calculators: list[str]
    ) -> dict[str, Any]:
        backend = self.backend
        if backend is None:
            return {"calculator": "MISSING", "detail": "no backend"}
        if "vasp" not in calculators and "vaspc" not in calculators:
            return {
                "calculator": "MISSING",
                "detail": "vasp calculator not reported by `magus checkpack calculators`",
            }
        vasp_command = await backend._run_ssh(
            "command -v vasp_std 2>/dev/null | head -n 1"
        )
        return {
            "calculator": "READY",
            "vasp_std_on_path": (
                vasp_command.stdout.strip()[:300] if vasp_command.ok else ""
            ),
            "launcher": (
                "READY"
                if (self.vasp_script or self.env_script)
                else "MISSING"
            ),
            "detail": (
                "MAGUS VASP calculator present; launcher requires "
                "SCNET_MAGUS_VASP_SCRIPT or SCNET_MAGUS_ENV_SCRIPT"
                if not (self.vasp_script or self.env_script)
                else "VASP env script configured; sourced inside MAGUS jobs"
            ),
        }

    async def _probe_psp_readiness(self) -> dict[str, Any]:
        backend = self.backend
        if backend is None:
            return {"configured": False, "detail": "no backend"}
        if not self.vasp_pp_path:
            return {
                "configured": False,
                "detail": (
                    "SCNET_MAGUS_VASP_PP_PATH not set; MAGUS+VASP searches "
                    "cannot resolve POTCAR"
                ),
            }
        resolved = await resolve_remote_psp_library(self.vasp_pp_path, backend)
        if resolved is None:
            return {
                "configured": True,
                "resolved_library": None,
                "layout": None,
                "detail": (
                    f"no known POTCAR layout found under {self.vasp_pp_path}; "
                    "expected <root>/<setup>/POTCAR, <root>/potpaw_PBE/<setup>/POTCAR "
                    "or <root>/potpaw_PBE.64/<setup>/POTCAR"
                ),
            }
        library, layout = resolved
        return {
            "configured": True,
            "resolved_library": library,
            "layout": layout,
            "detail": f"POTCAR layout detected: {layout}",
        }

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    def validate_search_request(self, request: MagusSearchRequest) -> list[str]:
        """Typed validation problems for a search request."""
        problems: list[str] = []
        if request.structure_type not in self.search_types:
            problems.append(
                f"structure_type {request.structure_type!r} not exposed by this "
                f"MAGUS installation; supported: {self.search_types}"
            )
        if request.structure_type == "surface" and request.slab is None:
            problems.append(
                "surface search requires a slab configuration (bulk_file and "
                "slab layers); provide MagusSearchRequest.slab"
            )
        if request.calculator == "vasp" and request.execution_mode == "parallel":
            problems.append(
                "parallel VASP mode submits nested sbatch jobs (MAGUS "
                "queuemanage) and is not supported inside a PhotoMatAgent "
                "allocation; use execution_mode=serial"
            )
        return problems

    # ------------------------------------------------------------------
    # preparation (local, no I/O beyond the job tree)
    # ------------------------------------------------------------------

    def _build_manifest(
        self,
        *,
        operation: str,
        request: MagusGenerateRequest | MagusSearchRequest,
        input_yaml: str,
        job_dir: Path,
    ) -> dict[str, Any]:
        from datetime import datetime, timezone

        from pymatgen.core import Composition

        amounts = dict(
            zip(request.composition.symbols, request.composition.formula, strict=True)
        )
        formula_string = Composition(amounts).reduced_formula
        config = MagusExecutionConfig(
            backend="scnet",
            magus_root=self.magus_root,
            executable=self.executable or "auto-discover",
            operation=operation,
            search_type=request.structure_type,
            calculator=(
                request.calculator
                if isinstance(request, MagusSearchRequest)
                else ""
            ),
            execution_mode=(
                request.execution_mode
                if isinstance(request, MagusSearchRequest)
                else "serial"
            ),
            generation_parameters=request.model_dump(exclude={"composition"}),
            remote_root=self.remote_root,
            limitations=[
                "candidates are UNVALIDATED_GENERATED_STRUCTURE until an "
                "internal calculator evaluates them",
                "structure generation is not energy validation",
            ],
        )
        manifest: dict[str, Any] = {
            "application": "magus",
            "status": "PREPARED",
            "operation": operation,
            "search_type": request.structure_type,
            "composition": formula_string,
            "request": request.model_dump(),
            "backend": config.backend,
            "magus_version": "",
            "input_hash": manifest_input_hash(input_yaml),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "candidate_lineage_root": f"magus_{operation}_{request.structure_type}",
            "execution": config.model_dump(),
        }
        (job_dir / "photomat_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest

    @staticmethod
    def _write_legacy_manifest(
        job_dir: Path, operation: str, request: MagusGenerateRequest | MagusSearchRequest
    ) -> None:
        """Backward-compatible ``magus_manifest.json`` pointer file."""
        (job_dir / "magus_manifest.json").write_text(
            json.dumps(
                {
                    "application": "magus",
                    "status": "PREPARED",
                    "operation": operation,
                    "search_type": request.structure_type,
                    "composition": request.composition.model_dump(),
                    "deprecated": True,
                    "see": "photomat_manifest.json",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def prepare_generate(
        self, request: MagusGenerateRequest, job_dir: str | Path
    ) -> dict[str, Any]:
        """Write input.yaml + magus.slurm + photomat_manifest.json for
        ``magus generate``. Never submits."""
        if request.structure_type not in self.search_types:
            raise MagusPrerequisiteError(
                f"structure_type {request.structure_type!r} not supported by "
                f"this installation; supported: {self.search_types}"
            )
        job = Path(job_dir).expanduser().resolve()
        job.mkdir(parents=True, exist_ok=True)
        input_yaml = render_generate_input(request)
        (job / "input.yaml").write_text(input_yaml, encoding="utf-8")
        script = self.render_slurm(
            job_name=f"magus-gen-{job.name[:40]}",
            request=request,
            resource=ResourceRequest(
                partition=os.environ.get("SCNET_PARTITION", "normal"),
                nodes=1,
                tasks_per_node=8,
                walltime_minutes=int(
                    os.environ.get("PHOTOMATAGENT_HPC_MAX_WALLTIME_MINUTES", "120")
                    or "120"
                ),
            ),
        )
        (job / "magus.slurm").write_text(script, encoding="utf-8")
        manifest = self._build_manifest(
            operation="generate",
            request=request,
            input_yaml=input_yaml,
            job_dir=job,
        )
        self._write_legacy_manifest(job, "generate", request)
        manifest["job_dir"] = str(job)
        return manifest

    def prepare_search(
        self, request: MagusSearchRequest, job_dir: str | Path
    ) -> dict[str, Any]:
        """Write the full job tree for ``magus search``:
        input.yaml + inputFold/VASP/INCAR (VASP only) + magus.slurm +
        photomat_manifest.json. Never submits."""
        problems = self.validate_search_request(request)
        if problems:
            raise MagusPrerequisiteError("; ".join(problems))
        job = Path(job_dir).expanduser().resolve()
        job.mkdir(parents=True, exist_ok=True)
        input_yaml = render_search_input(request)
        (job / "input.yaml").write_text(input_yaml, encoding="utf-8")
        if request.calculator == "vasp":
            vasp_input = job / "inputFold" / "VASP"
            vasp_input.mkdir(parents=True, exist_ok=True)
            (vasp_input / "INCAR").write_text(MAGUS_VASP_INCAR, encoding="utf-8")
        (job / "Seeds").mkdir(parents=True, exist_ok=True)
        self._write_legacy_manifest(job, "search", request)
        script = self.render_slurm(
            job_name=f"magus-search-{job.name[:40]}",
            request=request,
            resource=ResourceRequest(
                partition=os.environ.get("SCNET_PARTITION", "normal"),
                nodes=1,
                tasks_per_node=8,
                walltime_minutes=int(
                    os.environ.get("PHOTOMATAGENT_HPC_MAX_WALLTIME_MINUTES", "120")
                    or "120"
                ),
            ),
        )
        (job / "magus.slurm").write_text(script, encoding="utf-8")
        manifest = self._build_manifest(
            operation="search",
            request=request,
            input_yaml=input_yaml,
            job_dir=job,
        )
        manifest["job_dir"] = str(job)
        manifest["pseudopotentials"] = [
            req.model_dump()
            for req in self.pseudopotential_requirements(request)
        ]
        (job / "photomat_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest

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
        """Backward-compatible wrapper: full job tree for a bulk search
        (no longer a manifest-only job)."""
        del target_dir  # seeds are not supported in v1; the tree is self-contained
        request = MagusSearchRequest.from_composition(
            composition,
            structure_type=search_type,  # type: ignore[arg-type]
            calculator="vasp",
            execution_mode="serial",
            init_size=population_size,
            population_size=population_size,
            generations=generations,
            save_good=min(2, population_size),
        )
        return self.prepare_search(request, output_dir)

    def pseudopotential_requirements(
        self, request: MagusSearchRequest
    ) -> list[MagusPseudopotentialRequirement]:
        """Build element/setup requirements for the request composition."""
        if request.calculator != "vasp":
            return []
        if request.pseudopotentials:
            return list(request.pseudopotentials)
        return [
            MagusPseudopotentialRequirement(element=symbol)
            for symbol in request.composition.symbols
        ]

    def render_slurm(
        self,
        *,
        job_name: str,
        request: MagusGenerateRequest | MagusSearchRequest,
        resource: ResourceRequest,
    ) -> str:
        """Deterministic Slurm script (launcher empty; JOB_SYSTEM=SLURM)."""
        operation = "generate" if isinstance(request, MagusGenerateRequest) else "search"
        args = magus_arguments(operation, request)
        return render_magus_slurm(
            job_name=job_name,
            executable=self.executable or f"{self.magus_root}/bin/magus",
            args=args,
            resource=resource,
            magus_root=self.magus_root,
            env_script=self.env_script,
            vasp_script=self.vasp_script,
            vasp_pp_path=self.vasp_pp_path,
        )

    # ------------------------------------------------------------------
    # submission / status / collect
    # ------------------------------------------------------------------

    async def submit(
        self,
        *,
        job_name: str,
        prepared_dir: str | Path,
        resource: ResourceRequest | None = None,
    ) -> RemoteJobRef:
        """Upload a prepared job tree and submit via Slurm. Rejects
        manifest-only trees and refuses without authorization."""
        if self.backend is None:
            raise MagusUnconfiguredError("MAGUS backend is not configured")
        root = Path(prepared_dir).expanduser().resolve()
        missing = [name for name in MAGUS_JOB_FILES if not (root / name).is_file()]
        if missing:
            raise ValueError(
                "prepared tree is incomplete (missing: "
                + ", ".join(missing)
                + "); run magus.prepare_generate / magus.prepare_search first"
            )
        manifest = self._read_manifest(root)
        operation = manifest.get("operation", "")
        calculator = str(manifest.get("request", {}).get("calculator", ""))
        if operation == "search" and calculator == "vasp":
            requirements = [
                (item["element"], item.get("setup", ""))
                for item in manifest.get("pseudopotentials", [])
            ]
            if not self.vasp_pp_path:
                raise MagusPrerequisiteError(
                    "VASP search requires SCNET_MAGUS_VASP_PP_PATH (ASE "
                    "VASP_PP_PATH parent of potpaw_PBE)"
                )
            missing_psp = await remote_potcar_check(
                self.backend, self.vasp_pp_path, requirements
            )
            if missing_psp:
                raise MagusPseudopotentialMissingError(missing_psp)
        executable = self.executable or manifest.get("executable", "")
        safe_name = job_name.replace("/", "-")[:48] or "magus"
        remote_directory = (
            f"{self.remote_root}/magus/{safe_name}-{uuid.uuid4().hex[:8]}"
        )
        validate_remote_path(remote_directory)
        await self.backend.upload_tree(root, remote_directory)
        request = resource or ResourceRequest(
            partition=os.environ.get("SCNET_PARTITION", "normal"),
            nodes=1,
            tasks_per_node=8,
            walltime_minutes=int(
                os.environ.get("PHOTOMATAGENT_HPC_MAX_WALLTIME_MINUTES", "120")
                or "120"
            ),
        )
        try:
            return await self.backend.submit_script(
                RemoteJobSpec(
                    application="magus",
                    job_name=safe_name,
                    remote_directory=remote_directory,
                    script_name="magus.slurm",
                    resource=request,
                    executable=executable or f"{self.magus_root}/bin/magus",
                    provenance={
                        "prepared_dir": str(root),
                        "operation": operation,
                        "input_hash": manifest.get("input_hash", ""),
                    },
                )
            )
        except Exception as exc:
            if "HPC submission is disabled" in str(exc) or "exceeds policy" in str(exc):
                raise MagusSubmissionBlockedError(str(exc)) from exc
            raise

    @staticmethod
    def _read_manifest(root: Path) -> dict[str, Any]:
        manifest_path = root / "photomat_manifest.json"
        if not manifest_path.is_file():
            legacy = root / "magus_manifest.json"
            if legacy.is_file():
                return json.loads(legacy.read_text(encoding="utf-8"))
            raise FileNotFoundError("photomat_manifest.json missing; run prepare first")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    async def status(self, job_id: str) -> HPCJobState:
        if self.backend is None:
            return HPCJobState.UNKNOWN
        return await self.backend.job_status(job_id)

    async def collect(
        self, *, job_ref: RemoteJobRef, local_dir: str | Path
    ) -> dict[str, Any]:
        """Download bounded artifacts and produce a structured report."""
        if self.backend is None:
            raise MagusUnconfiguredError("MAGUS backend is not configured")
        local = Path(local_dir).expanduser().resolve()
        local.mkdir(parents=True, exist_ok=True)
        artifacts = await self.backend.list_remote_artifacts(job_ref.remote_directory)
        interesting = [
            artifact
            for artifact in artifacts
            if self._interesting_artifact(artifact.name)
            and (artifact.size_bytes or 0) <= _MAX_DOWNLOAD_BYTES
        ][:_MAX_DOWNLOAD_FILES]
        downloaded: list[Path] = []
        for artifact in interesting:
            # Nested artifacts (results/best.traj) keep their relative
            # structure locally; flat files land in the root.
            target_dir = local
            if "/" in artifact.name:
                target_dir = local / Path(artifact.name).parent
            path = await self.backend.download_file(
                job_ref.remote_directory, artifact.name, target_dir
            )
            if path is not None:
                downloaded.append(path)
        manifest = self._read_manifest(local)
        operation = manifest.get("operation", "")
        inspected = self.inspect_results(
            local,
            operation=operation,
            expected_number=(
                manifest.get("request", {}).get("number")
                if operation == "generate"
                else None
            ),
        )
        return {
            "job_id": job_ref.job_id,
            "application": "magus",
            "operation": operation,
            "scheduler_state": job_ref.state.value,
            "downloaded": [path.name for path in downloaded],
            "artifact_count": len(interesting),
            "artifacts": [
                {
                    "name": artifact.name,
                    "size_bytes": artifact.size_bytes,
                    "remote_path": artifact.remote_path,
                }
                for artifact in interesting[:100]
            ],
            "candidates": inspected["candidates"],
            "candidate_count": inspected["candidate_count"],
            "summary": inspected["summary"],
            "note": (
                "execution acceptance only; candidates remain "
                "UNVALIDATED_GENERATED_STRUCTURE unless energies were "
                "actually computed by the internal calculator"
            ),
        }

    @staticmethod
    def _interesting_artifact(name: str) -> bool:
        base = Path(name).name
        if base in _TEXT_ARTIFACTS:
            return True
        return base.endswith(
            (".traj", ".vasp", ".out", ".err", ".yaml", ".json")
        ) or base in {"INCAR", "POSCAR", "CONTCAR", "OUTCAR", "EIGENVAL"}

    def inspect_results(
        self,
        result_dir: str | Path,
        *,
        operation: str = "generate",
        expected_number: int | None = None,
    ) -> dict[str, Any]:
        """Parse bounded candidate info from collected artifacts. Never
        fabricates: anything unverifiable is reported as unknown."""
        root = Path(result_dir).expanduser().resolve()
        candidates: list[dict[str, Any]] = []
        summary_text = ""
        summary_file = root / "summary"
        if summary_file.is_file():
            summary_text = (
                summary_file.read_text(encoding="utf-8", errors="replace")[:6000]
            )
            candidates = self._parse_summary_rows(summary_text)
        traj_files = []
        if operation == "generate":
            traj_files.append(root / "gen.traj")
        else:
            for name in ("results/best.traj", "results/good.traj", "best.traj", "good.traj"):
                traj_files.append(root / name)
        traj_counts: dict[str, int] = {}
        for traj in traj_files:
            if not traj.is_file() or traj.stat().st_size > _MAX_DOWNLOAD_BYTES:
                continue
            count = self._count_traj_frames(traj)
            if count is not None:
                traj_counts[traj.relative_to(root).as_posix()] = count
        if traj_counts:
            for name, count in sorted(traj_counts.items()):
                candidates.append({"artifact": name, "frames": count})
        if operation == "generate" and not traj_counts and expected_number is not None:
            candidates = [
                {"requested": expected_number, "verified_from_artifact": False}
            ]
        return {
            "operation": operation,
            "candidates": candidates[:200],
            "candidate_count": (
                len(candidates)
                if candidates
                else None
            ),
            "summary": summary_text,
            "note": (
                "candidate count derived from downloaded artifacts only; "
                "no energies are reported without an internal calculator"
            ),
        }

    @staticmethod
    def _parse_summary_rows(summary_text: str) -> list[dict[str, Any]]:
        """Parse the MAGUS ``summary`` table rows (bounded, best-effort)."""
        rows: list[dict[str, Any]] = []
        header_seen = False
        for line in (summary_text or "").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if "symmetry" in stripped.lower() and "enthalpy" in stripped.lower():
                header_seen = True
                continue
            if not header_seen:
                continue
            parts = stripped.split()
            if len(parts) < 3:
                continue
            rows.append(
                {
                    "symmetry": " ".join(parts[:2]),
                    "enthalpy": parts[2] if len(parts) > 2 else None,
                    "formula": parts[3] if len(parts) > 3 else None,
                }
            )
            if len(rows) >= 200:
                break
        return rows

    @staticmethod
    def _count_traj_frames(traj: Path) -> int | None:
        """Count frames in an ASE traj file; None when unreadable."""
        try:
            from ase.io import read

            frames = read(str(traj), index=":")
            return len(frames)
        except Exception:
            return None
