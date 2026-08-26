"""VaspApplication: VASP-specific adapter on top of a generic HPC backend.

Sprint 3 section 20-21: the donor ``VaspCloudAdapter`` is split into
``SCNetBackend`` (generic) + this application (VASP knowledge). The
application never implements SSH/Slurm; it validates, prepares, submits,
collects, validates output and parses results. ``run_workflow`` exists only
as a bounded convenience API; the core contract is detached
prepare -> submit -> status -> collect.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import uuid
from pathlib import Path
from typing import Any

from photomatagent.scientific.applications.vasp.inputs import VaspInputGenerator
from photomatagent.scientific.applications.vasp.profiles import (
    VaspProfile,
    get_profile,
)
from photomatagent.scientific.applications.vasp.validation import (
    REQUIRED_VASP_RESULT_FILES,
    parse_result,
    validate_output,
)
from photomatagent.scientific.remote.models import (
    HPCJobState,
    RemoteJobRef,
    RemoteJobSpec,
    RemoteServerConfig,
    ResourcePolicy,
    ResourceRequest,
)
from photomatagent.scientific.remote.scheduler import render_slurm_script
from photomatagent.scientific.remote.scnet import SCNetBackend, validate_remote_path

VASP_REQUIRED_INPUTS = ("POSCAR", "INCAR", "KPOINTS")


def env_source_preamble(env_script: str, executable: str) -> str:
    """Bash preamble that sources the SCNet product environment.

    SCNet product ``env.sh`` may end with a best-effort accounting write
    that returns non-zero for users. Preserve the exported environment,
    then explicitly verify the executable before running anything.
    """
    validate_remote_path(env_script, allow_tilde=False)
    return (
        "set +e\n"
        f"source {shlex.quote(env_script)}\n"
        "set -e\n"
        f"command -v {shlex.quote(executable)} >/dev/null"
    )


def potcar_assembly_preamble(
    remote_psp_dir: str, potcar_symbols: list[str]
) -> str:
    """Bash preamble that assembles POTCAR on the remote host, in element
    order, from ``SCNET_VASP_PSP_DIR``.

    The curated POTCAR datasets stay on the cluster: no POTCAR content is
    uploaded, logged or returned. Only the element sequence (a list of
    single-symbol tokens) is interpolated into the script.
    """
    validate_remote_path(remote_psp_dir)
    if remote_psp_dir.startswith("~/"):
        psp_value = '"${HOME}/' + remote_psp_dir[2:] + '"'
    else:
        psp_value = shlex.quote(remote_psp_dir)
    lines = [
        "if [ ! -s POTCAR ]; then",
        f"  psp_base={psp_value}",
        "  : > POTCAR",
        (
            "  # layout detection: direct <root>/<setup>/POTCAR, "
            "then potpaw_PBE, then legacy potpaw_PBE.64"
        ),
        '  for cand in "$psp_base" "$psp_base/potpaw_PBE" '
        '"$psp_base/potpaw_PBE.64"; do',
    ]
    for symbol in potcar_symbols:
        if not symbol.isalpha():
            raise ValueError(f"unsafe POTCAR symbol: {symbol!r}")
        lines.extend(
            [
                f'    if [ -z "$psp_lib" ] && [ -s "$cand/{symbol}/POTCAR" ]; then',
                f'      psp_lib="$cand"',
            ]
        )
    lines.extend(
        [
            "  done",
            '  test -n "$psp_lib"',
            '  for sym in ' + " ".join(potcar_symbols) + "; do",
            '    cat "$psp_lib/$sym/POTCAR" >> POTCAR',
            "  done",
        ]
    )
    lines.append("fi")
    return "\n".join(lines)


class VaspApplication:
    """VASP workflows: profiles, input preparation, submission, validation."""

    def __init__(
        self,
        backend: SCNetBackend | Any | None = None,
        *,
        workspace: str | Path | None = None,
        psp_dir: str | None = None,
        jobs_local_dir: str | Path = "output/vasp_inputs",
        policy: ResourcePolicy | None = None,
        module_name: str = "",
        env_script: str = "",
        remote_root: str = "~/photomatagent",
        remote_psp_dir: str = "",
    ) -> None:
        self.workspace = Path(workspace or Path.cwd()).expanduser().resolve()
        self.policy = policy or ResourcePolicy.from_environment()
        self.backend = backend
        self.module_name = module_name
        self.env_script = env_script
        self.remote_root = remote_root.rstrip("/")
        self.remote_psp_dir = remote_psp_dir.rstrip("/")
        self.generator = VaspInputGenerator(
            psp_dir=psp_dir, jobs_local_dir=jobs_local_dir
        )

    # -- environment --------------------------------------------------------

    def probe_environment(self) -> dict[str, Any]:
        """Read-only probe: backend, Slurm, pseudopotential resolution."""
        from photomatagent.scientific.applications.vasp.psp import (
            resolve_local_psp_library,
        )

        local_psp = resolve_local_psp_library(self.generator.psp_dir)
        report: dict[str, Any] = {
            "application": "vasp",
            "backend": getattr(self.backend, "name", "none"),
            "profiles": [profile.name for profile in self._profiles()],
            "soc_supported": True,  # vasp_ncl profile available
            "potcar_policy": (
                "POTCAR resolved from PMG_VASP_PSP_DIR or remote location "
                "at submit time; never committed or logged"
            ),
            "psp_dir_local": (
                str(local_psp[0]) if local_psp is not None else None
            ),
            "psp_layout_local": local_psp[1] if local_psp is not None else None,
            "submission_authorized": self.policy.allow_hpc_submit,
            "module": self.module_name,
            "env_script_configured": bool(self.env_script),
            "remote_psp_configured": bool(self.remote_psp_dir),
        }
        if self.backend is not None:
            import asyncio

            try:
                asyncio.get_running_loop()
            except RuntimeError:
                try:
                    report["connection"] = asyncio.run(
                        self.backend.check_connection()
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
        else:
            report["connection"] = {
                "connected": "false",
                "error": "no backend configured",
            }
        return report

    async def probe_environment_async(self) -> dict[str, Any]:
        """Async environment probe for MCP/CLI event loops."""
        report = self.probe_environment()
        if self.backend is None:
            return report
        report["connection"] = await self.backend.check_connection()
        if report["connection"].get("connected") == "true":
            report["available_partitions"] = await self.backend.available_partitions()
            report["software"] = await self.backend.probe_module(
                self.module_name, "vasp_std"
            )
        return report

    @staticmethod
    def _profiles() -> list[VaspProfile]:
        from photomatagent.scientific.applications.vasp.profiles import profiles

        return profiles()

    # -- validation ---------------------------------------------------------

    def validate_inputs(
        self, structure_path: str | Path, profile_name: str
    ) -> list[str]:
        """Return input problems (empty means acceptable)."""
        problems: list[str] = []
        try:
            profile = get_profile(profile_name)
        except ValueError as exc:
            return [str(exc)]
        path = Path(structure_path).expanduser().resolve()
        if not path.is_file():
            problems.append(f"structure file does not exist: {path}")
            return problems
        try:
            structure = self.generator.load_structure(path)
        except Exception as exc:
            problems.append(f"structure unreadable: {type(exc).__name__}: {exc}")
            return problems
        if profile.soc and any(
            element.is_actinoid for element in structure.composition.elements
        ):
            problems.append(
                "actinide elements detected: LMAXMIX must be set to 6; "
                "narrow_gap_soc profile currently uses LMAXMIX=4"
            )
        if profile.name == "namd_preparation" and profile.needs_configuration:
            problems.append(
                "namd_preparation is gated: confirm the SCNet Hefei-NAMD "
                "module/environment before production use (see profile "
                "limitations)"
            )
        return problems

    # -- preparation --------------------------------------------------------

    def prepare_inputs(
        self,
        *,
        structure_path: str | Path,
        profile_name: str,
        output_dir: str | Path | None = None,
        spec_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate the full stage workflow; never submits anything."""
        problems = self.validate_inputs(structure_path, profile_name)
        if problems:
            raise ValueError("; ".join(problems))
        root = Path(output_dir) if output_dir else (
            self.generator.jobs_local_dir / Path(structure_path).stem
        )
        return self.generator.prepare_workflow(
            structure_path=structure_path,
            profile_name=profile_name,
            output_root=root,
            spec_overrides=spec_overrides,
        )

    def resolve_potcar(self, input_dir: str | Path) -> Path | None:
        """Assemble POTCAR from the local psp dir; None when unresolvable."""
        from photomatagent.scientific.applications.vasp.psp import (
            resolve_local_psp_library,
        )

        input_dir = Path(input_dir)
        resolved = resolve_local_psp_library(self.generator.psp_dir)
        if resolved is None:
            return None
        psp, _ = resolved
        policy = input_dir / "POTCAR.policy"
        symbols: list[str] = []
        if policy.is_file():
            for line in policy.read_text(encoding="utf-8").splitlines():
                if line.startswith("  ") and ": " in line:
                    symbol = line.strip().split(":")[0].strip()
                    if symbol.isalpha():
                        symbols.append(symbol)
        if not symbols:
            return None
        target = input_dir / "POTCAR"
        with target.open("wb") as destination:
            for symbol in symbols:
                source = psp / symbol / "POTCAR"
                if not source.is_file():
                    return None
                with source.open("rb") as handle:
                    shutil.copyfileobj(handle, destination)
        return target

    def render_slurm(
        self,
        *,
        job_name: str,
        profile: VaspProfile,
        resource: ResourceRequest | None = None,
        executable: str | None = None,
        potcar_symbols: list[str] | None = None,
    ) -> str:
        """Render the submission script for one stage."""
        preamble = ""
        preamble_parts: list[str] = []
        if self.env_script:
            # Single source of the product-environment preamble (also reused
            # by the isolated-molecule runner so there is exactly one Slurm
            # template instead of divergent copies).
            preamble_parts.append(
                env_source_preamble(
                    self.env_script, executable or profile.executable
                )
            )
        if self.remote_psp_dir and potcar_symbols:
            preamble_parts.append(
                potcar_assembly_preamble(
                    self.remote_psp_dir, list(potcar_symbols)
                )
            )
        preamble = "\n".join(preamble_parts)
        return render_slurm_script(
            job_name=job_name,
            resource=resource or profile.default_resource,
            module_load="" if self.env_script else self.module_name,
            executable=executable or profile.executable,
            preamble=preamble,
        )

    @staticmethod
    def _potcar_symbols(input_dir: Path) -> list[str]:
        policy = input_dir / "POTCAR.policy"
        symbols: list[str] = []
        if policy.is_file():
            for line in policy.read_text(encoding="utf-8").splitlines():
                if line.startswith("  ") and ": " in line:
                    symbol = line.strip().split(":", 1)[0]
                    if symbol.isalpha() and symbol not in symbols:
                        symbols.append(symbol)
        return symbols

    @staticmethod
    def _validate_incar_for_resource(
        incar: Path, resource: ResourceRequest
    ) -> None:
        raw = incar.read_bytes()
        if b"\r\n" in raw:
            raise ValueError(
                "INCAR uses CRLF line endings; convert it with dos2unix INCAR"
            )
        text = raw.decode("utf-8", errors="replace")
        total_tasks = resource.nodes * resource.tasks_per_node
        for key in ("NCORE", "NPAR"):
            match = re.search(
                rf"(?im)^\s*{key}\s*=\s*(\d+)\b", text
            )
            if not match:
                continue
            value = int(match.group(1))
            if value < 1 or total_tasks % value:
                raise ValueError(
                    f"{key}={value} must divide total Slurm tasks "
                    f"({resource.nodes} x {resource.tasks_per_node} = "
                    f"{total_tasks})"
                )

    # -- submission ---------------------------------------------------------

    async def submit_stage(
        self,
        *,
        job_name: str,
        input_dir: str | Path,
        profile_name: str,
        remote_root: str | None = None,
        resource: ResourceRequest | None = None,
        unique_remote_directory: bool = True,
    ) -> RemoteJobRef:
        """Upload one stage directory and submit; detached by default.

        ``unique_remote_directory`` defaults to True: two jobs must never
        write into the same remote directory. Callers that pass False
        explicitly accept the (legacy) shared-directory risk.
        """
        if self.backend is None:
            raise RuntimeError("VASP backend is not configured")
        profile = get_profile(profile_name)
        request = resource or profile.default_resource
        input_dir = Path(input_dir).expanduser().resolve()
        missing = [
            name for name in VASP_REQUIRED_INPUTS if not (input_dir / name).is_file()
        ]
        if missing:
            raise ValueError(f"missing VASP inputs: {', '.join(missing)}")
        self._validate_incar_for_resource(input_dir / "INCAR", request)
        potcar = self.resolve_potcar(input_dir)
        symbols = self._potcar_symbols(input_dir)
        if (
            potcar is None
            and not (input_dir / "POTCAR").is_file()
            and not (self.remote_psp_dir and symbols)
        ):
            raise ValueError(
                "POTCAR cannot be resolved: configure PMG_VASP_PSP_DIR "
                "(local) or SCNET_VASP_PSP_DIR (remote)"
            )
        files = [input_dir / name for name in VASP_REQUIRED_INPUTS]
        for name in ("POTCAR", "POTCAR.policy"):
            if (input_dir / name).is_file():
                files.append(input_dir / name)
        safe_name = job_name.replace("/", "-")[:64] or "vasp"
        root = (remote_root or f"{self.remote_root}/vasp").rstrip("/")
        suffix = f"-{uuid.uuid4().hex[:8]}" if unique_remote_directory else ""
        remote_directory = f"{root}/{safe_name}{suffix}"
        validate_remote_path(remote_directory)
        await self.backend.upload_files(files, remote_directory)
        script = self.render_slurm(
            job_name=safe_name,
            profile=profile,
            resource=request,
            potcar_symbols=(symbols if not (input_dir / "POTCAR").is_file() else []),
        )
        script_path = input_dir / "vasp.slurm"
        script_path.write_text(script, encoding="utf-8")
        await self.backend.upload_files([script_path], remote_directory)
        return await self.backend.submit_script(
            RemoteJobSpec(
                application="vasp",
                job_name=safe_name,
                remote_directory=remote_directory,
                script_name="vasp.slurm",
                resource=request,
                executable=profile.executable,
                provenance={
                    "profile": profile.name,
                    "stage_dir": str(input_dir),
                    "soc": profile.soc,
                },
            )
        )

    async def submit_workflow(
        self, *, workflow_dir: str | Path, profile_name: str
    ) -> dict[str, Any]:
        """Submit every stage sequentially, propagating dependencies.

        Convenience API: stages run one after another (each waits for the
        previous Slurm job). For production use, prefer preparing each stage
        and submitting detached jobs.
        """
        from photomatagent.scientific.applications.vasp.workflow import (
            run_vasp_workflow,
        )

        return await run_vasp_workflow(
            application=self,
            workflow_dir=Path(workflow_dir),
            profile_name=profile_name,
        )

    # -- status / collect ---------------------------------------------------

    async def status(self, job_id: str) -> HPCJobState:
        if self.backend is None:
            return HPCJobState.UNKNOWN
        return await self.backend.job_status(job_id)

    async def collect(
        self,
        *,
        job_ref: RemoteJobRef,
        local_dir: str | Path,
        profile_name: str,
    ) -> dict[str, Any]:
        """Download results, validate, parse; returns a structured report."""
        if self.backend is None:
            raise RuntimeError("VASP backend is not configured")
        local = Path(local_dir).expanduser().resolve()
        local.mkdir(parents=True, exist_ok=True)
        downloaded = await self.backend.download_files(
            job_ref.remote_directory,
            list(REQUIRED_VASP_RESULT_FILES),
            local,
        )
        problems = validate_output(local, profile_name=profile_name)
        parsed = parse_result(local)
        return {
            "job_id": job_ref.job_id,
            "profile": profile_name,
            "scheduler_state": job_ref.state.value,
            "downloaded": [path.name for path in downloaded],
            "validation_problems": problems,
            "scientifically_valid": not problems,
            "parsed": parsed,
            "artifacts": [str(path) for path in downloaded],
            "note": (
                "Slurm COMPLETED is scheduler state; scientific validity "
                "requires an empty validation_problems list"
            ),
        }

    def validate_output(self, result_dir: str | Path, profile_name: str) -> list[str]:
        return validate_output(result_dir, profile_name=profile_name)

    def parse_result(self, result_dir: str | Path) -> dict[str, Any]:
        return parse_result(result_dir)


def default_vasp_application() -> VaspApplication | None:
    """Build an application from environment config; None when unconfigured."""
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
    return VaspApplication(
        backend,
        module_name=(
            _env("SCNET_VASP_MODULE")
            or "vasp-6.4.2-intelmpi2017_ioptcell"
        ),
        env_script=_env("SCNET_VASP_ENV_SCRIPT"),
        remote_root=config.remote_root,
        remote_psp_dir=_env("SCNET_VASP_PSP_DIR"),
    )


def _env(name: str) -> str:
    import os

    return os.environ.get(name, "").strip()
