"""FastMCP stdio server exposing SCNet scientific applications as tools.

Usage:
    photomatagent-mcp-scnet            # stdio MCP server
    photomatagent-mcp-scnet --doctor   # diagnostic dump (no MCP handshake)

Environment (expanded by the MCP gateway from the process env / workspace
``.env``):
    SCNET_HOST / SCNET_USERNAME / SCNET_PORT / SCNET_PRIVATE_KEY_PATH /
    SCNET_REMOTE_ROOT / PHOTOMATAGENT_ALLOW_HPC_SUBMIT

The server starts even when SCNet is unconfigured; tools then return typed
``missing_prerequisites`` diagnostics instead of hallucinating results.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from fastmcp import FastMCP

from photomatagent.scientific.applications.vasp.application import (
    VaspApplication,
    default_vasp_application,
)

SERVER_NAME = "scnet-science"

mcp = FastMCP(SERVER_NAME)


def _application() -> VaspApplication | None:
    return default_vasp_application()


def _result_payload(result: Any) -> dict[str, Any]:
    """Convert a ScientificToolResult into a JSON-safe MCP payload."""
    payload = dict(result.data) if getattr(result, "data", None) else {}
    payload.setdefault("output", getattr(result, "output", ""))
    payload["is_error"] = bool(getattr(result, "is_error", False))
    return payload


def _error(message: str, error_type: str = "missing_prerequisites") -> dict[str, Any]:
    return {
        "is_error": True,
        "error_type": error_type,
        "message": message,
        "output": message,
    }


def _vasp_or_error() -> tuple[VaspApplication | None, dict[str, Any] | None]:
    application = _application()
    if application is None:
        return None, _error(
            "VASP is UNCONFIGURED: set SCNET_HOST / SCNET_USERNAME "
            "(or SUPERCOMPUTING_HOST / SUPERCOMPUTING_USERNAME)"
        )
    return application, None


# ---------------------------------------------------------------------------
# vasp.* tools (deferred application surface)
# ---------------------------------------------------------------------------


@mcp.tool()
async def vasp_capabilities() -> dict[str, Any]:
    """List VASP capabilities: profiles, SOC support, backend state, POTCAR policy, resource limits. Read-only; never submits."""
    application, error = _vasp_or_error()
    if error:
        return error
    from photomatagent.scientific.applications.vasp.profiles import profiles

    payload = await application.probe_environment_async()  # type: ignore[union-attr]
    payload["profiles"] = [
        {
            "name": profile.name,
            "description": profile.description,
            "soc": profile.soc,
            "stages": profile.stages,
            "needs_configuration": profile.needs_configuration,
        }
        for profile in profiles()
    ]
    payload["cost_class"] = "EXPENSIVE"
    return payload


@mcp.tool()
async def vasp_prepare(
    structure_path: str,
    profile: str,
    output_dir: str | None = None,
    encut_ev: float | None = None,
    kpoint_density: int | None = None,
    kpoint_grid: list[int] | None = None,
) -> dict[str, Any]:
    """Prepare VASP inputs for a profile from a structure file (CIF/POSCAR). Generates POSCAR/INCAR/KPOINTS + POTCAR.policy and workflow.json locally. NEVER submits. POTCAR is resolved at submit time from PMG_VASP_PSP_DIR or a remote pseudopotential location."""
    application, error = _vasp_or_error()
    if error:
        return error
    overrides: dict[str, Any] = {}
    if encut_ev is not None:
        overrides["encut_ev"] = encut_ev
    if kpoint_density is not None:
        overrides["kpoint_density"] = kpoint_density
    if kpoint_grid is not None:
        overrides["kpoint_grid"] = kpoint_grid
    try:
        manifest = application.prepare_inputs(  # type: ignore[union-attr]
            structure_path=structure_path,
            profile_name=profile,
            output_dir=output_dir,
            spec_overrides=overrides,
        )
    except Exception as exc:
        return _error(
            f"vasp.prepare failed: {type(exc).__name__}: {exc}",
            error_type=type(exc).__name__,
        )
    manifest["submitted"] = False
    return manifest


@mcp.tool()
async def vasp_submit(
    job_name: str,
    input_dir: str,
    profile: str,
    partition: str | None = None,
    nodes: int = 1,
    tasks_per_node: int = 32,
    walltime_minutes: int = 240,
) -> dict[str, Any]:
    """Submit one prepared VASP stage directory to SCNet via Slurm. Requires PHOTOMATAGENT_ALLOW_HPC_SUBMIT=1 and a passing resource policy. Returns a detached job ref; poll with vasp_status."""
    from photomatagent.scientific.remote.models import ResourceRequest

    application, error = _vasp_or_error()
    if error:
        return error
    try:
        selected_partition = partition or os.environ.get("SCNET_PARTITION", "").strip()
        if not selected_partition:
            return _error(
                "partition is required: call scnet_partitions (SCNet "
                "whichpartition) and pass one result, or set SCNET_PARTITION",
                error_type="missing_partition",
            )
        ref = await application.submit_stage(  # type: ignore[union-attr]
            job_name=job_name,
            input_dir=input_dir,
            profile_name=profile,
            resource=ResourceRequest(
                partition=selected_partition,
                nodes=nodes,
                tasks_per_node=tasks_per_node,
                walltime_minutes=walltime_minutes,
            ),
            unique_remote_directory=True,
        )
    except Exception as exc:
        return _error(
            f"vasp.submit refused: {type(exc).__name__}: {exc}",
            error_type=type(exc).__name__,
        )
    payload = ref.model_dump(mode="json")
    payload["note"] = (
        "detached job: poll with vasp_status; Slurm COMPLETED is not "
        "scientific validity -- collect with vasp_collect"
    )
    return payload


@mcp.tool()
async def vasp_status(job_id: str) -> dict[str, Any]:
    """Query the Slurm state of a VASP job id (PENDING/RUNNING/COMPLETED/FAILED/...). Scheduler state only; COMPLETED does not imply scientific validity."""
    application, error = _vasp_or_error()
    if error:
        return error
    try:
        state = await application.status(job_id)  # type: ignore[union-attr]
    except Exception as exc:
        return _error(f"vasp.status failed: {type(exc).__name__}: {exc}")
    return {
        "job_id": job_id,
        "state": state.value,
        "terminal": state.terminal,
        "note": "scheduler state only; use vasp_collect for validation",
    }


@mcp.tool()
async def vasp_collect(
    job_id: str, remote_directory: str, profile: str, local_dir: str | None = None
) -> dict[str, Any]:
    """Download a finished VASP job's results, validate the vasprun.xml contract, and parse bounded values into evidence. Returns validation problems explicitly."""
    from photomatagent.scientific.remote.models import RemoteJobRef

    application, error = _vasp_or_error()
    if error:
        return error
    job_ref = RemoteJobRef(
        backend="scnet",
        application="vasp",
        job_id=job_id,
        remote_directory=remote_directory,
    )
    try:
        report = await application.collect(  # type: ignore[union-attr]
            job_ref=job_ref,
            local_dir=local_dir or "output/vasp_results",
            profile_name=profile,
        )
    except Exception as exc:
        return _error(f"vasp.collect failed: {type(exc).__name__}: {exc}")
    return report


@mcp.tool()
async def vasp_inspect_result(result_dir: str, profile: str) -> dict[str, Any]:
    """Validate and parse a local VASP result directory (vasprun.xml + OUTCAR) without remote contact."""
    application, error = _vasp_or_error()
    if error:
        return error
    problems = application.validate_output(result_dir, profile_name=profile)  # type: ignore[union-attr]
    parsed = application.parse_result(result_dir)  # type: ignore[union-attr]
    return {
        "result_dir": result_dir,
        "profile": profile,
        "validation_problems": problems,
        "scientifically_valid": not problems,
        "parsed": parsed,
    }


@mcp.tool()
async def vasp_run_workflow(
    workflow_dir: str,
    profile: str,
    poll_interval_seconds: float = 30.0,
    timeout_seconds: float = 86400.0,
) -> dict[str, Any]:
    """Bounded convenience API: submit every prepared stage sequentially, wait for each job, collect and validate. Production use should prefer detached jobs."""
    application, error = _vasp_or_error()
    if error:
        return error
    from photomatagent.scientific.applications.vasp.workflow import (
        run_vasp_workflow,
    )
    from pathlib import Path

    try:
        report = await run_vasp_workflow(
            application=application,
            workflow_dir=Path(workflow_dir),
            profile_name=profile,
            poll_interval_seconds=poll_interval_seconds,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        return _error(f"vasp.run_workflow failed: {type(exc).__name__}: {exc}")
    return report


# ---------------------------------------------------------------------------
# namd.* tools
# ---------------------------------------------------------------------------


def _namd_application() -> Any:
    from photomatagent.scientific.applications.namd.application import (
        default_namd_application,
    )

    return default_namd_application()


def _namd_or_error() -> tuple[Any | None, dict[str, Any] | None]:
    application = _namd_application()
    if application is None:
        return None, _error(
            "Hefei-NAMD is UNCONFIGURED: set SCNET_HOST / SCNET_USERNAME / "
            "SCNET_NAMD_MODULE"
        )
    return application, None


@mcp.tool()
async def namd_capabilities() -> dict[str, Any]:
    """Probe the SCNet Hefei-NAMD environment: module availability, supported workflow (VASP AIMD trajectory + per-snapshot WAVECARs), required VASP artifacts. Never fabricates carrier-dynamics numbers."""
    application, error = _namd_or_error()
    if error:
        return error
    assert application is not None
    return await application.probe_environment_async()


@mcp.tool()
async def namd_validate_inputs(trajectory_dir: str) -> dict[str, Any]:
    """Validate a VASP AIMD trajectory tree for Hefei-NAMD: reference POSCAR + XDATCAR + OUTCAR, per-snapshot POSCAR/WAVECAR/OUTCAR, identical WAVECAR sizes."""
    application, error = _namd_or_error()
    if error:
        return error
    assert application is not None
    problems = application.validate_inputs(trajectory_dir)
    return {"trajectory_dir": trajectory_dir, "problems": problems, "valid": not problems}


@mcp.tool()
async def namd_prepare(
    trajectory_dir: str,
    output_dir: str,
    inp_path: str | None = None,
    inicon_path: str | None = None,
    parameters: dict[str, Any] | None = None,
    initial_conditions: list[list[int]] | None = None,
) -> dict[str, Any]:
    """Prepare a complete Hefei-NAMD tree. Supply version-matched inp/INICON paths, or explicit NAMDPARA values plus [[start_step, band], ...] initial conditions. Preserves run/NNNN/WAVECAR directories."""
    application, error = _namd_or_error()
    if error:
        return error
    assert application is not None
    try:
        return application.prepare(
            trajectory_dir=trajectory_dir,
            output_dir=output_dir,
            inp_path=inp_path,
            inicon_path=inicon_path,
            parameters=parameters,
            initial_conditions=initial_conditions,
        )
    except Exception as exc:
        return _error(f"namd.prepare failed: {type(exc).__name__}: {exc}")


@mcp.tool()
async def namd_submit(
    job_name: str,
    prepared_dir: str,
    partition: str | None = None,
    nodes: int = 1,
    tasks_per_node: int = 32,
    walltime_minutes: int = 720,
) -> dict[str, Any]:
    """Submit a runnable Hefei-NAMD tree to SCNet. Requires inp, INICON, preserved run/ snapshots, SCNET_NAMD_MODULE, an explicit/SCNET partition, and PHOTOMATAGENT_ALLOW_HPC_SUBMIT=1."""
    from photomatagent.scientific.remote.models import ResourceRequest

    application, error = _namd_or_error()
    if error:
        return error
    assert application is not None
    selected_partition = partition or os.environ.get("SCNET_PARTITION", "").strip()
    if not selected_partition:
        return _error(
            "partition is required: call scnet_partitions and pass one "
            "result, or set SCNET_PARTITION",
            error_type="missing_partition",
        )
    try:
        ref = await application.submit(
            job_name=job_name,
            prepared_dir=prepared_dir,
            resource=ResourceRequest(
                partition=selected_partition,
                nodes=nodes,
                tasks_per_node=tasks_per_node,
                walltime_minutes=walltime_minutes,
            ),
        )
    except Exception as exc:
        return _error(
            f"namd.submit refused: {type(exc).__name__}: {exc}",
            error_type=type(exc).__name__,
        )
    payload = ref.model_dump(mode="json")
    payload["note"] = "detached job; poll with namd_status"
    return payload


@mcp.tool()
async def namd_status(job_id: str) -> dict[str, Any]:
    """Query the Slurm state of a Hefei-NAMD job."""
    application, error = _namd_or_error()
    if error:
        return error
    assert application is not None
    try:
        state = await application.status(job_id)
    except Exception as exc:
        return _error(f"namd.status failed: {type(exc).__name__}: {exc}")
    return {"job_id": job_id, "state": state.value, "terminal": state.terminal}


@mcp.tool()
async def namd_collect(
    job_id: str, remote_directory: str, local_dir: str | None = None
) -> dict[str, Any]:
    """Download real Hefei-NAMD NATXT/EIGTXT/COUPCAR/PSICT.*/SHPROP.* outputs."""
    from photomatagent.scientific.remote.models import RemoteJobRef

    application, error = _namd_or_error()
    if error:
        return error
    assert application is not None
    ref = RemoteJobRef(
        backend="scnet",
        application="hefei-namd",
        job_id=job_id,
        remote_directory=remote_directory,
    )
    try:
        return await application.collect(
            job_ref=ref, local_dir=local_dir or "output/namd_results"
        )
    except Exception as exc:
        return _error(f"namd.collect failed: {type(exc).__name__}: {exc}")


@mcp.tool()
async def namd_inspect_result(result_dir: str) -> dict[str, Any]:
    """List bounded local Hefei-NAMD result artifacts without inventing derived values."""
    from pathlib import Path

    root = Path(result_dir).expanduser().resolve()
    files = [
        {"name": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ] if root.is_dir() else []
    return {"result_dir": str(root), "files": files[:500], "file_count": len(files)}


@mcp.tool()
async def scnet_partitions() -> dict[str, Any]:
    """Read-only queue discovery using SCNet whichpartition (sinfo fallback)."""
    application, error = _vasp_or_error()
    if error:
        return error
    assert application is not None and application.backend is not None
    try:
        partitions = await application.backend.available_partitions()
    except Exception as exc:
        return _error(f"partition discovery failed: {type(exc).__name__}: {exc}")
    return {
        "partitions": partitions,
        "configured_default": os.environ.get("SCNET_PARTITION", "").strip() or None,
        "note": "select a partition returned by the connected SCNet center",
    }


# ---------------------------------------------------------------------------
# magus.* tools
# ---------------------------------------------------------------------------


def _magus_application() -> Any:
    from photomatagent.scientific.applications.magus.application import (
        default_magus_application,
    )

    return default_magus_application()


@mcp.tool()
async def magus_capabilities() -> dict[str, Any]:
    """Probe the remote SCNet MAGUS installation (root, executable, version, commands, calculators, structure types, JOB_SYSTEM, VASP/pseudopotential readiness). Read-only; never submits."""
    application = _magus_application()
    if application is None:
        return _error(
            "MAGUS is UNCONFIGURED: set SCNET_HOST / SCNET_USERNAME / "
            "SCNET_PRIVATE_KEY_PATH (and SCNET_MAGUS_ROOT)",
            error_type="UNCONFIGURED",
        )
    return await application.probe_environment_async()


@mcp.tool()
async def magus_prepare_generate(
    composition: str,
    job_dir: str,
    structure_type: str = "bulk",
    number: int = 5,
    min_atoms: int | None = None,
    max_atoms: int | None = None,
) -> dict[str, Any]:
    """Prepare a MAGUS structure-generation job (input.yaml + magus.slurm + photomat_manifest.json); NEVER submits."""
    application = _magus_application()
    if application is None:
        return _error("MAGUS is UNCONFIGURED", error_type="UNCONFIGURED")
    from photomatagent.scientific.applications.magus.models import (
        MagusGenerateRequest,
    )

    try:
        request = MagusGenerateRequest.from_composition(
            composition,
            structure_type=structure_type,  # type: ignore[arg-type]
            number=number,
            min_atoms=min_atoms,
            max_atoms=max_atoms,
        )
        return application.prepare_generate(request, job_dir)
    except Exception as exc:
        return _error(
            f"magus.prepare_generate failed: {type(exc).__name__}: {exc}",
            error_type="INVALID_REQUEST",
        )


@mcp.tool()
async def magus_prepare_search(
    composition: str,
    job_dir: str,
    structure_type: str = "bulk",
    calculator: str = "vasp",
    init_size: int = 4,
    population_size: int = 4,
    generations: int = 1,
    save_good: int = 2,
    pressure_gpa: float = 0.0,
    min_atoms: int | None = None,
    max_atoms: int | None = None,
    slab: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare a constrained MAGUS structure-search job (input.yaml + inputFold/VASP/INCAR + magus.slurm + manifest); NEVER submits."""
    application = _magus_application()
    if application is None:
        return _error("MAGUS is UNCONFIGURED", error_type="UNCONFIGURED")
    from photomatagent.scientific.applications.magus.models import (
        MagusSearchRequest,
    )

    try:
        request = MagusSearchRequest.from_composition(
            composition,
            structure_type=structure_type,  # type: ignore[arg-type]
            calculator=calculator,  # type: ignore[arg-type]
            init_size=init_size,
            population_size=population_size,
            generations=generations,
            save_good=save_good,
            pressure_gpa=pressure_gpa,
            min_atoms=min_atoms,
            max_atoms=max_atoms,
        )
        if slab is not None:
            from photomatagent.scientific.applications.magus.models import (
                MagusSlabConfig,
            )

            request.slab = MagusSlabConfig(**slab)
        return application.prepare_search(request, job_dir)
    except Exception as exc:
        return _error(
            f"magus.prepare_search failed: {type(exc).__name__}: {exc}",
            error_type="INVALID_REQUEST",
        )


@mcp.tool()
async def magus_submit(
    job_name: str,
    prepared_dir: str,
    partition: str | None = None,
    nodes: int = 1,
    tasks_per_node: int = 8,
    walltime_minutes: int = 120,
) -> dict[str, Any]:
    """Submit a prepared MAGUS job tree to SCNet via Slurm (requires PHOTOMATAGENT_ALLOW_HPC_SUBMIT=1 + resource policy)."""
    from photomatagent.scientific.remote.models import ResourceRequest

    application = _magus_application()
    if application is None:
        return _error("MAGUS is UNCONFIGURED", error_type="UNCONFIGURED")
    try:
        selected_partition = partition or os.environ.get("SCNET_PARTITION", "").strip()
        if not selected_partition:
            return _error(
                "partition is required: call scnet_partitions and pass one "
                "result, or set SCNET_PARTITION",
                error_type="missing_partition",
            )
        ref = await application.submit(
            job_name=job_name,
            prepared_dir=prepared_dir,
            resource=ResourceRequest(
                partition=selected_partition,
                nodes=nodes,
                tasks_per_node=tasks_per_node,
                walltime_minutes=walltime_minutes,
            ),
        )
    except Exception as exc:
        return _error(
            f"magus.submit refused: {type(exc).__name__}: {exc}",
            error_type=type(exc).__name__,
        )
    payload = ref.model_dump(mode="json")
    payload["note"] = "detached job; poll with magus_status"
    return payload


@mcp.tool()
async def magus_status(job_id: str) -> dict[str, Any]:
    """Query the Slurm state of a MAGUS job id (scheduler state only)."""
    application = _magus_application()
    if application is None:
        return _error("MAGUS is UNCONFIGURED", error_type="UNCONFIGURED")
    try:
        state = await application.status(job_id)
    except Exception as exc:
        return _error(f"magus.status failed: {type(exc).__name__}: {exc}")
    return {"job_id": job_id, "state": state.value, "terminal": state.terminal}


@mcp.tool()
async def magus_collect(
    job_id: str, remote_directory: str, local_dir: str | None = None
) -> dict[str, Any]:
    """Download a finished MAGUS job's bounded artifacts and summarize candidates."""
    from photomatagent.scientific.remote.models import RemoteJobRef

    application = _magus_application()
    if application is None:
        return _error("MAGUS is UNCONFIGURED", error_type="UNCONFIGURED")
    job_ref = RemoteJobRef(
        backend="scnet",
        application="magus",
        job_id=job_id,
        remote_directory=remote_directory,
    )
    try:
        return await application.collect(
            job_ref=job_ref, local_dir=local_dir or "output/magus_results"
        )
    except Exception as exc:
        return _error(f"magus.collect failed: {type(exc).__name__}: {exc}")


@mcp.tool()
async def magus_inspect_results(
    result_dir: str, operation: str = "generate"
) -> dict[str, Any]:
    """Parse a collected MAGUS result directory (bounded summary + candidates)."""
    application = _magus_application()
    if application is None:
        application = _bare_magus_application()
    try:
        return application.inspect_results(result_dir, operation=operation)
    except Exception as exc:
        return _error(f"magus.inspect_results failed: {type(exc).__name__}: {exc}")


@mcp.tool()
async def magus_search_bulk(
    composition: str, target_dir: str, output_dir: str
) -> dict[str, Any]:
    """DEPRECATED prepare-only alias of magus_prepare_search (bulk, VASP calculator, serial)."""
    application = _magus_application()
    if application is None:
        return _error("MAGUS is UNCONFIGURED", error_type="UNCONFIGURED")
    try:
        return application.prepare(
            search_type="bulk",
            composition=composition,
            target_dir=target_dir,
            output_dir=output_dir,
        )
    except Exception as exc:
        return _error(
            f"magus.search_bulk failed: {type(exc).__name__}: {exc}",
            error_type=type(exc).__name__,
        )


def _bare_magus_application() -> Any:
    from photomatagent.scientific.applications.magus.application import (
        MagusApplication,
    )

    return MagusApplication()


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


async def build_doctor_report() -> dict[str, Any]:
    """Read-only SCNet diagnostics (no MCP needed; used by --doctor)."""
    from photomatagent.scientific.applications.namd.application import (
        NamdApplication,
    )

    report: dict[str, Any] = {}
    application = _application()
    if application is not None:
        try:
            assert application.backend is not None
            report["vasp"] = await application.probe_environment_async()
            report["scnet"] = await application.backend.doctor()
        except Exception as exc:
            report["vasp"] = {"error": f"{type(exc).__name__}: {exc}"}
    else:
        report["vasp"] = {"connected": "false", "error": "no backend configured"}
    try:
        namd = _namd_application()
        report["namd"] = (
            await namd.probe_environment_async()
            if namd is not None
            else {"status": "UNCONFIGURED", "error": "no backend configured"}
        )
    except Exception as exc:
        report["namd"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        magus = _magus_application()
        if magus is None:
            report["magus"] = {
                "status": "UNCONFIGURED",
                "error_type": "UNCONFIGURED",
            }
        else:
            report["magus"] = await magus.probe_environment_async()
    except Exception as exc:
        report["magus"] = {"error": f"{type(exc).__name__}: {exc}"}
    return report


@mcp.tool()
async def scnet_doctor() -> dict[str, Any]:
    """Read-only SCNet diagnostics: SSH connection, Slurm, VASP/NAMD/MAGUS environment probes. Never exposes private keys or tokens."""
    return await build_doctor_report()


def main() -> None:
    """Entry point: run the MCP stdio server (or --doctor dump)."""
    # The MCP gateway expands workspace .env values itself. Direct doctor
    # mode loads them only when explicitly requested, so an unconfigured
    # diagnostic never makes an unexpected network connection.
    if "--load-dotenv" in sys.argv:
        try:
            from dotenv import load_dotenv

            load_dotenv(os.path.join(os.getcwd(), ".env"), override=False)
        except Exception:
            pass
        sys.argv.remove("--load-dotenv")
    if "--doctor" in sys.argv:
        report = asyncio.run(build_doctor_report())
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
