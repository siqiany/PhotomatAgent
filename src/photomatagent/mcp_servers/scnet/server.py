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
# vasp_molecule.* tools (isolated-molecule DAG)
# ---------------------------------------------------------------------------


def _molecular_tool(name: str) -> Any:
    """Find one registered ``vasp_molecule.*`` Tool instance."""
    from photomatagent.scientific.applications.vasp.molecular.tool_pack import (
        molecular_vasp_pack,
    )

    pack = molecular_vasp_pack()
    for tool in pack.tools():
        if tool.name == name:
            return tool
    raise KeyError(name)


async def _call_molecular(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    tool = _molecular_tool(name)
    try:
        result = await tool.execute(arguments)
    except Exception as exc:
        return _error(
            f"{name} failed: {type(exc).__name__}: {exc}",
            error_type=type(exc).__name__,
        )
    payload = dict(result.data) if getattr(result, "data", None) else {}
    payload.setdefault("output", getattr(result, "output", ""))
    payload["is_error"] = bool(getattr(result, "is_error", False))
    return payload


@mcp.tool()
async def vasp_molecule_capabilities() -> dict[str, Any]:
    """List the isolated-molecule VASP runtime configuration (backend, pseudopotential dirs, module/environment, workspace paths) and the typed stage DAG. Read-only; never submits."""
    return await _call_molecular("vasp_molecule.capabilities", {})


@mcp.tool()
async def vasp_molecule_prepare(
    structure_path: str,
    total_charge: int,
    name: str | None = None,
    box_ang: float = 30.0,
    spin_multiplicity: int = 1,
    calculation_purpose: str = "unspecified",
    conformer_id: str | None = None,
    encut_ev: float | None = None,
    include_orbital_homo: bool = True,
    include_orbital_lumo: bool = True,
    include_esp: bool = True,
    include_hse06: bool = False,
    workflow_dir: str | None = None,
) -> dict[str, Any]:
    """Generate the isolated-molecule VASP stage tree + deterministic preflight offline. total_charge is explicit and NEVER inferred from names. Never submits."""
    arguments: dict[str, Any] = {
        "structure_path": structure_path,
        "total_charge": total_charge,
        "name": name,
        "box_ang": box_ang,
        "spin_multiplicity": spin_multiplicity,
        "calculation_purpose": calculation_purpose,
        "conformer_id": conformer_id,
        "encut_ev": encut_ev,
        "include_orbital_homo": include_orbital_homo,
        "include_orbital_lumo": include_orbital_lumo,
        "include_esp": include_esp,
        "include_hse06": include_hse06,
        "workflow_dir": workflow_dir,
    }
    arguments = {key: value for key, value in arguments.items() if value is not None}
    return await _call_molecular("vasp_molecule.prepare", arguments)


@mcp.tool()
async def vasp_molecule_preflight(workflow_dir: str) -> dict[str, Any]:
    """Run the deterministic offline molecular preflight (charge, POTCAR order, NELECT parity, vacuum, Gamma-only, DIPOL rendering, dependencies). Writes preflight.json."""
    return await _call_molecular(
        "vasp_molecule.preflight", {"workflow_dir": workflow_dir}
    )


@mcp.tool()
async def vasp_molecule_submit(
    workflow_dir: str,
    stage: str,
    wait: bool = False,
    wait_timeout_seconds: float = 3600.0,
    force_new_attempt: bool = False,
) -> dict[str, Any]:
    """Submit ONE isolated-molecule stage under the submit-once contract (unique remote dir, generated run.slurm, remote POTCAR assembly). Requires SCNET config + PHOTOMATAGENT_ALLOW_HPC_SUBMIT=1."""
    return await _call_molecular(
        "vasp_molecule.submit",
        {
            "workflow_dir": workflow_dir,
            "stage": stage,
            "wait": wait,
            "wait_timeout_seconds": wait_timeout_seconds,
            "force_new_attempt": force_new_attempt,
        },
    )


@mcp.tool()
async def vasp_molecule_status(workflow_dir: str, stage: str) -> dict[str, Any]:
    """Read lifecycle + scheduler state of one isolated-molecule stage. Query failures are UNKNOWN, never job failures."""
    return await _call_molecular(
        "vasp_molecule.status", {"workflow_dir": workflow_dir, "stage": stage}
    )


@mcp.tool()
async def vasp_molecule_collect(
    workflow_dir: str, stage: str, local_dir: str | None = None
) -> dict[str, Any]:
    """Download and validate one finished stage: COMPLETED -> COLLECTED -> VALIDATED. Evidence only when validation passes; task_state.json and the SQLite registry stay in sync."""
    arguments: dict[str, Any] = {"workflow_dir": workflow_dir, "stage": stage}
    if local_dir:
        arguments["local_dir"] = local_dir
    return await _call_molecular("vasp_molecule.collect", arguments)


@mcp.tool()
async def vasp_molecule_analyze_orbitals(
    result_dir: str,
    charge: int = 0,
    spin_multiplicity: int = 1,
    box_ang: float | None = None,
) -> dict[str, Any]:
    """HOMO/LUMO identification + vacuum alignment from EIGENVAL + LOCPOT (offline). Raw values must never be compared across molecules."""
    arguments: dict[str, Any] = {
        "result_dir": result_dir,
        "charge": charge,
        "spin_multiplicity": spin_multiplicity,
    }
    if box_ang is not None:
        arguments["box_ang"] = box_ang
    return await _call_molecular("vasp_molecule.analyze_orbitals", arguments)


@mcp.tool()
async def vasp_molecule_analyze_esp(result_dir: str) -> dict[str, Any]:
    """ESP/LOCPOT grid metadata (offline); the potential grid content stays on disk."""
    return await _call_molecular(
        "vasp_molecule.analyze_esp", {"result_dir": result_dir}
    )


@mcp.tool()
async def vasp_molecule_binding_energy(
    complex_name: str,
    complex_dir: str,
    references: list[dict[str, Any]],
    alternative_references: list[dict[str, Any]] | None = None,
    charge: int = 0,
) -> dict[str, Any]:
    """Electronic binding energy (ΔE/ΔΔE) from validated E0 values with box/functional/ENCUT consistency checks. Electronic-only; no vibrational/thermal terms are claimed."""
    arguments: dict[str, Any] = {
        "complex_name": complex_name,
        "complex_dir": complex_dir,
        "references": references,
        "charge": charge,
    }
    if alternative_references:
        arguments["alternative_references"] = alternative_references
    return await _call_molecular("vasp_molecule.binding_energy", arguments)


@mcp.tool()
async def vasp_molecule_resume_workflow(
    workflow_dir: str,
    wait: bool = True,
    collect: bool = True,
    stop_on_failure: bool = True,
    only: list[str] | None = None,
    wait_timeout_seconds: float = 3600.0,
) -> dict[str, Any]:
    """Run or resume the full isolated-molecule DAG from task_state.json. Completed stages are never resubmitted; failures block dependents."""
    arguments: dict[str, Any] = {
        "workflow_dir": workflow_dir,
        "wait": wait,
        "collect": collect,
        "stop_on_failure": stop_on_failure,
        "wait_timeout_seconds": wait_timeout_seconds,
    }
    if only:
        arguments["only"] = only
    return await _call_molecular("vasp_molecule.resume_workflow", arguments)


# ---------------------------------------------------------------------------
# vasp_study.* orchestration tools (thin adapters over the study pack)
# ---------------------------------------------------------------------------


def _study_tool(name: str) -> Any:
    """Find one registered ``vasp_study.*`` Tool instance."""
    from photomatagent.scientific.applications.vasp.study.tools import (
        vasp_study_pack,
    )

    pack = vasp_study_pack()
    for tool in pack.tools():
        if tool.name == name:
            return tool
    raise KeyError(name)


async def _call_study(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    tool = _study_tool(name)
    try:
        result = await tool.execute(arguments)
    except Exception as exc:
        return _error(
            f"{name} failed: {type(exc).__name__}: {exc}",
            error_type=type(exc).__name__,
        )
    payload = dict(result.data) if getattr(result, "data", None) else {}
    payload.setdefault("output", getattr(result, "output", ""))
    payload["is_error"] = bool(getattr(result, "is_error", False))
    return payload


@mcp.tool()
async def vasp_study_plan(
    systems: list[dict[str, Any]],
    original_request: str = "",
    study_id: str = "",
    property_requests: list[str] | None = None,
    allow_assumed_structures: bool = True,
    max_candidates_per_system: int = 3,
    user_requested_computation: bool = False,
    max_core_hours: float = 64.0,
    functional: str = "PBE-D3(BJ)",
    encut_ev: float | None = None,
    box_ang: float = 20.0,
    workspace: str = "",
) -> dict[str, Any]:
    """Compile a typed VASP study plan (offline; never submits)."""
    return await _call_study(
        "vasp_study.plan",
        {
            "systems": systems,
            "original_request": original_request,
            "study_id": study_id,
            "property_requests": property_requests or [],
            "allow_assumed_structures": allow_assumed_structures,
            "max_candidates_per_system": max_candidates_per_system,
            "user_requested_computation": user_requested_computation,
            "max_core_hours": max_core_hours,
            "functional": functional,
            "encut_ev": encut_ev,
            "box_ang": box_ang,
            "workspace": workspace,
        },
    )


@mcp.tool()
async def vasp_study_execute(
    study_id: str,
    study_dir: str = "",
    user_requested_computation: bool = False,
    wait: bool = True,
) -> dict[str, Any]:
    """Execute or resume a planned study through vasp_molecule.*."""
    return await _call_study(
        "vasp_study.execute",
        {
            "study_id": study_id,
            "study_dir": study_dir,
            "user_requested_computation": user_requested_computation,
            "wait": wait,
        },
    )


@mcp.tool()
async def vasp_study_status(study_id: str, study_dir: str = "") -> dict[str, Any]:
    """Read persisted study task/binding states."""
    return await _call_study(
        "vasp_study.status", {"study_id": study_id, "study_dir": study_dir}
    )


@mcp.tool()
async def vasp_study_resume(
    study_id: str,
    study_dir: str = "",
    user_requested_computation: bool = False,
) -> dict[str, Any]:
    """Resume an interrupted study (molecular resume semantics)."""
    return await _call_study(
        "vasp_study.resume",
        {
            "study_id": study_id,
            "study_dir": study_dir,
            "user_requested_computation": user_requested_computation,
        },
    )


@mcp.tool()
async def vasp_study_collect(study_id: str, study_dir: str = "") -> dict[str, Any]:
    """Collect + validate scheduler-COMPLETED / COLLECTED tasks."""
    return await _call_study(
        "vasp_study.collect", {"study_id": study_id, "study_dir": study_dir}
    )


@mcp.tool()
async def vasp_study_report(study_id: str, study_dir: str = "") -> dict[str, Any]:
    """Generate results.json / results.csv / figures / report.md."""
    return await _call_study(
        "vasp_study.report", {"study_id": study_id, "study_dir": study_dir}
    )


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
