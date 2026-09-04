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
import re
import sys
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

SERVER_NAME = "scnet-science"

mcp = FastMCP(SERVER_NAME)

_MAX_ERROR_CHARS = 512
_SENSITIVE_KEY_VALUE = re.compile(
    r"(?i)\b(?:password|token|secret|private[_-]?key|api[_-]?key)\b\s*[:=]\s*"
    r"(?:'[^']*'|\"[^\"]*\"|\S+)"
)
_ABSOLUTE_PATH = re.compile(r"(?<![\w.-])/(?:[^\s'\"\\]|\\.)+")
_PYDANTIC_INPUT_LINE = re.compile(r"(?im)^\s*input_value\s*=\s*.*$")
_PYDANTIC_INPUT_INLINE = re.compile(
    r"(?i)\binput_value\s*=\s*(?:'[^']*'|\"[^\"]*\"|[^,\]]*)"
)


def _safe_error_message(message: str) -> str:
    """Bound and redact potentially untrusted backend or validation text."""
    sanitized = _PYDANTIC_INPUT_LINE.sub(
        "input_value=<redacted>", str(message)
    )
    sanitized = _PYDANTIC_INPUT_INLINE.sub(
        "input_value=<redacted>", sanitized
    )
    sanitized = _SENSITIVE_KEY_VALUE.sub("<redacted>", sanitized)
    sanitized = _ABSOLUTE_PATH.sub("<path>", sanitized)
    sanitized = " ".join(sanitized.split())
    if len(sanitized) > _MAX_ERROR_CHARS:
        sanitized = sanitized[: _MAX_ERROR_CHARS - 1] + "…"
    return sanitized or "operation failed"


def _error(message: str, error_type: str = "missing_prerequisites") -> dict[str, Any]:
    safe_message = _safe_error_message(message)
    return {
        "is_error": True,
        "error_type": error_type,
        "message": safe_message,
        "output": safe_message,
    }

_vasp_graph: Any | None = None


def _unified_vasp_service() -> Any:
    """Return the process-lifetime VASP service (test injection only)."""
    global _vasp_graph
    if _vasp_graph is None:
        from photomatagent.scientific.applications.vasp.application import (
            default_vasp_application,
        )
        from photomatagent.scientific.applications.vasp.unified.factory import (
            build_unified_vasp_graph,
        )

        _vasp_graph = build_unified_vasp_graph(
            application=default_vasp_application(), workspace=Path.cwd()
        )
    return getattr(_vasp_graph, "service", _vasp_graph)


def _set_unified_vasp_graph_for_test(graph: Any | None) -> None:
    """Inject/reset the server graph for deterministic in-process tests."""
    global _vasp_graph
    _vasp_graph = graph


def _bounded_json(value: Any) -> Any:
    """Remove secrets and cap recursively serialized service output."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {
            str(key): _bounded_json(item)
            for key, item in list(value.items())[:64]
            if not any(secret in str(key).lower() for secret in ("password", "token", "secret", "private_key"))
        }
    if isinstance(value, list):
        return [_bounded_json(item) for item in value[:64]]
    if isinstance(value, str):
        return _safe_error_message(value) if len(value) > 0 else value
    return value


def _service_payload(result: Any) -> dict[str, Any]:
    payload = _bounded_json(result)
    if not isinstance(payload, dict):
        payload = {"data": payload}
    payload.setdefault("ok", not bool(payload.get("errors")))
    payload["is_error"] = not bool(payload["ok"])
    return payload


def _validated_plan(workflow_kind: str, scientific_spec: dict[str, Any]) -> Any:
    from photomatagent.scientific.applications.vasp.unified.models import (
        UnifiedVaspRequest,
    )

    return UnifiedVaspRequest.model_validate(
        {"workflow_kind": workflow_kind, "scientific_spec": scientific_spec}
    )


def _validated_report(report_request: dict[str, Any]) -> Any:
    from photomatagent.scientific.applications.vasp.unified.models import ReportRequest

    return ReportRequest.model_validate(report_request)


@mcp.tool()
async def vasp_capabilities(workflow_kind: str | None = None) -> dict[str, Any]:
    """List the unified VASP lifecycle capability surface."""
    try:
        return _service_payload(_unified_vasp_service().capabilities(workflow_kind))
    except Exception as exc:
        return _error(str(exc), type(exc).__name__)


@mcp.tool()
async def vasp_plan(
    workflow_kind: str, scientific_spec: dict[str, Any]
) -> dict[str, Any]:
    """Create a unified periodic, molecular, or study workflow manifest."""
    try:
        return _service_payload(_unified_vasp_service().plan(_validated_plan(workflow_kind, scientific_spec)))
    except Exception as exc:
        return _error(str(exc), type(exc).__name__)


async def _workflow_call(method: str, workflow_id: str, *args: Any) -> dict[str, Any]:
    if not workflow_id:
        return _error("workflow_id is required", "ValidationError")
    try:
        result = await getattr(_unified_vasp_service(), method)(workflow_id, *args)
        return _service_payload(result)
    except Exception as exc:
        return _error(str(exc), type(exc).__name__)


@mcp.tool()
async def vasp_prepare(workflow_id: str) -> dict[str, Any]:
    """Prepare one planned unified VASP workflow."""
    return await _workflow_call("prepare", workflow_id)


@mcp.tool()
async def vasp_preflight(workflow_id: str) -> dict[str, Any]:
    """Run deterministic preflight for one unified workflow."""
    return await _workflow_call("preflight", workflow_id)


@mcp.tool()
async def vasp_submit(workflow_id: str, stage: str | None = None) -> dict[str, Any]:
    """Submit a workflow stage through the idempotent unified lifecycle."""
    return await _workflow_call("submit", workflow_id, stage)


@mcp.tool()
async def vasp_status(workflow_id: str) -> dict[str, Any]:
    """Read unified VASP workflow status."""
    return await _workflow_call("status", workflow_id)


@mcp.tool()
async def vasp_wait(workflow_id: str) -> dict[str, Any]:
    """Check the unified lifecycle at its bounded wait boundary."""
    return await _workflow_call("wait", workflow_id)


@mcp.tool()
async def vasp_resume(workflow_id: str) -> dict[str, Any]:
    """Reconcile or resume a workflow under the unified policy."""
    return await _workflow_call("resume", workflow_id)


@mcp.tool()
async def vasp_collect(workflow_id: str) -> dict[str, Any]:
    """Collect and validate a workflow's scientific outputs."""
    return await _workflow_call("collect", workflow_id)


@mcp.tool()
async def vasp_report(
    workflow_id: str, report_request: dict[str, Any]
) -> dict[str, Any]:
    """Produce a typed report from a unified workflow."""
    try:
        result = await _unified_vasp_service().report(
            workflow_id, _validated_report(report_request)
        )
        return _service_payload(result)
    except Exception as exc:
        return _error(str(exc), type(exc).__name__)


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


def _partition_backend() -> Any | None:
    """Return the configured SCNet backend for queue discovery only.

    This is intentionally independent of the unified VASP lifecycle: queue
    discovery remains available to NAMD and MAGUS without exposing a second
    VASP execution path.
    """
    from photomatagent.scientific.applications.vasp.application import (
        default_vasp_application,
    )

    application = default_vasp_application()
    return application.backend if application is not None else None


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
    backend = _partition_backend()
    if backend is None:
        return _error(
            "SCNet is UNCONFIGURED: set SCNET_HOST / SCNET_USERNAME",
        )
    try:
        partitions = await backend.available_partitions()
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
    from photomatagent.scientific.applications.vasp.application import (
        default_vasp_application,
    )

    report: dict[str, Any] = {}
    application = default_vasp_application()
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
