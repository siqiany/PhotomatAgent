"""Sequential multi-stage VASP workflow runner (bounded convenience API).

The core PhotoMatAgent contract is detached prepare -> submit -> status ->
collect. ``run_vasp_workflow`` exists for small, fully-authorized smoke runs:
it submits stages one at a time, waits each Slurm job to a terminal state,
collects, validates, and stops at the first invalid stage. It never polls
longer than ``timeout_seconds`` and never submits without the resource
policy authorization (enforced by the backend).
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from pathlib import Path
from typing import Any

from photomatagent.scientific.remote.models import HPCJobState

TERMINAL_SUCCESS = {HPCJobState.COMPLETED}
TERMINAL_FAILURE = {
    HPCJobState.FAILED,
    HPCJobState.CANCELLED,
    HPCJobState.TIMEOUT,
    HPCJobState.OUT_OF_MEMORY,
    HPCJobState.NODE_FAIL,
}


def _prepare_stage_dependencies(
    stage: dict[str, Any], completed: dict[str, Path]
) -> None:
    """Copy upstream CONTCAR -> POSCAR and CHGCAR into the stage directory."""
    dependency = stage.get("depends_on")
    if not dependency:
        return
    upstream = completed[dependency]
    stage_dir = Path(stage["directory"])
    for filename in stage.get("required_outputs", []):
        source = upstream / filename
        if not source.is_file():
            raise FileNotFoundError(f"required upstream output missing: {source}")
        target_name = "POSCAR" if filename == "CONTCAR" else filename
        shutil.copy2(source, stage_dir / target_name)


async def run_vasp_workflow(
    *,
    application: Any,
    workflow_dir: Path,
    profile_name: str,
    poll_interval_seconds: float = 30.0,
    timeout_seconds: float = 86400.0,
) -> dict[str, Any]:
    """Execute a prepared workflow; returns per-stage records."""
    manifest_path = workflow_dir / "workflow.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"workflow.json missing in {workflow_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stages = manifest["stages"]
    stage_records: list[dict[str, Any]] = []
    completed: dict[str, Path] = {}
    started = time.monotonic()
    for stage in stages:
        _prepare_stage_dependencies(stage, completed)
        stage_dir = Path(stage["directory"])
        job_ref = await application.submit_stage(
            job_name=f"{workflow_dir.name}-{stage['stage']}",
            input_dir=stage_dir,
            profile_name=profile_name,
        )
        scheduler_state = HPCJobState.SUBMITTED
        while True:
            scheduler_state = await application.status(job_ref.job_id)
            if scheduler_state.terminal:
                break
            if time.monotonic() - started >= timeout_seconds:
                scheduler_state = HPCJobState.TIMEOUT
                break
            await asyncio.sleep(poll_interval_seconds)
        result_dir = workflow_dir / "results" / stage["stage"]
        report = await application.collect(
            job_ref=job_ref,
            local_dir=result_dir,
            profile_name=stage["stage"],
        )
        stage_records.append(
            {
                "stage": stage["stage"],
                "job_id": job_ref.job_id,
                "scheduler_state": scheduler_state.value,
                "validation_problems": report["validation_problems"],
                "scientifically_valid": report["scientifically_valid"],
            }
        )
        if scheduler_state not in TERMINAL_SUCCESS:
            raise RuntimeError(
                f"stage {stage['stage']} ended as {scheduler_state.value}"
            )
        if not report["scientifically_valid"]:
            raise RuntimeError(
                f"stage {stage['stage']} failed scientific validation: "
                + "; ".join(report["validation_problems"])
            )
        completed[stage["stage"]] = result_dir
    return {
        "workflow": workflow_dir.name,
        "profile": profile_name,
        "stages": stage_records,
        "all_valid": True,
    }
