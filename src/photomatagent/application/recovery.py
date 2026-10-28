"""Conservative recovery decisions for interrupted local workbench runs."""

from __future__ import annotations

from enum import StrEnum
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from photomatagent.application.models import RunRecord
from photomatagent.application.store import WorkbenchStore
from photomatagent.scientific.remote.registry import JobLifecycleState, JobRegistry


class RecoveryStatus(StrEnum):
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"
    NO_RECOVERY_NEEDED = "NO_RECOVERY_NEEDED"


class RecoveryConclusion(BaseModel):
    run_id: str
    status: RecoveryStatus
    message: str
    external_job_ids: list[str] = []


def inspect_recovery(
    store: WorkbenchStore, run: RunRecord, *, workspace_root: Path | str | None = None
) -> RecoveryConclusion:
    """Inspect durable facts only; never resubmit an external computation here."""
    job_ids: list[str] = []
    request_ids: set[str] = set()
    for event in store.events_after(run.task_id, 0):
        if event.run_id != run.run_id:
            continue
        if event.type != "tool_completed":
            continue
        output = event.payload.get("output")
        if not isinstance(output, str):
            continue
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            continue
        for key, value in _identifiers(parsed):
            if key == "request_id":
                request_ids.add(value)
            elif value not in job_ids:
                job_ids.append(value)

    root = Path(workspace_root).resolve() if workspace_root is not None else Path.cwd()
    for path in (
        root / ".photomatagent" / "state" / "jobs.sqlite3",
        root / ".photomatagent" / "vasp" / "jobs.sqlite3",
    ):
        if not path.is_file():
            continue
        with JobRegistry(path) as registry:
            for record in registry.list():
                linked = (
                    record.request_id in request_ids
                    or record.provenance.get("run_id") == run.run_id
                    or record.provenance.get("task_id") == run.task_id
                    or record.state in {
                        JobLifecycleState.SUBMITTING,
                        JobLifecycleState.SUBMITTED,
                        JobLifecycleState.PENDING,
                        JobLifecycleState.RUNNING,
                        # Scheduler completion alone has not validated artifacts.
                        JobLifecycleState.COMPLETED,
                        # Collection can persist even when result validation fails.
                        JobLifecycleState.COLLECTED,
                        JobLifecycleState.UNKNOWN_RECONCILIATION_REQUIRED,
                    }
                )
                if not linked:
                    continue
                request_ids.add(record.request_id)
                if record.job_id and record.job_id not in job_ids:
                    job_ids.append(record.job_id)
    for request_id in sorted(request_ids):
        if request_id not in job_ids:
            job_ids.append(request_id)
    if job_ids:
        return RecoveryConclusion(
            run_id=run.run_id,
            status=RecoveryStatus.RECONCILIATION_REQUIRED,
            message="External computation requires authoritative scheduler reconciliation before continuation.",
            external_job_ids=job_ids,
        )
    return RecoveryConclusion(
        run_id=run.run_id,
        status=RecoveryStatus.NO_RECOVERY_NEEDED,
        message="No external computation was recorded; create a new run to continue.",
    )


def _identifiers(value: Any) -> list[tuple[str, str]]:
    """Read identifiers from bounded SubmitOnce tool output envelopes."""
    if isinstance(value, dict):
        found: list[tuple[str, str]] = []
        for key, item in value.items():
            if key in {"request_id", "job_id"} and isinstance(item, str) and item:
                found.append((key, item))
            elif isinstance(item, (dict, list)):
                found.extend(_identifiers(item))
        return found
    if isinstance(value, list):
        return [pair for item in value for pair in _identifiers(item)]
    return []
