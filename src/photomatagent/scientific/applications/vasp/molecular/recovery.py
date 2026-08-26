"""Deterministic typed recovery for relax stages (rule-based, never LLM-INCAR).

Recovery decisions come from a closed decision table: a failure class maps
to an artifact to restart from and a small set of typed INCAR changes. The
LLM never free-edits INCAR; every attempt gets its own attempt_id/remote
directory, and auto-retries are bounded by ``RecoveryPolicy.max_auto_attempts``.

Classes covered (also the decision table in
``skills/vasp-hpc-operator/references/convergence-and-recovery.md``):
NSW_EXHAUSTED, WALLTIME, FORCE_PLATEAU, LINE_SEARCH_EXCURSION, OOM,
SCF_NOT_CONVERGED, AMBIGUOUS_SUBMISSION, STATUS_QUERY_FAILED, STATUS_UNKNOWN.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from photomatagent.scientific.applications.vasp.molecular.render import (
    render_incar,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RecoveryFailure(str, Enum):
    """Deterministic failure classes used by the decision table."""

    NSW_EXHAUSTED = "NSW_EXHAUSTED"
    WALLTIME = "WALLTIME"
    FORCE_PLATEAU = "FORCE_PLATEAU"
    LINE_SEARCH_EXCURSION = "LINE_SEARCH_EXCURSION"
    OOM = "OOM"
    SCF_NOT_CONVERGED = "SCF_NOT_CONVERGED"
    AMBIGUOUS_SUBMISSION = "AMBIGUOUS_SUBMISSION"
    STATUS_QUERY_FAILED = "STATUS_QUERY_FAILED"
    STATUS_UNKNOWN = "STATUS_UNKNOWN"
    NONE = "NONE"


class RelaxAttempt(BaseModel):
    """One typed recovery attempt of a relax stage."""

    attempt_id: str
    retry_index: int = 0
    retry_of_request_id: str = ""
    request_id: str = ""
    job_id: str = ""
    remote_directory: str = ""
    restart_from: str = ""  # "CONTCAR" | "XDATCAR_BEST" | ""
    parameter_changes: list[str] = Field(default_factory=list)
    practical_convergence: bool = False
    recorded_at: str = Field(default_factory=_now)
    provenance: dict[str, Any] = Field(default_factory=dict)


class RecoveryPolicy(BaseModel):
    """Bounded, typed recovery policy (no free parameter editing)."""

    max_auto_attempts: int = Field(default=1, ge=0)
    allow_potim_reduction: bool = True
    allow_ibrion_switch: bool = False
    allow_ediffg_relaxation: bool = True
    ediffg_relax_factor: float = Field(default=2.0, gt=1.0)
    require_contcar_for_restart: bool = True


class RecoveryDecision(BaseModel):
    """One deterministic recovery decision."""

    action: str  # RESUBMIT | STOP | RECONCILE | STATUS_ONLY
    failure: str = RecoveryFailure.NONE.value
    restart_from: str = ""  # "CONTCAR" | "XDATCAR_BEST" | ""
    parameter_changes: list[str] = Field(default_factory=list)
    incar_changes: dict[str, Any] = Field(default_factory=dict)
    new_attempt_id: str = ""
    reason: str = ""
    practical_convergence: bool = False
    practical_convergence_note: str = ""
    provenance: dict[str, Any] = Field(default_factory=dict)


def _next_attempt_id(retry_index: int) -> str:
    return f"relax-attempt-{retry_index + 1}"


def outcar_force_history(path: str | Path) -> list[float]:
    """Per-ionic-step max forces from OUTCAR (``FORCES: max atom, RMS`` rows,
    with a per-block fallback from TOTAL-FORCE blocks)."""
    maxima: list[float] = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "FORCES: max atom, RMS" in line:
                tokens = line.split()
                match = [
                    float(token)
                    for token in tokens
                    if _is_number(token)
                ]
                if match:
                    maxima.append(match[0])
    if maxima:
        return maxima
    # Fallback: max per-row norm of every TOTAL-FORCE block.
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    blocks, _ = _parse_total_force_blocks(text)
    return [float(max(block)) for block in blocks if block]


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _parse_total_force_blocks(text: str) -> tuple[list[list[float]], int]:
    blocks: list[list[float]] = []
    current: list[float] = []
    in_block = False
    block_count = 0
    for line in text.splitlines():
        if "TOTAL-FORCE" in line:
            in_block = True
            current = []
            block_count += 1
            continue
        if in_block:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("-"):
                if current:
                    blocks.append(current)
                    in_block = False
                continue
            tokens = stripped.split()
            if len(tokens) >= 6:
                try:
                    fx, fy, fz = (
                        float(tokens[3]),
                        float(tokens[4]),
                        float(tokens[5]),
                    )
                    current.append(float((fx * fx + fy * fy + fz * fz) ** 0.5))
                except ValueError:
                    continue
    if in_block and current:
        blocks.append(current)
    return blocks, block_count


def classify_relax_failure(
    *,
    convergence: dict[str, Any] | None = None,
    lifecycle_state: str | None = None,
    scheduler_state: str | None = None,
    query_failed: bool = False,
    force_history: list[float] | None = None,
) -> RecoveryFailure:
    """Map observed state onto one closed failure class (deterministic)."""
    convergence = dict(convergence or {})
    if query_failed or lifecycle_state == "STATUS_QUERY_FAILED":
        return RecoveryFailure.STATUS_QUERY_FAILED
    if lifecycle_state in {
        "UNKNOWN_RECONCILIATION_REQUIRED",
        "UNKNOWN",
    } or (
        lifecycle_state is None
        and scheduler_state in {"UNKNOWN", "UNKNOWN_RECONCILIATION_REQUIRED"}
    ):
        return RecoveryFailure.AMBIGUOUS_SUBMISSION
    if lifecycle_state in {"TIMEOUT"} or scheduler_state in {"TIMEOUT"}:
        return RecoveryFailure.WALLTIME
    if lifecycle_state in {"OUT_OF_MEMORY"} or scheduler_state in {
        "OUT_OF_MEMORY"
    }:
        return RecoveryFailure.OOM
    detected = [str(item) for item in convergence.get("detected_errors", [])]
    if any(
        token in detected
        for token in ("out of memory", "cannot allocate", "allocation would exceed")
    ):
        return RecoveryFailure.OOM
    if convergence.get("exhausted_nsw"):
        return RecoveryFailure.NSW_EXHAUSTED
    if convergence.get("electronic_converged") is False:
        return RecoveryFailure.SCF_NOT_CONVERGED
    history = list(force_history or [])
    if len(history) >= 3:
        tail = history[-3:]
        spread = max(tail) - min(tail)
        scale = max(1e-9, abs(float(history[-1])))
        if spread <= 0.1 * scale and history[-1] > 1e-4:
            return RecoveryFailure.FORCE_PLATEAU
        earlier = history[:-1]
        if earlier and history[-1] > 2.0 * max(1e-6, float(max(earlier))):
            return RecoveryFailure.LINE_SEARCH_EXCURSION
    if convergence.get("ionic_converged") is True:
        return RecoveryFailure.NONE
    return RecoveryFailure.STATUS_UNKNOWN


def decide_recovery(
    policy: RecoveryPolicy,
    *,
    failure: RecoveryFailure,
    attempts_used: int = 0,
    has_contcar: bool = False,
    has_xdatcar_best: bool = False,
    max_force: float | None = None,
    ediffg: float | None = None,
) -> RecoveryDecision:
    """Closed decision table. Never resubmits identical resources."""
    provenance: dict[str, Any] = {"policy": policy.model_dump(mode="json")}
    attempt_id = _next_attempt_id(attempts_used)
    if attempts_used >= policy.max_auto_attempts:
        return RecoveryDecision(
            action="STOP",
            failure=failure.value,
            reason=(
                f"auto-recovery limit reached ({attempts_used} >= "
                f"{policy.max_auto_attempts}); manual intervention required"
            ),
            new_attempt_id=attempt_id,
            provenance=provenance,
        )

    def restart_decision(
        *,
        restart_from: str,
        parameter_changes: list[str],
        incar_changes: dict[str, Any],
        reason: str,
        practical: bool = False,
        practical_note: str = "",
    ) -> RecoveryDecision:
        return RecoveryDecision(
            action="RESUBMIT",
            failure=failure.value,
            restart_from=restart_from,
            parameter_changes=parameter_changes,
            incar_changes=incar_changes,
            new_attempt_id=attempt_id,
            reason=reason,
            practical_convergence=practical,
            practical_convergence_note=practical_note,
            provenance=provenance,
        )

    # -- status/reconciliation first: never re-sbatch on a query failure.
    if failure is RecoveryFailure.STATUS_QUERY_FAILED:
        return RecoveryDecision(
            action="STATUS_ONLY",
            failure=failure.value,
            reason=(
                "status query failed: refresh/reconcile only, NEVER "
                "re-submit this job"
            ),
            new_attempt_id=attempt_id,
            provenance=provenance,
        )
    if failure is RecoveryFailure.AMBIGUOUS_SUBMISSION:
        return RecoveryDecision(
            action="RECONCILE",
            failure=failure.value,
            reason=(
                "submission outcome ambiguous: reconcile via registry + "
                "squeue/sacct before any decision"
            ),
            new_attempt_id=attempt_id,
            provenance=provenance,
        )

    # -- hard failures: STOP (identical resources must not be repeated).
    if failure in {RecoveryFailure.OOM, RecoveryFailure.SCF_NOT_CONVERGED}:
        return RecoveryDecision(
            action="STOP",
            failure=failure.value,
            reason=(
                "not auto-retried by the typed policy: "
                + (
                    "change resources (tasks/memory/LREAL) first"
                    if failure is RecoveryFailure.OOM
                    else "inspect electronic mixing/SIGMA/NELM first"
                )
                + "; resubmitting identical inputs is forbidden"
            ),
            new_attempt_id=attempt_id,
            provenance=provenance,
        )
    if failure is RecoveryFailure.STATUS_UNKNOWN:
        return RecoveryDecision(
            action="STOP",
            failure=failure.value,
            reason="unknown failure mode; manual inspection of OUTCAR required",
            new_attempt_id=attempt_id,
            provenance=provenance,
        )
    if failure is RecoveryFailure.NONE:
        return RecoveryDecision(
            action="STOP",
            failure=failure.value,
            reason="no failure to recover",
            new_attempt_id=attempt_id,
            provenance=provenance,
        )

    # -- restartable classes ------------------------------------------------
    if failure is RecoveryFailure.WALLTIME:
        if policy.require_contcar_for_restart and not has_contcar:
            return RecoveryDecision(
                action="STOP",
                failure=failure.value,
                reason="CONTCAR missing; cannot continue the geometry",
                new_attempt_id=attempt_id,
                provenance=provenance,
            )
        return restart_decision(
            restart_from="CONTCAR",
            parameter_changes=[],
            incar_changes={},
            reason=(
                "walltime: continue from the last complete CONTCAR with a "
                "new attempt and new remote directory"
            ),
        )

    if failure is RecoveryFailure.NSW_EXHAUSTED:
        if policy.require_contcar_for_restart and not has_contcar:
            return RecoveryDecision(
                action="STOP",
                failure=failure.value,
                reason="CONTCAR missing; cannot continue the geometry",
                new_attempt_id=attempt_id,
                provenance=provenance,
            )
        # practical convergence: forces below the RELAXED threshold (recorded
        # explicitly; never claimed as the original EDIFFG).
        changes: dict[str, Any] = {}
        practical = False
        practical_note = ""
        if (
            policy.allow_ediffg_relaxation
            and ediffg
            and max_force is not None
            and max_force > ediffg
            and max_force <= ediffg * policy.ediffg_relax_factor
        ):
            old = ediffg
            new = round(ediffg * policy.ediffg_relax_factor, 6)
            changes["EDIFFG"] = -new
            practical = True
            practical_note = (
                f"EDIFFG relaxed -{old:.6f} -> -{new:.6f} eV/A (reason: "
                f"max force {max_force:.6f} within "
                f"{policy.ediffg_relax_factor:g}x of the original threshold); "
                "this is PRACTICAL convergence, NOT the originally required "
                "accuracy; recorded in provenance, never presented as "
                "satisfying the original EDIFFG"
            )
        return restart_decision(
            restart_from="CONTCAR",
            parameter_changes=[
                *(f"{key} = {value}" for key, value in changes.items()),
            ],
            incar_changes=changes,
            reason=(
                "NSW exhausted: restart from CONTCAR with a new attempt "
                "(never from the initial POSCAR)"
            ),
            practical=practical,
            practical_note=practical_note,
        )

    if failure is RecoveryFailure.FORCE_PLATEAU:
        if policy.require_contcar_for_restart and not has_contcar:
            return RecoveryDecision(
                action="STOP",
                failure=failure.value,
                reason="CONTCAR missing; cannot continue the geometry",
                new_attempt_id=attempt_id,
                provenance=provenance,
            )
        changes: dict[str, Any] = {}
        if policy.allow_potim_reduction:
            changes["POTIM"] = 0.5
        elif policy.allow_ibrion_switch:
            changes["IBRION"] = 1
        return restart_decision(
            restart_from="CONTCAR",
            parameter_changes=[
                f"{key} = {value}" for key, value in changes.items()
            ],
            incar_changes=changes,
            reason=(
                "force plateau detected; "
                + (
                    "reduce POTIM to 0.5"
                    if changes.get("POTIM") is not None
                    else (
                        "switch IBRION to 1"
                        if changes.get("IBRION") is not None
                        else "no typed parameter change available; STOP"
                    )
                )
            ),
        )

    if failure is RecoveryFailure.LINE_SEARCH_EXCURSION:
        if has_xdatcar_best:
            return restart_decision(
                restart_from="XDATCAR_BEST",
                parameter_changes=[],
                incar_changes={},
                reason=(
                    "line-search excursion: NOT continuing from the worsened "
                    "latest geometry; restart from the historical lowest-force "
                    "snapshot"
                ),
            )
        return RecoveryDecision(
            action="STOP",
            failure=failure.value,
            reason=(
                "line-search excursion and no historical best structure; "
                "do NOT blindly continue from the obviously worsened latest "
                "geometry"
            ),
            new_attempt_id=attempt_id,
            provenance=provenance,
        )

    return RecoveryDecision(
        action="STOP",
        failure=failure.value,
        reason=f"unhandled failure class {failure.value}",
        new_attempt_id=attempt_id,
        provenance=provenance,
    )


def apply_incar_changes(incar_text: str, changes: dict[str, Any]) -> str:
    """Rewrite an INCAR text with typed changes only (old/new recorded by
    the caller into recovery provenance)."""
    if not changes:
        return incar_text
    lines = incar_text.splitlines()
    rendered = render_incar(changes).splitlines()
    updates = {
        rendered_line.split(" = ")[0]: rendered_line for rendered_line in rendered
    }
    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            out.append(line)
            continue
        key = stripped.split(" = ")[0].split()[0]
        if key in updates:
            out.append(updates[key])
            seen.add(key)
        else:
            out.append(line)
    for key, rendered in updates.items():
        if key not in seen:
            out.append(rendered)
    return "\n".join(out) + "\n"


def materialize_recovery_stage_dir(
    *,
    previous_stage_dir: str | Path,
    restart_structure: str | Path,
    incar_changes: dict[str, Any],
    attempt_id: str,
    workflow_dir: str | Path,
    reason: str,
    practical_convergence: bool = False,
    practical_convergence_note: str = "",
) -> Path:
    """Build the restart stage dir: CONTCAR snapshot -> POSCAR, typed INCAR
    changes, KPOINTS/meta copies and a recovery_provenance.json."""
    previous = Path(previous_stage_dir).expanduser().resolve()
    target = (
        Path(workflow_dir).expanduser().resolve()
        / f"stage_relax_attempt_{attempt_id}"
    )
    target.mkdir(parents=True, exist_ok=True)
    source_structure = Path(restart_structure).expanduser().resolve()
    if not source_structure.is_file():
        raise FileNotFoundError(
            f"recovery restart artifact missing: {source_structure}"
        )
    shutil.copy2(source_structure, target / "POSCAR")
    for name in ("KPOINTS", "POTCAR.meta", "POTCAR.policy"):
        source_file = previous / name
        if source_file.is_file():
            shutil.copy2(source_file, target / name)
    incar_path = previous / "INCAR"
    if incar_path.is_file():
        (target / "INCAR").write_text(
            apply_incar_changes(
                incar_path.read_text(encoding="utf-8", errors="replace"),
                incar_changes,
            ),
            encoding="utf-8",
        )
    else:
        (target / "INCAR").write_text(
            render_incar(incar_changes) if incar_changes else "",
            encoding="utf-8",
        )
    old_values = dict(incar_changes)
    (target / "recovery_provenance.json").write_text(
        json.dumps(
            {
                "attempt_id": attempt_id,
                "restart_from": source_structure.name,
                "incar_changes": old_values,
                "reason": reason,
                "practical_convergence": practical_convergence,
                "practical_convergence_note": practical_convergence_note,
                "created_at": _now(),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return target
