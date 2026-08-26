"""Auditable resource calibration records for production VASP workflows.

A ``CalibrationRecord`` is a measured, attributed statement about one real
VASP run (memory, timing, electron count, grid, parallelisation): it exists
so that a production submission never runs on guessed resources. The record
is content-addressed through ``derive_calibration_id`` and its applicability
is verified deterministically by :func:`calibration_applicable` (formula /
atom count / box / ENCUT must match the molecule being planned).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class CalibrationRecord(BaseModel):
    """One measured resource calibration from a real representative VASP run."""

    calibration_id: str = ""
    atom_count: int = Field(gt=0)
    formula: str = ""
    elements: list[str] = Field(default_factory=list)
    nbands: int | None = None
    box_ang: float | None = None
    grid: tuple[int, int, int] | None = None
    encut_ev: float | None = None
    prec: str = ""
    lreal: bool | None = None
    addgrid: bool | None = None
    tasks: int = Field(default=8, gt=0)
    ncore: int | None = None
    electronic_steps: int | None = None
    elapsed_seconds: float = Field(default=0.0, ge=0)
    max_rss_bytes: int = Field(default=0, ge=0)
    vasp_version: str = ""
    source_job_id: str = ""
    applicable_to: str = ""
    recorded_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
    )
    notes: list[str] = Field(default_factory=list)


def _substantive_payload(record: CalibrationRecord) -> dict[str, Any]:
    return {
        "atom_count": record.atom_count,
        "formula": record.formula,
        "elements": record.elements,
        "nbands": record.nbands,
        "box_ang": record.box_ang,
        "grid": list(record.grid) if record.grid else None,
        "encut_ev": record.encut_ev,
        "prec": record.prec,
        "lreal": record.lreal,
        "addgrid": record.addgrid,
        "tasks": record.tasks,
        "ncore": record.ncore,
        "electronic_steps": record.electronic_steps,
        "elapsed_seconds": record.elapsed_seconds,
        "max_rss_bytes": record.max_rss_bytes,
        "vasp_version": record.vasp_version,
        "applicable_to": record.applicable_to,
    }


def derive_calibration_id(record: CalibrationRecord) -> str:
    """Content-addressed id: same measured facts -> same id."""
    payload = json.dumps(_substantive_payload(record), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def finalize_calibration(record: CalibrationRecord) -> CalibrationRecord:
    """Fill the calibration_id from content if it is empty."""
    if record.calibration_id:
        return record
    return record.model_copy(update={"calibration_id": derive_calibration_id(record)})


def calibration_applicable(
    record: CalibrationRecord,
    *,
    formula: str | None = None,
    atom_count: int | None = None,
    box_ang: float | None = None,
    encut_ev: float | None = None,
) -> tuple[bool, list[str]]:
    """Deterministic scope check: is this calibration evidence for THAT run?

    The record must match the molecule's formula/atom count/box/ENCUT;
    mismatches are returned as reasons why the calibration must NOT be used.
    A missing field (None) is not itself a mismatch: the check only refuses
    when a claimed value contradicts the record.
    """
    problems: list[str] = []
    if formula and record.formula and formula != record.formula:
        problems.append(
            f"calibration formula {record.formula!r} != planned {formula!r}"
        )
    if (
        atom_count is not None
        and record.atom_count
        and atom_count != record.atom_count
    ):
        problems.append(
            f"calibration atom_count {record.atom_count} != planned {atom_count}"
        )
    if (
        box_ang is not None
        and record.box_ang
        and abs(box_ang - record.box_ang) > 1e-6
    ):
        problems.append(
            f"calibration box {record.box_ang:g} A != planned "
            f"{box_ang:g} A"
        )
    if (
        encut_ev is not None
        and record.encut_ev
        and abs(encut_ev - record.encut_ev) > 1e-6
    ):
        problems.append(
            f"calibration ENCUT {record.encut_ev:g} != planned "
            f"{encut_ev:g}"
        )
    return not problems, problems

