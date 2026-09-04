"""Conformer screening funnel: cheap static screens rank candidates by E0.

Only the lowest-E0 candidate of a chemical formula ever enters the expensive
production stages (relax + corrected_static + orbitals/ESP). Every candidate
keeps its screen E0, relative energy, job_id and elimination reason; total
energies are NEVER compared across different formulas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from photomatagent.scientific.applications.vasp.molecular.models import (
    MoleculeSpec,
    StageName,
)
from photomatagent.scientific.applications.vasp.molecular.workflow import (
    build_molecule_workflow,
)


class ConformerScreenRecord(BaseModel):
    """One candidate's cheap screen outcome."""

    candidate_index: int
    structure_path: str
    state: str = ""  # VALIDATED | FAILED | SKIPPED
    e0_ev: float | None = None
    relative_e0_ev: float | None = None
    job_id: str = ""
    request_id: str = ""
    screen_workflow_dir: str = ""
    elimination_reason: str = ""


class ConformerScreenReport(BaseModel):
    """Full screening report for one unique calculation."""

    system_id: str
    task_id: str
    formula: str = ""
    screen_complete: bool = False
    selected_structure_path: str = ""
    selected_candidate_index: int | None = None
    records: list[ConformerScreenRecord] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "formula": self.formula,
            "candidates_screened": len(self.records),
            "selected_structure_path": self.selected_structure_path,
            "selected_candidate_index": self.selected_candidate_index,
            "notes": self.notes,
        }


def persist_screen_report(
    report: ConformerScreenReport, study_dir: str | Path
) -> Path:
    """Persist the per-task screening record under study_dir/screening/."""
    directory = Path(study_dir).expanduser().resolve() / "screening"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{report.task_id}.json"
    target.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return target


def load_screen_reports(study_dir: str | Path) -> dict[str, ConformerScreenReport]:
    """Load all persisted screen reports (resume contract)."""
    directory = Path(study_dir).expanduser().resolve() / "screening"
    reports: dict[str, ConformerScreenReport] = {}
    if not directory.is_dir():
        return reports
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            report = ConformerScreenReport.model_validate(payload)
            reports[report.task_id] = report
        except Exception:
            continue
    return reports


class ConformerScreener:
    """Runs the cheap static screen for one task's candidates."""

    def __init__(
        self,
        *,
        runtime: Any,
        screens_root: str | Path,
        method: Any,
    ) -> None:
        self.runtime = runtime
        self.screens_root = Path(screens_root).expanduser().resolve()
        self.screens_root.mkdir(parents=True, exist_ok=True)
        self.method = method

    async def screen(
        self,
        *,
        task: Any,
        candidates: list[str],
        box_ang: float,
    ) -> ConformerScreenReport:
        """Screen every candidate on the SAME formula/charge/spin/box."""
        report = ConformerScreenReport(
            system_id=task.system_id,
            task_id=task.task_id,
            formula=task.formula,
        )
        from photomatagent.scientific.applications.vasp.molecular.tools import (
            MolecularVaspTools,
        )

        scores: list[tuple[int, float, ConformerScreenRecord]] = []
        for index, candidate in enumerate(candidates):
            molecule = MoleculeSpec(
                name=f"{task.display_name}_screen_{index}",
                structure_path=Path(candidate),
                total_charge=task.total_charge,
                spin_multiplicity=task.spin_multiplicity,
                box_ang=box_ang,
                functional=self.method.functional,
                potcar_set=self._potcar_set(),
                calculation_purpose="conformer_screen",
            )
            workflow = build_molecule_workflow(
                molecule,
                psp_dir=self.runtime.psp_dir,
                encut_ev=self._encut(),
                screen_only=True,
                resource_profile=self.method.profile(),
            )
            screen_dir = self.screens_root / f"{task.task_id}" / f"candidate_{index}"
            screen_dir.mkdir(parents=True, exist_ok=True)
            facade = MolecularVaspTools(
                session=self.runtime.session,
                backend=self.runtime.backend,
                psp_dir=self.runtime.psp_dir,
                workflow_dir=screen_dir,
                log_dir=self.runtime.log_dir,
                module_name=self.runtime.module_name,
                env_script=self.runtime.env_script,
                remote_psp_dir=self.runtime.remote_psp_dir,
                configured=self.runtime.configured,
            )
            record = ConformerScreenRecord(
                candidate_index=index,
                structure_path=str(Path(candidate).resolve()),
                screen_workflow_dir=str(screen_dir),
            )
            try:
                runner = await facade.run_workflow(
                    workflow,
                    wait=True,
                    collect=True,
                    stop_on_failure=True,
                    wait_timeout_seconds=120.0,
                )
                task_state_payload = _stage_results(screen_dir)
                e0 = task_state_payload.get("e0_ev")
                runner_blocked = bool(runner.get("blocked"))
                summary = runner.get("summary") or {}
                if (
                    runner_blocked
                    or not runner.get("ok")
                    or e0 is None
                    or not summary.get("evidence_count")
                ):
                    record.state = "FAILED"
                    record.elimination_reason = (
                        "screen run not validated; "
                        + str(runner.get("errors") or ["no E0"])[0]
                    )
                else:
                    record.state = "VALIDATED"
                    record.e0_ev = e0
                    record.job_id = _screen_job_id(screen_dir) or ""
                    record.request_id = _screen_request_id(screen_dir) or ""
            except Exception as exc:
                record.state = "FAILED"
                record.elimination_reason = f"{type(exc).__name__}: {exc}"
            report.records.append(record)
            if record.e0_ev is not None:
                scores.append((index, record.e0_ev, record))

        if not scores:
            report.screen_complete = False
            report.notes.append(
                "no candidate produced a validated screen E0; production "
                "falls back to the first candidate with the screen "
                "incompleteness recorded"
            )
        else:
            report.screen_complete = True
            scores.sort(key=lambda item: (item[1], item[0]))
            best_index, best_e0, best_record = scores[0]
            report.selected_structure_path = best_record.structure_path
            report.selected_candidate_index = best_index
            for _, e0, record in scores:
                record.relative_e0_ev = round(float(e0 - best_e0), 8)
                if record is not best_record:
                    record.elimination_reason = (
                        f"higher E0 by {record.relative_e0_ev:+.6f} eV; only "
                        "the lowest-energy candidate enters production"
                    )
            report.notes.append(
                "scores ranked within the same chemical formula only; "
                "cross-formula energy comparisons are never made"
            )
        return report

    def _potcar_set(self) -> str:
        return self.method.potcar_set

    def _encut(self) -> float | None:
        return self.method.encut_ev


def _stage_results(workflow_dir: Path) -> dict[str, Any]:
    """E0 of the screen's static single point from persisted results.json."""
    payload_path = (
        Path(workflow_dir) / "results" / StageName.STATIC.value / "results.json"
    )
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    energy = payload.get("energy") or {}
    return {
        "e0_ev": energy.get("e_0_ev"),
        "validated": payload.get("validated") is True,
    }


def _screen_job_id(workflow_dir: Path) -> str | None:
    from photomatagent.scientific.applications.vasp.molecular.workflow import (
        load_task_state,
    )

    state = load_task_state(workflow_dir)
    if state is None or not state.stages:
        return None
    return state.stages[0].job_id or None


def _screen_request_id(workflow_dir: Path) -> str | None:
    from photomatagent.scientific.applications.vasp.molecular.workflow import (
        load_task_state,
    )

    state = load_task_state(workflow_dir)
    if state is None or not state.stages:
        return None
    return state.stages[0].request_id or None
