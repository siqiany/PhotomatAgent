"""Study executor: drives the existing vasp_molecule.* machinery.

The executor never re-implements submission, POTCAR, Slurm or monitoring.
Each unique calculation in the matrix is run through
:class:`MolecularVaspTools` (prepare/preflight/run_workflow/collect), whose
submit-once + resume semantics are reused verbatim. Study-level concerns
handled here: per-task state persistence (study_state.json), authorization
gates, resource budget enforcement, conformer fallback, binding gating
(only after every required energy is VALIDATED) and isolation of failures.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from photomatagent.scientific.applications.vasp.molecular.models import (
    MoleculeSpec,
    ResourceProfile,
    StageName,
)
from photomatagent.scientific.applications.vasp.molecular.workflow import (
    build_molecule_workflow,
)
from photomatagent.scientific.applications.vasp.study.models import (
    PropertyRequest,
    StudyTaskState,
    VaspStudySpec,
)
from photomatagent.scientific.applications.vasp.study.planner import (
    budget_status,
)
from photomatagent.scientific.capabilities.chemistry.storage import read_xyz


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stage_energy_dir(workflow_dir: Path) -> Path | None:
    """The production-energy stage results dir (corrected_static then relax)."""
    for stage in (StageName.CORRECTED_STATIC, StageName.RELAX):
        candidate = workflow_dir / "results" / stage.value
        if (candidate / "results.json").is_file():
            return candidate
    return None


def _structure_extent(path: Path) -> float:
    """Max per-axis extent of a structure (Angstrom)."""
    try:
        _symbols, coords, _ = read_xyz(path)
    except ValueError:
        return 8.0
    if not len(coords):
        return 0.0
    return float(np.max(coords.max(axis=0) - coords.min(axis=0)))


class StudyExecutor:
    """One resumable study execution session bound to a molecular runtime."""

    def __init__(
        self,
        spec: VaspStudySpec,
        runtime: Any,
    ) -> None:
        self.spec = spec
        self.runtime = runtime
        self.workflows_root = Path(spec.study_dir) / "workflows"
        self.workflows_root.mkdir(parents=True, exist_ok=True)

    # -- persisted study state ----------------------------------------------

    @property
    def state_path(self) -> Path:
        return Path(self.spec.study_dir) / "study_state.json"

    def load_state(self) -> dict[str, Any]:
        if self.state_path.is_file():
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            return payload
        return {
            "study_id": self.spec.study_id,
            "tasks": {},
            "binding_groups": {},
            "updated_at": _now(),
        }

    def save_state(self, state: dict[str, Any]) -> Path:
        state["updated_at"] = _now()
        self.state_path.write_text(
            json.dumps(state, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return self.state_path

    def _sync_matrix_from_state(
        self, state: dict[str, Any]
    ) -> None:
        """Copy persisted task/groups states back into the matrix model."""
        tasks = state.get("tasks", {})
        for task in self.spec.calculation_matrix.tasks:
            if task.task_id in tasks:
                stored = tasks[task.task_id]
                task.state = stored.get("state", task.state)
                task.request_id = stored.get("request_id", task.request_id)
                task.conformer_index = stored.get(
                    "conformer_index", task.conformer_index
                )
                task.error = stored.get("error", task.error)
                task.workflow_dir = stored.get(
                    "workflow_dir", task.workflow_dir
                )
                task.results_dir = stored.get("results_dir", task.results_dir)
        for group in self.spec.calculation_matrix.binding_groups:
            stored = state.get("binding_groups", {}).get(group.complex_task_id)
            if stored:
                group.state = stored.get("state", group.state)
                group.delta_e_ev = stored.get("delta_e_ev")
                group.delta_delta_e_ev = stored.get("delta_delta_e_ev")
                group.error = stored.get("error", group.error)

    def _state_entries(self) -> dict[str, Any]:
        return {
            "tasks": {
                task.task_id: {
                    "state": task.state,
                    "request_id": task.request_id,
                    "conformer_index": task.conformer_index,
                    "error": task.error,
                    "workflow_dir": task.workflow_dir,
                    "results_dir": task.results_dir,
                }
                for task in self.spec.calculation_matrix.tasks
            },
            "binding_groups": {
                group.complex_task_id: {
                    "state": group.state,
                    "delta_e_ev": group.delta_e_ev,
                    "delta_delta_e_ev": group.delta_delta_e_ev,
                    "error": group.error,
                }
                for group in self.spec.calculation_matrix.binding_groups
            },
        }

    # -- per-task helpers -----------------------------------------------------

    def _authorized(self) -> bool:
        request = self.spec.request
        backend_policy = getattr(self.runtime.backend, "policy", None)
        allow = bool(
            getattr(backend_policy, "allow_hpc_submit", False)
            if backend_policy is not None
            else False
        )
        return bool(
            request.execution_policy.user_requested_computation
            and allow
        )

    def _facade(self, workflow_dir: Path) -> Any:
        from photomatagent.scientific.applications.vasp.molecular.tools import (
            MolecularVaspTools,
        )

        return MolecularVaspTools(
            session=self.runtime.session,
            backend=self.runtime.backend,
            psp_dir=self.runtime.psp_dir,
            workflow_dir=workflow_dir,
            log_dir=self.runtime.log_dir,
            module_name=self.runtime.module_name,
            env_script=self.runtime.env_script,
            remote_psp_dir=self.runtime.remote_psp_dir,
            configured=self.runtime.configured,
        )

    def _group_box_ang(self, task: Any) -> float:
        """One box for the whole connected binding component.

        The binding-consistency rule requires E(complex), E(fragment1), ...
        computed in the SAME box. A fragment shared across multiple binding
        groups (Li+, TVM) therefore forces every task in its connected
        component to use the component's maximum needed box.
        """
        request = self.spec.request
        matrix = self.spec.calculation_matrix
        task_map = matrix.task_map()
        parent: dict[str, str] = {}

        def find(key: str) -> str:
            parent.setdefault(key, key)
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        def union(left: str, right: str) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for group in matrix.binding_groups:
            members = [group.complex_task_id, *group.fragment_task_ids]
            for member in members[1:]:
                union(members[0], member)
        root = task.task_id in parent and find(task.task_id) or task.task_id
        group_task_ids = [
            key for key in parent if find(key) == root
        ]
        group_tasks = [
            task_map[key] for key in group_task_ids if key in task_map
        ]
        if task.task_id not in group_task_ids:
            group_tasks.append(task)  # standalone systems size their own box
        extents = [
            _structure_extent(Path(other.structure_path)) + 2.0 * 10.0
            for other in group_tasks
            if other.structure_path
        ]
        needed = max(
            [request.method.box_ang, *extents],
        )
        return needed

    def _build_workflow(self, task: Any) -> Any:
        request = self.spec.request
        method = request.method
        profile = method.profile()
        calibration = method.calibration_record()
        tasks = (calibration.tasks if calibration is not None else None) or 8
        walltime = 20
        if calibration is not None and calibration.elapsed_seconds > 0:
            walltime = max(
                20, int(calibration.elapsed_seconds / 60.0 * 1.5) + 5
            )
        assert task.structure_path is not None
        molecule = MoleculeSpec(
            name=task.display_name,
            structure_path=Path(task.structure_path),
            total_charge=task.total_charge,
            spin_multiplicity=task.spin_multiplicity,
            box_ang=self._group_box_ang(task),
            functional=method.functional,
            potcar_set=method.potcar_set,
            calculation_purpose="vasp_study",
        )
        assists = set(task.assists)
        return build_molecule_workflow(
            molecule,
            psp_dir=self.runtime.psp_dir,
            encut_ev=method.encut_ev,
            # ISPIN is derived from the typed MoleculeSpec (multiplicity > 1
            # or odd electrons -> ISPIN=2). Passing ``spin`` here means
            # "explicit ISPIN override" -- it is NEVER the multiplicity.
            spin=method.spin,
            include_orbital_homo=PropertyRequest.HOMO_LUMO in assists,
            include_orbital_lumo=PropertyRequest.HOMO_LUMO in assists,
            include_esp=PropertyRequest.ESP in assists,
            resource_profile=profile,
            tasks_per_node=tasks,
            walltime_minutes=walltime,
            calibration=calibration,
        )

    # -- execution ------------------------------------------------------------

    async def execute(self) -> dict[str, Any]:
        """Run one pass over the matrix (idempotent, resumable)."""
        state = self.load_state()
        self._sync_matrix_from_state(state)
        report: dict[str, Any] = {
            "study_id": self.spec.study_id,
            "submitted": [],
            "resumed": [],
            "skipped": [],
            "failed": [],
            "budget": budget_status(self.spec),
            "authorized": self._authorized(),
        }
        if not self._authorized():
            for task in self.spec.calculation_matrix.tasks:
                if task.state == StudyTaskState.PLANNED.value:
                    task.state = StudyTaskState.BLOCKED_NO_AUTHORIZATION.value
                    task.error = (
                        "study-level authorization missing: "
                        "user_requested_computation=True and "
                        "PHOTOMATAGENT_ALLOW_HPC_SUBMIT are both required"
                    )
            self._persist(report)
            return report

        spent_core_hours = 0.0
        budget = self.spec.request.resource_budget
        authorized = True
        for task in self.spec.calculation_matrix.tasks:
            if not task.structure_path:
                task.state = StudyTaskState.SKIPPED_PROXY.value
                report["skipped"].append(task.task_id)
                continue
            if task.state == StudyTaskState.VALIDATED.value:
                report["resumed"].append(task.task_id)
                continue
            if task.state in {
                StudyTaskState.SKIPPED_PROXY.value,
                StudyTaskState.SKIPPED_BUDGET.value,
                StudyTaskState.BLOCKED_NO_AUTHORIZATION.value,
                StudyTaskState.FAILED.value,
            }:
                report["skipped"].append(task.task_id)
                continue
            if not authorized:
                task.state = StudyTaskState.SKIPPED_BUDGET.value
                task.error = "resource budget exhausted; no new jobs started"
                report["skipped"].append(task.task_id)
                continue
            if spent_core_hours + task.estimated_core_hours > budget.max_core_hours + 1e-9:
                task.state = StudyTaskState.SKIPPED_BUDGET.value
                task.error = (
                    f"budget {budget.max_core_hours:g} core-h exceeded "
                    f"(spent {spent_core_hours:.1f} + est "
                    f"{task.estimated_core_hours:.1f})"
                )
                authorized = False
                report["skipped"].append(task.task_id)
                continue
            spent_core_hours += task.estimated_core_hours
            outcome = await self._run_task(task)
            report["submitted"].append(task.task_id)
            if outcome["state"] in {
                StudyTaskState.VALIDATED.value,
            }:
                report["resumed"].append(task.task_id)
            elif outcome["state"] in {
                StudyTaskState.COLLECTED.value,
                StudyTaskState.COMPLETED.value,
            }:
                report["failed"].append(
                    {"task": task.task_id, "reason": outcome["error"]}
                )
            else:
                report["failed"].append(
                    {"task": task.task_id, "reason": outcome["error"]}
                )
        self._compute_bindings()
        self._persist(report)
        return report

    async def _run_task(self, task: Any) -> dict[str, Any]:
        """Run one unique calculation.

        With the screening funnel enabled (B3), every candidate gets a cheap
        static E0 screen first and ONLY the selected lowest-energy candidate
        enters the expensive production stages; there is no expensive
        fallback loop over candidates after screening. Without screening the
        legacy conformer fallback remains.
        """
        policy = self.spec.request.execution_policy
        candidates = list(task.structure_candidates)
        if policy.screen_conformers and len(candidates) > 1:
            from photomatagent.scientific.applications.vasp.study.screening import (
                load_screen_reports,
            )

            # Resume safety: never re-screen a task that already has a
            # completed, persisted screen report (no duplicate screen jobs).
            existing = load_screen_reports(self.spec.study_dir).get(task.task_id)
            if (
                existing is not None
                and existing.screen_complete
                and existing.selected_structure_path
            ):
                task.structure_path = Path(existing.selected_structure_path)
                task.conformer_index = existing.selected_candidate_index or 0
                return await self._run_task_with_conformer(task)
            screened = await self._screen_task(task, candidates)
            if screened["selected"] is not None:
                task.structure_path = Path(screened["selected"])
                task.conformer_index = screened["index"]
            return await self._run_task_with_conformer(task)
        max_attempts = max(1, len(task.structure_candidates))
        for attempt in range(max_attempts):
            if attempt > 0:
                task.state = StudyTaskState.CONFORMER_RETRY.value
                task.conformer_index = attempt
                task.error = ""
            outcome = await self._run_task_with_conformer(task)
            if outcome["state"] == StudyTaskState.VALIDATED.value:
                return outcome
            if task.conformer_index + 1 < max_attempts:
                task.conformer_index += 1
                continue
            return outcome
        return {"state": task.state, "error": task.error}

    async def _screen_task(
        self, task: Any, candidates: list[str]
    ) -> dict[str, Any]:
        """Run the deterministic conformer screen funnel (cheap statics)."""
        from photomatagent.scientific.applications.vasp.study.screening import (
            ConformerScreener,
            persist_screen_report,
        )

        policy = self.spec.request.execution_policy
        candidates = candidates[: policy.max_screen_candidates]
        box = self._group_box_ang(task)
        screener = ConformerScreener(
            runtime=self.runtime,
            screens_root=self.spec.study_dir / "screening",
            method=self.spec.request.method,
        )
        report = await screener.screen(
            task=task, candidates=candidates, box_ang=box
        )
        persist_screen_report(report, self.spec.study_dir)
        return {
            "selected": (
                report.selected_structure_path if report.screen_complete else None
            ),
            "index": report.selected_candidate_index,
            "report": report,
        }

    async def _run_task_with_conformer(self, task: Any) -> dict[str, Any]:
        """Prepare + submit + collect one conformer through vasp_molecule.*."""
        assert task.structure_path is not None
        if task.structure_candidates:
            candidates = task.structure_candidates
            index = min(task.conformer_index, len(candidates) - 1)
            task.structure_path = Path(candidates[index])
        workflow_dir = self.workflows_root / task.system_id
        workflow_dir.mkdir(parents=True, exist_ok=True)
        workflow = self._build_workflow(task)
        facade = self._facade(workflow_dir)
        prepared = await facade.prepare(workflow, output_dir=workflow_dir)
        task.workflow_dir = str(workflow_dir)
        if not prepared["ok"]:
            task.state = StudyTaskState.PREFLIGHT_FAILED.value
            task.error = "; ".join(prepared["errors"][:3])
            return {"state": task.state, "error": task.error}
        # Zero-electron references (bare Li+): VASP cannot run NELECT=0. The
        # study uses a declared reference model (E = 0 eV convention, ΔΔE
        # recommended) instead of submitting a job that cannot converge.
        nelect = (prepared.get("summary") or {}).get("nelect")
        if nelect is not None and float(nelect) <= 0:
            self._write_zero_electron_reference(workflow_dir, task)
            task.state = StudyTaskState.VALIDATED.value
            task.results_dir = str(
                workflow_dir / "results" / "reference_zero_electron"
            )
            task.error = (
                "zero-electron reference model (E=0 eV convention; "
                "VASP cannot run NELECT=0)"
            )
            return {"state": task.state, "error": task.error}
        report = await facade.run_workflow(
            workflow,
            wait=self.spec.request.execution_policy.wait,
            collect=True,
            stop_on_failure=self.spec.request.execution_policy.stop_on_failure,
            wait_timeout_seconds=(
                self.spec.request.execution_policy.wait_timeout_seconds
            ),
        )
        payload_errors = report.get("errors", [])
        if report.get("error") or payload_errors:
            task.state = StudyTaskState.FAILED.value
            task.error = (
                report.get("error")
                or "; ".join(str(item) for item in payload_errors[:3])
            )
            return {"state": task.state, "error": task.error}
        # The authoritative stage ledger is the persisted task_state.json
        # (the tool payload only carries stage-name strings).
        molecular_stages = self._molecular_stages(workflow_dir)
        if not molecular_stages:
            task.state = StudyTaskState.FAILED.value
            task.error = "workflow returned no stage report"
            return {"state": task.state, "error": task.error}
        all_validated = all(
            item.state == "VALIDATED" and item.validated
            for item in molecular_stages
        )
        if all_validated:
            task.state = StudyTaskState.VALIDATED.value
        else:
            failed = next(
                (
                    item
                    for item in molecular_stages
                    if item.state == "FAILED"
                ),
                None,
            )
            task.state = (
                StudyTaskState.COLLECTED.value
                if any(
                    item.state == "COLLECTED" for item in molecular_stages
                )
                else StudyTaskState.FAILED.value
            )
            task.error = (
                failed.error
                if failed
                else "; ".join(
                    item.error
                    for item in molecular_stages
                    if item.error
                )
            )
        energy_dir = _stage_energy_dir(workflow_dir)
        task.results_dir = str(energy_dir) if energy_dir else ""
        return {"state": task.state, "error": task.error}

    def _write_zero_electron_reference(
        self,
        workflow_dir: Path,
        task: Any,
    ) -> Path:
        """Persist the declared zero-electron reference results.json."""
        box = self._group_box_ang(task)
        result_dir = workflow_dir / "results" / "reference_zero_electron"
        result_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "validated": True,
            "errors": [],
            "warnings": [],
            "explicit_reference_assumption": True,
            "reference_kind": "zero_electron_bare_ion",
            "reference_model": {
                "kind": "zero_electron_bare_ion",
                "convention": "E = 0 eV by definition (VASP cannot run "
                "NELECT = 0)",
                "note": "use relative binding energies (ΔΔE) or an "
                "alternative reference to reduce bare-ion error",
            },
            "not_a_vasp_result": True,
            "identity": {
                "formula": task.formula or "Li",
                "charge": task.total_charge,
                "spin_multiplicity": task.spin_multiplicity,
            },
            "method": {
                "functional": "PE",  # matches analyze_result_dir's GGA tag
                "encut_ev": self.spec.request.method.encut_ev or 400.0,
                "box_ang": box,
                "gamma_only": True,
            },
            "energy": {
                "e_0_ev": 0.0,
                "e_fr_ev": 0.0,
                "source": "declared zero-electron reference model",
                "note": "not a VASP result; explicit model assumption",
            },
            "limitations": [
                "bare-ion reference error; prefer ΔΔE comparisons",
                "no VASP run was performed for this reference",
            ],
        }
        (result_dir / "results.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return result_dir

    # -- collect / status ------------------------------------------------------

    async def collect(self) -> dict[str, Any]:
        """Collect+validate COMPLETED tasks, re-validate COLLECTED ones."""
        state = self.load_state()
        self._sync_matrix_from_state(state)
        report: dict[str, Any] = {"study_id": self.spec.study_id, "collected": []}
        for task in self.spec.calculation_matrix.tasks:
            if task.state not in {
                StudyTaskState.COMPLETED.value,
                StudyTaskState.COLLECTED.value,
            }:
                continue
            workflow_dir = Path(task.workflow_dir or "")
            if not (workflow_dir / "task_state.json").is_file():
                report["collected"].append(
                    {"task": task.task_id, "state": task.state, "error": "no workflow"}
                )
                continue
            facade = self._facade(workflow_dir)
            # Re-run the molecular runner with collect=True: COMPLETED stages
            # are downloaded/validated, COLLECTED stages re-validated, and
            # nothing is resubmitted (molecular resume semantics).
            manifest = workflow_dir / "workflow.json"
            if not manifest.is_file():
                report["collected"].append(
                    {"task": task.task_id, "state": task.state, "error": "no workflow.json"}
                )
                continue
            from photomatagent.scientific.applications.vasp.molecular.models import (
                WorkflowSpec,
            )

            workflow = WorkflowSpec.model_validate_json(
                manifest.read_text(encoding="utf-8")
            )
            runner_report = await facade.run_workflow(
                workflow,
                wait=False,
                collect=True,
                only=None,
                wait_timeout_seconds=60.0,
            )
            task.workflow_dir = str(workflow_dir)
            if runner_report.get("error") or runner_report.get("errors"):
                task.error = (
                    runner_report.get("error")
                    or "; ".join(
                        str(item)
                        for item in runner_report.get("errors", [])[:3]
                    )
                )
                report["collected"].append(
                    {"task": task.task_id, "state": task.state, "error": task.error}
                )
                continue
            if all(
                item.state == "VALIDATED"
                for item in self._molecular_stages(workflow_dir)
            ):
                task.state = StudyTaskState.VALIDATED.value
            else:
                task.state = StudyTaskState.COLLECTED.value
            energy_dir = _stage_energy_dir(workflow_dir)
            task.results_dir = str(energy_dir) if energy_dir else ""
            report["collected"].append(
                {"task": task.task_id, "state": task.state, "error": task.error}
            )
        self._compute_bindings()
        self._persist(report)
        return report

    @staticmethod
    def _molecular_stages(workflow_dir: Path) -> list[Any]:
        from photomatagent.scientific.applications.vasp.molecular.workflow import (
            load_task_state,
        )

        task_state = load_task_state(workflow_dir)
        return list(task_state.stages) if task_state is not None else []

    async def status(self) -> dict[str, Any]:
        """Read persisted study state; refresh registry records when known."""
        state = self.load_state()
        self._sync_matrix_from_state(state)
        rows: list[dict[str, Any]] = []
        for task in self.spec.calculation_matrix.tasks:
            rows.append(
                {
                    "task": task.task_id,
                    "system": task.display_name,
                    "charge": task.total_charge,
                    "reliability": task.reliability,
                    "state": task.state,
                    "conformer": task.conformer_index,
                    "error": task.error,
                    "workflow_dir": task.workflow_dir,
                }
            )
        return {
            "study_id": self.spec.study_id,
            "tasks": rows,
            "binding_groups": [
                {
                    "complex": group.complex_task_id,
                    "state": group.state,
                    "delta_e_ev": group.delta_e_ev,
                    "error": group.error,
                }
                for group in self.spec.calculation_matrix.binding_groups
            ],
            "budget": budget_status(self.spec),
        }

    # -- binding ---------------------------------------------------------------

    def _compute_bindings(self) -> None:
        """E_binding only after every required energy is VALIDATED."""
        from photomatagent.scientific.applications.vasp.molecular.binding import (
            BindingEnergyInput,
            BindingReference,
            compute_binding_energy,
        )

        matrix = self.spec.calculation_matrix
        task_map = matrix.task_map()
        for group in matrix.binding_groups:
            if group.state == StudyTaskState.VALIDATED.value:
                continue
            complex_task = task_map.get(group.complex_task_id)
            if complex_task is None:
                group.state = StudyTaskState.FAILED.value
                group.error = (
                    f"complex task {group.complex_task_id} missing from matrix"
                )
                continue
            fragment_tasks = [
                task_map[fragment_id]
                for fragment_id in group.fragment_task_ids
                if fragment_id in task_map
            ]
            required = [complex_task, *fragment_tasks]
            if any(
                task is None
                or task.state != StudyTaskState.VALIDATED.value
                or not task.results_dir
                for task in required
            ):
                pending = [
                    task.task_id
                    for task in required
                    if task is None
                    or task.state != StudyTaskState.VALIDATED.value
                ]
                group.state = StudyTaskState.PLANNED.value
                group.error = f"waiting for VALIDATED energies: {pending}"
                continue
            try:
                result = compute_binding_energy(
                    BindingEnergyInput(
                        complex_name=complex_task.display_name,
                        complex_dir=complex_task.results_dir,
                        references=[
                            BindingReference(
                                name=fragment.display_name,
                                results_dir=fragment.results_dir,
                                charge=fragment.total_charge,
                                role=(
                                    "ion"
                                    if fragment.role == "ion"
                                    else "fragment"
                                ),
                            )
                            for fragment in fragment_tasks
                        ],
                        charge=group.total_charge,
                    )
                )
            except Exception as exc:
                group.state = StudyTaskState.FAILED.value
                group.error = f"{type(exc).__name__}: {exc}"
                continue
            if not result.get("ok"):
                group.state = StudyTaskState.FAILED.value
                group.error = "; ".join(result.get("errors", []))
                continue
            group.state = StudyTaskState.VALIDATED.value
            group.delta_e_ev = (result.get("results") or {}).get("delta_e_ev")
            group.delta_delta_e_ev = (result.get("results") or {}).get(
                "delta_delta_e_ev"
            )
            group.uses_declared_reference_assumption = bool(
                (result.get("results") or {}).get(
                    "uses_declared_reference_assumption"
                )
            )
            group.high_risk_absolute_binding_energy = bool(
                (result.get("results") or {}).get(
                    "high_risk_absolute_binding_energy"
                )
            )
            group.error = ""

    def _persist(self, report: dict[str, Any]) -> None:
        state = self.load_state()
        state.update(self._state_entries())
        self.save_state(state)
        # Keep calculation_matrix.json in lockstep (resume reads it).
        (Path(self.spec.study_dir) / "calculation_matrix.json").write_text(
            json.dumps(
                self.spec.calculation_matrix.model_dump(mode="json"),
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        report["study_state_path"] = str(self.state_path)
