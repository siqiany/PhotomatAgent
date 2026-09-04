"""Standalone molecular VASP tool surface (bounded, offline-friendly).

Tools:
    vasp_molecule.prepare / preflight / submit / status / collect /
    analyze_orbitals / analyze_esp / binding_energy

Contract: every tool returns a bounded payload (<= 4000 characters), writes
its full detail to a log file, and persists ``task_state.json`` under the
workflow directory so different sessions can resume the same workflow.
Scheduling states never become scientific evidence by themselves: evidence is
produced only when result validation passes.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from photomatagent.scientific.applications.vasp.molecular.binding import (
    BindingEnergyInput,
    compute_binding_energy,
)
from photomatagent.scientific.applications.vasp.molecular.models import (
    StageName,
    WorkflowSpec,
)
from photomatagent.scientific.remote.models import ResourceRequest

MAX_TOOL_CHARS = 4000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def bounded_payload(
    *, ok: bool, summary: dict[str, Any], errors: list[str],
    warnings: list[str] | None = None, artifacts: list[str] | None = None,
    note: str = "", evidence: int = 0, **extra: Any
) -> dict[str, Any]:
    """Assemble a bounded tool payload (never dumps source or big files)."""
    payload: dict[str, Any] = {
        "ok": ok,
        "summary": summary,
        "errors": errors[:10],
        "warnings": (warnings or [])[:10],
        "artifacts": (artifacts or [])[:12],
        "evidence_count": evidence,
        "note": note,
        "chars": 0,
        "timestamp": _now(),
    }
    payload.update(extra)
    payload["chars"] = len(json.dumps(payload, ensure_ascii=False))
    if payload["chars"] > MAX_TOOL_CHARS:
        # Last-resort trim: drop per-error details, keep the envelope.
        payload["summary"] = {k: v for k, v in payload["summary"].items()
                              if not isinstance(v, (dict, list)) or k in ("state",)}
        payload["errors"] = payload["errors"][:4]
        payload["chars"] = len(json.dumps(payload, ensure_ascii=False))
        payload["note"] = (payload["note"] + " [output trimmed]").strip()
    return payload


class MolecularVaspTools:
    """Session-bound facade over generation, preflight, lifecycle, analysis."""

    def __init__(
        self,
        *,
        session: Any,
        backend: Any,
        psp_dir: str | Path | None = None,
        workflow_dir: str | Path | None = None,
        log_dir: str | Path | None = None,
        module_name: str = "",
        env_script: str = "",
        remote_psp_dir: str = "",
        configured: bool = True,
    ) -> None:
        self.session = session
        self.backend = backend
        self.psp_dir = psp_dir
        self.workflow_dir = (
            Path(workflow_dir).expanduser().resolve()
            if workflow_dir is not None
            else None
        )
        self.log_dir = (
            Path(log_dir).expanduser().resolve() if log_dir is not None else None
        )
        self.module_name = module_name
        self.env_script = env_script
        self.remote_psp_dir = remote_psp_dir
        self.configured = configured

    # -- helpers ------------------------------------------------------------

    def _load_or_create_task_state(self, root: Path) -> Any:
        from photomatagent.scientific.applications.vasp.molecular.workflow import (
            TaskState,
            load_task_state,
        )

        state = load_task_state(root)
        if state is None:
            state = TaskState(workflow_dir=str(root), molecule_name="molecule")
        return state

    def _persist_stage_entry(self, root: Path, update: dict[str, Any]) -> Any:
        from photomatagent.scientific.applications.vasp.molecular.workflow import (
            save_task_state,
        )

        state = self._load_or_create_task_state(root)
        entries = {item.stage: item for item in state.stages}
        existing = entries.get(update["stage"])
        if existing is None:
            from photomatagent.scientific.applications.vasp.molecular.workflow import (
                StageTask,
            )

            existing = StageTask(stage=update["stage"], state="PREPARED")
            entries[existing.stage] = existing
        for key, value in update.items():
            if key != "stage":
                setattr(existing, key, value)
        state.stages = list(entries.values())
        save_task_state(root, state)
        return state

    def _log(self, tool: str, text: str) -> Path:
        if self.log_dir is None:
            self.log_dir = Path.cwd() / "output" / "molecule_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f"{tool}.log"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"--- {_now()} ---\n{text}\n")
        return path

    def _resolve_workflow(self, workflow: WorkflowSpec | None) -> tuple[WorkflowSpec, Path]:
        if workflow is None:
            if self.workflow_dir is None:
                raise ValueError("workflow or workflow_dir is required")
            manifest = self.workflow_dir / "workflow.json"
            if not manifest.is_file():
                raise ValueError(f"workflow.json missing in {self.workflow_dir}")
            workflow = WorkflowSpec.model_validate_json(
                manifest.read_text(encoding="utf-8")
            )
        root = (
            self.workflow_dir
            or Path.cwd() / "output" / f"molecule_{workflow.molecule.name}"
        )
        return workflow, root

    # -- tools --------------------------------------------------------------

    async def prepare(
        self,
        workflow: WorkflowSpec | None = None,
        *,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Generate the typed stage tree + preflight.json + task_state.json."""
        from photomatagent.scientific.applications.vasp.molecular.generator import (
            MolecularVaspGenerator,
        )
        from photomatagent.scientific.applications.vasp.molecular.preflight import (
            run_molecular_preflight,
            save_preflight_report,
        )
        from photomatagent.scientific.applications.vasp.molecular.workflow import (
            TaskState,
            load_task_state,
        )

        workflow, root = self._resolve_workflow(workflow)
        if output_dir is not None:
            root = Path(output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        try:
            result = MolecularVaspGenerator(self.psp_dir).generate(
                workflow, root / "inputs", write_potcar=False
            )
            # The generator's manifest (inputs/workflow.json) lacks the full
            # typed StageSpec INCARs; the workflow root keeps the complete
            # typed spec so resumed sessions can re-run preflight and submit.
            (root / "workflow.json").write_text(
                workflow.model_dump_json(indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            return bounded_payload(
                ok=False, summary={}, errors=[f"prepare failed: {exc}"],
                note="no inputs were written",
            )
        preflight = run_molecular_preflight(
            workflow,
            psp_dir=self.psp_dir,
            output_dir=root,
        )
        # signature: save_preflight_report(report, output_dir)
        save_preflight_report(preflight, root)
        if load_task_state(root) is None:
            save = TaskState(workflow_dir=str(root), molecule_name=workflow.molecule.name)
            (root / "task_state.json").write_text(
                json.dumps(save.model_dump(mode="json"), indent=2) + "\n",
                encoding="utf-8",
            )
        summary = {
            "preflight_passed": preflight.passed,
            "stages": [stage.name.value for stage in workflow.stages],
            "nelect": preflight.summary.nelect if preflight.summary else None,
            "formula": preflight.summary.formula if preflight.summary else None,
        }
        errors = [issue.message for issue in preflight.errors[:10]]
        self._log("prepare", json.dumps(result, ensure_ascii=False, indent=2))
        return bounded_payload(
            ok=preflight.passed,
            summary=summary,
            errors=errors,
            warnings=[issue.message for issue in preflight.warnings[:10]],
            artifacts=[str(root / "workflow.json"), str(root / "preflight.json")],
            note="prepared offline; nothing was submitted",
        )

    async def preflight(
        self, workflow: WorkflowSpec | None = None
    ) -> dict[str, Any]:
        from photomatagent.scientific.applications.vasp.molecular.preflight import (
            render_agent_text,
            run_molecular_preflight,
        )

        workflow, root = self._resolve_workflow(workflow)
        report = run_molecular_preflight(
            workflow, psp_dir=self.psp_dir, output_dir=root
        )
        text = render_agent_text(report)
        self._log("preflight", text)
        return bounded_payload(
            ok=report.passed,
            summary={
                "passed": report.passed,
                "errors": len(report.errors),
                "warnings": len(report.warnings),
                "checks": len(report.checks),
                "formula": report.summary.formula if report.summary else None,
                "nelect": report.summary.nelect if report.summary else None,
            },
            errors=[issue.message for issue in report.errors[:10]],
            warnings=[issue.message for issue in report.warnings[:10]],
            artifacts=[str(root / "preflight.json")],
            note="offline deterministic preflight; see preflight.json",
        )

    async def submit(
        self,
        stage: str | StageName,
        workflow: WorkflowSpec | None = None,
        *,
        wait: bool = False,
        wait_timeout_seconds: float = 3600.0,
        force_new_attempt: bool = False,
        resource: ResourceRequest | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Submit one stage under the preflight gate (submit-once semantics)."""
        from photomatagent.scientific.applications.vasp.molecular.preflight import (
            preflight_gate,
            run_molecular_preflight,
        )
        from photomatagent.scientific.applications.vasp.molecular.workflow import (
            load_task_state,
            sanitize_resource,
            wait_for_terminal_state,
        )
        from photomatagent.scientific.remote.lifecycle import RemoteArtifactCopy

        workflow, root = self._resolve_workflow(workflow)
        stage_name = StageName(stage)
        stage_spec = next(
            (s for s in workflow.stages if s.name is stage_name), None
        )
        if stage_spec is None:
            return bounded_payload(
                ok=False, summary={},
                errors=[f"stage {stage_name.value} not in workflow"],
            )
        report = run_molecular_preflight(
            workflow, psp_dir=self.psp_dir, output_dir=root
        )
        gate = preflight_gate(report, report_path=str(root / "preflight.json"))
        task_state = load_task_state(root)
        task_map = task_state.stage_map() if task_state is not None else {}
        existing = task_map.get(stage_name.value) if task_map else None
        if existing is not None and existing.state in {
            "COMPLETED", "COLLECTED", "VALIDATED",
        }:
            if existing.state == "VALIDATED":
                return bounded_payload(
                    ok=True,
                    summary={
                        "stage": stage_name.value,
                        "state": existing.state,
                        "already": True,
                    },
                    errors=[],
                    note=(
                        "stage already validated; resume with "
                        "vasp_molecule.resume_workflow for downstream stages"
                    ),
                )
            if not force_new_attempt:
                return bounded_payload(
                    ok=False,
                    summary={
                        "stage": stage_name.value,
                        "state": existing.state,
                    },
                    errors=[
                        "stage already reached scheduler completion; "
                        "collect and validate instead of resubmitting "
                        "(submit-once never creates a second job)"
                    ],
                    note=(
                        "run vasp_molecule.collect (or resume_workflow) to "
                        "validate the existing job"
                    ),
                )
        stage_dir = root / "inputs" / (
            f"{workflow.stages.index(stage_spec) + 1:02d}_{stage_name.value}"
        )
        from photomatagent.scientific.applications.vasp.molecular.slurm import (
            cleanup_materialized_potcar,
            materialize_stage_potcar,
            potcar_mode_of_stage,
            potcar_symbols_from_stage,
            render_stage_slurm,
        )

        potcar_mode = potcar_mode_of_stage(
            stage_dir,
            remote_psp_dir=self.remote_psp_dir,
            psp_dir=self.psp_dir,
        )
        if potcar_mode == "none":
            return bounded_payload(
                ok=False,
                summary={"stage": stage_name.value},
                errors=[
                    "no POTCAR strategy: materialize POTCAR locally or "
                    "configure SCNET_VASP_PSP_DIR; submission refused"
                ],
                note="molecular submissions always need a POTCAR strategy",
            )
        # Local mode without a curated POTCAR: assemble the concatenated
        # POTCAR from the resolved local PAW-PBE library in POSCAR order and
        # upload it to the unique remote job directory only. The assembled
        # bytes are removed again after the upload attempt; they never enter
        # logs, the registry, JSON payloads or model output.
        potcar_materialized = (
            potcar_mode == "local"
            and not (stage_dir / "POTCAR").is_file()
        )
        if potcar_materialized:
            try:
                potcar_materialized = materialize_stage_potcar(
                    stage_dir,
                    self.psp_dir,
                    potcar_symbols_from_stage(stage_dir),
                )
            except Exception as exc:
                return bounded_payload(
                    ok=False,
                    summary={"stage": stage_name.value},
                    errors=[f"local POTCAR assembly failed: {exc}"],
                    note="no upload or sbatch was performed",
                )
        # Dependency structure: copy the relax-chain CONTCAR into POSCAR for
        # stages that declare it (same rule as the workflow runner).
        if stage_spec.depends_on is not None and "CONTCAR" in stage_spec.required_upstream_outputs:
            if task_map:
                contcar_source: Path | None = None
                positions = {
                    stage.name.value: index
                    for index, stage in enumerate(workflow.stages)
                }
                cursor: StageName | None = stage_spec.depends_on
                while cursor is not None:
                    upstream = task_map.get(cursor.value)
                    if upstream is not None and upstream.results_dir:
                        candidate = Path(upstream.results_dir) / "CONTCAR"
                        if candidate.is_file():
                            contcar_source = candidate
                            break
                    cursor_index = positions.get(cursor.value)
                    if cursor_index is None:
                        break
                    cursor = workflow.stages[cursor_index].depends_on
                if contcar_source is None:
                    return bounded_payload(
                        ok=False, summary={"stage": stage_name.value},
                        errors=[
                            "stage requires CONTCAR from the relax chain but "
                            "no completed upstream provides it"
                        ],
                    )
                stage_dir.mkdir(parents=True, exist_ok=True)
                import shutil

                shutil.copy2(contcar_source, stage_dir / "POSCAR")
        remote_copies: list[RemoteArtifactCopy] = []
        if stage_spec.depends_on is not None:
            upstream = task_map.get(stage_spec.depends_on.value)
            if upstream is not None and upstream.remote_directory:
                remote_copies = [
                    RemoteArtifactCopy(
                        source_remote_directory=upstream.remote_directory,
                        filename=filename,
                    )
                    for filename in stage_spec.required_upstream_outputs
                    if filename in {"WAVECAR", "CHGCAR"}
                ]
        try:
            submit = await self.session.submit_once(
                application="vasp_molecular",
                workflow_stage=stage_name.value,
                job_name=f"{workflow.molecule.name}-{stage_name.value}",
                local_input_dir=stage_dir,
                gate=gate,
                resource=resource or sanitize_resource(workflow, stage_spec),
                executable="vasp_std",
                script_name="run.slurm",
                request_id=request_id,
                force_new_attempt=force_new_attempt,
                remote_copies=remote_copies,
                script_renderer=lambda job_name, resource: render_stage_slurm(
                    job_name=job_name,
                    resource=resource,
                    stage_dir=stage_dir,
                    module_name=self.module_name,
                    env_script=self.env_script,
                    remote_psp_dir=self.remote_psp_dir,
                ),
                potcar_mode=potcar_mode,
                potcar_symbols=potcar_symbols_from_stage(stage_dir),
                remote_psp_dir=self.remote_psp_dir,
            )
        finally:
            cleanup_materialized_potcar(
                stage_dir, materialized=potcar_materialized
            )
        self._persist_stage_entry(
            root,
            {
                "stage": stage_name.value,
                "state": submit.record.get("state", "PREPARED"),
                "request_id": submit.request_id,
                "job_id": submit.record.get("job_id") or "",
                "remote_directory": submit.record.get("remote_directory") or "",
                "stage_dir": str(stage_dir),
                "error": submit.error,
            },
        )
        state = submit.record.get("state")
        if wait and submit.submitted:
            state = await wait_for_terminal_state(
                self.session, submit.request_id,
                timeout_seconds=wait_timeout_seconds,
            )
            self._persist_stage_entry(root, {"stage": stage_name.value, "state": state})
        detail = json.dumps(submit.model_dump(mode="json"), ensure_ascii=False, indent=2)
        self._log(f"submit_{stage_name.value}", detail)
        errors = [submit.error] if submit.error else []
        if not submit.submitted and not errors:
            errors.append("submission did not produce a job")
        return bounded_payload(
            ok=submit.submitted,
            summary={
                "stage": stage_name.value,
                "request_id": submit.request_id,
                "state": state,
                "job_id": submit.record.get("job_id"),
                "remote_directory": submit.record.get("remote_directory"),
            },
            errors=errors,
            note=(
                "reconciliation required before any retry"
                if submit.needs_reconciliation
                else "submit-once: same request_id never creates a second job"
            ),
        )

    async def status(self, stage: str | StageName) -> dict[str, Any]:
        """Read the registry record; query the scheduler when job_id exists."""
        from photomatagent.scientific.applications.vasp.molecular.workflow import (
            load_task_state,
        )

        task_state = load_task_state(self.workflow_dir) if self.workflow_dir else None
        entry = None
        if task_state is not None:
            entry = task_state.stage_map().get(StageName(stage).value)
        if entry is None or not entry.request_id:
            return bounded_payload(
                ok=False, summary={"stage": StageName(stage).value},
                errors=["no task_state entry for this stage; run prepare first"],
            )
        refresh = await self.session.refresh_status(entry.request_id)
        record = self.session.registry.get(entry.request_id)
        self._log(
            f"status_{StageName(stage).value}",
            json.dumps(
                refresh.model_dump(mode="json"), ensure_ascii=False, indent=2
            ),
        )
        return bounded_payload(
            ok=refresh.ok and refresh.state is not None,
            summary={
                "stage": StageName(stage).value,
                "job_id": refresh.job_id,
                "scheduler_state": refresh.scheduler_state,
                "lifecycle_state": record.state.value if record else None,
            },
            errors=[refresh.error] if refresh.error else [],
            note=(
                "query failure is UNKNOWN, not a job failure"
                if refresh.query_failed
                else "scheduler state; not scientific validity"
            ),
        )

    async def collect(
        self,
        stage: str | StageName,
        *,
        local_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Download, parse and validate one stage's results; evidence gated."""
        from photomatagent.scientific.applications.vasp.molecular.results import (
            analyze_result_dir,
            scientific_evidence,
        )
        from photomatagent.scientific.applications.vasp.molecular.workflow import (
            load_task_state,
            stage_result_files,
        )
        from photomatagent.scientific.remote.registry import JobLifecycleState

        stage_name = StageName(stage)
        task_state = load_task_state(self.workflow_dir) if self.workflow_dir else None
        if task_state is None:
            return bounded_payload(
                ok=False, summary={"stage": stage_name.value},
                errors=["no task_state; run prepare first"],
            )
        entry = task_state.stage_map().get(stage_name.value)
        if entry is None or not entry.remote_directory:
            return bounded_payload(
                ok=False, summary={"stage": stage_name.value},
                errors=["stage has no remote directory recorded"],
            )
        if entry.state == JobLifecycleState.VALIDATED.value:
            return bounded_payload(
                ok=True,
                errors=[],
                summary={
                    "stage": stage_name.value,
                    "state": JobLifecycleState.VALIDATED.value,
                    "already": True,
                },
                note="stage already validated; results are on disk",
            )
        if entry.state not in {
            JobLifecycleState.COMPLETED.value,
            JobLifecycleState.COLLECTED.value,
        }:
            return bounded_payload(
                ok=False, summary={"stage": stage_name.value, "state": entry.state},
                errors=[
                    "only scheduler-COMPLETED stages are (re-)collected; "
                    f"state is {entry.state}"
                ],
            )
        root = Path(str(self.workflow_dir))
        result_dir = Path(local_dir) if local_dir is not None else root / "results" / stage_name.value
        result_dir.mkdir(parents=True, exist_ok=True)
        downloaded = await self.backend.download_files(
            entry.remote_directory, stage_result_files(stage_name), result_dir
        )
        if stage_name in {
            StageName.ORBITAL_HOMO,
            StageName.ORBITAL_LUMO,
        }:
            # VASP 5.4.4 writes PARCHG under several names; discover every
            # PARCHG* artifact so the orbital density is never lost.
            from photomatagent.scientific.applications.vasp.molecular.workflow import (
                _download_parchg_artifacts,
            )

            downloaded.extend(
                await _download_parchg_artifacts(
                    self.backend, entry.remote_directory, result_dir
                )
            )
        # Mirror the run inputs into the result dir so EDIFF/NSW/ISPIN and
        # the structure are analyzed from the exact submitted files.
        import shutil

        for name in ("INCAR", "KPOINTS", "POSCAR"):
            source_file = Path(str(entry.stage_dir or "")) / name
            if source_file.is_file():
                shutil.copy2(source_file, result_dir / name)
        workflow = WorkflowSpec.model_validate_json(
            (root / "workflow.json").read_text(encoding="utf-8")
        )
        analysis = analyze_result_dir(
            result_dir,
            charge=workflow.molecule.total_charge,
            spin_multiplicity=workflow.molecule.spin_multiplicity,
            box_ang=workflow.molecule.box_ang,
        )
        evidence = scientific_evidence(analysis, tool="vasp_molecule.collect")
        (result_dir / "results.json").write_text(
            json.dumps(analysis, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (result_dir / "evidence.json").write_text(
            json.dumps(
                [item.model_dump(mode="json") for item in evidence],
                indent=2, ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        validated = bool(analysis.get("validated"))
        if entry.request_id:
            # SQLite registry: COMPLETED -> COLLECTED -> VALIDATED (only when
            # the scientific analysis passes). task_state stays in lockstep.
            self.session.mark_result_state(
                entry.request_id,
                collected=True,
                validated=validated,
                evidence=len(evidence),
                error="; ".join(analysis.get("errors", [])),
            )
        persist = task_state.stage_map().get(stage_name.value)
        if persist is not None:
            persist.results_dir = str(result_dir)
            persist.validated = validated
            persist.state = (
                JobLifecycleState.VALIDATED.value
                if validated
                else JobLifecycleState.COLLECTED.value
            )
            persist.error = (
                "" if validated else "; ".join(analysis.get("errors", []))
            )
            from photomatagent.scientific.applications.vasp.molecular.workflow import (
                save_task_state,
            )

            save_task_state(root, task_state)
        summary = {
            "stage": stage_name.value,
            "validated": validated,
            "state": persist.state if persist is not None else None,
            "e0_ev": (analysis.get("energy") or {}).get("e_0_ev"),
            "formula": (analysis.get("identity") or {}).get("formula"),
            "scf_converged": (analysis.get("scf") or {}).get("converged"),
            "evidence": len(evidence),
        }
        self._log(
            f"collect_{stage_name.value}",
            json.dumps(analysis, ensure_ascii=False, indent=2),
        )
        return bounded_payload(
            ok=validated,
            summary=summary,
            errors=analysis.get("errors", []),
            warnings=analysis.get("warnings", []),
            artifacts=[path.name for path in downloaded] + ["results.json"],
            evidence=len(evidence),
            note=(
                "evidence was generated only because validation passed"
                if validated
                else "not validated; no scientific evidence was produced"
            ),
        )

    async def analyze_orbitals(
        self, result_dir: str | Path, *, charge: int = 0, spin_multiplicity: int = 1,
        box_ang: float | None = None,
    ) -> dict[str, Any]:
        """HOMO/LUMO + vacuum alignment from EIGENVAL + LOCPOT (offline)."""
        from photomatagent.scientific.applications.vasp.molecular.results import (
            analyze_result_dir,
            scientific_evidence,
        )

        directory = Path(result_dir)
        workflow_dir = self.workflow_dir
        box = box_ang
        if box is None and workflow_dir is not None:
            manifest = Path(workflow_dir) / "workflow.json"
            if manifest.is_file():
                workflow = WorkflowSpec.model_validate_json(
                    manifest.read_text(encoding="utf-8")
                )
                box = workflow.molecule.box_ang
                charge = workflow.molecule.total_charge
                spin_multiplicity = workflow.molecule.spin_multiplicity
        analysis = analyze_result_dir(
            directory, charge=charge, spin_multiplicity=spin_multiplicity,
            box_ang=box,
        )
        evidence = scientific_evidence(analysis, tool="vasp_molecule.analyze_orbitals")
        orbitals = analysis.get("orbitals", {})
        vacuum = analysis.get("vacuum", {})
        summary = {
            "homo_band": orbitals.get("homo_band"),
            "lumo_band": orbitals.get("lumo_band"),
            "homo_raw_ev": orbitals.get("homo_raw_ev"),
            "lumo_raw_ev": orbitals.get("lumo_raw_ev"),
            "ks_gap_ev": orbitals.get("ks_gap_ev"),
            "vacuum_level_ev": vacuum.get("level_ev"),
            "aligned_homo_ev": vacuum.get("aligned_homo_ev"),
            "aligned_lumo_ev": vacuum.get("aligned_lumo_ev"),
        }
        self._log(
            "analyze_orbitals",
            json.dumps(analysis, ensure_ascii=False, indent=2),
        )
        return bounded_payload(
            ok=bool(analysis.get("validated")),
            summary=summary,
            errors=analysis.get("errors", []),
            warnings=analysis.get("warnings", []),
            artifacts=[str(directory / "EIGENVAL"), str(directory / "LOCPOT")],
            evidence=len(evidence),
            note=(
                "vacuum alignment uses LOCPOT; raw values must not be "
                "compared across molecules"
                if vacuum.get("level_ev") is not None
                else "LOCPOT absent; energies are raw, not vacuum-aligned"
            ),
        )

    async def analyze_esp(
        self, result_dir: str | Path
    ) -> dict[str, Any]:
        """ESP/LOCPOT metadata; the potential grid itself never leaves disk."""
        from photomatagent.scientific.applications.vasp.molecular.results import (
            esp_metadata,
        )

        metadata = esp_metadata(result_dir)
        incar = Path(result_dir) / "INCAR"
        lvhar = False
        if incar.is_file():
            from photomatagent.scientific.applications.vasp.molecular.render import (
                parse_bool,
                parse_incar,
            )

            lvhar = parse_bool(str(parse_incar(incar.read_text(encoding="utf-8")).get("LVHAR", ""))) is True
        metadata["lvhar_declared"] = lvhar
        self._log("analyze_esp", json.dumps(metadata, ensure_ascii=False, indent=2))
        return bounded_payload(
            ok=bool(metadata.get("has_locpot")),
            summary={
                "has_locpot": metadata.get("has_locpot"),
                "grid": metadata.get("grid"),
                "spacing_ang": metadata.get("spacing_ang"),
                "lvhar_declared": lvhar,
            },
            errors=[metadata.get("parse_error", "")] if metadata.get("parse_error") else [],
            artifacts=[str(Path(result_dir) / "LOCPOT")],
            note="only grid metadata is returned; LOCPOT content stays on disk",
        )

    async def binding_energy(
        self, inputs: BindingEnergyInput | dict[str, Any]
    ) -> dict[str, Any]:
        """Electronic binding energy with parameter-consistency checks."""
        if isinstance(inputs, dict):
            inputs = BindingEnergyInput.model_validate(inputs)
        result = compute_binding_energy(inputs)
        self._log("binding_energy", json.dumps(result, ensure_ascii=False, indent=2))
        return bounded_payload(
            ok=bool(result.get("ok")),
            summary={
                "complex": inputs.complex_name,
                "delta_e_ev": (result.get("results") or {}).get("delta_e_ev"),
                "delta_delta_e_ev": (result.get("results") or {}).get("delta_delta_e_ev"),
                "electronic_only": (result.get("results") or {}).get("electronic_only"),
            },
            errors=result.get("errors", []),
            warnings=result.get("warnings", []),
            note=result.get("method", {}).get("kind", ""),
        )

    async def run_workflow(
        self,
        workflow: WorkflowSpec | None = None,
        *,
        wait: bool = True,
        collect: bool = True,
        stop_on_failure: bool = True,
        only: list[str] | None = None,
        wait_timeout_seconds: float = 3600.0,
    ) -> dict[str, Any]:
        """Run the full DAG with resume support (completed stages are kept)."""
        from photomatagent.scientific.applications.vasp.molecular.workflow import (
            run_molecule_workflow,
        )

        workflow, root = self._resolve_workflow(workflow)
        report = await run_molecule_workflow(
            workflow, root, session=self.session, backend=self.backend,
            psp_dir=self.psp_dir, wait=wait, collect=collect,
            stop_on_failure=stop_on_failure, only=only,
            wait_timeout_seconds=wait_timeout_seconds,
            module_name=self.module_name,
            env_script=self.env_script,
            remote_psp_dir=self.remote_psp_dir,
        )
        self._log("run_workflow", json.dumps(report, ensure_ascii=False, indent=2))
        return bounded_payload(
            ok=not report.get("error") and not report.get("blocked"),
            summary={
                "stages": [stage["stage"] for stage in report.get("stages", [])],
                "completed": report.get("completed", []),
                "resumed": report.get("resumed", []),
                "blocked": report.get("blocked", []),
                "evidence_count": report.get("evidence_count", 0),
                "preflight_passed": report.get("preflight_passed"),
            },
            errors=[report["error"]] if report.get("error") else [],
            artifacts=[str(root / "task_state.json")],
            note=(
                "workflow finished with a persistent task_state; resume "
                "later without resubmitting completed stages"
            ),
        )
