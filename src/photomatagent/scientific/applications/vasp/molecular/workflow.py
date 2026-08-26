"""Isolated-molecule VASP workflow DAG: build, persist, run, resume.

Canonical DAG (typed stages, explicit requires/produces contracts):

    structure
      -> relax
      -> static_preconverge        (CONTCAR; WAVECAR/CHGCAR seeds; no LDIPOL)
      -> corrected_static          (restart; strict EDIFF; dipole correction)
         |-> orbital_homo          (IBAND from corrected_static occupations;
         |                            LVHAR LOCPOT for vacuum alignment)
         |-> orbital_lumo          (same for LUMO)
         `-> esp                   (optimized structure + CHGCAR; LVHAR LOCPOT)

Any stage failure blocks its dependents; successful stages are recorded in
``task_state.json`` so the workflow can be resumed later without resubmitting
completed jobs (the receive-only registry + submit-once session enforce this
at the submission layer as well).
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from photomatagent.scientific.applications.vasp.molecular.models import (
    MoleculeSpec,
    MonopoleMethod,
    ResourceClass,
    ResourceProfile,
    ScreenDecision,
    StageName,
    StageSpec,
    WorkflowSpec,
)
from photomatagent.scientific.applications.vasp.molecular.calibration import (
    CalibrationRecord,
    calibration_applicable,
)
from photomatagent.scientific.applications.vasp.molecular.psp_metadata import (
    resolve_potcar_metadata,
)
from photomatagent.scientific.applications.vasp.molecular.structures import (
    grouped_symbols,
    read_structure,
)
from photomatagent.scientific.applications.vasp.molecular.templates import (
    corrected_static_incar,
    esp_incar,
    make_stage,
    orbital_single_incar,
    relax_incar,
    static_incar,
    static_hse_incar,
    static_preconverge_incar,
)


def _assemble_workflow(
    *,
    molecule: MoleculeSpec,
    stages: list[StageSpec],
    scientific_method: str,
    profile: ResourceProfile,
    encut: float,
    monopole: MonopoleMethod,
    dipole: bool,
    tasks_per_node: int,
    walltime_minutes: int,
    ispin: int,
    nupdown: int | None,
    magmom: list[float] | None,
    calibration: CalibrationRecord | None,
    atom_count: int,
    screen_decision: ScreenDecision | None = None,
    provenance_extra: dict[str, Any] | None = None,
) -> WorkflowSpec:
    """Shared WorkflowSpec assembly (normal DAG and screen-only workflows)."""
    from photomatagent.scientific.applications.vasp.molecular.calibration import (
        finalize_calibration,
    )
    from photomatagent.scientific.applications.vasp.molecular.models import (
        CorrectionPolicy,
        PreflightConfig,
        ResourceCeiling,
        ResourcePlan,
        SmokeBaseline,
    )

    finalized_calibration: CalibrationRecord | None = None
    if calibration is not None:
        finalized_calibration = finalize_calibration(calibration)
        applicable, reasons = calibration_applicable(
            finalized_calibration,
            atom_count=atom_count,
            box_ang=molecule.box_ang,
            encut_ev=encut,
        )
        if not applicable:
            raise ValueError(
                "calibration record does not apply to this molecule: "
                + "; ".join(reasons)
            )

    return WorkflowSpec(
        molecule=molecule,
        stages=stages,
        scientific_method=scientific_method,
        correction_policy=CorrectionPolicy(
            monopole_method=monopole, dipole=dipole
        ),
        resource_ceiling=ResourceCeiling(
            nodes=1,
            tasks_per_node=tasks_per_node,
            walltime_minutes=walltime_minutes,
        ),
        resource_plan=ResourcePlan(
            profile=profile,
            tasks_per_node=tasks_per_node,
            walltime_minutes=walltime_minutes,
            calibration=finalized_calibration,
            calibration_note=(
                f"calibration {finalized_calibration.calibration_id} from "
                f"job {finalized_calibration.source_job_id}"
                if finalized_calibration is not None
                else ""
            ),
        ),
        smoke_baseline=SmokeBaseline(),
        screen_decision=screen_decision,
        preflight_config=PreflightConfig(
            encut_floor_ev=400.0
            if profile is ResourceProfile.SMOKE
            else 520.0,
            encut_max_enmax_ratio=(
                # The verified smoke run ran ENCUT = max ENMAX (400 eV).
                1.0 if profile is ResourceProfile.SMOKE else 1.3
            ),
            # The verified smoke run used a 20 A box: 10 A of vacuum per
            # side is only reachable in production-size cells (>=30 A).
            min_vacuum_per_side_ang=(
                5.0 if profile is ResourceProfile.SMOKE else 10.0
            ),
        ),
        provenance={
            "builder": "build_molecule_workflow",
            "builder_version": 1,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "spin": {
                "spin_multiplicity": molecule.spin_multiplicity,
                "ispin": ispin,
                "spin_polarized": molecule.spin_polarized,
                "nupdown": nupdown,
                "magmom_declared": magmom is not None,
                "assumptions": molecule.spin_assumptions(),
            },
            **(provenance_extra or {}),
        },
    )


def build_molecule_workflow(
    molecule: MoleculeSpec,
    *,
    psp_dir: str | Path | None = None,
    encut_ev: float | None = None,
    spin: int | None = None,
    nupdown: int | None = None,
    magmom: list[float] | None = None,
    lmono: bool | None = None,
    dipole: bool = True,
    include_orbital_homo: bool = True,
    include_orbital_lumo: bool = True,
    include_esp: bool = True,
    include_hse06: bool = False,
    screen_decision: ScreenDecision | None = None,
    screen_only: bool = False,
    resource_profile: ResourceProfile = ResourceProfile.SMOKE,
    tasks_per_node: int = 8,
    walltime_minutes: int = 20,
    ncore: int = 2,
    calibration: CalibrationRecord | None = None,
    scientific_method: str = "isolated-molecule PBE-D3(BJ) Gamma-only DAG",
) -> WorkflowSpec:
    """Build the canonical typed DAG from a molecule spec.

    ``encut_ev`` defaults to 400 eV for the smoke profile and 520 eV for the
    production profile; NELECT comes from the real POTCAR metadata when
    ``psp_dir`` resolves, otherwise INCARs omit NELECT and the preflight
    refuses submission (PSP_METADATA_UNREADABLE).
    """
    profile = ResourceProfile(resource_profile)
    encut = encut_ev if encut_ev is not None else (520.0 if profile is ResourceProfile.PRODUCTION else 400.0)
    # ISPIN is NEVER the spin multiplicity: a triplet must not render
    # ISPIN = 3. Explicit ``spin`` wins; otherwise the molecule's typed
    # derivation applies (odd electrons / multiplicity > 1 -> ISPIN 2).
    ispin = molecule.effective_ispin() if spin is None else int(spin)
    if ispin not in {1, 2}:
        raise ValueError(
            "ISPIN must be 1 or 2; got "
            f"{ispin} (spin_multiplicity={molecule.spin_multiplicity} is "
            "not an ISPIN value)"
        )
    nupdown = molecule.nupdown if nupdown is None else nupdown
    magmom = molecule.magmom if magmom is None else magmom
    if calibration is not None:
        # Measured calibration drives the per-node task count and a
        # walltime derived from the measured per-stage elapsed time
        # (still capped by the caller's explicit parameters).
        tasks_per_node = calibration.tasks or tasks_per_node
        if calibration.elapsed_seconds > 0:
            from_calibration = max(
                20, int(calibration.elapsed_seconds / 60.0 * 1.5) + 5
            )
            walltime_minutes = (
                min(walltime_minutes, from_calibration)
                if walltime_minutes
                else from_calibration
            )
    needs_monopole = molecule.total_charge != 0
    if lmono is None:
        lmono = needs_monopole  # charged systems default to LMONO
    if needs_monopole and lmono:
        # Phase-1 correction policy: LMONO replaces the dipole correction.
        dipole = False
    assert molecule.structure_path is not None
    structure = read_structure(
        molecule.structure_path, kind=molecule.structure_kind
    )
    elements, counts = grouped_symbols(structure.symbols)
    nelect: float | None = None
    if psp_dir is not None:
        resolution = resolve_potcar_metadata(
            molecule, elements, psp_dir=psp_dir
        )
        neutral = sum(
            block.zval * count
            for block, count in zip(resolution.blocks, counts, strict=True)
        )
        nelect = neutral - molecule.total_charge

    monopole = MonopoleMethod.LMONO if lmono and needs_monopole else MonopoleMethod.NONE
    stages: list[StageSpec] = []

    if screen_only:
        # Cheap conformer screen: ONE static single point on the candidate
        # geometry. It ranks candidates by E0 only; it is never a production
        # value and never feeds downstream stages.
        stages.append(
            make_stage(
                StageName.STATIC,
                incar=static_incar(
                    spin=ispin,
                    encut=encut,
                    nelect=nelect,
                    lmono=monopole is MonopoleMethod.LMONO,
                    dipole=dipole,
                    ncore=ncore,
                ),
                produced_outputs=["OSZICAR", "EIGENVAL", "vasprun.xml"],
                resource_class=ResourceClass.SMOKE,
                description=(
                    "conformer static screen (cheap E0 ranking; never a "
                    "production value)"
                ),
            )
        )
        return _assemble_workflow(
            molecule=molecule,
            stages=stages,
            profile=profile,
            encut=encut,
            monopole=monopole,
            dipole=dipole,
            tasks_per_node=tasks_per_node,
            walltime_minutes=walltime_minutes,
            ispin=ispin,
            nupdown=nupdown,
            magmom=magmom,
            calibration=calibration,
            atom_count=len(structure.symbols),
            scientific_method=(
                "conformer static screen; E0 ranking only, not production"
            ),
        )

    stages.append(
        make_stage(
            StageName.RELAX,
            incar=relax_incar(
                spin=ispin,
                encut=encut,
                nelect=nelect,
                lmono=monopole is MonopoleMethod.LMONO,
                dipole=dipole,
                ncore=ncore,
                nupdown=nupdown,
                magmom=magmom,
            ),
            produced_outputs=["CONTCAR", "CHGCAR", "OSZICAR", "vasprun.xml"],
            resource_class=ResourceClass.STANDARD,
            description="geometry relaxation in the fixed vacuum box",
        )
    )
    stages.append(
        make_stage(
            StageName.STATIC_PRECONVERGE,
            depends_on=StageName.RELAX,
            incar=static_preconverge_incar(
                spin=ispin,
                encut=encut,
                nelect=nelect,
                lmono=monopole is MonopoleMethod.LMONO,
                dipole=False,  # deliberate: cheap WAVECAR/CHGCAR source
                ncore=ncore,
            ),
            required_upstream_outputs=["CONTCAR"],
            produced_outputs=["WAVECAR", "CHGCAR", "EIGENVAL", "OSZICAR", "vasprun.xml"],
            resource_class=ResourceClass.SMOKE,
            description=(
                "preconvergence single point (loose EDIFF, no LDIPOL); "
                "produces the WAVECAR/CHGCAR sources; never a production value"
            ),
        )
    )
    stages.append(
        make_stage(
            StageName.CORRECTED_STATIC,
            depends_on=StageName.STATIC_PRECONVERGE,
            incar=corrected_static_incar(
                spin=ispin,
                encut=encut,
                nelect=nelect,
                lmono=monopole is MonopoleMethod.LMONO,
                dipole=dipole,
                ncore=ncore,
            ),
            required_upstream_outputs=["CONTCAR", "WAVECAR", "CHGCAR"],
            produced_outputs=[
                "WAVECAR", "CHGCAR", "EIGENVAL", "OSZICAR", "vasprun.xml",
            ],
            resource_class=ResourceClass.STANDARD,
            description=(
                "corrected static single point with strict EDIFF and the "
                "declared electrostatic corrections; primary production stage"
            ),
        )
    )
    if include_orbital_homo:
        stages.append(
            make_stage(
                StageName.ORBITAL_HOMO,
                depends_on=StageName.CORRECTED_STATIC,
                incar=orbital_single_incar(
                    spin=ispin,
                    encut=encut,
                    nelect=nelect,
                    lmono=monopole is MonopoleMethod.LMONO,
                    dipole=dipole,
                    ncore=ncore,
                    iband=0,  # replaced by the runner from occupancy analysis
                ),
                required_upstream_outputs=["WAVECAR", "EIGENVAL"],
                produced_outputs=[
                    "LOCPOT", "EIGENVAL", "OSZICAR",
                    "PARCHG", "PARCHG.ALL",
                ],
                resource_class=ResourceClass.SMALL,
                description="HOMO single point + LVHAR LOCPOT (vacuum alignment)",
            )
        )
    if include_orbital_lumo:
        stages.append(
            make_stage(
                StageName.ORBITAL_LUMO,
                depends_on=StageName.CORRECTED_STATIC,
                incar=orbital_single_incar(
                    spin=ispin,
                    encut=encut,
                    nelect=nelect,
                    lmono=monopole is MonopoleMethod.LMONO,
                    dipole=dipole,
                    ncore=ncore,
                    iband=0,
                ),
                required_upstream_outputs=["WAVECAR", "EIGENVAL"],
                produced_outputs=[
                    "LOCPOT", "EIGENVAL", "OSZICAR",
                    "PARCHG", "PARCHG.ALL",
                ],
                resource_class=ResourceClass.SMALL,
                description="LUMO single point + LVHAR LOCPOT (vacuum alignment)",
            )
        )
    if include_esp:
        stages.append(
            make_stage(
                StageName.ESP,
                depends_on=StageName.CORRECTED_STATIC,
                incar=esp_incar(
                    spin=ispin,
                    encut=encut,
                    nelect=nelect,
                    lmono=monopole is MonopoleMethod.LMONO,
                    dipole=dipole,
                    ncore=ncore,
                ),
                required_upstream_outputs=["CONTCAR", "CHGCAR"],
                produced_outputs=["LOCPOT", "OSZICAR", "vasprun.xml"],
                resource_class=ResourceClass.STANDARD,
                description="ESP run: optimized structure + LVHAR LOCPOT",
            )
        )
    if include_hse06:
        stages.append(
            make_stage(
                StageName.STATIC_HSE,
                depends_on=StageName.CORRECTED_STATIC,
                incar=static_hse_incar(
                    spin=ispin,
                    encut=encut,
                    nelect=nelect,
                    lmono=monopole is MonopoleMethod.LMONO,
                    dipole=dipole,
                    ncore=ncore,
                ),
                required_upstream_outputs=["CONTCAR", "WAVECAR", "CHGCAR"],
                produced_outputs=["EIGENVAL", "OSZICAR", "vasprun.xml"],
                resource_class=ResourceClass.LARGE,
                description="HSE06 single point (gated by screen_decision)",
            )
        )
    return _assemble_workflow(
        molecule=molecule,
        stages=stages,
        scientific_method=scientific_method,
        profile=profile,
        encut=encut,
        monopole=monopole,
        dipole=dipole,
        tasks_per_node=tasks_per_node,
        walltime_minutes=walltime_minutes,
        ispin=ispin,
        nupdown=nupdown,
        magmom=magmom,
        calibration=calibration,
        atom_count=len(structure.symbols),
        screen_decision=screen_decision,
    )


# --------------------------------------------------------------------------
# persistent task state (workflow resume contract)
# --------------------------------------------------------------------------


class StageTask(BaseModel):
    stage: str
    state: str  # JobLifecycleState value
    request_id: str = ""
    job_id: str = ""
    stage_dir: str = ""
    remote_directory: str = ""
    results_dir: str = ""
    validated: bool = False
    error: str = ""
    attempt_id: str = ""
    retry_count: int = 0
    recovery_attempts: list[dict] = Field(default_factory=list)
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


class TaskState(BaseModel):
    workflow_dir: str
    molecule_name: str
    stages: list[StageTask] = Field(default_factory=list)
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def stage_map(self) -> dict[str, StageTask]:
        return {item.stage: item for item in self.stages}

    def completed(self) -> list[str]:
        return [
            item.stage
            for item in self.stages
            if item.state in _STAGE_DONE
        ]


_STAGE_DONE = ("VALIDATED",)


def stage_done(state: str) -> bool:
    """Only a validated stage satisfies downstream scientific dependencies.

    Scheduler COMPLETED carries no scientific meaning on its own, and
    COLLECTED-but-not-validated must be re-validated before it can feed any
    dependent stage. Neither is considered done.
    """
    return state in _STAGE_DONE


def needs_revalidation(state: str) -> bool:
    """COMPLETED/COLLECTED stages need collection or re-validation on resume."""
    return state in {"COMPLETED", "COLLECTED"}


def load_task_state(workflow_dir: str | Path) -> TaskState | None:
    path = Path(workflow_dir) / "task_state.json"
    if not path.is_file():
        return None
    return TaskState.model_validate_json(path.read_text(encoding="utf-8"))


def save_task_state(workflow_dir: str | Path, state: TaskState) -> Path:
    state.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path = Path(workflow_dir) / "task_state.json"
    path.write_text(
        json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return path


TASK_STATE_KEYS = ("task_state.json", "workflow.json")


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------


MOLECULAR_RESULT_FILES: dict[str, list[str]] = {
    # OUTCAR is a REQUIRED collection artifact for relax (ionic force
    # criterion) and corrected_static (E0 + runtime-error scan).
    "relax": ["CONTCAR", "OUTCAR", "OSZICAR", "EIGENVAL", "vasprun.xml"],
    "static_preconverge": ["OSZICAR", "EIGENVAL", "vasprun.xml"],
    "corrected_static": ["OUTCAR", "OSZICAR", "EIGENVAL", "vasprun.xml"],
    "orbital_homo": [
        "LOCPOT", "EIGENVAL", "OSZICAR", "PARCHG", "PARCHG.ALL",
    ],
    "orbital_lumo": [
        "LOCPOT", "EIGENVAL", "OSZICAR", "PARCHG", "PARCHG.ALL",
    ],
    "esp": ["LOCPOT", "OSZICAR", "EIGENVAL", "vasprun.xml"],
    "static_hse": ["OSZICAR", "EIGENVAL", "vasprun.xml"],
}


def stage_result_files(stage: StageName) -> list[str]:
    return MOLECULAR_RESULT_FILES.get(stage.value, ["OSZICAR", "EIGENVAL", "vasprun.xml"])


def stage_local_inputs(
    workflow: WorkflowSpec,
    stage: StageSpec,
    workflow_dir: str | Path,
    completed_stages: dict[str, StageTask],
) -> Path:
    """Materialize the stage's local input dir (CONTCAR -> POSCAR etc.)."""
    root = Path(workflow_dir)
    stage_dir = root / f"stage_{stage.name.value}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    if stage.depends_on is not None and "CONTCAR" in stage.required_upstream_outputs:
        # CONTCAR lives in the relax stage's results; walk the depends_on
        # chain and pick the first completed ancestor that actually has it.
        positions = {s.name.value: index for index, s in enumerate(workflow.stages)}
        cursor: StageName | None = stage.depends_on
        while cursor is not None:
            upstream = completed_stages.get(cursor.value)
            if upstream is not None and upstream.results_dir:
                contcar = Path(upstream.results_dir) / "CONTCAR"
                if contcar.is_file():
                    shutil.copy2(contcar, stage_dir / "POSCAR")
                    break
            cursor_pos = positions.get(cursor.value)
            if cursor_pos is None:
                break
            cursor = workflow.stages[cursor_pos].depends_on
        else:
            raise RuntimeError(
                f"stage {stage.name.value} needs CONTCAR from the relax "
                "chain, but no completed ancestor provides it"
            )
    return stage_dir


def persist_stage_task(
    workflow_dir: str | Path,
    task_state: TaskState,
    update: dict[str, Any],
) -> TaskState:
    stage = update["stage"]
    entries = {item.stage: item for item in task_state.stages}
    existing = entries.get(stage)
    fields = dict(update)
    if existing is not None:
        for key, value in fields.items():
            setattr(existing, key, value)
        entries[stage] = existing
    else:
        entries[stage] = StageTask(
            **fields,
        )
    task_state.stages = list(entries.values())
    save_task_state(workflow_dir, task_state)
    return task_state


def determine_ibands(
    eigenval_dir: str | Path,
) -> tuple[int | None, int | None]:
    """HOMO/LUMO band indices from a corrected-static EIGENVAL (occupations)."""
    from photomatagent.scientific.applications.vasp.molecular.results import (
        determine_orbital_bands,
        parse_eigenval,
    )

    eigenval = Path(eigenval_dir) / "EIGENVAL"
    if not eigenval.is_file():
        return None, None
    bands = determine_orbital_bands(parse_eigenval(eigenval))
    return bands.homo_band, bands.lumo_band


async def _collect_and_validate_stage(
    *,
    backend: Any,
    session: Any,
    workflow: WorkflowSpec,
    stage: StageSpec,
    entry: StageTask,
    stage_dir: Path,
    results_root: Path,
) -> tuple[StageTask, dict[str, Any], int]:
    """Download (when needed) and validate one completed stage. Never submits.

    ``entry.state`` must be COMPLETED or COLLECTED. COMPLETED entries are
    downloaded from their remote directory; COLLECTED entries are re-validated
    from the results already on disk. The returned entry is VALIDATED only
    when the scientific analysis passes; otherwise it stays COLLECTED with
    the analysis errors, and no evidence is produced.
    """
    from photomatagent.scientific.applications.vasp.molecular.results import (
        analyze_result_dir,
        scientific_evidence,
    )
    from photomatagent.scientific.remote.registry import JobLifecycleState

    name = stage.name
    result_dir = results_root / name.value
    result_dir.mkdir(parents=True, exist_ok=True)
    need_download = (
        entry.state == JobLifecycleState.COMPLETED.value
        or not entry.results_dir
        or not (Path(entry.results_dir) / "OSZICAR").is_file()
    )
    if need_download:
        if not entry.remote_directory:
            return (
                entry.model_copy(
                    update={
                        "state": JobLifecycleState.FAILED.value,
                        "error": (
                            "COMPLETED stage has no remote directory; "
                            "collection refused"
                        ),
                    }
                ),
                {"errors": ["no remote directory recorded for collection"]},
                0,
            )
        try:
            await backend.download_files(
                entry.remote_directory,
                stage_result_files(name),
                result_dir,
            )
            # VASP 5.4.4 writes orbital densities under several names
            # (PARCHG, PARCHG.<band>, PARCHG.0001.<band>...). Discover and
            # fetch every PARCHG* artifact so the orbital file is never lost.
            if stage.name in {StageName.ORBITAL_HOMO, StageName.ORBITAL_LUMO}:
                await _download_parchg_artifacts(
                    backend, entry.remote_directory, result_dir
                )
        except Exception as exc:
            return (
                entry.model_copy(
                    update={
                        "state": JobLifecycleState.FAILED.value,
                        "error": f"collect failed: {exc}",
                    }
                ),
                {"errors": [f"collect failed: {exc}"]},
                0,
            )
    else:
        result_dir = Path(entry.results_dir)
    # Mirror the submitted inputs so EDIFF/NSW/ISPIN and the structure are
    # analyzed from the exact files that ran.
    for input_name in ("INCAR", "KPOINTS", "POSCAR"):
        source_file = stage_dir / input_name
        if source_file.is_file() and not (result_dir / input_name).is_file():
            shutil.copy2(source_file, result_dir / input_name)
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
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    validated = analysis.get("validated") is True
    if entry.request_id:
        session.mark_result_state(
            entry.request_id,
            collected=True,
            validated=validated,
            evidence=len(evidence),
            error="; ".join(analysis.get("errors", [])),
        )
    updated = entry.model_copy(
        update={
            "state": (
                JobLifecycleState.VALIDATED.value
                if validated
                else JobLifecycleState.COLLECTED.value
            ),
            "results_dir": str(result_dir),
            "validated": validated,
            "error": "" if validated else "; ".join(analysis.get("errors", [])),
        }
    )
    return updated, analysis, len(evidence)


async def _download_parchg_artifacts(
    backend: Any, remote_directory: str, result_dir: Path
) -> list[Path]:
    """Download every PARCHG* file present in the remote job directory.

    VASP 5.4.4 writes partial charge densities as ``PARCHG``, ``PARCHG.ALL``,
    ``PARCHG.<band>`` or ``PARCHG.0001.<band>`` depending on LPARD/LSEPB
    settings. Discovery via the backend listing keeps collection robust to
    every naming variant; when the listing is unavailable the common names
    are attempted directly.
    """
    names: list[str] = []
    listing = getattr(backend, "list_remote_artifacts", None)
    if listing is not None:
        try:
            artifacts = await listing(remote_directory)
            names = [
                artifact.name
                for artifact in artifacts
                if artifact.name == "PARCHG"
                or artifact.name.startswith("PARCHG.")
            ]
        except Exception:
            names = []
    if not names:
        names = ["PARCHG", "PARCHG.ALL"]
    return await backend.download_files(remote_directory, names, result_dir)


async def _recover_collected_relax(
    *,
    backend: Any,
    session: Any,
    workflow: WorkflowSpec,
    entry: StageTask,
    stage_dir: Path,
    root: Path,
    task_state: TaskState,
    psp_dir: str | Path | None,
    module_name: str,
    env_script: str,
    remote_psp_dir: str,
    recovery_policy: Any,
    analysis: dict[str, Any],
    wait_timeout_seconds: float,
    provenance: dict[str, Any] | None,
) -> tuple[bool, StageTask, dict[str, Any], int]:
    """One deterministic relax recovery attempt (rule-based, typed policy).

    Only a COLLECTED-but-not-converged relax enters here. The decision table
    decides restart artifact (CONTCAR / XDATCAR best), typed INCAR changes
    and whether to submit at all; every attempt gets a NEW attempt_id and a
    NEW remote directory (submit-once + unique remote dirs are preserved).
    """
    from photomatagent.scientific.applications.vasp.molecular.preflight import (
        preflight_gate,
        run_molecular_preflight,
    )
    from photomatagent.scientific.applications.vasp.molecular.recovery import (
        classify_relax_failure,
        decide_recovery,
        materialize_recovery_stage_dir,
        outcar_force_history,
    )
    from photomatagent.scientific.applications.vasp.molecular.slurm import (
        cleanup_materialized_potcar,
        materialize_stage_potcar,
        potcar_mode_of_stage,
        potcar_symbols_from_stage,
        render_stage_slurm,
    )
    from photomatagent.scientific.remote.registry import JobLifecycleState

    convergence = analysis.get("convergence") or {}
    force_history: list[float] = []
    result_dir = Path(entry.results_dir) if entry.results_dir else Path(entry.stage_dir)
    outcar = result_dir / "OUTCAR"
    if outcar.is_file():
        try:
            force_history = outcar_force_history(outcar)
        except Exception:
            force_history = []
    failure = classify_relax_failure(
        convergence=convergence,
        lifecycle_state=entry.state,
        force_history=force_history,
    )
    contcar = result_dir / "CONTCAR"
    decision = decide_recovery(
        recovery_policy,
        failure=failure,
        attempts_used=entry.retry_count,
        has_contcar=contcar.is_file(),
        max_force=convergence.get("max_force_ev_ang"),
        ediffg=convergence.get("ediffg_ev_ang"),
    )
    if decision.action != "RESUBMIT":
        updated = entry.model_copy(
            update={
                "error": (
                    (entry.error + "; " if entry.error else "")
                    + f"recovery {decision.action}: {decision.reason}"
                )
            }
        )
        task_state = persist_stage_task(
            root, task_state,
            {"stage": "relax", "error": updated.error},
        )
        return False, updated, analysis, 0

    restart_source = contcar if decision.restart_from == "CONTCAR" else (
        result_dir / "XDATCAR_BEST"
    )
    restart_dir = materialize_recovery_stage_dir(
        previous_stage_dir=stage_dir,
        restart_structure=str(restart_source),
        incar_changes=decision.incar_changes,
        attempt_id=decision.new_attempt_id,
        workflow_dir=str(root),
        reason=decision.reason,
        practical_convergence=decision.practical_convergence,
        practical_convergence_note=decision.practical_convergence_note,
    )
    # Re-run the deterministic gate over the FULL workflow, pointing only the
    # relax stage at the restart directory (typed changes are re-checked).
    inputs_root = root / "inputs"
    stage_dirs = {
        stage.name: inputs_root / f"{index + 1:02d}_{stage.name.value}"
        for index, stage in enumerate(workflow.stages)
    }
    stage_dirs[StageName.RELAX] = restart_dir
    report = run_molecular_preflight(
        workflow, psp_dir=psp_dir, stage_dirs=stage_dirs
    )
    gate = preflight_gate(report, report_path=str(root / "preflight.json"))
    stage_spec = workflow.stages[0]
    potcar_mode = potcar_mode_of_stage(
        restart_dir, remote_psp_dir=remote_psp_dir, psp_dir=psp_dir
    )
    if potcar_mode == "none":
        return False, entry, analysis, 0
    potcar_materialized = False
    if potcar_mode == "local" and not (restart_dir / "POTCAR").is_file():
        try:
            potcar_materialized = materialize_stage_potcar(
                restart_dir, psp_dir, potcar_symbols_from_stage(restart_dir)
            )
        except Exception as exc:
            return False, entry, analysis, 0
    try:
        submit = await session.submit_once(
            application="vasp_molecular",
            workflow_stage="relax",
            job_name=(
                f"{workflow.molecule.name}-relax-"
                f"{decision.new_attempt_id}"
            ),
            local_input_dir=restart_dir,
            gate=gate,
            resource=sanitize_resource(workflow, stage_spec),
            executable="vasp_std",
            script_name="run.slurm",
            provenance={
                "workflow_dir": str(root),
                "molecule": workflow.molecule.name,
                "recovery": decision.provenance,
                "recovery_reason": decision.reason,
                "recovery_attempt_id": decision.new_attempt_id,
                "recovery_restart_from": decision.restart_from,
                **dict(provenance or {}),
            },
            remote_copies=[],
            script_renderer=lambda job_name, resource: render_stage_slurm(
                job_name=job_name,
                resource=resource,
                stage_dir=restart_dir,
                module_name=module_name,
                env_script=env_script,
                remote_psp_dir=remote_psp_dir,
            ),
            potcar_mode=potcar_mode,
            potcar_symbols=potcar_symbols_from_stage(restart_dir),
            remote_psp_dir=remote_psp_dir,
        )
    finally:
        cleanup_materialized_potcar(
            restart_dir, materialized=potcar_materialized
        )
    if submit.blocked or submit.needs_reconciliation or not submit.submitted:
        return False, entry, analysis, 0
    terminal = await wait_for_terminal_state(
        session, submit.request_id, timeout_seconds=wait_timeout_seconds
    )
    if terminal != JobLifecycleState.COMPLETED.value:
        return False, entry, analysis, 0
    synthetic = StageTask(
        stage="relax",
        state=JobLifecycleState.COMPLETED.value,
        request_id=submit.request_id,
        job_id=submit.record.get("job_id") or "",
        remote_directory=submit.record.get("remote_directory") or "",
        stage_dir=str(restart_dir),
        results_dir="",
    )
    updated2, analysis2, evidence_count = await _collect_and_validate_stage(
        backend=backend,
        session=session,
        workflow=workflow,
        stage=stage_spec,
        entry=synthetic,
        stage_dir=restart_dir,
        results_root=root / "results",
    )
    if decision.practical_convergence:
        convergence2 = dict(analysis2.get("convergence") or {})
        convergence2["practical_convergence"] = True
        convergence2["practical_convergence_note"] = (
            decision.practical_convergence_note
        )
        analysis2["convergence"] = convergence2
        analysis2["warnings"] = [
            *analysis2.get("warnings", []),
            "PRACTICAL_CONVERGENCE: " + decision.practical_convergence_note,
        ]
        (Path(updated2.results_dir) / "results.json").write_text(
            json.dumps(analysis2, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    attempt_record = {
        "attempt_id": decision.new_attempt_id,
        "failure": decision.failure,
        "restart_from": decision.restart_from,
        "parameter_changes": decision.parameter_changes,
        "reason": decision.reason,
        "request_id": submit.request_id,
        "job_id": submit.record.get("job_id") or "",
        "remote_directory": submit.record.get("remote_directory") or "",
        "validated": updated2.validated,
        "practical_convergence": decision.practical_convergence,
        "practical_convergence_note": decision.practical_convergence_note,
    }
    task_state = persist_stage_task(
        root, task_state,
        {
            "stage": "relax",
            "state": updated2.state,
            "results_dir": updated2.results_dir,
            "remote_directory": updated2.remote_directory,
            "validated": updated2.validated,
            "error": updated2.error,
            "attempt_id": decision.new_attempt_id,
            "retry_count": entry.retry_count + 1,
            "recovery_attempts": [
                *entry.recovery_attempts,
                attempt_record,
            ],
        },
    )
    return True, updated2, analysis2, evidence_count


async def run_molecule_workflow(
    workflow: WorkflowSpec,
    workflow_dir: str | Path,
    *,
    session: Any,
    backend: Any,
    psp_dir: str | Path | None = None,
    module_name: str = "",
    env_script: str = "",
    remote_psp_dir: str = "",
    wait: bool = True,
    collect: bool = True,
    stop_on_failure: bool = True,
    only: list[str] | None = None,
    wait_timeout_seconds: float = 3600.0,
    provenance: dict[str, Any] | None = None,
    recovery_policy: Any | None = None,
) -> dict[str, Any]:
    """Execute the DAG stage by stage; resumes from task_state.json.

    Completed stages are never resubmitted; a stage failure blocks every
    dependent stage. WAVECAR/CHGCAR are staged between remote directories
    (never downloaded locally). Returns a bounded report.
    """
    from photomatagent.scientific.applications.vasp.molecular.generator import (
        MolecularVaspGenerator,
    )
    from photomatagent.scientific.applications.vasp.molecular.preflight import (
        preflight_gate,
        run_molecular_preflight,
    )
    from photomatagent.scientific.applications.vasp.molecular.results import (
        analyze_result_dir,
        scientific_evidence,
    )
    from photomatagent.scientific.remote.lifecycle import RemoteArtifactCopy
    from photomatagent.scientific.remote.registry import JobLifecycleState

    root = Path(workflow_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    inputs_root = root / "inputs"
    results_root = root / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    generator = MolecularVaspGenerator(psp_dir=psp_dir)
    generator.generate(workflow, inputs_root, write_potcar=False)
    preflight_report = run_molecular_preflight(
        workflow, psp_dir=psp_dir, stage_dirs={
            stage.name: inputs_root / f"{i+1:02d}_{stage.name.value}"
            for i, stage in enumerate(workflow.stages)
        },
    )
    gate = preflight_gate(preflight_report, report_path=str(root / "preflight.json"))

    task_state = load_task_state(root) or TaskState(
        workflow_dir=str(root), molecule_name=workflow.molecule.name
    )
    stage_tasks = task_state.stage_map()
    requested = set(only or [stage.name.value for stage in workflow.stages])
    completed = {item.stage: item for item in task_state.stages if stage_done(item.state)}
    report: dict[str, Any] = {
        "workflow_dir": str(root),
        "preflight_passed": preflight_report.passed,
        "stages": [],
        "resumed": [],
        "blocked": [],
        "evidence_count": 0,
    }
    if not preflight_report.passed:
        report["error"] = (
            "preflight failed; no submission attempted "
            f"({len(preflight_report.errors)} errors)"
        )
        return report

    for stage in workflow.stages:
        name = stage.name.value
        entry = stage_tasks.get(name)
        if name not in requested:
            continue
        if entry is not None and stage_done(entry.state):
            report["resumed"].append(name)
            report["stages"].append(
                {
                    "stage": name,
                    "state": JobLifecycleState.VALIDATED.value,
                    "resumed": True,
                    "validated": entry.validated,
                }
            )
            continue
        if entry is not None and needs_revalidation(entry.state):
            # Scheduler COMPLETED or COLLECTED: resume means collect and
            # validate, never resubmit. A COMPLETED job may already exist
            # remotely; the registry owns its request_id and submit-once
            # guarantees no second job.
            generated_dir = (
                inputs_root / f"{workflow.stages.index(stage) + 1:02d}_{name}"
            )
            stage_dir = (
                Path(entry.stage_dir)
                if entry.stage_dir
                else generated_dir
            )
            stage_dir.mkdir(parents=True, exist_ok=True)
            for filename in (
                "POSCAR", "INCAR", "KPOINTS",
                "POTCAR.meta", "POTCAR.policy",
            ):
                source_file = generated_dir / filename
                if source_file.is_file() and not (stage_dir / filename).exists():
                    shutil.copy2(source_file, stage_dir / filename)
            updated, analysis, evidence_count = await _collect_and_validate_stage(
                backend=backend,
                session=session,
                workflow=workflow,
                stage=stage,
                entry=task_state.stage_map().get(name) or entry,
                stage_dir=stage_dir,
                results_root=results_root,
            )
            task_state = persist_stage_task(
                root, task_state,
                {
                    "stage": name,
                    "state": updated.state,
                    "results_dir": updated.results_dir,
                    "validated": updated.validated,
                    "error": updated.error,
                },
            )
            report["evidence_count"] += evidence_count
            report["resumed"].append(name)
            report["stages"].append(
                {
                    "stage": name,
                    "state": updated.state,
                    "resumed": True,
                    "validated": updated.validated,
                    "evidence": evidence_count,
                    "errors": analysis.get("errors", []),
                }
            )
            if updated.state != JobLifecycleState.VALIDATED.value:
                if (
                    recovery_policy is not None
                    and stage.name is StageName.RELAX
                ):
                    recovered, updated, analysis, evidence_count = (
                        await _recover_collected_relax(
                            backend=backend,
                            session=session,
                            workflow=workflow,
                            entry=task_state.stage_map().get(name) or entry,
                            stage_dir=stage_dir,
                            root=root,
                            task_state=task_state,
                            psp_dir=psp_dir,
                            module_name=module_name,
                            env_script=env_script,
                            remote_psp_dir=remote_psp_dir,
                            recovery_policy=recovery_policy,
                            analysis=analysis,
                            wait_timeout_seconds=wait_timeout_seconds,
                            provenance=provenance,
                        )
                    )
                    if recovered:
                        report["evidence_count"] += evidence_count
                        report["stages"].append(
                            {
                                "stage": "relax",
                                "state": updated.state,
                                "recovered": True,
                                "validated": updated.validated,
                                "evidence": evidence_count,
                                "recovery_attempts": (
                                    task_state.stage_map().get("relax") or updated
                                ).recovery_attempts,
                                "errors": analysis.get("errors", []),
                            }
                        )
                        continue
                report["blocked"] = [
                    s.name.value
                    for s in workflow.stages
                    if workflow.stages.index(s) > workflow.stages.index(stage)
                ]
                break
            continue
        try:
            stage_dir = stage_local_inputs(
                workflow, stage, root, {item.stage: item for item in task_state.stages}
            )
        except Exception as exc:
            persist_stage_task(
                root, task_state,
                {"stage": name, "state": "FAILED", "error": str(exc)},
            )
            report["blocked"].append(name)
            report["stages"].append(
                {"stage": name, "state": "FAILED", "error": str(exc)}
            )
            break

        # Copy the generated inputs (POSCAR/INCAR/KPOINTS) into the stage dir,
        # then apply stage-specific substitutions (CONTCAR already handled by
        # stage_local_inputs for dependency stages).
        generated_dir = inputs_root / f"{workflow.stages.index(stage) + 1:02d}_{name}"
        for filename in ("POSCAR", "INCAR", "KPOINTS", "POTCAR.meta", "POTCAR.policy"):
            source_file = generated_dir / filename
            if source_file.is_file() and not (stage_dir / filename).exists():
                shutil.copy2(source_file, stage_dir / filename)

        if stage.name in {StageName.ORBITAL_HOMO, StageName.ORBITAL_LUMO}:
            # Read from the LIVE task state: corrected_static completed in
            # this very run (stage_tasks was captured before the loop).
            static_task = task_state.stage_map().get(
                StageName.CORRECTED_STATIC.value
            )
            homo_band, lumo_band = None, None
            if static_task is not None and static_task.results_dir:
                homo_band, lumo_band = determine_ibands(static_task.results_dir)
            iband = homo_band if stage.name is StageName.ORBITAL_HOMO else lumo_band
            if iband is None:
                persist_stage_task(
                    root, task_state,
                    {
                        "stage": name,
                        "state": "FAILED",
                        "error": (
                            "IBAND could not be determined: corrected_static "
                            "EIGENVAL missing or unreadable"
                        ),
                    },
                )
                report["blocked"].append(name)
                report["stages"].append(
                    {"stage": name, "state": "FAILED", "error": "IBAND undetermined"}
                )
                break
            incar_path = stage_dir / "INCAR"
            lines = [
                line if not line.startswith("IBAND") else f"IBAND = {iband}"
                for line in incar_path.read_text(encoding="utf-8").splitlines()
            ]
            incar_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        remote_copies: list[RemoteArtifactCopy] = []
        if stage.depends_on is not None:
            upstream = stage_tasks.get(stage.depends_on.value)
            if upstream is not None and upstream.remote_directory:
                for filename in stage.required_upstream_outputs:
                    if filename in {"WAVECAR", "CHGCAR"}:
                        remote_copies.append(
                            RemoteArtifactCopy(
                                source_remote_directory=upstream.remote_directory,
                                filename=filename,
                            )
                        )

        task_state = persist_stage_task(
            root, task_state,
            {
                "stage": name,
                "state": JobLifecycleState.PREFLIGHT_PASSED.value,
                "stage_dir": str(stage_dir),
                "error": "",
            },
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
            remote_psp_dir=remote_psp_dir,
            psp_dir=psp_dir,
        )
        if potcar_mode == "none":
            task_state = persist_stage_task(
                root, task_state,
                {
                    "stage": name,
                    "state": JobLifecycleState.FAILED.value,
                    "error": (
                        "no POTCAR strategy: materialize POTCAR locally or "
                        "configure SCNET_VASP_PSP_DIR"
                    ),
                },
            )
            report["blocked"].append(name)
            report["stages"].append(
                {"stage": name, "state": "FAILED",
                "error": "no POTCAR strategy; submission refused"}
            )
            break
        # Local POTCAR strategy without a curated file: assemble from the
        # resolved local PAW-PBE library in POSCAR order, upload to the
        # unique remote job directory, then remove the local bytes again.
        potcar_materialized = (
            potcar_mode == "local"
            and not (stage_dir / "POTCAR").is_file()
        )
        if potcar_materialized:
            try:
                potcar_materialized = materialize_stage_potcar(
                    stage_dir, psp_dir, potcar_symbols_from_stage(stage_dir)
                )
            except Exception as exc:
                task_state = persist_stage_task(
                    root, task_state,
                    {
                        "stage": name,
                        "state": JobLifecycleState.FAILED.value,
                        "error": f"local POTCAR assembly failed: {exc}",
                    },
                )
                report["blocked"].append(name)
                report["stages"].append(
                    {"stage": name, "state": "FAILED",
                     "error": f"local POTCAR assembly failed: {exc}"}
                )
                break
        try:
            submit = await session.submit_once(
                application="vasp_molecular",
                workflow_stage=name,
                job_name=f"{workflow.molecule.name}-{name}",
                local_input_dir=stage_dir,
                gate=gate,
                resource=sanitize_resource(workflow, stage),
                executable="vasp_std",
                script_name="run.slurm",
                provenance={
                    "workflow_dir": str(root),
                    "molecule": workflow.molecule.name,
                    **dict(provenance or {}),
                },
                remote_copies=remote_copies,
                script_renderer=lambda job_name, resource: render_stage_slurm(
                    job_name=job_name,
                    resource=resource,
                    stage_dir=stage_dir,
                    module_name=module_name,
                    env_script=env_script,
                    remote_psp_dir=remote_psp_dir,
                ),
                potcar_mode=potcar_mode,
                potcar_symbols=potcar_symbols_from_stage(stage_dir),
                remote_psp_dir=remote_psp_dir,
            )
        finally:
            cleanup_materialized_potcar(
                stage_dir, materialized=potcar_materialized
            )
        task_state = persist_stage_task(
            root, task_state,
            {
                "stage": name,
                "state": submit.record.get("state", "PREPARED"),
                "request_id": submit.request_id,
                "job_id": submit.record.get("job_id") or "",
                "remote_directory": submit.record.get("remote_directory") or "",
                "error": submit.error,
            },
        )
        if submit.blocked or submit.needs_reconciliation or not submit.submitted:
            report["stages"].append(
                {
                    "stage": name,
                    "state": submit.record.get("state"),
                    "error": submit.error,
                    "needs_reconciliation": submit.needs_reconciliation,
                }
            )
            if stop_on_failure:
                report["blocked"] = [
                    s.name.value
                    for s in workflow.stages
                    if workflow.stages.index(s) > workflow.stages.index(stage)
                ]
                break
            continue

        state = submit.record.get("state")
        if wait:
            terminal = await wait_for_terminal_state(
                session, submit.request_id, timeout_seconds=wait_timeout_seconds
            )
            state = terminal
            task_state = persist_stage_task(
                root, task_state, {"stage": name, "state": terminal}
            )
        if state != JobLifecycleState.COMPLETED.value:
            report["stages"].append(
                {
                    "stage": name,
                    "state": state,
                    "error": "scheduler state is not COMPLETED; no collection",
                }
            )
            if stop_on_failure:
                report["blocked"] = [
                    s.name.value
                    for s in workflow.stages
                    if workflow.stages.index(s) > workflow.stages.index(stage)
                ]
                break
            continue

        if not collect:
            report["stages"].append({"stage": name, "state": "COMPLETED", "collected": False})
            continue
        synthetic = StageTask(
            stage=name,
            state=JobLifecycleState.COMPLETED.value,
            request_id=submit.request_id,
            job_id=submit.record.get("job_id") or "",
            remote_directory=submit.record.get("remote_directory") or "",
            stage_dir=str(stage_dir),
            results_dir="",
        )
        updated, analysis, evidence_count = await _collect_and_validate_stage(
            backend=backend,
            session=session,
            workflow=workflow,
            stage=stage,
            entry=synthetic,
            stage_dir=stage_dir,
            results_root=results_root,
        )
        task_state = persist_stage_task(
            root, task_state,
            {
                "stage": name,
                "state": updated.state,
                "results_dir": updated.results_dir,
                "validated": updated.validated,
                "error": updated.error,
            },
        )
        report["evidence_count"] += evidence_count
        report["stages"].append(
            {
                "stage": name,
                "state": updated.state,
                "validated": updated.validated,
                "evidence": evidence_count,
                "errors": analysis.get("errors", []),
            }
        )
        if updated.state != JobLifecycleState.VALIDATED.value:
            # Validation failed: the stage is collected but not valid; every
            # dependent stage is blocked (no scientific dependency is ever
            # satisfied by a merely-collected result).
            if (
                recovery_policy is not None
                and stage.name is StageName.RELAX
            ):
                recovered, updated, analysis, evidence_count = (
                    await _recover_collected_relax(
                        backend=backend,
                        session=session,
                        workflow=workflow,
                        entry=task_state.stage_map().get(name) or synthetic,
                        stage_dir=stage_dir,
                        root=root,
                        task_state=task_state,
                        psp_dir=psp_dir,
                        module_name=module_name,
                        env_script=env_script,
                        remote_psp_dir=remote_psp_dir,
                        recovery_policy=recovery_policy,
                        analysis=analysis,
                        wait_timeout_seconds=wait_timeout_seconds,
                        provenance=provenance,
                    )
                )
                if recovered:
                    report["evidence_count"] += evidence_count
                    report["stages"].append(
                        {
                            "stage": "relax",
                            "state": updated.state,
                            "recovered": True,
                            "validated": updated.validated,
                            "evidence": evidence_count,
                            "recovery_attempts": (
                                task_state.stage_map().get("relax") or updated
                            ).recovery_attempts,
                            "errors": analysis.get("errors", []),
                        }
                    )
                    continue
            if stop_on_failure:
                report["blocked"] = [
                    s.name.value
                    for s in workflow.stages
                    if workflow.stages.index(s) > workflow.stages.index(stage)
                ]
                break
            continue
    report["completed"] = task_state.completed()
    report["task_state"] = str(root / "task_state.json")
    return report


async def wait_for_terminal_state(
    session: Any, request_id: str, *, timeout_seconds: float
) -> str:
    """Poll the registry until a terminal lifecycle state or timeout."""
    import asyncio
    import time

    from photomatagent.scientific.remote.registry import JobLifecycleState

    deadline = time.monotonic() + timeout_seconds
    while True:
        # Push the scheduler state into the registry first: the record only
        # advances when a status refresh succeeds.
        await session.refresh_status(request_id)
        record = session.registry.get(request_id)
        if record is not None and record.state.terminal:
            return record.state.value
        if time.monotonic() >= deadline:
            return "TIMEOUT"
        await asyncio.sleep(2.0)


def sanitize_resource(workflow: WorkflowSpec, stage: StageSpec) -> Any:
    """Map one stage onto the workflow's resource plan (capped)."""
    from photomatagent.scientific.remote.models import ResourceRequest

    plan = workflow.resource_plan
    return ResourceRequest(
        partition=workflow.resource_ceiling.partition,
        nodes=1,
        tasks_per_node=plan.tasks_per_node,
        walltime_minutes=plan.walltime_minutes,
    )
