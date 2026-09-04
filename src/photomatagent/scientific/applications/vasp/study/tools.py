"""Hidden migration-only ``vasp_study.*`` helpers: plan / execute / status /
resume / collect / report.

The study layer is a thin orchestrator over the existing molecular executor
plus the generic ``chemistry`` package. Natural-language input is never
parsed by an embedded LLM. These helpers are retained only for direct Python
migration callers; the outer agent uses the unified ``vasp.*`` service.
"""

# Migration-only module: legacy vasp_study.* Tool classes are not
# registered in the model-visible ToolRegistry. New public entry points
# use the unified vasp.* service.


from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from photomatagent.scientific.applications.vasp.molecular.runtime import (
    MolecularVaspRuntime,
    default_molecular_runtime,
)
from photomatagent.scientific.applications.vasp.study.executor import (
    StudyExecutor,
)
from photomatagent.scientific.applications.vasp.study.models import (
    CalculationMatrix,
    ExecutionPolicy,
    MethodSpec,
    PropertyRequest,
    ResourceBudget,
    StructurePolicy,
    StudySystem,
    VaspStudyRequest,
)
from photomatagent.scientific.applications.vasp.study.planner import (
    budget_status,
    load_planned_study,
    plan_study,
)
from photomatagent.scientific.capabilities.base import (
    CapabilityPack,
    CapabilityStatus,
    ProbeResult,
)
from photomatagent.scientific.capabilities.contracts import ScientificToolResult
from photomatagent.tools.base import Tool
from photomatagent.tools.exposure import ToolExposure

MAX_TOOL_CHARS = 4000


def bounded_payload(
    *,
    ok: bool,
    summary: dict[str, Any],
    errors: list[str],
    warnings: list[str] | None = None,
    artifacts: list[str] | None = None,
    note: str = "",
    **extra: Any,
) -> dict[str, Any]:
    """Study-tool bounded payload (<=4000 chars), with a matrix-safe trim."""
    payload: dict[str, Any] = {
        "ok": ok,
        "summary": summary,
        "errors": errors[:10],
        "warnings": (warnings or [])[:10],
        "artifacts": (artifacts or [])[:12],
        "chars": 0,
        "note": note,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    payload.update(extra)
    payload["chars"] = len(json.dumps(payload, ensure_ascii=False))
    if payload["chars"] > MAX_TOOL_CHARS:
        # Compact the big list fields first (matrix rows survive capped).
        def _trim(value: Any, cap: int = 10) -> Any:
            if isinstance(value, list):
                if len(value) > cap:
                    return [*value[:cap], "..."]
                return value
            if isinstance(value, dict):
                return {
                    key: _trim(item, cap) for key, item in value.items()
                }
            return value

        payload["summary"] = _trim(payload["summary"], cap=10)
        payload["errors"] = payload["errors"][:6]
        payload["warnings"] = payload["warnings"][:6]
        payload["artifacts"] = payload["artifacts"][:8]
        payload["chars"] = len(json.dumps(payload, ensure_ascii=False))
        if payload["chars"] > MAX_TOOL_CHARS:
            # Last resort: scalars only.
            payload["summary"] = {
                key: value for key, value in payload["summary"].items()
                if not isinstance(value, (dict, list))
            }
            payload["chars"] = len(json.dumps(payload, ensure_ascii=False))
        payload["note"] = (payload["note"] + " [output trimmed]").strip()
    return payload


def _result(payload: dict[str, Any]) -> ScientificToolResult:
    return ScientificToolResult(
        output=json.dumps(payload, ensure_ascii=False, indent=2),
        data=payload,
        is_error=not bool(payload.get("ok", True)),
    )


def _study_dir_argument(
    arguments: dict[str, Any], runtime: MolecularVaspRuntime
) -> Path | None:
    raw = arguments.get("study_dir")
    if raw:
        return Path(raw).expanduser().resolve()
    study_id = arguments.get("study_id")
    if study_id:
        workflow_dir = getattr(runtime, "workflow_dir", None)
        if workflow_dir is not None:
            root = Path(workflow_dir)
            workspace = (
                root.parents[1]
                if root.name == "vasp_molecule"
                else root.parent
            )
        else:
            workspace = Path.cwd()
        return workspace / "output" / "vasp_study" / str(study_id)
    return None


class _StudyTool(Tool):
    """Shared defaults for the vasp_study tool family."""

    namespace = "vasp_study"
    source = "photomatagent study orchestration"
    # Migration-only compatibility helpers are never model-visible.
    exposure = ToolExposure.HIDDEN
    tags = ("vasp", "study", "molecule", "dft")
    cost_class = "EXPENSIVE"

    def __init__(self, runtime: MolecularVaspRuntime | None = None) -> None:
        self.runtime = runtime or default_molecular_runtime()

    def _executor(self, study_dir: Path) -> StudyExecutor:
        spec = load_planned_study(study_dir)
        return StudyExecutor(spec, self.runtime)

    def _matrix_payload(
        self, matrix: CalculationMatrix, *, budget: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "unique_tasks": len(matrix.tasks),
            "total_jobs": matrix.total_jobs,
            "total_core_hours": matrix.total_core_hours,
            "estimated_disk_gb": matrix.estimated_disk_gb,
            "budget": budget,
            "tasks": [
                (
                    f"{task.task_id}|{task.display_name}|q{task.total_charge:+d}"
                    f"|rel{task.reliability}|{task.state}"
                    f"|{','.join(str(getattr(item, 'value', item)) for item in task.assists)}"
                )
                for task in matrix.tasks
            ],
            "binding_groups": [
                (
                    f"{group.complex_task_id}+{'+'.join(group.fragment_task_ids)}"
                    f"|q{group.total_charge:+d}|{group.state}"
                )
                for group in matrix.binding_groups
            ],
        }


class VaspStudyPlanTool(_StudyTool):
    name = "vasp_study.plan"
    description = (
        "Compile a full typed study plan from the outer agent's structured "
        "parameters (original natural-language request, systems, requested "
        "properties, method, structure policy, resource budget): resolve or "
        "generate every structure with recorded provenance, build the "
        "deduplicated calculation matrix (charges explicit, fragments "
        "expanded, shared references deduplicated), estimate jobs/core-hours/"
        "disk and persist study_request.json + structure_manifest.json + "
        "calculation_matrix.json. Never submits."
    )
    short_description = "Compile a VASP study plan (offline)."
    cost_class = "MODERATE"
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "original_request": {"type": "string"},
            "study_id": {"type": "string"},
            "systems": {
                "type": "array",
                "items": {"type": "object"},
            },
            "property_requests": {
                "type": "array",
                "items": {"type": "string"},
            },
            "allow_assumed_structures": {"type": "boolean"},
            "max_candidates_per_system": {"type": "integer"},
            "user_requested_computation": {"type": "boolean"},
            "max_core_hours": {"type": "number"},
            "functional": {"type": "string"},
            "encut_ev": {"type": "number"},
            "box_ang": {"type": "number"},
            "seed": {"type": "integer"},
            "workspace": {"type": "string"},
        },
        "required": ["systems"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        raw_systems = arguments.get("systems") or []
        systems: list[StudySystem] = []
        for raw in raw_systems:
            properties: list[PropertyRequest] = []
            for value in raw.get("properties", []):
                try:
                    properties.append(PropertyRequest(str(value)))
                except ValueError:
                    continue
            systems.append(
                StudySystem(
                    system_id=str(raw["system_id"]),
                    display_name=str(raw.get("display_name") or ""),
                    aliases=list(raw.get("aliases") or []),
                    smiles=raw.get("smiles"),
                    structure_path=(
                        Path(raw["structure_path"])
                        if raw.get("structure_path")
                        else None
                    ),
                    total_charge=(
                        int(raw["total_charge"])
                        if raw.get("total_charge") is not None
                        else None
                    ),
                    spin_multiplicity=int(raw.get("spin_multiplicity", 1)),
                    role=str(raw.get("role") or "molecule"),
                    properties=properties,
                )
            )
        property_requests = [
            PropertyRequest(str(value))
            for value in (arguments.get("property_requests") or [])
            if str(value) in {item.value for item in PropertyRequest}
        ]
        request = VaspStudyRequest(
            study_id=str(arguments.get("study_id") or ""),
            original_request=str(arguments.get("original_request") or ""),
            systems=systems,
            property_requests=property_requests,
            structure_policy=StructurePolicy(
                allow_assumed_structures=bool(
                    arguments.get("allow_assumed_structures", True)
                ),
                max_candidates_per_system=int(
                    arguments.get("max_candidates_per_system", 3)
                ),
                seed=int(arguments.get("seed", 20260825)),
            ),
            execution_policy=ExecutionPolicy(
                user_requested_computation=bool(
                    arguments.get("user_requested_computation", False)
                ),
            ),
            resource_budget=ResourceBudget(
                max_core_hours=float(arguments.get("max_core_hours", 64.0))
            ),
            method=MethodSpec(
                functional=str(arguments.get("functional", "PBE-D3(BJ)")),
                encut_ev=(
                    float(arguments["encut_ev"])
                    if arguments.get("encut_ev") is not None
                    else None
                ),
                box_ang=float(arguments.get("box_ang", 20.0)),
            ),
        )
        workspace = Path(arguments.get("workspace") or Path.cwd()).expanduser()
        try:
            spec = plan_study(request, workspace)
        except Exception as exc:
            return _result(
                bounded_payload(
                    ok=False,
                    summary={},
                    errors=[f"plan failed: {type(exc).__name__}: {exc}"],
                )
            )
        payload = bounded_payload(
            ok=True,
            summary={
                "study_id": spec.study_id,
                "study_dir": str(spec.study_dir),
                "matrix": self._matrix_payload(
                    spec.calculation_matrix,
                    budget=budget_status(spec),
                ),
                "structures": [
                    {
                        "system": structure.identity.system_id,
                        "formula": structure.identity.formula,
                        "charge": structure.identity.total_charge,
                        "reliability": structure.reliability_grade().value,
                        "status": structure.provenance.status.value,
                    }
                    for structure in _manifest_structures(spec)
                ],
            },
            errors=[],
            warnings=spec.calculation_matrix.notes,
            artifacts=[
                str(spec.study_dir / "study_request.json"),
                str(spec.study_dir / "structure_manifest.json"),
                str(spec.study_dir / "calculation_matrix.json"),
            ],
            note=(
                "plan only: nothing submitted. Run vasp_study.execute with "
                "user_requested_computation=True once authorized."
            ),
        )
        return _result(payload)


def _manifest_structures(spec: Any) -> list[Any]:
    from photomatagent.scientific.capabilities.chemistry.models import (
        ChemicalIdentity,
        GeneratedStructure,
        StructureProvenance,
    )

    manifest = spec.structure_manifest_path
    if not manifest.is_file():
        return []
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    structures: list[GeneratedStructure] = []
    for row in payload.get("structures", []):
        provenance = StructureProvenance(**row.get("provenance", {}))
        identity = ChemicalIdentity(
            system_id=row["system_id"],
            display_name=row["display_name"],
            formula=row.get("formula", ""),
            total_charge=row["total_charge"],
            spin_multiplicity=row.get("spin_multiplicity", 1),
            role=row.get("role", "molecule"),
        )
        structures.append(
            GeneratedStructure(
                identity=identity,
                structure_path=Path(row["structure_path"]),
                format=row.get("format", "xyz"),
                atom_count=row.get("atom_count", 0),
                formal_charge=row.get("formal_charge", 0),
                provenance=provenance,
            )
        )
    return structures


class VaspStudyExecuteTool(_StudyTool):
    name = "vasp_study.execute"
    description = (
        "Execute (or resume) a planned study. Every unique calculation runs "
        "through the existing vasp_molecule.* prepare/preflight/submit/"
        "collect machinery with submit-once semantics; VALIDATED tasks are "
        "never resubmitted, COMPLETED tasks are collected and validated, "
        "binding energies are computed only after all required energies are "
        "VALIDATED. Requires study-level authorization "
        "(user_requested_computation=True) plus the backend HPC-submit "
        "policy."
    )
    short_description = "Run/resume a planned VASP study."
    cost_class = "EXPENSIVE"
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "study_id": {"type": "string"},
            "study_dir": {"type": "string"},
            "user_requested_computation": {"type": "boolean"},
            "wait": {"type": "boolean"},
        },
        "required": ["study_id"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        study_dir = _study_dir_argument(arguments, self.runtime)
        if study_dir is None or not (study_dir / "study_request.json").is_file():
            return _result(
                bounded_payload(
                    ok=False,
                    summary={},
                    errors=[
                        "study not found; run vasp_study.plan first "
                        "(or pass study_dir)"
                    ],
                )
            )
        spec = load_planned_study(study_dir)
        spec.request.execution_policy.user_requested_computation = bool(
            arguments.get("user_requested_computation", False)
        )
        if "wait" in arguments:
            spec.request.execution_policy.wait = bool(arguments["wait"])
        executor = StudyExecutor(spec, self.runtime)
        report = await executor.execute()
        payload = bounded_payload(
            ok=not report["failed"] or not report["authorized"],
            summary={
                "study_id": report["study_id"],
                "authorized": report["authorized"],
                "submitted": report["submitted"],
                "resumed": report["resumed"],
                "skipped": report["skipped"],
                "failed": report["failed"],
                "budget": report["budget"],
                "binding_groups": [
                    {
                        "complex": group.complex_task_id,
                        "state": group.state,
                        "delta_e_ev": group.delta_e_ev,
                    }
                    for group in spec.calculation_matrix.binding_groups
                ],
            },
            errors=[
                item["reason"]
                for item in report["failed"]
                if isinstance(item, dict)
            ],
            artifacts=[str(executor.state_path)],
            note=(
                "study state persisted; resume with vasp_study.resume "
                "anytime"
            ),
        )
        return _result(payload)


class VaspStudyStatusTool(_StudyTool):
    name = "vasp_study.status"
    description = "Read a study's persisted task and binding states."
    short_description = "Study task states (read-only)."
    cost_class = "CHEAP"
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "study_id": {"type": "string"},
            "study_dir": {"type": "string"},
        },
        "required": ["study_id"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        study_dir = _study_dir_argument(arguments, self.runtime)
        if study_dir is None or not (study_dir / "study_request.json").is_file():
            return _result(
                bounded_payload(ok=False, summary={}, errors=["study not found"])
            )
        executor = self._executor(study_dir)
        status = await executor.status()
        return _result(
            bounded_payload(
                ok=True,
                summary=status,
                errors=[],
                note="scheduler queries are never mistaken for job failure",
            )
        )


class VaspStudyResumeTool(_StudyTool):
    name = "vasp_study.resume"
    description = (
        "Resume a study after an interruption: VALIDATED tasks stay done, "
        "COMPLETED tasks are collected and validated (never resubmitted), "
        "COLLECTED tasks re-validate from disk, failed conformers fall back "
        "to pre-generated candidates. Uses the molecular resume semantics."
    )
    short_description = "Resume a study workflow."
    cost_class = "EXPENSIVE"
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "study_id": {"type": "string"},
            "study_dir": {"type": "string"},
            "user_requested_computation": {"type": "boolean"},
        },
        "required": ["study_id"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        study_dir = _study_dir_argument(arguments, self.runtime)
        if study_dir is None or not (study_dir / "study_request.json").is_file():
            return _result(
                bounded_payload(ok=False, summary={}, errors=["study not found"])
            )
        spec = load_planned_study(study_dir)
        spec.request.execution_policy.user_requested_computation = bool(
            arguments.get("user_requested_computation", False)
        )
        executor = StudyExecutor(spec, self.runtime)
        report = await executor.execute()
        return _result(
            bounded_payload(
                ok=True,
                summary={
                    "study_id": report["study_id"],
                    "attempted": report["submitted"],
                    "resumed": report["resumed"],
                    "skipped": report["skipped"],
                    "failed": report["failed"],
                    "study_state_path": str(executor.state_path),
                },
                errors=[],
                note=(
                    "resume ran the same submit-once executor; no job was "
                    "created twice"
                ),
            )
        )


class VaspStudyCollectTool(_StudyTool):
    name = "vasp_study.collect"
    description = (
        "Collect and validate every scheduler-COMPLETED task and re-validate "
        "COLLECTED tasks through the molecular executor; recompute binding "
        "energies once all required energies are VALIDATED."
    )
    short_description = "Collect+validate study results."
    cost_class = "EXPENSIVE"
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "study_id": {"type": "string"},
            "study_dir": {"type": "string"},
        },
        "required": ["study_id"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        study_dir = _study_dir_argument(arguments, self.runtime)
        if study_dir is None or not (study_dir / "study_request.json").is_file():
            return _result(
                bounded_payload(ok=False, summary={}, errors=["study not found"])
            )
        executor = self._executor(study_dir)
        report = await executor.collect()
        rows = report.get("collected", [])
        return _result(
            bounded_payload(
                ok=True,
                summary={
                    "study_id": report["study_id"],
                    "collected": rows,
                },
                errors=[],
                note=(
                    "only VALIDATED analyses produce scientific evidence; "
                    "scheduler COMPLETED alone is not evidence"
                ),
            )
        )


class VaspStudyReportTool(_StudyTool):
    name = "vasp_study.report"
    description = (
        "Generate the final study artifacts: results.json, results.csv, "
        "figures (vacuum-aligned HOMO/LUMO levels, PARCHG isosurfaces with "
        "the molecular skeleton, ESP surface maps with colorbar, binding "
        "chart) and report.md with all 14 mandated sections including every "
        "structure assumption and reliability grade."
    )
    short_description = "Final study results, figures and report."
    cost_class = "MODERATE"
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "study_id": {"type": "string"},
            "study_dir": {"type": "string"},
        },
        "required": ["study_id"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        from photomatagent.scientific.applications.vasp.study.analyze import (
            analyze_study,
            write_results_csv,
        )
        from photomatagent.scientific.applications.vasp.study.plotting import (
            plot_binding_energies,
            plot_esp_surface,
            plot_orbital_isosurface,
            plot_orbital_levels,
        )

        study_dir = _study_dir_argument(arguments, self.runtime)
        if study_dir is None or not (study_dir / "study_request.json").is_file():
            return _result(
                bounded_payload(ok=False, summary={}, errors=["study not found"])
            )
        spec = load_planned_study(study_dir)
        executor = self._executor(study_dir)
        state = executor.load_state()
        executor._sync_matrix_from_state(state)
        # ``load_planned_study`` contains the original matrix, before runtime
        # structure selection and child-workflow paths were persisted.  The
        # executor has just merged those durable fields from study_state.json;
        # use that synchronized spec for analysis and figure structure lookup.
        spec = executor.spec
        results = analyze_study(spec)
        figures_dir = study_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        generated_figures: list[str] = []
        orbital_rows = [
            row for row in results["systems"]
            if row.get("homo_aligned_ev") is not None
        ]
        if orbital_rows:
            levels = figures_dir / "orbital_levels.png"
            plot_orbital_levels(orbital_rows, levels)
            generated_figures.append(str(levels))
        for row in results["systems"]:
            workflow_dir = _row_workflow_dir(study_dir, row)
            figure_id = str(
                row.get("system_id") or row["task_id"].split("|", 1)[0]
            )
            structure_path = _row_structure_path(spec, row)
            for stage, tag in (
                ("orbital_homo", "homo"),
                ("orbital_lumo", "lumo"),
            ):
                stage_dir = workflow_dir / "results" / stage
                parchg_files = sorted(stage_dir.glob("PARCHG*"))
                for parchg in parchg_files[:1]:
                    out = figures_dir / f"{tag}_isosurface_{figure_id}.png"
                    try:
                        plot_orbital_isosurface(parchg, structure_path, out)
                        generated_figures.append(str(out))
                    except Exception:
                        pass
            locpot = workflow_dir / "results" / "esp" / "LOCPOT"
            if locpot.is_file():
                out = figures_dir / f"esp_surface_{figure_id}.png"
                try:
                    plot_esp_surface(locpot, structure_path, out)
                    generated_figures.append(str(out))
                except Exception:
                    pass
        if results["binding_energies"]:
            out = figures_dir / "binding_energies.png"
            plot_binding_energies(results["binding_energies"], out)
            generated_figures.append(str(out))
        (study_dir / "results.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        write_results_csv(results, study_dir / "results.csv")
        from photomatagent.scientific.applications.vasp.study.report import (
            render_report,
        )

        report = render_report(spec, results)
        (study_dir / "report.md").write_text(report, encoding="utf-8")
        return _result(
            bounded_payload(
                ok=True,
                summary={
                    "study_id": spec.study_id,
                    "summary": results["summary"],
                    "artifacts": [
                        str(study_dir / "results.json"),
                        str(study_dir / "results.csv"),
                        str(study_dir / "report.md"),
                        *generated_figures,
                    ],
                },
                errors=[],
                warnings=[
                    item
                    for row in results["structure_assumptions"]
                    for item in row["assumptions"]
                ][:10],
                note=(
                    "report includes every structure assumption; C/D-grade "
                    "results carry the mandated hypothesis-model warning"
                ),
            )
        )


def _row_structure_path(spec: Any, row: dict[str, Any]) -> Path:
    for task in spec.calculation_matrix.tasks:
        if task.task_id != row["task_id"]:
            continue
        if task.structure_path:
            return Path(task.structure_path)
        # Backward-compatible recovery for study_state files written before
        # the selected structure path was persisted.  The conformer index is
        # durable, and candidates remain part of the planned matrix.
        if task.structure_candidates:
            index = min(task.conformer_index, len(task.structure_candidates) - 1)
            return Path(task.structure_candidates[index])
    return Path.cwd()


def _row_workflow_dir(study_dir: Path, row: dict[str, Any]) -> Path:
    """Return a persisted child path only when it remains inside the study."""
    candidate = Path(str(row.get("workflow_dir") or "")).expanduser().resolve()
    study_root = study_dir.resolve()
    try:
        candidate.relative_to(study_root)
    except ValueError:
        return study_root / "workflows" / "missing"
    return candidate


class VaspStudyCapabilityPack(CapabilityPack):
    """Hidden migration pack for the retired ``vasp_study.*`` family."""

    name = "vasp_study"
    description = (
        "VASP study orchestration: typed requests into deduplicated "
        "calculation matrices executed through the vasp_molecule.* executor, "
        "with figures and final reports."
    )
    execution_mode = "mcp/scnet"
    backend_name = "SCNet (SSH + Slurm)"

    def __init__(self, runtime: MolecularVaspRuntime | None = None) -> None:
        self._runtime_instance = runtime

    def _runtime(self) -> MolecularVaspRuntime:
        if self._runtime_instance is not None:
            return self._runtime_instance
        return default_molecular_runtime()

    def probe(self) -> ProbeResult:
        runtime = self._runtime()
        if not runtime.configured:
            return ProbeResult(
                status=CapabilityStatus.UNCONFIGURED,
                detail="no SCNet backend configured (SCNET_HOST/USERNAME)",
            )
        return ProbeResult(
            status=CapabilityStatus.AVAILABLE,
            detail="study layer ready; execution gated by policy",
            version="vasp-study",
        )

    def tools(self) -> list[Tool]:
        runtime = self._runtime()
        return [
            VaspStudyPlanTool(runtime),
            VaspStudyExecuteTool(runtime),
            VaspStudyStatusTool(runtime),
            VaspStudyResumeTool(runtime),
            VaspStudyCollectTool(runtime),
            VaspStudyReportTool(runtime),
        ]


def vasp_study_pack(
    runtime: MolecularVaspRuntime | None = None,
) -> VaspStudyCapabilityPack:
    return VaspStudyCapabilityPack(runtime)
