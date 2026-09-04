"""Legacy ``vasp_molecule.*`` tool pack (migration only).

These classes are retained as internal compatibility helpers while the
unified ``vasp.*`` surface is authoritative. They are NOT registered in the
model-visible ToolRegistry and are NOT DEFERRED model tools.

Every tool stays HIDDEN in this legacy module and is never registered in the
model-visible registry. Direct Python callers still receive the same bounded
payload (``<= 4000`` characters) with no source code, no full file listings
and no POTCAR content. Scheduling states never become scientific evidence by
themselves: evidence is attached only when result validation passes.

The tools are thin adapters over :class:`MolecularVaspTools`: exactly the
same deterministic generator / preflight / lifecycle / parser that the
offline tests exercise. New code should use the unified service instead.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from photomatagent.scientific.applications.vasp.molecular.binding import (
    BindingEnergyInput,
)
from photomatagent.scientific.applications.vasp.molecular.models import (
    MoleculeSpec,
    WorkflowSpec,
)
from photomatagent.scientific.applications.vasp.molecular.runtime import (
    MolecularVaspRuntime,
    default_molecular_runtime,
)
from photomatagent.scientific.applications.vasp.molecular.workflow import (
    build_molecule_workflow,
)
from photomatagent.scientific.capabilities.base import CapabilityPack
from photomatagent.scientific.capabilities.contracts import ScientificToolResult
from photomatagent.tools.base import Tool
from photomatagent.tools.exposure import ToolExposure


def _result(payload: dict[str, Any]) -> ScientificToolResult:
    return ScientificToolResult(
        output=json.dumps(payload, ensure_ascii=False, indent=2),
        data=payload,
        is_error=not bool(payload.get("ok", True)),
    )


def _unconfigured() -> ScientificToolResult:
    return ScientificToolResult(
        output=json.dumps(
            {
                "ok": False,
                "errors": [
                    "SCNet is not configured for submission; set "
                    "SCNET_HOST/SCNET_USERNAME (or SUPERCOMPUTING_HOST/"
                    "SUPERCOMPUTING_USERNAME)"
                ],
                "note": "offline prepare/preflight/analysis remain available",
            },
            ensure_ascii=False,
            indent=2,
        ),
        is_error=True,
        data={
            "error_type": "missing_prerequisites",
            "missing": ["SCNET_HOST", "SCNET_USERNAME"],
            "ok": False,
        },
    )


class _MolecularTool(Tool):
    """Shared defaults for the molecular VASP tool family."""

    namespace = "vasp_molecule"
    source = "photomatagent molecular VASP"
    # Migration-only compatibility helpers are never model-visible.
    exposure = ToolExposure.HIDDEN
    tags = ("vasp", "molecule", "dft")
    cost_class = "EXPENSIVE"

    def __init__(self, runtime: MolecularVaspRuntime) -> None:
        self.runtime = runtime

    def _facade(self, workflow_dir: str | Path | None = None) -> Any:
        from photomatagent.scientific.applications.vasp.molecular.tools import (
            MolecularVaspTools,
        )

        resolved = (
            Path(workflow_dir).expanduser().resolve()
            if workflow_dir is not None
            else self.runtime.workflow_dir
        )
        return MolecularVaspTools(
            session=self.runtime.session,
            backend=self.runtime.backend,
            psp_dir=self.runtime.psp_dir,
            workflow_dir=resolved,
            log_dir=self.runtime.log_dir,
            module_name=self.runtime.module_name,
            env_script=self.runtime.env_script,
            remote_psp_dir=self.runtime.remote_psp_dir,
            configured=self.runtime.configured,
        )

    def _workflow(self, arguments: dict[str, Any]) -> tuple[WorkflowSpec, Path]:
        raw = arguments.get("workflow_dir")
        if raw:
            root = Path(raw).expanduser().resolve()
        elif self.runtime.workflow_dir is not None:
            root = self.runtime.workflow_dir
        else:
            raise ValueError("workflow_dir is required")
        manifest = root / "workflow.json"
        if not manifest.is_file():
            raise ValueError(f"workflow.json missing in {root}")
        workflow = WorkflowSpec.model_validate_json(
            manifest.read_text(encoding="utf-8")
        )
        return workflow, root


class MolecularVaspCapabilitiesTool(_MolecularTool):
    name = "vasp_molecule.capabilities"
    description = (
        "List isolated-molecule VASP capabilities: runtime configuration "
        "(SCNet backend, pseudopotential dirs, module/environment), "
        "workspace-scoped workflow/log/registry paths and the POTCAR policy. "
        "Read-only; never submits."
    )
    short_description = "Isolated-molecule VASP runtime configuration."
    cost_class = "CHEAP"
    input_schema: dict[str, Any] = {"type": "object", "properties": {}}

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        payload = self.runtime.capabilities_payload()
        payload["ok"] = True
        payload["stages"] = [
            "relax", "static_preconverge", "corrected_static",
            "orbital_homo", "orbital_lumo", "esp",
        ]
        payload["note"] = (
            "molecular DAG is structure -> relax -> static_preconverge -> "
            "corrected_static -> {orbital_homo, orbital_lumo, esp}"
        )
        return _result(payload)


class MolecularVaspPrepareTool(_MolecularTool):
    name = "vasp_molecule.prepare"
    description = (
        "Generate the typed isolated-molecule VASP stage tree "
        "(POSCAR/INCAR/KPOINTS/POTCAR.meta/POTCAR.policy per stage), run the "
        "deterministic preflight and write workflow.json + preflight.json + "
        "task_state.json. Requires an explicit total_charge (never inferred "
        "from names). Never submits."
    )
    short_description = "Generate isolated-molecule VASP inputs offline."
    cost_class = "MODERATE"
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "structure_path": {"type": "string"},
            "structure_kind": {
                "type": "string",
                "enum": ["xyz", "sdf", "mol", "poscar"],
                "description": (
                    "Optional format hint. Extensionless files named exactly "
                    "POSCAR or CONTCAR are auto-detected as POSCAR; .xyz/.sdf/"
                    ".mol/.poscar suffixes are also auto-detected."
                ),
            },
            "name": {"type": "string"},
            "total_charge": {"type": "integer"},
            "spin_multiplicity": {"type": "integer", "minimum": 1},
            "box_ang": {"type": "number", "minimum": 5.01, "maximum": 100.0},
            "functional": {"type": "string"},
            "calculation_purpose": {"type": "string"},
            "conformer_id": {"type": "string"},
            "encut_ev": {"type": "number", "minimum": 200, "maximum": 1000},
            "include_orbital_homo": {"type": "boolean"},
            "include_orbital_lumo": {"type": "boolean"},
            "include_esp": {"type": "boolean"},
            "include_hse06": {"type": "boolean"},
            "workflow_dir": {"type": "string"},
        },
        "required": ["structure_path", "total_charge"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        raw_dir = arguments.get("workflow_dir")
        root = (
            Path(raw_dir).expanduser().resolve()
            if raw_dir
            else (
                self.runtime.workflow_dir.resolve()
                if self.runtime.workflow_dir is not None
                else Path.cwd() / "output" / "vasp_molecule"
            )
        )
        manifest = root / "workflow.json"
        if manifest.is_file():
            workflow = WorkflowSpec.model_validate_json(
                manifest.read_text(encoding="utf-8")
            )
        else:
            name = arguments.get("name") or Path(
                arguments["structure_path"]
            ).stem
            molecule = MoleculeSpec(
                name=str(name),
                structure_path=Path(arguments["structure_path"]),
                structure_kind=(
                    arguments.get("structure_kind") or None
                ),
                total_charge=int(arguments["total_charge"]),
                spin_multiplicity=int(arguments.get("spin_multiplicity", 1)),
                box_ang=float(arguments.get("box_ang", 30.0)),
                functional=str(arguments.get("functional", "PBE-D3(BJ)")),
                calculation_purpose=str(
                    arguments.get("calculation_purpose", "unspecified")
                ),
                conformer_id=(
                    str(arguments["conformer_id"])
                    if arguments.get("conformer_id")
                    else None
                ),
            )
            workflow = build_molecule_workflow(
                molecule,
                psp_dir=self.runtime.psp_dir,
                encut_ev=(
                    float(arguments["encut_ev"])
                    if arguments.get("encut_ev") is not None
                    else None
                ),
                spin=int(arguments.get("spin_multiplicity", 1)),
                include_orbital_homo=bool(
                    arguments.get("include_orbital_homo", True)
                ),
                include_orbital_lumo=bool(
                    arguments.get("include_orbital_lumo", True)
                ),
                include_esp=bool(arguments.get("include_esp", True)),
                include_hse06=bool(arguments.get("include_hse06", False)),
            )
        payload = await self._facade().prepare(workflow, output_dir=root)
        return _result(payload)


class MolecularVaspPreflightTool(_MolecularTool):
    name = "vasp_molecule.preflight"
    description = (
        "Run the deterministic offline preflight on a prepared molecular "
        "workflow directory. Returns passed/errors/warnings and writes "
        "preflight.json. Charge, POTCAR ordering, NELECT parity, vacuum, "
        "Γ-only sampling, dipolar rendering and stage dependencies are all "
        "checked before any submission is allowed."
    )
    short_description = "Deterministic molecular VASP preflight (offline)."
    cost_class = "MODERATE"
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {"workflow_dir": {"type": "string"}},
        "required": ["workflow_dir"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        try:
            workflow, root = self._workflow(arguments)
        except ValueError as exc:
            return _result(
                {"ok": False, "errors": [str(exc)], "summary": {}}
            )
        payload = await self._facade(root).preflight(workflow)
        return _result(payload)


class MolecularVaspSubmitTool(_MolecularTool):
    name = "vasp_molecule.submit"
    description = (
        "Submit ONE prepared molecular stage to SCNet under the submit-once "
        "contract: only a passing preflight allows submission; the same "
        "request_id never creates a second job; each job gets its own unique "
        "remote directory. Generates and uploads run.slurm (shared Slurm "
        "template, srun --mpi=pmi2 vasp_std) and assembles POTCAR remotely "
        "from SCNET_VASP_PSP_DIR in POSCAR element order. Requires "
        "PHOTOMATAGENT_ALLOW_HPC_SUBMIT=1."
    )
    short_description = "Submit one isolated-molecule stage (submit-once)."
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "workflow_dir": {"type": "string"},
            "stage": {
                "type": "string",
                "enum": [
                    "relax", "static_preconverge", "corrected_static",
                    "orbital_homo", "orbital_lumo", "esp", "static_hse",
                ],
            },
            "wait": {"type": "boolean"},
            "wait_timeout_seconds": {"type": "number", "minimum": 1},
            "force_new_attempt": {"type": "boolean"},
        },
        "required": ["workflow_dir", "stage"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        if not self.runtime.configured:
            return _unconfigured()
        workflow, root = self._workflow(arguments)
        payload = await self._facade(root).submit(
            arguments["stage"],
            workflow,
            wait=bool(arguments.get("wait", False)),
            wait_timeout_seconds=float(
                arguments.get("wait_timeout_seconds", 3600.0)
            ),
            force_new_attempt=bool(arguments.get("force_new_attempt", False)),
        )
        return _result(payload)


class MolecularVaspStatusTool(_MolecularTool):
    name = "vasp_molecule.status"
    description = (
        "Read the lifecycle + scheduler state of one submitted molecular "
        "stage. A query failure is reported as UNKNOWN, never as a job "
        "failure. Scheduler COMPLETED is not scientific validity."
    )
    short_description = "Poll one isolated-molecule stage state."
    cost_class = "CHEAP"
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {"workflow_dir": {"type": "string"}, "stage": {"type": "string"}},
        "required": ["workflow_dir", "stage"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        if not self.runtime.configured:
            return _unconfigured()
        root = Path(arguments["workflow_dir"]).expanduser().resolve()
        payload = await self._facade(root).status(arguments["stage"])
        return _result(payload)


class MolecularVaspCollectTool(_MolecularTool):
    name = "vasp_molecule.collect"
    description = (
        "Download one scheduler-COMPLETED molecular stage, parse E0/OSZICAR/"
        "EIGENVAL, and advance the lifecycle COMPLETED -> COLLECTED -> "
        "VALIDATED. Evidence is produced only when the scientific validation "
        "passes; a failed validation keeps COLLECTED and produces no "
        "evidence. Both task_state.json and the SQLite registry are updated."
    )
    short_description = "Collect and validate one isolated-molecule stage."
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "workflow_dir": {"type": "string"},
            "stage": {"type": "string"},
            "local_dir": {"type": "string"},
        },
        "required": ["workflow_dir", "stage"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        if not self.runtime.configured:
            return _unconfigured()
        root = Path(arguments["workflow_dir"]).expanduser().resolve()
        payload = await self._facade(root).collect(
            arguments["stage"],
            local_dir=(
                arguments["local_dir"] if arguments.get("local_dir") else None
            ),
        )
        return _result(payload)


class MolecularVaspAnalyzeOrbitalsTool(_MolecularTool):
    name = "vasp_molecule.analyze_orbitals"
    description = (
        "HOMO/LUMO identification from EIGENVAL occupations plus vacuum "
        "alignment from LOCPOT (offline). Returns band indices, raw and "
        "aligned energies; raw values must never be compared across "
        "molecules without the same vacuum reference."
    )
    short_description = "HOMO/LUMO + vacuum alignment (offline)."
    cost_class = "MODERATE"
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "result_dir": {"type": "string"},
            "charge": {"type": "integer"},
            "spin_multiplicity": {"type": "integer", "minimum": 1},
            "box_ang": {"type": "number", "minimum": 5.01},
        },
        "required": ["result_dir"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        payload = await self._facade().analyze_orbitals(
            arguments["result_dir"],
            charge=int(arguments.get("charge", 0)),
            spin_multiplicity=int(arguments.get("spin_multiplicity", 1)),
            box_ang=(
                float(arguments["box_ang"])
                if arguments.get("box_ang") is not None
                else None
            ),
        )
        return _result(payload)


class MolecularVaspAnalyzeEspTool(_MolecularTool):
    name = "vasp_molecule.analyze_esp"
    description = (
        "ESP/LOCPOT grid metadata for one result directory (offline). Only "
        "metadata is returned; the potential grid content stays on disk."
    )
    short_description = "ESP/LOCPOT grid metadata (offline)."
    cost_class = "MODERATE"
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {"result_dir": {"type": "string"}},
        "required": ["result_dir"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        payload = await self._facade().analyze_esp(arguments["result_dir"])
        return _result(payload)


class MolecularVaspBindingEnergyTool(_MolecularTool):
    name = "vasp_molecule.binding_energy"
    description = (
        "Electronic binding energy (ΔE and ΔΔE) from validated E0 values "
        "with strict parameter-consistency checks: same box, functional, "
        "ENCUT and corrections. No vibrational/thermal corrections are "
        "claimed; the result is labelled electronic-only."
    )
    short_description = "Electronic binding energy (validated E0 only)."
    cost_class = "MODERATE"
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "complex_name": {"type": "string"},
            "complex_dir": {"type": "string"},
            "references": {"type": "array", "items": {"type": "object"}},
            "alternative_references": {
                "type": "array", "items": {"type": "object"},
            },
            "charge": {"type": "integer"},
        },
        "required": ["complex_name", "complex_dir", "references"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        inputs = BindingEnergyInput(
            complex_name=str(arguments["complex_name"]),
            complex_dir=str(arguments["complex_dir"]),
            references=arguments["references"],
            alternative_references=arguments.get(
                "alternative_references", []
            ),
            charge=int(arguments.get("charge", 0)),
        )
        payload = await self._facade().binding_energy(inputs)
        return _result(payload)


class MolecularVaspResumeWorkflowTool(_MolecularTool):
    name = "vasp_molecule.resume_workflow"
    description = (
        "Run (or resume) the full molecular DAG from task_state.json: "
        "completed stages (COMPLETED/COLLECTED/VALIDATED) are never "
        "resubmitted; a stage failure blocks every dependent. Collects and "
        "validates each finished stage and persists task_state.json for "
        "later sessions."
    )
    short_description = "Run or resume the isolated-molecule DAG."
    cost_class = "VERY_EXPENSIVE"
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "workflow_dir": {"type": "string"},
            "wait": {"type": "boolean"},
            "collect": {"type": "boolean"},
            "stop_on_failure": {"type": "boolean"},
            "only": {"type": "array", "items": {"type": "string"}},
            "wait_timeout_seconds": {"type": "number", "minimum": 1},
        },
        "required": ["workflow_dir"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        if not self.runtime.configured:
            return _unconfigured()
        workflow, root = self._workflow(arguments)
        payload = await self._facade(root).run_workflow(
            workflow,
            wait=bool(arguments.get("wait", True)),
            collect=bool(arguments.get("collect", True)),
            stop_on_failure=bool(arguments.get("stop_on_failure", True)),
            only=arguments.get("only"),
            wait_timeout_seconds=float(
                arguments.get("wait_timeout_seconds", 3600.0)
            ),
        )
        return _result(payload)


class MolecularVaspCapabilityPack(CapabilityPack):
    """Hidden migration pack for the retired ``vasp_molecule.*`` family."""

    name = "vasp_molecule"
    description = (
        "Isolated-molecule VASP on SCNet (typed DAG, deterministic "
        "preflight, submit-once lifecycle, orbital/ESP/binding analysis)."
    )
    execution_mode = "mcp/scnet"
    backend_name = "SCNet (SSH + Slurm)"

    def __init__(self, runtime: MolecularVaspRuntime | None = None) -> None:
        self.runtime = runtime or default_molecular_runtime

    def probe(self) -> Any:
        from photomatagent.scientific.capabilities.base import (
            CapabilityStatus,
            ProbeResult,
        )

        runtime = self._runtime()
        if not runtime.configured:
            return ProbeResult(
                status=CapabilityStatus.UNCONFIGURED,
                detail="no SCNet backend configured (SCNET_HOST/USERNAME)",
            )
        return ProbeResult(
            status=CapabilityStatus.AVAILABLE,
            detail="SCNet backend configured; molecular tools ready",
            version="isolated-molecule vasp",
        )

    def tools(self) -> list[Tool]:
        runtime = self._runtime()
        return [
            MolecularVaspCapabilitiesTool(runtime),
            MolecularVaspPrepareTool(runtime),
            MolecularVaspPreflightTool(runtime),
            MolecularVaspSubmitTool(runtime),
            MolecularVaspStatusTool(runtime),
            MolecularVaspCollectTool(runtime),
            MolecularVaspAnalyzeOrbitalsTool(runtime),
            MolecularVaspAnalyzeEspTool(runtime),
            MolecularVaspBindingEnergyTool(runtime),
            MolecularVaspResumeWorkflowTool(runtime),
        ]

    def _runtime(self) -> MolecularVaspRuntime:
        if callable(self.runtime):
            return self.runtime()
        return self.runtime


def molecular_vasp_pack(runtime: MolecularVaspRuntime | None = None) -> MolecularVaspCapabilityPack:
    return MolecularVaspCapabilityPack(runtime)
