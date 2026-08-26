"""Study-level result aggregation (never re-parses VASP outputs)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from photomatagent.scientific.applications.vasp.study.executor import (
    _stage_energy_dir,
)
from photomatagent.scientific.applications.vasp.study.models import (
    PropertyRequest,
    StudyTaskState,
    VaspStudySpec,
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _stage_results(workflow_dir: Path, stage: str) -> dict[str, Any]:
    return _load_json(Path(workflow_dir) / "results" / stage / "results.json")


def task_result_row(
    spec: VaspStudySpec, task: Any
) -> dict[str, Any]:
    """One results row per unique calculation (all values from validated
    stage results.json files; never re-parsed here)."""
    workflow_dir = Path(task.workflow_dir or "")
    energy_dir = Path(task.results_dir) if task.results_dir else None
    energy: dict[str, Any] = {}
    scf: dict[str, Any] = {}
    orbitals: dict[str, Any] = {}
    vacuum: dict[str, Any] = {}
    esp: dict[str, Any] = {}
    if energy_dir is not None and (energy_dir / "results.json").is_file():
        payload = _load_json(energy_dir / "results.json")
        energy = payload.get("energy") or {}
        scf = payload.get("scf") or {}
    orbital_stage = None
    for stage_name, key in (
        ("orbital_homo", "homo"),
        ("orbital_lumo", "lumo"),
    ):
        payload = _stage_results(workflow_dir, stage_name)
        if not payload.get("validated"):
            continue
        orbital_stage = stage_name
        orbitals = payload.get("orbitals") or {}
        vacuum = payload.get("vacuum") or {}
        break
    esp_payload = _stage_results(workflow_dir, "esp")
    if esp_payload.get("validated"):
        esp = esp_payload.get("esp") or {}
    parchg_files = sorted(
        path.name
        for stage in ("orbital_homo", "orbital_lumo")
        for path in (Path(workflow_dir) / "results" / stage).glob("PARCHG*")
        if path.is_file()
    )
    return {
        "task_id": task.task_id,
        "system": task.display_name,
        "role": task.role,
        "formula": task.formula,
        "charge": task.total_charge,
        "spin_multiplicity": task.spin_multiplicity,
        "reliability": task.reliability,
        "structure_status": task.structure_status,
        "state": task.state,
        "conformer_index": task.conformer_index,
        "e0_ev": energy.get("e_0_ev"),
        "e_fr_ev": energy.get("e_fr_ev"),
        "energy_source": energy.get("source"),
        "scf_converged": scf.get("converged"),
        "homo_raw_ev": orbitals.get("homo_raw_ev"),
        "lumo_raw_ev": orbitals.get("lumo_raw_ev"),
        "homo_aligned_ev": vacuum.get("aligned_homo_ev"),
        "lumo_aligned_ev": vacuum.get("aligned_lumo_ev"),
        "vacuum_level_ev": vacuum.get("level_ev"),
        "ks_gap_ev": orbitals.get("ks_gap_ev"),
        "esp_has_locpot": esp.get("has_locpot"),
        "parchg_files": parchg_files,
        "error": task.error,
        "workflow_dir": task.workflow_dir,
    }


def analyze_study(
    spec: VaspStudySpec,
) -> dict[str, Any]:
    """Build results.json content: task rows + binding table + provenance."""
    rows = [
        task_result_row(spec, task)
        for task in spec.calculation_matrix.tasks
    ]
    binding_rows: list[dict[str, Any]] = []
    task_map = spec.calculation_matrix.task_map()
    for group in spec.calculation_matrix.binding_groups:
        complex_task = task_map.get(group.complex_task_id)
        fragments = [
            task_map.get(fragment_id)
            for fragment_id in group.fragment_task_ids
        ]
        binding_rows.append(
            {
                "label": group.label,
                "complex": (
                    complex_task.display_name if complex_task else group.complex_task_id
                ),
                "complex_formula": complex_task.formula if complex_task else "",
                "complex_reliability": (
                    complex_task.reliability if complex_task else ""
                ),
                "complex_charge": group.total_charge,
                "fragments": [
                    {
                        "name": fragment.display_name if fragment else fragment_id,
                        "formula": fragment.formula if fragment else "",
                        "charge": fragment.total_charge if fragment else 0,
                        "reliability": (
                            fragment.reliability if fragment else ""
                        ),
                    }
                    for fragment, fragment_id in zip(
                        fragments, group.fragment_task_ids, strict=False
                    )
                ],
                "delta_e_ev": group.delta_e_ev,
                "delta_delta_e_ev": group.delta_delta_e_ev,
                "state": group.state,
                "error": group.error,
                "uses_declared_reference_assumption": (
                    group.uses_declared_reference_assumption
                ),
                "high_risk_absolute_binding_energy": (
                    group.high_risk_absolute_binding_energy
                ),
                "zero_electron_references": [
                    fragment.display_name
                    for fragment, fragment_id in (
                        (task_map.get(fragment_id), fragment_id)
                        for fragment_id in group.fragment_task_ids
                    )
                    if fragment is not None
                    and "zero-electron" in fragment.error
                ],
                "note": (
                    "electronic binding energy only; no vibrational, thermal "
                    "or solvation free energy is claimed"
                ),
            }
        )
    assumptions: list[dict[str, Any]] = []
    manifest = spec.structure_manifest_path
    if manifest.is_file():
        payload = _load_json(manifest)
        for row in payload.get("structures", []):
            provenance = row.get("provenance", {})
            if provenance.get("assumptions") or row.get("reliability") in {
                "C", "D",
            }:
                assumptions.append(
                    {
                        "system": row.get("display_name"),
                        "reliability": row.get("reliability"),
                        "status": provenance.get("status"),
                        "assumptions": provenance.get("assumptions", []),
                        "source": provenance.get("source"),
                        "confidence": provenance.get("confidence"),
                    }
                )
    screening: list[dict[str, Any]] = []
    from photomatagent.scientific.applications.vasp.study.screening import (
        load_screen_reports,
    )

    for report in load_screen_reports(Path(spec.study_dir)).values():
        screening.append(report.summary())

    return {
        "study_id": spec.study_id,
        "summary": {
            "unique_calculations": sum(
                1
                for task in spec.calculation_matrix.tasks
                if task.structure_path
            ),
            "validated": sum(
                1
                for task in spec.calculation_matrix.tasks
                if task.state == StudyTaskState.VALIDATED.value
            ),
            "skipped_proxy": sum(
                1
                for task in spec.calculation_matrix.tasks
                if task.state == StudyTaskState.SKIPPED_PROXY.value
            ),
            "skipped_budget": sum(
                1
                for task in spec.calculation_matrix.tasks
                if task.state == StudyTaskState.SKIPPED_BUDGET.value
            ),
            "failed": sum(
                1
                for task in spec.calculation_matrix.tasks
                if task.state in {
                    StudyTaskState.FAILED.value,
                    StudyTaskState.PREFLIGHT_FAILED.value,
                    StudyTaskState.COLLECTED.value,
                }
            ),
            "binding_groups_computed": sum(
                1
                for group in spec.calculation_matrix.binding_groups
                if group.state == StudyTaskState.VALIDATED.value
            ),
        },
        "systems": rows,
        "binding_energies": binding_rows,
        "conformer_screening": screening,
        "structure_assumptions": assumptions,
        "method": {
            "functional": spec.request.method.functional,
            "encut_ev": spec.request.method.encut_ev or 400.0,
            "box_ang": spec.request.method.box_ang,
            "gamma_only": True,
            "corrections": "per-system preflight + binding parameter checks",
        },
        "limitations": [
            "electronic energies only (no vibration/thermal/solvation)",
            "raw HOMO/LUMO are not comparable across molecules; only "
            "vacuum-aligned values are",
            "VM/TVM structures are representative oligomer proxies; results "
            "describe the constructed model, not the real network",
            "bare-ion references (Li+) carry well-known vacuum-reference "
            "error; prefer ΔΔE comparisons",
        ],
    }


def write_results_csv(results: dict[str, Any], path: Path) -> Path:
    """results.csv: one row per unique calculation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "task_id", "system", "role", "formula", "charge",
        "spin_multiplicity", "reliability", "structure_status", "state",
        "e0_ev", "scf_converged", "homo_raw_ev", "lumo_raw_ev",
        "homo_aligned_ev", "lumo_aligned_ev", "vacuum_level_ev",
        "ks_gap_ev", "esp_has_locpot", "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in results.get("systems", []):
            writer.writerow(row)
    return path
