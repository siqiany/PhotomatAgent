"""Study planning: resolve structures, build the matrix, persist artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from photomatagent.scientific.applications.vasp.study.matrix import (
    build_calculation_matrix,
)
from photomatagent.scientific.applications.vasp.study.models import (
    StudyTaskState,
    VaspStudyRequest,
    VaspStudySpec,
)
from photomatagent.scientific.capabilities.chemistry.conformers import (
    ChemistryError,
)
from photomatagent.scientific.capabilities.chemistry.resolver import (
    StructureRequest,
    resolve_structure,
)
from photomatagent.scientific.capabilities.chemistry.storage import (
    write_structure_manifest,
    write_structure_thumbnails,
)


def derive_study_id(request: VaspStudyRequest) -> str:
    """Content-addressed study id: stable across resume calls."""
    if request.study_id:
        return request.study_id
    payload = json.dumps(
        {
            "systems": [
                {
                    "id": system.system_id,
                    "charge": system.total_charge,
                    "smiles": system.smiles,
                    "structure_path": (
                        str(system.structure_path)
                        if system.structure_path
                        else None
                    ),
                    "properties": [p.value for p in system.properties],
                }
                for system in request.systems
            ],
            "properties": [p.value for p in request.property_requests],
            "method": request.method.model_dump(mode="json"),
        },
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:12]
    return f"study-{digest}"


def plan_study(
    request: VaspStudyRequest,
    workspace: str | Path,
) -> VaspStudySpec:
    """Resolve structures, build the matrix and persist plan artifacts."""
    workspace_path = Path(workspace).expanduser().resolve()
    study_id = derive_study_id(request)
    study_dir = workspace_path / "output" / "vasp_study" / study_id
    structures_dir = study_dir / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)

    policy = request.structure_policy
    # 1) explicit request systems
    resolved: dict[str, list[Any]] = {}
    for system in request.systems:
        try:
            structures = resolve_structure(
                StructureRequest(
                    system_id=system.system_id,
                    display_name=system.display_name or system.system_id,
                    aliases=list(system.aliases),
                    smiles=system.smiles,
                    structure_path=system.structure_path,
                    total_charge=system.total_charge,
                    spin_multiplicity=system.spin_multiplicity,
                    role=system.role,
                    allow_assumed=policy.allow_assumed_structures,
                    max_candidates=policy.max_candidates_per_system,
                    seed=policy.seed,
                ),
                structures_dir,
            )
        except ChemistryError as exc:
            # A failing structure never blocks the whole study: the system
            # enters the matrix as SKIPPED_PROXY.
            resolved[system.system_id] = []
            (study_dir / "structure_errors.jsonl").open("a", encoding="utf-8").write(
                json.dumps(
                    {
                        "system_id": system.system_id,
                        "error": f"{exc.code}: {exc}",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            continue
        resolved[system.system_id] = structures

    # 2) implicit fragments referenced by complex provenance (Li+, TFSI-...)
    for structures in list(resolved.values()):
        for structure in structures:
            for parent_id in structure.provenance.parent_structures:
                if parent_id in resolved:
                    continue
                try:
                    resolved[parent_id] = resolve_structure(
                        StructureRequest(
                            system_id=parent_id,
                            display_name=parent_id,
                            allow_assumed=policy.allow_assumed_structures,
                            max_candidates=policy.max_candidates_per_system,
                            seed=policy.seed,
                        ),
                        structures_dir,
                    )
                except ChemistryError:
                    resolved[parent_id] = []

    manifest_path = write_structure_manifest(
        [structure for structures in resolved.values() for structure in structures],
        study_dir / "structure_manifest.json",
    )
    write_structure_thumbnails(
        [structure for structures in resolved.values() for structure in structures],
        study_dir / "figures" / "structures",
    )

    matrix = build_calculation_matrix(request, resolved)
    (study_dir / "study_request.json").write_text(
        json.dumps(request.model_dump(mode="json"), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    (study_dir / "calculation_matrix.json").write_text(
        json.dumps(matrix.model_dump(mode="json"), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return VaspStudySpec(
        study_id=study_id,
        request=request,
        study_dir=study_dir,
        calculation_matrix=matrix,
        structure_manifest_path=manifest_path,
    )


def load_planned_study(study_dir: str | Path) -> VaspStudySpec:
    """Reload a planned study from disk (resume contract)."""
    root = Path(study_dir).expanduser().resolve()
    request = VaspStudyRequest.model_validate_json(
        (root / "study_request.json").read_text(encoding="utf-8")
    )
    matrix_payload = json.loads(
        (root / "calculation_matrix.json").read_text(encoding="utf-8")
    )
    from photomatagent.scientific.applications.vasp.study.models import (
        CalculationMatrix,
    )

    matrix = CalculationMatrix.model_validate(matrix_payload)
    manifest = root / "structure_manifest.json"
    return VaspStudySpec(
        study_id=request.study_id or root.name,
        request=request,
        study_dir=root,
        calculation_matrix=matrix,
        structure_manifest_path=manifest,
    )


def budget_status(spec: VaspStudySpec) -> dict[str, Any]:
    budget = spec.request.resource_budget
    return {
        "total_core_hours": spec.calculation_matrix.total_core_hours,
        "budget_core_hours": budget.max_core_hours,
        "within_budget": (
            spec.calculation_matrix.total_core_hours
            <= budget.max_core_hours + 1e-9
        ),
        "max_jobs": budget.max_jobs,
        "total_jobs": spec.calculation_matrix.total_jobs,
        "estimated_disk_gb": spec.calculation_matrix.estimated_disk_gb,
    }
