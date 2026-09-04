"""Task 1: typed unified VASP requests, manifests, and fingerprints."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from photomatagent.scientific.applications.vasp.molecular.models import (
    MoleculeSpec,
    StageName,
    StageSpec,
    WorkflowSpec,
)
from photomatagent.scientific.applications.vasp.unified.fingerprints import (
    canonical_json,
    execution_fingerprint,
    scientific_fingerprint,
)
from photomatagent.scientific.applications.vasp.unified.models import (
    MolecularScientificSpec,
    PeriodicScientificSpec,
    StudyScientificSpec,
    UnifiedStage,
    UnifiedVaspManifest,
    UnifiedVaspRequest,
    VaspWorkflowKind,
    WorkflowState,
)
from photomatagent.scientific.applications.vasp.study.models import VaspStudyRequest
from photomatagent.scientific.remote.models import ResourceRequest


def write_structure(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def periodic_spec(
    structure_path: str | Path,
    *,
    profile: str = "standard_semiconductor",
    overrides: dict | None = None,
    potcar_policy: str = "configured",
) -> PeriodicScientificSpec:
    return PeriodicScientificSpec(
        structure_path=str(structure_path),
        profile=profile,
        scientific_overrides=overrides or {},
        potcar_policy=potcar_policy,
    )


def molecular_workflow(
    tmp_path: Path,
    *,
    charge: int = 0,
    spin: int = 1,
    structure_content: str = "2\nLi\nLi 0 0 0\nLi 1 0 0\n",
) -> WorkflowSpec:
    structure = tmp_path / "molecule.xyz"
    write_structure(structure, structure_content)
    return WorkflowSpec(
        molecule=MoleculeSpec(
            name="Li2",
            structure_path=structure,
            structure_kind="xyz",
            total_charge=charge,
            spin_multiplicity=spin,
        ),
        stages=[StageSpec(name=StageName.RELAX, depends_on=None)],
        scientific_method="PBE-D3(BJ)",
    )


def manifest(
    spec,
    stages=None,
    *,
    workflow_id: str = "wf-test",
    workflow_kind: VaspWorkflowKind | None = None,
) -> UnifiedVaspManifest:
    kind = workflow_kind or VaspWorkflowKind(spec.kind)
    resolved_stages = stages if stages is not None else [UnifiedStage(name="stage")]
    return UnifiedVaspManifest(
        workflow_id=workflow_id,
        workflow_kind=kind,
        scientific_spec=spec,
        scientific_fingerprint=scientific_fingerprint(spec, resolved_stages),
        stages=resolved_stages,
    )


def test_molecular_request_without_explicit_total_charge_fails(tmp_path):
    structure = write_structure(tmp_path / "missing.xyz", "1\nX\nX 0 0 0\n")
    with pytest.raises(ValidationError):
        WorkflowSpec.model_validate(
            {
                "molecule": {
                    "name": "X",
                    "structure_path": str(structure),
                    "structure_kind": "xyz",
                    "spin_multiplicity": 1,
                    # total_charge intentionally omitted
                },
                "stages": [{"name": "relax"}],
                "scientific_method": "PBE-D3(BJ)",
            }
        )


def test_request_rejects_workflow_kind_spec_mismatch(tmp_path):
    structure = write_structure(tmp_path / "periodic.cif", "periodic structure")
    periodic = periodic_spec(structure)
    with pytest.raises(ValidationError):
        UnifiedVaspRequest(
            workflow_kind=VaspWorkflowKind.MOLECULAR,
            scientific_spec=periodic,
        )


def test_resource_change_does_not_alter_scientific_hash(tmp_path):
    structure = write_structure(tmp_path / "a.cif", "same")
    spec = periodic_spec(structure)
    stages_a = [
        UnifiedStage(
            name="relax",
            resource_recommendation=ResourceRequest(
                nodes=1, tasks_per_node=8, walltime_minutes=60
            ),
        )
    ]
    stages_b = [
        UnifiedStage(
            name="relax",
            resource_recommendation=ResourceRequest(
                nodes=4, tasks_per_node=64, walltime_minutes=720
            ),
        )
    ]
    assert scientific_fingerprint(spec, stages_a) == scientific_fingerprint(
        spec, stages_b
    )


def test_encut_change_alters_scientific_hash(tmp_path):
    structure = write_structure(tmp_path / "a.cif", "same")
    base = periodic_spec(structure, overrides={"encut_ev": 500})
    changed = periodic_spec(structure, overrides={"encut_ev": 520})
    assert scientific_fingerprint(base) != scientific_fingerprint(changed)


def test_kpoints_change_alters_scientific_hash(tmp_path):
    structure = write_structure(tmp_path / "a.cif", "same")
    base = periodic_spec(structure, overrides={"kpoint_grid": [3, 3, 3]})
    changed = periodic_spec(structure, overrides={"kpoint_grid": [4, 4, 4]})
    assert scientific_fingerprint(base) != scientific_fingerprint(changed)


def test_soc_change_alters_scientific_hash(tmp_path):
    structure = write_structure(tmp_path / "a.cif", "same")
    no_soc = periodic_spec(structure, profile="standard_semiconductor")
    soc = periodic_spec(structure, profile="narrow_gap_soc")
    assert scientific_fingerprint(no_soc) != scientific_fingerprint(soc)


def test_charge_and_spin_changes_alter_scientific_hash(tmp_path):
    neutral = molecular_workflow(tmp_path, charge=0, spin=1)
    charged = molecular_workflow(tmp_path, charge=-1, spin=1)
    triplet = molecular_workflow(tmp_path, charge=0, spin=3)

    neutral_spec = MolecularScientificSpec(workflow=neutral)
    charged_spec = MolecularScientificSpec(workflow=charged)
    triplet_spec = MolecularScientificSpec(workflow=triplet)

    assert scientific_fingerprint(neutral_spec) != scientific_fingerprint(charged_spec)
    assert scientific_fingerprint(neutral_spec) != scientific_fingerprint(triplet_spec)


def test_structure_content_change_alters_scientific_hash(tmp_path):
    structure = write_structure(tmp_path / "a.cif", "initial content")
    first = periodic_spec(structure)
    before = scientific_fingerprint(first)
    write_structure(structure, "changed content")
    second = periodic_spec(structure)
    after = scientific_fingerprint(second)
    assert before != after


def test_potcar_policy_change_alters_scientific_hash(tmp_path):
    structure = write_structure(tmp_path / "a.cif", "same")
    configured = periodic_spec(structure, potcar_policy="configured")
    local = periodic_spec(structure, potcar_policy="local")
    assert scientific_fingerprint(configured) != scientific_fingerprint(local)


def test_stage_change_alters_scientific_hash(tmp_path):
    structure = write_structure(tmp_path / "a.cif", "same")
    spec = periodic_spec(structure)
    stages_a = [UnifiedStage(name="relax"), UnifiedStage(name="static")]
    stages_b = [UnifiedStage(name="relax"), UnifiedStage(name="band")]
    assert scientific_fingerprint(spec, stages_a) != scientific_fingerprint(
        spec, stages_b
    )


def test_dictionary_order_does_not_alter_hash(tmp_path):
    structure = write_structure(tmp_path / "a.cif", "same")
    first = periodic_spec(
        structure,
        overrides={
            "encut_ev": 520,
            "kpoint_grid": [4, 4, 4],
            "scientific_tags": ["x", "y"],
        },
    )
    second = periodic_spec(
        structure,
        overrides={
            "scientific_tags": ["x", "y"],
            "kpoint_grid": [4, 4, 4],
            "encut_ev": 520,
        },
    )
    assert scientific_fingerprint(first) == scientific_fingerprint(second)
    assert canonical_json(first.model_dump()) == canonical_json(
        second.model_dump()
    )


def test_equivalent_relative_paths_do_not_alter_hash(tmp_path):
    # Both spellings are workspace-relative and normalize to the same path.
    direct = periodic_spec(Path("nested") / "a.cif")
    indirect = periodic_spec(Path("nested") / ".." / "nested" / "a.cif")
    assert scientific_fingerprint(direct) == scientific_fingerprint(indirect)


def test_fingerprints_are_sha256_and_stable(tmp_path):
    structure = write_structure(tmp_path / "a.cif", "same")
    request = UnifiedVaspRequest(
        workflow_kind=VaspWorkflowKind.PERIODIC,
        scientific_spec=periodic_spec(structure),
    )
    first = scientific_fingerprint(request)
    second = scientific_fingerprint(request)
    assert first == second
    assert len(first) == 64
    assert len(execution_fingerprint(first)) == 64


def test_execution_fingerprint_includes_resources_and_stage(tmp_path):
    structure = write_structure(tmp_path / "a.cif", "same")
    spec = periodic_spec(structure)
    scientific = scientific_fingerprint(spec)
    base = execution_fingerprint(
        scientific,
        resource=ResourceRequest(nodes=1, tasks_per_node=8, walltime_minutes=60),
        stage="relax",
    )
    changed = execution_fingerprint(
        scientific,
        resource=ResourceRequest(nodes=2, tasks_per_node=16, walltime_minutes=120),
        stage="relax",
    )
    changed_stage = execution_fingerprint(
        scientific,
        resource=ResourceRequest(nodes=1, tasks_per_node=8, walltime_minutes=60),
        stage="static",
    )
    assert base != changed
    assert base != changed_stage


def test_study_spec_accepts_typed_request():
    study_request = VaspStudyRequest(study_id="s1", systems=[])
    spec = StudyScientificSpec(request=study_request)
    assert spec.kind == "study"
    unified = UnifiedVaspRequest(
        workflow_kind=VaspWorkflowKind.STUDY,
        scientific_spec=spec,
    )
    assert unified.scientific_spec.request.study_id == "s1"
