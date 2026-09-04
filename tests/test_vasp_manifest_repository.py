"""Task 2: workspace-contained atomic manifest persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from photomatagent.errors import ToolExecutionError
from photomatagent.scientific.applications.vasp.unified.fingerprints import (
    scientific_fingerprint,
)
from photomatagent.scientific.applications.vasp.unified.models import (
    MolecularScientificSpec,
    PeriodicScientificSpec,
    StudyScientificSpec,
    UnifiedVaspManifest,
    UnifiedVaspRequest,
    VaspWorkflowKind,
)
from photomatagent.scientific.applications.vasp.molecular.models import (
    MoleculeSpec,
    StageName,
    StageSpec,
    WorkflowSpec,
)
from photomatagent.scientific.applications.vasp.study.models import (
    StudySystem,
    VaspStudyRequest,
)
from photomatagent.scientific.applications.vasp.unified.repository import (
    ManifestConflictError,
    ManifestRepository,
)
from photomatagent.workspace import Workspace


def make_request(workspace: Workspace, *, path: str = "structure.cif") -> UnifiedVaspRequest:
    source = workspace.root / path
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("initial structure", encoding="utf-8")
    spec = PeriodicScientificSpec(
        structure_path=path,
        profile="standard_semiconductor",
        scientific_overrides={"encut_ev": 500},
    )
    return UnifiedVaspRequest(
        workflow_kind=VaspWorkflowKind.PERIODIC,
        scientific_spec=spec,
    )


def test_create_generates_workflow_id_and_snapshots(tmp_path):
    workspace = Workspace(tmp_path)
    repo = ManifestRepository(workspace)
    request = make_request(workspace)

    manifest = repo.create(request)

    assert manifest.workflow_id.startswith("vasp_")
    assert manifest.revision == 0
    assert manifest.state.value == "PLANNED"
    assert (repo.workflow_dir(manifest.workflow_id) / "manifest.json").exists()
    snapshot_path = workspace.resolve(manifest.scientific_spec.structure_path)
    assert snapshot_path.exists()
    assert "initial structure" in snapshot_path.read_text(encoding="utf-8")
    assert scientific_fingerprint(manifest.scientific_spec) == manifest.scientific_fingerprint


def test_absolute_and_dotdot_escapes_rejected(tmp_path):
    workspace = Workspace(tmp_path)
    repo = ManifestRepository(workspace)
    request = UnifiedVaspRequest(
        workflow_kind=VaspWorkflowKind.PERIODIC,
        scientific_spec=PeriodicScientificSpec(
            structure_path="/etc/passwd",
            profile="standard_semiconductor",
        ),
    )
    with pytest.raises(ToolExecutionError):
        repo.create(request)

    escaped = UnifiedVaspRequest(
        workflow_kind=VaspWorkflowKind.PERIODIC,
        scientific_spec=PeriodicScientificSpec(
            structure_path=str(Path("..") / "outside.cif"),
            profile="standard_semiconductor",
        ),
    )
    with pytest.raises(ToolExecutionError):
        repo.create(escaped)


def molecular_request(path: str | Path) -> UnifiedVaspRequest:
    return UnifiedVaspRequest(
        workflow_kind=VaspWorkflowKind.MOLECULAR,
        scientific_spec=MolecularScientificSpec(workflow=WorkflowSpec(
            molecule=MoleculeSpec(
                name="X", structure_path=path, structure_kind="xyz", total_charge=0,
            ),
            stages=[StageSpec(name=StageName.RELAX)],
            scientific_method="PBE-D3(BJ)",
        )),
    )


def test_load_rejects_tampered_molecular_structure_paths(tmp_path):
    workspace = Workspace(tmp_path)
    repo = ManifestRepository(workspace)
    source = workspace.root / "molecule.xyz"
    source.write_text("1\nX\nX 0 0 0\n", encoding="utf-8")
    manifest = repo.create(molecular_request("molecule.xyz"))
    raw = json.loads(repo._manifest_path(manifest.workflow_id).read_text(encoding="utf-8"))
    external = tmp_path.parent / f"{tmp_path.name}-external.xyz"
    external.write_text("1\nX\nX 0 0 0\n", encoding="utf-8")
    escape_link = workspace.root / "escape.xyz"
    escape_link.symlink_to(external)
    try:
        for path in (
            str(workspace.root / "molecule.xyz"),
            str(external),
            "../molecule.xyz",
            escape_link.name,
        ):
            raw["scientific_spec"]["workflow"]["molecule"]["structure_path"] = path
            repo._manifest_path(manifest.workflow_id).write_text(
                json.dumps(raw), encoding="utf-8"
            )
            with pytest.raises(ToolExecutionError, match="molecular structure path"):
                repo.load(manifest.workflow_id)
    finally:
        escape_link.unlink(missing_ok=True)
        external.unlink(missing_ok=True)


def test_load_rejects_path_like_or_malformed_workflow_ids_without_creating_paths(
    tmp_path,
):
    workspace = Workspace(tmp_path)
    repo = ManifestRepository(workspace)
    absolute_escape = tmp_path.parent / f"{tmp_path.name}-absolute-workflow-escape"
    assert not absolute_escape.exists()
    cases = [
        (str(absolute_escape), absolute_escape),
        (
            "../vasp-manifest-parent-escape",
            workspace.root / ".photomatagent" / "vasp" / "vasp-manifest-parent-escape",
        ),
        ("", workspace.root / ".photomatagent" / "vasp" / "workflows"),
        (
            "nested/workflow",
            workspace.root / ".photomatagent" / "vasp" / "workflows" / "nested",
        ),
        (
            "not-a-generated-workflow",
            workspace.root
            / ".photomatagent"
            / "vasp"
            / "workflows"
            / "not-a-generated-workflow",
        ),
    ]

    try:
        for workflow_id, unexpected_directory in cases:
            with pytest.raises(ToolExecutionError, match="workflow ID"):
                repo.load(workflow_id)

            assert not unexpected_directory.exists()
    finally:
        if absolute_escape.exists():
            absolute_escape.rmdir()


def test_loading_unknown_valid_workflow_id_does_not_create_a_workflow_directory(tmp_path):
    workspace = Workspace(tmp_path)
    repo = ManifestRepository(workspace)
    workflow_id = "vasp_0123456789abcdef"

    with pytest.raises(KeyError, match="unknown workflow"):
        repo.load(workflow_id)

    assert not (
        workspace.root / ".photomatagent" / "vasp" / "workflows" / workflow_id
    ).exists()


def test_create_rejects_absolute_dotdot_and_external_symlink_source_references(tmp_path):
    workspace = Workspace(tmp_path)
    repo = ManifestRepository(workspace)
    inside_source = workspace.root / "inside.cif"
    inside_source.write_text("data_inside", encoding="utf-8")
    outside_source = tmp_path.parent / f"{tmp_path.name}-outside-source.cif"
    source_link = workspace.root / "outside-source-link.cif"
    assert not outside_source.exists()
    assert not source_link.exists()
    outside_source.write_text("data_outside", encoding="utf-8")
    source_link.symlink_to(outside_source)
    paths = [
        str(tmp_path.parent / f"{tmp_path.name}-absolute-source.cif"),
        str(inside_source),
        "../outside-source.cif",
        source_link.name,
    ]

    try:
        for source_path in paths:
            request = UnifiedVaspRequest(
                workflow_kind=VaspWorkflowKind.PERIODIC,
                scientific_spec=PeriodicScientificSpec(
                    structure_path=source_path,
                    profile="standard_semiconductor",
                ),
            )

            with pytest.raises(ToolExecutionError, match="VASP manifest source path"):
                repo.create(request)
    finally:
        source_link.unlink(missing_ok=True)
        outside_source.unlink(missing_ok=True)


def test_workflow_directory_symlink_to_another_workspace_location_is_rejected(tmp_path):
    workspace = Workspace(tmp_path)
    repo = ManifestRepository(workspace)
    redirected_directory = workspace.root / "redirected-workflows"
    workflows_link = workspace.root / ".photomatagent" / "vasp" / "workflows"
    redirected_workflow = redirected_directory / "vasp_0123456789abcdef"
    redirected_directory.mkdir()
    workflows_link.parent.mkdir(parents=True)
    workflows_link.symlink_to(redirected_directory, target_is_directory=True)

    try:
        with pytest.raises(ToolExecutionError, match="symlink"):
            repo.workflow_dir("vasp_0123456789abcdef")

        assert not redirected_workflow.exists()
    finally:
        workflows_link.unlink(missing_ok=True)
        if redirected_workflow.exists():
            redirected_workflow.rmdir()
        redirected_directory.rmdir()


def test_changed_source_structure_does_not_mutate_existing_snapshot(tmp_path):
    workspace = Workspace(tmp_path)
    repo = ManifestRepository(workspace)
    request = make_request(workspace)
    manifest = repo.create(request)
    snapshot = workspace.resolve(manifest.scientific_spec.structure_path)
    original_snapshot_text = snapshot.read_text(encoding="utf-8")

    original = workspace.root / "structure.cif"
    original.write_text("changed source after snapshot", encoding="utf-8")

    loaded = repo.load(manifest.workflow_id)
    reloaded_snapshot = workspace.resolve(loaded.scientific_spec.structure_path)
    assert reloaded_snapshot.read_text(encoding="utf-8") == original_snapshot_text
    assert scientific_fingerprint(loaded.scientific_spec) == loaded.scientific_fingerprint


def test_interrupted_temporary_output_leaves_previous_manifest_readable(tmp_path):
    workspace = Workspace(tmp_path)
    repo = ManifestRepository(workspace)
    manifest = repo.create(make_request(workspace))
    saved = repo.save(manifest, expected_revision=0)
    assert saved.revision == 1

    workflow_dir = repo.workflow_dir(manifest.workflow_id)
    (workflow_dir / ".manifest.json.tmp.deadbeef").write_text(
        "{}", encoding="utf-8"
    )

    loaded = repo.load(manifest.workflow_id)
    assert loaded.revision == 1
    assert loaded.workflow_id == manifest.workflow_id


def test_stale_revision_writes_raise_manifest_conflict(tmp_path):
    workspace = Workspace(tmp_path)
    repo = ManifestRepository(workspace)
    manifest = repo.create(make_request(workspace))

    first = repo.save(manifest, expected_revision=0)
    assert first.revision == 1

    with pytest.raises(ManifestConflictError):
        repo.save(manifest, expected_revision=0)


def test_unknown_schema_version_is_rejected(tmp_path):
    workspace = Workspace(tmp_path)
    repo = ManifestRepository(workspace)
    manifest = repo.create(make_request(workspace))
    path = repo._manifest_path(manifest.workflow_id)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["schema_version"] = "9.9"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported VASP manifest schema_version"):
        repo.load(manifest.workflow_id)


def test_current_schema_scientific_fingerprint_tampering_is_rejected(tmp_path):
    workspace = Workspace(tmp_path)
    repo = ManifestRepository(workspace)
    manifest = repo.create(make_request(workspace))
    path = repo._manifest_path(manifest.workflow_id)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["scientific_fingerprint"] = "0" * 64
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="scientific fingerprint"):
        repo.load(manifest.workflow_id)


def test_snapshot_bytes_and_recorded_hash_are_verified_on_load(tmp_path):
    workspace = Workspace(tmp_path)
    repo = ManifestRepository(workspace)
    manifest = repo.create(make_request(workspace))
    snapshot = repo.workflow_dir(manifest.workflow_id) / "source" / "structure.cif"
    snapshot.write_text("tampered snapshot", encoding="utf-8")

    with pytest.raises(ValueError, match="snapshot"):
        repo.load(manifest.workflow_id)

    # Restore source bytes but tamper the immutable snapshot inventory itself.
    snapshot.write_text("structure", encoding="utf-8")
    inventory = repo.workflow_dir(manifest.workflow_id) / "snapshot.json"
    raw = json.loads(inventory.read_text(encoding="utf-8"))
    raw["files"][0]["sha256"] = "0" * 64
    inventory.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot"):
        repo.load(manifest.workflow_id)


def test_explicit_legacy_manifest_migration(tmp_path):
    workspace = Workspace(tmp_path)
    repo = ManifestRepository(workspace)
    manifest = repo.create(make_request(workspace))
    raw = json.loads(repo._manifest_path(manifest.workflow_id).read_text(encoding="utf-8"))
    raw.pop("schema_version", None)

    path = repo._manifest_path(manifest.workflow_id)
    path.write_text(json.dumps(raw), encoding="utf-8")

    migrated = repo.load(manifest.workflow_id)
    assert migrated.schema_version == "2.0"
    assert migrated.workflow_id == manifest.workflow_id
    assert migrated.revision == 0


def test_legacy_migration_resets_privileged_execution_metadata_and_snapshots_sources(tmp_path):
    workspace = Workspace(tmp_path)
    repo = ManifestRepository(workspace)
    manifest = repo.create(make_request(workspace))
    path = repo._manifest_path(manifest.workflow_id)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.pop("schema_version", None)
    raw["scientific_spec"]["structure_path"] = "structure.cif"
    raw["state"] = "RUNNING"
    raw["execution_fingerprint"] = "e" * 64
    raw["decision_epoch"] = 9
    raw["events"] = [{"event_type": "submitted", "timestamp": "2026-01-01T00:00:00Z"}]
    raw["child_workflow_ids"] = {"child": "vasp_fedcba9876543210"}
    raw["stages"] = [{
        "name": "relax", "state": "SUBMITTED", "request_id": "old-request",
        "execution_fingerprint": "e" * 64, "decision_epoch": 4,
    }]
    path.write_text(json.dumps(raw), encoding="utf-8")

    migrated = repo.load(manifest.workflow_id)

    assert migrated.state.value == "PLANNED"
    assert migrated.execution_fingerprint is None
    assert migrated.decision_epoch == 0
    assert migrated.events == []
    assert migrated.child_workflow_ids == {}
    assert migrated.stages == []
    assert migrated.scientific_spec.structure_path.startswith(".photomatagent/vasp/workflows/")
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == "2.0"
    assert persisted["scientific_spec"]["structure_path"] == migrated.scientific_spec.structure_path
    assert repo.load(manifest.workflow_id) == migrated


@pytest.mark.parametrize("kind", ["molecular", "study"])
def test_legacy_migration_snapshots_contained_molecular_and_study_sources(tmp_path, kind):
    workspace = Workspace(tmp_path)
    repo = ManifestRepository(workspace)
    source = workspace.root / "source.xyz"
    source.write_text("1\nX\nX 0 0 0\n", encoding="utf-8")
    if kind == "molecular":
        request = molecular_request("source.xyz")
    else:
        request = UnifiedVaspRequest(
            workflow_kind=VaspWorkflowKind.STUDY,
            scientific_spec=StudyScientificSpec(request=VaspStudyRequest(
                study_id="legacy-study",
                systems=[StudySystem(system_id="x", structure_path="source.xyz", total_charge=0)],
            )),
        )
    manifest = repo.create(request)
    raw = json.loads(repo._manifest_path(manifest.workflow_id).read_text(encoding="utf-8"))
    raw.pop("schema_version", None)
    if kind == "molecular":
        raw["scientific_spec"]["workflow"]["molecule"]["structure_path"] = "source.xyz"
    else:
        raw["scientific_spec"]["request"]["systems"][0]["structure_path"] = "source.xyz"
    repo._manifest_path(manifest.workflow_id).write_text(json.dumps(raw), encoding="utf-8")

    migrated = repo.load(manifest.workflow_id)

    if kind == "molecular":
        snapshot = migrated.scientific_spec.workflow.molecule.structure_path
    else:
        snapshot = migrated.scientific_spec.request.systems[0].structure_path
    assert str(snapshot).startswith(".photomatagent/vasp/workflows/")


@pytest.mark.parametrize("source_path", ["../outside.xyz", "/tmp/outside.xyz"])
def test_legacy_migration_rejects_unsafe_source_references(tmp_path, source_path):
    workspace = Workspace(tmp_path)
    repo = ManifestRepository(workspace)
    manifest = repo.create(make_request(workspace))
    raw = json.loads(repo._manifest_path(manifest.workflow_id).read_text(encoding="utf-8"))
    raw.pop("schema_version", None)
    raw["scientific_spec"]["structure_path"] = source_path
    repo._manifest_path(manifest.workflow_id).write_text(json.dumps(raw), encoding="utf-8")

    legacy_text = json.dumps(raw)
    repo._manifest_path(manifest.workflow_id).write_text(legacy_text, encoding="utf-8")
    with pytest.raises(ToolExecutionError):
        repo.load(manifest.workflow_id)
    assert repo._manifest_path(manifest.workflow_id).read_text(encoding="utf-8") == legacy_text
