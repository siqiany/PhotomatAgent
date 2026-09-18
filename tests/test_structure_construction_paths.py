import json
import os
import shutil
from pathlib import Path

import pytest

from photomatagent.scientific.capabilities.structure.artifacts import (
    StructureArtifactError,
    publish_structures,
    resolve_structure_input,
    structure_hash,
)
from photomatagent.scientific.capabilities.structure.construction_models import (
    SiteReplacement,
    SubstitutionRequest,
    SupercellRequest,
)
from pymatgen.core import Lattice, Structure
from photomatagent.workspace import Workspace


@pytest.mark.parametrize("path", ["/tmp/outside.cif", "../outside.cif", "a/../b.cif"])
def test_construction_rejects_absolute_and_parent_paths(tmp_path, path):
    with pytest.raises(ValueError):
        resolve_structure_input(Workspace(tmp_path), path)


def test_construction_rejects_directory_and_symlink_escape(tmp_path):
    workspace = Workspace(tmp_path)
    (tmp_path / "dir").mkdir()
    with pytest.raises(ValueError):
        resolve_structure_input(workspace, "dir")
    outside = tmp_path.parent / "outside.cif"
    outside.write_text("bad")
    os.symlink(outside, tmp_path / "linked.cif")
    with pytest.raises(ValueError):
        resolve_structure_input(workspace, "linked.cif")


def test_publish_is_atomic_and_reuses_complete_operation(tmp_path):
    workspace = Workspace(tmp_path)
    structure = Structure(Lattice.cubic(4), ["Si"], [[0, 0, 0]])
    request = SupercellRequest(path="input.cif", scaling=(1, 1, 1), task_slug="task")
    first = publish_structures(workspace, request, "a" * 64, [structure])
    assert len(first) == 1
    second = publish_structures(workspace, request, "a" * 64, [structure])
    assert second[0].structure_hash == first[0].structure_hash
    output = workspace.root / first[0].output_path
    output.unlink()
    with pytest.raises(StructureArtifactError) as exc:
        publish_structures(workspace, request, "a" * 64, [structure])
    assert exc.value.code == "ARTIFACT_INCOMPLETE"

def _structure():
    return Structure(Lattice.cubic(4), ["Si"], [[0, 0, 0]])


def test_publish_rejects_output_and_atom_limits(tmp_path):
    workspace = Workspace(tmp_path)
    request = SupercellRequest(path="i.cif", scaling=(1, 1, 1), task_slug="t")
    with pytest.raises(StructureArtifactError):
        publish_structures(workspace, request, "a" * 64, [_structure()] * 33)
    large = Structure(Lattice.cubic(4), ["Si"] * 513, [[0,0,0]] * 513)
    with pytest.raises(StructureArtifactError):
        publish_structures(workspace, request, "b" * 64, [large])

def test_expected_formula_mismatch_leaves_no_destination(tmp_path):
    workspace = Workspace(tmp_path)
    request = SubstitutionRequest(
        path="i.cif",
        replacements=[SiteReplacement(index=0, from_element="Si", to_element="Ge")],
        expected_formula="Na1",
        hypothesis_id="h",
        task_slug="t",
    )
    with pytest.raises(StructureArtifactError):
        publish_structures(workspace, request, "a" * 64, [_structure()])
    assert not list((tmp_path / "user_output").rglob("manifest.json"))

def test_manifest_tamper_and_cif_tamper_are_conflicts(tmp_path):
    workspace = Workspace(tmp_path)
    request = SupercellRequest(path="i.cif", scaling=(1, 1, 1), task_slug="t")
    record = publish_structures(workspace, request, "a" * 64, [_structure()])[0]
    directory = (tmp_path / record.output_path).parent
    manifest = directory / "manifest.json"
    payload = json.loads(manifest.read_text())
    payload["input_sha256"] = "b" * 64
    manifest.write_text(json.dumps(payload))
    with pytest.raises(StructureArtifactError) as exc:
        publish_structures(workspace, request, "a" * 64, [_structure()])
    assert exc.value.code == "ARTIFACT_CONFLICT"
    manifest.write_text(json.dumps({**payload, "input_sha256": "a" * 64}))
    (directory / "structure_0000.cif").write_text("garbage")
    with pytest.raises(StructureArtifactError) as exc: publish_structures(workspace, request, "a"*64, [_structure()])
    assert exc.value.code == "ARTIFACT_CONFLICT"

def test_preexisting_directories_and_lock_are_untouched(tmp_path):
    workspace = Workspace(tmp_path)
    request = SupercellRequest(path="i.cif", scaling=(1, 1, 1), task_slug="t")
    from photomatagent.scientific.capabilities.structure.artifacts import operation_id
    op = operation_id(request, "a" * 64)
    destination = tmp_path / "user_output" / "t" / "structures" / op
    destination.mkdir(parents=True)
    (destination / "keep").write_text("x")
    with pytest.raises(StructureArtifactError):
        publish_structures(workspace, request, "a" * 64, [_structure()])
    assert (destination / "keep").read_text() == "x"
    shutil.rmtree(destination)
    lock = destination.parent / f".{op}.lock"
    lock.mkdir(parents=True)
    with pytest.raises(StructureArtifactError) as exc:
        publish_structures(workspace, request, "a" * 64, [_structure()])
    assert exc.value.code == "ARTIFACT_CONFLICT" and lock.is_dir()

def test_destination_appearing_at_publish_is_not_replaced(tmp_path, monkeypatch):
    import photomatagent.scientific.capabilities.structure.artifacts as artifacts
    workspace = Workspace(tmp_path)
    request = SupercellRequest(path="i.cif", scaling=(1, 1, 1), task_slug="race")
    operation = artifacts.operation_id(request, "a" * 64)
    destination = tmp_path / "user_output" / "race" / "structures" / operation

    def race(_stage, target):
        destination.mkdir(parents=True)
        (destination / "keep").write_text("preserve")
        raise artifacts.StructureArtifactError(artifacts.ARTIFACT_CONFLICT, "injected race")

    monkeypatch.setattr(artifacts, "_rename_noreplace", race)
    with pytest.raises(artifacts.StructureArtifactError) as exc:
        publish_structures(workspace, request, "a" * 64, [_structure()])
    assert exc.value.code == "ARTIFACT_CONFLICT"
    assert (destination / "keep").read_text() == "preserve"
    assert not (destination / "manifest.json").exists()
    assert not list((tmp_path / "tmp").glob("structures-*"))
    assert not (destination.parent / f".{operation}.lock").exists()

def test_manifest_structure_matcher_tamper_is_conflict(tmp_path):
    import photomatagent.scientific.capabilities.structure.artifacts as artifacts
    workspace = Workspace(tmp_path)
    request = SupercellRequest(path="i.cif", scaling=(1, 1, 1), task_slug="matcher")
    record = publish_structures(workspace, request, "a" * 64, [_structure()])[0]
    manifest = (tmp_path / record.output_path).parent / "manifest.json"
    payload = json.loads(manifest.read_text())
    payload["structure_matcher"] = {"ltol": 0.2}
    manifest.write_text(json.dumps(payload))
    with pytest.raises(artifacts.StructureArtifactError) as exc:
        publish_structures(workspace, request, "a" * 64, [_structure()])
    assert exc.value.code == "ARTIFACT_CONFLICT"

def test_rename_noreplace_preserves_existing_destination(tmp_path):
    import photomatagent.scientific.capabilities.structure.artifacts as artifacts
    source = tmp_path / "stage"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (destination / "keep").write_text("preserve")
    with pytest.raises(artifacts.StructureArtifactError) as exc:
        artifacts._rename_noreplace(source, destination)
    assert exc.value.code == "ARTIFACT_CONFLICT"
    assert (destination / "keep").read_text() == "preserve"
    assert source.is_dir()

def test_different_parameters_have_different_operation_ids(tmp_path):
    from photomatagent.scientific.capabilities.structure.artifacts import operation_id
    assert operation_id(SupercellRequest(path="i.cif", scaling=(1,1,1), task_slug="t"), "a"*64) != operation_id(SupercellRequest(path="i.cif", scaling=(2,1,1), task_slug="t"), "a"*64)

def test_output_parent_symlink_escape_does_not_write_outside(tmp_path):
    workspace = Workspace(tmp_path)
    outside = tmp_path.parent / "outside-structures"
    outside.mkdir()
    (tmp_path / "user_output" / "evil").symlink_to(outside, target_is_directory=True)
    request = SupercellRequest(path="i.cif", scaling=(1, 1, 1), task_slug="evil")
    with pytest.raises(ValueError):
        publish_structures(workspace, request, "a" * 64, [_structure()])
    assert list(outside.iterdir()) == []

def test_oversize_sparse_input_is_rejected(tmp_path):
    workspace = Workspace(tmp_path)
    path = tmp_path / "large.cif"
    with path.open("wb") as handle: handle.truncate(10 * 1024 * 1024 + 1)
    with pytest.raises(StructureArtifactError) as exc:
        resolve_structure_input(workspace, "large.cif")
    assert exc.value.code == "INPUT_TOO_LARGE"

def test_corrupt_cif_is_rejected(tmp_path):
    workspace = Workspace(tmp_path)
    (tmp_path / "bad.cif").write_text("not a cif")
    from photomatagent.scientific.capabilities.structure.artifacts import load_structure_input
    with pytest.raises(StructureArtifactError) as exc:
        load_structure_input(workspace, "bad.cif")
    assert exc.value.code == "INVALID_STRUCTURE"

def test_write_failure_cleans_stage_and_publishes_no_manifest(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path)
    request = SupercellRequest(path="i.cif", scaling=(1, 1, 1), task_slug="write-failure")
    original = Structure.to
    def fail_once(self, *args, **kwargs):
        raise OSError("injected write failure")
    monkeypatch.setattr(Structure, "to", fail_once)
    with pytest.raises(OSError):
        publish_structures(workspace, request, "a" * 64, [_structure()])
    assert not list((tmp_path / "user_output").rglob("manifest.json"))
    assert not list((tmp_path / "tmp").glob("structures-*"))
    monkeypatch.setattr(Structure, "to", original)

def test_input_structure_is_not_modified_by_publish(tmp_path):
    workspace = Workspace(tmp_path)
    request = SupercellRequest(path="i.cif", scaling=(1, 1, 1), task_slug="unchanged")
    structure = _structure()
    before = structure_hash(structure)
    publish_structures(workspace, request, "a"*64, [structure])
    assert structure_hash(structure) == before and len(structure) == 1
