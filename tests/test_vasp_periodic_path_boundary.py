"""Workspace-boundary coverage for persisted periodic manifests."""

from __future__ import annotations

from pathlib import Path

import pytest

from photomatagent.errors import ToolExecutionError
from photomatagent.scientific.applications.vasp.application import VaspApplication
from photomatagent.scientific.applications.vasp.unified.fingerprints import (
    scientific_fingerprint,
)
from photomatagent.scientific.applications.vasp.unified.models import (
    PeriodicScientificSpec,
    UnifiedStage,
    UnifiedVaspManifest,
    VaspWorkflowKind,
)
from photomatagent.scientific.applications.vasp.unified.periodic import (
    PeriodicVaspExecutor,
)
from photomatagent.scientific.remote.fake import FakeSCNetBackend


def test_tampered_periodic_manifest_structure_path_outside_workspace_is_rejected(
    tmp_path,
):
    outside_structure = tmp_path.parent / f"{tmp_path.name}-tampered-structure.cif"
    assert not outside_structure.exists()
    outside_structure.write_text("data_outside", encoding="utf-8")
    spec = PeriodicScientificSpec(
        structure_path=str(outside_structure),
        profile="standard_semiconductor",
    )
    manifest = UnifiedVaspManifest(
        workflow_id="vasp_0123456789abcdef",
        workflow_kind=VaspWorkflowKind.PERIODIC,
        scientific_spec=spec,
        scientific_fingerprint=scientific_fingerprint(spec),
    )
    executor = PeriodicVaspExecutor.__new__(PeriodicVaspExecutor)
    executor.application = VaspApplication(workspace=tmp_path)

    try:
        with pytest.raises(ToolExecutionError, match="workspace-relative"):
            executor._workspace_path(manifest.scientific_spec.structure_path)
    finally:
        outside_structure.unlink(missing_ok=True)


def test_tampered_periodic_manifest_workflow_id_cannot_escape_managed_paths(tmp_path):
    spec = PeriodicScientificSpec(
        structure_path="structure.cif",
        profile="standard_semiconductor",
    )
    manifest = UnifiedVaspManifest(
        workflow_id="../../../../periodic-escape",
        workflow_kind=VaspWorkflowKind.PERIODIC,
        scientific_spec=spec,
        scientific_fingerprint=scientific_fingerprint(spec),
    )
    executor = PeriodicVaspExecutor.__new__(PeriodicVaspExecutor)
    executor.application = VaspApplication(workspace=tmp_path)

    with pytest.raises(ToolExecutionError, match="workflow ID"):
        executor._stage_root(manifest)


def test_periodic_managed_paths_reject_internal_workflow_directory_symlinks(tmp_path):
    redirected_directory = tmp_path / "redirected-workflows"
    workflows_link = tmp_path / ".photomatagent" / "vasp" / "workflows"
    redirected_directory.mkdir()
    workflows_link.parent.mkdir(parents=True)
    workflows_link.symlink_to(redirected_directory, target_is_directory=True)
    executor = PeriodicVaspExecutor.__new__(PeriodicVaspExecutor)
    executor.application = VaspApplication(workspace=tmp_path)

    try:
        with pytest.raises(ToolExecutionError, match="symlink"):
            executor._managed_workflow_path(
                "workflows", "vasp_0123456789abcdef", "inputs"
            )

        assert not (redirected_directory / "vasp_0123456789abcdef").exists()
    finally:
        workflows_link.unlink(missing_ok=True)
        redirected_directory.rmdir()


def test_tampered_periodic_manifest_absolute_contained_structure_path_is_rejected(
    tmp_path,
):
    contained_structure = tmp_path / "tampered-structure.cif"
    spec = PeriodicScientificSpec(
        structure_path=str(contained_structure),
        profile="standard_semiconductor",
    )
    manifest = UnifiedVaspManifest(
        workflow_id="vasp_0123456789abcdef",
        workflow_kind=VaspWorkflowKind.PERIODIC,
        scientific_spec=spec,
        scientific_fingerprint=scientific_fingerprint(spec),
    )
    executor = PeriodicVaspExecutor.__new__(PeriodicVaspExecutor)
    executor.application = VaspApplication(workspace=tmp_path)

    with pytest.raises(ToolExecutionError, match="workspace-relative"):
        executor._workspace_path(manifest.scientific_spec.structure_path)


def test_tampered_periodic_manifest_structure_path_with_traversal_is_rejected(tmp_path):
    spec = PeriodicScientificSpec(
        structure_path="../tampered-structure.cif",
        profile="standard_semiconductor",
    )
    manifest = UnifiedVaspManifest(
        workflow_id="vasp_0123456789abcdef",
        workflow_kind=VaspWorkflowKind.PERIODIC,
        scientific_spec=spec,
        scientific_fingerprint=scientific_fingerprint(spec),
    )
    executor = PeriodicVaspExecutor.__new__(PeriodicVaspExecutor)
    executor.application = VaspApplication(workspace=tmp_path)

    with pytest.raises(ToolExecutionError, match="workspace-relative"):
        executor._workspace_path(manifest.scientific_spec.structure_path)


@pytest.mark.parametrize(
    ("link_parts", "registry_parent_parts"),
    [
        ((".photomatagent",), ("vasp",)),
        ((".photomatagent", "vasp"), ()),
    ],
)
def test_periodic_constructor_rejects_managed_root_symlink_without_external_state(
    tmp_path, link_parts, registry_parent_parts
):
    redirected_directory = tmp_path.parent / (
        f"{tmp_path.name}-constructor-redirect-{'-'.join(link_parts)}"
    )
    redirected_directory.mkdir()
    managed_link = tmp_path.joinpath(*link_parts)
    managed_link.parent.mkdir(parents=True, exist_ok=True)
    managed_link.symlink_to(redirected_directory, target_is_directory=True)
    app = VaspApplication(FakeSCNetBackend(), workspace=tmp_path)
    executor = None
    registry_path = redirected_directory.joinpath(
        *registry_parent_parts, "jobs.sqlite3"
    )

    try:
        with pytest.raises(ToolExecutionError, match="symlink"):
            executor = PeriodicVaspExecutor(app)

        assert not registry_path.exists()
    finally:
        if executor is not None:
            executor.session.registry.close()
        for suffix in ("", "-shm", "-wal"):
            (registry_path.parent / f"{registry_path.name}{suffix}").unlink(
                missing_ok=True
            )
        markers = registry_path.parent / "markers"
        if markers.exists():
            markers.rmdir()
        if (
            registry_path.parent != redirected_directory
            and registry_path.parent.exists()
        ):
            registry_path.parent.rmdir()
        managed_link.unlink(missing_ok=True)
        redirected_directory.rmdir()


@pytest.mark.asyncio
async def test_prepare_rejects_managed_path_redirected_during_application_call(
    tmp_path, monkeypatch
):
    (tmp_path / "structure.cif").write_text("data_structure", encoding="utf-8")
    manifest = UnifiedVaspManifest(
        workflow_id="vasp_0123456789abcdef",
        workflow_kind=VaspWorkflowKind.PERIODIC,
        scientific_spec=PeriodicScientificSpec(
            structure_path="structure.cif", profile="standard_semiconductor"
        ),
        scientific_fingerprint="test-fingerprint",
    )
    redirected_directory = tmp_path.parent / f"{tmp_path.name}-prepare-redirect"
    workflow_root = tmp_path / ".photomatagent" / "vasp" / "workflows"
    redirected_directory.mkdir()
    workflow_root.mkdir(parents=True)
    app = VaspApplication(workspace=tmp_path)
    executor = PeriodicVaspExecutor.__new__(PeriodicVaspExecutor)
    executor.application = app

    def redirecting_prepare_inputs(**_kwargs):
        workflow_root.rmdir()
        workflow_root.symlink_to(redirected_directory, target_is_directory=True)
        return {"stages": []}

    monkeypatch.setattr(app, "prepare_inputs", redirecting_prepare_inputs)
    try:
        result = await executor.prepare(manifest)

        assert not result.ok
        assert any("symlink" in error for error in result.errors)
    finally:
        if workflow_root.is_symlink():
            workflow_root.unlink()
        elif workflow_root.exists():
            workflow_root.rmdir()
        if redirected_directory.exists():
            redirected_directory.rmdir()


@pytest.mark.asyncio
async def test_collect_rejects_result_path_redirected_during_application_call(
    tmp_path, monkeypatch
):
    (tmp_path / "structure.cif").write_text("data_structure", encoding="utf-8")
    manifest = UnifiedVaspManifest(
        workflow_id="vasp_0123456789abcdef",
        workflow_kind=VaspWorkflowKind.PERIODIC,
        scientific_spec=PeriodicScientificSpec(
            structure_path="structure.cif", profile="standard_semiconductor"
        ),
        scientific_fingerprint="test-fingerprint",
        stages=[UnifiedStage(name="relax")],
    )
    redirected_directory = tmp_path.parent / f"{tmp_path.name}-collect-redirect"
    results_root = tmp_path / ".photomatagent" / "vasp" / "results"
    redirected_directory.mkdir()
    results_root.mkdir(parents=True)
    app = VaspApplication(workspace=tmp_path)
    executor = PeriodicVaspExecutor.__new__(PeriodicVaspExecutor)
    executor.application = app

    class Registry:
        @staticmethod
        def get(_request_id):
            return type("Record", (), {"job_id": "123", "remote_directory": "remote"})()

    executor.session = type("Session", (), {"registry": Registry()})()

    async def redirecting_collect(**_kwargs):
        results_root.rmdir()
        results_root.symlink_to(redirected_directory, target_is_directory=True)
        return {"scientifically_valid": False, "validation_problems": ["invalid"]}

    monkeypatch.setattr(app, "collect", redirecting_collect)
    try:
        result = await executor.collect(manifest)

        assert not result.ok
        assert any("symlink" in gap for gap in result.evidence_gaps)
    finally:
        results_root.unlink(missing_ok=True)
        redirected_directory.rmdir()
