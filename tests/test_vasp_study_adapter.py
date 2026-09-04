"""Task 9: study adapter preserves orchestration over child workflows."""

from __future__ import annotations

from pathlib import Path
import json

import pytest

from photomatagent.scientific.applications.vasp.molecular.runtime import (
    MolecularVaspRuntime,
)
from photomatagent.scientific.applications.vasp.study.models import VaspStudyRequest
from photomatagent.scientific.applications.vasp.study.models import StudySystem
from photomatagent.scientific.applications.vasp.study.models import BindingGroup
from photomatagent.scientific.applications.vasp.unified.executors import (
	CollectionResult,
	OperationResult,
	PreflightResult,
	RecoveryResult,
	ReportResult,
	StatusResult,
    SubmissionResult,
    ServiceResult,
    VaspWorkflowExecutor,
)
from photomatagent.scientific.applications.vasp.unified.fingerprints import (
    scientific_fingerprint,
)
from photomatagent.scientific.applications.vasp.unified.models import (
    StudyScientificSpec,
    ReportKind,
    ReportRequest,
    UnifiedStage,
    UnifiedVaspRequest,
    UnifiedVaspManifest,
    VaspWorkflowKind,
    WorkflowState,
)
from photomatagent.scientific.applications.vasp.unified.approvals import ApprovalReceiptStore
from photomatagent.scientific.applications.vasp.unified.repository import ManifestRepository
from photomatagent.scientific.applications.vasp.unified.router import UnifiedVaspRouter
from photomatagent.scientific.applications.vasp.unified.service import UnifiedVaspService
from photomatagent.scientific.applications.vasp.unified.resources import AutomaticBudget, ResourceAuthorizationService
from photomatagent.scientific.remote.models import ResourcePolicy
from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.workspace import Workspace
from photomatagent.scientific.applications.vasp.unified.study import (
    VaspStudyExecutorAdapter,
)
from photomatagent.scientific.applications.vasp.unified.molecular import (
    MolecularVaspExecutorAdapter,
)
from photomatagent.scientific.remote.models import ResourceRequest
from photomatagent.scientific.remote.fake import FakeSCNetBackend
from photomatagent.scientific.remote.registry import JobRegistry


def make_manifest() -> UnifiedVaspManifest:
    spec = StudyScientificSpec(request=VaspStudyRequest(study_id="s1"))
    stages = [UnifiedStage(name="study")]
    return UnifiedVaspManifest(
        workflow_id="vasp_abcdef0123456789",
        workflow_kind=VaspWorkflowKind.STUDY,
        scientific_spec=spec,
        scientific_fingerprint=scientific_fingerprint(spec, stages),
        stages=stages,
    )


def manifest_with_tasks(
    tmp_path: Path, repository: ManifestRepository | None = None
) -> UnifiedVaspManifest:
    structure = tmp_path / "molecule.xyz"
    structure.write_text("1\nX\nX 0 0 0\n", encoding="utf-8")
    spec = StudyScientificSpec(
        request=VaspStudyRequest(
            study_id="s1",
            systems=[
                StudySystem(
                    system_id="x",
                    structure_path=structure,
                    total_charge=0,
                )
            ],
        )
    )
    if repository is not None:
        return repository.create(
            UnifiedVaspRequest(
                workflow_kind=VaspWorkflowKind.STUDY,
                scientific_spec=StudyScientificSpec(
                    request=spec.request.model_copy(
                        update={"systems": [
                            spec.request.systems[0].model_copy(
                                update={"structure_path": "molecule.xyz"}
                            )
                        ]}
                    )
                ),
            )
        )
    stages = [UnifiedStage(name="study")]
    return UnifiedVaspManifest(
        workflow_id="vasp_abcdef0123456789",
        workflow_kind=VaspWorkflowKind.STUDY,
        scientific_spec=spec,
        scientific_fingerprint=scientific_fingerprint(spec, stages),
        stages=stages,
    )


class FakeChildAdapter:
    def __init__(self, *, preflight_ok: bool = True) -> None:
        self.prepares = 0
        self.preflights = 0
        self.submits = 0
        self.preflight_ok = preflight_ok

    async def prepare(self, manifest):
        self.prepares += 1
        return OperationResult(ok=True)

    async def preflight(self, manifest):
        self.preflights += 1
        return PreflightResult(
            ok=self.preflight_ok,
            passed=self.preflight_ok,
            errors=[] if self.preflight_ok else ["child preflight failed"],
        )

    async def submit(self, manifest, stage, resource):
        self.submits += 1
        return SubmissionResult(
            ok=True,
            submitted=True,
            request_id=f"req-{manifest.workflow_id}",
            job_id="job-1",
        )

    async def status(self, manifest):
        return StatusResult(ok=True, stage_states={"relax": "RUNNING"})

    async def reconcile(self, manifest):
        return RecoveryResult(ok=True, action="AUTO_RESUME")

    async def collect(self, manifest):
        return CollectionResult(ok=True, validated=False)

    async def report(self, manifest, request):
        return ReportResult(ok=True, report_kind=request.kind)


class FakeChildService:
    def __init__(self, adapter: FakeChildAdapter) -> None:
        self.adapter = adapter
        self.manifests = {}
        self.next_id = 0

    def plan(self, request):
        self.next_id += 1
        manifest = make_manifest_with_workflow(request, f"vasp_{self.next_id:016x}")
        self.manifests[manifest.workflow_id] = manifest
        return manifest

    def load_manifest(self, workflow_id):
        return self.manifests[workflow_id]

    async def prepare(self, workflow_id):
        return await self.adapter.prepare(self.manifests[workflow_id])

    async def preflight(self, workflow_id):
        return await self.adapter.preflight(self.manifests[workflow_id])

    async def submit(self, workflow_id, stage=None):
        manifest = self.manifests[workflow_id]
        return await self.adapter.submit(manifest, manifest.stages[0], ResourceRequest())

    async def status(self, workflow_id):
        return await self.adapter.status(self.manifests[workflow_id])

    async def resume(self, workflow_id):
        return await self.adapter.reconcile(self.manifests[workflow_id])

    async def collect(self, workflow_id):
        return await self.adapter.collect(self.manifests[workflow_id])

    async def report(self, workflow_id, request):
        return await self.adapter.report(self.manifests[workflow_id], request)


class PersistedFakeMolecularExecutor:
    def __init__(self) -> None:
        self.submissions = 0

    async def prepare(self, manifest):
        return OperationResult(ok=True, data={"prepared": True})

    async def preflight(self, manifest):
        return PreflightResult(ok=True, passed=True)

    async def submit(self, manifest, stage, resource):
        self.submissions += 1
        return SubmissionResult(ok=True, submitted=True, request_id=f"req-{self.submissions}", job_id=str(self.submissions))

    async def status(self, manifest):
        return StatusResult(ok=True, stage_states={stage.name: "COMPLETED" for stage in manifest.stages})

    async def reconcile(self, manifest):
        return RecoveryResult(ok=True, action="AUTO_RESUME")

    async def collect(self, manifest):
        return CollectionResult(ok=True, validated=True, stage_states={stage.name: "VALIDATED" for stage in manifest.stages})

    async def report(self, manifest, request):
        return ReportResult(ok=True, report_kind=request.kind)


class RecordingLifecycleExecutor(PersistedFakeMolecularExecutor):
    """Local executor below the real unified service boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.stage_submissions: list[str] = []

    async def submit(self, manifest, stage, resource):
        self.stage_submissions.append(stage.name)
        return SubmissionResult(
            ok=True,
            submitted=True,
            request_id=f"request-{stage.name}-{len(self.stage_submissions)}",
            job_id=f"job-{len(self.stage_submissions)}",
            data={"authorized": resource.model_dump(mode="json")},
        )

    async def collect(self, manifest):
        submitted = self.stage_submissions
        states = {
            stage.name: ("VALIDATED" if stage.name in submitted else "PLANNED")
            for stage in manifest.stages
        }
        return CollectionResult(
            ok=True,
            validated=len(submitted) == len(manifest.stages),
            evidence=[ScientificEvidence(
                subject="X", property="total_energy", value=-1.0, unit="eV",
                source="local-fake", source_type="calculation", method="PBE",
                fidelity="dft",
            )],
            stage_states=states,
            evidence_gaps=[] if len(submitted) == len(manifest.stages) else ["next stage pending"],
        )


def persisted_child_service(tmp_path, executor):
    repo = ManifestRepository(Workspace(tmp_path))
    approvals = ApprovalReceiptStore(tmp_path / "approvals")
    resources = ResourceAuthorizationService(
        approvals,
        policy=ResourcePolicy(allow_hpc_submit=True),
        automatic_budget=AutomaticBudget(max_nodes=4, max_tasks_per_node=64, max_walltime_minutes=600),
    )
    return UnifiedVaspService(repo, approvals, UnifiedVaspRouter(molecular=executor), resources=resources)


def make_manifest_with_workflow(request, workflow_id):
    from photomatagent.scientific.applications.vasp.unified.models import MolecularScientificSpec
    return UnifiedVaspManifest(
        workflow_id=workflow_id,
        workflow_kind=VaspWorkflowKind.MOLECULAR,
        scientific_spec=MolecularScientificSpec(workflow=request.scientific_spec.workflow),
        scientific_fingerprint=scientific_fingerprint(request.scientific_spec),
        stages=[UnifiedStage(name=stage.name.value) for stage in request.scientific_spec.workflow.stages],
    )


def test_study_adapter_satisfies_executor_protocol(tmp_path):
    runtime = MolecularVaspRuntime(
        backend=None,
        application=None,
        configured=False,
        workflow_dir=tmp_path / "wf",
        log_dir=tmp_path / "logs",
        registry_path=tmp_path / "jobs.sqlite3",
    )
    adapter = VaspStudyExecutorAdapter(runtime, study_dir=tmp_path / "study")
    assert isinstance(adapter, VaspWorkflowExecutor)


@pytest.mark.asyncio
async def test_study_adapter_prepare_returns_typed_result(tmp_path):
    runtime = MolecularVaspRuntime(
        backend=None,
        application=None,
        configured=False,
        workflow_dir=tmp_path / "wf",
        log_dir=tmp_path / "logs",
        registry_path=tmp_path / "jobs.sqlite3",
    )
    adapter = VaspStudyExecutorAdapter(runtime, study_dir=tmp_path / "study")
    result = await adapter.prepare(make_manifest())
    assert result.ok is True
    assert "study_id" in result.data


@pytest.mark.asyncio
async def test_prepare_is_planning_only_and_persists_child_ids(tmp_path):
    runtime = MolecularVaspRuntime(
        backend=None,
        application=None,
        configured=False,
        workflow_dir=tmp_path / "wf",
        log_dir=tmp_path / "logs",
        registry_path=tmp_path / "jobs.sqlite3",
    )
    child = FakeChildAdapter()
    manifest = manifest_with_tasks(tmp_path)
    adapter = VaspStudyExecutorAdapter(
        runtime, study_dir=tmp_path / "study", child_service=FakeChildService(child)
    )

    result = await adapter.prepare(manifest)

    assert result.ok
    assert child.prepares == 1
    assert child.preflights == 0
    assert child.submits == 0
    assert manifest.child_workflow_ids
    assert (tmp_path / "study" / "study_state.json").is_file()


@pytest.mark.asyncio
async def test_preflight_aggregates_children_and_submit_delegates(tmp_path):
    runtime = MolecularVaspRuntime(
        backend=None,
        application=None,
        configured=False,
        workflow_dir=tmp_path / "wf",
        log_dir=tmp_path / "logs",
        registry_path=tmp_path / "jobs.sqlite3",
    )
    child = FakeChildAdapter()
    manifest = manifest_with_tasks(tmp_path)
    adapter = VaspStudyExecutorAdapter(
        runtime, study_dir=tmp_path / "study", child_service=FakeChildService(child)
    )

    await adapter.prepare(manifest)
    preflight = await adapter.preflight(manifest)
    submitted = await adapter.submit(
        manifest, UnifiedStage(name="study"), ResourceRequest()
    )

    assert preflight.ok and preflight.passed
    assert child.preflights == 1
    assert submitted.ok and submitted.submitted
    assert child.submits == 1


@pytest.mark.asyncio
async def test_restart_reuses_child_ids_and_does_not_submit_again(tmp_path):
    runtime = MolecularVaspRuntime(
        backend=None,
        application=None,
        configured=False,
        workflow_dir=tmp_path / "wf",
        log_dir=tmp_path / "logs",
        registry_path=tmp_path / "jobs.sqlite3",
    )
    parent_repo = ManifestRepository(Workspace(tmp_path))
    manifest = manifest_with_tasks(tmp_path, parent_repo)
    child_executor = PersistedFakeMolecularExecutor()
    service = persisted_child_service(tmp_path, child_executor)
    first = VaspStudyExecutorAdapter(runtime, study_dir=tmp_path / "study", child_service=service)
    await first.prepare(manifest)
    parent_repo.save(manifest, expected_revision=0)
    await first.preflight(manifest)
    await first.submit(manifest, UnifiedStage(name="study"), ResourceRequest())
    ids = dict(manifest.child_workflow_ids)
    assert child_executor.submissions == 1

    # A new service/adapter models process restart.  Parent and child
    # manifests are reloaded from the real repository; no in-memory cache is
    # reused.
    reloaded_parent = parent_repo.load(manifest.workflow_id)
    restarted_executor = PersistedFakeMolecularExecutor()
    restarted_service = persisted_child_service(tmp_path, restarted_executor)
    restarted = VaspStudyExecutorAdapter(runtime, study_dir=tmp_path / "study", child_service=restarted_service)
    await restarted.prepare(reloaded_parent)
    await restarted.submit(reloaded_parent, UnifiedStage(name="study"), ResourceRequest())

    assert reloaded_parent.child_workflow_ids == ids
    assert restarted_executor.submissions == 0


@pytest.mark.asyncio
async def test_real_parent_service_advances_persisted_child_stages_without_duplicate(tmp_path):
    runtime = MolecularVaspRuntime(
        backend=None, application=None, configured=False,
        workflow_dir=tmp_path / "wf", log_dir=tmp_path / "logs",
        registry_path=tmp_path / "jobs.sqlite3",
    )
    child_executor = RecordingLifecycleExecutor()
    child_service = persisted_child_service(tmp_path, child_executor)
    adapter = VaspStudyExecutorAdapter(
        runtime, study_dir=tmp_path / "study", child_service=child_service
    )
    parent_repo = ManifestRepository(Workspace(tmp_path))
    parent_service = UnifiedVaspService(
        parent_repo,
        ApprovalReceiptStore(tmp_path / "parent-approvals"),
        UnifiedVaspRouter(study=adapter),
        resources=child_service.resources,
    )
    structure = tmp_path / "molecule.xyz"
    structure.write_text("1\nX\nX 0 0 0\n", encoding="utf-8")
    parent = parent_service.plan(UnifiedVaspRequest(
        workflow_kind=VaspWorkflowKind.STUDY,
        scientific_spec=StudyScientificSpec(request=VaspStudyRequest(
            study_id="integration",
            systems=[StudySystem(system_id="x", structure_path="molecule.xyz", total_charge=0)],
        )),
    ))
    prepared = await parent_service.prepare(parent.workflow_id)
    assert prepared.ok
    preflighted = await parent_service.preflight(parent.workflow_id)
    assert preflighted.ok
    first = await parent_service.submit(parent.workflow_id)
    assert first.ok
    persisted_parent = parent_repo.load(parent.workflow_id)
    assert persisted_parent.study_task_states
    assert next(iter(persisted_parent.study_task_states.values()))["request_id"]
    child_id = persisted_parent.child_workflow_ids
    assert len(child_id) == 1
    child_workflow_id = next(iter(child_id.values()))
    assert child_executor.stage_submissions == ["relax"]

    await child_service.status(child_workflow_id)
    await child_service.collect(child_workflow_id)
    progressed = await parent_service.collect(parent.workflow_id)
    assert progressed.state is WorkflowState.PREFLIGHTED
    second = await parent_service.submit(parent.workflow_id)
    assert second.ok
    assert child_executor.stage_submissions == ["relax", "static_preconverge"]

    duplicate = await parent_service.submit(parent.workflow_id)
    assert duplicate.ok and duplicate.data.get("duplicate") is True
    assert child_executor.stage_submissions == ["relax", "static_preconverge"]


class OneFailedOneValidatedChildService(FakeChildService):
    """Two child results that must never validate their study parent."""

    async def collect(self, workflow_id):
        index = sorted(self.manifests).index(workflow_id)
        if index == 0:
            return ServiceResult(
                ok=False,
                workflow_id=workflow_id,
                state=WorkflowState.FAILED,
                errors=["deterministic child validation failure"],
                evidence_gaps=["child evidence invalid"],
            )
        return ServiceResult(
            ok=True,
            workflow_id=workflow_id,
            state=WorkflowState.VALIDATED,
            evidence=[ScientificEvidence(
                subject="X", property="total_energy", value=-1.0, unit="eV",
                source="fake", source_type="calculation", method="PBE", fidelity="dft",
            )],
        )


def two_task_manifest(tmp_path: Path) -> UnifiedVaspManifest:
    for name in ("first", "second"):
        (tmp_path / f"{name}.xyz").write_text("1\nX\nX 0 0 0\n", encoding="utf-8")
    spec = StudyScientificSpec(request=VaspStudyRequest(
        study_id="two-child-study",
        systems=[
            StudySystem(system_id="first", structure_path=Path("first.xyz"), total_charge=0),
            StudySystem(system_id="second", structure_path=Path("second.xyz"), total_charge=0),
        ],
    ))
    return UnifiedVaspManifest(
        workflow_id="vasp_abcdef0123456789",
        workflow_kind=VaspWorkflowKind.STUDY,
        scientific_spec=spec,
        scientific_fingerprint=scientific_fingerprint(spec),
        stages=[UnifiedStage(name="study")],
    )


@pytest.mark.asyncio
async def test_parent_collect_fails_closed_for_one_failed_and_one_validated_child(tmp_path):
    runtime = MolecularVaspRuntime(
        backend=None, application=None, configured=False,
        workflow_dir=tmp_path / "wf", log_dir=tmp_path / "logs",
        registry_path=tmp_path / "jobs.sqlite3",
    )
    child_service = OneFailedOneValidatedChildService(FakeChildAdapter())
    adapter = VaspStudyExecutorAdapter(
        runtime, study_dir=tmp_path / "study", child_service=child_service
    )
    parent_repo = ManifestRepository(Workspace(tmp_path))
    parent_service = UnifiedVaspService(
        parent_repo,
        ApprovalReceiptStore(tmp_path / "parent-approvals"),
        UnifiedVaspRouter(study=adapter),
        resources=ResourceAuthorizationService(
            ApprovalReceiptStore(tmp_path / "resource-approvals"),
            policy=ResourcePolicy(allow_hpc_submit=True),
        ),
    )
    parent = parent_service.plan(UnifiedVaspRequest(
        workflow_kind=VaspWorkflowKind.STUDY,
        scientific_spec=two_task_manifest(tmp_path).scientific_spec,
    ))
    assert (await parent_service.prepare(parent.workflow_id)).ok

    collected = await parent_service.collect(parent.workflow_id)
    persisted = parent_repo.load(parent.workflow_id)

    assert not collected.ok
    assert collected.state is WorkflowState.VALIDATION_FAILED
    assert persisted.state is WorkflowState.VALIDATION_FAILED
    assert any("deterministic child validation failure" in error for error in collected.errors)
    failed_tasks = [
        entry for entry in persisted.study_task_states.values()
        if entry["state"] == "FAILED"
    ]
    assert failed_tasks
    assert any("child evidence invalid" in gap for gap in collected.evidence_gaps)


@pytest.mark.asyncio
async def test_tampered_duplicate_child_id_fails_collect_and_persists_bounded_tasks(tmp_path):
    runtime = MolecularVaspRuntime(
        backend=None, application=None, configured=False,
        workflow_dir=tmp_path / "wf", log_dir=tmp_path / "logs",
        registry_path=tmp_path / "jobs.sqlite3",
    )
    manifest = two_task_manifest(tmp_path)
    adapter = VaspStudyExecutorAdapter(
        runtime, study_dir=tmp_path / "study", child_service=FakeChildService(FakeChildAdapter())
    )
    await adapter.prepare(manifest)
    task_ids = list(manifest.child_workflow_ids)
    manifest.child_workflow_ids[task_ids[1]] = manifest.child_workflow_ids[task_ids[0]]

    collected = await adapter.collect(manifest)

    assert not collected.ok and not collected.validated
    assert any("duplicate child workflow ID" in error for error in collected.errors)
    assert all(
        manifest.study_task_states[task_id]["state"] == "FAILED"
        for task_id in task_ids
    )
    assert all(
        state == WorkflowState.FAILED.value
        for state in collected.stage_states.values()
    )


@pytest.mark.asyncio
async def test_binding_rejects_tampered_complex_or_duplicate_fragment_child_ids(tmp_path):
    runtime = MolecularVaspRuntime(
        backend=None, application=None, configured=False,
        workflow_dir=tmp_path / "wf", log_dir=tmp_path / "logs",
        registry_path=tmp_path / "jobs.sqlite3",
    )
    manifest = two_task_manifest(tmp_path)
    service = FakeChildService(FakeChildAdapter())
    adapter = VaspStudyExecutorAdapter(runtime, study_dir=tmp_path / "study", child_service=service)
    await adapter.prepare(manifest)
    spec = adapter._planned_spec
    assert spec is not None
    task_ids = list(manifest.child_workflow_ids)
    spec.calculation_matrix.binding_groups = [
        BindingGroup(
            complex_task_id=task_ids[0],
            fragment_task_ids=[task_ids[1], task_ids[1]],
            label="tampered",
            total_charge=0,
        )
    ]
    for child in service.manifests.values():
        child.state = WorkflowState.VALIDATED
    adapter._plan = lambda _: spec  # type: ignore[method-assign]

    result = await adapter.report(
        manifest, ReportRequest(kind=ReportKind.BINDING_ENERGY)
    )

    assert not result.ok
    assert any("duplicate" in error for error in result.errors)


@pytest.mark.asyncio
async def test_manifest_task_state_is_reapplied_after_memory_is_discarded(tmp_path):
    runtime = MolecularVaspRuntime(
        backend=None, application=None, configured=False,
        workflow_dir=tmp_path / "wf", log_dir=tmp_path / "logs",
        registry_path=tmp_path / "jobs.sqlite3",
    )
    parent_repo = ManifestRepository(Workspace(tmp_path))
    child_service = persisted_child_service(tmp_path, PersistedFakeMolecularExecutor())
    adapter = VaspStudyExecutorAdapter(runtime, study_dir=tmp_path / "study", child_service=child_service)
    (tmp_path / "molecule.xyz").write_text("1\nX\nX 0 0 0\n", encoding="utf-8")
    parent = parent_repo.create(UnifiedVaspRequest(
        workflow_kind=VaspWorkflowKind.STUDY,
        scientific_spec=StudyScientificSpec(request=VaspStudyRequest(
            study_id="recover-progress",
            systems=[StudySystem(system_id="x", structure_path=Path("molecule.xyz"), total_charge=0)],
        )),
    ))
    await adapter.prepare(parent)
    task_id = next(iter(parent.study_task_states))
    parent.study_task_states[task_id]["state"] = "FAILED"
    parent.study_task_states[task_id]["request_id"] = "persisted-request"
    parent_repo.save(parent, expected_revision=parent.revision)

    reloaded = parent_repo.load(parent.workflow_id)
    restarted = VaspStudyExecutorAdapter(runtime, study_dir=tmp_path / "study", child_service=child_service)
    spec = restarted._plan(reloaded)
    task = next(item for item in spec.calculation_matrix.tasks if item.task_id == task_id)

    assert task.state == "FAILED"
    assert task.request_id == "persisted-request"


def test_study_derived_state_rejects_symlink_parent_and_target(tmp_path):
    runtime = MolecularVaspRuntime(
        backend=None, application=None, configured=False,
        workflow_dir=tmp_path / "wf", log_dir=tmp_path / "logs",
        registry_path=tmp_path / "jobs.sqlite3",
    )
    adapter = VaspStudyExecutorAdapter(runtime, study_dir=tmp_path / "study")
    target = tmp_path / "redirect"
    target.mkdir()
    (tmp_path / "study" / "children").rmdir()
    (tmp_path / "study").rmdir()
    (tmp_path / "study").symlink_to(target, target_is_directory=True)
    with pytest.raises(Exception, match="symlink"):
        adapter._atomic_write_state({"tasks": {}})
    (tmp_path / "study").unlink()
    (tmp_path / "study").mkdir()
    (tmp_path / "study" / "study_state.json").symlink_to(target / "state.json")
    with pytest.raises(Exception, match="symlink"):
        adapter._atomic_write_state({"tasks": {}})


@pytest.mark.asyncio
async def test_study_submission_uses_real_molecular_submit_once_chain_once(tmp_path):
    """The study path reaches the real molecular facade and registry once."""
    structure = tmp_path / "h2.xyz"
    structure.write_text("2\nH2\nH 0 0 0\nH 0 0 0.74\n", encoding="utf-8")
    for element, zval in (("H", 1.0), ("C", 4.0)):
        psp = tmp_path / "psp" / element
        psp.mkdir(parents=True)
        psp.joinpath("POTCAR").write_text(
            f"TITEL = PAW_PBE {element}\nPOMASS = 1; ZVAL = {zval:.3f}\nENMAX = 250.000 eV\n",
            encoding="utf-8",
        )
    backend = FakeSCNetBackend(policy=ResourcePolicy(allow_hpc_submit=True), strict=True)
    registry = JobRegistry(tmp_path / "state" / "jobs.sqlite3")
    runtime = MolecularVaspRuntime(
        backend=backend, configured=True, psp_dir=tmp_path / "psp",
        workflow_dir=tmp_path / "molecular", log_dir=tmp_path / "logs",
        registry_path=tmp_path / "state" / "jobs.sqlite3",
        module_name="vasp-5.4.4", remote_psp_dir="~/photomatagent/psp",
    )
    child_adapter = MolecularVaspExecutorAdapter(
        runtime, workflow_dir=tmp_path / "molecular"
    )
    approvals = ApprovalReceiptStore(tmp_path / "approvals")
    resources = ResourceAuthorizationService(
        approvals, policy=ResourcePolicy(allow_hpc_submit=True),
        automatic_budget=AutomaticBudget(max_nodes=4, max_tasks_per_node=64, max_walltime_minutes=600),
    )
    child_service = UnifiedVaspService(
        ManifestRepository(Workspace(tmp_path)), approvals,
        UnifiedVaspRouter(molecular=child_adapter), resources=resources,
    )
    study_adapter = VaspStudyExecutorAdapter(
        runtime, study_dir=tmp_path / "study", child_service=child_service
    )
    parent_service = UnifiedVaspService(
        ManifestRepository(Workspace(tmp_path)), ApprovalReceiptStore(tmp_path / "parent-approvals"),
        UnifiedVaspRouter(study=study_adapter), resources=resources,
    )
    parent = parent_service.plan(UnifiedVaspRequest(
        workflow_kind=VaspWorkflowKind.STUDY,
        scientific_spec=StudyScientificSpec(request=VaspStudyRequest(
            study_id="real-chain",
            systems=[StudySystem(system_id="h2", structure_path=Path("h2.xyz"), total_charge=0)],
        )),
    ))
    prepared = await parent_service.prepare(parent.workflow_id)
    assert prepared.ok, prepared.errors
    preflighted = await parent_service.preflight(parent.workflow_id)
    assert preflighted.ok, preflighted.model_dump()
    first = await parent_service.submit(parent.workflow_id)
    duplicate = await parent_service.submit(parent.workflow_id)

    assert first.ok and first.data["submitted_tasks"]
    assert duplicate.ok and duplicate.data["duplicate"] is True
    assert len(backend.submitted_scripts) == 1
    assert len(registry.list()) == 1
    record = registry.list()[0]
    assert record.request_id == next(
        entry["request_id"]
        for entry in parent_service.repository.load(parent.workflow_id).study_task_states.values()
    )
    assert record.resource.nodes >= 1
    assert record.resource.tasks_per_node >= 1

@pytest.mark.asyncio
async def test_budget_exhaustion_preserves_partial_plan_without_child_submission(tmp_path):
    runtime = MolecularVaspRuntime(
        backend=None,
        application=None,
        configured=False,
        workflow_dir=tmp_path / "wf",
        log_dir=tmp_path / "logs",
        registry_path=tmp_path / "jobs.sqlite3",
    )
    child = FakeChildAdapter()
    manifest = manifest_with_tasks(tmp_path)
    manifest.scientific_spec.request.resource_budget.max_core_hours = 0
    adapter = VaspStudyExecutorAdapter(runtime, study_dir=tmp_path / "study", child_service=FakeChildService(child))

    await adapter.prepare(manifest)
    result = await adapter.submit(manifest, UnifiedStage(name="study"), ResourceRequest())

    assert result.ok
    assert child.submits == 0
    state = json.loads((tmp_path / "study" / "study_state.json").read_text())
    assert state["tasks"]
