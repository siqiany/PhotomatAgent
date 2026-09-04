"""Unified study adapter.

Studies only orchestrate typed molecular child workflows. Planning is local
and persisted; child lifecycle calls are delegated to the injected executor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, cast

from photomatagent.scientific.applications.vasp.molecular.runtime import MolecularVaspRuntime
from photomatagent.scientific.applications.vasp.study.executor import StudyExecutor
from photomatagent.scientific.applications.vasp.study.models import StudyTaskState, VaspStudySpec
from photomatagent.scientific.applications.vasp.unified.executors import (
    CollectionResult,
    OperationResult,
    PreflightResult,
    RecoveryResult,
    ReportResult,
    StatusResult,
    SubmissionResult,
)
from photomatagent.scientific.applications.vasp.unified.models import (
    MolecularScientificSpec,
    ReportRequest,
    StudyScientificSpec,
    UnifiedStage,
    UnifiedVaspManifest,
    UnifiedVaspRequest,
    VaspWorkflowKind,
    WorkflowState,
)
from photomatagent.scientific.applications.vasp.unified.molecular import MolecularVaspExecutorAdapter
from photomatagent.scientific.applications.vasp.unified.service import UnifiedVaspService
from photomatagent.scientific.applications.vasp.unified.repository import ManifestRepository
from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.remote.models import ResourceRequest
from photomatagent.workspace import Workspace


class UnifiedChildLifecycle(Protocol):
    """Narrow service contract used by the study orchestrator.

    The service owns repositories, resource decisions, approvals, and stage
    transitions.  A study never receives an executor or ResourceRequest for a
    child directly.
    """

    def plan(self, request: UnifiedVaspRequest) -> UnifiedVaspManifest: ...
    def load_manifest(self, workflow_id: str) -> UnifiedVaspManifest: ...
    async def prepare(self, workflow_id: str) -> Any: ...
    async def preflight(self, workflow_id: str) -> Any: ...
    async def submit(self, workflow_id: str, stage: str | None = None) -> Any: ...
    async def status(self, workflow_id: str) -> Any: ...
    async def resume(self, workflow_id: str) -> Any: ...
    async def collect(self, workflow_id: str) -> Any: ...
    async def report(self, workflow_id: str, request: ReportRequest) -> Any: ...


class VaspStudyExecutorAdapter:
    """Orchestrate child adapters without invoking legacy study execution."""

    def __init__(
        self,
        runtime: MolecularVaspRuntime,
        study_dir: str | Path | None = None,
        *,
        child_service: UnifiedChildLifecycle | None = None,
    ) -> None:
        self.runtime = runtime
        configured_root = (
            Path(study_dir).expanduser()
            if study_dir is not None
            else (
                runtime.workflow_dir.parent / "study_unified"
                if runtime.workflow_dir is not None
                else Path.cwd() / "output" / "vasp_study_unified"
            )
        )
        self._assert_safe_root(configured_root)
        authority = getattr(
            getattr(child_service, "repository", None), "workspace", None
        )
        if authority is None:
            authority = getattr(runtime, "workspace", None)
        if authority is None and runtime.workflow_dir is not None:
            authority = Path(runtime.workflow_dir).parent
        if authority is None:
            raise ValueError("study adapter requires an authoritative Workspace")
        authority_root = (
            Path(authority.root)
            if hasattr(authority, "root") and not isinstance(authority, Path)
            else Path(authority)
        )
        resolved_root = configured_root.resolve()
        if not resolved_root.is_relative_to(authority_root.resolve()):
            raise ValueError("study root is outside the authoritative workspace")
        self.study_dir = configured_root.resolve()
        self._assert_safe_root(self.study_dir)
        self.study_dir.mkdir(parents=True, exist_ok=True)
        self._planned_spec: VaspStudySpec | None = None
        self._preflighted_child_ids: set[str] = set()
        if child_service is None:
            from photomatagent.scientific.applications.vasp.unified.approvals import ApprovalReceiptStore
            from photomatagent.scientific.applications.vasp.unified.router import UnifiedVaspRouter
            base = authority_root
            child_executor = MolecularVaspExecutorAdapter(runtime, workflow_dir=self.study_dir / "children")
            child_service = UnifiedVaspService(
                ManifestRepository(Workspace(base)),
                ApprovalReceiptStore(base),
                UnifiedVaspRouter(molecular=child_executor),
            )
        self.child_service = child_service
        artifact_repository = getattr(child_service, "repository", None)
        self._derived_artifacts: ManifestRepository = (
            artifact_repository
            if isinstance(artifact_repository, ManifestRepository)
            else ManifestRepository(Workspace(authority_root))
        )

    def _study_executor(self, manifest: UnifiedVaspManifest) -> StudyExecutor:
        request = cast(StudyScientificSpec, manifest.scientific_spec).request
        # Snapshotting serializes request paths; restore their typed Path
        # representation before passing the request into the legacy planner.
        request = request.model_copy(update={
            "systems": [
                system.model_copy(update={"structure_path": Path(system.structure_path)})
                if system.structure_path is not None and not isinstance(system.structure_path, Path)
                else system
                for system in request.systems
            ]
        })
        spec = VaspStudySpec(
            study_id=request.study_id or manifest.workflow_id,
            request=request,
            study_dir=self.study_dir,
        )
        return StudyExecutor(spec, self.runtime)

    def _plan(self, manifest: UnifiedVaspManifest) -> VaspStudySpec:
        spec = self._study_executor(manifest).plan_only()
        state_path = self.study_dir / "study_state.json"
        stored: dict[str, Any] = {
            "tasks": manifest.study_task_states,
            "child_workflow_ids": manifest.child_workflow_ids,
            "resource_budget_consumed_core_hours": manifest.resource_budget_consumed_core_hours,
            "resource_budget_submitted_jobs": manifest.resource_budget_submitted_jobs,
        }
        # The JSON file is a derived report.  Use it only for legacy parents
        # that predate typed execution metadata in the parent manifest.
        if not manifest.study_task_states and state_path.is_file():
            try:
                stored = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                stored = {}
            manifest.resource_budget_consumed_core_hours = float(
                stored.get("resource_budget_consumed_core_hours", manifest.resource_budget_consumed_core_hours)
            )
            manifest.resource_budget_submitted_jobs = int(
                stored.get("resource_budget_submitted_jobs", manifest.resource_budget_submitted_jobs)
            )
            for task_id, child_id in (stored.get("child_workflow_ids") or {}).items():
                manifest.child_workflow_ids.setdefault(task_id, child_id)
        # Manifest metadata is authoritative on every re-plan/restart.  Do
        # not let the derived JSON view overwrite it once it exists.
        for task in spec.calculation_matrix.tasks:
            entry = (stored.get("tasks") or {}).get(task.task_id) or {}
            for name in ("state", "request_id", "workflow_dir", "results_dir", "error"):
                if name in entry and entry[name] is not None:
                    setattr(task, name, entry[name])
        self._planned_spec = spec
        return spec

    @staticmethod
    def _assert_safe_root(root: Path) -> None:
        cursor = Path(root.anchor)
        for part in root.parts[1:]:
            cursor /= part
            if cursor.is_symlink():
                raise ValueError(f"study root contains symlink: {cursor}")

    def _child_request(self, parent: UnifiedVaspManifest, task: Any) -> UnifiedVaspRequest:
        planner = self._study_executor(parent)
        workflow = planner._build_workflow(task)
        # Child ManifestRepository snapshots only workspace-relative source
        # references.  Convert the planner's absolute local path at this
        # boundary; never weaken repository path validation.
        workspace = getattr(getattr(self.child_service, "repository", None), "workspace", None)
        if workspace is not None and workflow.molecule.structure_path is not None:
            try:
                relative = workflow.molecule.structure_path.resolve().relative_to(workspace.root.resolve())
            except ValueError as exc:
                raise ValueError("child structure is outside the child workspace") from exc
            workflow = workflow.model_copy(update={
                "molecule": workflow.molecule.model_copy(update={"structure_path": relative})
            })
        return UnifiedVaspRequest(
            workflow_kind=VaspWorkflowKind.MOLECULAR,
            scientific_spec=MolecularScientificSpec(workflow=workflow),
        )

    def _children(
        self, manifest: UnifiedVaspManifest, *, require_existing: bool = True
    ) -> list[tuple[Any, UnifiedVaspManifest]]:
        spec = self._plan(manifest)
        duplicate_ids = self._duplicate_child_ids(manifest, spec)
        if duplicate_ids:
            self._mark_duplicate_child_ids(manifest, spec, duplicate_ids)
            self._persist_planning_state(manifest, spec)
            details = "; ".join(
                f"{child_id}: {', '.join(task_ids)}"
                for child_id, task_ids in duplicate_ids.items()
            )
            raise ValueError(f"duplicate child workflow ID mapping: {details}")
        children: list[tuple[Any, UnifiedVaspManifest]] = []
        for task in spec.calculation_matrix.tasks:
            # Proxy tasks have no real child. FAILED tasks with an assigned
            # child remain in the expected set so collect fails closed.
            if not task.structure_path or task.state == StudyTaskState.SKIPPED_PROXY.value:
                continue
            child_id = manifest.child_workflow_ids.get(task.task_id)
            if not child_id:
                if require_existing:
                    raise ValueError(f"missing persisted child ID for task {task.task_id}")
                continue
            try:
                child = self.child_service.load_manifest(child_id)
            except Exception as exc:
                task.state = StudyTaskState.FAILED.value
                task.error = f"child manifest unavailable: {str(exc)[:240]}"
                raise ValueError(f"missing child manifest {child_id}: {exc}") from exc
            children.append((task, child))
        return children

    @staticmethod
    def _expected_tasks(spec: VaspStudySpec) -> list[Any]:
        return [
            task for task in spec.calculation_matrix.tasks
            if task.structure_path and task.state != StudyTaskState.SKIPPED_PROXY.value
        ]

    def _duplicate_child_ids(
        self, manifest: UnifiedVaspManifest, spec: VaspStudySpec
    ) -> dict[str, list[str]]:
        owners: dict[str, list[str]] = {}
        for task in self._expected_tasks(spec):
            child_id = manifest.child_workflow_ids.get(task.task_id)
            if child_id:
                owners.setdefault(child_id, []).append(task.task_id)
        return {child_id: task_ids for child_id, task_ids in owners.items() if len(task_ids) > 1}

    @staticmethod
    def _mark_duplicate_child_ids(
        manifest: UnifiedVaspManifest,
        spec: VaspStudySpec,
        duplicate_ids: dict[str, list[str]],
    ) -> None:
        tasks = {task.task_id: task for task in spec.calculation_matrix.tasks}
        for child_id, task_ids in duplicate_ids.items():
            detail = f"duplicate child workflow ID {child_id} shared by {', '.join(task_ids)}"
            for task_id in task_ids:
                task = tasks[task_id]
                task.state = StudyTaskState.FAILED.value
                task.error = detail

    def _persist_planning_state(self, manifest: UnifiedVaspManifest, spec: VaspStudySpec) -> None:
        child_stages: dict[str, dict[str, str]] = {}
        for task_id, child_id in manifest.child_workflow_ids.items():
            try:
                child = self.child_service.load_manifest(child_id)
            except Exception:
                continue
            child_stages[child_id] = {
                item.name: item.state.value for item in child.stages
            }
        task_states = {
            task.task_id: {
                "state": task.state,
                "workflow_id": manifest.child_workflow_ids.get(task.task_id, ""),
                "request_id": task.request_id,
                "workflow_dir": task.workflow_dir,
                "results_dir": task.results_dir,
                "error": task.error,
            }
            for task in spec.calculation_matrix.tasks
        }
        manifest.study_task_states = task_states
        manifest.study_stage_states = child_stages
        payload = {
            "study_id": spec.study_id,
            "child_workflow_ids": dict(manifest.child_workflow_ids),
            "resource_budget_consumed_core_hours": manifest.resource_budget_consumed_core_hours,
            "resource_budget_submitted_jobs": manifest.resource_budget_submitted_jobs,
            "child_stage_states": child_stages,
            "tasks": {
                task.task_id: {
                    "state": task.state,
                    **task_states[task.task_id],
                }
                for task in spec.calculation_matrix.tasks
            },
        }
        self._atomic_write_state(payload)

    def _atomic_write_state(self, payload: dict[str, Any]) -> None:
        relative = self._derived_artifacts.workspace.relative(
            self.study_dir / "study_state.json"
        )
        self._derived_artifacts.write_derived_json(relative, payload)

    async def prepare(self, manifest: UnifiedVaspManifest) -> OperationResult:
        spec = self._plan(manifest)
        errors: list[str] = []
        prepared = 0
        for task in spec.calculation_matrix.tasks:
            if not task.structure_path or task.state == StudyTaskState.SKIPPED_PROXY.value:
                continue
            child_id = manifest.child_workflow_ids.get(task.task_id)
            try:
                if child_id:
                    child = self.child_service.load_manifest(child_id)
                else:
                    child = self.child_service.plan(self._child_request(manifest, task))
                    manifest.child_workflow_ids[task.task_id] = child.workflow_id
                if child.state is WorkflowState.PLANNED:
                    result = await self.child_service.prepare(child.workflow_id)
                    if not getattr(result, "ok", False):
                        errors.extend(list(getattr(result, "errors", []))[:3])
                        task.state = StudyTaskState.FAILED.value
                        task.error = "; ".join(list(getattr(result, "errors", []))[:3])
                    else:
                        prepared += 1
            except Exception as exc:
                errors.append(f"{task.task_id}: {exc}")
                task.state = StudyTaskState.FAILED.value
                task.error = str(exc)
        self._persist_planning_state(manifest, spec)
        return OperationResult(
            ok=not errors,
            data={
                "study_id": spec.study_id,
                "child_workflow_ids": dict(manifest.child_workflow_ids),
                "planned_tasks": len(spec.calculation_matrix.tasks),
                "prepared_children": prepared,
            },
            errors=errors[:10],
            evidence_gaps=errors[:10],
            artifacts=[str(self.study_dir / "study_state.json")],
        )

    async def preflight(self, manifest: UnifiedVaspManifest) -> PreflightResult:
        try:
            children = self._children(manifest)
        except Exception as exc:
            self._persist_planning_state(manifest, self._planned_spec or self._plan(manifest))
            return PreflightResult(
                ok=False,
                passed=False,
                errors=[f"study child discovery failed: {str(exc)[:240]}"],
                evidence_gaps=["a planned child manifest is missing"],
            )
        results: list[Any] = []
        errors: list[str] = []
        gaps: list[str] = []
        for task, child in children:
            try:
                result = await self.child_service.preflight(child.workflow_id)
            except Exception as exc:
                task.state = StudyTaskState.FAILED.value
                task.error = f"child preflight failed: {str(exc)[:240]}"
                errors.append(f"{task.task_id}: {task.error}")
                gaps.append(f"{task.task_id}: child preflight did not complete")
                continue
            results.append(result)
            errors.extend(list(getattr(result, "errors", []))[:2])
            gaps.extend(list(getattr(result, "evidence_gaps", []))[:2])
            if not getattr(result, "ok", False):
                task.state = StudyTaskState.FAILED.value
                task.error = "; ".join(list(getattr(result, "errors", []))[:3])
            else:
                self._preflighted_child_ids.add(child.workflow_id)
        passed = bool(children) and not errors and all(bool(getattr(result, "ok", False)) for result in results)
        if not results:
            errors.append("study has no executable child workflows")
        self._persist_planning_state(manifest, self._plan(manifest))
        return PreflightResult(
            ok=passed,
            passed=passed,
            errors=errors,
            evidence_gaps=gaps,
            data={"child_count": len(results), "children_passed": sum(bool(getattr(result, "ok", False)) for result in results)},
        )

    async def submit(self, manifest: UnifiedVaspManifest, stage: UnifiedStage, resource: ResourceRequest) -> SubmissionResult:
        # The parent-facing resource is intentionally not passed to a child;
        # UnifiedVaspService recomputes and authorizes the child's exact
        # stage resource.
        del stage, resource
        try:
            children = self._children(manifest)
        except Exception as exc:
            self._persist_planning_state(manifest, self._planned_spec or self._plan(manifest))
            return SubmissionResult(
                ok=False,
                request_id=f"study-{manifest.workflow_id}",
                errors=[f"study child discovery failed: {str(exc)[:240]}"],
                evidence_gaps=["a planned child manifest is missing"],
            )
        # ``_children`` retains the exact planned object whose task states are
        # updated below; planning it again would discard those updates before
        # they are persisted for a process restart.
        spec = self._planned_spec
        if spec is None:
            spec = self._plan(manifest)
        submitted: list[str] = []
        request_ids: list[str] = []
        errors: list[str] = []
        pending_decision: Any = None
        spent_core_hours = manifest.resource_budget_consumed_core_hours
        started_jobs = manifest.resource_budget_submitted_jobs
        budget = spec.request.resource_budget
        for task, child in children:
            if child.state in {WorkflowState.FAILED, WorkflowState.VALIDATION_FAILED}:
                task.state = StudyTaskState.FAILED.value
                task.error = f"child workflow is terminal {child.state.value}"
                errors.append(f"{task.task_id}: {task.error}")
                continue
            if any(item.state in {
                WorkflowState.SUBMITTED,
                WorkflowState.RUNNING,
                WorkflowState.RECONCILING,
            } for item in child.stages):
                continue
            child_stage = self._next_stage(child)
            if child_stage is None and child.workflow_id not in self._preflighted_child_ids and child.state in {
                WorkflowState.PLANNED,
                WorkflowState.PREPARED,
                WorkflowState.VALIDATED,
            }:
                try:
                    preflight = await self.child_service.preflight(child.workflow_id)
                    if not getattr(preflight, "ok", False):
                        child_errors = list(getattr(preflight, "errors", []))[:3]
                        errors.extend(child_errors)
                        task.state = StudyTaskState.FAILED.value
                        task.error = "; ".join(child_errors)
                        continue
                    child = self.child_service.load_manifest(child.workflow_id)
                    child_stage = self._next_stage(child, allow_planned=True)
                except Exception as exc:
                    task.state = StudyTaskState.FAILED.value
                    task.error = f"child preflight failed: {str(exc)[:240]}"
                    errors.append(f"{task.task_id}: {task.error}")
                    continue
            elif child_stage is None:
                child_stage = self._next_stage(child, allow_planned=True)
            if child_stage is None:
                # All stages are either terminal or waiting on a dependency;
                # a stale coarse study task must never trigger a duplicate.
                continue
            stage_cost = task.estimated_core_hours / max(len(child.stages), 1)
            if (
                budget.max_jobs is not None
                and started_jobs >= budget.max_jobs
            ) or (
                spent_core_hours + stage_cost
                > budget.max_core_hours + 1e-9
            ):
                task.state = StudyTaskState.SKIPPED_BUDGET.value
                task.error = "resource budget exhausted; no new child started"
                continue
            try:
                result = await self.child_service.submit(child.workflow_id, stage=child_stage.name)
            except Exception as exc:
                task.state = StudyTaskState.FAILED.value
                task.error = f"child submission failed: {str(exc)[:240]}"
                errors.append(f"{task.task_id}: {task.error}")
                continue
            if not getattr(result, "ok", False):
                child_errors = list(getattr(result, "errors", []))[:3]
                errors.extend(child_errors)
                task.error = "; ".join(child_errors)
                pending = getattr(result, "pending_decision", None)
                if pending is not None and getattr(result, "state", None) in {
                    WorkflowState.AWAITING_RESOURCE_CONFIRMATION,
                    WorkflowState.AWAITING_SCIENTIFIC_CONFIRMATION,
                }:
                    pending_decision = pending
                    task.state = (
                        StudyTaskState.AWAITING_RESOURCE_CONFIRMATION.value
                        if getattr(pending.kind, "value", pending.kind) == "resource"
                        else StudyTaskState.AWAITING_SCIENTIFIC_CONFIRMATION.value
                    )
                else:
                    task.state = StudyTaskState.FAILED.value
                try:
                    loaded = self.child_service.load_manifest(child.workflow_id)
                except Exception:
                    loaded = child
                task.request_id = next((item.request_id for item in loaded.stages if item.request_id), "")
                continue
            task.state = StudyTaskState.SUBMITTED.value
            try:
                loaded = self.child_service.load_manifest(child.workflow_id)
            except Exception:
                loaded = child
            persisted_stage = next(
                (item for item in loaded.stages if item.name == child_stage.name),
                child_stage,
            )
            task.request_id = str(
                persisted_stage.request_id
                or getattr(result, "data", {}).get("request_id", "")
                or ""
            )
            spent_core_hours += stage_cost
            started_jobs += 1
            manifest.resource_budget_consumed_core_hours = spent_core_hours
            manifest.resource_budget_submitted_jobs = started_jobs
            submitted.append(task.task_id)
            request_ids.append(task.request_id or child.workflow_id)
        self._persist_planning_state(manifest, spec)
        return SubmissionResult(
            ok=not errors,
            request_id=request_ids[0] if request_ids else f"study-{manifest.workflow_id}",
            submitted=bool(submitted) or not children,
            duplicate=not submitted and not errors,
            errors=errors[:10],
            data={"submitted_tasks": submitted, "child_workflow_ids": dict(manifest.child_workflow_ids)},
            pending_decision=pending_decision,
        )

    @staticmethod
    def _next_stage(
        child: UnifiedVaspManifest, *, allow_planned: bool = False
    ) -> UnifiedStage | None:
        by_name = {candidate.name: candidate for candidate in child.stages}
        for item in child.stages:
            if item.state is not WorkflowState.PREFLIGHTED and not (
                allow_planned and item.state is WorkflowState.PLANNED
            ):
                continue
            if all(
                by_name.get(dependency) is not None
                and by_name[dependency].state
                in {WorkflowState.SCHEDULER_COMPLETED, WorkflowState.VALIDATED}
                for dependency in item.depends_on
            ):
                return item
        return None

    async def status(self, manifest: UnifiedVaspManifest) -> StatusResult:
        states: dict[str, str] = {}
        errors: list[str] = []
        try:
            children = self._children(manifest)
        except Exception as exc:
            self._persist_planning_state(manifest, self._planned_spec or self._plan(manifest))
            return StatusResult(
                ok=False, query_failed=True,
                errors=[f"study child discovery failed: {str(exc)[:240]}"],
                evidence_gaps=["a planned child manifest is missing"],
            )
        for task, child in children:
            try:
                result = await self.child_service.status(child.workflow_id)
                states[manifest.child_workflow_ids[task.task_id]] = getattr(getattr(result, "state", None), "value", "UNKNOWN")
                errors.extend(list(getattr(result, "errors", []))[:2])
                if getattr(result, "state", None) in {WorkflowState.FAILED, WorkflowState.VALIDATION_FAILED}:
                    task.state = StudyTaskState.FAILED.value
                    task.error = f"child workflow is terminal {result.state.value}"
            except Exception as exc:
                task.state = StudyTaskState.FAILED.value
                task.error = f"child status failed: {str(exc)[:240]}"
                errors.append(f"{task.task_id}: {task.error}")
        self._persist_planning_state(manifest, self._planned_spec or self._plan(manifest))
        return StatusResult(ok=not errors, stage_states=states, errors=errors[:10], query_failed=bool(errors))

    async def reconcile(self, manifest: UnifiedVaspManifest) -> RecoveryResult:
        errors: list[str] = []
        states: dict[str, str] = {}
        try:
            children = self._children(manifest)
        except Exception as exc:
            self._persist_planning_state(manifest, self._planned_spec or self._plan(manifest))
            return RecoveryResult(
                ok=False,
                action="STOP",
                errors=[f"study child discovery failed: {str(exc)[:240]}"],
                stage_states={},
            )
        for task, child in children:
            try:
                result = await self.child_service.resume(child.workflow_id)
                errors.extend(list(getattr(result, "errors", []))[:2])
                states[manifest.child_workflow_ids[task.task_id]] = getattr(getattr(result, "state", None), "value", "UNKNOWN")
                if getattr(result, "state", None) in {WorkflowState.FAILED, WorkflowState.VALIDATION_FAILED}:
                    task.state = StudyTaskState.FAILED.value
            except Exception as exc:
                task.state = StudyTaskState.FAILED.value
                task.error = f"child resume failed: {str(exc)[:240]}"
                errors.append(f"{task.task_id}: {task.error}")
        self._persist_planning_state(manifest, self._planned_spec or self._plan(manifest))
        return RecoveryResult(ok=not errors, action="AUTO_RESUME" if not errors else "RECONCILE", errors=errors[:10], stage_states=states)

    async def collect(self, manifest: UnifiedVaspManifest) -> CollectionResult:
        spec = self._plan(manifest)
        results: list[Any] = []
        errors: list[str] = []
        evidence: list[Any] = []
        gaps: list[str] = []
        stage_states: dict[str, str] = {}
        expected = self._expected_tasks(spec)
        duplicate_ids = self._duplicate_child_ids(manifest, spec)
        duplicate_tasks = {
            task_id for task_ids in duplicate_ids.values() for task_id in task_ids
        }
        if duplicate_ids:
            self._mark_duplicate_child_ids(manifest, spec, duplicate_ids)
        for task in expected:
            child_id = manifest.child_workflow_ids.get(task.task_id)
            if task.task_id in duplicate_tasks:
                detail = f"duplicate child workflow ID {child_id} for expected task"
                errors.append(f"{task.task_id}: {detail}")
                gaps.append(f"{task.task_id}: expected child mapping is not one-to-one")
                stage_states[f"{child_id}:{task.task_id}"] = WorkflowState.FAILED.value
                continue
            if not child_id:
                task.state = StudyTaskState.FAILED.value
                task.error = "missing persisted child workflow ID"
                errors.append(f"{task.task_id}: {task.error}")
                gaps.append(f"{task.task_id}: expected child workflow is missing")
                stage_states[f"missing:{task.task_id}"] = WorkflowState.FAILED.value
                continue
            try:
                child = self.child_service.load_manifest(child_id)
            except Exception as exc:
                task.state = StudyTaskState.FAILED.value
                task.error = f"child manifest unavailable: {str(exc)[:240]}"
                errors.append(f"{task.task_id}: {task.error}")
                gaps.append(f"{task.task_id}: expected child manifest is missing")
                stage_states[child_id] = WorkflowState.FAILED.value
                continue
            try:
                result = await self.child_service.collect(child.workflow_id)
            except Exception as exc:
                task.state = StudyTaskState.FAILED.value
                task.error = f"child collect failed: {str(exc)[:240]}"
                errors.append(f"{task.task_id}: {task.error}")
                gaps.append(f"{task.task_id}: child collect did not complete")
                continue
            results.append(result)
            state = getattr(getattr(result, "state", None), "value", "UNKNOWN")
            stage_states[child.workflow_id] = str(state)
            errors.extend(list(getattr(result, "errors", []))[:2])
            gaps.extend(list(getattr(result, "evidence_gaps", []))[:2])
            child_state = getattr(result, "state", None)
            if child_state in {WorkflowState.FAILED, WorkflowState.VALIDATION_FAILED}:
                task.state = StudyTaskState.FAILED.value
                task.error = f"child workflow is terminal {child_state.value}"
                if task.error not in errors:
                    errors.append(f"{task.task_id}: {task.error}")
                if not any(task.task_id in gap for gap in gaps):
                    gaps.append(f"{task.task_id}: child validation failed")
            elif child_state is WorkflowState.VALIDATED:
                task.state = StudyTaskState.VALIDATED.value
            else:
                task.state = StudyTaskState.COLLECTED.value
        raw_evidence = [item for result in results for item in list(getattr(result, "evidence", []))]
        evidence = [item for item in raw_evidence if isinstance(item, ScientificEvidence)]
        if len(evidence) != len(raw_evidence):
            gaps.append("study collection received non-ScientificEvidence child evidence")
        validated = (
            bool(expected)
            and len(results) == len(expected)
            and not errors
            and not gaps
            and all(
                bool(getattr(result, "ok", False))
                and getattr(getattr(result, "state", None), "value", "") == WorkflowState.VALIDATED.value
                and bool(getattr(result, "evidence", []))
                and all(isinstance(item, ScientificEvidence) for item in getattr(result, "evidence", []))
                for result in results
            )
        )
        if not evidence:
            gaps.append("validated study collection requires typed ScientificEvidence")
        if not validated and not gaps:
            gaps.append("not every expected child workflow is validated")
        self._persist_planning_state(manifest, spec)
        return CollectionResult(
            ok=validated,
            validated=validated,
            evidence=evidence,
            errors=errors,
            evidence_gaps=gaps[:10],
            stage_states=stage_states,
        )

    async def report(self, manifest: UnifiedVaspManifest, request: ReportRequest) -> ReportResult:
        try:
            children = self._children(manifest)
        except Exception as exc:
            self._persist_planning_state(manifest, self._planned_spec or self._plan(manifest))
            return ReportResult(
                ok=False,
                report_kind=request.kind,
                errors=[f"study child discovery failed: {str(exc)[:240]}"],
                evidence_gaps=["a planned child manifest is missing"],
            )
        if request.kind.value == "binding_energy":
            known_ids = set(manifest.child_workflow_ids.values())
            unrelated = [item for item in request.related_workflow_ids if item not in known_ids]
            if unrelated:
                return ReportResult(
                    ok=False,
                    report_kind=request.kind,
                    errors=["related_workflow_ids must belong to this study parent"],
                    evidence_gaps=["binding reference is not a persisted study child"],
                )
            spec = self._planned_spec or self._plan(manifest)
            groups = spec.calculation_matrix.binding_groups
            if not groups:
                return ReportResult(
                    ok=False,
                    report_kind=request.kind,
                    errors=["no persisted study binding relationship is available"],
                    evidence_gaps=["complete binding references are missing"],
                )
            task_children = {task.task_id: child for task, child in children}
            reports: list[Any] = []
            for group in groups:
                if group.state == StudyTaskState.FAILED.value or group.missing_fragment_ids:
                    missing = ", ".join(group.missing_fragment_ids)
                    return ReportResult(
                        ok=False,
                        report_kind=request.kind,
                        errors=[group.error or "binding relationship is not computable"],
                        evidence_gaps=[
                            f"binding group is incomplete{': ' + missing if missing else ''}"
                        ],
                    )
                complex_child = task_children.get(group.complex_task_id)
                reference_ids = [manifest.child_workflow_ids.get(item, "") for item in group.fragment_task_ids]
                reference_children = [task_children.get(item) for item in group.fragment_task_ids]
                complex_id = manifest.child_workflow_ids.get(group.complex_task_id, "")
                if (
                    len(group.fragment_task_ids) != len(set(group.fragment_task_ids))
                    or len(reference_ids) != len(set(reference_ids))
                    or complex_id in reference_ids
                ):
                    return ReportResult(
                        ok=False,
                        report_kind=request.kind,
                        errors=["binding relationship contains duplicate complex/fragment child IDs"],
                        evidence_gaps=["binding references must form a one-to-one child set"],
                    )
                if (
                    complex_child is None
                    or complex_child.state is not WorkflowState.VALIDATED
                    or not reference_ids
                    or any(not item for item in reference_ids)
                    or any(child is None or child.state is not WorkflowState.VALIDATED for child in reference_children)
                ):
                    return ReportResult(
                        ok=False,
                        report_kind=request.kind,
                        errors=["binding relationship has incomplete persisted child IDs"],
                        evidence_gaps=["binding report requires every real child reference"],
                    )
                try:
                    reports.append(await self.child_service.report(
                        complex_child.workflow_id,
                        ReportRequest(kind=request.kind, related_workflow_ids=reference_ids),
                    ))
                except Exception as exc:
                    return ReportResult(
                        ok=False,
                        report_kind=request.kind,
                        errors=[f"binding child report failed: {str(exc)[:240]}"],
                        evidence_gaps=["binding child report did not complete"],
                    )
            errors = [error for report in reports for error in list(getattr(report, "errors", []))][:10]
            return ReportResult(
                ok=not errors and all(bool(getattr(report, "ok", False)) for report in reports),
                report_kind=request.kind,
                data={"groups": [report.data for report in reports]},
                errors=errors,
                evidence_gaps=[gap for report in reports for gap in list(getattr(report, "evidence_gaps", []))][:10],
            )
        if request.kind.value in {"orbitals", "esp"}:
            reports = []
            for task, child in children:
                try:
                    reports.append(await self.child_service.report(child.workflow_id, request))
                except Exception as exc:
                    task.state = StudyTaskState.FAILED.value
                    task.error = f"child report failed: {str(exc)[:240]}"
                    return ReportResult(
                        ok=False,
                        report_kind=request.kind,
                        errors=[task.error],
                        evidence_gaps=["child report did not complete"],
                    )
            errors = [error for report in reports for error in list(getattr(report, "errors", []))][:10]
            return ReportResult(
                ok=not errors and all(bool(getattr(report, "ok", False)) for report in reports),
                report_kind=request.kind,
                data={"children": [report.data for report in reports]},
                errors=errors,
                evidence_gaps=[gap for report in reports for gap in list(getattr(report, "evidence_gaps", []))][:10],
            )
        return ReportResult(
            ok=True,
            report_kind=request.kind,
            data={"study_id": manifest.workflow_id, "child_workflow_ids": dict(manifest.child_workflow_ids)},
        )
