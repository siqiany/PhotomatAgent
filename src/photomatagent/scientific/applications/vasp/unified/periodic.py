"""Periodic VASP adapter that owns lifecycle through SubmitOnceSession."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, cast

from photomatagent.scientific.applications.vasp.application import VaspApplication
from photomatagent.scientific.applications.vasp.profiles import get_profile
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
    PeriodicScientificSpec,
    ReportRequest,
    UnifiedStage,
    UnifiedVaspManifest,
)
from photomatagent.scientific.applications.vasp.unified.fingerprints import execution_fingerprint
from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.applications.vasp.unified.repository import (
    ensure_managed_vasp_directory,
    managed_vasp_path,
    revalidate_managed_vasp_path,
    resolve_workspace_relative_path,
    validate_workflow_id,
)
from photomatagent.scientific.remote.fake import FakeSCNetBackend
from photomatagent.scientific.remote.lifecycle import (
    SubmitOnceSession,
    SubmissionGate,
)
from photomatagent.scientific.remote.models import (
    HPCJobState,
    RemoteJobRef,
    ResourceRequest,
)
from photomatagent.scientific.remote.registry import (
    JobLifecycleState,
    JobRegistry,
    derive_request_id,
)
from photomatagent.workspace import Workspace


class PeriodicVaspExecutor:
    """Periodic VASP implementation of the unified executor Protocol."""

    def __init__(
        self,
        application: VaspApplication,
        session: SubmitOnceSession | None = None,
    ) -> None:
        self.application = application
        if session is None:
            if application.backend is None:
                raise RuntimeError("PeriodicVaspExecutor requires a backend")
            workspace = Workspace(application.workspace)
            registry_path = managed_vasp_path(workspace, "jobs.sqlite3")
            marker_directory = managed_vasp_path(workspace, "markers")
            ensure_managed_vasp_directory(workspace, registry_path.parent)
            ensure_managed_vasp_directory(workspace, marker_directory)
            session = SubmitOnceSession(
                JobRegistry(registry_path),
                application.backend,
                marker_temp_dir=marker_directory,
            )
            revalidate_managed_vasp_path(workspace, registry_path)
            revalidate_managed_vasp_path(workspace, marker_directory)
        self.session = session

    # -- path / input helpers ----------------------------------------------

    def _workspace_path(self, path: str) -> Path:
        return resolve_workspace_relative_path(
            Workspace(self.application.workspace), path, must_exist=False
        )

    def _managed_workflow_path(
        self, area: str, workflow_id: str, *parts: str
    ) -> Path:
        workflow_id = validate_workflow_id(workflow_id)
        return managed_vasp_path(
            Workspace(self.application.workspace), area, workflow_id, *parts
        )

    def _prepare_data(
        self, manifest: UnifiedVaspManifest
    ) -> dict[str, Any]:
        spec = cast(PeriodicScientificSpec, manifest.scientific_spec)
        structure = self._workspace_path(spec.structure_path)
        output_dir = self._stage_root(manifest)
        self._revalidate_application_paths(manifest, output_dir)
        data = self.application.prepare_inputs(
            structure_path=structure,
            profile_name=spec.profile,
            output_dir=output_dir,
            spec_overrides=dict(spec.scientific_overrides),
        )
        self._revalidate_application_paths(manifest, output_dir)
        return data

    def _stage_root(self, manifest: UnifiedVaspManifest) -> Path:
        return self._managed_workflow_path(
            "workflows", manifest.workflow_id, "inputs"
        )

    def _revalidate_application_paths(
        self, manifest: UnifiedVaspManifest, managed_path: Path
    ) -> None:
        spec = cast(PeriodicScientificSpec, manifest.scientific_spec)
        self._workspace_path(spec.structure_path)
        revalidate_managed_vasp_path(
            Workspace(self.application.workspace), managed_path
        )

    def _stage_dir(
        self, manifest: UnifiedVaspManifest, stage: UnifiedStage
    ) -> Path | None:
        data = self._prepare_data(manifest)
        for item in data.get("stages", []):
            if item.get("stage") == stage.name:
                return Path(item["directory"])
        return None

    def _request_id(
        self, manifest: UnifiedVaspManifest, stage: UnifiedStage
    ) -> str:
        if stage.request_id:
            return stage.request_id
        execution = stage.execution_fingerprint or manifest.execution_fingerprint
        if not execution:
            execution = execution_fingerprint(
                manifest.scientific_fingerprint, stage.resource_recommendation,
                stage, stage.attempt_inputs or None,
            )
        return derive_request_id(
            f"periodic:{manifest.workflow_id}", stage.name, execution
        )

    # -- executor contract --------------------------------------------------

    async def prepare(self, manifest: UnifiedVaspManifest) -> OperationResult:
        try:
            data = self._prepare_data(manifest)
        except Exception as exc:
            return OperationResult(
                ok=False,
                errors=[f"{type(exc).__name__}: {exc}"],
            )
        return OperationResult(ok=True, data=data)

    async def preflight(self, manifest: UnifiedVaspManifest) -> PreflightResult:
        spec = cast(PeriodicScientificSpec, manifest.scientific_spec)
        try:
            structure = self._workspace_path(spec.structure_path)
            problems = self.application.validate_inputs(
                structure, spec.profile
            )
        except Exception as exc:
            problems = [f"{type(exc).__name__}: {exc}"]
        # POTCAR resolvability is part of submission readiness: a workflow
        # whose POTCAR cannot be supplied locally or remotely would submit
        # and die instantly on the cluster (VASP aborts: POTCAR not found).
        problems.extend(self._potcar_problems(manifest))
        passed = not problems
        return PreflightResult(
            ok=passed,
            passed=passed,
            errors=problems,
            data={"profile": spec.profile, "stage_names": [s.name for s in manifest.stages]},
            evidence_gaps=[] if passed else problems,
        )

    def _potcar_problems(
        self, manifest: UnifiedVaspManifest
    ) -> list[str]:
        """Return [] when a submit-time POTCAR strategy can be supplied.

        Element symbols are derived from the structure itself (not from the
        policy file) so an unconfigured local library is still caught: the
        policy only lists per-element lines when a local library is present.
        """
        spec = cast(PeriodicScientificSpec, manifest.scientific_spec)
        from photomatagent.scientific.applications.vasp.inputs import (
            VaspInputGenerator,
        )
        from photomatagent.scientific.applications.vasp.psp import (
            potcar_element_name,
        )

        try:
            structure = VaspInputGenerator.load_structure(
                self._workspace_path(spec.structure_path)
            )
        except Exception as exc:
            return [f"{type(exc).__name__}: {exc}"]
        symbols = [
            potcar_element_name(str(element.symbol))
            for element in structure.composition.elements
        ]
        if not symbols:
            return []
        try:
            stage_dirs = self._prepare_data(manifest).get("stages", [])
        except Exception as exc:
            return [f"{type(exc).__name__}: {exc}"]
        if not stage_dirs:
            return ["no stage input directories prepared; run vasp.prepare first"]
        directory = Path(stage_dirs[0]["directory"])
        if self.application.resolve_potcar(directory) is not None:
            return []
        if self.application.remote_psp_dir:
            return []
        return [
            "POTCAR cannot be resolved for "
            + ", ".join(symbols)
            + ": configure PMG_VASP_PSP_DIR with the required setups "
            "(e.g. Ba_sv) or SCNET_VASP_PSP_DIR on SCNet"
        ]

    async def submit(
        self,
        manifest: UnifiedVaspManifest,
        stage: UnifiedStage,
        resource: ResourceRequest,
    ) -> SubmissionResult:
        preflight = await self.preflight(manifest)
        request_id = self._request_id(manifest, stage)
        if not preflight.passed:
            return SubmissionResult(
                ok=False,
                request_id=request_id,
                submitted=False,
                errors=preflight.errors,
                evidence_gaps=preflight.errors,
            )
        # After vasp.resume confirmed a terminal scheduler failure, the
        # workflow is reset to PREFLIGHTED and this submit is a deliberate
        # new attempt: the registry still holds the failed record under the
        # derived request_id, so alone it would return a terminal duplicate.
        # Force a fresh attempt (new request_id with a parent pointer) only
        # when the recorded job is scheduler-confirmed failed -- never for
        # RUNNING/PENDING, COMPLETED (must be collected), or ambiguous state.
        existing = self.session.registry.get(request_id)
        force_new_attempt = bool(
            existing is not None
            and existing.job_id is not None
            and existing.state
            in {
                JobLifecycleState.FAILED,
                JobLifecycleState.TIMEOUT,
                JobLifecycleState.OUT_OF_MEMORY,
                JobLifecycleState.CANCELLED,
            }
        )
        stage_dir = self._stage_dir(manifest, stage)
        if stage_dir is None:
            return SubmissionResult(
                ok=False,
                request_id=request_id,
                submitted=False,
                errors=[f"stage {stage.name} was not prepared"],
            )
        spec = cast(PeriodicScientificSpec, manifest.scientific_spec)
        profile = get_profile(spec.profile)
        symbols = self.application._potcar_symbols(stage_dir)
        local_potcar = self.application.resolve_potcar(stage_dir)
        potcar_mode = "local" if local_potcar is not None else (
            "remote" if self.application.remote_psp_dir and symbols else "none"
        )
        if potcar_mode == "none":
            # Never submit a job that is guaranteed to die: VASP aborts
            # immediately when POTCAR is absent.
            return SubmissionResult(
                ok=False,
                request_id=request_id,
                submitted=False,
                errors=[
                    "POTCAR cannot be supplied: local PMG_VASP_PSP_DIR lacks "
                    "the required setups and no remote SCNET_VASP_PSP_DIR is "
                    "configured; nothing was submitted"
                ],
            )
        if potcar_mode == "local" and (stage_dir / "POTCAR").stat().st_size == 0:
            return SubmissionResult(
                ok=False,
                request_id=request_id,
                submitted=False,
                errors=["local POTCAR resolved to an empty file; nothing submitted"],
            )

        def render(job_name: str, request_resource: ResourceRequest) -> str:
            return self.application.render_slurm(
                job_name=job_name,
                profile=profile,
                resource=request_resource,
                executable=profile.executable,
                potcar_symbols=symbols if potcar_mode == "remote" else [],
            )

        try:
            result = await self.session.submit_once(
                application="vasp",
                workflow_stage=stage.name,
                job_name=f"{manifest.workflow_id[:12]}-{stage.name}",
                local_input_dir=stage_dir,
                gate=SubmissionGate(
                    passed=True,
                    errors=[],
                    summary={"workflow_id": manifest.workflow_id},
                ),
                resource=resource,
                executable=profile.executable,
                script_name="vasp.slurm",
                script_renderer=render,
                request_id=request_id,
                force_new_attempt=force_new_attempt,
                provenance={
                    "workflow_id": manifest.workflow_id,
                    "scientific_fingerprint": manifest.scientific_fingerprint,
                    "execution_fingerprint": manifest.execution_fingerprint,
                },
                potcar_mode=potcar_mode,
                potcar_symbols=symbols,
                remote_psp_dir=self.application.remote_psp_dir,
            )
        except Exception as exc:
            return SubmissionResult(
                ok=False,
                request_id=request_id,
                submitted=False,
                errors=[f"{type(exc).__name__}: {exc}"],
            )
        record = result.record or {}
        return SubmissionResult(
            ok=not result.blocked and not result.error,
            request_id=result.request_id,
            job_id=record.get("job_id"),
            submitted=result.submitted,
            duplicate=result.duplicate,
            needs_reconciliation=result.needs_reconciliation,
            errors=[result.error] if result.error else [],
            data={"remote_directory": record.get("remote_directory"), "record": result.record},
        )

    async def status(self, manifest: UnifiedVaspManifest) -> StatusResult:
        stage_states: dict[str, str] = {}
        failed = False
        for stage in manifest.stages:
            rid = self._request_id(manifest, stage)
            record = self.session.registry.get(rid)
            if record is None or record.job_id is None:
                # Stage was never submitted (or lost its job id): this is
                # not a query failure -- later stages of a chain are simply
                # not started yet, and a caller must not treat them as an
                # error that hides the real state of submitted stages.
                stage_states[stage.name] = "NOT_FOUND"
                continue
            refresh = await self.session.refresh_status(rid)
            if refresh.query_failed:
                failed = True
                stage_states[stage.name] = "UNKNOWN"
            else:
                stage_states[stage.name] = refresh.state or "UNKNOWN"
        return StatusResult(
            ok=not failed,
            stage_states=stage_states,
            query_failed=failed,
        )

    async def reconcile(self, manifest: UnifiedVaspManifest) -> RecoveryResult:
        stage_states: dict[str, str] = {}
        for stage in manifest.stages:
            rid = self._request_id(manifest, stage)
            result = await self.session.reconcile(rid)
            if result.outcome in {"FOUND_MULTIPLE", "FOUND_UNIQUE_UNKNOWN_STATE"}:
                return RecoveryResult(
                    ok=False,
                    action="RECONCILE",
                    stage_states=stage_states,
                    errors=[result.error or result.outcome],
                )
            if result.outcome == "NOT_FOUND":
                # No job can be found for this request: either it was never
                # submitted or the cluster lost it. Not a failure to report.
                stage_states[stage.name] = "NOT_FOUND"
                continue
            if result.error:
                return RecoveryResult(
                    ok=False,
                    action="STOP",
                    stage_states=stage_states,
                    errors=[result.error],
                )
            record = self.session.registry.get(rid)
            stage_states[stage.name] = (
                record.state.value if record is not None else "UNKNOWN"
            )
        return RecoveryResult(
            ok=True,
            action="AUTO_RESUME",
            stage_states=stage_states,
            data={"reconciled": True},
        )

    async def collect(self, manifest: UnifiedVaspManifest) -> CollectionResult:
        evidence_gaps: list[str] = []
        evidence: list[ScientificEvidence] = []
        for stage in manifest.stages:
            rid = self._request_id(manifest, stage)
            record = self.session.registry.get(rid)
            if record is None or record.job_id is None:
                evidence_gaps.append(f"{stage.name}: no job_id in registry")
                continue
            local_dir = self._managed_workflow_path(
                "results", manifest.workflow_id, stage.name
            )
            try:
                self._revalidate_application_paths(manifest, local_dir)
                report = await self.application.collect(
                    job_ref=RemoteJobRef(
                        backend="scnet",
                        application="vasp",
                        job_id=record.job_id,
                        remote_directory=record.remote_directory or "",
                    ),
                    local_dir=local_dir,
                    profile_name=cast(
                        PeriodicScientificSpec, manifest.scientific_spec
                    ).profile,
                )
                self._revalidate_application_paths(manifest, local_dir)
            except Exception as exc:
                evidence_gaps.append(f"{stage.name}: {type(exc).__name__}: {exc}")
                continue
            if not report.get("scientifically_valid", False):
                evidence_gaps.append(
                    f"{stage.name}: " + "; ".join(report.get("validation_problems", []))
                )
                continue
            self.session.mark_result_state(
                rid, collected=True, validated=True, evidence=len(evidence)
            )
            evidence.extend(
                [
                    # Existing periodic tools create evidence from parsed energy.
                    # The unified service may map this via the tool pack; here
                    # we keep validator-owned outputs in data.
                ]
            )
        return CollectionResult(
            ok=not evidence_gaps,
            validated=not evidence_gaps,
            evidence=evidence,
            evidence_gaps=evidence_gaps,
        )

    async def report(
        self, manifest: UnifiedVaspManifest, request: ReportRequest
    ) -> ReportResult:
        return ReportResult(ok=True, report_kind=request.kind)
