"""Periodic VASP adapter that owns lifecycle through SubmitOnceSession."""

from __future__ import annotations

import hashlib
import shutil
import uuid
from pathlib import Path
from typing import Any, cast

from photomatagent.scientific.applications.vasp.application import VaspApplication
from photomatagent.scientific.applications.vasp.profiles import (
    get_profile,
    periodic_stage_specs,
)
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
    WorkflowState,
)
from photomatagent.scientific.applications.vasp.unified.fingerprints import execution_fingerprint
from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.capabilities.structure.artifacts import structure_hash
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
    canonical_input_hash,
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
        if any(
            isinstance((stage.attempt_inputs or {}).get("staged_input_sha256"), dict)
            and bool((stage.attempt_inputs or {}).get("staged_input_sha256"))
            for stage in manifest.stages
        ):
            # A workflow can be prepared twice during recovery/resume.  Never
            # regenerate the whole input tree after any stage has an
            # authorized staged-input hash; that would overwrite the exact
            # bytes represented by the execution fingerprint.
            stages: list[dict[str, Any]] = []
            for stage in manifest.stages:
                expected = (stage.attempt_inputs or {}).get(
                    "staged_input_sha256"
                )
                existing = self._existing_stage_dir(manifest, stage)
                if isinstance(expected, dict) and expected:
                    if existing is None:
                        raise ValueError(
                            f"authorized stage {stage.name!r} inputs are missing; "
                            "refusing to regenerate and overwrite the binding"
                        )
                    errors = self._verify_authorized_inputs(existing, stage)
                    if errors:
                        raise ValueError(
                            f"authorized stage {stage.name!r} inputs changed; "
                            "refusing to overwrite: " + "; ".join(errors)
                        )
                    stages.append(
                        {"stage": stage.name, "directory": str(existing)}
                    )
                else:
                    directory = self._prepare_stage_dir(manifest, stage)
                    stages.append(
                        {"stage": stage.name, "directory": str(directory)}
                    )
            self._revalidate_application_paths(manifest, output_dir)
            profile = get_profile(spec.profile)
            return {
                "profile": profile.name,
                "soc": profile.soc,
                "executable": profile.executable,
                "stages": stages,
                "potcar_policy": "POTCAR resolved at submit time; never committed",
                "notes": [
                    "bound stage directories were preserved; only unbound "
                    "stage directories were generated"
                ],
            }
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
        existing = self._existing_stage_dir(manifest, stage)
        expected = (stage.attempt_inputs or {}).get("staged_input_sha256")
        if existing is not None and isinstance(expected, dict) and expected:
            return existing
        try:
            return self._prepare_stage_dir(manifest, stage)
        except Exception:
            return None

    def _prepare_stage_dir(
        self, manifest: UnifiedVaspManifest, stage: UnifiedStage
    ) -> Path:
        """Generate only the requested stage directory.

        ``_prepare_data`` generates the complete input tree.  Rewriting every
        stage there would overwrite already-bound downstream inputs (for
        example static's POSCAR) when a later stage is prepared.  This helper
        keeps preparation stage-local and leaves previously bound directories
        byte-for-byte untouched.
        """
        spec = cast(PeriodicScientificSpec, manifest.scientific_spec)
        profile = get_profile(spec.profile)
        structure = self._workspace_path(spec.structure_path)
        problems = self.application.validate_inputs(structure, spec.profile)
        if problems:
            raise ValueError("; ".join(problems))
        root = self._stage_root(manifest)
        for index, stage_spec in enumerate(
            periodic_stage_specs(spec.profile), start=1
        ):
            if stage_spec.name != stage.name:
                continue
            output = root / f"{index:02d}_{stage.name}"
            self._revalidate_application_paths(manifest, output)
            self.application.generator.generate_stage(
                structure_path=structure,
                profile=profile,
                stage=stage.name,
                output_dir=output,
                spec_overrides=dict(spec.scientific_overrides),
            )
            self._revalidate_application_paths(manifest, output)
            return output
        raise ValueError(f"stage {stage.name!r} is not part of profile {spec.profile!r}")

    def _required_upload_inputs(
        self,
        manifest: UnifiedVaspManifest,
        stage: UnifiedStage,
        stage_dir: Path,
    ) -> tuple[dict[str, str], list[str]]:
        """Map the stage authorization onto the files the job must receive.

        Only artifacts that this stage actually depends on -- its upstream
        ``required_outputs`` that are not ordinary VASP inputs -- are bound
        here.  Ordinary POSCAR/INCAR/KPOINTS/POTCAR staging keeps its
        existing separate policy so the upload set never silently grows.
        """
        from photomatagent.scientific.remote.scheduler import (
            REMOTE_INPUT_ALLOWLIST,
        )

        expected = (stage.attempt_inputs or {}).get("staged_input_sha256")
        if not isinstance(expected, dict) or not expected:
            return {}, []
        upstream_names = set(self._stage_required_outputs(manifest, stage))
        if not upstream_names:
            return {}, []
        selected: dict[str, str] = {}
        errors: list[str] = []
        for name in sorted(upstream_names):
            if name in {"POSCAR", "INCAR", "KPOINTS", "CONTCAR", "POTCAR.meta"}:
                # CONTCAR is materialized as this stage's POSCAR (already
                # bound under "POSCAR" above); uploading it again would ship
                # an unrequested duplicate under a second name.
                continue
            if name not in REMOTE_INPUT_ALLOWLIST:
                errors.append(
                    f"upstream artifact {name!r} is not an uploadable VASP "
                    "input; refusing to submit an incomplete stage"
                )
                continue
            digest = expected.get(name)
            if not isinstance(digest, str) or len(digest) != 64:
                errors.append(
                    f"upstream artifact {name!r} is required by stage "
                    f"{stage.name!r} but is not bound in the authorized "
                    "inputs; re-prepare and re-approve the stage"
                )
                continue
            path = stage_dir / name
            if path.is_symlink() or not path.is_file():
                errors.append(
                    f"authorized upstream artifact {name!r} is missing or "
                    "symlinked in the stage directory"
                )
                continue
            if _sha256_file(path) != digest:
                errors.append(
                    f"authorized upstream artifact {name!r} changed after "
                    "approval"
                )
                continue
            selected[name] = digest
        return selected, errors

    def _stage_required_outputs(
        self, manifest: UnifiedVaspManifest, stage: UnifiedStage
    ) -> tuple[str, ...]:
        """Required upstream artifacts for one stage.

        The manifest's explicit ``required_outputs`` wins; it is populated
        from the same profile DAG, and this fallback keeps stages built
        without explicit outputs upload-correct instead of silently empty.
        """
        if stage.required_outputs:
            return tuple(stage.required_outputs)
        spec = cast(PeriodicScientificSpec, manifest.scientific_spec)
        for stage_spec in periodic_stage_specs(spec.profile):
            if stage_spec.name == stage.name:
                return tuple(stage_spec.required_outputs)
        return ()

    def _existing_stage_dir(
        self, manifest: UnifiedVaspManifest, stage: UnifiedStage
    ) -> Path | None:
        """Locate an already-generated stage directory without rewriting it."""
        profile = cast(PeriodicScientificSpec, manifest.scientific_spec).profile
        root = self._stage_root(manifest)
        for index, stage_spec in enumerate(
            periodic_stage_specs(profile), start=1
        ):
            if stage_spec.name == stage.name:
                candidate = root / f"{index:02d}_{stage.name}"
                return candidate if candidate.is_dir() else None
        return None

    def _staged_input_hashes(self, stage_dir: Path) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for name in ("POSCAR", "INCAR", "KPOINTS", "CHGCAR", "XDATCAR"):
            path = stage_dir / name
            if path.is_file() and not path.is_symlink():
                hashes[name] = _sha256_file(path)
        return hashes

    def _verify_authorized_inputs(
        self,
        stage_dir: Path,
        stage: UnifiedStage,
    ) -> list[str]:
        """Verify staged inputs against the execution identity already bound."""
        expected = stage.attempt_inputs or {}
        expected_files = expected.get("staged_input_sha256")
        if not isinstance(expected_files, dict) or not expected_files:
            return []
        errors: list[str] = []
        for name, digest in expected_files.items():
            path = stage_dir / str(name)
            if path.is_symlink() or not path.is_file():
                errors.append(f"authorized input {name!r} is missing or symlinked")
            elif _sha256_file(path) != str(digest):
                errors.append(f"authorized input {name!r} changed after approval")
        expected_directory = expected.get("input_directory_sha256")
        if isinstance(expected_directory, str) and expected_directory:
            current = canonical_input_hash(stage_dir)
            if current != expected_directory:
                errors.append(
                    "execution input directory changed after approval; "
                    "a new authorization is required"
                )
        return errors

    async def prepare_stage_inputs(
        self, manifest: UnifiedVaspManifest, stage: UnifiedStage
    ) -> OperationResult:
        """Materialize deterministic local inputs for one stage.

        Upstream artifacts are copied only from already-VALIDATED stages in
        this workflow.  The returned ``attempt_inputs`` binds every source and
        destination hash into the execution fingerprint.
        """
        spec = cast(PeriodicScientificSpec, manifest.scientific_spec)
        stage_spec = next(
            (
                item
                for item in periodic_stage_specs(spec.profile)
                if item.name == stage.name
            ),
            None,
        )
        if stage_spec is None:
            return OperationResult(
                ok=False,
                errors=[f"stage {stage.name!r} is not part of profile {spec.profile!r}"],
            )

        expected = stage.attempt_inputs or {}
        expected_files = expected.get("staged_input_sha256")
        stage_dir = self._existing_stage_dir(manifest, stage)
        if isinstance(expected_files, dict) and expected_files:
            if stage_dir is None:
                return OperationResult(
                    ok=False,
                    errors=[
                        "authorized stage inputs are missing; re-prepare the "
                        "workflow and authorize a new execution identity"
                    ],
                )
            errors = self._verify_authorized_inputs(stage_dir, stage)
            if errors:
                return OperationResult(ok=False, errors=errors)
            return OperationResult(
                ok=True,
                data={"directory": str(stage_dir), "attempt_inputs": {}},
            )

        # First authorization pass: generate only this stage directory, then
        # stage the upstream outputs.  Never regenerate sibling stage dirs.
        try:
            stage_dir = self._prepare_stage_dir(manifest, stage)
        except Exception as exc:
            return OperationResult(
                ok=False,
                errors=[f"{type(exc).__name__}: {exc}"],
            )
        upstream_artifacts: list[dict[str, Any]] = []
        try:
            workspace = Workspace(self.application.workspace)
            for dependency in stage_spec.depends_on:
                upstream = next(
                    (item for item in manifest.stages if item.name == dependency),
                    None,
                )
                if upstream is None:
                    raise ValueError(f"required upstream stage {dependency!r} is missing")
                if upstream.state is not WorkflowState.VALIDATED:
                    raise ValueError(
                        f"required upstream stage {dependency!r} is "
                        f"{upstream.state.value}, not VALIDATED"
                    )
                upstream_dir = self._managed_workflow_path(
                    "results", manifest.workflow_id, upstream.name
                )
                revalidate_managed_vasp_path(workspace, upstream_dir)
                if upstream_dir.is_symlink() or not upstream_dir.is_dir():
                    raise ValueError(
                        f"validated upstream results for {dependency!r} are absent"
                    )
                request_id = self._request_id(manifest, upstream)
                for artifact in stage_spec.required_outputs:
                    source = upstream_dir / artifact
                    if source.is_symlink() or not source.is_file():
                        raise ValueError(
                            f"required upstream artifact {artifact!r} from "
                            f"{dependency!r} is missing or symlinked"
                        )
                    try:
                        source.relative_to(workspace.root)
                    except ValueError as exc:
                        raise ValueError(
                            f"upstream artifact {source} is outside the workspace"
                        ) from exc
                    destination_name = "POSCAR" if artifact == "CONTCAR" else artifact
                    destination = stage_dir / destination_name
                    shutil.copy2(source, destination)
                    source_digest = _sha256_file(source)
                    destination_digest = _sha256_file(destination)
                    upstream_artifacts.append(
                        {
                            "source_stage": dependency,
                            "source_request_id": request_id,
                            "artifact": artifact,
                            "destination": destination_name,
                            "source_sha256": source_digest,
                            "destination_sha256": destination_digest,
                        }
                    )
        except Exception as exc:
            return OperationResult(
                ok=False,
                errors=[f"{type(exc).__name__}: {exc}"],
                evidence_gaps=["upstream artifact staging was blocked"],
            )

        attempt_inputs = {
            "input_directory_sha256": canonical_input_hash(stage_dir),
            "staged_input_sha256": self._staged_input_hashes(stage_dir),
            "upstream_artifacts": upstream_artifacts,
        }
        return OperationResult(
            ok=True,
            data={"directory": str(stage_dir), "attempt_inputs": attempt_inputs},
        )

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
            directory = self._existing_stage_dir(manifest, manifest.stages[0])
            if directory is None:
                stage_dirs = self._prepare_data(manifest).get("stages", [])
                if not stage_dirs:
                    return [
                        "no stage input directories prepared; run vasp.prepare first"
                    ]
                directory = Path(stage_dirs[0]["directory"])
        except Exception as exc:
            return [f"{type(exc).__name__}: {exc}"]
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
        prepared_inputs = await self.prepare_stage_inputs(manifest, stage)
        if not prepared_inputs.ok:
            return SubmissionResult(
                ok=False,
                request_id=request_id,
                submitted=False,
                errors=prepared_inputs.errors,
                evidence_gaps=prepared_inputs.evidence_gaps,
            )
        stage_dir = Path(str(prepared_inputs.data["directory"]))
        spec = cast(PeriodicScientificSpec, manifest.scientific_spec)
        profile = get_profile(spec.profile)
        # The exact bytes authorized for this execution identity must be the
        # bytes that reach the cluster.  An approved list that cannot be
        # satisfied locally is a hard stop, not a silent partial upload.
        authorized_inputs, authorization_errors = self._required_upload_inputs(
            manifest, stage, stage_dir
        )
        if authorization_errors:
            return SubmissionResult(
                ok=False,
                request_id=request_id,
                submitted=False,
                errors=authorization_errors,
            )
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
                required_input_sha256=authorized_inputs,
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

    @staticmethod
    def _record_scheduler_state(record: Any) -> HPCJobState:
        """Return the observed scheduler enum, never a default SUBMITTED."""
        scheduler_state = getattr(record, "scheduler_state", None)
        if isinstance(scheduler_state, HPCJobState):
            return scheduler_state
        state = getattr(record, "state", None)
        if state in {
            JobLifecycleState.COMPLETED,
            JobLifecycleState.COLLECTED,
            JobLifecycleState.VALIDATED,
        }:
            return HPCJobState.COMPLETED
        if isinstance(state, JobLifecycleState):
            return HPCJobState(state.value)
        try:
            return HPCJobState(str(state))
        except (TypeError, ValueError):
            return HPCJobState.UNKNOWN

    @staticmethod
    def _next_stage_from_states(
        manifest: UnifiedVaspManifest,
        stage_states: dict[str, str],
    ) -> str | None:
        for stage in manifest.stages:
            if stage_states.get(stage.name) == "VALIDATED":
                continue
            if all(
                stage_states.get(dependency) == "VALIDATED"
                for dependency in stage.depends_on
            ):
                return stage.name
        return None

    async def collect(self, manifest: UnifiedVaspManifest) -> CollectionResult:
        stage_states: dict[str, str] = {}
        pending_stages: list[str] = []
        validated_stages: list[str] = []
        evidence: list[ScientificEvidence] = []
        evidence_gaps: list[str] = []
        errors: list[str] = []
        diagnostics: list[dict[str, Any]] = []

        for stage in manifest.stages:
            rid = self._request_id(manifest, stage)
            record = self.session.registry.get(rid)
            if record is None or record.job_id is None:
                stage_states[stage.name] = "NOT_FOUND"
                pending_stages.append(stage.name)
                evidence_gaps.append(
                    f"{stage.name}: no job_id in registry; not submitted"
                )
                continue

            # Production JobRecords always carry ``state``.  A minimal test
            # surrogate may omit it; such a record already has a job_id, so
            # collection proceeds through the normal application validation
            # path instead of being classed as unsubmitted.
            record_state = getattr(record, "state", None)

            if record_state is not None and record_state in {
                JobLifecycleState.FAILED,
                JobLifecycleState.TIMEOUT,
                JobLifecycleState.OUT_OF_MEMORY,
                JobLifecycleState.CANCELLED,
            }:
                message = (
                    f"{stage.name}: scheduler state {record_state.value}; "
                    "no result evidence was produced"
                )
                stage_states[stage.name] = record_state.value
                errors.append(message)
                evidence_gaps.append(message)
                diagnostics.append(
                    await self._collect_stage_diagnostics(
                        manifest, stage, record, record_state.value
                    )
                )
                continue

            if record_state is JobLifecycleState.UNKNOWN_RECONCILIATION_REQUIRED:
                message = (
                    f"{stage.name}: {record_state.value}; reconcile required "
                    "before collection"
                )
                stage_states[stage.name] = record_state.value
                pending_stages.append(stage.name)
                evidence_gaps.append(message)
                continue

            if record_state in {
                JobLifecycleState.PREPARED,
                JobLifecycleState.PREFLIGHT_PASSED,
                JobLifecycleState.SUBMITTING,
                JobLifecycleState.SUBMITTED,
                JobLifecycleState.PENDING,
                JobLifecycleState.RUNNING,
            }:
                stage_states[stage.name] = record_state.value
                pending_stages.append(stage.name)
                evidence_gaps.append(
                    f"{stage.name}: {record_state.value}; result is not ready "
                    "to collect"
                )
                continue

            if record_state is not None and record_state not in {
                JobLifecycleState.COMPLETED,
                JobLifecycleState.COLLECTED,
                JobLifecycleState.VALIDATED,
            }:
                # Unknown lifecycle values are reconciliation ambiguities, not
                # scheduler success and not missing/unsubmitted stages.
                stage_states[stage.name] = record_state.value
                pending_stages.append(stage.name)
                evidence_gaps.append(
                    f"{stage.name}: unrecognized lifecycle state "
                    f"{record_state.value!r}; reconcile required"
                )
                continue

            scheduler_state = self._record_scheduler_state(record)
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
                        state=scheduler_state,
                    ),
                    local_dir=local_dir,
                    profile_name=cast(
                        PeriodicScientificSpec, manifest.scientific_spec
                    ).profile,
                    stage=stage.name,
                )
                self._revalidate_application_paths(manifest, local_dir)
            except Exception as exc:
                message = f"{stage.name}: {type(exc).__name__}: {exc}"
                stage_states[stage.name] = "UNKNOWN"
                errors.append(message)
                evidence_gaps.append(message)
                continue

            if not report.get("scientifically_valid", False):
                problems = [
                    str(problem)
                    for problem in report.get("validation_problems", [])
                ]
                message = f"{stage.name}: " + (
                    "; ".join(problems) if problems else "validation failed"
                )
                stage_states[stage.name] = "VALIDATION_FAILED"
                evidence_gaps.append(message)
                self.session.mark_result_state(
                    rid, collected=True, validated=False, error=message
                )
                diagnostics.append(
                    await self._collect_stage_diagnostics(
                        manifest,
                        stage,
                        record,
                        "VALIDATION_FAILED",
                    )
                )
                continue

            spec = cast(PeriodicScientificSpec, manifest.scientific_spec)
            try:
                parent_path = self._workspace_path(spec.structure_path)
                parent_structure = _load_trusted_structure(parent_path)
                parent_digest = structure_hash(parent_structure)
                parent_sha256 = _sha256_file(parent_path)
                structure_path = _trusted_result_structure(local_dir)
                structure = _load_trusted_structure(structure_path)
            except Exception as exc:
                message = (
                    f"{stage.name}: trusted structure scope unavailable: "
                    f"{type(exc).__name__}: {exc}"
                )
                stage_states[stage.name] = "VALIDATION_FAILED"
                evidence_gaps.append(message)
                self.session.mark_result_state(
                    rid, collected=True, validated=False, error=message
                )
                continue

            parsed = report.get("parsed", {})
            energy = parsed.get("final_energy_eV") if isinstance(parsed, dict) else None
            if not isinstance(energy, (int, float)) or isinstance(energy, bool):
                message = (
                    f"{stage.name}: validated result has no parsed total energy"
                )
                stage_states[stage.name] = "VALIDATION_FAILED"
                evidence_gaps.append(message)
                self.session.mark_result_state(
                    rid, collected=True, validated=False, error=message
                )
                continue

            digest = structure_hash(structure)
            input_sha256 = _sha256_file(structure_path)
            parent_candidate_id = (
                f"cand_{parent_digest[:24]}"
                if f"cand_{parent_digest[:24]}" != f"cand_{digest[:24]}"
                else None
            )
            parent_structure_hash = parent_digest if parent_candidate_id else None
            # Stable identity: a re-collection of the same request/stage/output
            # must be idempotent, while a changed output or property creates a
            # new evidence identity instead of hiding the change behind the old
            # record.
            evidence_identity = "|".join(
                [
                    manifest.workflow_id,
                    stage.name,
                    rid,
                    "total_energy",
                    f"{float(energy):.12g}",
                    digest,
                    input_sha256,
                ]
            )
            evidence_id = (
                "sev_" + hashlib.sha256(evidence_identity.encode("utf-8")).hexdigest()[:12]
            )
            evidence.append(
                ScientificEvidence(
                    id=evidence_id,
                    subject=structure.composition.reduced_formula,
                    property="total_energy",
                    value=float(energy),
                    unit="eV",
                    source="SCNet VASP calculation",
                    source_type="dft_calculation",
                    method=f"VASP (profile={spec.profile}, stage={stage.name})",
                    fidelity="dft",
                    candidate_id=f"cand_{digest[:24]}",
                    structure_hash=digest,
                    summary=(
                        f"VASP total energy {float(energy):.4f} eV "
                        f"(workflow {manifest.workflow_id}, stage {stage.name})"
                    ),
                    limitations="; ".join(report.get("validation_problems", [])),
                    provenance={
                        "tool": "vasp.collect",
                        "workflow_id": manifest.workflow_id,
                        "stage": stage.name,
                        "job_id": record.job_id,
                        "profile": spec.profile,
                        "scheduler_state": report.get("scheduler_state"),
                        "job_lifecycle_state": (
                            record_state.value
                            if isinstance(record_state, JobLifecycleState)
                            else getattr(record, "state", None)
                        ),
                        "input_structure_path": str(structure_path),
                        "input_sha256": input_sha256,
                        "output_sha256": input_sha256,
                        "structure_hash": digest,
                        "parent_candidate_id": parent_candidate_id,
                        "parent_structure_hash": parent_structure_hash,
                        "parent_input_sha256": parent_sha256,
                    },
                )
            )
            stage_states[stage.name] = "VALIDATED"
            validated_stages.append(stage.name)
            self.session.mark_result_state(
                rid, collected=True, validated=True, evidence=len(evidence)
            )

        validated = bool(manifest.stages) and not errors and all(
            stage_states.get(stage.name) == "VALIDATED"
            for stage in manifest.stages
        )
        next_stage = self._next_stage_from_states(manifest, stage_states)
        data: dict[str, Any] = {
            "pending_stages": pending_stages,
            "validated_stages": validated_stages,
        }
        if any(stage.name in {"band", "dos"} for stage in manifest.stages):
            data["unsupported_property_gaps"] = {
                "band_gap": (
                    "UNSUPPORTED: no fixed periodic SOC/noncollinear "
                    "band/static sample is available; no numeric band_gap "
                    "evidence is emitted (task 14 LIMITED)"
                )
            }
        if next_stage is not None:
            data["next_stage"] = next_stage
        if errors:
            data["blocked_reason"] = "; ".join(errors)
        elif pending_stages:
            data["blocked_reason"] = "pending: " + ", ".join(pending_stages)
        if diagnostics:
            data["diagnostics"] = diagnostics
            data["next_action"] = _diagnostic_next_action(diagnostics)
        return CollectionResult(
            ok=not errors,
            validated=validated,
            evidence=evidence,
            evidence_gaps=evidence_gaps,
            errors=errors,
            stage_states=stage_states,
            data=data,
        )

    async def _collect_stage_diagnostics(
        self,
        manifest: UnifiedVaspManifest,
        stage: UnifiedStage,
        record: Any,
        stage_state: str,
    ) -> dict[str, Any]:
        """Gather bounded failure diagnostics for one non-validated stage.

        Diagnostics are explicitly separated from scientific evidence: they
        never enter ``evidence``, never change the scheduler/lifecycle state,
        never trigger a resubmit, and a missing or unreadable log is reported
        as a typed gap instead of being inferred from error text.
        """
        job_id = str(getattr(record, "job_id", "") or "")
        remote_directory = getattr(record, "remote_directory", None)
        base: dict[str, Any] = {
            "stage": stage.name,
            "request_id": getattr(record, "request_id", None),
            "job_id": job_id,
            "stage_state": stage_state,
            "scheduler_state": self._record_scheduler_state(record).value
            if record is not None
            else "UNKNOWN",
            "remote_directory": remote_directory,
            "diagnostic_status": "UNAVAILABLE",
            "artifacts": [],
            "missing_artifacts": [],
            "diagnostic_gaps": [],
        }
        if not job_id or not remote_directory:
            base["diagnostic_gaps"].append(
                "no confirmed job id / remote directory; failure logs are "
                "not addressable from the local registry"
            )
            return base
        diagnostic_dir = self._managed_workflow_path(
            "diagnostics", manifest.workflow_id, stage.name
        )
        try:
            revalidate_managed_vasp_path(
                Workspace(self.application.workspace), diagnostic_dir
            )
            report = await self.application.collect_diagnostics(
                job_ref=RemoteJobRef(
                    backend="scnet",
                    application="vasp",
                    job_id=job_id,
                    remote_directory=remote_directory,
                    state=self._record_scheduler_state(record),
                ),
                local_dir=diagnostic_dir,
                diagnostic_dir=diagnostic_dir,
            )
        except Exception as exc:
            base["diagnostic_status"] = "QUERY_FAILED"
            base["diagnostic_gaps"].append(
                f"diagnostic collection failed: {type(exc).__name__}: {exc}"
            )
            return base
        # ``requested`` travels with the gaps so a consumer can tell an
        # unqueried name apart from a queried-but-missing one.
        for key in ("requested", "artifacts", "missing_artifacts", "diagnostic_gaps"):
            value = report.get(key)
            if isinstance(value, list):
                base[key] = value
        status = report.get("diagnostic_status")
        if isinstance(status, str) and status:
            base["diagnostic_status"] = status
        return base

    async def report(
        self, manifest: UnifiedVaspManifest, request: ReportRequest
    ) -> ReportResult:
        return ReportResult(ok=True, report_kind=request.kind)


def _diagnostic_next_action(diagnostics: list[dict[str, Any]]) -> str:
    """Deterministic next step for a failed-stage diagnostic set.

    A failed job whose logs are still absent must pause and be reported; the
    caller must not be nudged into guessing from source code or source
    exploration.
    """
    collected = [
        item
        for item in diagnostics
        if item.get("diagnostic_status") == "COLLECTED" and item.get("artifacts")
    ]
    if collected:
        stages = ", ".join(str(item.get("stage")) for item in collected)
        return (
            "READ_FAILURE_DIAGNOSTICS: inspect the collected OUTCAR/"
            f"vasprun.xml/OSZICAR excerpts for {stages}; they are not "
            "scientific evidence and do not authorize a resubmit"
        )
    return (
        "PAUSE_FOR_MISSING_DIAGNOSTICS: no failure log is available locally; "
        "report the stage as FAILED with UNKNOWN remote cause and wait for "
        "log retrieval -- do not infer the error from source code and do not "
        "resubmit without the normal preflight/approval gates"
    )


def _load_trusted_structure(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError("structure artifact is missing or symlinked")
    from pymatgen.core import Structure

    return Structure.from_file(str(path))


def _trusted_result_structure(local_dir: Path) -> Path:
    """Select a validated final geometry from the contained result directory."""
    for name in ("CONTCAR", "POSCAR"):
        path = local_dir / name
        if path.is_file() and not path.is_symlink():
            return path
    raise ValueError("validated VASP result has no CONTCAR/POSCAR structure")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
