"""Freeze and run one bounded paper example through the existing sandbox."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import AwareDatetime, Field, FiniteFloat, SerializerFunctionWrapHandler, model_serializer, model_validator

from photomatagent.capability_acquisition.container_backend import PodmanSandboxBackend
from photomatagent.capability_acquisition.models import (
    ExecutionRecord, ExecutionRequest, SourceTreeLayout, StrictModel, validate_source_tree_path,
)
from photomatagent.capability_acquisition.sandbox import SandboxService
from photomatagent.capability_acquisition.web_sources import safe_result_url, tool_result
from photomatagent.errors import ToolExecutionError
from photomatagent.persistence.atomic import atomic_write
from photomatagent.scientific.capabilities.literature.documents import DocumentService
from photomatagent.tools.base import Tool, ToolResult
from photomatagent.tools.exposure import ToolExposure
from photomatagent.workspace import Workspace


class ManagedReferenceAnchor(StrictModel):
    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    object_id: str = Field(min_length=1, max_length=200)
    sentence: str = Field(min_length=1, max_length=500)


class NumericReference(StrictModel):
    metric: str = Field(min_length=1, max_length=100)
    expected: FiniteFloat
    absolute_tolerance: FiniteFloat = Field(ge=0)
    unit: str = Field(max_length=40)
    source_anchor: str = Field(min_length=1, max_length=500)
    managed_value: ManagedReferenceAnchor | None = None
    managed_tolerance: ManagedReferenceAnchor | None = None
    tolerance_rationale: str = Field(default="", max_length=500)

    @model_serializer(mode="wrap")
    def serialize_reference(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return _without_empty(handler(self), "managed_value", "managed_tolerance", "tolerance_rationale")


def _without_empty(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        if not payload.get(key):
            payload.pop(key, None)
    return payload


class ReproductionAsset(StrictModel):
    input_key: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    role: Literal["data", "weights", "patch"]
    source_url: str = Field(min_length=1, max_length=2000)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=1, le=4_000_000)
    version: str = Field(min_length=1, max_length=250)
    license: str = Field(min_length=1, max_length=120)
    acquired_at: AwareDatetime


class ReproductionChange(StrictModel):
    kind: Literal["interface", "scientific"]
    rationale: str = Field(min_length=1, max_length=500)
    patch_input_key: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")


class ReproductionSpec(StrictModel):
    schema_version: Literal[1] = 1
    source: str = Field(min_length=1, max_length=2000)
    paper_anchor: str = Field(min_length=1, max_length=400)
    source_commit: str = Field(default="", pattern=r"^([0-9a-f]{40}|[0-9a-f]{64})?$")
    implementation: Literal["author_code", "third_party", "reimplementation"]
    script_path: str = Field(min_length=1, max_length=300)
    environment_id: str = Field(pattern=r"^env_[0-9a-f]{24}$")
    argv: tuple[str, ...] = Field(default=(), max_length=24)
    timeout_seconds: int = Field(default=60, ge=1, le=120)
    reference: NumericReference | None = None

    source_root: str = Field(default="", max_length=300)
    source_files: tuple[str, ...] = Field(default=(), max_length=64)
    source_bundle_id: str = Field(default="", pattern=r"^([0-9a-f]{64})?$")
    input_refs: dict[str, str] = Field(default_factory=dict, max_length=16)
    assets: tuple[ReproductionAsset, ...] = Field(default=(), max_length=16)
    target: str = Field(default="", max_length=300)
    requested_scope: Literal["IMPORT", "EXAMPLE", "PAPER_SUBSET", "PAPER_FULL", "REIMPLEMENTATION"] | None = None
    seed: int | None = Field(default=None, ge=0, le=2**64 - 1)
    seed_flag: str = Field(default="", pattern=r"^(?:--[A-Za-z][A-Za-z0-9-]{0,39})?$")
    environment_profile: Literal["python", "native", "cuda"] = "python"
    build_backend: str = Field(default="", max_length=100)
    repair_limit: int | None = Field(default=None, ge=0, le=8)
    changes: tuple[ReproductionChange, ...] = Field(default=(), max_length=8)
    adaptation_differences: tuple[str, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def validate_contract(self) -> ReproductionSpec:
        assets = {asset.input_key: asset for asset in self.assets}
        if len(assets) != len(self.assets) or not assets.keys() <= self.input_refs.keys():
            raise ValueError("REPRODUCTION_ASSET_BINDING_INVALID")
        if any(not safe_result_url(asset.source_url) for asset in self.assets):
            raise ValueError("REPRODUCTION_ASSET_SOURCE_INVALID")
        if len(self.changes) > (self.repair_limit if self.repair_limit is not None else 2):
            raise ValueError("REPRODUCTION_REPAIR_LIMIT_EXCEEDED")
        if any(change.patch_input_key not in assets or assets[change.patch_input_key].role != "patch" for change in self.changes):
            raise ValueError("REPRODUCTION_PATCH_REQUIRED")
        if bool(self.seed_flag) != (self.seed is not None) or (self.seed_flag and (
                self.argv.count(self.seed_flag) != 1 or (self.seed_flag, str(self.seed)) not in zip(self.argv, self.argv[1:]))):
            raise ValueError("REPRODUCTION_SEED_NOT_BOUND_TO_ARGV")
        return self

    @model_serializer(mode="wrap")
    def serialize_spec(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        payload: dict[str, Any] = handler(self)
        if self.environment_profile == "python":
            payload.pop("environment_profile", None)
        for key in ("seed", "repair_limit"):
            if payload.get(key) is None:
                payload.pop(key, None)
        return _without_empty(payload, "source_files", "source_root", "source_bundle_id", "input_refs", "assets",
                              "target", "requested_scope", "seed_flag", "build_backend", "changes", "adaptation_differences")


class ReproductionPlan(StrictModel):
    spec: ReproductionSpec
    script_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    source_hashes: dict[str, str] = Field(default_factory=dict, max_length=64)
    input_hashes: dict[str, str] = Field(default_factory=dict, max_length=16)

    @model_serializer(mode="wrap")
    def serialize_plan(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        payload: dict[str, Any] = handler(self)
        if not self.source_hashes:
            payload.pop("source_hashes", None)
        if not self.input_hashes:
            payload.pop("input_hashes", None)
        return payload

    def source_bindings(self) -> tuple[SourceTreeLayout | None, dict[str, str], dict[str, str]]:
        if not self.source_hashes:
            return None, dict(self.spec.input_refs), dict(self.input_hashes)
        script = Path(self.spec.script_path)
        root = Path(self.spec.source_root) if self.spec.source_root else script.parent
        paths = {f"file_{i}": name for i, name in enumerate(self.source_hashes)}
        layout = SourceTreeLayout(entry=script.relative_to(root).as_posix(), inputs=paths)
        return (layout, {**self.spec.input_refs, **{key: (root / name).as_posix() for key, name in paths.items()}},
                {**self.input_hashes, **{key: self.source_hashes[name] for key, name in paths.items()}})

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()

    @property
    def execution_id(self) -> str:
        return f"exec_{self.sha256[:32]}"


def check_reference(record: ExecutionRecord, reference: NumericReference | None) -> dict[str, Any]:
    if reference is None or record.state != "COMPLETED" or record.exit_code != 0:
        return {"result": "UNKNOWN"}
    try:
        output = json.loads(record.stdout)
        value = output[reference.metric]
        unit = output["unit"]
        if type(value) not in (int, float) or unit != reference.unit:
            raise ValueError("Metric or unit mismatch")
        difference = abs(value - reference.expected)
        if difference == float("inf") or difference != difference:
            raise ValueError("Nonfinite metric")
    except (KeyError, IndexError, TypeError, ValueError, RecursionError, OverflowError):
        return {"result": "UNKNOWN", "reason": "REFERENCE_OUTPUT_UNAVAILABLE"}
    return {"result": "PASS" if difference <= reference.absolute_tolerance else "FAIL",
            "observed": value, "unit": unit, "absolute_error": difference,
            "reference": reference.model_dump(mode="json"),
            "scope": "comparison_to_supplied_reference"}


def configured_reproduction_sandbox(workspace: Workspace) -> SandboxService | None:
    """Opt-in ordinary-chat adapter for an existing workspace Podman store."""
    relative = os.environ.get("PHOTOMATAGENT_REPRODUCTION_ENV_STORAGE", "")
    if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        return None
    try:
        storage = workspace.resolve(relative)
    except ToolExecutionError:
        return None
    if not storage.is_dir() or workspace.relative(storage) != Path(relative).as_posix():
        return None
    backend = PodmanSandboxBackend(storage)
    if backend.executable is None:
        return None
    return SandboxService(
        workspace=workspace.root, trial_id="reproduction",
        genome_sha256=hashlib.sha256(b"photomatagent-reproduction").hexdigest(),
        scripts={}, dynamic_script_prefix="tmp/reproduction",
        backend=backend, restore_persisted_podman=True,
    )


class ReproductionService:
    def __init__(
        self, workspace: Workspace, sandbox: SandboxService | None, *, scratch: str | None = None,
    ) -> None:
        self.workspace = workspace
        self.sandbox = sandbox if isinstance(sandbox, SandboxService) else None
        prefix = self.sandbox.dynamic_script_prefix if self.sandbox is not None else None
        self._scratch = Path(prefix or scratch or "tmp/reproduction")
        self._scope = hashlib.sha256(self._scratch.as_posix().encode()).hexdigest()[:24]
        self._documents = DocumentService(workspace)

    def _managed_sentence(self, anchor: ManagedReferenceAnchor) -> str:
        from photomatagent.scientific.capabilities.literature.evidence import _sentences

        artifact, obj = self._documents.cached_anchored_object(
            anchor.document_id, anchor.revision, anchor.source_sha256, anchor.object_id,
        )
        if artifact.media_type != "application/pdf" or not safe_result_url(artifact.source_url):
            raise ValueError("REPRODUCTION_REFERENCE_SOURCE_UNVERIFIED")
        if _sentences(obj.text).count(anchor.sentence) != 1:
            raise ValueError("REPRODUCTION_REFERENCE_SENTENCE_NOT_BOUND")
        return anchor.sentence

    def _reference_authority(self, reference: NumericReference | None) -> str:
        from photomatagent.scientific.capabilities.literature.evidence import extract_evidence_from_text

        if reference is None:
            return "NO_REFERENCE"
        if reference.managed_value is None:
            if reference.managed_tolerance is not None:
                raise ValueError("REPRODUCTION_TOLERANCE_WITHOUT_VALUE")
            return "SUPPLIED"
        values = [item for item in extract_evidence_from_text(self._managed_sentence(reference.managed_value))
                  if item.property == reference.metric and item.unit == reference.unit]
        if (len(values) != 1 or values[0].method not in {"reported experimental value", "calculated or predicted value"}
                or values[0].provenance.get("quantity_ambiguous") or type(values[0].value) not in (int, float)
                or values[0].value != reference.expected):
            raise ValueError("REPRODUCTION_REFERENCE_VALUE_NOT_BOUND")
        tolerance = reference.managed_tolerance
        if tolerance is None:
            return "MANAGED_VALUE_ONLY"
        if (tolerance.document_id, tolerance.revision) != (reference.managed_value.document_id, reference.managed_value.revision):
            raise ValueError("REPRODUCTION_TOLERANCE_VERSION_MISMATCH")
        sentence = self._managed_sentence(tolerance)
        match = re.fullmatch(
            rf"(?:the\s+)?absolute(?:\s+comparison)?\s+tolerance\s+(?:is|was|of|=)\s+"
            rf"(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+{re.escape(reference.unit)}\.?",
            sentence, re.IGNORECASE,
        )
        if match is None or float(match[1]) != reference.absolute_tolerance:
            raise ValueError("REPRODUCTION_TOLERANCE_NOT_BOUND")
        return "MANAGED_VALUE_AND_TOLERANCE"

    def _plan_path(self, digest: str) -> Path:
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("PLAN_ID_INVALID")
        return self.workspace.resolve(f".photomatagent/reproduction/{self._scope}/plans/{digest}.json", must_exist=False)

    def _run_pointer(self, run_id: str) -> Path:
        if (not run_id.startswith("exec_") or len(run_id) != 37
                or any(ch not in "0123456789abcdef" for ch in run_id[5:])):
            raise ValueError("RUN_ID_INVALID")
        return self.workspace.resolve(
            f".photomatagent/reproduction/{self._scope}/runs/{run_id}", must_exist=False,
        )

    def _source_hashes(self, spec: ReproductionSpec, script: Path) -> dict[str, str]:
        if not spec.source_files:
            if spec.source_root:
                raise ValueError("SOURCE_TREE_FILES_REQUIRED")
            return {}
        relative_root = Path(spec.source_root) if spec.source_root else script.parent
        if (relative_root.is_absolute() or ".." in relative_root.parts
                or not relative_root.is_relative_to(self._scratch)
                or not script.is_relative_to(relative_root)):
            raise ValueError("SOURCE_TREE_OUTSIDE_TASK_SCRATCH")
        layout = SourceTreeLayout(
            entry=script.relative_to(relative_root).as_posix(),
            inputs={str(i): name for i, name in enumerate(spec.source_files)},
        )
        total = 0
        result = {}
        for name in sorted(layout.inputs.values()):
            relative = (relative_root / name).as_posix()
            path = self.workspace.resolve(relative)
            if self.workspace.relative(path) != relative or not path.is_file():
                raise ValueError("SOURCE_TREE_FILE_UNAVAILABLE")
            with path.open("rb") as handle:
                data = handle.read(32 * 1024 * 1024 - total + 1)
            total += len(data)
            if total > 32 * 1024 * 1024:
                raise ValueError("SOURCE_TREE_TOO_LARGE")
            result[validate_source_tree_path(name)] = hashlib.sha256(data).hexdigest()
        return result

    def _verify_bundle(self, spec: ReproductionSpec, script_hash: str, hashes: dict[str, str]) -> None:
        if not spec.source_bundle_id:
            return
        metadata = self.workspace.resolve(
            f".photomatagent/reproduction/sources/bundle-{spec.source_bundle_id}.json"
        )
        manifest = json.loads(metadata.read_bytes())
        root = Path(spec.source_root)
        entry = Path(spec.script_path).relative_to(root).as_posix()
        if (not spec.source_root or manifest.get("bundle_id") != spec.source_bundle_id
                or manifest.get("source_root") != spec.source_root or manifest.get("script_path") != spec.script_path
                or manifest.get("commit") != spec.source_commit
                or manifest.get("source_files") != sorted(spec.source_files)):
            raise ValueError("SOURCE_ARCHIVE_BINDING_CHANGED")
        records = manifest.get("files", {})
        if {name: row["sha256"] for name, row in records.items()} != {entry: script_hash, **hashes}:
            raise ValueError("SOURCE_ARCHIVE_BINDING_CHANGED")

    def prepare(self, spec: ReproductionSpec) -> dict[str, Any]:
        if not safe_result_url(spec.source):
            raise ValueError("SOURCE_HTTPS_REQUIRED")
        if spec.implementation != "reimplementation" and not spec.source_commit:
            raise ValueError("SOURCE_COMMIT_REQUIRED")
        self._reference_authority(spec.reference)
        relative = Path(spec.script_path)
        if (relative.is_absolute() or relative.suffix != ".py" or ".." in relative.parts
                or not relative.is_relative_to(self._scratch)):
            raise ValueError("SCRIPT_OUTSIDE_TASK_SCRATCH")
        script = self.workspace.resolve(spec.script_path)
        if (self.workspace.relative(script) != relative.as_posix()
                or not script.is_file() or script.stat().st_size > 1_000_000):
            raise ValueError("SCRIPT_UNAVAILABLE_OR_TOO_LARGE")
        sandbox = self.sandbox
        environment = None
        if (sandbox is not None and isinstance(sandbox.backend, PodmanSandboxBackend)
                and spec.environment_profile == "python" and not spec.build_backend):
            environment = next((item for item in sandbox.available_environments()
                                if item.environment_id == spec.environment_id and item.backend == "podman-rootless"), None)
        script_hash = hashlib.sha256(script.read_bytes()).hexdigest()
        hashes = self._source_hashes(spec, relative)
        self._verify_bundle(spec, script_hash, hashes)
        inputs: dict[str, str] = {}
        assets = {asset.input_key: asset for asset in spec.assets}
        for key, name in spec.input_refs.items():
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", key) or key.startswith("file_"):
                raise ValueError("REPRODUCTION_INPUT_KEY_INVALID")
            relative_input = Path(name)
            if (".." in relative_input.parts or not relative_input.is_relative_to(self._scratch)
                    or len(name) > 300):
                raise ValueError("INPUT_OUTSIDE_TASK_SCRATCH")
            path = self.workspace.resolve(name)
            if self.workspace.relative(path) != relative_input.as_posix() or not path.is_file():
                raise ValueError("REPRODUCTION_INPUT_UNAVAILABLE")
            with path.open("rb") as handle:
                data = handle.read(4_000_001)
            if len(data) > 4_000_000:
                raise ValueError("REPRODUCTION_INPUT_TOO_LARGE")
            inputs[key] = hashlib.sha256(data).hexdigest()
            asset = assets.get(key)
            if asset is not None and (asset.sha256 != inputs[key] or asset.size_bytes != len(data)):
                raise ValueError("REPRODUCTION_ASSET_BYTES_CHANGED")
        plan = ReproductionPlan(spec=spec, script_sha256=script_hash,
                                environment_sha256=environment.environment_sha256 if environment else None,
                                source_hashes=hashes, input_hashes=inputs)
        atomic_write(self._plan_path(plan.sha256), plan.model_dump_json().encode())
        atomic_write(self._run_pointer(plan.execution_id), plan.sha256.encode("ascii"))
        return self._status(plan)

    def _load(self, digest: str) -> ReproductionPlan:
        plan = ReproductionPlan.model_validate_json(self._plan_path(digest).read_bytes())
        if plan.sha256 != digest:
            raise ValueError("PLAN_HASH_CHANGED")
        return plan

    def _status(self, plan: ReproductionPlan) -> dict[str, Any]:
        execution_id = plan.execution_id
        record_path = self.workspace.resolve(
            f".photomatagent/executions/{execution_id}/record.json", must_exist=False,
        )
        claim = self.workspace.resolve(f"tmp/capability-executions/{execution_id}", must_exist=False)
        spec = plan.spec
        try:
            authority = self._reference_authority(spec.reference)
        except (ValueError, OSError, ToolExecutionError):
            authority = "MANAGED_REFERENCE_UNAVAILABLE"
        requested = spec.requested_scope
        adapted = bool(spec.adaptation_differences or any(change.kind == "scientific" for change in spec.changes))
        subset = (requested in {"PAPER_SUBSET", "PAPER_FULL"} and spec.target
                  and authority == "MANAGED_VALUE_AND_TOLERANCE" and not adapted)
        scope = ("REIMPLEMENTATION" if spec.implementation == "reimplementation"
                 else "PAPER_SUBSET" if subset else "IMPORT" if requested == "IMPORT" else "EXAMPLE")
        payload: dict[str, Any] = {
            "plan_id": plan.sha256, "plan_sha256": plan.sha256, "run_id": execution_id,
            "scope": scope, "requested_scope": requested, "reference_authority": authority,
            "paper_reproduction": "UNKNOWN", "asset_count": len(spec.assets), "scientific_adaptation": adapted,
            "execution": "PREPARED" if plan.environment_sha256 else "BLOCKED",
            "checker": {"result": "UNKNOWN"}, "applicability": "UNKNOWN",
        }
        if plan.source_hashes:
            payload["source_file_count"] = len(plan.source_hashes) + 1
        if plan.environment_sha256 is None:
            payload["diagnostic"] = ({"cuda": "CUDA_PROFILE_UNAVAILABLE", "native": "NATIVE_PROFILE_UNAVAILABLE"}.get(
                spec.environment_profile) or ("OFFLINE_BUILD_BACKEND_UNAVAILABLE" if spec.build_backend
                                             else "EXECUTION_BACKEND_OR_ENVIRONMENT_UNAVAILABLE"))
        if record_path.is_file():
            record = ExecutionRecord.model_validate_json(record_path.read_bytes())
            if (record.backend != "podman" or record.execution_id != execution_id
                    or record.environment_sha256 != plan.environment_sha256
                    or record.script_sha256 != plan.script_sha256):
                raise ValueError("EXECUTION_BINDING_MISMATCH")
            if plan.source_hashes or plan.input_hashes or spec.seed is not None:
                request_data = json.loads(record_path.with_name("request.json").read_bytes())
                request = ExecutionRequest.model_validate(request_data["request"])
                layout, paths, hashes = plan.source_bindings()
                if (request.execution_id != execution_id or request.script_path != plan.spec.script_path
                        or request.script_digest != plan.script_sha256 or request.environment_id != plan.spec.environment_id
                        or request.argv != plan.spec.argv or request.timeout_seconds != plan.spec.timeout_seconds
                        or request.source_tree != layout
                        or request.input_refs != paths or request.input_sha256 != hashes):
                    raise ValueError("EXECUTION_BINDING_MISMATCH")
            payload["execution"] = ("INTERRUPTED" if record.state == "INTERRUPTED" or record.timed_out
                                    else "SUCCEEDED" if record.state == "COMPLETED" and record.exit_code == 0
                                    else "FAILED")
            payload["checker"] = check_reference(record, plan.spec.reference)
            if scope == "PAPER_SUBSET" and payload["checker"]["result"] in {"PASS", "FAIL"}:
                payload["checker"]["scope"] = "comparison_to_managed_paper_reference"
                payload["paper_reproduction"] = payload["checker"]["result"]
            payload["artifacts"] = [item.model_dump(mode="json") for item in record.artifacts[:20]]
        elif claim.exists():
            payload.update(execution="UNKNOWN", diagnostic="EXECUTION_CLAIMED_WITHOUT_TERMINAL_RECORD")
        return payload

    def qualification_source(self, execution_id: str) -> dict[str, Any] | None:
        """Resolve host-owned method provenance from the scoped plan and its execution."""
        pointer = self._run_pointer(execution_id)
        if not pointer.is_file():
            return None
        plan = self._load(pointer.read_text(encoding="ascii"))
        status = self._status(plan)
        if plan.execution_id != execution_id or status["execution"] != "SUCCEEDED":
            raise ValueError("REPRODUCTION_NOT_ELIGIBLE")
        return {"plan_sha256": plan.sha256, "source": plan.spec.source,
                "paper_anchor": plan.spec.paper_anchor, "implementation": plan.spec.implementation,
                "source_commit": plan.spec.source_commit, "source_bundle_id": plan.spec.source_bundle_id,
                "input_sha256": plan.input_hashes, "reference_authority": status["reference_authority"],
                "paper_reproduction": status["paper_reproduction"]}

    def inspect(self, digest: str | None = None, run_id: str | None = None) -> dict[str, Any]:
        if bool(digest) == bool(run_id):
            raise ValueError("PLAN_OR_RUN_ID_REQUIRED")
        if run_id:
            digest = self._run_pointer(run_id).read_text(encoding="ascii")
        assert digest is not None
        plan = self._load(digest)
        if run_id and plan.execution_id != run_id:
            raise ValueError("RUN_BINDING_MISMATCH")
        return self._status(plan)

    async def execute(self, digest: str, confirmation: str) -> dict[str, Any]:
        if digest != confirmation:
            raise ValueError("PLAN_HASH_MISMATCH")
        plan = await asyncio.to_thread(self._load, digest)
        status = await asyncio.to_thread(self._status, plan)
        sandbox = self.sandbox
        layout, paths, hashes = plan.source_bindings()
        if status["execution"] == "UNKNOWN" and sandbox is not None and plan.environment_sha256 is not None:
            recovered = await sandbox.reconcile_persisted_podman(
                plan.execution_id, expected_script_sha256=plan.script_sha256,
                expected_environment_sha256=plan.environment_sha256, expected_source_tree=layout,
                expected_input_sha256=hashes,
            )
            if recovered.record is not None:
                return await asyncio.to_thread(self._status, plan)
            status["diagnostic"] = recovered.code
            if recovered.code == "EXECUTION_RUNNING":
                status["execution"] = "RUNNING"
            return status
        if status["execution"] != "PREPARED":
            return status
        if sandbox is None or not isinstance(sandbox.backend, PodmanSandboxBackend):
            return {**status, "execution": "BLOCKED", "diagnostic": "EXECUTION_BACKEND_UNAVAILABLE"}
        result = await sandbox.run(
            script_path=plan.spec.script_path, environment_id=plan.spec.environment_id,
            argv=plan.spec.argv, timeout_seconds=plan.spec.timeout_seconds,
            execution_id=plan.execution_id, expected_script_sha256=plan.script_sha256,
            expected_environment_sha256=plan.environment_sha256,
            input_refs=paths or None, expected_input_sha256=hashes or None, source_tree=layout,
        )
        status = await asyncio.to_thread(self._status, plan)
        if result.code != "OK" and status["execution"] == "PREPARED":
            status.update(execution="BLOCKED", diagnostic=result.code)
        return status


class ReproductionTool(Tool):
    namespace = "reproduction"
    source = "builtin"
    exposure = ToolExposure.DEFERRED
    tags = ("paper", "code", "reproduction", "isolated")

    def __init__(self, service: ReproductionService, operation: Literal["prepare", "execute", "inspect"]) -> None:
        self.name = f"reproduction.{operation}"
        self._operation = operation
        self._service = service
        self.description = {
            "prepare": "Freeze a task-local Python example, optional source tree, and isolated environment.",
            "execute": "Execute one frozen reproduction plan in a configured Podman environment, under runtime approval.",
            "inspect": "Read the immutable reproduction plan's execution and independent reference-comparison status.",
        }[operation]
        self.short_description = self.description
        digest = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
        fields: dict[str, Any] = {"plan_id": digest}
        if operation == "execute":
            fields["plan_sha256"] = digest
        if operation == "prepare":
            fields = {"spec": ReproductionSpec.model_json_schema()}
        if operation == "inspect":
            fields["run_id"] = {"type": "string", "pattern": "^exec_[0-9a-f]{32}$"}
        self.input_schema = {"type": "object", "properties": fields,
                             "required": [] if operation == "inspect" else list(fields), "additionalProperties": False}
        if operation == "prepare":
            nested = fields["spec"]
            definitions = nested.pop("$defs", None)
            if definitions:
                self.input_schema["$defs"] = definitions

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            if self._operation == "prepare":
                spec = ReproductionSpec.model_validate(arguments["spec"])
                result = await asyncio.to_thread(self._service.prepare, spec)
            elif self._operation == "inspect":
                result = await asyncio.to_thread(self._service.inspect, arguments.get("plan_id"), arguments.get("run_id"))
            else:
                result = await self._service.execute(arguments["plan_id"], arguments["plan_sha256"])
        except (ValueError, OSError, ToolExecutionError) as exc:
            reason = str(exc)
            code = reason if reason and len(reason) <= 80 and reason.replace("_", "").isalnum() and reason.isupper() else "REPRODUCTION_INPUT_OR_STORAGE_INVALID"
            return tool_result({"diagnostic": code}, error=True)
        return tool_result(result)
