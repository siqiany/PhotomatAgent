"""Immutable, provenance-rich contracts for quality-diversity evolution."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Any, Literal
from pydantic import StrictBool, StrictFloat, StrictInt, StrictStr

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer, model_validator

Sha256 = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[StrictStr, Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9_.:-]+$")]
PathRef = Annotated[StrictStr, Field(min_length=1, max_length=500)]
CheckKind = Literal["schema", "finite", "numeric_range", "reference_error", "required_evidence", "human_review", "registered_checker"]


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON_KEY_INVALID")
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("JSON_VALUE_INVALID")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


class QDModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False, str_strip_whitespace=True)

    @model_validator(mode="after")
    def freeze_nested_json(self) -> QDModel:
        for name, value in self.__dict__.items():
            if isinstance(value, (Mapping, list)):
                object.__setattr__(self, name, _freeze_json(value))
        return self

    @model_serializer(mode="wrap")
    def serialize_frozen_json(self, handler: Any, info: Any) -> Any:
        del info
        snapshot = self.model_copy(deep=False)
        for name, value in self.__dict__.items():
            if isinstance(value, Mapping):
                object.__setattr__(snapshot, name, _thaw_json(value))
        return handler(snapshot)


class ArtifactDigest(QDModel):
    path: PathRef
    sha256: Sha256
    size_bytes: StrictInt = Field(ge=0)
    media_type: StrictStr = Field(min_length=1)

    @field_validator("path")
    @classmethod
    def relative_safe_path(cls, value: str) -> str:
        if value.startswith("/") or re.match(r"^[A-Za-z]:", value) or "\\" in value or any(p in ("", ".", "..") for p in value.split("/")):
            raise ValueError("PATH_INVALID")
        return value


class AcceptanceCheck(QDModel):
    check_id: Identifier
    kind: CheckKind
    artifact_role: Identifier
    parameters: Mapping[StrictStr, Any] = Field(default_factory=dict)
    required: StrictBool = True

    @field_validator("parameters")
    @classmethod
    def json_finite(cls, value: Mapping[StrictStr, Any]) -> Mapping[StrictStr, Any]:
        try:
            json.dumps(value, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("CHECK_PARAMETERS_INVALID") from exc
        return value


class EvaluationContract(QDModel):
    contract_id: Identifier
    task_family: Identifier
    version: StrictStr = Field(min_length=1)
    input_schema: Mapping[StrictStr, Any]
    output_schema: Mapping[StrictStr, Any]
    checks: tuple[AcceptanceCheck, ...]
    quality_metric: StrictStr = Field(min_length=1)
    quality_direction: Literal["minimize", "maximize"]
    quality_tolerance: StrictFloat = Field(ge=0)
    evaluation_suite_sha256: Sha256
    seed_set: tuple[StrictInt, ...]
    descriptor_config: Mapping[StrictStr, Any]


class StepSpec(QDModel):
    step_id: Identifier
    kind: Literal["script", "registered_tool"]
    depends_on: tuple[Identifier, ...] = ()
    input_roles: tuple[Identifier, ...] = ()
    output_roles: tuple[Identifier, ...] = ()
    binding_id: StrictStr = Field(min_length=1, max_length=200)
    parameters: Mapping[StrictStr, Any] = Field(default_factory=dict)
    input_units: Mapping[StrictStr, StrictStr] = Field(default_factory=dict)
    output_units: Mapping[StrictStr, StrictStr] = Field(default_factory=dict)
    input_schema_sha256: Sha256 | None = None
    output_schema_sha256: Sha256 | None = None
    assumptions: tuple[StrictStr, ...] = ()

    @field_validator("binding_id")
    @classmethod
    def safe_binding(cls, value: str) -> str:
        if value.startswith("/") or re.match(r"^[A-Za-z]:", value) or ".." in value.split("/") or "\\" in value:
            raise ValueError("PATH_INVALID")
        return value

    @model_validator(mode="after")
    def valid_module_contract(self) -> StepSpec:
        if (not set(self.input_units).issubset(self.input_roles)
                or not set(self.output_units).issubset(self.output_roles)
                or any(not value.strip() or len(value) > 100
                       for value in (*self.input_units.values(), *self.output_units.values()))):
            raise ValueError("STEP_MODULE_CONTRACT_INVALID")
        return self


class WorkflowGenome(QDModel):
    @classmethod
    def from_behavior(cls, **data: Any) -> WorkflowGenome:
        normalized = dict(data)
        requested_digest = normalized.pop("content_sha256", None)
        normalized["steps"] = tuple(StepSpec.model_validate(item) for item in normalized["steps"])
        normalized["script_refs"] = tuple(ArtifactDigest.model_validate(item) for item in normalized["script_refs"])
        provisional = cls.model_construct(content_sha256="0" * 64, **normalized)
        actual_digest = canonical_genome_sha256(provisional)
        if requested_digest is not None and requested_digest != actual_digest:
            raise ValueError("GENOME_CONTENT_HASH_MISMATCH")
        normalized["content_sha256"] = actual_digest
        return cls.model_validate(normalized)

    genome_id: Identifier
    content_sha256: Sha256
    task_family: Identifier
    steps: tuple[StepSpec, ...]
    entrypoint: Identifier
    script_refs: tuple[ArtifactDigest, ...]
    environment_ref: StrictStr = Field(min_length=1)
    applicability: tuple[StrictStr, ...]
    recovery: tuple[StrictStr, ...]

    @model_validator(mode="after")
    def valid_workflow(self) -> WorkflowGenome:
        validate_dag(self.steps)
        if self.content_sha256 != canonical_genome_sha256(self):
            raise ValueError("GENOME_CONTENT_HASH_MISMATCH")
        paths = [ref.path for ref in self.script_refs]
        if len(paths) != len(set(paths)):
            raise ValueError("GENOME_DUPLICATE_SCRIPT_PATH")
        if self.entrypoint not in {step.step_id for step in self.steps}:
            raise ValueError("DAG_ENTRYPOINT_UNKNOWN")
        return self


class ProposalRecord(QDModel):
    proposal_id: Identifier
    parent_ids: tuple[Identifier, ...]
    operator: Identifier
    feedback_refs: tuple[Identifier, ...]
    change_summary: StrictStr = Field(min_length=1)
    genome_id: Identifier
    model_provenance: Mapping[StrictStr, Any]


class RevisionIssue(QDModel):
    issue_id: Identifier
    source_kind: Literal["human", "judge", "checker", "tool"]
    source_ref: Identifier
    artifact_sha256: Sha256
    status: Literal["open", "resolved", "deferred"]
    severity: Literal["low", "medium", "high", "critical"]
    step_ids: tuple[Identifier, ...]
    requested_actions: tuple[StrictStr, ...]
    preserve: tuple[StrictStr, ...]
    acceptance_check_ids: tuple[Identifier, ...]


class TrialRecord(QDModel):
    trial_id: Identifier
    event_id: Identifier
    genome_id: Identifier
    contract_sha256: Sha256
    input_digests: tuple[ArtifactDigest, ...]
    environment_sha256: Sha256
    seed_set: tuple[StrictInt, ...]
    status: Literal["RESERVED", "RUNNING", "COLLECTED", "VALIDATED", "FAILED", "CANCELLED", "INTERRUPTED"]
    session_id: Identifier
    attempt_dir: PathRef
    runtime_trace_ref: StrictStr
    cost: Mapping[StrictStr, StrictFloat]
    output_refs: tuple[ArtifactDigest, ...]
    resource_profile: Identifier = "default"

    @field_validator("attempt_dir")
    @classmethod
    def safe_attempt_path(cls, value: str) -> str:
        if value.startswith("/") or re.match(r"^[A-Za-z]:", value) or "\\" in value or any(part in ("", ".", "..") for part in value.split("/")):
            raise ValueError("PATH_INVALID")
        return value


class CheckOutcome(QDModel):
    check_id: Identifier
    verdict: Literal["PASS", "FAIL", "UNKNOWN"]
    measured_value: StrictFloat | None = None
    evidence_refs: tuple[Identifier, ...]
    reason_code: Identifier
    required: StrictBool = True

    @model_validator(mode="after")
    def pass_has_evidence(self) -> CheckOutcome:
        if self.verdict == "PASS" and not self.evidence_refs:
            raise ValueError("PASS_EVIDENCE_REQUIRED")
        return self


class ValidationRecord(QDModel):
    validation_id: Identifier
    trial_id: Identifier
    genome_id: Identifier = "unknown"
    genome_sha256: Sha256
    bundle_sha256: Sha256
    contract_sha256: Sha256
    suite_sha256: Sha256
    checker_version: StrictStr
    qualification: Literal["PASS", "FAIL", "UNKNOWN", "INCOMPARABLE"]
    outcomes: tuple[CheckOutcome, ...]
    quality: StrictFloat | None
    cost: Mapping[StrictStr, StrictFloat]
    provenance: Mapping[StrictStr, Any]
    resource_profile: Identifier = "default"
    seed_set: tuple[StrictInt, ...] = ()
    runtime_samples_seconds: tuple[StrictFloat, ...] = ()
    direct_dependency_count: StrictInt = Field(default=0, ge=0)

    @field_validator("runtime_samples_seconds")
    @classmethod
    def valid_runtime_samples(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if any(not math.isfinite(item) or item < 0 for item in value):
            raise ValueError("RUNTIME_SAMPLES_INVALID")
        return value

    @model_validator(mode="after")
    def qualification_matches_evidence(self) -> ValidationRecord:
        if self.qualification == "INCOMPARABLE":
            return self
        required = [outcome.verdict for outcome in self.outcomes if outcome.required]
        expected = "FAIL" if "FAIL" in required else "UNKNOWN" if not required or "UNKNOWN" in required else "PASS"
        if self.qualification != expected:
            raise ValueError("VALIDATION_QUALIFICATION_MISMATCH")
        if self.qualification == "PASS" and not self.provenance:
            raise ValueError("PASS_PROVENANCE_REQUIRED")
        return self


class ArchiveEntry(QDModel):
    archive_key: Identifier
    cell_key: Identifier
    genome_id: Identifier
    trial_ids: tuple[Identifier, ...]
    validation_ids: tuple[Identifier, ...]
    quality: StrictFloat
    cost: Mapping[StrictStr, StrictFloat]
    revision: StrictInt = Field(ge=0)
    previous_entry_ref: StrictStr | None = None
    use_count: StrictInt = Field(default=0, ge=0)
    last_used_at_ns: StrictInt = Field(default=0, ge=0)

    @model_validator(mode="after")
    def valid_lineage(self) -> ArchiveEntry:
        if (not self.trial_ids or len(self.trial_ids) != len(self.validation_ids)
                or len(set(self.trial_ids)) != len(self.trial_ids)
                or len(set(self.validation_ids)) != len(self.validation_ids)):
            raise ValueError("ARCHIVE_LINEAGE_INVALID")
        return self


class ProposalUsage(QDModel):
    model_calls: StrictInt = Field(ge=0)
    model_tokens: StrictInt = Field(ge=0)


class ProposalBatch(QDModel):
    proposals: tuple[ProposalRecord, ...]
    diagnostics: tuple[StrictStr, ...]
    usage: ProposalUsage


class TrialOutcome(QDModel):
    trial: TrialRecord
    validation: ValidationRecord | None
    reason_code: Identifier | None = None

    @model_validator(mode="after")
    def requires_failure_reason_without_validation(self) -> TrialOutcome:
        if self.validation is None and self.reason_code is None:
            raise ValueError("TRIAL_OUTCOME_REASON_REQUIRED")
        if self.validation is not None and self.validation.trial_id != self.trial.trial_id:
            raise ValueError("TRIAL_VALIDATION_ID_MISMATCH")
        return self


class ArchiveDecision(QDModel):
    action: Literal["INSERT", "REPLACE", "KEEP", "REJECT", "INCOMPARABLE"]
    reason: StrictStr
    old_entry: ArchiveEntry | None = None
    new_entry: ArchiveEntry | None = None


class EventOutcome(QDModel):
    status: StrictStr
    best_trial_id: StrictStr | None
    unresolved_issues: tuple[Identifier, ...]
    cost: Mapping[StrictStr, StrictFloat]


class SkillUseRecord(QDModel):
    task_id: Identifier
    skill_id: Identifier
    bundle_sha256: Sha256
    stage: StrictStr
    trial_id: Identifier


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def canonical_genome_sha256(genome: WorkflowGenome) -> str:
    """Hash behavior only; identity, stored digest, lineage and results are excluded."""
    payload = {
        "task_family": genome.task_family,
        "steps": [step.model_dump(mode="json") for step in genome.steps],
        "entrypoint": genome.entrypoint,
        "script_refs": [ref.model_dump(mode="json") for ref in sorted(genome.script_refs, key=lambda ref: (ref.path, ref.sha256, ref.size_bytes, ref.media_type))],
        "environment_ref": genome.environment_ref,
        "applicability": genome.applicability,
        "recovery": genome.recovery,
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()


def validate_dag(steps: tuple[StepSpec, ...]) -> None:
    ids = [step.step_id for step in steps]
    if len(ids) != len(set(ids)):
        raise ValueError("DAG_DUPLICATE_STEP_ID")
    known = set(ids)
    for step in steps:
        if any(parent not in known for parent in step.depends_on):
            raise ValueError("DAG_UNKNOWN_DEPENDENCY")
    outputs = [role for step in steps for role in step.output_roles]
    if len(outputs) != len(set(outputs)):
        raise ValueError("DAG_DUPLICATE_OUTPUT_ROLE")
    parents = {step.step_id: set(step.depends_on) for step in steps}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError("DAG_CYCLE")
        if node in visited:
            return
        visiting.add(node)
        for parent in parents[node]:
            visit(parent)
        visiting.remove(node)
        visited.add(node)

    for node in ids:
        visit(node)


class SkillPackage(QDModel):
    skill_id: Identifier
    version: StrictStr = Field(min_length=1)
    genome_id: Identifier
    bundle_sha256: Sha256
    description: StrictStr = Field(min_length=1)
    applicability: tuple[StrictStr, ...]
    file_digests: tuple[ArtifactDigest, ...]
    validation_refs: tuple[Identifier, ...]
    experience_refs: tuple[Identifier, ...]
    status: Literal["DRAFT", "REPLAY_PASSED", "ELIGIBLE", "PUBLISHED", "REVOKED"]
    replay_validation_ref: Identifier | None = None

    @model_validator(mode="after")
    def replay_status_has_credential(self) -> SkillPackage:
        if self.status in ("REPLAY_PASSED", "ELIGIBLE", "PUBLISHED") and self.replay_validation_ref is None:
            raise ValueError("SKILL_REPLAY_CREDENTIAL_REQUIRED")
        return self


class PublicationApproval(QDModel):
    approval_id: Identifier
    actor: StrictStr = Field(min_length=1)
    approved_at: StrictStr = Field(min_length=1)
    bundle_sha256: Sha256
    scope: StrictStr = Field(min_length=1)


class ReviewedTrialLink(QDModel):
    """Verified sidecar from an executed QD trial to reviewed legacy evidence."""

    qd_task_id: Identifier
    trial_id: Identifier
    genome_id: Identifier
    genome_sha256: Sha256
    bundle_sha256: Sha256
    evolution_id: Identifier
    episode_version: Identifier
    runtime_session_id: Identifier
    artifact_sha256: Sha256
    comparison_ids: tuple[Identifier, ...] = Field(min_length=1)
    experience_id: Identifier
    experience_evolution_id: Identifier
    task_group_id: Identifier
    environment_sha256: Sha256
    script_bundle_sha256: Sha256
