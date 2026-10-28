"""Strict data contracts for capability discovery and reproducible plans."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_SCRIPT_ARGV_TOKENS = 32


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PackageCandidate(StrictModel):
    candidate_id: str
    normalized_name: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=100)
    summary: str = Field(max_length=1000)
    source_urls: tuple[str, ...] = Field(min_length=1, max_length=5)
    retrieved_at: datetime
    ecosystem: Literal["pypi"] = "pypi"

    @field_validator("normalized_name")
    @classmethod
    def package_name_is_normalized(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value):
            raise ValueError("INVALID_PACKAGE_NAME")
        return value

    @field_validator("version")
    @classmethod
    def package_version_is_path_safe(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+!-]{0,99}", value):
            raise ValueError("INVALID_PACKAGE_VERSION")
        return value


class DiscoveryResult(StrictModel):
    code: Literal[
        "OK", "DISCOVERY_UNAVAILABLE", "NETWORK_BLOCKED", "NETWORK_ERROR",
        "NETWORK_FAILED", "INVALID_RESPONSE",
    ]
    candidates: tuple[PackageCandidate, ...] = ()
    diagnostics: tuple[str, ...] = ()


class WheelArtifact(StrictModel):
    filename: str = Field(min_length=1, max_length=255)
    url: str = Field(min_length=1, max_length=2048)
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    size: int = Field(gt=0, le=2_000_000_000)
    distribution: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=100)
    requires_python: str = Field(default="", max_length=200)
    tags: tuple[str, ...] = Field(min_length=1, max_length=64)

    @field_validator("filename")
    @classmethod
    def wheel_only(cls, value: str) -> str:
        if not value.endswith(".whl") or "/" in value or "\\" in value:
            raise ValueError("SDIST_OR_INVALID_WHEEL")
        return value

    @field_validator("url")
    @classmethod
    def approved_source(cls, value: str) -> str:
        from photomatagent.capability_acquisition.discovery import validate_public_https_url

        validate_public_https_url(value)
        from urllib.parse import urlsplit

        if (urlsplit(value).hostname or "").lower() != "files.pythonhosted.org":
            raise ValueError("UNAPPROVED_WHEEL_SOURCE")
        return value


class InstallPlan(StrictModel):
    plan_id: str
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    python_version: str
    platform_tag: str
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    execution_backend: Literal["podman", "local_dev"] = "podman"
    installer_version: str
    wheels: tuple[WheelArtifact, ...] = Field(min_length=1, max_length=256)
    direct_requirements: tuple[str, ...] = Field(min_length=1, max_length=32)
    estimated_bytes: int = Field(ge=0, le=10_000_000_000)
    expires_at: datetime
    approved_index: Literal["https://pypi.org/simple"] = "https://pypi.org/simple"

    @model_validator(mode="after")
    def verify_digest(self) -> InstallPlan:
        if computed_plan_sha256(self) != self.plan_sha256:
            raise ValueError("INSTALL_PLAN_HASH_MISMATCH")
        if self.expires_at.tzinfo is None or self.expires_at <= datetime.now(timezone.utc):
            raise ValueError("INVALID_PLAN_EXPIRY")
        return self


EnvironmentStatus = Literal["PREPARING", "READY", "FAILED"]
ExecutionState = Literal["RUNNING", "COMPLETED", "FAILED", "INTERRUPTED", "UNKNOWN"]


class EnvironmentRecord(StrictModel):
    environment_id: str
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: EnvironmentStatus
    lock_ref: str
    installed_manifest_ref: str
    entrypoints: dict[str, str]
    backend: str
    created_at: datetime
    environment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason_code: str | None = Field(default=None, max_length=100)


class ExecutionRequest(StrictModel):
    """Server-bound execution contract; created by SandboxService, not model input."""

    execution_id: str
    trial_id: str
    genome_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_id: str
    script_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    script_path: str
    input_refs: dict[str, str]
    input_sha256: dict[str, str] = Field(default_factory=dict)
    argv: tuple[str, ...] = Field(max_length=MAX_SCRIPT_ARGV_TOKENS)
    timeout_seconds: int = Field(ge=1, le=120)
    output_limit_bytes: int = Field(ge=1024, le=1_048_576)
    artifact_limit_bytes: int = Field(default=67_108_864, ge=1024, le=1_073_741_824)
    workspace_root: str

    @field_validator("argv")
    @classmethod
    def arguments_are_bounded(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(len(value) > 1000 or "\x00" in value for value in values):
            raise ValueError("INVALID_EXECUTION_ARGUMENT")
        return values


class ArtifactRecord(StrictModel):
    path: str = Field(min_length=1, max_length=1024)
    size: int = Field(ge=0, le=1_000_000_000)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExecutionRecord(StrictModel):
    execution_id: str
    backend: Literal["podman", "local_dev"] = "podman"
    state: ExecutionState
    exit_code: int | None
    timed_out: bool = False
    stdout: str = Field(default="", max_length=1_048_576)
    stderr: str = Field(default="", max_length=1_048_576)
    artifacts: tuple[ArtifactRecord, ...] = Field(default=(), max_length=256)
    environment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    script_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    started_at: datetime
    ended_at: datetime
    reason_code: str | None = Field(default=None, max_length=100)


class SandboxResult(StrictModel):
    code: str
    environment: EnvironmentRecord | None = None
    record: ExecutionRecord | None = None
    diagnostics: tuple[str, ...] = ()


def canonical_plan_payload(**fields: object) -> bytes:
    return json.dumps(
        fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False, default=lambda value: value.isoformat() if isinstance(value, datetime) else value,
    ).encode("utf-8")


def make_candidate(
    *, name: str, version: str, summary: str, source_urls: tuple[str, ...],
    retrieved_at: datetime | None = None,
) -> PackageCandidate:
    normalized = re.sub(r"[-_.]+", "-", name).lower()
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", normalized):
        raise ValueError("INVALID_PACKAGE_NAME")
    timestamp = retrieved_at or datetime.now(timezone.utc)
    identity = hashlib.sha256(f"pypi\0{normalized}\0{version}\0{source_urls[0]}".encode()).hexdigest()
    return PackageCandidate(
        candidate_id=f"pkg_{identity[:24]}", normalized_name=normalized, version=version,
        summary=summary[:1000], source_urls=source_urls, retrieved_at=timestamp,
    )


def new_plan_id() -> str:
    return f"plan_{uuid4().hex}"


def computed_plan_sha256(plan: InstallPlan) -> str:
    payload = {
        "python_version": plan.python_version,
        "platform_tag": plan.platform_tag,
        "image_digest": plan.image_digest,
        "installer_version": plan.installer_version,
        "wheels": [wheel.model_dump(mode="json") for wheel in plan.wheels],
        "direct_requirements": list(plan.direct_requirements),
        "estimated_bytes": plan.estimated_bytes,
        "approved_index": plan.approved_index,
    }
    if plan.execution_backend != "podman":
        payload["execution_backend"] = plan.execution_backend
    return hashlib.sha256(canonical_plan_payload(**payload)).hexdigest()
