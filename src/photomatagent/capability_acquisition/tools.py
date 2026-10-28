"""Deferred runtime tools for real package search and exact metadata inspect."""
from __future__ import annotations

import json
from typing import Any

from photomatagent.capability_acquisition.discovery import (
    CapabilityDiscoveryService,
    _network_failure_code,
)
from photomatagent.capability_acquisition.models import MAX_SCRIPT_ARGV_TOKENS
from photomatagent.capability_acquisition.sandbox import SandboxService
from photomatagent.tools.base import Tool, ToolResult
from photomatagent.tools.exposure import ToolExposure


class CapabilitySearchTool(Tool):
    name = "capability_search"
    description = "Find source-grounded Python packages. The built-in PyPI lookup accepts an exact package name or name==version; broad keyword search requires a configured endpoint."
    short_description = "Look up a Python package by exact name."
    exposure = ToolExposure.DEFERRED
    namespace = "capability"
    tags = ("capability", "search", "package")
    input_schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 500},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10},
            "ecosystem": {"type": "string", "enum": ["pypi", "native_cli"]},
        },
        "required": ["query"],
    }

    def __init__(self, service: CapabilityDiscoveryService) -> None:
        self.service = service

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        if set(arguments) - {"query", "limit", "ecosystem"}:
            return ToolResult(output="INVALID_SCHEMA", is_error=True)
        if arguments.get("ecosystem", "pypi") != "pypi":
            return ToolResult(output="NATIVE_CLI_UNSUPPORTED: only Python wheels and their console_scripts are supported", is_error=True,
                              data={"code": "NATIVE_CLI_UNSUPPORTED"})
        result = await self.service.search(str(arguments.get("query", "")), limit=int(arguments.get("limit", 5)))
        payload = {
            "code": result.code,
            "candidates": [
                {
                    "candidate_id": item.candidate_id,
                    "normalized_name": item.normalized_name,
                    "version": item.version,
                    "summary": item.summary[:200],
                    "source_urls": [url[:250] for url in item.source_urls[:1]],
                    "retrieved_at": item.retrieved_at.isoformat(),
                }
                for item in result.candidates
            ],
            "diagnostics": list(result.diagnostics),
        }
        output = _bounded_json(payload, max_chars=8000)
        return ToolResult(
            output=output,
            is_error=result.code != "OK", data={"code": result.code, "candidate_ids": [item.candidate_id for item in result.candidates]},
        )


class CapabilityInspectTool(Tool):
    name = "capability_inspect"
    description = "Inspect exact PyPI metadata for a search result and report installation-resolution status."
    short_description = "Inspect one source-bound PyPI candidate."
    exposure = ToolExposure.DEFERRED
    namespace = "capability"
    tags = ("capability", "metadata", "inspect", "package")
    input_schema = {
        "type": "object", "additionalProperties": False,
        "properties": {"candidate_id": {"type": "string", "minLength": 1, "maxLength": 128}},
        "required": ["candidate_id"],
    }

    def __init__(self, service: CapabilityDiscoveryService, *, local_dev: bool = False) -> None:
        self.service = service
        self.local_dev = local_dev

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        if set(arguments) != {"candidate_id"}:
            return ToolResult(output="INVALID_SCHEMA", is_error=True)
        candidate = self.service.candidate(str(arguments["candidate_id"]))
        if candidate is None:
            return ToolResult(output="CANDIDATE_NOT_FOUND", is_error=True, data={"code": "CANDIDATE_NOT_FOUND"})
        try:
            metadata = await self.service.inspect(candidate.candidate_id)
        except Exception as exc:
            if isinstance(exc, LookupError):
                code = "CANDIDATE_NOT_FOUND"
            elif (network_code := _network_failure_code(exc)) is not None:
                code = network_code
            elif isinstance(exc, ValueError):
                code = "INVALID_METADATA"
            else:
                code = "NETWORK_ERROR"
            return ToolResult(output=code[:500], is_error=True, data={"code": code})
        plan = None
        plan_code = "OK"
        try:
            plan = await self.service.resolve_install_plan(candidate.candidate_id)
        except Exception as exc:
            if isinstance(exc, LookupError):
                plan_code = "CANDIDATE_NOT_FOUND"
            elif isinstance(exc, ValueError):
                plan_code = "INVALID_INSTALL_PLAN"
            else:
                plan_code = "INSTALL_RESOLUTION_UNAVAILABLE"
        raw_info = metadata.get("info")
        info: dict[str, Any] = raw_info if isinstance(raw_info, dict) else metadata
        # Metadata can contain untrusted descriptions and arbitrary release data.
        # Keep the model-visible observation narrow and explicitly non-executable.
        payload = {
            "candidate_id": candidate.candidate_id,
            "normalized_name": candidate.normalized_name,
            "version": candidate.version,
            "summary": str(info.get("summary") or "")[:1000],
            "requires_python": str(info.get("requires_python") or "")[:200],
            "project_url": candidate.source_urls[0],
            "installation_plan": (
                {
                    "plan_id": plan.plan_id,
                    "plan_sha256": plan.plan_sha256,
                    "direct_requirements": list(plan.direct_requirements),
                    "dependency_count": len(plan.wheels),
                    "estimated_bytes": plan.estimated_bytes,
                    "expires_at": plan.expires_at.isoformat(),
                    "python_version": plan.python_version,
                    "platform_tag": plan.platform_tag,
                    "image_digest": plan.image_digest,
                    "installer_version": plan.installer_version,
                    "index": plan.approved_index,
                    "execution_backend": plan.execution_backend,
                    "execution_profile": (
                        "local_dev; host process; task venv; no OS isolation; offline wheel install"
                        if self.local_dev else "podman-rootless; network=none; offline wheel install"
                    ),
                    "wheels": [
                        {
                            "distribution": wheel.distribution,
                            "version": wheel.version,
                            "direct": f"{wheel.distribution}=={wheel.version}" in plan.direct_requirements,
                            "filename": wheel.filename,
                            "sha256": wheel.sha256,
                            "size": wheel.size,
                            "source": wheel.url,
                        }
                        for wheel in plan.wheels[:32]
                    ],
                    "wheels_omitted": max(0, len(plan.wheels) - 32),
                }
                if plan else None
            ),
            "diagnostics": [] if plan else [plan_code],
        }
        return ToolResult(
            output=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            data={
                "candidate_id": candidate.candidate_id,
                "plan_id": plan.plan_id if plan else None,
                "code": plan_code,
            },
        )


class CapabilityInstallTool(Tool):
    name = "capability_install"
    description = "Install an immutable hashed wheel plan in the configured backend (local_dev host venv or rootless Podman)."
    short_description = "Install one exact Python capability plan."
    exposure = ToolExposure.DEFERRED
    namespace = "capability"
    tags = ("capability", "install", "sandbox")
    input_schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "plan_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "plan_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
        "required": ["plan_id", "plan_sha256"],
    }

    def __init__(self, discovery: CapabilityDiscoveryService, sandbox: SandboxService) -> None:
        self.discovery = discovery
        self.sandbox = sandbox

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        if set(arguments) != {"plan_id", "plan_sha256"}:
            return ToolResult(output="INVALID_SCHEMA", is_error=True)
        plan = self.discovery.install_plan(str(arguments["plan_id"]))
        if plan is None:
            return ToolResult(output="INSTALL_PLAN_NOT_FOUND", is_error=True, data={"code": "INSTALL_PLAN_NOT_FOUND"})
        if arguments["plan_sha256"] != plan.plan_sha256:
            return ToolResult(output="INSTALL_PLAN_CHANGED", is_error=True, data={"code": "INSTALL_PLAN_CHANGED"})
        result = await self.sandbox.install(plan)
        payload = {
            "code": result.code,
            "environment_id": result.environment.environment_id if result.environment else None,
            "status": result.environment.status if result.environment else None,
            "backend": result.environment.backend if result.environment else plan.execution_backend,
            "entrypoints": list(result.environment.entrypoints)[:32] if result.environment else [],
            "plan_sha256": result.environment.plan_sha256 if result.environment else plan.plan_sha256,
            "diagnostics": list(result.diagnostics),
        }
        return ToolResult(
            output=_bounded_json(payload, max_chars=8000),
            is_error=result.code != "OK",
            data={"code": result.code, "environment_id": payload["environment_id"]},
        )


class ScriptRunTool(Tool):
    name = "script_run"
    description = "Run a bound relative Python script, or entrypoint:<name> from an installed local_dev wheel, using argv without a shell. Bound inputs are at PHOTOMAT_INPUT_DIR/<key>. local_dev runs on the host and is not OS-isolated."
    short_description = "Run a task Python script or installed CLI entrypoint."
    exposure = ToolExposure.DEFERRED
    namespace = "capability"
    tags = ("capability", "script", "run", "sandbox")
    input_schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "script_path": {"type": "string", "minLength": 1, "maxLength": 500},
            "environment_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "input_refs": {"type": "object", "additionalProperties": {"type": "string"}},
            "argv": {
                "type": "array", "maxItems": MAX_SCRIPT_ARGV_TOKENS,
                "items": {"type": "string", "maxLength": 1000},
            },
            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 120},
        },
        "required": ["script_path", "environment_id"],
    }

    def __init__(self, sandbox: SandboxService, published_skill_source: Any = None) -> None:
        self.sandbox = sandbox
        self.published_skill_source = published_skill_source

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        allowed = {"script_path", "environment_id", "input_refs", "argv", "timeout_seconds"}
        if set(arguments) - allowed:
            return ToolResult(output="INVALID_SCHEMA", is_error=True)
        script_path = str(arguments.get("script_path", ""))
        authorized_skill = None
        if self.published_skill_source is not None:
            try:
                authorized_skill = self.published_skill_source.authorize_execution(script_path)
            except ValueError:
                return ToolResult(
                    output="PUBLISHED_SKILL_UNAVAILABLE", is_error=True,
                    data={"code": "PUBLISHED_SKILL_UNAVAILABLE"},
                )
        result = await self.sandbox.run(
            script_path=script_path,
            environment_id=str(arguments.get("environment_id", "")),
            input_refs=arguments.get("input_refs", {}),
            argv=arguments.get("argv", []),
            timeout_seconds=int(arguments.get("timeout_seconds", 60)),
        )
        record = result.record
        attributed_skill = None
        if record is not None and self.published_skill_source is not None:
            attributed_skill = self.published_skill_source.record_execution(
                str(arguments.get("script_path", "")),
                code=result.code,
                execution_id=record.execution_id,
                expected_skill=authorized_skill,
            )
        payload = {
            "code": result.code,
            "execution_id": record.execution_id if record else None,
            "state": record.state if record else None,
            "backend": record.backend if record else None,
            "exit_code": record.exit_code if record else None,
            "timed_out": record.timed_out if record else False,
            "stdout": record.stdout[:1200] if record else "",
            "stderr": record.stderr[:1200] if record else "",
            "artifacts": [
                {"path": artifact.path[:160], "size": artifact.size, "sha256": artifact.sha256}
                for artifact in record.artifacts[:10]
            ] if record else [],
            "artifacts_omitted": max(0, len(record.artifacts) - 10) if record else 0,
            "diagnostics": list(result.diagnostics),
        }
        return ToolResult(
            output=_bounded_json(payload, max_chars=8000),
            is_error=result.code != "OK",
            data={
                "code": result.code, "execution_id": payload["execution_id"],
                "published_skill": attributed_skill,
                "skill_usage_stage": "script_executed" if attributed_skill else None,
                "skill_execution_succeeded": result.code == "OK" if attributed_skill else None,
            },
        )


def _bounded_json(payload: dict[str, Any], *, max_chars: int) -> str:
    """Return valid bounded JSON even when external summaries are unusually long."""
    bounded = dict(payload)
    while True:
        encoded = json.dumps(bounded, ensure_ascii=False, separators=(",", ":"))
        if len(encoded) <= max_chars:
            return encoded
        for key in ("candidates", "artifacts", "wheels"):
            values = bounded.get(key)
            if isinstance(values, list) and values:
                bounded[key] = values[:-1]
                bounded[f"{key}_omitted"] = int(bounded.get(f"{key}_omitted", 0)) + 1
                break
        else:
            for key in ("stdout", "stderr", "summary"):
                value = bounded.get(key)
                if isinstance(value, str) and value:
                    bounded[key] = value[: max(0, len(value) // 2)]
                    break
            else:
                return json.dumps({"code": bounded.get("code", "OUTPUT_TRUNCATED")}, separators=(",", ":"))
