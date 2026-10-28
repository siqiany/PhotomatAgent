"""Deterministic wheel-only plan validation and content addressing."""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import sys
import sysconfig
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Awaitable, Callable, Protocol
from typing import Literal

from photomatagent.capability_acquisition.models import (
    InstallPlan,
    WheelArtifact,
    canonical_plan_payload,
    computed_plan_sha256,
    new_plan_id,
)

_PINNED_REQUIREMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*==[A-Za-z0-9][A-Za-z0-9_.+!-]*$")


class InstallPlanResolver(Protocol):
    async def resolve(self, package_name: str, version: str) -> InstallPlan: ...


def create_install_plan(
    *, wheels: list[WheelArtifact], direct_requirements: list[str],
    python_version: str, platform_tag: str, image_digest: str,
    installer_version: str, estimated_bytes: int,
    execution_backend: Literal["podman", "local_dev"] = "podman",
    expires_at: datetime | None = None,
    compatible_wheel_tags: frozenset[str] | None = None,
) -> InstallPlan:
    if any(not _PINNED_REQUIREMENT.fullmatch(item) for item in direct_requirements):
        raise ValueError("INVALID_INSTALL_ARGUMENT")
    if not wheels:
        raise ValueError("INSTALL_WHEELS_REQUIRED")
    wheel_versions = {
        (wheel.distribution.lower().replace("_", "-").replace(".", "-"), wheel.version)
        for wheel in wheels
    }
    for requirement in direct_requirements:
        name, version = requirement.split("==", 1)
        normalized_name = name.lower().replace("_", "-").replace(".", "-")
        if (normalized_name, version) not in wheel_versions:
            raise ValueError("DIRECT_WHEEL_MISMATCH")
    if len({wheel.distribution.lower().replace("_", "-") for wheel in wheels}) != len(wheels):
        raise ValueError("DUPLICATE_DISTRIBUTION")
    if compatible_wheel_tags is not None:
        incompatible = any(not (set(wheel.tags) & compatible_wheel_tags) for wheel in wheels)
    else:
        incompatible = any(platform_tag not in wheel.tags and "py3-none-any" not in wheel.tags for wheel in wheels)
    if incompatible:
        raise ValueError("INCOMPATIBLE_WHEEL")
    if estimated_bytes < sum(wheel.size for wheel in wheels):
        raise ValueError("INSTALL_SIZE_UNDERSTATED")
    expiry = expires_at or datetime.now(timezone.utc) + timedelta(minutes=30)
    if expiry.tzinfo is None or expiry <= datetime.now(timezone.utc):
        raise ValueError("INVALID_PLAN_EXPIRY")
    wheels_sorted = tuple(sorted(wheels, key=lambda item: (item.distribution.lower(), item.version, item.filename)))
    requirements = tuple(sorted(set(direct_requirements), key=str.lower))
    payload = {
        "python_version": python_version,
        "platform_tag": platform_tag,
        "image_digest": image_digest,
        "installer_version": installer_version,
        "wheels": [wheel.model_dump(mode="json") for wheel in wheels_sorted],
        "direct_requirements": list(requirements),
        "estimated_bytes": estimated_bytes,
        "approved_index": "https://pypi.org/simple",
    }
    if execution_backend != "podman":
        payload["execution_backend"] = execution_backend
    digest = hashlib.sha256(canonical_plan_payload(**payload)).hexdigest()
    plan = InstallPlan(
        plan_id=new_plan_id(), plan_sha256=digest, python_version=python_version,
        platform_tag=platform_tag, image_digest=image_digest,
        execution_backend=execution_backend,
        installer_version=installer_version, wheels=wheels_sorted,
        direct_requirements=requirements, estimated_bytes=estimated_bytes,
        expires_at=expiry,
    )
    if computed_plan_sha256(plan) != digest:
        raise ValueError("INSTALL_PLAN_HASH_MISMATCH")
    return plan


class UVPlanResolver:
    """Resolve exact PyPI candidates to a complete, wheel-only hashed lock."""

    def __init__(
        self,
        *,
        image_digest: str,
        metadata_provider,
        uv_executable: str = "uv",
        execution_backend: Literal["podman", "local_dev"] = "podman",
        command_runner: Callable[[list[str], bytes], Awaitable[tuple[int, bytes, bytes]]] | None = None,
    ) -> None:
        self.image_digest = image_digest
        self.metadata_provider = metadata_provider
        self.uv_executable = uv_executable
        self.execution_backend = execution_backend
        self.command_runner = command_runner or self._run_uv

    async def resolve(self, package_name: str, version: str) -> InstallPlan:
        if not _PINNED_REQUIREMENT.fullmatch(f"{package_name}=={version}"):
            raise ValueError("INVALID_INSTALL_ARGUMENT")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.image_digest):
            raise ValueError("IMAGE_DIGEST_REQUIRED")
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        if python_version != "3.12":
            raise RuntimeError("INSTALL_PYTHON_UNSUPPORTED")
        platform_triple = _current_uv_platform()
        if platform_triple is None:
            raise RuntimeError("INSTALL_PLATFORM_UNSUPPORTED")
        binary = shutil.which(self.uv_executable)
        if binary is None:
            raise RuntimeError("INSTALLER_UNAVAILABLE")
        version_code, version_out, _ = await self.command_runner([binary, "--version"], b"")
        if version_code != 0:
            raise RuntimeError("INSTALLER_VERSION_UNAVAILABLE")
        installer_version = version_out.decode("utf-8", errors="replace").strip()[:100]
        arguments = [
            binary, "pip", "compile", "-", "-o", "-", "--generate-hashes",
            "--only-binary", ":all:", "--no-annotate", "--no-header", "--no-config",
            "--index-url", "https://pypi.org/simple", "--python-version", python_version,
            "--python-platform", platform_triple,
        ]
        code, stdout, _ = await self.command_runner(arguments, f"{package_name}=={version}\n".encode())
        if code != 0:
            raise RuntimeError("DEPENDENCY_RESOLUTION_FAILED")
        if len(stdout) > 1_000_000:
            raise ValueError("INSTALL_LOCK_TOO_LARGE")
        locked = _parse_hashed_requirements(stdout.decode("utf-8", errors="strict"))
        if (package_name.lower().replace("_", "-").replace(".", "-"), version) not in locked:
            raise ValueError("DIRECT_REQUIREMENT_NOT_LOCKED")
        try:
            from packaging.tags import sys_tags
            from packaging.utils import parse_wheel_filename
        except ImportError as exc:
            raise RuntimeError("WHEEL_TAG_PARSER_UNAVAILABLE") from exc
        supported_order = list(sys_tags())
        supported = frozenset(str(tag) for tag in supported_order)
        wheels: list[WheelArtifact] = []
        total_bytes = 0
        for (distribution, locked_version), hashes in sorted(locked.items()):
            metadata = await self.metadata_provider.inspect(distribution, locked_version)
            releases = metadata.get("urls")
            if not isinstance(releases, list):
                raise ValueError("INVALID_PYPI_RELEASE_FILES")
            candidates = []
            for release in releases:
                if not isinstance(release, dict) or release.get("packagetype") != "bdist_wheel":
                    continue
                digest = (release.get("digests") or {}).get("sha256")
                url = release.get("url")
                filename = release.get("filename")
                if not isinstance(digest, str) or digest not in hashes or not isinstance(filename, str):
                    continue
                try:
                    wheel_name, wheel_version, _, tags = parse_wheel_filename(filename)
                except (ValueError, TypeError):
                    continue
                compatible = [tag for tag in supported_order if tag in tags]
                normalized_wheel_name = str(wheel_name).lower().replace("_", "-").replace(".", "-")
                if (
                    not compatible or normalized_wheel_name != distribution
                    or str(wheel_version) != locked_version
                ):
                    continue
                candidates.append((supported_order.index(compatible[0]), release, tags))
            if not candidates:
                raise ValueError("NO_COMPATIBLE_HASHED_WHEEL")
            _, selected, tags = min(candidates, key=lambda item: item[0])
            file_url = selected.get("url")
            size = selected.get("size")
            digest = (selected.get("digests") or {}).get("sha256")
            if not isinstance(file_url, str) or not isinstance(size, int) or not isinstance(digest, str):
                raise ValueError("INVALID_PYPI_RELEASE_FILE")
            wheel = WheelArtifact(
                filename=selected["filename"], url=file_url, sha256=digest,
                size=size, distribution=distribution, version=locked_version,
                requires_python=str(selected.get("requires_python") or ""),
                tags=tuple(sorted(str(tag) for tag in tags)),
            )
            wheels.append(wheel)
            total_bytes += size
        return create_install_plan(
            wheels=wheels, direct_requirements=[f"{package_name}=={version}"],
            python_version=python_version, platform_tag=sysconfig.get_platform(),
            image_digest=self.image_digest, installer_version=installer_version,
            estimated_bytes=total_bytes, compatible_wheel_tags=supported,
            execution_backend=self.execution_backend,
        )

    async def _run_uv(self, arguments: list[str], stdin: bytes) -> tuple[int, bytes, bytes]:
        from photomatagent.capability_acquisition.container_backend import _run_bounded_process

        with tempfile.TemporaryDirectory(prefix="photomat-cap-plan-") as temporary:
            env = {
                "PATH": f"{Path(arguments[0]).parent}{os.pathsep}{os.defpath}",
                "HOME": temporary,
                "UV_CACHE_DIR": str(Path(temporary) / "uv-cache"),
                "UV_NO_CONFIG": "1",
                "PYTHONIOENCODING": "utf-8",
            }
            code, stdout, stderr, timed_out, overflow = await _run_bounded_process(
                arguments, timeout_seconds=60.0, output_limit=1_000_000,
                cwd=Path(temporary), env=env, stdin_data=stdin,
            )
            if timed_out:
                raise RuntimeError("DEPENDENCY_RESOLUTION_TIMEOUT")
            if overflow:
                raise ValueError("INSTALL_RESOLVER_OUTPUT_LIMIT")
            return code, stdout, stderr


def _parse_hashed_requirements(text: str) -> dict[tuple[str, str], set[str]]:
    locked: dict[tuple[str, str], set[str]] = {}
    current: tuple[str, str] | None = None
    for raw in text.splitlines():
        line = raw.strip().rstrip("\\").strip()
        if not line or line.startswith("#"):
            continue
        requirement = re.match(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)==([A-Za-z0-9][A-Za-z0-9_.+!-]*)", line)
        if requirement:
            name = requirement.group(1).lower().replace("_", "-").replace(".", "-")
            current = (name, requirement.group(2))
            locked.setdefault(current, set())
        if "--hash=" in line:
            if current is None:
                raise ValueError("ORPHAN_INSTALL_HASH")
            hashes = re.findall(r"--hash=sha256:([0-9a-fA-F]{64})", line)
            if not hashes:
                raise ValueError("INVALID_INSTALL_HASH")
            locked[current].update(value.lower() for value in hashes)
    if not locked or any(not hashes for hashes in locked.values()):
        raise ValueError("INSTALL_HASH_REQUIRED")
    return locked


def _current_uv_platform() -> str | None:
    machine = sysconfig.get_platform().lower()
    if machine.startswith("linux-"):
        architecture = machine.removeprefix("linux-").replace("-", "_")
        mapped = {"x86_64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}.get(architecture)
        return f"{mapped}-unknown-linux-gnu" if mapped else None
    return None
