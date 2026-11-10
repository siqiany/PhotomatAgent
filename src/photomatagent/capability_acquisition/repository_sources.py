"""GitHub repository metadata and commit-pinned script acquisition."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, urlencode

from pydantic import Field

from photomatagent.capability_acquisition.models import StrictModel
from photomatagent.capability_acquisition.web_sources import (
    PublicSourceClient,
    SourceError,
    safe_result_url,
    tool_result,
)
from photomatagent.persistence.atomic import atomic_write
from photomatagent.tools.base import Tool, ToolResult
from photomatagent.tools.exposure import ToolExposure
from photomatagent.workspace import Workspace

_RAW_HOST = "raw.githubusercontent.com"
_API_HOST = "api.github.com"


class RepositorySearchService:
    def __init__(self, client: PublicSourceClient | None = None) -> None:
        self._client = client or PublicSourceClient(frozenset({_API_HOST}), request_quota=64)

    async def search(self, query: str, limit: int) -> dict[str, Any]:
        if os.environ.get("PHOTOMATAGENT_GITHUB_DISCOVERY") != "1":
            return {"diagnostic": "NOT_CONFIGURED"}
        params = urlencode({"q": f"{query} is:public", "per_page": limit})
        try:
            payload = await self._client.json(f"https://{_API_HOST}/search/repositories?{params}")
        except SourceError as exc:
            return {"diagnostic": str(exc)}
        if not isinstance(payload.get("items"), list):
            return {"diagnostic": "INVALID_RESPONSE"}
        hits = []
        for item in payload["items"][:limit]:
            if not isinstance(item, dict) or item.get("private") is not False:
                continue
            license_metadata = item.get("license")
            license_id = license_metadata.get("spdx_id") if isinstance(license_metadata, dict) else None
            if url := safe_result_url(item.get("html_url")):
                hits.append({"url": url, "repository": str(item.get("full_name", ""))[:150],
                             "description": str(item.get("description") or "")[:400],
                             "license_hint": str(license_id or "unknown")[:50],
                             "default_branch": str(item.get("default_branch") or "")[:100],
                             "pushed_at": str(item.get("pushed_at") or "")[:60],
                             "archived": item.get("archived") if isinstance(item.get("archived"), bool) else None})
        return {"hits": hits, "coverage": "repository_metadata", "scientific_validation": "UNKNOWN"}


class RepositorySearchTool(Tool):
    name = "web.search_repositories"
    namespace = "web"
    source = "builtin"
    exposure = ToolExposure.DEFERRED
    description = "Search public GitHub repository metadata for implementation leads. Repository claims are unverified."
    short_description = "Search public software repositories."
    tags = ("github", "software", "code", "repository")
    input_schema = {"type": "object", "properties": {
        "query": {"type": "string", "minLength": 1, "maxLength": 500},
        "limit": {"type": "integer", "minimum": 1, "maximum": 8},
    }, "required": ["query"], "additionalProperties": False}

    def __init__(self, client: PublicSourceClient | None = None, *, service: RepositorySearchService | None = None) -> None:
        self.service = service or RepositorySearchService(client)

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        data = await self.service.search(arguments["query"], int(arguments.get("limit", 5)))
        return tool_result(data, error="diagnostic" in data)


class ScriptRequest(StrictModel):
    owner: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9-]{0,38}$")
    repository: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,100}$")
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    file: str = Field(min_length=1, max_length=250)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license: str = Field(min_length=1, max_length=80)
    license_source: str = Field(min_length=1, max_length=1000)


class FetchRepositoryScriptTool(Tool):
    name = "reproduction.fetch_script"
    namespace = "reproduction"
    source = "builtin"
    exposure = ToolExposure.DEFERRED
    description = "Acquire a single SHA-256-pinned public Python source file at a GitHub commit. Acquisition does not execute code."
    short_description = "Acquire a pinned Python reference script."
    tags = ("github", "code", "source", "reproduction")
    input_schema = ScriptRequest.model_json_schema()

    def __init__(
        self, workspace: Workspace, scratch_prefix: str | None,
        client: PublicSourceClient | None = None,
    ) -> None:
        self._workspace = workspace
        self._scratch = scratch_prefix or "tmp/reproduction"
        self._client = client or PublicSourceClient(frozenset({_RAW_HOST}), request_quota=64)

    def _store(self, request: ScriptRequest, url: str, body: bytes) -> dict[str, Any]:
        relative = f"{self._scratch}/reproduction-sources/{request.sha256}.py"
        path = self._workspace.resolve(relative, must_exist=False)
        if self._workspace.relative(path) != relative:
            raise ValueError("SOURCE_PATH_CHANGED")
        if path.exists() and path.read_bytes() != body:
            raise ValueError("SOURCE_PATH_CHANGED")
        atomic_write(path, body)
        manifest = {
            "sha256": request.sha256, "script_path": relative, "url": url, "size": len(body),
            "commit": request.commit, "license_declared": request.license,
            "license_source": request.license_source, "license_verified": False,
            "retrieved_at": datetime.now(UTC).isoformat(),
        }
        # The bytes may be shared by multiple upstream locations and commits.
        origin = hashlib.sha256(url.encode()).hexdigest()
        metadata = self._workspace.resolve(
            f".photomatagent/reproduction/sources/{request.sha256}-{origin}.json", must_exist=False,
        )
        atomic_write(metadata, json.dumps(manifest, ensure_ascii=False).encode())
        return manifest

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        if os.environ.get("PHOTOMATAGENT_GITHUB_DISCOVERY") != "1":
            return tool_result({"diagnostic": "NOT_CONFIGURED"}, error=True)
        request = ScriptRequest.model_validate(arguments)
        path = PurePosixPath(request.file)
        if (path.is_absolute() or ".." in path.parts or "\\" in request.file or path.suffix != ".py"
                or request.repository in {".", ".."} or not safe_result_url(request.license_source)):
            return tool_result({"diagnostic": "INVALID_SOURCE_FILE_OR_LICENSE_REFERENCE"}, error=True)
        url = f"https://{_RAW_HOST}/{request.owner}/{request.repository}/{request.commit}/{quote(str(path), safe='/')}"
        try:
            response = await self._client.get(url, max_bytes=1_000_000,
                                              accept=frozenset({"text/plain", "application/octet-stream"}))
            if response.url != url or hashlib.sha256(response.body).hexdigest() != request.sha256:
                raise SourceError("SOURCE_HASH_OR_ORIGIN_CHANGED")
            result = await asyncio.to_thread(self._store, request, url, response.body)
        except SourceError as exc:
            return tool_result({"diagnostic": str(exc)}, error=True)
        except (OSError, ValueError):
            return tool_result({"diagnostic": "SOURCE_STORAGE_FAILED"}, error=True)
        return tool_result(result)


class ArchiveRequest(StrictModel):
    owner: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9-]{0,38}$")
    repository: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,100}$")
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entry: str = Field(min_length=1, max_length=250)
    files: tuple[str, ...] = Field(min_length=1, max_length=64)
    license: str = Field(min_length=1, max_length=80)
    license_source: str = Field(min_length=1, max_length=1000)


_ARCHIVE_DOWNLOAD_LIMIT = 12 * 1024 * 1024
_ARCHIVE_EXPANDED_LIMIT = 40 * 1024 * 1024
_ARCHIVE_SELECTED_LIMIT = 32 * 1024 * 1024


def _archive_files(content: bytes, wanted: set[str]) -> dict[str, bytes]:
    """Read only explicitly selected regular members, without extracting a tree."""
    import gzip
    from io import BytesIO
    import tarfile

    with gzip.GzipFile(fileobj=BytesIO(content)) as compressed:
        raw = compressed.read(_ARCHIVE_EXPANDED_LIMIT + 1)
    if len(raw) > _ARCHIVE_EXPANDED_LIMIT:
        raise ValueError("SOURCE_ARCHIVE_EXPANSION_LIMIT")
    selected = {}
    root = None
    seen: set[str] = set()
    total = 0
    with tarfile.open(fileobj=BytesIO(raw), mode="r:") as archive:
        for index, member in enumerate(archive):
            if index >= 2048:
                raise ValueError("SOURCE_ARCHIVE_MEMBER_LIMIT")
            name = member.name.removesuffix("/") if member.isdir() else member.name
            path = PurePosixPath(name)
            if (path.is_absolute() or path.as_posix() != name or ".." in path.parts
                    or not path.parts or len(name) > 500
                    or any(ord(ch) < 32 or ch in "\\:" for ch in name)
                    or (not member.isdir() and (not member.isreg() or member.issparse()))):
                raise ValueError("SOURCE_ARCHIVE_MEMBER_INVALID")
            if root is None:
                root = path.parts[0]
            if path.parts[0] != root or name.casefold() in seen:
                raise ValueError("SOURCE_ARCHIVE_MEMBER_INVALID")
            seen.add(name.casefold())
            relative = "/".join(path.parts[1:])
            if relative not in wanted:
                continue
            if not member.isreg() or member.size < 0 or member.size + total > _ARCHIVE_SELECTED_LIMIT:
                raise ValueError("SOURCE_ARCHIVE_MEMBER_INVALID")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError("SOURCE_ARCHIVE_MEMBER_INVALID")
            with handle:
                data = handle.read(member.size + 1)
            if len(data) != member.size:
                raise ValueError("SOURCE_ARCHIVE_MEMBER_INVALID")
            total += len(data)
            selected[relative] = data
    if set(selected) != wanted:
        raise ValueError("SOURCE_ARCHIVE_MEMBER_MISSING")
    return selected


class FetchRepositoryArchiveTool(Tool):
    name = "reproduction.fetch_archive"
    namespace = "reproduction"
    source = "builtin"
    exposure = ToolExposure.DEFERRED
    description = "Acquire selected regular files from a hash-pinned public GitHub commit archive for offline reproduction."
    short_description = "Acquire a verified set of repository files."
    tags = ("github", "code", "archive", "reproduction")
    input_schema = ArchiveRequest.model_json_schema()

    def __init__(
        self, workspace: Workspace, scratch_prefix: str | None, client: PublicSourceClient | None = None,
    ) -> None:
        self._workspace = workspace
        self._scratch = scratch_prefix or "tmp/reproduction"
        self._client = client or PublicSourceClient(frozenset({_API_HOST, "codeload.github.com"}), request_quota=64)

    def _store(
        self, request: ArchiveRequest, url: str, final_url: str, content: bytes, wanted: set[str],
    ) -> dict[str, Any]:
        import tempfile

        selected = _archive_files(content, wanted)
        bundle_id = hashlib.sha256(
            json.dumps([url, request.sha256, request.entry, sorted(wanted)], separators=(",", ":")).encode()
        ).hexdigest()
        relative = f"{self._scratch}/reproduction-archives/{bundle_id}"
        destination = self._workspace.resolve(relative, must_exist=False)
        if self._workspace.relative(destination) != relative:
            raise ValueError("SOURCE_PATH_CHANGED")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
            staging = Path(scratch) / "bundle"
            for name, data in selected.items():
                atomic_write(staging / name, data)
            try:
                staging.rename(destination)
            except OSError:
                if not destination.is_dir() or destination.is_symlink():
                    raise
                for name, data in selected.items():
                    candidate = self._workspace.resolve(f"{relative}/{name}")
                    if candidate != destination / name:
                        raise ValueError("SOURCE_PATH_CHANGED")
                    with candidate.open("rb") as handle:
                        if handle.read(len(data) + 1) != data:
                            raise ValueError("SOURCE_PATH_CHANGED")
        manifest = {
            "bundle_id": bundle_id, "source_root": relative, "script_path": f"{relative}/{request.entry}",
            "source_files": sorted(wanted - {request.entry}), "commit": request.commit,
            "url": url, "final_url": final_url, "archive_sha256": request.sha256,
            "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
                      for name, data in sorted(selected.items())},
            "license_declared": request.license, "license_source": request.license_source,
            "license_verified": False, "retrieved_at": datetime.now(UTC).isoformat(),
        }
        metadata = self._workspace.resolve(
            f".photomatagent/reproduction/sources/bundle-{bundle_id}.json", must_exist=False,
        )
        atomic_write(metadata, json.dumps(manifest, ensure_ascii=False).encode())
        return {key: value for key, value in manifest.items() if key != "files"}

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        from urllib.parse import urlsplit
        from photomatagent.capability_acquisition.models import SourceTreeLayout
        import tarfile
        import zlib

        if os.environ.get("PHOTOMATAGENT_GITHUB_DISCOVERY") != "1":
            return tool_result({"diagnostic": "NOT_CONFIGURED"}, error=True)
        try:
            request = ArchiveRequest.model_validate(arguments)
            layout = SourceTreeLayout(entry=request.entry, inputs={str(i): name for i, name in enumerate(request.files)})
            if request.repository in {".", ".."} or not safe_result_url(request.license_source):
                raise ValueError("SOURCE_ARCHIVE_INPUT_INVALID")
            wanted = {layout.entry, *layout.inputs.values()}
            url = f"https://{_API_HOST}/repos/{request.owner}/{request.repository}/tarball/{request.commit}"
            response = await self._client.get(
                url, max_bytes=_ARCHIVE_DOWNLOAD_LIMIT,
                accept=frozenset({"application/x-gzip", "application/gzip", "application/octet-stream"}),
            )
            if (urlsplit(response.url).hostname != "codeload.github.com"
                    or hashlib.sha256(response.body).hexdigest() != request.sha256):
                raise SourceError("SOURCE_HASH_OR_ORIGIN_CHANGED")
            result = await asyncio.to_thread(self._store, request, url, response.url, response.body, wanted)
        except SourceError as exc:
            return tool_result({"diagnostic": str(exc)}, error=True)
        except (ValueError, OSError, EOFError, tarfile.TarError, zlib.error) as exc:
            reason = str(exc)
            code = reason if reason.startswith("SOURCE_ARCHIVE_") and reason.replace("_", "").isupper() else "SOURCE_ARCHIVE_INVALID"
            return tool_result({"diagnostic": code}, error=True)
        return tool_result(result)
