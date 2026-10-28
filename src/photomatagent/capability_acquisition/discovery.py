"""Real, source-grounded capability discovery with bounded HTTPS responses."""
from __future__ import annotations

import asyncio
import ipaddress
import json
import hashlib
import re
import socket
from datetime import datetime, timezone
from typing import Any, Literal, Protocol
from urllib.parse import urlencode, urljoin, urlsplit
from pathlib import Path

from photomatagent.capability_acquisition.models import (
    DiscoveryResult,
    InstallPlan,
    PackageCandidate,
    make_candidate,
)
from photomatagent.capability_acquisition.installation import InstallPlanResolver

MAX_RESPONSE_BYTES = 1_000_000
MAX_REDIRECTS = 3
DEFAULT_TIMEOUT_SECONDS = 10.0


class DiscoveryProvider(Protocol):
    async def search(self, query: str, limit: int) -> list[dict[str, Any]]: ...


class PackageMetadataProvider(Protocol):
    async def inspect(self, name: str, version: str) -> dict[str, Any]: ...


def _network_failure_code(
    exc: Exception,
) -> Literal["NETWORK_BLOCKED", "NETWORK_ERROR"] | None:
    """Classify known network/security failures without masking bad responses."""
    if isinstance(exc, ValueError):
        diagnostic = str(exc)
        if diagnostic == "PRIVATE_NETWORK_REDIRECT":
            return "NETWORK_BLOCKED"
        if diagnostic == "HOST_RESOLUTION_FAILED":
            return "NETWORK_ERROR"
        return None
    if isinstance(exc, (RuntimeError, TimeoutError, ConnectionError, OSError)):
        return "NETWORK_ERROR"
    if type(exc).__module__.startswith("httpx"):
        return "NETWORK_ERROR"
    return None


def validate_public_https_url(url: str, *, resolve: bool = False) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https" or not parsed.hostname or parsed.username
        or parsed.password or parsed.port not in (None, 443)
    ):
        raise ValueError("HTTPS_URL_REQUIRED")
    hostname = parsed.hostname.rstrip(".").lower()
    try:
        addresses = [ipaddress.ip_address(hostname)]
    except ValueError:
        addresses = []
        if resolve:
            try:
                addresses = [
                    ipaddress.ip_address(item[4][0])
                    for item in socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
                ]
            except (OSError, ValueError) as exc:
                raise ValueError("HOST_RESOLUTION_FAILED") from exc
    if any(not address.is_global for address in addresses):
        raise ValueError("PRIVATE_NETWORK_REDIRECT")
    return url


async def bounded_https_get(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = MAX_RESPONSE_BYTES,
) -> tuple[int, dict[str, str], bytes]:
    """Fetch one bounded HTTPS response and revalidate each redirect target."""
    try:
        import httpx
    except ImportError as exc:  # optional capability; base runtime remains usable
        raise RuntimeError("HTTP_CLIENT_UNAVAILABLE") from exc
    current = url
    current_headers = dict(headers or {})
    for _ in range(MAX_REDIRECTS + 1):
        validate_public_https_url(current, resolve=True)
        async with httpx.AsyncClient(
            timeout=timeout_seconds, follow_redirects=False, trust_env=False,
        ) as client:
            async with client.stream("GET", current, headers=current_headers) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("INVALID_REDIRECT")
                    next_url = urljoin(current, location)
                    validate_public_https_url(next_url, resolve=True)
                    before = urlsplit(current)
                    after = urlsplit(next_url)
                    if (before.scheme, before.hostname, before.port) != (
                        after.scheme, after.hostname, after.port
                    ):
                        current_headers = {
                            key: value for key, value in current_headers.items()
                            if key.lower() not in {"authorization", "cookie", "proxy-authorization"}
                        }
                    current = next_url
                    continue
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > max_bytes:
                        raise ValueError("RESPONSE_TOO_LARGE")
                return response.status_code, dict(response.headers), bytes(body)
    raise ValueError("TOO_MANY_REDIRECTS")


async def bounded_https_download(
    url: str,
    destination: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    timeout_seconds: float = 60.0,
    max_bytes: int = 1_000_000_000,
) -> None:
    """Stream one approved wheel to disk and verify exact size and SHA-256."""
    if not 0 < expected_size <= max_bytes:
        raise ValueError("DOWNLOAD_SIZE_INVALID")
    current = url
    total = 0
    digest = hashlib.sha256()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        for _ in range(MAX_REDIRECTS + 1):
            validate_public_https_url(current, resolve=True)
            if (urlsplit(current).hostname or "").lower() != "files.pythonhosted.org":
                raise ValueError("UNAPPROVED_WHEEL_SOURCE")
            async with _httpx_client(timeout_seconds) as client:
                async with client.stream("GET", current, headers={"Accept-Encoding": "identity"}) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise ValueError("INVALID_REDIRECT")
                        current = urljoin(current, location)
                        validate_public_https_url(current, resolve=True)
                        if (urlsplit(current).hostname or "").lower() != "files.pythonhosted.org":
                            raise ValueError("UNAPPROVED_WHEEL_SOURCE")
                        continue
                    if response.status_code < 200 or response.status_code >= 300:
                        raise RuntimeError("DOWNLOAD_HTTP_FAILED")
                    with destination.open("xb") as handle:
                        async for chunk in response.aiter_bytes():
                            total += len(chunk)
                            if total > min(expected_size, max_bytes):
                                raise ValueError("DOWNLOAD_SIZE_MISMATCH")
                            digest.update(chunk)
                            handle.write(chunk)
                    break
        else:
            raise ValueError("TOO_MANY_REDIRECTS")
        if total != expected_size or digest.hexdigest() != expected_sha256:
            raise ValueError("ARTIFACT_HASH_MISMATCH")
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def _httpx_client(timeout_seconds: float):
    import httpx

    return httpx.AsyncClient(
        timeout=timeout_seconds, follow_redirects=False, trust_env=False,
    )


class HTTPSearchProvider:
    """Adapter for a user-configured HTTPS search endpoint returning JSON hits."""

    def __init__(
        self, endpoint: str, *, token: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        validate_public_https_url(endpoint)
        self.endpoint = endpoint
        self.token = token
        self.timeout_seconds = timeout_seconds

    async def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        url = self.endpoint + ("&" if "?" in self.endpoint else "?") + urlencode({"q": query, "limit": limit})
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        status, _, body = await bounded_https_get(
            url, headers=headers, timeout_seconds=self.timeout_seconds,
        )
        if status < 200 or status >= 300:
            raise RuntimeError("SEARCH_HTTP_FAILED")
        payload = json.loads(body)
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ValueError("INVALID_SEARCH_RESPONSE")
        return payload["results"][:limit]


class PyPIMetadataProvider:
    """Exact-name/version lookup against PyPI's JSON API; never keyword search."""

    async def inspect(self, name: str, version: str) -> dict[str, Any]:
        from urllib.parse import quote

        normalized = name.lower().replace("_", "-").replace(".", "-")
        if (
            not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", normalized)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+!-]{0,99}", version)
        ):
            raise ValueError("INVALID_PACKAGE_COORDINATE")
        url = f"https://pypi.org/pypi/{quote(normalized, safe='')}/{quote(version, safe='')}/json"
        status, _, body = await bounded_https_get(url)
        if status == 404:
            raise LookupError("PACKAGE_VERSION_NOT_FOUND")
        if status < 200 or status >= 300:
            raise RuntimeError("PYPI_METADATA_FAILED")
        payload = json.loads(body)
        info = payload.get("info")
        if not isinstance(info, dict):
            raise ValueError("INVALID_PYPI_METADATA")
        if info.get("name", "").lower().replace("_", "-").replace(".", "-") != normalized:
            raise ValueError("PACKAGE_NAME_MISMATCH")
        if info.get("version") != version:
            raise ValueError("PACKAGE_VERSION_MISMATCH")
        return payload


class ExactPyPISearchProvider:
    """Built-in name lookup for workbench use when no keyword endpoint is set."""

    async def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        from urllib.parse import quote

        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_.-]{0,199})(?:==([A-Za-z0-9][A-Za-z0-9_.+!-]{0,99}))?", query.strip())
        if match is None:
            raise ValueError("EXACT_PACKAGE_NAME_REQUIRED")
        name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
        version = match.group(2)
        if version is None:
            url = f"https://pypi.org/pypi/{quote(name, safe='')}/json"
            status, _, body = await bounded_https_get(url)
            if status == 404:
                return []
            if status < 200 or status >= 300:
                raise RuntimeError("PYPI_METADATA_FAILED")
            info = json.loads(body).get("info")
            if not isinstance(info, dict) or not isinstance(info.get("version"), str):
                raise ValueError("INVALID_PYPI_METADATA")
            version = info["version"]
        return [{
            "title": f"{name} {version}",
            "url": f"https://pypi.org/project/{quote(name, safe='')}/{quote(version, safe='')}/",
            "snippet": "Exact PyPI package lookup",
            "name": name, "version": version,
        }][:limit]


class CapabilityDiscoveryService:
    def __init__(
        self,
        *,
        search_provider: DiscoveryProvider | None = None,
        metadata_provider: PackageMetadataProvider | None = None,
        plan_resolver: InstallPlanResolver | None = None,
    ) -> None:
        self.search_provider = search_provider
        self.metadata_provider = metadata_provider
        self.plan_resolver = plan_resolver
        self._candidates: dict[str, PackageCandidate] = {}
        self._plans: dict[str, InstallPlan] = {}

    async def search(self, query: str, *, limit: int) -> DiscoveryResult:
        if not query.strip() or len(query) > 500 or not 1 <= limit <= 10:
            return DiscoveryResult(code="INVALID_RESPONSE", diagnostics=("INVALID_QUERY_OR_LIMIT",))
        if self.search_provider is None:
            return DiscoveryResult(code="DISCOVERY_UNAVAILABLE", diagnostics=("SEARCH_PROVIDER_NOT_CONFIGURED",))
        if self.metadata_provider is None:
            return DiscoveryResult(code="DISCOVERY_UNAVAILABLE", diagnostics=("PACKAGE_METADATA_PROVIDER_NOT_CONFIGURED",))
        try:
            results = await asyncio.wait_for(self.search_provider.search(query, limit), timeout=DEFAULT_TIMEOUT_SECONDS + 2)
        except ValueError as exc:
            network_code = _network_failure_code(exc)
            if network_code:
                prefix = "SEARCH_BLOCKED" if network_code == "NETWORK_BLOCKED" else "SEARCH_FAILED"
                return DiscoveryResult(code=network_code, diagnostics=(f"{prefix}:{exc}",))
            return DiscoveryResult(code="INVALID_RESPONSE", diagnostics=(f"INVALID_SEARCH_RESPONSE:{type(exc).__name__}",))
        except Exception as exc:
            return DiscoveryResult(code="NETWORK_ERROR", diagnostics=(f"SEARCH_FAILED:{type(exc).__name__}",))
        deadline = asyncio.get_running_loop().time() + 30.0
        try:
            candidates: list[PackageCandidate] = []
            for item in results[:limit]:
                if not isinstance(item, dict):
                    raise ValueError("INVALID_SEARCH_RESULT")
                title, url, snippet = item.get("title"), item.get("url"), item.get("snippet")
                name, version = item.get("name"), item.get("version")
                if not all(isinstance(value, str) and value.strip() for value in (title, url, snippet, name, version)):
                    raise ValueError("INVALID_SEARCH_RESULT")
                title, url, snippet = str(title), str(url), str(snippet)
                name, version = str(name), str(version)
                validate_public_https_url(url)
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError("SEARCH_TOTAL_TIMEOUT")
                metadata = await asyncio.wait_for(
                    self.metadata_provider.inspect(name, version),
                    timeout=min(DEFAULT_TIMEOUT_SECONDS + 2, remaining),
                )
                info = metadata.get("info") if "info" in metadata else metadata
                if not isinstance(info, dict):
                    raise ValueError("INVALID_PACKAGE_METADATA")
                canonical_name = info.get("name")
                canonical_version = info.get("version")
                if not isinstance(canonical_name, str) or canonical_version != version:
                    raise ValueError("PACKAGE_METADATA_MISMATCH")
                project_urls = info.get("project_urls")
                project_url = info.get("project_url")
                if not project_url and isinstance(project_urls, dict):
                    project_url = project_urls.get("Homepage")
                sources = tuple(dict.fromkeys([url, project_url] if isinstance(project_url, str) else [url]))
                for source in sources:
                    validate_public_https_url(source)
                candidate = make_candidate(
                    name=canonical_name, version=version,
                    summary=str(info.get("summary") or snippet), source_urls=sources,
                    retrieved_at=datetime.now(timezone.utc),
                )
                self._candidates[candidate.candidate_id] = candidate
                candidates.append(candidate)
            return DiscoveryResult(code="OK", candidates=tuple(candidates))
        except Exception as exc:
            network_code = _network_failure_code(exc)
            if network_code:
                prefix = "METADATA_BLOCKED" if network_code == "NETWORK_BLOCKED" else "METADATA_LOOKUP_FAILED"
                detail = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
                return DiscoveryResult(code=network_code, diagnostics=(f"{prefix}:{detail}",))
            return DiscoveryResult(code="INVALID_RESPONSE", diagnostics=(f"INVALID_SEARCH_RESULT:{type(exc).__name__}",))

    async def inspect(self, candidate_id: str) -> dict[str, Any]:
        candidate = self._candidates.get(candidate_id)
        if candidate is None or self.metadata_provider is None:
            raise LookupError("CANDIDATE_NOT_FOUND")
        return await asyncio.wait_for(
            self.metadata_provider.inspect(candidate.normalized_name, candidate.version),
            timeout=DEFAULT_TIMEOUT_SECONDS + 2,
        )

    def candidate(self, candidate_id: str) -> PackageCandidate | None:
        return self._candidates.get(candidate_id)

    async def resolve_install_plan(self, candidate_id: str) -> InstallPlan:
        candidate = self._candidates.get(candidate_id)
        if candidate is None:
            raise LookupError("CANDIDATE_NOT_FOUND")
        if self.plan_resolver is None:
            raise RuntimeError("INSTALL_RESOLUTION_UNAVAILABLE")
        plan = await asyncio.wait_for(
            self.plan_resolver.resolve(candidate.normalized_name, candidate.version),
            timeout=60.0,
        )
        self._plans[plan.plan_id] = plan
        return plan

    def install_plan(self, plan_id: str) -> InstallPlan | None:
        return self._plans.get(plan_id)


def validate_search_sources(rows: list[dict[str, Any]]) -> None:
    """Reject source-less or non-HTTPS search records before metadata lookup."""
    for row in rows:
        url = row.get("url") if isinstance(row, dict) else None
        if not isinstance(url, str) or not url.strip():
            raise ValueError("DISCOVERY_SOURCE_REQUIRED")
        validate_public_https_url(url)
