"""Bounded public HTTPS requests with explicit destination policy."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode, urljoin, urlsplit

from photomatagent.tools.base import Tool, ToolResult
from photomatagent.tools.exposure import ToolExposure


class SourceError(ValueError):
    """Safe source diagnostic without URLs or credentials."""


@dataclass(frozen=True)
class SourceResponse:
    url: str
    content_type: str
    body: bytes


async def _addresses(host: str) -> list[str]:
    records = await asyncio.get_running_loop().getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(str(record[4][0]) for record in records))


Resolver = Callable[[str], Awaitable[list[str]]]
URLPolicy = Callable[[str], bool]


def _host(url: str) -> str:
    try:
        parts = urlsplit(url)
        if (parts.scheme == "https" and parts.hostname and parts.port in (None, 443)
                and parts.username is None and parts.password is None and not parts.fragment
                and not any(ch.isspace() or ord(ch) < 32 for ch in url) and "\\" not in url
                and len(url) <= 8192):
            return parts.hostname.rstrip(".").lower().encode("idna").decode("ascii")
    except (ValueError, UnicodeError):
        pass
    raise SourceError("URL_NOT_ALLOWED")


class PublicSourceClient:
    def __init__(
        self, hosts: frozenset[str], *, resolver: Resolver = _addresses, transport: Any = None,
        url_policy: URLPolicy | None = None, fixture_logical_host: bool = False,
        request_quota: int | None = None,
    ) -> None:
        self._hosts = hosts
        self._resolver = resolver
        self._transport = transport
        self._url_policy = url_policy
        self._fixture_logical_host = fixture_logical_host
        self._remaining_requests = request_quota

    async def get(
        self, url: str, *, max_bytes: int = 1_000_000, headers: dict[str, str] | None = None,
        accept: frozenset[str] | None = None, max_bytes_by_media: dict[str, int] | None = None,
    ) -> SourceResponse:
        try:
            import httpx
        except ImportError as exc:
            raise SourceError("HTTP_CLIENT_UNAVAILABLE") from exc

        initial_host = _host(url)
        try:
            async with asyncio.timeout(20):
                for _ in range(4):
                    host = _host(url)
                    if host not in self._hosts or (self._url_policy and not self._url_policy(url)):
                        raise SourceError("HOST_NOT_ALLOWED")
                    if self._remaining_requests is not None:
                        if self._remaining_requests <= 0:
                            raise SourceError("SOURCE_REQUEST_QUOTA_EXHAUSTED")
                        self._remaining_requests -= 1
                    target = httpx.URL(url)
                    # In-process compatibility fixtures assert logical URLs.
                    # A real transport always connects to a checked address.
                    if not (self._fixture_logical_host and isinstance(self._transport, httpx.MockTransport)):
                        addresses = [ipaddress.ip_address(value) for value in await self._resolver(host)]
                        if not addresses or any(not address.is_global or address.is_multicast or address.is_reserved for address in addresses):
                            raise SourceError("PRIVATE_OR_UNROUTABLE_ADDRESS")
                        address = next((item for item in addresses if item.version == 4), addresses[0])
                        target = target.copy_with(host=str(address))
                    request_headers = dict(headers or {}) if host == initial_host else {}
                    request_headers["Host"] = host
                    request_headers["Accept-Encoding"] = "identity"
                    async with httpx.AsyncClient(
                        transport=self._transport, timeout=10, follow_redirects=False, trust_env=False,
                    ) as client:
                        async with client.stream(
                            "GET", target, headers=request_headers, extensions={"sni_hostname": host},
                        ) as response:
                            if response.status_code in {301, 302, 303, 307, 308}:
                                location = response.headers.get("location")
                                if not location:
                                    raise SourceError("INVALID_REDIRECT")
                                url = urljoin(url, location)
                                continue
                            if response.status_code == 429:
                                raise SourceError("RATE_LIMITED")
                            if response.status_code in {401, 403}:
                                raise SourceError("ACCESS_DENIED")
                            if response.status_code == 404:
                                raise SourceError("NOT_FOUND")
                            if response.status_code != 200:
                                raise SourceError("UPSTREAM_HTTP_ERROR")
                            media = response.headers.get("content-type", "").split(";", 1)[0].lower().strip()
                            if accept is not None and media not in accept:
                                raise SourceError("UNSUPPORTED_CONTENT_TYPE")
                            if response.headers.get("content-encoding", "identity").lower() != "identity":
                                raise SourceError("UNSUPPORTED_CONTENT_ENCODING")
                            limit = min(max_bytes, (max_bytes_by_media or {}).get(media, max_bytes))
                            body = bytearray()
                            chunks = (_loaded_body(response.content) if response.is_stream_consumed
                                      else response.aiter_raw(chunk_size=65536))
                            async for chunk in chunks:
                                if len(body) + len(chunk) > limit:
                                    raise SourceError("RESPONSE_TOO_LARGE")
                                body.extend(chunk)
                            return SourceResponse(url, media, bytes(body))
        except (httpx.HTTPError, TimeoutError, OSError) as exc:
            raise SourceError("NETWORK_UNAVAILABLE") from exc
        raise SourceError("REDIRECT_LIMIT")

    async def json(self, url: str, *, headers: dict[str, str] | None = None) -> dict[str, Any]:
        response = await self.get(url, headers=headers, accept=frozenset({"application/json", "application/vnd.github+json"}))
        try:
            value = json.loads(response.body)
        except (ValueError, RecursionError) as exc:
            raise SourceError("INVALID_JSON") from exc
        if not isinstance(value, dict):
            raise SourceError("INVALID_JSON")
        return value


async def _loaded_body(content: bytes):
    yield content


async def public_dns_for_synthetic_addresses(host: str) -> list[str]:
    """Opt-in resolver for fake-IP DNS. The requesting client still validates and pins every address."""
    addresses = await _addresses(host)
    synthetic = ipaddress.ip_network("198.18.0.0/15")
    if not addresses or not all(ipaddress.ip_address(address) in synthetic for address in addresses):
        return addresses

    async def bootstrap(_host: str) -> list[str]:
        return ["8.8.8.8"]

    dns = PublicSourceClient(
        frozenset({"dns.google"}), resolver=bootstrap,
        url_policy=lambda url: urlsplit(url).path == "/resolve",
        request_quota=1,
    )
    query = urlencode({"name": host, "type": "A", "edns_client_subnet": "0.0.0.0/0"})
    try:
        reply = await dns.json(f"https://dns.google/resolve?{query}")
        answers = reply.get("Answer")
        if (reply.get("Status") != 0 or reply.get("TC") is not False
                or reply.get("Question") != [{"name": host + ".", "type": 1}]
                or not isinstance(answers, list) or len(answers) > 64):
            raise ValueError("Invalid DNS reply")
        return [str(ipaddress.IPv4Address(item["data"])) for item in answers if item["type"] == 1]
    except (ValueError, KeyError, TypeError) as exc:
        raise SourceError("PUBLIC_DNS_UNAVAILABLE") from exc


def tool_result(data: dict[str, Any], *, error: bool = False) -> ToolResult:
    return ToolResult(output=json.dumps(data, ensure_ascii=False), data=data, is_error=error)


def safe_result_url(value: object) -> str:
    if not isinstance(value, str) or len(value) > 2048:
        return ""
    try:
        _host(value)
    except SourceError:
        return ""
    return value


class WebSearchService:
    def __init__(self, endpoint: str, token: str = "", client: PublicSourceClient | None = None) -> None:
        self._endpoint = endpoint
        self._token = token
        self._client = client
        if endpoint and client is None:
            try:
                self._client = PublicSourceClient(frozenset({_host(endpoint)}), request_quota=64)
            except SourceError:
                pass

    async def search(self, query: str, limit: int) -> dict[str, Any]:
        if not self._endpoint:
            return {"diagnostic": "NOT_CONFIGURED"}
        try:
            if self._client is None:
                raise SourceError("URL_NOT_ALLOWED")
            client = self._client
            url = self._endpoint + ("&" if "?" in self._endpoint else "?") + urlencode({"q": query, "limit": limit})
            headers = {"Authorization": f"Bearer {self._token}"} if self._token else None
            payload = await client.json(url, headers=headers)
            rows = payload.get("results")
            if not isinstance(rows, list):
                raise SourceError("INVALID_SEARCH_RESPONSE")
        except SourceError as exc:
            return {"diagnostic": str(exc)}
        hits = []
        for row in rows[:limit]:
            if not isinstance(row, dict) or not (url := safe_result_url(row.get("url"))):
                continue
            hits.append({"url": url, "title": str(row.get("title", ""))[:240],
                         "snippet": str(row.get("snippet", ""))[:400]})
        return {"hits": hits, "source_role": "discovery_leads"}


class WebSearchTool(Tool):
    name = "web.search"
    description = "Search public web sources through the operator-configured HTTPS provider. Results are discovery leads."
    short_description = "Search public web pages."
    namespace = "web"
    source = "builtin"
    exposure = ToolExposure.DEFERRED
    tags = ("search", "website", "software", "official documentation")
    input_schema = {
        "type": "object", "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 500},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10},
        }, "required": ["query"], "additionalProperties": False,
    }

    def __init__(self, endpoint: str, token: str = "", client: PublicSourceClient | None = None,
                 *, service: WebSearchService | None = None) -> None:
        self.service = service or WebSearchService(endpoint, token, client)

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        data = await self.service.search(arguments["query"], int(arguments.get("limit", 5)))
        return tool_result(data, error="diagnostic" in data)


class WebFetchTool(Tool):
    name = "web.fetch"
    description = "Read bounded page text from an operator-approved public host. Page text is unattested."
    short_description = "Read text on an approved public website."
    namespace = "web"
    source = "builtin"
    exposure = ToolExposure.DEFERRED
    tags = ("read", "public web", "website", "documentation", "fetch")
    input_schema = {"type": "object", "properties": {"url": {"type": "string", "maxLength": 2048}},
                    "required": ["url"], "additionalProperties": False}

    def __init__(self, client: PublicSourceClient) -> None:
        self._client = client

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        from photomatagent.scientific.capabilities.literature.source_fetch import _PageText

        url = arguments["url"]
        try:
            _host(url)
            if urlsplit(url).query:
                raise SourceError("QUERY_PARAMETERS_NOT_ALLOWED")
            page = await self._client.get(url, accept=frozenset({"text/html", "text/plain", "application/xhtml+xml"}))
        except SourceError as exc:
            return tool_result({"diagnostic": str(exc)}, error=True)
        text = page.body.decode("utf-8", errors="replace")
        if page.content_type != "text/plain":
            parser = _PageText()
            parser.feed(text)
            text = "\n".join(parser.parts)
        return tool_result({"url": page.url.split("?", 1)[0], "sha256": hashlib.sha256(page.body).hexdigest(),
                            "text": text[:8000], "truncated": len(text) > 8000,
                            "content_role": "unattested_web_text"})
