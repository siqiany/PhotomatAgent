"""Federated bibliography and open-access location tools."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from copy import deepcopy
import os
import re
import time
from typing import Any
from urllib.parse import quote, urlencode, urlsplit

from photomatagent.capability_acquisition.web_sources import (
    PublicSourceClient,
    SourceError,
    public_dns_for_synthetic_addresses,
    safe_result_url,
    tool_result,
)
from photomatagent.errors import ToolExecutionError
from photomatagent.scientific.capabilities.literature.documents import DocumentService
from photomatagent.tools.base import Tool, ToolResult
from photomatagent.tools.exposure import ToolExposure

_ENDPOINTS = {
    "crossref": "https://api.crossref.org/works",
    "openalex": "https://api.openalex.org/works",
    "semantic_scholar": "https://api.semanticscholar.org/graph/v1/paper/search",
    "zenodo": "https://zenodo.org/api/records",
}
_UNPAYWALL = "https://api.unpaywall.org/v2/"
_HOSTS = frozenset(str(urlsplit(url).hostname) for url in (*_ENDPOINTS.values(), _UNPAYWALL))
_DOI = re.compile(r"10\.\d{4,9}/[^\s?#]{1,200}", re.IGNORECASE)
_ARXIV = re.compile(r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?", re.IGNORECASE)


def _doi(value: object) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip().casefold().removeprefix("https://doi.org/").removeprefix("doi:")
    return value if _DOI.fullmatch(value) else ""


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _title(value: object) -> str:
    if isinstance(value, list) and value:
        value = value[0]
    return value[:350] if isinstance(value, str) else ""


def _crossref_relations(record: dict[str, Any]) -> list[dict[str, str]]:
    links = []
    for relation, values in _mapping(record.get("relation")).items():
        if not isinstance(relation, str) or not isinstance(values, list):
            continue
        for target in values[:8]:
            if (isinstance(target, dict) and target.get("id-type") == "doi"
                    and (doi := _doi(target.get("id")))):
                links.append({"relationship": relation[:100], "doi": doi})
            if len(links) == 8:
                return links
    return links


def _row(provider: str, record: dict[str, Any]) -> dict[str, Any]:
    if provider == "openalex":
        return {"id": record.get("id"), "doi": _doi(record.get("doi")),
                "title": _title(record.get("display_name")), "record_type": record.get("type"),
                "year": record.get("publication_year"), "url": safe_result_url(record.get("id")),
                "fulltext_hint": safe_result_url(_mapping(record.get("best_oa_location")).get("pdf_url"))}
    if provider == "crossref":
        date = _mapping(record.get("published")).get("date-parts", [])
        year = date[0][0] if isinstance(date, list) and date and isinstance(date[0], list) and date[0] else None
        return {"id": record.get("DOI"), "doi": _doi(record.get("DOI")),
                "title": _title(record.get("title")), "record_type": record.get("type"),
                "year": year, "url": safe_result_url(record.get("URL")),
                "relationships": _crossref_relations(record)}
    if provider == "semantic_scholar":
        identifiers = _mapping(record.get("externalIds"))
        return {"id": record.get("paperId"), "doi": _doi(identifiers.get("DOI")),
                "title": _title(record.get("title")), "record_type": "paper",
                "year": record.get("year"), "url": safe_result_url(record.get("url")),
                "arxiv_id": identifiers.get("ArXiv"),
                "fulltext_hint": safe_result_url(_mapping(record.get("openAccessPdf")).get("url"))}
    metadata = _mapping(record.get("metadata"))
    return {"id": record.get("id"), "doi": _doi(metadata.get("doi")),
            "title": _title(metadata.get("title")), "record_type": "archive_record",
            "url": safe_result_url(_mapping(record.get("links")).get("html"))}


class FederatedLiterature:
    def __init__(
        self, enabled: frozenset[str], client: PublicSourceClient | None = None,
        credentials: dict[str, str] | None = None,
        *, documents: DocumentService | None = None,
    ) -> None:
        self.enabled = enabled
        self.client = client or PublicSourceClient(_HOSTS, request_quota=96)
        self.credentials = credentials or {}
        self.documents = documents
        self._search_cache: OrderedDict[tuple[str, str, int, int], tuple[float, str, list[dict[str, Any]]]] = OrderedDict()

    @classmethod
    def configured(cls, documents: DocumentService | None = None) -> FederatedLiterature:
        enabled = frozenset(name.strip() for name in os.environ.get("PHOTOMATAGENT_LITERATURE_SOURCES", "").split(","))
        client = None
        if os.environ.get("PHOTOMATAGENT_LITERATURE_PUBLIC_DNS") == "1":
            client = PublicSourceClient(_HOSTS, resolver=public_dns_for_synthetic_addresses, request_quota=96)
        return cls(enabled, client=client, credentials={
            "openalex": os.environ.get("OPENALEX_API_KEY", ""),
            "crossref": os.environ.get("CROSSREF_MAILTO", ""),
            "semantic_scholar": os.environ.get("SEMANTIC_SCHOLAR_API_KEY", ""),
            "zenodo": os.environ.get("ZENODO_API_TOKEN", ""),
            "unpaywall": os.environ.get("UNPAYWALL_EMAIL", ""),
        }, documents=documents)

    async def _search(self, provider: str, query: str, limit: int, page: int) -> tuple[str, list[dict[str, Any]]]:
        if provider not in self.enabled:
            return "NOT_CONFIGURED", []
        key = (provider, query, limit, page)
        cached = self._search_cache.get(key)
        if cached is not None and time.monotonic() < cached[0]:
            self._search_cache.move_to_end(key)
            return cached[1], deepcopy(cached[2])
        status, records = await self._request_page(provider, query, limit, page)
        if status != "SOURCE_REQUEST_QUOTA_EXHAUSTED":
            self._search_cache[key] = (time.monotonic() + (300 if status == "OK" else 15), status, deepcopy(records))
            self._search_cache.move_to_end(key)
            if len(self._search_cache) > 64:
                self._search_cache.popitem(last=False)
        return status, records

    async def _request_page(self, provider: str, query: str, limit: int, page: int) -> tuple[str, list[dict[str, Any]]]:
        credential = self.credentials.get(provider, "")
        headers: dict[str, str] = {}
        params: dict[str, str | int]
        if provider == "crossref":
            params = {"query.bibliographic": query, "rows": limit, "offset": (page - 1) * limit}
            if credential:
                params["mailto"] = credential
        elif provider == "openalex":
            params = {"search": query, "per-page": limit, "page": page,
                      "select": "id,doi,display_name,type,publication_year,best_oa_location"}
            if credential:
                params["api_key"] = credential
        elif provider == "semantic_scholar":
            params = {"query": query, "limit": limit, "offset": (page - 1) * limit,
                      "fields": "title,year,url,externalIds,openAccessPdf"}
            if credential:
                headers["x-api-key"] = credential
        else:
            params = {"q": query, "size": limit, "page": page}
            if credential:
                headers["Authorization"] = f"Bearer {credential}"
        try:
            response = await self.client.json(f"{_ENDPOINTS[provider]}?{urlencode(params)}", headers=headers)
            if provider == "crossref":
                rows = _mapping(response.get("message")).get("items")
            elif provider == "openalex":
                rows = response.get("results")
            elif provider == "semantic_scholar":
                rows = response.get("data", [] if response.get("total") == 0 else None)
            else:
                rows = _mapping(response.get("hits")).get("hits")
            if not isinstance(rows, list):
                return "INVALID_RESPONSE", []
        except SourceError as exc:
            return str(exc), []
        records: list[dict[str, Any]] = []
        for row in rows[:limit]:
            if not isinstance(row, dict):
                continue
            entry = _row(provider, row)
            if not isinstance(entry.get("id"), (int, str)) or not entry["title"]:
                continue
            # Limit even unexpected upstream string fields.
            bounded = {k: v[:2048] if isinstance(v, str) else v for k, v in entry.items()
                       if v is None or isinstance(v, (str, int))}
            if entry.get("relationships"):
                bounded["relationships"] = entry["relationships"]
            records.append(bounded)
        return "OK", records

    async def search(self, query: str, sources: tuple[str, ...], limit: int, page: int = 1) -> dict[str, Any]:
        if not 1 <= page <= 5 or not 1 <= limit <= 8:
            raise ValueError("SEARCH_PAGE_OR_LIMIT_INVALID")
        routes = tuple(dict.fromkeys(source for source in sources if source in _ENDPOINTS))
        results = await asyncio.gather(*(self._search(source, query, limit, page) for source in routes))
        status = {source: result[0] for source, result in zip(routes, results, strict=True)}
        groups: dict[str, dict[str, Any]] = {}
        # Interleave sources before applying the observation limit.
        for i in range(limit):
            for source, (_status, records) in zip(routes, results, strict=True):
                if i >= len(records):
                    continue
                row = records[i]
                if self.documents is not None and source == "crossref" and row.get("doi"):
                    for link in row.get("relationships", ()):
                        if link["relationship"] in {
                            "is-preprint-of", "has-preprint", "is-version-of", "has-version",
                            "is-correction-of", "is-corrected-by", "is-supplement-to", "is-supplemented-by",
                        }:
                            try:
                                await asyncio.to_thread(self.documents.record_publication_relation,
                                    "doi:" + row["doi"], link["relationship"], "doi:" + link["doi"])
                            except (ValueError, OSError, ToolExecutionError):
                                status[source] = "PUBLICATION_RELATION_PERSISTENCE_UNAVAILABLE"
                key = row.get("doi") or f"{source}:{row['id']}"
                if key not in groups and len(groups) >= limit:
                    continue
                group = groups.setdefault(key, {"doi": row.get("doi"), "title": row["title"], "records": []})
                group["records"].append({"provider": source, **row})
        return {"hits": list(groups.values()), "source_status": status, "page": page,
                "coverage": "metadata_only", "scientific_validation": "UNKNOWN"}

    async def resolve(self, identifier: str) -> dict[str, Any]:
        arxiv = identifier.removeprefix("https://arxiv.org/abs/").removeprefix("arXiv:")
        if _ARXIV.fullmatch(arxiv):
            return {"locations": [{"url": f"https://arxiv.org/pdf/{arxiv}",
                                   "version": "preprint", "provider": "arxiv"}],
                    "coverage": "location_only"}
        doi = _doi(identifier)
        if not doi:
            return {"diagnostic": "DOI_OR_ARXIV_ID_REQUIRED", "locations": []}
        email = self.credentials.get("unpaywall", "")
        if "unpaywall" not in self.enabled or not email:
            return {"diagnostic": "NOT_CONFIGURED", "locations": []}
        try:
            payload = await self.client.json(f"{_UNPAYWALL}{quote(doi, safe='/')}?{urlencode({'email': email})}")
        except SourceError as exc:
            return {"diagnostic": str(exc), "locations": []}
        if _doi(payload.get("doi")) != doi:
            return {"diagnostic": "IDENTIFIER_MISMATCH", "locations": []}
        locations = []
        best = _mapping(payload.get("best_oa_location"))
        if url := safe_result_url(best.get("url_for_pdf") or best.get("url")):
            locations.append({"url": url, "version": str(best.get("version", "unknown"))[:80],
                              "license": str(best.get("license") or "unknown")[:80], "provider": "unpaywall"})
        return {"doi": doi, "locations": locations, "coverage": "location_only"}


class LiteratureSearchTool(Tool):
    name = "literature.search"
    description = "Search configured academic metadata sources. Missing or failed sources are reported separately."
    short_description = "Search multiple academic sources."
    namespace = "literature"
    source = "builtin"
    exposure = ToolExposure.DEFERRED
    tags = ("papers", "literature", "openalex", "crossref", "semantic scholar", "zenodo")
    input_schema = {"type": "object", "properties": {
        "query": {"type": "string", "minLength": 1, "maxLength": 500},
        "sources": {"type": "array", "minItems": 1, "maxItems": len(_ENDPOINTS),
                    "items": {"type": "string", "enum": list(_ENDPOINTS)}},
        "limit": {"type": "integer", "minimum": 1, "maximum": 8},
        "page": {"type": "integer", "minimum": 1, "maximum": 5},
    }, "required": ["query"], "additionalProperties": False}

    def __init__(self, service: FederatedLiterature) -> None:
        self._service = service

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        return tool_result(await self._service.search(arguments["query"], tuple(arguments.get("sources", _ENDPOINTS)),
                                                      int(arguments.get("limit", 5)), int(arguments.get("page", 1))))


class ResolveFulltextTool(Tool):
    name = "literature.resolve_fulltext"
    description = "Find a public full-text location for a DOI or arXiv ID. This does not retrieve or verify the paper."
    short_description = "Locate accessible paper versions."
    namespace = "literature"
    source = "builtin"
    exposure = ToolExposure.DEFERRED
    tags = ("papers", "literature", "doi", "open access", "unpaywall", "arxiv")
    input_schema = {"type": "object", "properties": {"identifier": {"type": "string", "maxLength": 250}},
                    "required": ["identifier"], "additionalProperties": False}

    def __init__(self, service: FederatedLiterature) -> None:
        self._service = service

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        result = await self._service.resolve(arguments["identifier"])
        return tool_result(result, error="diagnostic" in result)
