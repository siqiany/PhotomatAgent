"""Provider-independent progressive tool-surface planning and diagnostics."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from photomatagent.models.types import ToolDefinition
from photomatagent.tools.base import Tool
from photomatagent.tools.exposure import ToolExposure
from photomatagent.tools.registry import ToolRegistry

BRIDGE_TOOL_NAMES = frozenset({"tool_search", "tool_describe", "tool_call"})
PROGRESSIVE_HELPER_TOOL_NAMES = BRIDGE_TOOL_NAMES


def estimate_tokens(chars: int) -> int:
    """Cheap diagnostics estimate; provider usage remains authoritative."""
    return math.ceil(max(chars, 0) / 4)


def serialized_definition(definition: ToolDefinition) -> str:
    """Serialize only fields that provider adapters put on the wire."""
    return json.dumps(
        {
            "name": definition.name,
            "description": definition.description,
            "input_schema": definition.input_schema,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def serialized_definitions(definitions: list[ToolDefinition]) -> str:
    if not definitions:
        return ""
    return "[" + ",".join(serialized_definition(item) for item in definitions) + "]"


class ToolSurfaceConfig(BaseModel):
    manifest_max_tokens: int = Field(default=2000, ge=0)
    search_default_limit: int = Field(default=5, ge=1, le=20)
    search_max_limit: int = Field(default=20, ge=1, le=20)
    mode: Literal["progressive", "eager"] = "progressive"


class ToolSchemaStats(BaseModel):
    name: str
    exposure: ToolExposure
    schema_chars: int
    description_chars: int
    parameter_schema_chars: int
    estimated_schema_tokens: int


class ToolSurfaceStats(BaseModel):
    registered_tools: int
    direct_tools: int
    deferred_tools: int
    hidden_tools: int
    direct_schema_chars: int
    deferred_schema_chars: int
    bridge_schema_chars: int
    manifest_chars: int
    visible_schema_chars: int
    estimated_direct_schema_tokens: int
    estimated_deferred_schema_tokens: int
    estimated_bridge_schema_tokens: int
    estimated_manifest_tokens: int
    estimated_visible_schema_tokens: int
    estimated_avoided_tokens: int
    schemas: list[ToolSchemaStats] = Field(default_factory=list)


@dataclass(frozen=True)
class ToolCatalogEntry:
    name: str
    short_description: str
    full_description: str
    namespace: str
    source: str
    tags: tuple[str, ...]
    parameter_names: tuple[str, ...]
    full_schema_reference: ToolDefinition

    @property
    def search_text(self) -> str:
        return " ".join(
            (
                self.name,
                self.short_description,
                self.full_description,
                self.namespace,
                self.source,
                *self.tags,
                *self.parameter_names,
            )
        )

    @property
    def required_parameters(self) -> list[str]:
        required = self.full_schema_reference.input_schema.get("required", [])
        return [str(item) for item in required] if isinstance(required, list) else []


@dataclass(frozen=True)
class ToolSearchMatch:
    entry: ToolCatalogEntry
    score: float


class ToolCatalog:
    """A zero-copy catalog over deferred definitions with local BM25 search."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def entries(self) -> list[ToolCatalogEntry]:
        return [self._entry(tool) for tool in self._registry.tools_for_exposure(ToolExposure.DEFERRED)]

    def get(self, name: str) -> ToolCatalogEntry | None:
        return next((entry for entry in self.entries() if entry.name == name), None)

    def search(
        self, query: str, *, limit: int = 5, namespace: str | None = None
    ) -> list[ToolSearchMatch]:
        entries = [
            entry
            for entry in self.entries()
            if namespace is None or entry.namespace == namespace
        ]
        if not entries or not query.strip():
            return []
        query_terms = _terms(query)
        documents = [_terms(entry.search_text) for entry in entries]
        average_length = sum(len(document) for document in documents) / len(documents)
        document_frequency = Counter(
            term for document in documents for term in set(document)
        )
        scored: list[ToolSearchMatch] = []
        for entry, document in zip(entries, documents, strict=True):
            frequencies = Counter(document)
            score = 0.0
            for term in query_terms:
                frequency = frequencies[term]
                if not frequency:
                    continue
                inverse_frequency = math.log(
                    1 + (len(documents) - document_frequency[term] + 0.5)
                    / (document_frequency[term] + 0.5)
                )
                denominator = frequency + 1.5 * (
                    1 - 0.75 + 0.75 * len(document) / max(average_length, 1)
                )
                score += inverse_frequency * frequency * 2.5 / denominator
            if score > 0:
                scored.append(ToolSearchMatch(entry, score))
        if not scored:
            lowered = query.casefold()
            scored = [
                ToolSearchMatch(entry, 0.1)
                for entry in entries
                if lowered in entry.search_text.casefold()
                or any(term in entry.search_text.casefold() for term in query_terms)
            ]
        scored.sort(key=lambda match: (-match.score, match.entry.name))
        return scored[: min(20, max(1, limit))]

    def _entry(self, tool: Tool) -> ToolCatalogEntry:
        definition = self._registry.definition(tool.name)
        properties = definition.input_schema.get("properties", {})
        parameter_names = tuple(properties) if isinstance(properties, dict) else ()
        short = tool.short_description.strip() or _first_sentence(tool.description)
        return ToolCatalogEntry(
            name=tool.name,
            short_description=short,
            full_description=tool.description,
            namespace=tool.namespace,
            source=tool.source,
            tags=tool.tags,
            parameter_names=parameter_names,
            full_schema_reference=definition,
        )


class CapabilityManifest(BaseModel):
    text: str = ""
    chars: int = 0
    estimated_tokens: int = 0
    format: Literal["empty", "full", "names", "namespaces", "truncated"] = "empty"


class ModelVisibleTools(BaseModel):
    definitions: list[ToolDefinition]
    manifest: CapabilityManifest
    stats: ToolSurfaceStats


class ToolSurfacePlanner:
    """Turn the registered universe into one stable model-visible surface."""

    def __init__(
        self, registry: ToolRegistry, config: ToolSurfaceConfig | None = None
    ) -> None:
        self.registry = registry
        self.config = config or ToolSurfaceConfig()
        self.catalog = ToolCatalog(registry)

    def plan(self) -> ModelVisibleTools:
        direct = self.registry.definitions(ToolExposure.DIRECT)
        deferred = self.registry.definitions(ToolExposure.DEFERRED)
        hidden = self.registry.definitions(ToolExposure.HIDDEN)
        visible = (
            direct
            if self.config.mode == "progressive"
            else [
                definition
                for definition in [*direct, *deferred]
                if definition.name not in PROGRESSIVE_HELPER_TOOL_NAMES
            ]
        )
        manifest = (
            build_manifest(self.catalog.entries(), self.config.manifest_max_tokens)
            if self.config.mode == "progressive"
            else CapabilityManifest()
        )
        direct_chars = len(serialized_definitions(direct))
        deferred_chars = len(serialized_definitions(deferred))
        visible_chars = len(serialized_definitions(visible))
        bridge = [definition for definition in direct if definition.name in BRIDGE_TOOL_NAMES]
        bridge_chars = len(serialized_definitions(bridge)) if bridge else 0
        schema_stats = [_schema_stats(self.registry, tool) for tool in self.registry.list_tools()]
        stats = ToolSurfaceStats(
            registered_tools=len(direct) + len(deferred) + len(hidden),
            direct_tools=len(direct),
            deferred_tools=len(deferred),
            hidden_tools=len(hidden),
            direct_schema_chars=direct_chars,
            deferred_schema_chars=deferred_chars,
            bridge_schema_chars=bridge_chars,
            manifest_chars=manifest.chars,
            visible_schema_chars=visible_chars,
            estimated_direct_schema_tokens=estimate_tokens(direct_chars),
            estimated_deferred_schema_tokens=estimate_tokens(deferred_chars),
            estimated_bridge_schema_tokens=estimate_tokens(bridge_chars),
            estimated_manifest_tokens=manifest.estimated_tokens,
            estimated_visible_schema_tokens=estimate_tokens(visible_chars),
            estimated_avoided_tokens=(
                estimate_tokens(deferred_chars) if self.config.mode == "progressive" else 0
            ),
            schemas=schema_stats,
        )
        return ModelVisibleTools(definitions=visible, manifest=manifest, stats=stats)


def build_manifest(
    entries: list[ToolCatalogEntry], max_tokens: int
) -> CapabilityManifest:
    if not entries or max_tokens <= 0:
        return CapabilityManifest()
    budget = max_tokens * 4
    intro = (
        "Deferred capabilities are available. Use tool_search to discover them, "
        "tool_describe for calling details, and tool_call to invoke one.\n"
    )
    full = intro + "\n".join(
        f"{entry.name} — {entry.short_description}" for entry in entries
    )
    if len(entries) <= 12 and len(full) <= budget:
        return _manifest(full, "full")
    grouped: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        grouped[entry.namespace].append(entry.name)
    names = intro + "\n".join(
        f"{namespace}: {', '.join(names)}"
        for namespace, names in sorted(grouped.items())
    )
    if len(names) <= budget:
        return _manifest(names, "names")
    namespaces = intro + "\n".join(
        f"{namespace} — {len(names)} tools"
        for namespace, names in sorted(grouped.items())
    )
    if len(namespaces) <= budget:
        return _manifest(namespaces, "namespaces")
    marker = "\n[capability manifest truncated; use tool_search]"
    delivered = namespaces[: max(0, budget - len(marker))] + marker
    return _manifest(delivered[:budget], "truncated")


def compact_parameter_help(definition: ToolDefinition) -> dict[str, object]:
    schema = definition.input_schema
    properties = schema.get("properties", {})
    raw_required = schema.get("required", [])
    required = raw_required if isinstance(raw_required, list) else []
    compact: dict[str, object] = {}
    if isinstance(properties, dict):
        for name, raw in properties.items():
            value = raw if isinstance(raw, dict) else {}
            help_item = {
                key: value[key]
                for key in ("type", "description", "enum", "minimum", "maximum")
                if key in value
            }
            help_item["required"] = name in required
            compact[str(name)] = help_item
    return compact


def _manifest(text: str, format_name: Literal["full", "names", "namespaces", "truncated"]) -> CapabilityManifest:
    return CapabilityManifest(
        text=text,
        chars=len(text),
        estimated_tokens=estimate_tokens(len(text)),
        format=format_name,
    )


def _schema_stats(registry: ToolRegistry, tool: Tool) -> ToolSchemaStats:
    definition = registry.definition(tool.name)
    serialized = serialized_definition(definition)
    description_chars = len(
        json.dumps(definition.description, ensure_ascii=False, separators=(",", ":"))
    )
    parameter_chars = len(
        json.dumps(
            definition.input_schema,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return ToolSchemaStats(
        name=tool.name,
        exposure=tool.exposure,
        schema_chars=len(serialized),
        description_chars=description_chars,
        parameter_schema_chars=parameter_chars,
        estimated_schema_tokens=estimate_tokens(len(serialized)),
    )


def _terms(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold().replace("_", " ").replace(".", " "))


def _first_sentence(description: str) -> str:
    line = " ".join(description.split())
    match = re.match(r"(.+?[.!?])(?:\s|$)", line)
    return (match.group(1) if match else line)[:240]
