"""Stable identities for evaluator-accepted scientific observations."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.evidence import Evidence

_HASH_KEYS = frozenset(
    {
        "artifact_hash",
        "artifact_sha",
        "artifact_sha256",
        "content_hash",
        "content_sha",
        "content_sha256",
        "sha256",
    }
)
_VOLATILE_KEYS = frozenset(
    {
        "artifact",
        "artifact_path",
        "created_at",
        "evidence_id",
        "filename",
        "file_name",
        "input_path",
        "output_name",
        "output_path",
        "path",
        "reason",
        "request_id",
        "timestamp",
        "tool_call_id",
    }
)
_NUMERIC_QUANTUM = 1e-3


def observation_key(
    evidence: Evidence | ScientificEvidence, property_name: str, outcome: str
) -> str:
    return (
        f"question:{property_name}|outcome:{outcome}|"
        f"observation:{stable_observation_identity(evidence)}"
    )


def stable_observation_identity(evidence: Evidence | ScientificEvidence) -> str:
    """Hash scientific content while ignoring producer-controlled volatility."""

    explicit = _hash_values(getattr(evidence, "provenance", {}))
    if isinstance(evidence, Evidence):
        try:
            content_payload = json.loads(evidence.content)
        except (TypeError, json.JSONDecodeError):
            content_payload = {}
        explicit.update(_hash_values(content_payload))
    if isinstance(evidence, ScientificEvidence):
        payload = {
            "property": evidence.property,
            "value": _stable_value(evidence.value),
            "unit": evidence.unit,
            "source_type": evidence.source_type,
            "fidelity": evidence.fidelity,
            "method": evidence.method,
            "structure_hash": evidence.structure_hash,
            "conditions": _stable_value(evidence.conditions),
        }
        if explicit:
            payload["hashes"] = sorted(explicit)
    else:
        payload = {
            "type": evidence.type,
            "content": _stable_content(evidence.content),
            "confidence": _stable_value(evidence.confidence),
            "provenance": _stable_value(evidence.provenance),
        }
        if explicit:
            payload["hashes"] = sorted(explicit)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def _hash_values(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in _HASH_KEYS and item not in (None, ""):
                found.add(f"{normalized}:{item}")
            found.update(_hash_values(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_hash_values(item))
    return found


def _stable_content(content: str) -> Any:
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return " ".join(content.split())
    return _stable_value(parsed)


def _stable_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return round(float(value) / _NUMERIC_QUANTUM) * _NUMERIC_QUANTUM
    if isinstance(value, dict):
        return {
            str(key): _stable_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            if str(key).casefold() not in _VOLATILE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    return str(value)
