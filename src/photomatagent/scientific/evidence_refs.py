"""Bounded public references for internal scientific evidence identities."""

from __future__ import annotations

import hashlib


def opaque_evidence_ref(evidence_id: str) -> str:
    digest = hashlib.sha256(evidence_id.encode("utf-8")).hexdigest()[:20]
    return f"eref_{digest}"


def matches_evidence_ref(reference: str, evidence_id: str) -> bool:
    return reference == evidence_id or reference == opaque_evidence_ref(evidence_id)


__all__ = ["matches_evidence_ref", "opaque_evidence_ref"]
