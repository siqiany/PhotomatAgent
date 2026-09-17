"""Bounded public references for internal scientific evidence identities."""

from __future__ import annotations

import hashlib
import re

_CANONICAL_OPAQUE_REF = re.compile(r"^eref_[0-9a-f]{20}$")


def opaque_evidence_ref(evidence_id: str) -> str:
    digest = hashlib.sha256(evidence_id.encode("utf-8")).hexdigest()[:20]
    return f"eref_{digest}"


def is_canonical_opaque_evidence_ref(value: object) -> bool:
    """Return whether ``value`` is exactly one generated opaque reference."""

    return isinstance(value, str) and bool(_CANONICAL_OPAQUE_REF.fullmatch(value))


def matches_evidence_ref(reference: str, evidence_id: str) -> bool:
    return reference == evidence_id or (
        is_canonical_opaque_evidence_ref(reference)
        and reference == opaque_evidence_ref(evidence_id)
    )


__all__ = [
    "is_canonical_opaque_evidence_ref",
    "matches_evidence_ref",
    "opaque_evidence_ref",
]
