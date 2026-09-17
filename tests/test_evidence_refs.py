from __future__ import annotations

import pytest

from photomatagent.scientific.evidence_refs import (
    is_canonical_opaque_evidence_ref,
    matches_evidence_ref,
    opaque_evidence_ref,
)


@pytest.mark.parametrize(
    "value",
    [
        "eref_SECRET_TOKEN",
        "eref_" + "A" * 20,
        "eref_" + "a" * 19,
        "eref_" + "a" * 21,
        "eref_",
    ],
)
def test_only_exact_lowercase_opaque_refs_are_canonical(value: str) -> None:
    assert is_canonical_opaque_evidence_ref(value) is False


def test_generated_opaque_ref_is_canonical_and_not_double_hashed() -> None:
    reference = opaque_evidence_ref("/private/SECRET_TOKEN")

    assert is_canonical_opaque_evidence_ref(reference) is True
    assert matches_evidence_ref(reference, "/private/SECRET_TOKEN") is True
    assert matches_evidence_ref("eref_SECRET_TOKEN", "/private/SECRET_TOKEN") is False
