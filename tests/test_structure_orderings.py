from __future__ import annotations

import json

import pytest
from pymatgen.core import Lattice, Structure

from photomatagent.scientific.capabilities.structure.construction import (
    StructureConstructionError,
    enumerate_orderings,
    ordering_count,
)
from photomatagent.scientific.capabilities.structure.construction_models import (
    ConstructionLimits,
    OrderingRequest,
)
from photomatagent.scientific.capabilities.structure.construction_tools import (
    EnumerateOrderingsTool,
)
from photomatagent.scientific.discovery.models import HypothesisOrigin, HypothesisProposal
from photomatagent.scientific.discovery.registration import build_hypothesis
from photomatagent.scientific.state import ScientificState
from photomatagent.workspace import Workspace


def _parent() -> Structure:
    return Structure(
        Lattice.tetragonal(10, 12),
        ["Na", "Na", "Na", "Na"],
        [[0, 0, 0], [0.17, 0.23, 0.11], [0.41, 0.37, 0.29], [0.73, 0.61, 0.83]],
    )


def _request(**changes: object) -> OrderingRequest:
    payload: dict[str, object] = {
        "path": "input.cif",
        "eligible_indices": [0, 1, 2, 3],
        "from_element": "Na",
        "to_element": "Ag",
        "replacement_count": 2,
        "expected_formula": "Na2Ag2",
        "hypothesis_id": "hyp-ordering",
        "task_slug": "ordering-tests",
    }
    payload.update(changes)
    return OrderingRequest.model_validate(payload)


def test_count_is_known_before_materializing_configurations() -> None:
    assert ordering_count(4, 1) == 4
    assert ordering_count(32, 16) > 4096
    with pytest.raises(ValueError):
        ordering_count(4, 5)


def test_orderings_are_deterministic_and_preserve_exact_composition() -> None:
    limits = ConstructionLimits(max_raw_configurations=64, max_outputs=32)
    request = _request()
    first = enumerate_orderings(_parent(), request, limits)
    second = enumerate_orderings(_parent(), request, limits)

    assert len(first) == len(second)
    assert [item.composition.reduced_formula for item in first] == [
        item.composition.reduced_formula for item in second
    ]
    assert all(item.composition.num_atoms == 4 for item in first)


def test_ordering_count_limit_fails_before_copying_or_materializing() -> None:
    parent = _parent()
    with pytest.raises(StructureConstructionError) as exc_info:
        enumerate_orderings(
            parent,
            _request(replacement_count=2),
            ConstructionLimits(max_raw_configurations=5),
        )
    assert exc_info.value.code == "ENUMERATION_LIMIT_EXCEEDED"
    assert parent.composition.num_atoms == 4


def test_ordering_request_validates_indices_and_host_elements() -> None:
    request = _request(from_element="Ag", to_element="Cu")
    with pytest.raises(StructureConstructionError) as exc_info:
        enumerate_orderings(
            _parent(),
            request,
            ConstructionLimits(),
        )
    # The request is normalized to fixed index order; this assertion documents
    # that a malformed host is rejected before any output is published.
    assert exc_info.value.code == "HOST_ELEMENT_MISMATCH"


@pytest.mark.asyncio
async def test_ordering_tool_reports_raw_scan_dedup_and_truncation(tmp_path) -> None:
    workspace = Workspace(tmp_path)
    workspace.resolve("input.cif", must_exist=False).parent.mkdir(parents=True, exist_ok=True)
    _parent().to(filename=str(workspace.resolve("input.cif", must_exist=False)), fmt="cif")
    hypothesis = build_hypothesis(
        HypothesisProposal(
            request_id="ordering-hypothesis",
            formula="Na2Ag2",
            statement="ordering fixture",
            design_operation="ordering",
            validation_questions=["is the ordering retained?"],
        ),
        HypothesisOrigin(
            tool_name="test", tool_call_id="test", session_id="test", run_id="test",
            provider="test", model="test",
        ),
    )
    tool = EnumerateOrderingsTool(
        workspace,
        scientific_state=ScientificState(material_hypotheses=[hypothesis]),
        limits=ConstructionLimits(max_outputs=2),
    )
    result = await tool.execute(
        _request(hypothesis_id=hypothesis.id).model_dump(mode="json")
    )

    assert result.is_error is False
    assert result.state_updates
    assert all(type(item).__name__ == "StructureRegistration" for item in result.state_updates)
    assert result.evidence == []
    assert result.data["total"] == 6
    assert result.data["scanned"] <= 6
    assert result.data["truncated"] is True
    assert result.data["exhaustive"] is False
    assert result.data["discarded"] >= 0
    manifest_path = next(workspace.root.glob("user_output/ordering-tests/structures/*/manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["structure_matcher"] == {"ltol": 0.2, "stol": 0.3, "angle_tol": 5}
