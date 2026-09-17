from __future__ import annotations

import pytest

from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.discovery import composition as composition_module
from photomatagent.scientific.discovery.models import HypothesisOrigin, HypothesisProposal
from photomatagent.scientific.discovery.registration import build_hypothesis
from photomatagent.scientific.loop.candidate import (
    candidate_from_formula,
    extract_candidate_from_state,
    extract_candidates_from_state,
    project_candidates_from_state,
)
from photomatagent.scientific.state import ScientificState


def _hypothesis(request_id: str, formula: str, statement: str):
    return build_hypothesis(
        HypothesisProposal(
            request_id=request_id,
            formula=formula,
            statement=statement,
            design_operation="isovalent_substitution",
            validation_questions=["Is the composition stable?"],
        ),
        HypothesisOrigin(
            tool_name="generation.register_hypothesis",
            tool_call_id=f"call-{request_id}",
            session_id="session",
            run_id="run",
            provider="fake",
            model="fake",
        ),
    )


def test_projects_all_hypotheses_and_merges_equivalent_compositions():
    first = _hypothesis("h1", "Na0.75Ag0.25BiS2", "alloy the cation site")
    second = _hypothesis("h2", "Na3AgBi4S8", "order the cation site")
    distinct = _hypothesis("h3", "NaBiS2", "retain the ternary end member")
    scientific = ScientificState(material_hypotheses=[first, second, distinct])

    candidates = extract_candidates_from_state(scientific, iteration=3)

    assert [candidate.candidate_id for candidate in candidates] == [
        first.candidate_id,
        distinct.candidate_id,
    ]
    merged = candidates[0]
    assert merged.representation == {
        "formula": "Na0.75Ag0.25BiS2",
        "composition": {"Ag": 1, "Bi": 4, "Na": 3, "S": 8},
        "hypothesis_ids": [first.id, second.id],
    }
    assert merged.created_iteration == 3
    assert "properties" not in merged.representation


def test_reduced_formula_candidates_have_the_same_fingerprint():
    decimal = candidate_from_formula("Na0.75Ag0.25BiS2")
    integer = candidate_from_formula("Na3AgBi4S8")

    assert decimal.fingerprint == integer.fingerprint


def test_mechanism_and_legacy_candidates_coexist_by_composition_identity():
    hypothesis = _hypothesis("h1", "NaBiS2", "mechanism proposal")
    evidence = ScientificEvidence(
        subject="HgTe",
        property="candidate_formula",
        value="HgTe",
        source="vae",
        source_type="generative_model",
        provenance={"tool": "generation.vae_formula"},
    )
    scientific = ScientificState(material_hypotheses=[hypothesis], evidence=[evidence])

    candidates = extract_candidates_from_state(scientific)

    assert [candidate.formula for candidate in candidates] == ["NaBiS2", "HgTe"]
    assert candidates[0].generation_method == "mechanism_reasoning"
    assert candidates[1].generation_method == "generation.vae_formula"


def test_legacy_candidate_with_equivalent_formula_does_not_duplicate_hypothesis():
    hypothesis = _hypothesis("h1", "Na3AgBi4S8", "mechanism proposal")
    first_evidence = ScientificEvidence(
        subject="Na0.75Ag0.25BiS2",
        property="candidate_formula",
        value="Na0.75Ag0.25BiS2",
        source="vae",
        source_type="generative_model",
        provenance={"tool": "generation.vae_formula", "seed": 7},
    )
    second_evidence = ScientificEvidence(
        subject="AgNa3Bi4S8",
        property="candidate_formula",
        value="AgNa3Bi4S8",
        source="legacy import",
        source_type="model",
        provenance={"tool": "legacy.import", "record": "row-2"},
    )
    scientific = ScientificState(
        material_hypotheses=[hypothesis],
        evidence=[first_evidence, second_evidence],
    )

    candidates = extract_candidates_from_state(scientific)

    assert len(candidates) == 1
    assert candidates[0].candidate_id == hypothesis.candidate_id
    assert candidates[0].representation["hypothesis_ids"] == [hypothesis.id]
    assert candidates[0].evidence_ids == [first_evidence.id, second_evidence.id]
    assert candidates[0].lineage is not None
    assert candidates[0].lineage.source_artifacts == [
        first_evidence.id,
        second_evidence.id,
    ]
    assert candidates[0].generation_parameters["legacy_sources"] == [
        {
            "evidence_id": first_evidence.id,
            "source": "vae",
            "source_type": "generative_model",
            "method": "",
            "provenance": {"tool": "generation.vae_formula", "seed": 7},
        },
        {
            "evidence_id": second_evidence.id,
            "source": "legacy import",
            "source_type": "model",
            "method": "",
            "provenance": {"tool": "legacy.import", "record": "row-2"},
        },
    ]
    assert scientific.evidence == [first_evidence, second_evidence]
    assert "properties" not in candidates[0].representation


def test_legacy_only_extraction_remains_compatible():
    evidence = ScientificEvidence(
        subject="HgTe",
        property="candidate_formula",
        value="HgTe",
        source="legacy",
        source_type="generative_model",
    )
    scientific = ScientificState(evidence=[evidence])

    legacy = extract_candidate_from_state(scientific, iteration=2)
    projected = extract_candidates_from_state(scientific, iteration=2)

    assert legacy is not None
    assert len(projected) == 1
    assert projected[0].formula == legacy.formula == "HgTe"
    assert projected[0].created_iteration == legacy.created_iteration == 2


def test_legacy_projection_keeps_latest_record_for_stable_identity():
    earlier = ScientificEvidence(
        subject="HgTe",
        property="candidate_formula",
        value="HgTe",
        source="vae",
        source_type="generative_model",
        provenance={"tool": "generation.first"},
    )
    latest = ScientificEvidence(
        subject="TeHg",
        property="candidate_formula",
        value="TeHg",
        source="vae",
        source_type="generative_model",
        provenance={"tool": "generation.latest"},
    )

    candidates = extract_candidates_from_state(
        ScientificState(evidence=[earlier, latest])
    )

    assert len(candidates) == 1
    assert candidates[0].generation_method == "generation.latest"
    assert candidates[0].evidence_ids == [earlier.id, latest.id]
    assert candidates[0].lineage is not None
    assert candidates[0].lineage.source_artifacts == [earlier.id, latest.id]
    assert [
        item["provenance"]["tool"]
        for item in candidates[0].generation_parameters["legacy_sources"]
    ] == ["generation.first", "generation.latest"]


def test_invalid_legacy_record_does_not_erase_valid_projections():
    hypothesis = _hypothesis("h1", "NaBiS2", "valid mechanism proposal")
    invalid = ScientificEvidence(
        subject="not-a-formula",
        property="candidate_formula",
        value="not-a-formula",
        source="legacy import",
        source_type="model",
    )
    valid = ScientificEvidence(
        subject="HgTe",
        property="candidate_formula",
        value="HgTe",
        source="vae",
        source_type="generative_model",
    )

    result = project_candidates_from_state(
        ScientificState(
            material_hypotheses=[hypothesis],
            evidence=[invalid, valid],
        )
    )

    assert [candidate.formula for candidate in result.candidates] == [
        "NaBiS2",
        "HgTe",
    ]
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].code == "INVALID_LEGACY_CANDIDATE"
    assert result.diagnostics[0].evidence_id == invalid.id


def test_registered_hypothesis_survives_missing_optional_composition_parser(
    monkeypatch: pytest.MonkeyPatch,
):
    hypothesis = _hypothesis("h1", "NaBiS2", "already normalized")
    legacy = ScientificEvidence(
        subject="HgTe",
        property="candidate_formula",
        value="HgTe",
        source="legacy",
        source_type="model",
    )

    def unavailable():
        raise composition_module.CompositionCapabilityError("pymatgen unavailable")

    monkeypatch.setattr(composition_module, "_lazy_pymatgen_types", unavailable)

    result = project_candidates_from_state(
        ScientificState(material_hypotheses=[hypothesis], evidence=[legacy])
    )

    assert [candidate.candidate_id for candidate in result.candidates] == [
        hypothesis.candidate_id
    ]
    assert result.candidates[0].representation["composition"] == {
        "Bi": 1,
        "Na": 1,
        "S": 2,
    }
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].code == "COMPOSITION_CAPABILITY_UNAVAILABLE"
