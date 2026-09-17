from __future__ import annotations

import asyncio
import json

import pytest

from photomatagent.scientific.capabilities.generation.hypotheses import (
    RegisterHypothesisTool,
)
from photomatagent.scientific.capabilities.generation.tools import (
    GenerationCapabilitiesTool,
    VAERetrieveTool,
)
from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.loop.candidate import candidate_from_formula
from photomatagent.scientific.loop.evaluation import ScientificEvaluator
from photomatagent.scientific.loop.target import ConstraintSpec, TargetSpec
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.mock_calculation import MockCalculationTool


def _proposal() -> dict[str, object]:
    return {
        "request_id": "mechanism-workflow-1",
        "formula": "NaAgBiS2",
        "statement": "等价取代可能改变局域配位环境",
        "design_operation": "isovalent_substitution",
        "validation_questions": ["需要结构与电子性质验证"],
    }


def test_mechanism_registration_is_separate_from_vae() -> None:
    state = ScientificState()
    result = asyncio.run(RegisterHypothesisTool(state).execute(_proposal()))

    assert not result.is_error
    assert result.data["validation_status"] == "UNVALIDATED_HYPOTHESIS"
    assert result.state_updates
    assert not any("vae" in str(value).lower() for value in result.data.values())


def test_generation_capabilities_keep_mechanism_discoverable_without_torch(monkeypatch) -> None:
    import photomatagent.scientific.capabilities.generation.tools as generation_tools

    monkeypatch.setattr(generation_tools.importlib.util, "find_spec", lambda name: None)
    result = asyncio.run(GenerationCapabilitiesTool().execute({}))
    payload = json.loads(result.output)

    assert payload["mechanism_reasoning"]["status"] == "AVAILABLE"
    assert payload["vae_formula"]["status"] == "MISSING_DEPENDENCY"


def test_skills_keep_explicit_vae_and_known_retrieval_routes_distinct() -> None:
    from photomatagent.skills.loader import SkillLoader

    loader = SkillLoader()
    mechanism, _ = loader.view("mechanism-guided-material-design")
    composition, _ = loader.view("material-composition-generation")

    assert "generation.register_hypothesis" in mechanism
    assert "generation.vae_formula" not in mechanism
    assert "generation.vae_formula" in composition
    assert "generation.vae_retrieve" in composition


def test_known_material_retrieval_is_labeled_as_prior_not_new_generation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = tmp_path / "known.json"
    index.write_text(
        json.dumps([{"formula": "NaBiS2", "source": "known-record"}]),
        encoding="utf-8",
    )
    monkeypatch.setenv("VAE_INDEX_PATH", str(index))

    result = asyncio.run(VAERetrieveTool().execute({"formula": "NaBiS2"}))

    assert not result.is_error
    assert result.data["count"] == 1
    assert result.data["matches"][0]["source"] == "known-record"
    assert result.evidence == []
    assert result.state_updates == []
    assert "retrieval" in result.output.lower()
    assert "UNVALIDATED_GENERATED_STRUCTURE" not in result.output


@pytest.mark.parametrize(
    ("source", "source_type", "fidelity"),
    [
        ("mock", "calculation", "dft"),
        ("mechanism proposal", "generative_model", "ml_generated"),
    ],
)
def test_mock_or_unvalidated_results_cannot_make_evaluator_pass(
    source: str, source_type: str, fidelity: str
) -> None:
    evidence = ScientificEvidence(
        subject="NaBiS2",
        property="band_gap",
        value=0.1,
        unit="eV",
        source=source,
        source_type=source_type,
        method="untrusted fixture",
        fidelity=fidelity,
    )
    target = TargetSpec(
        goal="validate band gap",
        constraints=[
            ConstraintSpec(property="band_gap", operator="le", value=1.0, unit="eV")
        ],
    )

    report = ScientificEvaluator(target).evaluate(
        candidate_from_formula("NaBiS2"), ScientificState(evidence=[evidence])
    )

    assert report.constraint_results[0].result == "UNKNOWN"
    assert report.verdict == "INCONCLUSIVE"


def test_test_only_mock_tool_result_cannot_become_validated_material(
) -> None:
    result = asyncio.run(
        MockCalculationTool().execute(
            {"material": "GaAs", "calculation_type": "band_structure"}
        )
    )
    state = ScientificState()
    for update in result.state_updates:
        if hasattr(update, "content"):
            state.add_evidence(update)

    report = ScientificEvaluator(
        TargetSpec(
            goal="validate band gap",
            constraints=[
                ConstraintSpec(property="band_gap", operator="le", value=1.0, unit="eV")
            ],
        )
    ).evaluate(candidate_from_formula("GaAs"), state)

    assert report.constraint_results[0].result == "UNKNOWN"
    assert report.verdict == "INCONCLUSIVE"
