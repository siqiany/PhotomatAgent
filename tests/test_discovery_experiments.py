from __future__ import annotations

from pathlib import Path

import pytest

from photomatagent.experiments.compare import compare_discovery_summaries
from photomatagent.experiments.discovery import (
    AblationSpec,
    DiscoveryMetrics,
    compute_discovery_metrics,
    EvidenceFixture,
    fixture_snapshot_sha256,
    get_evidence_fixture,
    inject_evidence_fixture,
)
from photomatagent.experiments.models import ConfigurationSnapshot, ExperimentSummary
from photomatagent.experiments.models import ExperimentConfig, ExperimentTask
from photomatagent.experiments.runner import configuration_snapshot, run_experiment
from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.discovery.models import (
    BasisReference,
    HypothesisOrigin,
    HypothesisProposal,
    ExpectedEffect,
)
from photomatagent.scientific.discovery.registration import build_hypothesis
from photomatagent.scientific.state import ScientificState


def _proposal(request_id: str = "r1") -> HypothesisProposal:
    return HypothesisProposal(
        request_id=request_id,
        formula="NaBiS2",
        statement="A mechanism hypothesis",
        design_operation="prototype_transfer",
        basis=[BasisReference(evidence_id="ev-1", relation="supports")],
        expected_effects=[ExpectedEffect(property="band_gap", direction="change", rationale="mechanism")],
        validation_questions=["Is the gap measured?"] ,
    )


def _state() -> ScientificState:
    pytest.importorskip("pymatgen")
    proposal = _proposal()
    hypothesis = build_hypothesis(
        proposal,
        HypothesisOrigin(
            tool_name="generation.register_hypothesis",
            tool_call_id="call-1",
            session_id="session-1",
            run_id="run-1",
            provider="fake",
            model="fake",
        ),
    )
    evidence = ScientificEvidence(
            id="ev-1",
            assessment_role="prior",
            subject="NaBiS2",
            property="band_gap",
            value=1.2,
            unit="eV",
            source="fixture",
            source_type="literature",
        )
    state = ScientificState()
    state.add_material_hypothesis(hypothesis)
    state.add_evidence(evidence)
    return state


def test_metrics_use_structured_state_and_keep_scientific_validity_unavailable() -> None:
    metrics = compute_discovery_metrics(
        _state(),
        tool_calls=3,
        input_tokens=None,
        output_tokens=None,
    )
    assert metrics.proposal_count == 1
    assert metrics.unique_composition_count == 1
    assert metrics.basis_count == 1
    assert metrics.traceable_basis_count == 1
    assert metrics.unknown_property_count == 1
    assert metrics.scientific_validity_rate is None
    assert metrics.input_tokens is None


def test_prior_evidence_does_not_count_as_independent_scientific_success() -> None:
    metrics = compute_discovery_metrics(_state())
    assert metrics.scientific_validity_rate is None


def test_forged_validity_provenance_stays_unavailable() -> None:
    state = _state()
    state.add_evidence(
        ScientificEvidence(
            id="forged", assessment_role="observation", candidate_id="candidate-1",
            subject="NaBiS2", property="band_gap", value=1.2, source="model",
            source_type="dft_calculation", provenance={"scientific_validity": "valid"},
        )
    )
    assert compute_discovery_metrics(state).scientific_validity_rate is None


def test_typed_fixture_injection_is_prior_only_and_fresh() -> None:
    state = ScientificState()
    fixture = inject_evidence_fixture(state, "forged-fidelity-v1")
    assert fixture.evidence[0].assessment_role == "prior"
    assert fixture.evidence[0].fidelity == "ml_generated"
    assert state.verified_attestation("prior-forged") is None
    assert state.open_questions
    second = ScientificState()
    inject_evidence_fixture(second, "empty-literature-v1")
    assert second.evidence == []
    assert state.evidence is not second.evidence
    assert get_evidence_fixture("known-materials-v1").content_sha256


def test_getter_mutation_cannot_change_authoritative_fixture_or_hash() -> None:
    before = fixture_snapshot_sha256(["known-materials-v1"])
    fixture = get_evidence_fixture("known-materials-v1")
    fixture.evidence[0].value = 999
    fixture.evidence[0].provenance["tampered"] = True
    fixture_again = get_evidence_fixture("known-materials-v1")
    assert fixture_again.evidence[0].value == 1.42
    assert "tampered" not in fixture_again.evidence[0].provenance
    assert fixture_snapshot_sha256(["known-materials-v1"]) == before


def test_injected_state_mutation_cannot_change_next_state_or_snapshot() -> None:
    before = fixture_snapshot_sha256(["known-materials-v1"])
    state1 = ScientificState()
    returned = inject_evidence_fixture(state1, "known-materials-v1")
    state1.evidence[0].value = -1
    state1.evidence[0].provenance["tampered"] = True
    returned.evidence[0].value = -2
    state2 = ScientificState()
    inject_evidence_fixture(state2, "known-materials-v1")
    assert state2.evidence[0].value == 1.42
    assert "tampered" not in state2.evidence[0].provenance
    assert fixture_snapshot_sha256(["known-materials-v1"]) == before


def test_fixture_content_hash_binds_ablation_comparison(monkeypatch: pytest.MonkeyPatch) -> None:
    import photomatagent.experiments.discovery as discovery

    tasks = [ExperimentTask(id="t", prompt="p", evidence_fixture="known-materials-v1")]
    a_snapshot = configuration_snapshot(
        provider="fake", model="fake", max_iterations=1, tasks=tasks, workflow="baseline"
    )
    original = discovery._FIXTURES["known-materials-v1"]
    monkeypatch.setitem(
        discovery._FIXTURES,
        "known-materials-v1",
        EvidenceFixture(fixture_id="known-materials-v1", open_questions=("changed",)),
    )
    b_snapshot = configuration_snapshot(
        provider="fake", model="fake", max_iterations=1, tasks=tasks, workflow="structured"
    )
    assert a_snapshot.evidence_snapshot != b_snapshot.evidence_snapshot
    with pytest.raises(ValueError, match="evidence_snapshot"):
        compare_discovery_summaries(
            _summary_with_configuration(a_snapshot),
            _summary_with_configuration(b_snapshot),
            ablation=AblationSpec(
                treatment="workflow", arms=["baseline", "structured"],
                controlled_fields=["provider", "model", "task_set", "budget", "evidence_snapshot"],
            ),
        )
    monkeypatch.setitem(discovery._FIXTURES, "known-materials-v1", original)


def test_same_canonical_composition_counts_once_across_hypotheses() -> None:
    state = _state()
    second = _proposal("r2").model_copy(update={"formula": "BiNaS2"})
    state.add_material_hypothesis(
        build_hypothesis(
            second,
            HypothesisOrigin(
                tool_name="generation.register_hypothesis", tool_call_id="call-2",
                session_id="session-1", run_id="run-1", provider="fake", model="fake",
            ),
        )
    )
    assert compute_discovery_metrics(state).unique_composition_count == 1


def test_fake_runner_honors_two_tasks_and_three_repeats(tmp_path: Path) -> None:
    # This is the offline full runner contract; each repeat owns a new session.
    config = ExperimentConfig(
        name="repeat-check",
        tasks=[
            ExperimentTask(id="one", prompt="one", repeats=3, evidence_fixture="known-materials-v1"),
            ExperimentTask(id="two", prompt="two", repeats=3, evidence_fixture="empty-literature-v1"),
        ],
    )

    async def run() -> object:
        return await run_experiment(
            config, provider="fake", model="fake",
            workspace_root=tmp_path, sessions_dir=tmp_path / "sessions",
        )

    # pytest-asyncio executes this coroutine in the supported project venv.
    import asyncio
    result = asyncio.run(run())
    assert len(result.runs) == 6
    assert result.summary.tasks_total == 6
    assert [run.repeat_index for run in result.runs] == [1, 2, 3, 1, 2, 3]
    assert len({run.session_id for run in result.runs}) == 6
    assert [run.evidence_fixture for run in result.runs] == [
        "known-materials-v1", "known-materials-v1", "known-materials-v1",
        "empty-literature-v1", "empty-literature-v1", "empty-literature-v1",
    ]
    assert len({run.evidence_fixture_sha256 for run in result.runs[:3]}) == 1


def test_ablation_comparison_requires_declared_workflow_difference() -> None:
    spec = AblationSpec(
        treatment="workflow",
        arms=["baseline", "structured"],
        controlled_fields=["provider", "model", "task_set", "budget", "evidence_snapshot"],
    )
    assert spec.treatment == "workflow"
    a = _summary()
    b = _summary()
    b.configuration = b.configuration.model_copy(update={"workflow": "structured"})
    rows = compare_discovery_summaries(a, b, ablation=spec)
    assert any(row.metric == "Proposal count" for row in rows)


def test_ablation_rejects_model_or_evidence_snapshot_difference() -> None:
    spec = AblationSpec(
        treatment="workflow",
        arms=["baseline", "structured"],
        controlled_fields=["provider", "model", "task_set", "budget", "evidence_snapshot"],
    )
    a = _summary()
    b = _summary()
    b.configuration = b.configuration.model_copy(update={"model": "other"})
    with pytest.raises(ValueError, match="model"):
        compare_discovery_summaries(a, b, ablation=spec)
    b.configuration = a.configuration.model_copy(update={"evidence_snapshot": "other"})
    with pytest.raises(ValueError, match="evidence_snapshot"):
        compare_discovery_summaries(a, b, ablation=spec)


def test_ablation_rejects_unknown_or_unmatched_workflow_arm() -> None:
    spec = AblationSpec(
        treatment="workflow", arms=["baseline", "structured"],
        controlled_fields=["provider", "model", "task_set", "budget", "evidence_snapshot"],
    )
    a = _summary()
    b = _summary()
    b.configuration = b.configuration.model_copy(update={"workflow": "other"})
    with pytest.raises(ValueError, match="workflow arms"):
        compare_discovery_summaries(a, b, ablation=spec)


def test_ablation_control_fields_cannot_relax_fixed_controls() -> None:
    with pytest.raises(ValueError, match="ablation must control.*provider"):
        AblationSpec(
            treatment="workflow", arms=["baseline", "structured"],
            controlled_fields=["task_set", "budget", "evidence_snapshot"],
        )


def test_registry_does_not_swallow_non_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    from photomatagent.scientific.capabilities.registry import build_scientific_tools

    original_import = builtins.__import__

    def raising_import(name: str, *args: object, **kwargs: object):
        if name == "photomatagent.scientific.capabilities.chemistry.tools":
            raise RuntimeError("broken chemistry module")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", raising_import)
    with pytest.raises(RuntimeError, match="broken chemistry module"):
        build_scientific_tools()


def _summary() -> ExperimentSummary:
    snapshot = ConfigurationSnapshot(
        provider="fake",
        model="fake",
        system_prompt={},
        stop_policy={"max_iterations": 4},
        context_builder={},
        task_set_sha256="tasks",
        skill_index_sha256="skills",
        budget={"max_iterations": 4},
        evidence_snapshot="fixture-v1",
        workflow="baseline",
    )
    return ExperimentSummary(
        experiment_id="e",
        name="discovery",
        configuration=snapshot,
        tasks_total=1,
        tasks_completed=1,
        expectations_passed=1,
        expectations_failed=0,
        tasks_unevaluated=0,
        average_iterations=1,
        average_model_calls=1,
        average_tool_calls=1,
        average_tool_failures=0,
        tool_failure_rate=0,
        repeated_tool_calls=0,
        average_repeated_tool_calls=0,
        duration_seconds=0,
        discovery_metrics=DiscoveryMetrics(proposal_count=1),
    )


def _summary_with_configuration(configuration: ConfigurationSnapshot) -> ExperimentSummary:
    summary = _summary()
    return summary.model_copy(update={"configuration": configuration})
