from __future__ import annotations

from pathlib import Path

import pytest

import photomatagent.tools.factory as tools_factory
from photomatagent.cli.chat import build_runtime
from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.evidence import Evidence
from photomatagent.scientific.evolution.evidence import (
    build_inherited_scientific_state,
)
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.scientific_state_inspect import ScientificStateInspectTool


def _scientific(
    evidence_id: str,
    *,
    subject: str = "InAs",
    source_type: str = "dft_calculation",
    fidelity: str = "dft",
) -> ScientificEvidence:
    return ScientificEvidence(
        id=evidence_id,
        subject=subject,
        property="band_gap",
        value=0.15,
        unit="eV",
        source="calculation/result.json",
        source_type=source_type,
        method="VASP PBE",
        fidelity=fidelity,
        summary="computed band gap",
        provenance={"run_id": "run_001", "validated": True},
    )


def test_only_validated_structured_subject_compatible_evidence_is_carried() -> None:
    previous = ScientificState(
        goal="same task",
        hypotheses=["unverified hypothesis must not cross episodes"],
        open_questions=["old question"],
        evidence=[
            _scientific("sev_dft"),
            _scientific("sev_structured_no_extra_provenance").model_copy(
                update={"provenance": {}}
            ),
            _scientific("sev_model", source_type="model"),
            _scientific(
                "sev_generated",
                source_type="generative_model",
                fidelity="ml_generated",
            ),
            _scientific("sev_ml", fidelity="ml_generated"),
            _scientific("sev_explicit_unvalidated").model_copy(
                update={"provenance": {"validated": False}}
            ),
            _scientific("sev_invalid"),
            _scientific("sev_other_subject", subject="HgCdTe"),
            Evidence(
                id="ev_validated",
                type="experiment",
                source="lab notebook",
                content="measured response",
                confidence=0.9,
                provenance={"validated": True, "subject": "InAs"},
            ),
            Evidence(
                id="ev_unvalidated",
                type="literature",
                source="draft note",
                content="unverified prose",
                confidence=0.8,
                provenance={"subject": "InAs"},
            ),
        ],
    )

    inherited, decisions = build_inherited_scientific_state(
        previous,
        source_episode="v001",
        invalidated_evidence_ids={"sev_invalid"},
        subject="InAs",
    )

    assert [item.id for item in inherited.evidence] == [
        "sev_dft",
        "sev_structured_no_extra_provenance",
        "ev_validated",
    ]
    assert inherited.goal == "same task"
    assert inherited.hypotheses == []
    assert inherited.claims == []
    assert inherited.calculations == []
    assert inherited.open_questions == []
    assert inherited.contradictions == []
    assert inherited.pending_tasks == []
    assert {item.evidence_id for item in decisions if not item.carried} == {
        "sev_model",
        "sev_generated",
        "sev_ml",
        "sev_explicit_unvalidated",
        "sev_invalid",
        "sev_other_subject",
        "ev_unvalidated",
    }
    for item in inherited.evidence:
        assert item.provenance["inherited_from_episode"] == "v001"
        assert item.provenance["inherited_at"].endswith("Z")
    assert previous.evidence[0].provenance == {
        "run_id": "run_001",
        "validated": True,
    }


def test_build_runtime_binds_one_exact_inherited_scientific_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inherited = ScientificState(goal="same task")
    monkeypatch.setattr(tools_factory, "build_scientific_tools", lambda *args: [])

    runtime, _ = build_runtime(
        provider="fake",
        workspace_root=tmp_path,
        approval="deny",
        scientific_state=inherited,
        log_events=False,
    )

    assert runtime.scientific_state is inherited
    inspect_tool = runtime._tools.get("scientific_state_inspect")
    assert isinstance(inspect_tool, ScientificStateInspectTool)
    assert inspect_tool._state is inherited


def test_evidence_carry_contract_is_exported_from_evolution_package() -> None:
    from photomatagent.scientific.evolution import (
        EvidenceCarryDecision,
        build_inherited_scientific_state as exported_build,
        select_carry_forward_evidence,
    )

    assert EvidenceCarryDecision.__name__ == "EvidenceCarryDecision"
    assert exported_build is build_inherited_scientific_state
    assert callable(select_carry_forward_evidence)
