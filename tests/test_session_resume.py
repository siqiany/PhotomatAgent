from __future__ import annotations

import json

import pytest

from photomatagent.logging.event_logger import EventLogger
from photomatagent.models.fake import FakeModelProvider, FakeResponse, scripted_tool_call
from photomatagent.models.types import (
    AssistantMessage,
    ToolResultMessage,
    UserMessage,
)
from photomatagent.runtime.state import ConversationState
from photomatagent.runtime.context_engine import (
    CompactionState,
    ContextEngine,
    RelevantResource,
)
from photomatagent.scientific.evidence import Evidence
from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.state import EvidenceAttestation, ScientificState
from photomatagent.scientific.capabilities.generation.lineage import CandidateLineage
from photomatagent.scientific.discovery.structures import StructureDerivation
from photomatagent.sessions.store import (
    SESSION_STATE_FILENAME,
    SESSION_STATE_SCHEMA_VERSION,
    load_session_snapshot,
    save_session_snapshot,
    session_is_resumable,
)

from conftest import collect, make_runtime


def _hypothesis_arguments() -> dict[str, object]:
    return {
        "request_id": "resume-hypothesis-1",
        "formula": "Na0.75Ag0.25BiS2",
        "statement": "Preserve this mechanism hypothesis",
        "design_operation": "isovalent_substitution",
        "validation_questions": ["Does the ordered phase remain stable?"],
    }


def _structure_derivation(
    index: int, *, parent_candidate_id: str | None = None
) -> StructureDerivation:
    candidate_id = f"cand_structure_{index}"
    return StructureDerivation(
        id=f"der_structure_{index}",
        candidate_id=candidate_id,
        parent_candidate_id=parent_candidate_id,
        hypothesis_id="hypothesis-1",
        input_sha256="a" * 64,
        structure_hash=(str(index) * 64)[:64],
        output_path=(
            "user_output/resume-structures/structures/"
            f"op_{index:032x}/structure_0000.cif"
        ),
        operation="enumerate_orderings",
        parameters={"matcher": {"ltol": 0.2, "stol": 0.3, "angle_tol": 5}},
        normalized_composition=(("Ag", 1), ("Na", 1)),
        lineage=CandidateLineage(
            candidate_id=candidate_id,
            parent_candidate_id=parent_candidate_id,
            generated_by="structure_construction",
            transformation="enumerate_orderings",
        ).model_dump(mode="python"),
    )


def test_snapshot_roundtrip_preserves_all_resume_state(tmp_path):
    conversation = make_runtime(FakeModelProvider()).conversation_state
    conversation.add(UserMessage(content="compute GaAs"))
    scientific = ScientificState(goal="compute GaAs")
    scientific.add_evidence(
        Evidence(type="calculation", source="mock", content="E_g = 0.31 eV", confidence=0.7)
    )
    engine = ContextEngine()
    engine.restore(
        compaction_state=CompactionState(
            goal="goal",
            progress=["inspected"],
            relevant_resources=[RelevantResource(reference="a.py")],
        ).model_dump(mode="json"),
        compacted_message_count=3,
        compaction_count=1,
    )

    path = save_session_snapshot(
        tmp_path,
        conversation=conversation,
        scientific=scientific,
        engine=engine.snapshot(),
    )
    assert path.name == SESSION_STATE_FILENAME
    assert session_is_resumable(tmp_path)

    restored = load_session_snapshot(tmp_path)
    assert restored.conversation == conversation
    assert restored.scientific == scientific
    assert restored.engine is not None
    assert restored.engine.compacted_message_count == 3
    assert restored.engine.compaction_count == 1
    assert restored.engine.compaction_state is not None
    assert restored.engine.compaction_state.progress == ["inspected"]


@pytest.mark.asyncio
async def test_snapshot_roundtrip_preserves_structure_parent_child_lineage_and_inspect(tmp_path):
    parent = _structure_derivation(1)
    child = _structure_derivation(2, parent_candidate_id=parent.candidate_id)
    scientific = ScientificState(structure_derivations=[parent, child])
    path = save_session_snapshot(
        tmp_path,
        conversation=ConversationState(),
        scientific=scientific,
        engine=None,
    )
    restored = load_session_snapshot(tmp_path)
    assert path.name == SESSION_STATE_FILENAME
    assert [item.id for item in restored.scientific.structure_derivations] == [
        parent.id,
        child.id,
    ]
    assert restored.scientific.structure_derivations[1].parent_candidate_id == parent.candidate_id

    runtime = make_runtime(FakeModelProvider(), workspace=tmp_path)
    runtime.restore_session(restored)
    inspect = runtime._tools.get("scientific_state_inspect")
    result = await inspect.execute({"section": "structures", "limit": 10})
    assert result.data["total"] == 2
    assert result.data["items"][1]["parent_candidate_id"] == parent.candidate_id


def test_schema_v1_snapshot_loads_with_typed_authority_downgrade(tmp_path) -> None:
    evidence = ScientificEvidence(
        id="sev-v1",
        subject="HgTe",
        property="band_gap",
        value=0.1,
        unit="eV",
    )
    legacy = {
        "schema_version": 1,
        "conversation": {"messages": []},
        "scientific": {
            "evidence": [evidence.model_dump(mode="json")],
            "evidence_attestations": {
                evidence.id: EvidenceAttestation(
                    evidence_id=evidence.id,
                    authority="observation",
                    origin="trusted_builtin",
                    tool_name="electronic.band_summary",
                    tool_call_id="call-v1",
                ).model_dump(mode="json")
            },
        },
    }
    path = tmp_path / SESSION_STATE_FILENAME
    path.write_text(json.dumps(legacy), encoding="utf-8")

    restored = load_session_snapshot(tmp_path)

    assert restored.schema_version == SESSION_STATE_SCHEMA_VERSION
    assert restored.scientific.evidence[0].id == evidence.id
    assert restored.scientific.verified_attestation(evidence.id) is None
    assert [item.code for item in restored.migration_diagnostics] == [
        "EVIDENCE_AUTHORITY_DOWNGRADED"
    ]
    assert restored.migration_diagnostics[0].from_schema_version == 1


@pytest.mark.asyncio
async def test_restored_session_keeps_tool_state_and_continues(tmp_path):
    first_model = FakeModelProvider(
        [
            scripted_tool_call(
                "tool_call",
                {
                    "name": "mock.run_calculation",
                    "arguments": {
                        "material": "GaAs",
                        "calculation_type": "band_structure",
                    },
                },
                tool_call_id="resume-call-1",
            ),
            FakeResponse(text="gap is 0.31 eV"),
        ]
    )
    runtime = make_runtime(first_model, workspace=tmp_path)
    await collect(runtime, "compute GaAs band gap")
    assert len(runtime.scientific_state.evidence) == 1
    evidence_id = runtime.scientific_state.evidence[0].id
    assert runtime.scientific_state.evidence_attestations[evidence_id].authority == "background"

    save_session_snapshot(
        tmp_path / "session-state",
        conversation=runtime.conversation_state,
        scientific=runtime.scientific_state,
        engine=runtime.context_engine.snapshot(),
    )
    snapshot = load_session_snapshot(tmp_path / "session-state")
    assert [item.code for item in snapshot.migration_diagnostics] == [
        "EVIDENCE_AUTHORITY_DOWNGRADED"
    ]

    # A brand-new runtime (fresh scientific state) resumes the old session.
    second_runtime = make_runtime(
        FakeModelProvider([FakeResponse(text="continuing from previous session")]),
        workspace=tmp_path,
    )
    assert len(second_runtime.scientific_state.evidence) == 0
    second_runtime.restore_session(snapshot)
    assert len(second_runtime.scientific_state.evidence) == 1
    assert (
        second_runtime.scientific_state.evidence_attestations[evidence_id].authority
        == "background"
    )

    # Tools registered against the live scientific instance must see the
    # restored evidence (in-place mutation, not instance replacement).
    inspect = second_runtime._tools.get("scientific_state_inspect")
    result = await inspect.execute({"section": "all"})
    assert "Mock band_structure calculation for GaAs" in result.output

    # The restored conversation is preserved and a new turn continues on top.
    await collect(second_runtime, "follow-up question")
    assert any(
        isinstance(message, AssistantMessage)
        and message.text == "continuing from previous session"
        for message in second_runtime.conversation_state.messages
    )
    assert any(
        isinstance(message, AssistantMessage)
        and message.text == "gap is 0.31 eV"
        for message in second_runtime.conversation_state.messages
    )
    assert any(
        isinstance(message, UserMessage)
        and message.content == "compute GaAs band gap"
        for message in second_runtime.conversation_state.messages
    )
    assert any(
        isinstance(message, UserMessage) and message.content == "follow-up question"
        for message in second_runtime.conversation_state.messages
    )


@pytest.mark.asyncio
async def test_hypothesis_roundtrip_restores_in_place_and_retry_is_idempotent(tmp_path):
    arguments = _hypothesis_arguments()
    registrations = [
        {
            **arguments,
            "request_id": f"resume-hypothesis-{index}",
            "formula": f"Na{index}BiS2",
        }
        for index in range(1, 5)
    ]
    first_runtime = make_runtime(
        FakeModelProvider(
            [
                *[
                    scripted_tool_call(
                        "tool_call",
                        {
                            "name": "generation.register_hypothesis",
                            "arguments": registration,
                        },
                        tool_call_id=f"register-before-save-{index}",
                    )
                    for index, registration in enumerate(registrations, start=1)
                ],
                FakeResponse(text="saved"),
            ]
        ),
        workspace=tmp_path,
    )
    await collect(first_runtime, "register before save")
    originals = list(first_runtime.scientific_state.material_hypotheses)
    assert len(originals) == 4
    save_session_snapshot(
        tmp_path / "hypothesis-session",
        conversation=first_runtime.conversation_state,
        scientific=first_runtime.scientific_state,
        engine=first_runtime.context_engine.snapshot(),
    )

    retry_model = FakeModelProvider(
        [
            scripted_tool_call(
                "tool_call",
                    {
                        "name": "generation.register_hypothesis",
                        "arguments": registrations[0],
                },
                tool_call_id="register-after-restore",
            ),
            FakeResponse(text="retried"),
        ]
    )
    resumed = make_runtime(retry_model, workspace=tmp_path)
    live_state = resumed.scientific_state
    inspect_tool = resumed._tools.get("scientific_state_inspect")
    resumed.restore_session(load_session_snapshot(tmp_path / "hypothesis-session"))

    assert resumed.scientific_state is live_state
    restored_records = resumed.scientific_state.material_hypotheses
    assert restored_records == originals
    assert [record.model_dump(mode="json") for record in restored_records] == [
        record.model_dump(mode="json") for record in originals
    ]
    for restored, original in zip(restored_records, originals, strict=True):
        assert restored.lineage.candidate_id == original.lineage.candidate_id
        assert restored.lineage.generated_by == "mechanism_reasoning"
        assert restored.lineage.validation_status == "UNVALIDATED_HYPOTHESIS"
    inspected = await inspect_tool.execute(
        {"section": "hypotheses", "offset": 0, "limit": 50}
    )
    for original in originals:
        assert original.id in inspected.output
        assert original.lineage.candidate_id in inspected.output
    events = await collect(resumed, "retry restored request")
    assert len(resumed.scientific_state.material_hypotheses) == 4
    assert not any(event.kind == "hypothesis_registered" for event in events)


def test_snapshot_redacts_secrets_before_write(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_RESUME_API_KEY", "sk-secret-value-to-redact-12345")
    conversation, scientific = make_runtime(FakeModelProvider()).conversation_state, ScientificState()
    conversation.add(UserMessage(content="leak"))
    conversation.add(
        ToolResultMessage(
            tool_call_id="id-1",
            tool_name="bash",
            content="TEST_RESUME_API_KEY=sk-secret-value-to-redact-12345",
        )
    )
    save_session_snapshot(
        tmp_path,
        conversation=conversation,
        scientific=scientific,
        engine=None,
    )
    raw = (tmp_path / SESSION_STATE_FILENAME).read_text(encoding="utf-8")
    assert "sk-secret-value-to-redact-12345" not in raw
    assert "[REDACTED]" in raw


def test_event_logger_accepts_explicit_session_id(tmp_path):
    logger = EventLogger(tmp_path, session_id="20260801T000000_abc123")
    assert logger.session_id == "20260801T000000_abc123"
    assert logger.session_dir == tmp_path / "20260801T000000_abc123"
    assert logger.events_path == logger.session_dir / "events.jsonl"
    assert logger.session_dir.is_dir()


@pytest.mark.asyncio
async def test_run_chat_resume_continues_into_the_same_session(tmp_path):
    from photomatagent.cli.chat import run_chat

    await run_chat(
        provider="fake",
        approval="auto",
        goal="first goal",
        log_events=True,
        sessions_dir=tmp_path,
    )
    only_session = next(path for path in sorted(tmp_path.iterdir()) if path.is_dir())
    assert (only_session / SESSION_STATE_FILENAME).is_file()
    prior_events = (only_session / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert any("first goal" in line for line in prior_events)

    await run_chat(
        provider="fake",
        approval="auto",
        goal="second goal",
        resume=only_session.name,
        log_events=True,
        sessions_dir=tmp_path,
    )
    # No second directory: the resumed turn appends to the same trace.
    assert [path for path in sorted(tmp_path.iterdir()) if path.is_dir()] == [only_session]
    after_events = (only_session / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(after_events) > len(prior_events)
    assert any("second goal" in line for line in after_events)
    snapshot = load_session_snapshot(only_session)
    user_goals = [
        message.content
        for message in snapshot.conversation.messages
        if isinstance(message, UserMessage)
    ]
    assert user_goals == ["first goal", "second goal"]


@pytest.mark.asyncio
async def test_run_chat_resume_displays_bounded_migration_diagnostic(tmp_path, capsys):
    from photomatagent.cli.chat import run_chat

    session_dir = tmp_path / "legacy_session"
    evidence = ScientificEvidence(id="sev-chat-resume", subject="HgTe", property="band_gap", value=0.2, unit="eV")
    session_dir.mkdir()
    (session_dir / SESSION_STATE_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "conversation": {"messages": []},
                "scientific": {
                    "evidence": [evidence.model_dump(mode="json")],
                    "evidence_attestations": {
                        evidence.id: EvidenceAttestation(
                            evidence_id=evidence.id,
                            authority="observation",
                            origin="trusted_builtin",
                            tool_name="materials.get_summary",
                            tool_call_id="SECRET_RAW_CHAT_TOKEN",
                        ).model_dump(mode="json")
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    await run_chat(
        provider="fake",
        approval="auto",
        goal="continue legacy session",
        resume=session_dir.name,
        log_events=False,
        sessions_dir=tmp_path,
    )

    output = capsys.readouterr().out
    assert "EVIDENCE_AUTHORITY_DOWNGRADED" in output
    assert "SECRET_RAW_CHAT_TOKEN" not in output
