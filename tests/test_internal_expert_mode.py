from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest

from photomatagent.observability.trace import load_trace
from photomatagent.scientific.evolution.importer import (
    HistoricalSessionImporter,
    preview_historical_session,
)
from photomatagent.scientific.evolution.models import ExecutionMode
from photomatagent.scientific.evolution.service import EvolutionService
from photomatagent.scientific.evolution.store import EvolutionStore
from photomatagent.scientific.loop import TargetSpec
from photomatagent.scientific.state import ScientificState
from photomatagent.sessions.store import save_session_snapshot
from photomatagent.runtime.state import ConversationState
from photomatagent.models.types import AssistantMessage
from photomatagent.workspace import Workspace


def test_import_historical_session_materializes_provenance_bound_v001(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    session_dir = tmp_path / ".photomatagent" / "sessions" / "session-abc"
    session_dir.mkdir(parents=True)
    events = [
        {
            "kind": "loop_started",
            "session_id": "session-abc",
            "goal": "find a stable infrared absorber",
        },
        {
            "kind": "text_delta",
            "session_id": "session-abc",
            "iteration": 1,
            "text": "Historical final answer",
        },
    ]
    (session_dir / "events.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in events), encoding="utf-8"
    )
    original_state = ScientificState(
        goal="find a stable infrared absorber", open_questions=["verify stability"]
    )
    save_session_snapshot(
        session_dir,
        conversation=ConversationState(
            messages=[AssistantMessage(content="Historical final answer")]
        ),
        scientific=original_state,
    )
    target = TargetSpec(
        goal="find a stable infrared absorber",
        constraints=[{"property": "bandgap", "operator": "between", "value": [0.1, 0.5]}],
    )
    preview = preview_historical_session(workspace, session_dir)
    service = EvolutionService(EvolutionStore(workspace))
    task = HistoricalSessionImporter(workspace, service).import_session(
        session_dir, target=target
    )

    assert ExecutionMode.__args__[-1] == "IMPORTED_SESSION"
    assert task.current_version == "v001"
    episode = service.store.load_episode(task.evolution_id, "v001")
    assert episode.execution_mode == "IMPORTED_SESSION"
    assert episode.runtime_session_id == "session-abc"
    assert episode.summary is None
    assert episode.artifact is not None
    assert episode.artifact.sha256 == hashlib.sha256(
        b"Historical final answer\n"
    ).hexdigest()
    assert service.store.load_scientific_state(task.evolution_id, "v001").model_dump() == original_state.model_dump()
    assert preview.session_id == "session-abc"


def test_import_retries_after_reservation_interruption(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = Workspace(tmp_path)
    session_dir = tmp_path / ".photomatagent" / "sessions" / "session-retry"
    session_dir.mkdir(parents=True)
    (session_dir / "events.jsonl").write_text(
        json.dumps({"kind": "loop_started", "session_id": "session-retry", "goal": "goal"})
        + "\n"
        + json.dumps({"kind": "text_delta", "session_id": "session-retry", "iteration": 1, "text": "answer"})
        + "\n", encoding="utf-8"
    )
    target = TargetSpec(goal="goal", constraints=[{"property": "x", "operator": "ge", "value": 1}])
    service = EvolutionService(EvolutionStore(workspace))
    importer = HistoricalSessionImporter(workspace, service)
    original = service._reserve_imported_episode
    calls = 0

    def interrupt(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "_reserve_imported_episode", interrupt)
    with pytest.raises(RuntimeError, match="simulated"):
        importer.import_session(session_dir, target=target)
    task = importer.import_session(session_dir, target=target)
    assert task.current_version == "v001"
    assert service.store.load_episode(task.evolution_id, "v001").status == "COMPLETED"


def test_import_cleans_stale_partial_temporary_artifact(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    session_dir = tmp_path / ".photomatagent" / "sessions" / "session-temp"
    session_dir.mkdir(parents=True)
    (session_dir / "events.jsonl").write_text(
        json.dumps({"kind": "loop_started", "session_id": "session-temp", "goal": "goal"})
        + "\n" + json.dumps({"kind": "text_delta", "session_id": "session-temp", "iteration": 1, "text": "answer"}) + "\n",
        encoding="utf-8",
    )
    target = TargetSpec(goal="goal", constraints=[{"property": "x", "operator": "ge", "value": 1}])
    service = EvolutionService(EvolutionStore(workspace))
    evolution_id = "evo_import_" + hashlib.sha256(b"session-temp").hexdigest()[:32]
    result_dir = tmp_path / "user_output" / evolution_id / "v001"
    result_dir.mkdir(parents=True)
    (result_dir / ".result.md.importing-stale").write_bytes(b"partial")
    task = HistoricalSessionImporter(workspace, service).import_session(session_dir, target=target)
    assert task.current_version == "v001"
    assert not list(result_dir.glob(".result.md.importing-*"))
