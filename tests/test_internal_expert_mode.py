from __future__ import annotations

import hashlib
import json
from pathlib import Path

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
