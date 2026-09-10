"""Safe promotion of a completed runtime session into an evolution task."""

from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from photomatagent.observability.trace import AgentExecutionTrace, load_trace
from photomatagent.runtime.events import LoopStarted, TextDelta
from photomatagent.models.types import AssistantMessage
from photomatagent.scientific.evolution.artifacts import sha256_file
from photomatagent.scientific.evolution.models import (
    ArtifactRef,
    EvolutionTask,
)
from photomatagent.scientific.evolution.service import (
    EvolutionOperationConflict,
    EvolutionService,
)
from photomatagent.scientific.evolution.store import EvolutionAlreadyExistsError
from photomatagent.scientific.loop.target import TargetSpec
from photomatagent.scientific.state import ScientificState
from photomatagent.sessions.store import SessionSnapshot, load_session_snapshot
from photomatagent.workspace import Workspace


@dataclass(frozen=True, slots=True)
class HistoricalSessionPreview:
    session_id: str
    goal: str
    final_response: str
    event_log_path: str
    snapshot: SessionSnapshot | None = None


def _session_path(workspace: Workspace, source: str | Path) -> Path:
    candidate = workspace.resolve(str(source), must_exist=True)
    sessions_root = workspace.resolve(".photomatagent/sessions", must_exist=False)
    if candidate != sessions_root and sessions_root not in candidate.parents:
        raise ValueError("historical sessions must be inside .photomatagent/sessions")
    if not (candidate / "events.jsonl").is_file():
        raise ValueError("historical session has no events.jsonl")
    return candidate


def _extract(workspace: Workspace, trace: AgentExecutionTrace, snapshot: SessionSnapshot | None) -> HistoricalSessionPreview:
    goals = [event.goal for event in trace.events if isinstance(event, LoopStarted) and event.goal.strip()]
    goal = goals[-1] if goals else (snapshot.scientific.goal if snapshot else "")
    if not goal:
        raise ValueError("historical session has no recorded goal")
    run_id = next((event.run_id for event in reversed(trace.events) if isinstance(event, TextDelta) and event.text.strip()), None)
    pieces = [event.text for event in trace.events if isinstance(event, TextDelta) and event.text.strip() and (run_id is None or event.run_id == run_id)]
    if snapshot is not None:
        assistants = [message.text for message in snapshot.conversation.messages if isinstance(message, AssistantMessage) and message.text.strip()]
        if assistants:
            pieces = [assistants[-1]]
    final_response = "".join(pieces).strip()
    if not final_response:
        raise ValueError("historical session has no final assistant response")
    return HistoricalSessionPreview(
        session_id=trace.session_id,
        goal=goal,
        final_response=final_response,
        event_log_path=workspace.relative(trace.events_path),
        snapshot=snapshot,
    )


def preview_historical_session(
    workspace: Workspace,
    source: str | Path,
    *,
    snapshot: SessionSnapshot | None = None,
) -> HistoricalSessionPreview:
    session_dir = _session_path(workspace, source)
    trace = load_trace(session_dir, sessions_dir=session_dir.parent)
    if snapshot is None:
        try:
            snapshot = load_session_snapshot(session_dir)
        except FileNotFoundError:
            pass
    return _extract(workspace, trace, snapshot)


class HistoricalSessionImporter:
    def __init__(self, workspace: Workspace, service: EvolutionService) -> None:
        self.workspace = workspace
        self.service = service

    def import_session(
        self,
        source: str | Path,
        *,
        target: TargetSpec,
        goal: str | None = None,
        snapshot: SessionSnapshot | None = None,
        artifact_path: str | Path | None = None,
    ) -> EvolutionTask:
        if not target.constraints:
            raise ValueError("historical import requires at least one target constraint")
        preview = preview_historical_session(self.workspace, source, snapshot=snapshot)
        resolved_goal = goal or preview.goal
        if not resolved_goal.strip():
            raise ValueError("historical import requires a non-empty goal")
        evolution_id = "evo_import_" + hashlib.sha256(preview.session_id.encode()).hexdigest()[:32]
        input_sha = self.service._input_hash(resolved_goal, target)
        source_file = self.workspace.resolve(str(artifact_path), must_exist=True) if artifact_path else None
        expected_artifact_sha = sha256_file(source_file) if source_file else hashlib.sha256((preview.final_response + "\n").encode("utf-8")).hexdigest()
        try:
            task = self.service.create_task(
                goal=resolved_goal, target=target, evolution_id=evolution_id, input_sha256=input_sha
            ).entity
        except EvolutionAlreadyExistsError:
            task = self.service.get(evolution_id)
            if (
                task.goal != resolved_goal
                or task.input_sha256 != input_sha
                or task.target.model_dump(mode="json") != target.model_dump(mode="json")
            ):
                raise EvolutionOperationConflict("historical session import content conflicts with existing task")

        relative = f"user_output/{evolution_id}/v001/result.md"
        canonical = self.workspace.resolve(relative, must_exist=False)
        canonical.parent.mkdir(parents=True, exist_ok=True)
        if source_file is None:
            payload = (preview.final_response + "\n").encode("utf-8")
            self._materialize(canonical, payload, expected_artifact_sha)
        else:
            self._materialize_from(canonical, source_file, expected_artifact_sha)
        artifact = ArtifactRef(path=relative, media_type="text/markdown", size_bytes=canonical.stat().st_size, sha256=sha256_file(canonical))
        state = preview.snapshot.scientific if preview.snapshot is not None else ScientificState(goal=resolved_goal)
        owner = "import_owner_" + hashlib.sha256(preview.session_id.encode()).hexdigest()[:24]
        try:
            episode = self.service.store.load_episode(evolution_id, "v001")
        except FileNotFoundError:
            episode = self.service._reserve_imported_episode(evolution_id, owner_token=owner).entity
        if episode.execution_mode != "IMPORTED_SESSION" or episode.owner_token != owner:
            raise EvolutionOperationConflict("historical session provenance conflicts with existing import")
        if episode.status == "COMPLETED":
            if episode.artifact is None or episode.artifact.sha256 != expected_artifact_sha:
                raise EvolutionOperationConflict("historical session artifact hash conflicts")
            return task
        try:
            stored_state = self.service.store.load_scientific_state(evolution_id, "v001")
        except FileNotFoundError:
            self.service.store.write_scientific_state(evolution_id, "v001", state)
        else:
            if stored_state.model_dump(mode="json") != state.model_dump(mode="json"):
                raise EvolutionOperationConflict("historical scientific state conflicts")
        running = self.service.mark_episode_running(evolution_id, "v001", owner_token=owner, runtime_session_id=preview.session_id, event_log_path=preview.event_log_path).entity
        completed = running.model_copy(update={"artifact": artifact, "scientific_state_path": f".photomatagent/evolutions/{evolution_id}/episodes/v001.scientific.json"})
        return self.service.complete_episode(evolution_id, "v001", result=completed, owner_token=owner).entity and self.service.get(evolution_id)

    @staticmethod
    def _materialize(canonical: Path, payload: bytes, expected_sha: str) -> None:
        if canonical.is_file():
            if canonical.stat().st_size != len(payload) or sha256_file(canonical) != expected_sha:
                raise EvolutionOperationConflict("canonical historical artifact hash conflicts")
            return
        temporary = canonical.with_name(f".{canonical.name}.importing-{uuid.uuid4().hex}")
        try:
            with temporary.open("xb") as handle:
                handle.write(payload)
            if sha256_file(temporary) != expected_sha:
                raise EvolutionOperationConflict("temporary historical artifact hash mismatch")
            try:
                os.link(temporary, canonical)
                temporary.unlink()
            except FileExistsError:
                if not canonical.is_file() or sha256_file(canonical) != expected_sha:
                    raise EvolutionOperationConflict("canonical historical artifact hash conflicts")
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _materialize_from(canonical: Path, source: Path, expected_sha: str) -> None:
        if canonical.is_file():
            if sha256_file(canonical) != expected_sha:
                raise EvolutionOperationConflict("canonical historical artifact hash conflicts")
            return
        temporary = canonical.with_name(f".{canonical.name}.importing-{uuid.uuid4().hex}")
        try:
            with source.open("rb") as src, temporary.open("xb") as dst:
                shutil.copyfileobj(src, dst)
            if sha256_file(temporary) != expected_sha:
                raise EvolutionOperationConflict("temporary historical artifact hash mismatch")
            try:
                os.link(temporary, canonical)
                temporary.unlink()
            except FileExistsError:
                if not canonical.is_file() or sha256_file(canonical) != expected_sha:
                    raise EvolutionOperationConflict("canonical historical artifact hash conflicts")
        finally:
            temporary.unlink(missing_ok=True)

__all__ = ["HistoricalSessionImporter", "HistoricalSessionPreview", "preview_historical_session"]
