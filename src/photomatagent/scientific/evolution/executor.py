"""Application orchestration for one persisted scientific evolution episode."""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from photomatagent.logging.event_logger import EventLogger
from photomatagent.runtime.events import RuntimeEvent
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.scientific.evolution.artifacts import (
    EpisodeArtifactCollector,
    materialize_primary_result,
)
from photomatagent.scientific.evolution.models import (
    ArtifactRef,
    CostSnapshot,
    EpisodeRecord,
    EvolutionTask,
    RevisionPlan,
)
from photomatagent.scientific.evolution.service import (
    EvolutionService,
    MutationResult,
)
from photomatagent.scientific.evolution.store import EvolutionStore
from photomatagent.scientific.loop import (
    ScientificJudge,
    ScientificLoopConfig,
    ScientificLoopController,
    ScientificLoopSummary,
)

EventSink = Callable[[RuntimeEvent], Awaitable[None] | None]
_REVISION_INSTRUCTION_MAX_CHARS = 12_000


@dataclass(frozen=True, slots=True)
class EpisodeExecutionResult:
    """The durable completion produced by one executor invocation."""

    episode: EpisodeRecord
    runtime_session_id: str
    scientific_state_path: str
    scientific_summary: ScientificLoopSummary
    artifact: ArtifactRef
    cost: CostSnapshot


class ScientificEpisodeExecutor:
    """Run one reserved episode through the authoritative scientific loop.

    This class owns lifecycle orchestration only. Tool execution remains wholly
    inside ``AgentRuntime`` through the ``ScientificLoopController`` it wraps.
    """

    def __init__(
        self,
        store: EvolutionStore,
        *,
        event_logger: EventLogger | None = None,
    ) -> None:
        self.store = store
        self.event_logger = event_logger
        self.service = EvolutionService(store)

    async def execute(
        self,
        *,
        task: EvolutionTask,
        episode: EpisodeRecord,
        runtime: AgentRuntime,
        config: ScientificLoopConfig,
        revision: RevisionPlan | None = None,
        judge: ScientificJudge | None = None,
        on_event: EventSink | None = None,
    ) -> EpisodeExecutionResult:
        self._validate_execution(task=task, episode=episode, runtime=runtime)
        instruction = self._instruction(task, episode, revision)
        runtime_session_id = self._runtime_session_id(runtime)
        event_log_path = self._event_log_path()
        collector = EpisodeArtifactCollector()
        started_at = time.monotonic()
        running_result = self.service.mark_episode_running(
            task.evolution_id,
            episode.version,
            runtime_session_id=runtime_session_id,
            event_log_path=event_log_path,
        )
        running = running_result.entity

        controller = ScientificLoopController(
            target=episode.target_snapshot.to_target_spec(),
            runtime=runtime,
            config=config,
            judge=judge,
            event_sinks=(
                [self.event_logger.log] if self.event_logger is not None else []
            ),
            session_id=runtime_session_id,
        )
        try:
            await self._publish(running_result, on_event)
            async for event in controller.run(goal=instruction):
                collector.observe(event)
                await self._deliver(on_event, event)
            if controller.summary is None:
                raise RuntimeError("scientific loop ended without a summary")
            artifact = materialize_primary_result(
                workspace=self.store.workspace,
                evolution_id=task.evolution_id,
                version=episode.version,
                conversation=runtime.conversation_state,
                collector=collector,
            )
            state_path = self.store.write_scientific_state(
                task.evolution_id,
                episode.version,
                runtime.scientific_state,
            )
            scientific_state_path = self.store.workspace.relative(state_path)
            cost = self._cost_snapshot(runtime, started_at)
            completion = running.model_copy(
                update={
                    "scientific_state_path": scientific_state_path,
                    "summary": controller.summary,
                    "artifact": artifact,
                    "cost": cost,
                }
            )
            completed_result = self.service.complete_episode(
                task.evolution_id,
                episode.version,
                result=completion,
            )
        except BaseException as exc:
            try:
                persisted = self.store.load_episode(
                    task.evolution_id,
                    episode.version,
                )
                if persisted.status in {"RESERVED", "RUNNING"}:
                    failure = self.service.fail_episode(
                        task.evolution_id,
                        episode.version,
                        self._bounded_error(exc),
                    )
                    await self._publish(failure, on_event)
            except BaseException as recovery_error:
                exc.add_note(
                    "episode failure reconciliation also failed: "
                    f"{type(recovery_error).__name__}: {recovery_error}"
                )
            raise
        await self._publish(completed_result, on_event)
        completed = completed_result.entity
        return EpisodeExecutionResult(
            episode=completed,
            runtime_session_id=runtime_session_id,
            scientific_state_path=scientific_state_path,
            scientific_summary=controller.summary,
            artifact=artifact,
            cost=cost,
        )

    def _validate_execution(
        self,
        *,
        task: EvolutionTask,
        episode: EpisodeRecord,
        runtime: AgentRuntime,
    ) -> None:
        if runtime.workspace.root != self.store.workspace.root:
            raise ValueError("runtime and evolution store must use the same workspace")
        if task.evolution_id != episode.evolution_id:
            raise ValueError("task and episode belong to different evolution tasks")
        stored_task = self.store.load_task(task.evolution_id)
        stored_episode = self.store.load_episode(task.evolution_id, episode.version)
        if stored_task.current_version != episode.version:
            raise ValueError("episode is not the task's current reserved version")
        if stored_episode != episode or stored_episode.status != "RESERVED":
            raise ValueError("executor requires the exact persisted RESERVED episode")
        if task.goal != stored_task.goal or task.target != stored_task.target:
            raise ValueError("task does not match the persisted immutable task snapshot")

    def _runtime_session_id(self, runtime: AgentRuntime) -> str:
        public = getattr(runtime, "session_id", None)
        private = getattr(runtime, "_session_id", None)
        runtime_id = public or private
        logger_id = self.event_logger.session_id if self.event_logger is not None else None
        if runtime_id is None:
            runtime_id = logger_id
        if not isinstance(runtime_id, str) or not runtime_id:
            raise ValueError("runtime session ID is unavailable")
        if logger_id is not None and logger_id != runtime_id:
            raise ValueError("event logger and runtime session IDs do not match")
        return runtime_id

    def _event_log_path(self) -> str | None:
        if self.event_logger is None:
            return None
        path = self.event_logger.events_path.resolve()
        if self.store.workspace.contains(path):
            return self.store.workspace.relative(path)
        return str(path)

    async def _publish(
        self,
        result: MutationResult[object],
        on_event: EventSink | None,
    ) -> None:
        for event in result.events:
            if self.event_logger is not None:
                await self.event_logger.log(event)
            await self._deliver(on_event, event)

    @staticmethod
    async def _deliver(sink: EventSink | None, event: RuntimeEvent) -> None:
        if sink is None:
            return
        pending = sink(event)
        if inspect.isawaitable(pending):
            await pending

    @staticmethod
    def _instruction(
        task: EvolutionTask,
        episode: EpisodeRecord,
        revision: RevisionPlan | None,
    ) -> str:
        if revision is None:
            if episode.revision_plan_id is not None:
                raise ValueError("reserved revised episode requires its RevisionPlan")
            return task.goal
        if (
            not revision.confirmed
            or revision.has_blocking_ambiguity
            or revision.evolution_id != task.evolution_id
            or revision.revision_id != episode.revision_plan_id
            or revision.source_version != episode.parent_version
        ):
            raise ValueError("revision does not match the reserved episode")
        payload = {
            "contract_changes": revision.contract_changes,
            "evidence_requirements": revision.evidence_requirements,
            "output_schema_requirements": revision.output_schema_requirements,
            "preserved_facts": revision.preserved_facts,
            "preserved_evidence_ids": revision.preserved_evidence_ids,
            "prohibited_repeats": revision.prohibited_repeats,
            "invalidated_conclusions": revision.invalidated_conclusions,
            "machine_acceptance_tests": revision.machine_acceptance_tests,
            "human_acceptance_tests": revision.human_acceptance_tests,
            "strategy_arm": revision.strategy_arm,
            "strategy_reason": revision.strategy_reason,
        }
        revision_text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return (
            f"{task.goal}\n\n"
            "--- Confirmed structured revision contract ---\n"
            f"{revision_text[:_REVISION_INSTRUCTION_MAX_CHARS]}"
        )

    @staticmethod
    def _cost_snapshot(runtime: AgentRuntime, started_at: float) -> CostSnapshot:
        budget = runtime.budget
        return CostSnapshot(
            input_tokens=budget.input_tokens,
            output_tokens=budget.output_tokens,
            tool_calls=budget.tool_calls,
            runtime_seconds=max(0.0, time.monotonic() - started_at),
            hpc_cost=budget.compute_cost,
        )

    @staticmethod
    def _bounded_error(exc: BaseException) -> str:
        return f"{type(exc).__name__}: {exc}"[:1000]


__all__ = [
    "EpisodeExecutionResult",
    "EventSink",
    "ScientificEpisodeExecutor",
]
