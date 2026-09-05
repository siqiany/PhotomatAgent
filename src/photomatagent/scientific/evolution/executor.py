"""Application orchestration for one persisted scientific evolution episode."""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from photomatagent.errors import ToolExecutionError
from photomatagent.logging.event_logger import EventLogger
from photomatagent.redaction import redact_text
from photomatagent.runtime.events import RuntimeEvent
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.scientific.evolution.artifacts import (
    EpisodeArtifactCollector,
    materialize_primary_result,
)
from photomatagent.scientific.evolution.comparison import evaluate_machine_acceptance
from photomatagent.scientific.evolution.models import (
    ArtifactRef,
    CostSnapshot,
    EpisodeRecord,
    EvolutionTask,
    RevisionPlan,
)
from photomatagent.scientific.evolution.revision import format_revision_instruction
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
        owner_token: str | None = None,
    ) -> EpisodeExecutionResult:
        self.service.reconcile(task.evolution_id)
        self._validate_execution(
            task=task,
            episode=episode,
            runtime=runtime,
            owner_token=owner_token,
        )
        runtime_session_id = self._runtime_session_id(runtime)
        self._validate_unused_session(
            runtime_session_id,
            evolution_id=task.evolution_id,
            version=episode.version,
        )
        event_log_path = self._event_log_path()
        persisted_revision = self._persisted_revision(task, episode, revision)
        instruction = self._instruction(task, episode, persisted_revision)
        collector = EpisodeArtifactCollector()
        started_at = time.monotonic()
        runtime_logs_events = self._runtime_uses_event_logger(runtime)
        try:
            running_result = self.service.mark_episode_running(
                task.evolution_id,
                episode.version,
                owner_token=owner_token,
                runtime_session_id=runtime_session_id,
                event_log_path=event_log_path,
            )
            running = running_result.entity
            controller = ScientificLoopController(
                target=episode.target_snapshot.to_target_spec(),
                runtime=runtime,
                config=config,
                judge=judge,
                event_sinks=[],
                session_id=runtime_session_id,
            )
            await self._publish(running_result, on_event)
            async for event in controller.run(goal=instruction):
                collector.observe(event)
                if self.event_logger is not None and (
                    not runtime_logs_events or event.run_id == controller.run_id
                ):
                    await self.event_logger.log(event)
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
            if persisted_revision is not None:
                completion = completion.model_copy(
                    update={
                        "acceptance_results": evaluate_machine_acceptance(
                            plan=persisted_revision,
                            episode=completion,
                            state=runtime.scientific_state,
                        )
                    }
                )
            completed_result = self.service.complete_episode(
                task.evolution_id,
                episode.version,
                result=completion,
                owner_token=owner_token,
            )
            comparison_result = None
            if persisted_revision is not None and episode.parent_version is not None:
                comparison_result = self.service.compare(
                    task.evolution_id,
                    episode.parent_version,
                    episode.version,
                )
            await self._publish(completed_result, on_event)
            if comparison_result is not None:
                await self._publish(comparison_result, on_event)
        except BaseException as exc:
            await self._reconcile_exception(
                task=task,
                episode=episode,
                error=exc,
                on_event=on_event,
                owner_token=owner_token,
            )
            raise
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
        owner_token: str | None,
    ) -> None:
        if runtime.workspace.root != self.store.workspace.root:
            raise ValueError("runtime and evolution store must use the same workspace")
        if runtime.conversation_state.messages:
            raise ValueError("evolution episodes require a fresh runtime conversation")
        if any(value != 0 for value in runtime.budget.snapshot().values()):
            raise ValueError("evolution episodes require a fresh runtime budget")
        if task.evolution_id != episode.evolution_id:
            raise ValueError("task and episode belong to different evolution tasks")
        stored_task = self.store.load_task(task.evolution_id)
        stored_episode = self.store.load_episode(task.evolution_id, episode.version)
        if stored_task.current_version != episode.version:
            raise ValueError("episode is not the task's current reserved version")
        if stored_episode != episode or stored_episode.status != "RESERVED":
            raise ValueError("executor requires the exact persisted RESERVED episode")
        if (
            stored_episode.owner_token is not None
            and stored_episode.owner_token != owner_token
        ):
            raise ValueError("executor owner token does not match the reserved episode")
        if task.goal != stored_task.goal or task.target != stored_task.target:
            raise ValueError("task does not match the persisted immutable task snapshot")

    def _runtime_session_id(self, runtime: AgentRuntime) -> str:
        runtime_id = runtime.session_id
        logger_id = self.event_logger.session_id if self.event_logger is not None else None
        if not isinstance(runtime_id, str) or not runtime_id:
            raise ValueError("runtime session ID is unavailable")
        if logger_id is not None and logger_id != runtime_id:
            raise ValueError("event logger and runtime session IDs do not match")
        return runtime_id

    def _event_log_path(self) -> str | None:
        if self.event_logger is None:
            return None
        try:
            path = self.store.workspace.resolve(
                str(self.event_logger.events_path),
                must_exist=True,
            )
        except (OSError, ValueError, ToolExecutionError) as exc:
            raise ValueError("event log path must resolve inside the workspace") from exc
        if not path.is_file():
            raise ValueError("event log path must be a regular workspace file")
        return self.store.workspace.relative(path)

    def _validate_unused_session(
        self,
        session_id: str,
        *,
        evolution_id: str,
        version: str,
    ) -> None:
        for existing_task in self.store.list_tasks():
            if existing_task.current_version is None:
                continue
            latest = int(existing_task.current_version[1:])
            for number in range(1, latest + 1):
                existing_version = f"v{number:03d}"
                existing = self.store.load_episode(
                    existing_task.evolution_id,
                    existing_version,
                )
                if (
                    existing.runtime_session_id == session_id
                    and (existing.evolution_id, existing.version)
                    != (evolution_id, version)
                ):
                    raise ValueError(
                        "runtime session is already attributed to another episode"
                    )

    def _persisted_revision(
        self,
        task: EvolutionTask,
        episode: EpisodeRecord,
        supplied: RevisionPlan | None,
    ) -> RevisionPlan | None:
        if episode.revision_plan_id is None:
            if supplied is not None:
                raise ValueError("initial episode must not receive a revision")
            return None
        persisted = self.store.load_revision(
            episode.evolution_id,
            episode.revision_plan_id,
        )
        if supplied is None or supplied != persisted:
            raise ValueError("supplied revision does not equal the persisted RevisionPlan")
        if (
            not persisted.confirmed
            or persisted.has_blocking_ambiguity
            or persisted.evolution_id != episode.evolution_id
            or persisted.source_version != episode.parent_version
            or persisted.feedback_id != episode.applied_feedback_id
        ):
            raise ValueError("persisted RevisionPlan does not match the reserved episode")
        if episode.strategy_id is None:
            raise ValueError("revised episode has no persisted strategy")
        strategy = self.store.load_strategy(episode.evolution_id, episode.strategy_id)
        from photomatagent.scientific.evolution.strategy import FixedStrategySelector

        canonical = FixedStrategySelector().select(task, persisted)
        if (
            strategy != canonical
            or strategy.arm != episode.strategy_arm
            or strategy.parameters.get("revision_id") != persisted.revision_id
        ):
            raise ValueError("persisted strategy does not match the reserved episode")
        return persisted

    def _runtime_uses_event_logger(self, runtime: AgentRuntime) -> bool:
        if self.event_logger is None:
            return False
        expected_path = self.event_logger.events_path.resolve()
        for sink in runtime._event_sinks:
            owner = getattr(sink, "__self__", None)
            if (
                isinstance(owner, EventLogger)
                and owner.events_path.resolve() == expected_path
            ):
                return True
        return False

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

    def _instruction(
        self,
        task: EvolutionTask,
        episode: EpisodeRecord,
        revision: RevisionPlan | None,
    ) -> str:
        if revision is None:
            return task.goal
        revision_text = format_revision_instruction(
            revision,
            strategy=episode.strategy_arm,
        )
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
        return redact_text(f"{type(exc).__name__}: {exc}")[:1000]

    async def _reconcile_exception(
        self,
        *,
        task: EvolutionTask,
        episode: EpisodeRecord,
        error: BaseException,
        on_event: EventSink | None,
        owner_token: str | None,
    ) -> None:
        recovery_error: BaseException | None = None
        for _ in range(2):
            try:
                persisted = self.store.load_episode(
                    task.evolution_id,
                    episode.version,
                )
                persisted_task = self.store.load_task(task.evolution_id)
                if persisted.status == "COMPLETED":
                    if persisted_task.status == "RUNNING":
                        reconciled = self.service.complete_episode(
                            task.evolution_id,
                            episode.version,
                            result=persisted,
                            owner_token=owner_token,
                        )
                        await self._publish(reconciled, on_event)
                    return
                if persisted.status == "FAILED":
                    if persisted_task.status == "RUNNING":
                        reconciled = self.service.fail_episode(
                            task.evolution_id,
                            episode.version,
                            persisted.error or self._bounded_error(error),
                            owner_token=owner_token,
                        )
                        await self._publish(reconciled, on_event)
                    return
                if persisted.status in {"RESERVED", "RUNNING"}:
                    failed = self.service.fail_episode(
                        task.evolution_id,
                        episode.version,
                        self._bounded_error(error),
                        owner_token=owner_token,
                    )
                    await self._publish(failed, on_event)
                    return
            except BaseException as exc:
                recovery_error = exc
                continue
        if recovery_error is not None:
            safe_recovery_error = redact_text(
                f"{type(recovery_error).__name__}: {recovery_error}"
            )
            error.add_note(
                "episode terminal-state reconciliation also failed: "
                f"{safe_recovery_error}"
            )


__all__ = [
    "EpisodeExecutionResult",
    "EventSink",
    "ScientificEpisodeExecutor",
]
