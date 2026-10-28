"""ScientificLoopController: the evidence-guided scientific outer loop.

Architecture (task constraints honored):
    * the existing :class:`AgentRuntime` stays the inner execution loop and is
      the only place tools execute -- the controller never calls a backend
      directly and never bypasses ``_handle_tool_call`` (permission system,
      HPC gating and approval handlers stay authoritative);
    * this controller *contains* an AgentRuntime (outer depends on inner, not
      the reverse);
    * the Maker is the AgentRuntime; the Checker is the ScientificEvaluator;
      feedback is injected into the next maker turn as an explicit research
      instruction appended to the conversation (never into the static system
      prompt).

Deterministic stop conditions: SUCCESS (all hard constraints + confidence),
STALLED (stagnation detector), INCONCLUSIVE (no candidate/evidence possible),
BUDGET_EXHAUSTED (round/candidate caps).
"""

from __future__ import annotations

import inspect
import hashlib
import json
import time
from collections.abc import AsyncIterator, Callable
from uuid import uuid4

from pydantic import BaseModel, Field

from photomatagent.runtime.events import (
    CandidateEvaluated,
    CandidateJudged,
    CandidateProposed,
    RuntimeEvent,
    ScientificFeedbackGenerated,
    ScientificLoopCompleted,
    ScientificLoopDecisionMade,
    ScientificLoopStarted,
    ScientificLoopStalled,
)
from photomatagent.runtime.loop import AgentRuntime, EventSink
from photomatagent.scientific.loop.candidate import (
    CandidateState,
    extract_candidate_from_state,
    project_candidates_from_state,
)
from photomatagent.scientific.loop.evaluation import (
    EvaluationReport,
    ScientificEvaluator,
)
from photomatagent.scientific.loop.feedback import (
    FeedbackSignal,
    build_feedback,
    format_feedback_for_model,
)
from photomatagent.scientific.loop.judge import (
    JudgeReport,
    ScientificJudge,
    build_judge_snapshot,
)
from photomatagent.scientific.loop.judge import JUDGE_PROMPT_VERSION, JUDGE_SCHEMA_VERSION
from photomatagent.scientific.loop.judge_policy import JudgeMode, JudgePolicy
from photomatagent.scientific.loop.policy import (
    ScientificLoopDecision,
    ScientificLoopPolicy,
    ScientificLoopState,
    ScientificLoopSummary,
)
from photomatagent.scientific.loop.progress import progress_from_evaluation
from photomatagent.scientific.loop.stagnation import StagnationDetector
from photomatagent.scientific.loop.target import TargetSpec
from photomatagent.scientific.state import ScientificState

CandidateExtractor = Callable[[ScientificState, int], CandidateState | None]


class ScientificLoopConfig(BaseModel):
    max_rounds: int = Field(default=6, ge=1)
    max_candidates: int = Field(default=12, ge=1)
    patience: int = Field(default=3, ge=1)
    epsilon: float = Field(default=1e-3, ge=0.0)
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    judge_min_quality: float = Field(default=0.6, ge=0.0, le=1.0)
    require_judge: bool = Field(default=False)
    # None preserves the legacy every-round behavior for callers with old configs.
    judge_mode: JudgeMode | None = None
    judge_max_calls: int | None = Field(default=None, ge=0)
    judge_requested: bool = False


def format_loop_state_snapshot(
    state: ScientificLoopState, judge: JudgeReport | None = None
) -> str:
    """Compact dynamic loop context rendered as a trailing instruction.

    Kept bounded: best candidate, current violations, gaps, previous
    strategies and the next priority. Dynamic state never enters the static
    system prompt (cache-friendly layout preserved).
    """
    lines = ["--- Scientific loop state ---"]
    if state.best_candidate_id:
        lines.append(f"Best candidate: {state.best_candidate_id} (score {state.best_score:.3f})")
    latest = state.evaluations[-1] if state.evaluations else None
    if latest:
        if latest.violations:
            lines.append("Current violations:")
            lines += [f"- {v.short()}" for v in latest.violations]
        if latest.critical_evidence_gaps:
            lines.append("Evidence gaps:")
            lines += [f"- {gap}" for gap in latest.critical_evidence_gaps]
    if judge is not None and judge.available and judge.significant_issues:
        lines.append("Judge concerns (advisory):")
        lines += [
            f"- [{'HIGH' if issue.severity == 'HIGH' else 'MEDIUM'}] "
            f"{issue.description}"
            for issue in judge.significant_issues
        ]
    if state.feedback_history:
        last = state.feedback_history[-1]
        if last.prohibited_repeats:
            lines.append("Previous failed strategies:")
            lines += [f"- repeated: {item}" for item in last.prohibited_repeats]
        actions = sorted(last.recommended_actions, key=lambda a: a.priority)
        if actions:
            lines.append("Next priority:")
            lines += [
                f"{a.priority}. [{a.action_type}] {a.description}" for a in actions[:3]
            ]
    lines.append(f"Investigation round: {state.round}")
    return "\n".join(lines)


class ScientificLoopController:
    """Outer loop: MAKER (AgentRuntime) -> candidate -> CHECKER -> policy -> feedback."""

    def __init__(
        self,
        *,
        target: TargetSpec,
        runtime: AgentRuntime,
        evaluator: ScientificEvaluator | None = None,
        policy: ScientificLoopPolicy | None = None,
        stagnation: StagnationDetector | None = None,
        config: ScientificLoopConfig | None = None,
        candidate_extractor: CandidateExtractor | None = None,
        judge: ScientificJudge | None = None,
        event_sinks: list[EventSink] | None = None,
        session_id: str | None = None,
    ) -> None:
        self.target = target
        self.runtime = runtime
        self.evaluator = evaluator or ScientificEvaluator(target)
        self.config = config or ScientificLoopConfig()
        self.policy = policy or ScientificLoopPolicy(
            judge_min_quality=self.config.judge_min_quality,
            require_judge=self.config.require_judge,
        )
        self.stagnation = stagnation or StagnationDetector(
            patience=self.config.patience, epsilon=self.config.epsilon
        )
        self.candidate_extractor = candidate_extractor or extract_candidate_from_state
        self.judge = judge
        self.judge_policy = self._new_judge_policy()
        self.event_sinks = list(event_sinks or [])
        self.session_id = session_id
        self.run_id: str | None = None
        self.state = ScientificLoopState(target=target)
        self.summary: ScientificLoopSummary | None = None
        self._projection_diagnostics: list[str] = []
        self._last_judge_report: JudgeReport | None = None

    def _new_judge_policy(self) -> JudgePolicy:
        effective_judge_mode = self.config.judge_mode or "every_round"
        default_judge_calls = 1 if effective_judge_mode == "on_demand" else self.config.max_rounds
        return JudgePolicy(
            mode=effective_judge_mode,
            require_judge=self.config.require_judge,
            max_calls=(self.config.judge_max_calls if self.config.judge_max_calls is not None
                       else default_judge_calls),
        )

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    async def run(
        self,
        *,
        goal: str | None = None,
        max_rounds: int | None = None,
    ) -> AsyncIterator[RuntimeEvent]:
        """Run the outer loop, forwarding every inner runtime event.

        After the loop settles, ``self.summary`` holds the structured outcome
        and the full event trajectory was forwarded to the same sinks.
        """
        effective_goal = goal or self.target.goal
        effective_max_rounds = max_rounds or self.config.max_rounds
        self.run_id = uuid4().hex
        self.state = ScientificLoopState(target=self.target)
        self.summary = None
        self._projection_diagnostics = []
        self.judge_policy = self._new_judge_policy()
        feedback: FeedbackSignal | None = None
        self.state.status = "RUNNING"
        dry_evidence_rounds = 0

        yield await self._emit(
            ScientificLoopStarted(
                goal=effective_goal,
                max_rounds=effective_max_rounds,
                min_confidence=self.config.min_confidence,
            )
        )

        while True:
            if self.state.round >= effective_max_rounds:
                # max_rounds rounds completed without a terminal decision.
                decision = ScientificLoopDecision(
                    action="BUDGET_EXHAUSTED",
                    reason=(
                        f"max rounds {effective_max_rounds} exceeded without "
                        "scientific convergence"
                    ),
                    best_candidate_id=self.state.best_candidate_id,
                )
                self.state.status = decision.action
                break
            started = time.monotonic()
            self.state.round += 1
            instruction = self._round_instruction(effective_goal, feedback)
            async for event in self.runtime.run(instruction):
                yield event
            round_meta = self.runtime.last_run_meta
            # Only a wait whose workflow is still active at round end exempts
            # this round.  A wait observed earlier in the same round that has
            # since reached a terminal state, a runtime that declared
            # no_progress, or a runtime that exhausted its bounded diagnostic
            # exploration does not mask later stagnation.
            non_wait_terminations = {
                "no_progress",
                "diagnostic_budget_exhausted",
            }
            controlled_wait_observed = bool(
                round_meta.get("controlled_wait_active")
            ) and round_meta.get("termination_reason") not in non_wait_terminations

            scientific = self.runtime.scientific_state
            projected = self._extract_candidates(scientific, self.state.round)
            self.state.register_candidates(projected)
            candidate = self._select_candidate(scientific)
            if candidate is None:
                dry_evidence_rounds += 1
            else:
                dry_evidence_rounds = 0
                yield await self._emit(
                    CandidateProposed(
                        round=self.state.round,
                        candidate_id=candidate.candidate_id,
                        label=candidate.label,
                        fingerprint=candidate.fingerprint[:12],
                        generation_method=candidate.generation_method,
                        projection_diagnostics=list(self._projection_diagnostics),
                    )
                )

            evaluation = self.evaluator.evaluate(candidate, scientific)
            self.state.add_candidate(candidate, evaluation)
            if candidate is not None:
                if controlled_wait_observed:
                    self.stagnation.record_controlled_wait()
                    self.state.no_progress_rounds = 0
                else:
                    progress = progress_from_evaluation(
                        candidate,
                        evaluation,
                        scientific,
                    )
                    self.stagnation.record(
                        candidate, evaluation, progress=progress
                    )
                    self.state.no_progress_rounds = (
                        self.stagnation.no_progress_rounds
                    )

            yield await self._emit(
                CandidateEvaluated(
                    round=self.state.round,
                    candidate_id=candidate.candidate_id if candidate else "",
                    score=evaluation.score,
                    verdict=evaluation.verdict,
                    violations=[v.short() for v in evaluation.violations],
                    evidence_gaps=evaluation.critical_evidence_gaps,
                )
            )

            judge_report = await self._assess_judge(
                scientific=scientific,
                candidate=candidate,
                evaluation=evaluation,
            )
            self._last_judge_report = judge_report if candidate is not None else None
            if judge_report is not None:
                yield await self._emit(
                    CandidateJudged(
                        round=self.state.round,
                        candidate_id=candidate.candidate_id if candidate else "",
                        status=judge_report.status,
                        quality=judge_report.scientific_quality,
                        issues=[
                            f"[{issue.severity}] {issue.category}: {issue.description}"
                            for issue in judge_report.issues
                        ],
                        summary=judge_report.rationale[:200],
                    )
                )

            feedback = (
                build_feedback(
                    self.target,
                    candidate,
                    evaluation,
                    self.state.candidates,
                    judge=judge_report,
                    scientific=scientific,
                    prior_evaluations=self.state.historical_candidate_evaluations[:-1],
                )
                if candidate is not None
                else None
            )

            inconclusive_reason = None
            if dry_evidence_rounds >= 2:
                inconclusive_reason = (
                    "consecutive rounds produced neither a candidate nor new "
                    "scientific evidence (capability unavailable or tool failures)"
                )
            decision = self.policy.decide(
                evaluation=evaluation,
                state=self.state,
                stagnation=self.stagnation,
                max_rounds=effective_max_rounds,
                max_candidates=self.config.max_candidates,
                min_confidence=self.config.min_confidence,
                escalate_requested=bool(
                    feedback is not None and feedback.decision == "ESCALATE"
                ),
                inconclusive_reason=inconclusive_reason,
                judge=self._last_judge_report,
            )
            yield await self._emit(
                ScientificLoopDecisionMade(
                    round=self.state.round,
                    action=decision.action,
                    reason=decision.reason,
                    best_candidate_id=decision.best_candidate_id,
                    best_score=self.state.best_score,
                )
            )

            self.state.status = decision.action
            if decision.action in {"SUCCESS", "STALLED", "INCONCLUSIVE", "BUDGET_EXHAUSTED"}:
                break
            # CONTINUE / ESCALATE: turn feedback into the next maker turn.
            # A PASS verdict without sufficient confidence yields no feedback
            # (section 10: PASS produces no new feedback); the loop-state
            # snapshot still guides the next maker turn.
            if feedback is not None:
                self.state.feedback_history.append(feedback)
                yield await self._emit(
                    ScientificFeedbackGenerated(
                        round=self.state.round,
                        candidate_id=feedback.candidate_id,
                        decision=feedback.decision,
                        summary=feedback.summary,
                    )
                )

        self.summary = self._build_summary(decision)
        if decision.action == "STALLED":
            yield await self._emit(
                ScientificLoopStalled(
                    rounds=self.state.round,
                    best_score=self.state.best_score,
                    best_candidate_id=self.state.best_candidate_id,
                    no_progress_rounds=self.stagnation.no_progress_rounds,
                )
            )
        else:
            yield await self._emit(
                ScientificLoopCompleted(
                    status=self.summary.status,
                    rounds=self.summary.rounds,
                    candidate_count=self.summary.candidate_count,
                    best_candidate_id=self.summary.best_candidate_id,
                    best_score=self.summary.best_score,
                    termination_reason=self.summary.termination_reason,
                    projection_diagnostics=self.summary.projection_diagnostics,
                )
            )

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _round_instruction(
        self, goal: str, feedback: FeedbackSignal | None
    ) -> str:
        parts = [goal]
        if feedback is not None:
            parts.append(
                format_feedback_for_model(feedback, round_number=self.state.round - 1)
            )
        elif self.state.round > 1:
            # PASS verdict without sufficient confidence produces no feedback
            # by design (section 10); the loop-state snapshot still guides the
            # next maker turn.
            parts.append(format_loop_state_snapshot(self.state, self._last_judge_report))
        return "\n\n".join(parts)

    async def _assess_judge(
        self,
        *,
        scientific: ScientificState,
        candidate: CandidateState | None,
        evaluation: EvaluationReport,
    ) -> JudgeReport | None:
        """Run the advisory LLM judge, isolated and read-only.

        The judge can never overturn a deterministic verdict: failures here
        degrade to an UNAVAILABLE report, and judge availability never
        changes a FAIL/UNKNOWN. Returns None when no judge is configured or
        there is no candidate to judge.
        """
        if candidate is None:
            return None
        artifact_payload = build_judge_snapshot(
            target=self.target,
            candidate=candidate,
            scientific=scientific,
            evaluation=evaluation,
            evidence_limit=self.judge.evidence_limit if self.judge else 24,
        )
        artifact_sha256 = hashlib.sha256(json.dumps(
            artifact_payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")).hexdigest()
        contract_sha256 = hashlib.sha256(json.dumps(
            self.target.model_dump(mode="json"), sort_keys=True,
            separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")).hexdigest()
        provider = getattr(self.judge.model, "provider", "unknown") if self.judge else "unavailable"
        model_name = getattr(self.judge.model, "model", "unknown") if self.judge else "unavailable"
        deterministic_success = (
            evaluation.verdict == "PASS"
            and not evaluation.critical_evidence_gaps
            and evaluation.confidence >= self.config.min_confidence
        )
        decision = self.judge_policy.decide(
            artifact_sha256=artifact_sha256,
            contract_sha256=contract_sha256,
            provider=provider,
            model=model_name,
            prompt_version=JUDGE_PROMPT_VERSION,
            schema_version=JUDGE_SCHEMA_VERSION,
            user_requested=self.config.judge_requested,
            non_machine_issue=self.target.metadata.get("non_machine_review") is True,
            stagnated=self.stagnation.stalled,
            required_success=deterministic_success,
        )
        if decision.cache_key is None:
            return None
        cached = self.judge_policy.cached(decision.cache_key)
        if cached is not None:
            if candidate.candidate_id:
                cached.candidate_id = candidate.candidate_id
            return cached
        if not decision.call:
            if decision.reason == "required_budget_exhausted":
                return JudgeReport(
                    status="UNAVAILABLE", provider=provider, model=model_name,
                    error="judge call budget exhausted while required for SUCCESS",
                )
            return None
        if self.judge is None:
            report = JudgeReport(
                status="UNAVAILABLE", provider=provider, model=model_name,
                error="judge is required or triggered but no provider is configured",
            )
            self.judge_policy.remember(decision.cache_key, report)
            return report
        try:
            report = await self.judge.assess(
                target=self.target,
                candidate=candidate,
                scientific=scientific,
                evaluation=evaluation,
                round_number=self.state.round,
            )
        except Exception as exc:
            report = JudgeReport(
                status="UNAVAILABLE",
                provider=getattr(self.judge.model, "provider", "unknown"),
                model=getattr(self.judge.model, "model", "unknown"),
                error=f"judge failure must not break the loop: {type(exc).__name__}: {exc}",
            )
        self.judge_policy.remember(decision.cache_key, report)
        return report

    def _extract(self, scientific: ScientificState, round_number: int) -> CandidateState | None:
        return self.candidate_extractor(scientific, round_number)

    def _extract_candidates(
        self,
        scientific: ScientificState,
        round_number: int,
    ) -> list[CandidateState]:
        if self.candidate_extractor is not extract_candidate_from_state:
            candidate = self._extract(scientific, round_number)
            return [candidate] if candidate is not None else []
        projection = project_candidates_from_state(scientific, round_number)
        for diagnostic in projection.diagnostics:
            rendered = (
                f"{diagnostic.code}:{diagnostic.evidence_id}:{diagnostic.message}"
            )
            if rendered not in self._projection_diagnostics:
                self._projection_diagnostics.append(rendered)
        return projection.candidates

    def _select_candidate(self, scientific: ScientificState) -> CandidateState | None:
        active = self.state.candidate(self.state.active_candidate_id)
        if active is not None and self._has_new_applicable_evidence(active, scientific):
            return active
        if self.state.pending_candidate_ids:
            self.state.active_candidate_id = self.state.pending_candidate_ids.pop(0)
            return self.state.candidate(self.state.active_candidate_id)
        return active

    def _has_new_applicable_evidence(
        self,
        candidate: CandidateState,
        scientific: ScientificState,
    ) -> bool:
        prior_pairs = [
            (previous_candidate, previous_evaluation)
            for previous_candidate, previous_evaluation
            in self.state.historical_candidate_evaluations
            if previous_candidate.fingerprint == candidate.fingerprint
        ]
        if not prior_pairs:
            return False
        current = self.evaluator.evaluate(candidate, scientific)
        current_progress = progress_from_evaluation(
            candidate,
            current,
            scientific,
        )
        previous_observations: set[str] = set()
        previous_questions: set[str] = set()
        for previous_candidate, previous in prior_pairs:
            previous_progress = progress_from_evaluation(
                previous_candidate,
                previous,
                scientific,
            )
            previous_observations.update(previous_progress.observation_keys)
            previous_questions.update(previous_progress.resolved_questions)
        return bool(set(current_progress.observation_keys) - previous_observations) or bool(
            set(current_progress.resolved_questions) - previous_questions
        )

    def _build_summary(self, decision: ScientificLoopDecision) -> ScientificLoopSummary:
        final_evaluation: EvaluationReport | None = (
            self.state.evaluations[-1] if self.state.evaluations else None
        )
        return ScientificLoopSummary(
            status=decision.action,  # type: ignore[arg-type]
            rounds=self.state.round,
            candidate_count=len(self.state.candidates),
            best_candidate_id=self.state.best_candidate_id,
            best_score=self.state.best_score,
            final_evaluation=final_evaluation,
            unresolved_violations=(
                list(final_evaluation.violations) if final_evaluation else []
            ),
            unresolved_evidence_gaps=(
                list(final_evaluation.critical_evidence_gaps)
                if final_evaluation
                else []
            ),
            termination_reason=decision.reason,
            judge_report=self._last_judge_report,
            projection_diagnostics=list(self._projection_diagnostics),
        )

    async def _emit(self, event: RuntimeEvent) -> RuntimeEvent:
        event.session_id = self.session_id
        event.run_id = self.run_id
        for sink in self.event_sinks:
            result = sink(event)
            if inspect.isawaitable(result):
                await result
        return event
