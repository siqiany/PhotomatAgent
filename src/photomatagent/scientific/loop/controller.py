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
from photomatagent.scientific.loop.judge import JudgeReport, ScientificJudge
from photomatagent.scientific.loop.policy import (
    ScientificLoopDecision,
    ScientificLoopPolicy,
    ScientificLoopState,
    ScientificLoopSummary,
)
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
        self.event_sinks = list(event_sinks or [])
        self.session_id = session_id
        self.run_id: str | None = None
        self.state = ScientificLoopState(target=target)
        self.summary: ScientificLoopSummary | None = None
        self._note: str | None = None
        self._last_judge_report: JudgeReport | None = None

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

            scientific = self.runtime.scientific_state
            candidate = self._extract(scientific, self.state.round)
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
                    )
                )

            evaluation = self.evaluator.evaluate(candidate, scientific)
            self.state.add_candidate(candidate, evaluation)
            if candidate is not None:
                self.stagnation.record(candidate, evaluation)

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
        if self.judge is None or candidate is None:
            return None
        try:
            return await self.judge.assess(
                target=self.target,
                candidate=candidate,
                scientific=scientific,
                evaluation=evaluation,
                round_number=self.state.round,
            )
        except Exception as exc:
            return JudgeReport(
                status="UNAVAILABLE",
                provider=getattr(self.judge.model, "provider", "unknown"),
                model=getattr(self.judge.model, "model", "unknown"),
                error=f"judge failure must not break the loop: {type(exc).__name__}: {exc}",
            )

    def _extract(self, scientific: ScientificState, round_number: int) -> CandidateState | None:
        try:
            return self.candidate_extractor(scientific, round_number)
        except Exception as exc:  # a broken extractor must not kill the loop
            self._note = f"candidate extraction failed: {type(exc).__name__}: {exc}"
            return None

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
        )

    async def _emit(self, event: RuntimeEvent) -> RuntimeEvent:
        event.session_id = self.session_id
        event.run_id = self.run_id
        for sink in self.event_sinks:
            result = sink(event)
            if inspect.isawaitable(result):
                await result
        return event