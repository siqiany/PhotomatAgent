"""The complete, visible PhotomatAgent control loop.

Read this file top-to-bottom to follow:
context -> provider stream -> tool calls -> permission -> execution -> results
-> next iteration -> stop. No SDK or UI owns any part of that sequence.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from photomatagent.errors import ProviderError, ToolError, ToolValidationError
from photomatagent.runtime.evidence_attestation import (
    EvidenceAttestationPolicy,
    _RuntimeEvidenceAuthority,
)
from photomatagent.models.base import ModelProvider
from photomatagent.models.types import (
    AssistantMessage,
    ModelCompleted,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelStreamStarted as ProviderStreamStarted,
    ModelTextDelta,
    ModelToolCallArgumentsDelta,
    ModelToolCallCompleted,
    ModelToolCallStarted,
    ModelUsageUpdated,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from photomatagent.redaction import redact_text
from photomatagent.runtime.budget import BudgetState
from photomatagent.runtime.context import ContextBuilder, format_scientific_state
from photomatagent.runtime.context_engine import (
    ContextBuildCancelled,
    ContextBuildResult,
    ContextEngine,
    ContextEngineConfig,
    ContextLimitExceeded,
    ProviderContextSummarizer,
)
from photomatagent.runtime.context_budget import account_context, PromptTokenEstimator, prompt_token_upper_bound
from photomatagent.runtime.context_segments import (
    UnsafeContextHistory,
    split_atomic_spans,
)
from photomatagent.runtime.events import (
    BudgetUpdated,
    HypothesisRegistered,
    HypothesisRegistrationReasonCode,
    HypothesisRegistrationRejected,
    StructureDerived,
    LoopCompleted,
    LoopFailed,
    LoopIterationStarted,
    LoopStarted,
    ModelRequestStarted,
    ModelResponseCompleted,
    OutputRecoveryStarted,
    ModelStreamStarted,
    ProviderFailed,
    RuntimeEvent,
    ScientificTraceMeta,
    ScientificStateUpdated,
    SensitiveAccessBlocked,
    TextDelta,
    ToolApprovalRequired,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallStarted,
    ToolCompleted,
    ToolFailed,
    ToolPermissionDenied,
    ToolRequested,
    ToolStarted,
)
from photomatagent.runtime.permissions import (
    ApprovalHandler,
    ApprovalRequest,
    PermissionDecision,
    PermissionPolicy,
    default_permission_policy,
)
from photomatagent.runtime.observation import ObservationPolicy
from photomatagent.runtime.sensitive import SensitiveAccessError, SensitivePathPolicy
from photomatagent.runtime.state import ConversationState
from photomatagent.runtime.stop_policy import StopPolicy
from photomatagent.scientific.calculations import CalculationRecord
from photomatagent.scientific.claims import ScientificClaim
from photomatagent.scientific.evidence import Evidence
from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.capabilities.structure.artifacts import (
    file_sha256,
    load_structure_input,
    structure_hash as canonical_structure_hash,
    verify_structure_derivation,
    verify_structure_derivations,
)
from photomatagent.scientific.state import (
    EvidenceAttestation,
    ScientificReferenceUpdate,
    ScientificState,
)
from photomatagent.scientific.discovery.models import (
    HypothesisOrigin,
    HypothesisRegistration,
    ScientificHypothesis,
)
from photomatagent.scientific.discovery.structures import (
    StructureDerivation,
    StructureRegistration,
)
from photomatagent.scientific.discovery.registration import (
    build_hypothesis,
    validate_proposal,
)
from photomatagent.skills.loader import SkillLoader
from photomatagent.scientific.tasks import ScientificTask
from photomatagent.sessions.store import SessionSnapshot
from photomatagent.tools.base import ToolResult
from photomatagent.tools.registry import ToolRegistry
from photomatagent.tools.exposure import ToolExposure
from photomatagent.tools.surface import (
    ToolSurfaceConfig,
    ToolSurfacePlanner,
    compact_parameter_help,
)
from photomatagent.workspace import Workspace

EventSink = Callable[[RuntimeEvent], Awaitable[None] | None]
_HYPOTHESIS_REGISTRATION_TOOL = "generation.register_hypothesis"
# Scheduler states that mean a real calculation is still progressing.  A
# controlled wait is only trustworthy while its workflow is in one of these.
_ACTIVE_WAIT_STATES = frozenset(
    {"SUBMITTED", "PENDING", "RUNNING", "RECONCILING"}
)
# Tools that may grant a controlled-wait exemption.
_WAIT_GRANT_TOOLS = frozenset({"vasp.wait", "vasp.status"})
# Tools that report a workflow's current scheduler/lifecycle state and may
# therefore revoke an earlier exemption once that workflow has ended.
_WAIT_STATE_TOOLS = frozenset(
    {"vasp.wait", "vasp.status", "vasp.collect", "vasp.report", "vasp.resume"}
)
_EXPLICIT_TASK_SWITCH_MARKERS = (
    "新任务",
    "换个任务",
    "换一个任务",
    "另一个任务",
    "重新开始",
    "new task",
    "switch task",
    "different task",
)
_TASK_SWITCH_NEGATION = re.compile(
    r"(不要|不用|别|不是|没有|无需|不必|do not|don't|not|without)"
)
_TASK_SWITCH_REFERENCE = re.compile(
    r"(之前|刚才|曾经|说过|引用|旧指令|historical|previously|earlier)"
)
_TASK_SWITCH_CLAUSE_BOUNDARY = re.compile(r"[，,。.;；!！?？\n]")


def _pause_next_action(stop_reason: str, summary: dict[str, Any]) -> str:
    """Deterministic next-step line for a paused run (no model involved)."""
    if stop_reason == "output_truncated":
        if summary.get("observed_evidence"):
            return (
                "CONTINUE_FROM_SUMMARY: the model output was truncated; resume "
                "from the observed results above (see each item's "
                "classification) and re-request only the missing step"
            )
        return (
            "RETRY_WITH_SMALLER_STEP: the model output was truncated before any "
            "persisted result; split the task into a smaller step"
        )
    if stop_reason == "diagnostic_budget_exhausted":
        return (
            "REPORT_BLOCKER: the bounded diagnostic exploration budget is "
            "exhausted; report the blocker and the missing diagnostics instead "
            "of continuing to explore"
        )
    if stop_reason == "no_progress":
        return (
            "REPORT_BLOCKER: no new task progress was observed; explain what is "
            "missing or ask the user for the required input"
        )
    if stop_reason == "max_iterations":
        return "CONTINUE: raise the iteration budget only after inspecting why it ran out"
    if stop_reason == "output_truncated_scientific":
        return "REPORT_STATUS: report the persisted stage/evidence facts"
    return "REVIEW: inspect the persisted facts before the next action"


def _is_explicit_task_switch(goal: str) -> bool:
    """Return True only for a non-negated, non-quoted task-switch intent.

    Substring matching is not enough: ``不要开始新任务，继续原任务`` must keep
    the original goal, and references to an earlier instruction must not be
    treated as a fresh switch.
    """
    normalized = goal.strip().casefold()
    for marker in _EXPLICIT_TASK_SWITCH_MARKERS:
        start = 0
        while True:
            index = normalized.find(marker, start)
            if index < 0:
                break
            prefix = normalized[:index]
            last_clause = _TASK_SWITCH_CLAUSE_BOUNDARY.split(prefix)[-1]
            if not _TASK_SWITCH_NEGATION.search(last_clause) and not (
                _TASK_SWITCH_REFERENCE.search(last_clause)
            ):
                return True
            start = index + len(marker)
        # If a marker appears only in negated/quoted clauses, it is not a
        # switch; continue checking the remaining explicit markers.
    return False
_STRUCTURE_REGISTRATION_TOOLS = frozenset(
    {
        "structure.make_supercell",
        "structure.substitute_sites",
        "structure.enumerate_orderings",
    }
)
_UNCONFIRMED_ARGUMENTS_PLACEHOLDER = (
    "[tool arguments withheld: completion unavailable]"
)


@dataclass
class _BufferedArgumentDelta:
    """One argument fragment awaiting the authoritative completed call."""

    event: ModelToolCallArgumentsDelta
    resolved: bool = False
    replacement: ToolCallArgumentsDelta | None = None


def _bounded_event_text(value: object, *, limit: int) -> str:
    return redact_text(str(value))[:limit]


def _is_hypothesis_registration_call(
    tool_name: str, arguments: dict[str, object]
) -> bool:
    if tool_name == _HYPOTHESIS_REGISTRATION_TOOL:
        return True
    return (
        tool_name == "tool_call"
        and arguments.get("name") == _HYPOTHESIS_REGISTRATION_TOOL
    )


def _hypothesis_event_arguments(arguments: dict[str, object]) -> dict[str, object]:
    """Project a registration proposal to bounded identifiers for event sinks."""

    basis = arguments.get("basis")
    basis_items = basis if isinstance(basis, list) else []
    parent_ids = arguments.get("parent_hypothesis_ids")
    parent_items = parent_ids if isinstance(parent_ids, list) else []
    questions = arguments.get("validation_questions")
    question_items = questions if isinstance(questions, list) else []
    return {
        "request_id": _bounded_event_text(arguments.get("request_id", ""), limit=128),
        "formula": _bounded_event_text(arguments.get("formula", ""), limit=256),
        "design_operation": _bounded_event_text(
            arguments.get("design_operation", ""), limit=64
        ),
        "parent_hypothesis_ids": [
            _bounded_event_text(item, limit=128) for item in parent_items[:8]
        ],
        "basis_evidence_ids": [
            _bounded_event_text(item.get("evidence_id", ""), limit=128)
            for item in basis_items[:16]
            if isinstance(item, dict)
        ],
        "basis_count": len(basis_items),
        "validation_question_count": len(question_items),
    }


def _event_tool_arguments(
    tool_name: str, arguments: dict[str, object]
) -> dict[str, object]:
    if tool_name == _HYPOTHESIS_REGISTRATION_TOOL:
        return _hypothesis_event_arguments(arguments)
    if (
        tool_name == "tool_call"
        and arguments.get("name") == _HYPOTHESIS_REGISTRATION_TOOL
    ):
        nested = arguments.get("arguments")
        return {
            "name": _HYPOTHESIS_REGISTRATION_TOOL,
            "arguments": _hypothesis_event_arguments(
                nested if isinstance(nested, dict) else {}
            ),
        }
    return arguments


class AgentRuntime:
    def __init__(
        self,
        *,
        model: ModelProvider,
        tools: ToolRegistry,
        workspace: Workspace | Path | str | None = None,
        scientific_state: ScientificState | None = None,
        conversation_state: ConversationState | None = None,
        context_builder: ContextBuilder | None = None,
        tool_surface_planner: ToolSurfacePlanner | None = None,
        tool_surface_config: ToolSurfaceConfig | None = None,
        observation_policy: ObservationPolicy | None = None,
        context_engine: ContextEngine | None = None,
        context_engine_config: ContextEngineConfig | None = None,
        sensitive_path_policy: SensitivePathPolicy | None = None,
        model_context_limit: int | None = None,
        permission_policy: PermissionPolicy | None = None,
        stop_policy: StopPolicy | None = None,
        budget: BudgetState | None = None,
        approval_handler: ApprovalHandler | None = None,
        event_sinks: list[EventSink] | None = None,
        session_id: str | None = None,
        fresh_approval: bool = False,
        application_approval_root: Path | None = None,
        evidence_attestation_policy: EvidenceAttestationPolicy | None = None,
        max_no_progress_iterations: int = 3,
        diagnostic_observation_budget: int = 12,
        diagnostic_warning_at: int = 8,
    ) -> None:
        self._model = model
        self._tools = tools
        self._workspace = (
            workspace
            if isinstance(workspace, Workspace)
            else Workspace(workspace or Path.cwd())
        )
        self._scientific = scientific_state or ScientificState()
        self._evidence_authority = _RuntimeEvidenceAuthority()
        self._evidence_authority.bind(self._scientific, replace=True)
        self._evidence_attestation = (
            evidence_attestation_policy or EvidenceAttestationPolicy()
        )
        if context_builder is None and fresh_approval:
            fresh_skills_dir = self._workspace.resolve(
                ".photomatagent/fresh-context/skills",
                must_exist=False,
            )
            fresh_skills_dir.mkdir(parents=True, exist_ok=True)
            context_builder = ContextBuilder(SkillLoader(fresh_skills_dir))
        self._context_builder = context_builder or ContextBuilder(
            SkillLoader(published_root=self._workspace.root / ".photomatagent" / "qd" / "published-skills")
        )
        self._tool_surface = tool_surface_planner or ToolSurfacePlanner(
            tools, tool_surface_config
        )
        self._observation = observation_policy or ObservationPolicy()
        if context_engine is not None:
            engine_config = context_engine.config
        else:
            engine_config = context_engine_config or ContextEngineConfig()
        if model_context_limit is not None:
            config_data = engine_config.model_dump()
            config_data["context_limit_tokens"] = model_context_limit
            engine_config = ContextEngineConfig.model_validate(config_data)
        if context_engine is None:
            self._context_engine = ContextEngine(
                config=engine_config,
                summarizer=ProviderContextSummarizer(model, engine_config),
            )
        else:
            self._context_engine = context_engine
            if model_context_limit is not None:
                self._context_engine.config = engine_config
                summarizer = self._context_engine.summarizer
                if isinstance(summarizer, ProviderContextSummarizer):
                    summarizer.config = engine_config
        self._model_context_limit = self._context_engine.config.context_limit_tokens
        self._sensitive_paths = sensitive_path_policy or SensitivePathPolicy()
        self._permission = permission_policy or default_permission_policy()
        self._stop_policy = stop_policy or StopPolicy()
        self._budget = budget or BudgetState()
        self._prompt_token_estimator = PromptTokenEstimator()
        if isinstance(self._context_engine.summarizer, ProviderContextSummarizer):
            self._context_engine.summarizer.budget_state = self._budget
        self._approval_handler = approval_handler
        self._event_sinks = list(event_sinks or [])
        self._conversation = conversation_state or ConversationState()
        self._session_id = session_id or uuid4().hex
        self._run_id: str | None = None
        self._run_meta: dict[str, Any] = {}
        self._active_wait_workflows: set[str] = set()
        self._fresh_approval = fresh_approval
        self._application_approval_root = application_approval_root
        self._max_no_progress_iterations = max(
            1, int(max_no_progress_iterations)
        )
        # Bounded, configurable diagnostic exploration for a *blocked
        # scientific workflow*.  Distinct observations still grow knowledge,
        # but they must not reset the budget forever; only a trusted task
        # advance, new task-relevant evidence, a verified recovery action, or
        # a controlled scheduler wait ends the blocked period.
        self._diagnostic_budget = max(1, int(diagnostic_observation_budget))
        self._diagnostic_warning_at = max(
            1, min(int(diagnostic_warning_at), self._diagnostic_budget)
        )
        # Set when the current iteration produced real task progress (new
        # evidence or a trusted stage advance).  It ends the blocked
        # diagnostic period for that iteration.
        self._iteration_had_task_progress = False

    @property
    def scientific_state(self) -> ScientificState:
        return self._scientific

    @property
    def model_provider(self) -> ModelProvider:
        """Provider available to bounded, tool-free host workflows."""

        return self._model

    @property
    def conversation_state(self) -> ConversationState:
        return self._conversation

    @property
    def budget(self) -> BudgetState:
        return self._budget

    @property
    def context_engine(self) -> ContextEngine:
        return self._context_engine

    @property
    def permission_policy(self) -> PermissionPolicy:
        """Current policy, exposed for explicit interactive-session controls."""
        return self._permission

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    @property
    def session_id(self) -> str:
        """Stable read-only identity used to attribute one runtime session."""

        return self._session_id

    @property
    def last_run_meta(self) -> dict[str, Any]:
        """Read-only copy of bounded metadata from the most recent run.

        ``controlled_wait_active`` is the still-valid exemption: it is true
        only while at least one workflow's latest trusted state is an active
        scheduler wait.  ``controlled_wait_seen`` is the historical fact that
        a wait happened at some point in the run and is never revoked.
        """
        meta = dict(self._run_meta)
        if meta.get("termination_reason") not in {
            "no_progress",
            "diagnostic_budget_exhausted",
        }:
            meta["controlled_wait_active"] = bool(self._active_wait_workflows)
        return meta

    def _blocked_diagnostic_period(self) -> bool:
        """True while a scientific workflow is blocked on a known blocker.

        Only a real scientific workflow with an explicit blocker counts: an
        ordinary conversation or source-reading task keeps its existing
        behaviour and is never given a diagnostic budget.  The period also
        ends as soon as the workflow makes real progress (new evidence), so a
        successful collect or a completed stage restores the budget instead of
        being charged against it.
        """
        if not self._scientific.workflow_id:
            return False
        if self._iteration_had_task_progress:
            return False
        if self._scientific.next_stage:
            return True
        return bool(self._scientific.evidence_gaps)

    def _record_diagnostic_observation(self, *, count: int = 1) -> None:
        """Charge ``count`` new diagnostic observations to the blocked budget.

        Charged per tool observation, never per model iteration, so batched
        probes each pay.  Trusted scheduler waits and repeated identical
        probes contribute ``count == 0`` and are therefore free.  The budget is
        restored when the workflow stops being blocked (a trusted stage
        advance, new task-relevant evidence, or a verified recovery action
        clears the blocker).
        """
        if not self._blocked_diagnostic_period():
            self._run_meta["blocked"] = False
            self._run_meta["diagnostic_observations"] = 0
            self._run_meta["diagnostic_warning_issued"] = False
            self._run_meta["diagnostic_budget_remaining"] = self._diagnostic_budget
            return
        if count <= 0:
            return
        self._run_meta["blocked"] = True
        self._run_meta["diagnostic_observations"] = (
            int(self._run_meta.get("diagnostic_observations", 0)) + int(count)
        )
        used = int(self._run_meta["diagnostic_observations"])
        self._run_meta["diagnostic_budget_remaining"] = max(
            0, self._diagnostic_budget - used
        )

    def pause_summary(self) -> dict[str, Any]:
        """Deterministic wrap-up built only from already-persisted facts.

        Used when a run ends without a usable final answer (truncation, budget
        exhaustion, no-progress) or when the caller wants a visible stop
        summary.  It never calls the model and never invents a scientific
        conclusion: anything not observed is reported as UNKNOWN.
        """
        meta = self.last_run_meta
        evidence = list(self._scientific.evidence)
        workflow_id = self._scientific.workflow_id or ""
        stop_reason = str(meta.get("termination_reason") or "") or "UNKNOWN"

        def _provenance(item: Any) -> dict[str, Any]:
            value = getattr(item, "provenance", None)
            return value if isinstance(value, dict) else {}

        observed: list[dict[str, Any]] = []
        attested: list[dict[str, Any]] = []
        exploratory: list[dict[str, Any]] = []
        background: list[dict[str, Any]] = []
        for item in evidence:
            provenance = _provenance(item)
            status = str(
                provenance.get("assessment_status") or ""
            ).upper()
            acceptable = provenance.get("acceptable_as_target_evidence")
            # Host-held ledger only.  ``provenance`` is producer payload: any
            # tool state update can write it and a restored snapshot carries
            # it verbatim.  ``verified_attestation`` is the non-persistent
            # record the runtime created from the *executed registry tool*,
            # so producer payload may downgrade an item (exploratory) but can
            # never upgrade it.
            attestation = self._scientific.verified_attestation(
                str(getattr(item, "id", ""))
            )
            authority = (
                attestation.authority if attestation is not None else None
            )
            # Presence in ScientificState is not validation, and even a
            # trusted source is not validation of the target: the runtime
            # reports attested *observations*.  Deciding that a target is
            # scientifically validated belongs to the deterministic checker,
            # so this summary never claims it.
            if status == "EXPLORATORY" or acceptable is False:
                bucket = exploratory
                classification = "EXPLORATORY"
            elif authority == "observation":
                bucket = attested
                classification = "ATTESTED_OBSERVATION"
            elif authority == "synthetic":
                bucket = background
                classification = "SYNTHETIC"
            else:
                bucket = background
                classification = "BACKGROUND"
            record = {
                "property": getattr(item, "property", ""),
                "value": getattr(item, "value", None),
                "unit": getattr(item, "unit", ""),
                "stage": provenance.get("stage", ""),
                "job_id": provenance.get("job_id"),
                "assessment_status": provenance.get("assessment_status"),
                "classification": classification,
                # Host-held origin of the observation, not the payload's
                # self-description.  ``None`` means no runtime attestation.
                "attested": attestation is not None,
                "attestation_authority": authority,
                "attestation_origin": (
                    attestation.origin if attestation is not None else None
                ),
                "attestation_tool_name": (
                    attestation.tool_name if attestation is not None else None
                ),
            }
            observed.append(record)
            bucket.append(record)
        summary: dict[str, Any] = {
            "stop_reason": stop_reason,
            "workflow_id": workflow_id or "UNKNOWN",
            "next_stage": self._scientific.next_stage or "UNKNOWN",
            "active_wait_workflows": sorted(self._active_wait_workflows),
            # Neutral name: everything found in ScientificState, with an
            # explicit per-item classification derived from the host
            # attestation ledger.  Stored != validated.
            "observed_evidence": observed,
            # Kept for existing consumers, but deliberately never populated by
            # the runtime: only the deterministic checker validates a target,
            # and an attested observation is not validation.
            "validated_evidence": [],
            "validation_status": "NOT_DETERMINED_BY_RUNTIME",
            "attested_evidence": attested,
            "exploratory_evidence": exploratory,
            "background_evidence": background,
            "claims": len(self._scientific.claims),
            "evidence_gaps": list(self._scientific.evidence_gaps),
            "identified_gaps": list(meta.get("evidence_gaps_identified") or []),
            "capability_escalations": list(
                meta.get("capability_escalations") or []
            ),
            "diagnostic_observations": meta.get("diagnostic_observations"),
            "active_subtask": self._scientific.active_subtask or "UNKNOWN",
            "goal": self._scientific.goal or "UNKNOWN",
            "note": (
                "deterministic local stop summary; it contains only persisted "
                "facts and makes no scientific claim. Evidence present in "
                "ScientificState is not automatically validated: each item is "
                "classified from the host attestation ledger, and the runtime "
                "never asserts target validation (validation_status="
                "NOT_DETERMINED_BY_RUNTIME)."
            ),
        }
        summary["blocked_reason"] = meta.get("blocked_reason") or (
            summary["evidence_gaps"][0] if summary["evidence_gaps"] else "UNKNOWN"
        )
        summary["next_action"] = _pause_next_action(stop_reason, summary)
        return summary

    def restore_session(self, snapshot: SessionSnapshot) -> None:
        """Reload a saved session snapshot and continue on top of it.

        Mutates the existing live scientific state object in place so tool
        registries that hold a reference to it keep observing state updates.
        The conversation and ContextEngine compaction cursor are replaced only
        after the prospective snapshot validates, so a bad cursor cannot
        partially corrupt the currently running session.
        """
        engine_payload = self._validate_engine_snapshot(
            snapshot.conversation.messages, snapshot.engine
        )
        restored_scientific = snapshot.scientific
        self._scientific.__dict__.clear()
        self._scientific.__dict__.update(restored_scientific.__dict__)
        self._evidence_authority.bind(self._scientific, replace=True)
        self._conversation = snapshot.conversation
        if engine_payload is None:
            self._context_engine.restore(
                compaction_state=None,
                compacted_message_count=0,
                compaction_count=0,
            )
        else:
            self._context_engine.restore(**engine_payload)

    @staticmethod
    def _validate_engine_snapshot(
        messages: list[ModelMessage], engine: object | None
    ) -> dict[str, object] | None:
        if engine is None:
            return None
        if not hasattr(engine, "model_dump"):
            raise ValueError("session snapshot has no valid ContextEngine cursor")
        payload = engine.model_dump(mode="json")  # type: ignore[attr-defined]
        raw_state = payload.get("compaction_state")
        try:
            cursor = int(payload.get("compacted_message_count") or 0)
            count = int(payload.get("compaction_count") or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "session snapshot ContextEngine cursor is not an integer"
            ) from exc
        if cursor < 0 or cursor > len(messages):
            raise ValueError(
                "session snapshot compaction cursor is outside the conversation"
            )
        if count < 0:
            raise ValueError("session snapshot compaction count is negative")
        if cursor > 0 and raw_state is None:
            raise ValueError(
                "session snapshot compaction cursor is non-zero without a summary"
            )
        if cursor == 0 and raw_state:
            raise ValueError(
                "session snapshot has a summary but a zero compaction cursor"
            )
        if cursor not in (0, len(messages)):
            try:
                spans = split_atomic_spans(messages)
            except UnsafeContextHistory as exc:
                raise ValueError(
                    "session snapshot conversation has unsafe tool pairing"
                ) from exc
            if cursor not in {span.start for span in spans}:
                raise ValueError(
                    "session snapshot compaction cursor splits a tool transaction"
                )
        return payload

    async def run(self, goal: str) -> AsyncIterator[RuntimeEvent]:
        """Run one user turn while preserving conversation and scientific state."""
        self._close_pending_tool_calls()
        self._run_id = uuid4().hex
        self._run_observed_keys: set[tuple[object, ...]] = set()
        # Workflows whose latest trusted scheduler state is still active.  A
        # wait is only an exemption while at least one workflow remains here.
        self._active_wait_workflows = set()
        self._run_meta = {
            "skills_loaded": [],
            "published_skill_activity": [],
            "scientific_tools_used": [],
            "evidence_created": 0,
            "evidence_sources": [],
            "evidence_gaps_identified": [],
            "capability_escalations": [],
            "controlled_wait_seen": False,
            "controlled_wait_this_iteration": False,
            "controlled_wait_active": False,
            "termination_reason": "",
            "new_observation_seen": False,
            "diagnostic_observations": 0,
            "diagnostic_budget": self._diagnostic_budget,
            "diagnostic_budget_remaining": self._diagnostic_budget,
            "diagnostic_warning_issued": False,
            "blocked": False,
            "new_diagnostic_observations": 0,
            "new_wait_observations": 0,
            "wait_observations": 0,
        }
        no_progress_streak = 0
        diagnostic_notice_pending = False
        run_started = time.monotonic()
        if not self._scientific.goal:
            self._scientific.goal = goal
            self._scientific.active_subtask = goal
        elif _is_explicit_task_switch(goal):
            # An explicit task switch replaces the old research objective and
            # clears stale workflow pointers.  A plain continuation/refinement
            # below never overwrites the durable original goal.
            self._scientific.goal = goal
            self._scientific.active_subtask = goal
            self._scientific.workflow_id = ""
            self._scientific.next_stage = ""
            self._scientific.evidence_gaps = []
        else:
            self._scientific.active_subtask = goal
        self._conversation.add(UserMessage(content=goal))
        iteration = 0
        original_context_config = self._context_engine.config
        output_recoveries = 0
        try:
            yield await self._emit(
                LoopStarted(
                    goal=goal,
                    provider=self._model.provider,
                    model=self._model.model,
                    workspace=str(self._workspace.root),
                )
            )
            while True:
                iteration += 1
                self._budget.record_iteration()
                progress_before = self._progress_fingerprint()
                evidence_before = len(self._scientific.evidence)
                self._iteration_had_task_progress = False
                self._run_meta["controlled_wait_this_iteration"] = False
                self._run_meta["new_observation_seen"] = False
                self._run_meta["new_diagnostic_observations"] = 0
                self._run_meta["new_wait_observations"] = 0
                if diagnostic_notice_pending:
                    # Deterministic, factual budget notice: the model is told
                    # its remaining diagnostic budget and that it must report
                    # the blocker instead of continuing to explore.
                    diagnostic_notice_pending = False
                    remaining = int(
                        self._run_meta.get("diagnostic_budget_remaining", 0)
                    )
                    self._conversation.add(
                        UserMessage(
                            content=(
                                "runtime_notice: diagnostic exploration budget "
                                f"exhausted ({self._diagnostic_budget} distinct "
                                "observations) while a scientific workflow is "
                                f"blocked. Remaining budget: {remaining}. Stop "
                                "exploring source/data and report the blocker, "
                                "the evidence already gathered, and the missing "
                                "diagnostics instead."
                            )
                        )
                    )
                    self._run_meta["diagnostic_warning_issued"] = True
                yield await self._emit(LoopIterationStarted(iteration=iteration))

                surface = self._tool_surface.plan()
                try:
                    context = await self._context_engine.build(
                        conversation=self._conversation,
                        scientific=self._scientific,
                        context_builder=self._context_builder,
                        capability_manifest=surface.manifest.text,
                        surface=surface.stats,
                        session_id=self._session_id,
                    )
                except ContextBuildCancelled as exc:
                    for context_event in await self._consume_context_result(
                        exc.result
                    ):
                        yield context_event
                    raise
                context_events = await self._consume_context_result(context)
                for context_event in context_events:
                    yield context_event
                if not context.request_allowed:
                    raise self._context_limit_error(context, context_events)
                messages = context.messages
                context_budget = account_context(
                    messages,
                    surface.stats,
                    model_context_limit=self._model_context_limit,
                )
                max_output_tokens = self._context_engine.config.response_reserve_tokens
                fallback_model_tokens = 0
                request_tools = surface.definitions
                delivery_only = False
                estimated_input_tokens: int | None = None
                if self._budget.max_total_tokens is not None:
                    remaining_tokens = self._budget.max_total_tokens - self._budget.total_tokens
                    estimated_input_tokens = self._prompt_token_estimator.estimate(messages, surface.definitions)
                    if (self._prompt_token_estimator.calibrated and remaining_tokens > 0
                            and remaining_tokens < 2 * (estimated_input_tokens + max_output_tokens)):
                        delivery_only = True
                        request_tools = []
                        messages = [*messages, UserMessage(content=(
                            "runtime_budget_notice: remaining budget is reserved for final delivery. "
                            "Do not call tools or start new searches. Deliver a complete concise answer "
                            "from existing observations, distinguish completed work from drafts, and "
                            "mark missing evidence UNKNOWN. Explain any blocker and the next step."
                        ))]
                        estimated_input_tokens = self._prompt_token_estimator.estimate(messages, request_tools)
                    max_output_tokens = min(
                        max_output_tokens, remaining_tokens - estimated_input_tokens,
                    )
                    if max_output_tokens <= 0:
                        raise RuntimeError(
                            "TOKEN_BUDGET_EXHAUSTED: "
                            f"total_limit={self._budget.max_total_tokens}, "
                            f"consumed={self._budget.total_tokens}, "
                            f"remaining={remaining_tokens}, "
                            f"input_upper_bound={estimated_input_tokens}"
                        )
                    # Missing usage never benefits from the calibrated admission
                    # estimate. Reserve the byte bound plus requested output.
                    # Interrupted/unsettled requests are accounted separately by
                    # application runners at the full remaining case budget.
                    fallback_model_tokens = min(remaining_tokens,
                        prompt_token_upper_bound(messages, request_tools) + max_output_tokens)
                request = ModelRequest(
                    messages=messages,
                    tools=request_tools,
                    max_output_tokens=max_output_tokens,
                )
                yield await self._emit(
                    ModelRequestStarted(
                        iteration=iteration,
                        message_count=len(messages),
                        provider=self._model.provider,
                        model=self._model.model,
                        registered_tools=surface.stats.registered_tools,
                        direct_count=surface.stats.direct_tools,
                        deferred_count=surface.stats.deferred_tools,
                        hidden_count=surface.stats.hidden_tools,
                        visible_schema_chars=surface.stats.visible_schema_chars,
                        manifest_chars=surface.stats.manifest_chars,
                        estimated_schema_tokens=surface.stats.estimated_visible_schema_tokens,
                        estimated_direct_schema_tokens=(
                            surface.stats.estimated_direct_schema_tokens
                        ),
                        estimated_deferred_schema_tokens=(
                            surface.stats.estimated_deferred_schema_tokens
                        ),
                        estimated_manifest_tokens=surface.stats.estimated_manifest_tokens,
                        estimated_avoided_tokens=surface.stats.estimated_avoided_tokens,
                        estimated_bridge_schema_tokens=(
                            surface.stats.estimated_bridge_schema_tokens
                        ),
                        estimated_current_prompt_tokens=(
                            context_budget.estimated_current_prompt_tokens
                        ),
                        estimated_message_history_tokens=(
                            context_budget.estimated_message_history_tokens
                        ),
                        estimated_tool_result_tokens=(
                            context_budget.estimated_tool_result_tokens
                        ),
                        model_context_limit=context_budget.model_context_limit,
                        working_context_chars=context.size.chars,
                        durable_context_chars=context.durable_size.chars,
                        pruned_tool_results=context.pruned_tool_results,
                        compaction_count=context.compaction_count,
                        admission_input_tokens=estimated_input_tokens,
                        delivery_only=delivery_only,
                        max_output_tokens=max_output_tokens,
                    )
                )

                response: ModelResponse | None = None
                model_started = time.monotonic()
                stream_tool_names: dict[str, str] = {}
                ordered_stream_buffer: deque[
                    RuntimeEvent | _BufferedArgumentDelta
                ] = deque()
                pending_argument_deltas: dict[
                    str, list[_BufferedArgumentDelta]
                ] = {}

                def queue_stream_event(event: RuntimeEvent) -> list[RuntimeEvent]:
                    if ordered_stream_buffer:
                        ordered_stream_buffer.append(event)
                        return []
                    return [event]

                def drain_resolved_stream_events() -> list[RuntimeEvent]:
                    ready: list[RuntimeEvent] = []
                    while ordered_stream_buffer:
                        item = ordered_stream_buffer[0]
                        if isinstance(item, _BufferedArgumentDelta):
                            if not item.resolved:
                                break
                            ordered_stream_buffer.popleft()
                            if item.replacement is not None:
                                ready.append(item.replacement)
                        else:
                            ordered_stream_buffer.popleft()
                            ready.append(item)
                    return ready

                def resolve_pending_arguments(
                    call: ToolCall,
                    *,
                    index: int,
                ) -> None:
                    pending = pending_argument_deltas.pop(call.id, [])
                    if not pending:
                        return
                    if _is_hypothesis_registration_call(call.name, call.arguments):
                        first, *remaining = pending
                        first.replacement = ToolCallArgumentsDelta(
                            iteration=iteration,
                            tool_call_id=call.id,
                            delta=json.dumps(
                                _event_tool_arguments(call.name, call.arguments),
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                            index=index,
                        )
                        first.resolved = True
                        for buffered in remaining:
                            buffered.resolved = True
                        return
                    for buffered in pending:
                        streamed = buffered.event
                        buffered.replacement = ToolCallArgumentsDelta(
                            iteration=iteration,
                            tool_call_id=streamed.tool_call_id,
                            delta=streamed.delta,
                            index=streamed.index,
                        )
                        buffered.resolved = True

                def withhold_unconfirmed_arguments() -> list[RuntimeEvent]:
                    for pending in pending_argument_deltas.values():
                        if not pending:
                            continue
                        first, *remaining = pending
                        first.replacement = ToolCallArgumentsDelta(
                            iteration=iteration,
                            tool_call_id=first.event.tool_call_id,
                            delta=_UNCONFIRMED_ARGUMENTS_PLACEHOLDER,
                            index=first.event.index,
                        )
                        first.resolved = True
                        for buffered in remaining:
                            buffered.resolved = True
                    pending_argument_deltas.clear()
                    return drain_resolved_stream_events()

                stream_event: RuntimeEvent
                try:
                    async for model_event in self._model.stream(request):
                        if isinstance(model_event, ProviderStreamStarted):
                            stream_event = ModelStreamStarted(
                                iteration=iteration,
                                provider=model_event.provider,
                                model=model_event.model,
                                response_id=model_event.response_id,
                            )
                            for ready in queue_stream_event(stream_event):
                                yield await self._emit(ready)
                        elif isinstance(model_event, ModelTextDelta):
                            stream_event = TextDelta(
                                iteration=iteration, text=model_event.text
                            )
                            for ready in queue_stream_event(stream_event):
                                yield await self._emit(ready)
                        elif isinstance(model_event, ModelToolCallStarted):
                            stream_tool_names[model_event.tool_call_id] = (
                                model_event.tool_name
                            )
                            stream_event = ToolCallStarted(
                                iteration=iteration,
                                tool_call_id=model_event.tool_call_id,
                                tool_name=model_event.tool_name,
                                index=model_event.index,
                            )
                            for ready in queue_stream_event(stream_event):
                                yield await self._emit(ready)
                        elif isinstance(model_event, ModelToolCallArgumentsDelta):
                            streamed_name = stream_tool_names.get(
                                model_event.tool_call_id, ""
                            )
                            if streamed_name in {
                                "tool_call",
                                _HYPOTHESIS_REGISTRATION_TOOL,
                            }:
                                buffered = _BufferedArgumentDelta(event=model_event)
                                ordered_stream_buffer.append(buffered)
                                pending_argument_deltas.setdefault(
                                    model_event.tool_call_id, []
                                ).append(buffered)
                            else:
                                stream_event = ToolCallArgumentsDelta(
                                    iteration=iteration,
                                    tool_call_id=model_event.tool_call_id,
                                    delta=model_event.delta,
                                    index=model_event.index,
                                )
                                for ready in queue_stream_event(stream_event):
                                    yield await self._emit(ready)
                        elif isinstance(model_event, ModelToolCallCompleted):
                            call = model_event.tool_call
                            resolve_pending_arguments(call, index=model_event.index)
                            completed = ToolCallCompleted(
                                iteration=iteration,
                                tool_call_id=call.id,
                                tool_name=call.name,
                                arguments=_event_tool_arguments(
                                    call.name, call.arguments
                                ),
                                index=model_event.index,
                            )
                            for ready in queue_stream_event(completed):
                                yield await self._emit(ready)
                            for ready in drain_resolved_stream_events():
                                yield await self._emit(ready)
                        elif isinstance(model_event, ModelUsageUpdated):
                            # The final aggregate is booked once below to avoid
                            # double-counting incremental usage snapshots.
                            continue
                        elif isinstance(model_event, ModelCompleted):
                            response = model_event.response
                except ProviderError as exc:
                    for ready in withhold_unconfirmed_arguments():
                        yield await self._emit(ready)
                    yield await self._emit(
                        ProviderFailed(
                            iteration=iteration,
                            provider=self._model.provider,
                            model=self._model.model,
                            error=str(exc),
                            error_type=type(exc).__name__,
                            duration_ms=(time.monotonic() - model_started) * 1000,
                        )
                    )
                    raise

                if response is None:
                    no_completion_error = ProviderError(
                        self._model.provider, "stream ended without ModelCompleted"
                    )
                    for ready in withhold_unconfirmed_arguments():
                        yield await self._emit(ready)
                    yield await self._emit(
                        ProviderFailed(
                            iteration=iteration,
                            provider=self._model.provider,
                            model=self._model.model,
                            error=str(no_completion_error),
                            error_type=type(no_completion_error).__name__,
                            duration_ms=(time.monotonic() - model_started) * 1000,
                        )
                    )
                    raise no_completion_error

                self._budget.record_model_call(
                    response.usage, fallback_tokens=fallback_model_tokens,
                )
                if response.usage.resolved_total_tokens >= response.usage.input_tokens:
                    self._prompt_token_estimator.observe(
                        messages, request_tools, input_tokens=response.usage.input_tokens,
                    )
                yield await self._emit(
                    ModelResponseCompleted(
                        iteration=iteration,
                        provider=self._model.provider,
                        model=self._model.model,
                        response_id=response.response_id,
                        finish_reason=response.finish_reason,
                        tool_call_count=len(response.tool_calls),
                        usage=response.usage.model_dump(),
                        duration_ms=(time.monotonic() - model_started) * 1000,
                        max_output_tokens=max_output_tokens,
                        response_text_chars=len(response.text),
                    )
                )
                self._conversation.add(
                    AssistantMessage(text=response.text, tool_calls=response.tool_calls)
                )
                if delivery_only and response.tool_calls:
                    self._close_tool_calls(response.tool_calls, "not executed: final delivery budget reserve")
                    raise RuntimeError("DELIVERY_TOOLS_NOT_ALLOWED: final answer required within remaining budget")

                decision = self._stop_policy.should_stop(
                    iteration=iteration, response=response, budget=self._budget
                )
                if (decision.reason == "output_truncated"
                        and output_recoveries < original_context_config.output_recovery_attempts
                        and self._budget.model_calls < self._budget.max_iterations):
                    # Never execute or splice truncated tool arguments. Preserve
                    # the failed transaction and request a fresh, bounded response.
                    self._close_tool_calls(response.tool_calls, "not executed: output_truncated")
                    next_cap = min(
                        max_output_tokens * 2, 32_768,
                        original_context_config.context_limit_tokens
                        - original_context_config.safety_margin_tokens - 1_025,
                    )
                    if next_cap > max_output_tokens:
                        output_recoveries += 1
                        self._context_engine.config = original_context_config.model_copy(
                            update={"response_reserve_tokens": next_cap}
                        )
                        recovery_action = (
                            "Do not call tools or start new searches. Deliver a concise final answer "
                            "from existing observations, with blockers and unfinished work explicit. "
                            if delivery_only else
                            "Continue with fresh complete tool requests, save small intermediate "
                            "artifacts early, and give a concise final answer. "
                        )
                        self._conversation.add(UserMessage(content=(
                            "runtime_output_recovery: the preceding response was truncated; "
                            "its tool fragments were not executed and its text is incomplete. "
                            "Do not treat it as verified evidence. " + recovery_action +
                            "Retain the original task constraints. Missing evidence is "
                            "UNKNOWN. This recovery uses the existing call and token budgets."
                        )))
                        yield await self._emit(OutputRecoveryStarted(
                            iteration=iteration, attempt=output_recoveries,
                            previous_output_cap=max_output_tokens, next_output_cap=next_cap,
                        ))
                        yield await self._emit_budget(iteration)
                        continue
                if decision.should_stop:
                    self._close_tool_calls(
                        response.tool_calls,
                        f"not executed because loop stopped: {decision.reason}",
                    )
                    self._run_meta["termination_reason"] = decision.reason
                    yield await self._emit_budget(iteration)
                    yield await self._emit(
                        LoopCompleted(
                            iterations=iteration,
                            reason=decision.reason,
                            duration_ms=(time.monotonic() - run_started) * 1000,
                        )
                    )
                    yield await self._emit(
                        ScientificTraceMeta(
                            skills_loaded=self._run_meta["skills_loaded"],
                            scientific_tools_used=self._run_meta["scientific_tools_used"],
                            evidence_created=self._run_meta["evidence_created"],
                            evidence_sources=self._run_meta["evidence_sources"],
                            evidence_gaps_identified=self._run_meta["evidence_gaps_identified"],
                            capability_escalations=self._run_meta["capability_escalations"],
                        )
                    )
                    return

                tool_calls_seen = False
                for tool_call in response.tool_calls:
                    tool_calls_seen = True
                    async for event in self._handle_tool_call(tool_call, iteration):
                        yield event
                yield await self._emit_budget(iteration)
                scientific_progress = (
                    self._progress_fingerprint() != progress_before
                )
                new_evidence = (
                    len(self._scientific.evidence) > evidence_before
                )
                if new_evidence:
                    self._iteration_had_task_progress = True
                # A trusted task advance (new evidence) ends the blocked
                # diagnostic period: the workflow is moving again, so the
                # exploratory budget is restored instead of being drained by
                # legitimate progress.
                if new_evidence and self._run_meta.get("blocked"):
                    self._run_meta["blocked"] = False
                    self._run_meta["diagnostic_observations"] = 0
                    self._run_meta["diagnostic_warning_issued"] = False
                    self._run_meta["diagnostic_budget_remaining"] = (
                        self._diagnostic_budget
                    )
                blocked_now = self._blocked_diagnostic_period()
                if blocked_now:
                    # Inside a blocked period, one *new* diagnostic observation
                    # costs one budget unit and is counted per tool call, so a
                    # batched response with several probes pays for each probe.
                    # A trusted scheduler wait is operational and costs
                    # nothing; a repeated identical probe yields no new
                    # information, so it does not drain the budget either.
                    if tool_calls_seen:
                        self._record_diagnostic_observation(
                            count=int(
                                self._run_meta.get(
                                    "new_diagnostic_observations", 0
                                )
                            )
                        )
                        if (
                            scientific_progress
                            or new_evidence
                            or self._run_meta.get(
                                "controlled_wait_this_iteration"
                            )
                        ):
                            no_progress_streak = 0
                        elif self._run_meta.get("new_observation_seen"):
                            # New knowledge is not task progress: it only
                            # decays the repetition stall by one step, so a
                            # stalled workflow still stops deterministically
                            # while distinct exploration is not killed.
                            no_progress_streak = max(0, no_progress_streak - 1)
                        else:
                            # The same probe again: no new information at all.
                            no_progress_streak += 1
                    elif (
                        scientific_progress
                        or new_evidence
                        or self._run_meta.get("controlled_wait_this_iteration")
                    ):
                        no_progress_streak = 0
                    else:
                        # No observation was attempted at all: genuine model
                        # stagnation, not budget-bounded exploration.
                        no_progress_streak += 1
                    used = int(self._run_meta["diagnostic_observations"])
                    if (
                        not self._run_meta.get("diagnostic_warning_issued")
                        and used >= self._diagnostic_warning_at
                    ):
                        diagnostic_notice_pending = True
                    if used >= self._diagnostic_budget:
                        reason = "diagnostic_budget_exhausted"
                        blocker = (
                            f"runtime_diagnostic_budget_exhausted: workflow="
                            f"{self._scientific.workflow_id or 'unknown'}; "
                            f"next_stage="
                            f"{self._scientific.next_stage or 'unknown'}; "
                            f"observations={used}; "
                            f"evidence_created="
                            f"{self._run_meta.get('evidence_created', 0)}"
                        )
                        self._run_meta.setdefault(
                            "evidence_gaps_identified", []
                        ).append(blocker)
                        self._run_meta["blocked_reason"] = blocker
                        self._run_meta["termination_reason"] = reason
                        self._run_meta["controlled_wait_active"] = False
                        yield await self._emit(
                            LoopCompleted(
                                iterations=iteration,
                                reason=reason,
                                duration_ms=(
                                    time.monotonic() - run_started
                                ) * 1000,
                            )
                        )
                        yield await self._emit(
                            ScientificTraceMeta(
                                skills_loaded=self._run_meta["skills_loaded"],
                                scientific_tools_used=self._run_meta[
                                    "scientific_tools_used"
                                ],
                                evidence_created=self._run_meta[
                                    "evidence_created"
                                ],
                                evidence_sources=self._run_meta[
                                    "evidence_sources"
                                ],
                                evidence_gaps_identified=self._run_meta[
                                    "evidence_gaps_identified"
                                ],
                                capability_escalations=self._run_meta[
                                    "capability_escalations"
                                ],
                            )
                        )
                        return
                elif (
                    scientific_progress
                    or self._run_meta.get("controlled_wait_this_iteration")
                    or self._run_meta.get("new_observation_seen")
                ):
                    no_progress_streak = 0
                else:
                    no_progress_streak += 1
                # While a blocked workflow still has diagnostic budget left,
                # the budget (not the repetition stall) is the deterministic
                # stop; a restored budget lets ordinary stall accounting apply
                # again.  A stall with no tool call at all is always reported.
                if no_progress_streak >= self._max_no_progress_iterations:
                    blocker = (
                        f"runtime_no_progress: workflow="
                        f"{self._scientific.workflow_id or 'unknown'}; "
                        f"next_stage="
                        f"{self._scientific.next_stage or 'unknown'}; "
                        f"streak={no_progress_streak}"
                    )
                    self._run_meta.setdefault(
                        "evidence_gaps_identified", []
                    ).append(blocker)
                    # A runtime that stopped for lack of progress must not be
                    # reported as a controlled wait, even if a workflow was
                    # active earlier in the run.  The outer controller relies
                    # on this explicit termination contract.
                    self._run_meta["termination_reason"] = "no_progress"
                    self._run_meta["controlled_wait_active"] = False
                    yield await self._emit(
                        LoopCompleted(
                            iterations=iteration,
                            reason="no_progress",
                            duration_ms=(
                                time.monotonic() - run_started
                            ) * 1000,
                        )
                    )
                    yield await self._emit(
                        ScientificTraceMeta(
                            skills_loaded=self._run_meta["skills_loaded"],
                            scientific_tools_used=self._run_meta[
                                "scientific_tools_used"
                            ],
                            evidence_created=self._run_meta["evidence_created"],
                            evidence_sources=self._run_meta[
                                "evidence_sources"
                            ],
                            evidence_gaps_identified=self._run_meta[
                                "evidence_gaps_identified"
                            ],
                            capability_escalations=self._run_meta[
                                "capability_escalations"
                            ],
                        )
                    )
                    return
        except Exception as exc:
            yield await self._emit(
                LoopFailed(
                    error=f"{type(exc).__name__}: {exc}",
                    duration_ms=(time.monotonic() - run_started) * 1000,
                    error_type=type(exc).__name__,
                )
            )
            raise
        finally:
            self._context_engine.config = original_context_config

    async def _consume_context_result(
        self, result: ContextBuildResult
    ) -> list[RuntimeEvent]:
        """Emit context events and account every summary model call exactly once."""
        emitted: list[RuntimeEvent] = []
        for event in result.events:
            emitted.append(await self._emit(event))
        usages = list(result.compaction_usages)
        calls = max(result.compaction_model_calls, len(usages))
        for index in range(calls):
            self._budget.record_model_call(
                usages[index] if index < len(usages) else None
            )
        return emitted

    async def compact_working_context(self) -> list[RuntimeEvent]:
        """Developer hook used by the interactive ``/compact`` command."""
        surface = self._tool_surface.plan()
        try:
            result = await self._context_engine.build(
                conversation=self._conversation,
                scientific=self._scientific,
                context_builder=self._context_builder,
                capability_manifest=surface.manifest.text,
                surface=surface.stats,
                session_id=self._session_id,
                force_compaction=True,
            )
            emitted = await self._consume_context_result(result)
        except ContextBuildCancelled as exc:
            await self._consume_context_result(exc.result)
            raise
        if not result.request_allowed:
            raise self._context_limit_error(result, emitted)
        return emitted

    @staticmethod
    def _context_limit_error(
        result: ContextBuildResult, events: list[RuntimeEvent]
    ) -> ContextLimitExceeded:
        error = ContextLimitExceeded(
            result.limit_error
            or "working context exceeds the reserved model input budget"
        )
        # Interactive callers may render the events that were already emitted
        # before the runtime refused to send an oversized model request.
        error.result = result  # type: ignore[attr-defined]
        error.events = events  # type: ignore[attr-defined]
        return error

    def _progress_fingerprint(self) -> tuple[object, ...]:
        """Stable identity of deterministic scientific progress in this run.

        File reads and repeated framework-source inspection do not change any
        of these fields and therefore do not count as progress.
        """
        return (
            len(self._scientific.evidence),
            len(self._scientific.claims),
            len(self._scientific.calculations),
            len(self._scientific.pending_tasks),
            len(self._scientific.material_hypotheses),
            len(self._scientific.structure_derivations),
            self._scientific.workflow_id,
            self._scientific.next_stage,
            tuple(self._scientific.evidence_gaps),
        )

    @staticmethod
    def _canonical_observation_value(value: object) -> str:
        try:
            return json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        except (TypeError, ValueError):
            return repr(value)

    @classmethod
    def _observation_key(
        cls,
        name: str,
        result: ToolResult,
        arguments: dict[str, object],
    ) -> tuple[object, ...]:
        """Identity of one successful observation applied to progress.

        Repeated identical observations (same workflow state or same tool
        arguments/content) do not count as progress.  New data, changed
        content, or a changed normalized workflow stage state does.
        """
        raw_data = result.data
        data: dict[str, object] = raw_data if isinstance(raw_data, dict) else {}
        raw_nested = data.get("data")
        nested: dict[str, object] = raw_nested if isinstance(raw_nested, dict) else {}

        def pick(key: str) -> object:
            value = data.get(key)
            if value not in (None, "", [], {}):
                return value
            return nested.get(key)

        def as_str_tuple(value: object) -> tuple[str, ...]:
            if isinstance(value, (list, tuple, set)):
                return tuple(sorted(str(item) for item in value))
            return ()

        workflow_id = str(pick("workflow_id") or pick("job_id") or "")
        if workflow_id:
            stage_states = pick("stage_states")
            if not isinstance(stage_states, dict):
                stage_states = {}
            normalized_stages = tuple(
                sorted(
                    (str(stage), str(state))
                    for stage, state in stage_states.items()
                )
            )
            pending = as_str_tuple(pick("pending_stages"))
            validated = as_str_tuple(pick("validated_stages"))
            wait = data.get("wait")
            settled: object = None
            if isinstance(wait, dict):
                settled = wait.get("settled")
            return (
                "workflow",
                workflow_id,
                str(pick("state") or pick("scheduler_state") or ""),
                str(pick("next_stage") or ""),
                normalized_stages,
                pending,
                validated,
                settled,
            )

        output_digest = hashlib.sha256(
            result.output.encode("utf-8", errors="replace")
        ).hexdigest()
        return (
            "observation",
            name,
            cls._canonical_observation_value(arguments),
            output_digest,
        )

    @staticmethod
    def _structured_workflow_identity(result: ToolResult) -> str:
        """Return the workflow/job identity a structured result refers to.

        Reads the public result's top-level fields first and falls back to the
        nested ``data`` payload, matching the progress-observation contract.
        """
        raw_data = result.data
        data: dict[str, object] = raw_data if isinstance(raw_data, dict) else {}
        raw_nested = data.get("data")
        nested: dict[str, object] = raw_nested if isinstance(raw_nested, dict) else {}
        raw = data.get("workflow_id") or data.get("job_id")
        if raw in (None, ""):
            raw = nested.get("workflow_id") or nested.get("job_id")
        return str(raw) if raw not in (None, "") else ""

    @classmethod
    def _structured_wait_signal(
        cls, name: str, result: ToolResult
    ) -> tuple[str, str] | None:
        """Classify a scheduler-state result as an active or ended wait.

        Returns ``(workflow_key, "active"|"inactive")`` for tools that report a
        workflow's scheduler state, else ``None``.  A per-workflow "inactive"
        signal is what revokes an earlier wait exemption: a workflow that has
        reached a terminal or non-running state is no longer a controlled wait.
        """
        if name not in _WAIT_STATE_TOOLS:
            return None
        key = cls._structured_workflow_identity(result)
        if not key:
            return None
        raw_data = result.data
        data: dict[str, object] = raw_data if isinstance(raw_data, dict) else {}
        raw_nested = data.get("data")
        nested: dict[str, object] = raw_nested if isinstance(raw_nested, dict) else {}
        raw_state = data.get("state") or data.get("scheduler_state")
        if raw_state in (None, ""):
            raw_state = nested.get("state") or nested.get("scheduler_state")
        state = str(raw_state or "")
        raw_wait = data.get("wait")
        if not isinstance(raw_wait, dict):
            raw_wait = nested.get("wait")
        wait = raw_wait if isinstance(raw_wait, dict) else None
        settled = wait.get("settled") if wait is not None else None
        if state in _ACTIVE_WAIT_STATES or settled is False:
            return key, "active"
        if state or wait is not None:
            return key, "inactive"
        return None

    @classmethod
    def _is_controlled_wait_result(cls, name: str, result: ToolResult) -> bool:
        """Detect a controlled scheduler wait from structured tool data.

        This deliberately does not scan model-visible text for "RUNNING".
        The tool must identify a workflow/job and expose an active structured
        state or an explicit unsettled wait record.
        """
        if name not in _WAIT_GRANT_TOOLS:
            return False
        signal = cls._structured_wait_signal(name, result)
        return signal is not None and signal[1] == "active"

    def _close_pending_tool_calls(self) -> None:
        """Append factual terminal results for abandoned calls; never rewrite history."""
        messages = self._conversation.messages
        fulfilled = {
            message.tool_call_id
            for message in messages
            if isinstance(message, ToolResultMessage)
        }
        unfulfilled = {
            call.id
            for message in messages
            if isinstance(message, AssistantMessage)
            for call in message.tool_calls
        } - fulfilled
        if not unfulfilled:
            return
        calls = [
            call
            for message in messages
            if isinstance(message, AssistantMessage)
            for call in message.tool_calls
            if call.id in unfulfilled
        ]
        self._close_tool_calls(calls, "not executed because the previous turn ended")

    def _close_tool_calls(self, calls: list[ToolCall], reason: str) -> None:
        fulfilled = {
            message.tool_call_id
            for message in self._conversation.messages
            if isinstance(message, ToolResultMessage)
        }
        for call in calls:
            if call.id not in fulfilled:
                self._conversation.add(
                    ToolResultMessage(
                        tool_call_id=call.id,
                        tool_name=call.name,
                        content=reason,
                        is_error=True,
                    )
                )

    async def _handle_tool_call(
        self, tool_call: ToolCall, iteration: int
    ) -> AsyncIterator[RuntimeEvent]:
        bridge_tool: str | None = None
        protocol_tool_name = tool_call.name
        if tool_call.name == "tool_call":
            raw_target_name = tool_call.arguments.get("name")
            target_name = (
                raw_target_name if isinstance(raw_target_name, str) else None
            )
            bridge_validated = False
            try:
                bridge_arguments = self._tools.validate_arguments(
                    "tool_call", tool_call.arguments
                )
                bridge_validated = True
                target_name = str(bridge_arguments["name"])
                target_arguments = bridge_arguments["arguments"]
                target = self._tools.get(target_name)
                if target.exposure is not ToolExposure.DEFERRED:
                    raise ToolValidationError(
                        f"tool_call only accepts deferred tools; {target_name!r} is "
                        f"{target.exposure.value}"
                    )
                if not isinstance(target_arguments, dict):
                    raise ToolValidationError("tool_call arguments must be an object")
                tool_call = tool_call.model_copy(
                    update={"name": target_name, "arguments": target_arguments}
                )
                bridge_tool = "tool_call"
            except (ToolValidationError, KeyError) as exc:
                if target_name == _HYPOTHESIS_REGISTRATION_TOOL:
                    yield await self._emit_hypothesis_rejection(
                        tool_call,
                        "TOOL_UNAVAILABLE"
                        if bridge_validated and isinstance(exc, KeyError)
                        else "SCHEMA_VALIDATION_FAILED",
                    )
                async for event in self._record_tool_failure(
                    tool_call,
                    iteration,
                    self._bridge_error(tool_call, str(exc)),
                    error_type=type(exc).__name__,
                ):
                    yield event
                return

        name = tool_call.name
        try:
            registered = self._tools.get(name)
        except KeyError as exc:
            if name == _HYPOTHESIS_REGISTRATION_TOOL:
                yield await self._emit_hypothesis_rejection(
                    tool_call, "TOOL_UNAVAILABLE"
                )
            async for event in self._record_tool_failure(
                tool_call, iteration, str(exc), error_type=type(exc).__name__
            ):
                yield event
            return
        if registered.exposure is ToolExposure.HIDDEN:
            if name == _HYPOTHESIS_REGISTRATION_TOOL:
                yield await self._emit_hypothesis_rejection(
                    tool_call, "TOOL_UNAVAILABLE"
                )
            async for event in self._record_tool_failure(
                tool_call,
                iteration,
                f"tool {name!r} is hidden and unavailable",
                error_type="ToolUnavailable",
                bridge_tool=bridge_tool,
                protocol_tool_name=protocol_tool_name,
            ):
                yield event
            return
        try:
            self._sensitive_paths.check_tool_call(name, tool_call.arguments)
        except SensitiveAccessError as exc:
            if name == _HYPOTHESIS_REGISTRATION_TOOL:
                yield await self._emit_hypothesis_rejection(
                    tool_call, "SENSITIVE_ACCESS_BLOCKED"
                )
            yield await self._emit(
                SensitiveAccessBlocked(
                    iteration=iteration,
                    tool_call_id=tool_call.id,
                    tool_name=name,
                    path=exc.path,
                )
            )
            self._conversation.add(
                ToolResultMessage(
                    tool_call_id=tool_call.id,
                    tool_name=protocol_tool_name,
                    content=f"Blocked access to sensitive file: {exc.path}",
                    is_error=True,
                )
            )
            return
        yield await self._emit(
            ToolRequested(
                iteration=iteration,
                tool_call_id=tool_call.id,
                tool_name=name,
                arguments=_event_tool_arguments(name, tool_call.arguments),
                bridge_tool=bridge_tool,
                underlying_tool=name if bridge_tool else None,
            )
        )
        if (
            registered.exposure is ToolExposure.DEFERRED
            and bridge_tool is None
            and self._tool_surface.config.mode == "progressive"
        ):
            if name == _HYPOTHESIS_REGISTRATION_TOOL:
                yield await self._emit_hypothesis_rejection(
                    tool_call, "DEFERRED_TOOL_REQUIRES_BRIDGE"
                )
            payload = json.dumps(
                {
                    "error": "deferred_tool_requires_bridge",
                    "tool": name,
                    "hint": "Use tool_search/tool_describe, then tool_call.",
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            async for event in self._record_tool_failure(
                tool_call,
                iteration,
                payload,
                error_type="DeferredToolDirectCall",
            ):
                yield event
            return

        permission = await self._permission.check(name, tool_call.arguments)
        if permission.decision is PermissionDecision.DENY:
            if name == _HYPOTHESIS_REGISTRATION_TOOL:
                yield await self._emit_hypothesis_rejection(
                    tool_call, "PERMISSION_DENIED"
                )
            async for event in self._deny_tool(
                tool_call,
                iteration,
                permission.reason,
                bridge_tool=bridge_tool,
                protocol_tool_name=protocol_tool_name,
            ):
                yield event
            return
        if permission.decision is PermissionDecision.ASK:
            yield await self._emit(
                ToolApprovalRequired(
                    iteration=iteration,
                    tool_call_id=tool_call.id,
                    tool_name=name,
                    arguments=_event_tool_arguments(name, tool_call.arguments),
                    reason=permission.reason,
                    bridge_tool=bridge_tool,
                    underlying_tool=name if bridge_tool else None,
                )
            )
            approved = (
                self._approval_handler is not None
                and await self._approval_handler.request_approval(
                    ApprovalRequest(name, tool_call.arguments, permission.reason)
                )
            )
            if not approved:
                if name == _HYPOTHESIS_REGISTRATION_TOOL:
                    yield await self._emit_hypothesis_rejection(
                        tool_call, "APPROVAL_REJECTED"
                    )
                async for event in self._deny_tool(
                    tool_call,
                    iteration,
                    "approval denied",
                    bridge_tool=bridge_tool,
                    protocol_tool_name=protocol_tool_name,
                ):
                    yield event
                return

        try:
            arguments = self._tools.validate_arguments(name, tool_call.arguments)
        except (ToolValidationError, KeyError) as exc:
            error = (
                self._deferred_validation_error(name, tool_call.arguments, str(exc))
                if bridge_tool
                else str(exc)
            )
            if name == _HYPOTHESIS_REGISTRATION_TOOL:
                yield await self._emit_hypothesis_rejection(
                    tool_call, "SCHEMA_VALIDATION_FAILED"
                )
            async for event in self._record_tool_failure(
                tool_call,
                iteration,
                error,
                error_type=type(exc).__name__,
                bridge_tool=bridge_tool,
                protocol_tool_name=protocol_tool_name,
            ):
                yield event
            return

        yield await self._emit(
            ToolStarted(
                iteration=iteration,
                tool_name=name,
                tool_call_id=tool_call.id,
                bridge_tool=bridge_tool,
                underlying_tool=name if bridge_tool else None,
            )
        )
        self._budget.record_tool_call()
        tool_started = time.monotonic()
        try:
            result = await self._tools.get(name).execute(arguments)
        except Exception as exc:
            observation = self._observation.apply(
                name, f"{type(exc).__name__}: {exc}"
            )
            if name == _HYPOTHESIS_REGISTRATION_TOOL:
                yield await self._emit_hypothesis_rejection(
                    tool_call, "TOOL_EXECUTION_FAILED"
                )
            async for event in self._record_tool_failure(
                tool_call,
                iteration,
                observation.content,
                (time.monotonic() - tool_started) * 1000,
                error_type=type(exc).__name__,
                bridge_tool=bridge_tool,
                protocol_tool_name=protocol_tool_name,
                observation=observation,
            ):
                yield event
            return

        duration_ms = (time.monotonic() - tool_started) * 1000
        # Granting and revoking a wait exemption are deliberately separate.
        # A trusted application result that names this workflow and reports an
        # explicit inactive/terminal state revokes an earlier exemption even
        # when the tool marks the result as an error (for example a public
        # collect that returned VALIDATION_FAILED with no usable evidence).
        # Errors never grant a new wait and never count as scientific progress;
        # results without a structured workflow identity or state (a bare
        # network/query exception) leave eligibility untouched.
        wait_signal = self._structured_wait_signal(name, result)
        if wait_signal is not None:
            wait_key, wait_kind = wait_signal
            if wait_kind == "active":
                if not result.is_error:
                    self._run_meta["controlled_wait_seen"] = True
                    self._run_meta["controlled_wait_this_iteration"] = True
                    self._active_wait_workflows.add(wait_key)
            else:
                self._active_wait_workflows.discard(wait_key)
        if not result.is_error:
            observation_key = self._observation_key(name, result, arguments)
            if (
                observation_key is not None
                and observation_key not in self._run_observed_keys
            ):
                self._run_observed_keys.add(observation_key)
                self._run_meta["new_observation_seen"] = True
                # Charge the bounded diagnostic budget per *new* observation,
                # counted per tool call (not per model iteration) so a batched
                # response with several probes pays for each probe.  A result
                # that also carries a trusted active wait is exempt: the wait
                # is operational, not diagnostic exploration.
                if wait_signal is not None and wait_signal[1] == "active":
                    self._run_meta["new_wait_observations"] = (
                        int(self._run_meta.get("new_wait_observations", 0)) + 1
                    )
                    self._run_meta["wait_observations"] = (
                        int(self._run_meta.get("wait_observations", 0)) + 1
                    )
                else:
                    self._run_meta["new_diagnostic_observations"] = (
                        int(
                            self._run_meta.get(
                                "new_diagnostic_observations", 0
                            )
                        )
                        + 1
                    )
        if result.is_error:
            observation = self._observation.apply(name, result.output)
            if name == _HYPOTHESIS_REGISTRATION_TOOL:
                yield await self._emit_hypothesis_rejection(
                    tool_call, "TOOL_REJECTED"
                )
            async for event in self._record_tool_failure(
                tool_call,
                iteration,
                observation.content,
                duration_ms,
                bridge_tool=bridge_tool,
                protocol_tool_name=protocol_tool_name,
                observation=observation,
            ):
                yield event
            return

        try:
            prepared_updates = self._prepare_state_updates(
                result.state_updates,
                tool_name=name,
                tool_call_id=tool_call.id,
            )
        except (ToolError, ValueError) as exc:
            observation = self._observation.apply(
                name, f"{type(exc).__name__}: {exc}"
            )
            if name == _HYPOTHESIS_REGISTRATION_TOOL:
                yield await self._emit_hypothesis_rejection(
                    tool_call, "STATE_UPDATE_VALIDATION_FAILED"
                )
            async for event in self._record_tool_failure(
                tool_call,
                iteration,
                observation.content,
                duration_ms,
                error_type=type(exc).__name__,
                bridge_tool=bridge_tool,
                protocol_tool_name=protocol_tool_name,
                observation=observation,
            ):
                yield event
            return

        self._record_innovation_meta(name, registered.namespace, result, bridge_tool)
        if prepared_updates:
            for update in prepared_updates:
                self._apply_state_update(update)
        observation = self._observation.apply(name, result.output)
        yield await self._emit(
            ToolCompleted(
                iteration=iteration,
                tool_name=name,
                tool_call_id=tool_call.id,
                output=observation.content,
                duration_ms=duration_ms,
                bridge_tool=bridge_tool,
                underlying_tool=name if bridge_tool else None,
                truncated=observation.truncated,
                original_chars=observation.original_chars,
                delivered_chars=observation.delivered_chars,
                redacted=observation.redacted,
            )
        )
        for update in prepared_updates:
            if isinstance(update, ScientificHypothesis):
                yield await self._emit(
                    HypothesisRegistered(
                        hypothesis_id=update.id,
                        candidate_id=update.candidate_id,
                        request_id=update.proposal.request_id,
                    )
                )
            elif isinstance(update, StructureDerivation):
                yield await self._emit(
                    StructureDerived(
                        derivation_id=update.id,
                        candidate_id=update.candidate_id,
                        parent_candidate_id=update.parent_candidate_id,
                        structure_hash=update.structure_hash,
                    )
                )
        if prepared_updates:
            state_summary = format_scientific_state(self._scientific)
            registered_updates = [
                update
                for update in prepared_updates
                if isinstance(update, ScientificHypothesis)
            ]
            if registered_updates:
                state_summary = json.dumps(
                    {
                        "material_hypothesis_count": len(
                            self._scientific.material_hypotheses
                        ),
                        "registered": [
                            {
                                "hypothesis_id": update.id,
                                "candidate_id": update.candidate_id,
                            }
                            for update in registered_updates
                        ],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            elif any(isinstance(update, StructureDerivation) for update in prepared_updates):
                state_summary = json.dumps(
                    {
                        "structure_derivation_count": len(
                            self._scientific.structure_derivations
                        ),
                        "derived": [
                            {
                                "derivation_id": update.id,
                                "candidate_id": update.candidate_id,
                                "structure_hash": update.structure_hash,
                            }
                            for update in prepared_updates
                            if isinstance(update, StructureDerivation)
                        ],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            yield await self._emit(
                ScientificStateUpdated(summary=state_summary)
            )
        self._conversation.add(
            ToolResultMessage(
                tool_call_id=tool_call.id,
                tool_name=protocol_tool_name,
                content=observation.content,
                is_error=False,
            )
        )

    def _record_innovation_meta(
        self,
        tool_name: str,
        namespace: str,
        result: ToolResult,
        bridge_tool: str | None,
    ) -> None:
        """Accumulate per-run innovation telemetry without new state objects."""
        if not self._run_meta:
            return
        if tool_name == "skill_view":
            skill = str(result.data.get("skill", ""))
            if skill and skill not in self._run_meta["skills_loaded"]:
                self._run_meta["skills_loaded"].append(skill)
            if result.data.get("source") == "qd-published":
                self._run_meta["published_skill_activity"].append({
                    "stage": "loaded", "skill": skill,
                    "bundle_sha256": result.data.get("bundle_sha256"),
                })
        if tool_name == "script_run" and result.data.get("published_skill"):
            self._run_meta["published_skill_activity"].append({
                "stage": "script_executed",
                "skill": result.data.get("published_skill"),
                "execution_id": result.data.get("execution_id"),
                "succeeded": result.data.get("skill_execution_succeeded"),
            })
        if namespace not in {"", "core", "skills"}:
            if tool_name and tool_name not in self._run_meta["scientific_tools_used"]:
                self._run_meta["scientific_tools_used"].append(tool_name)
        if bridge_tool == "tool_call" and tool_name:
            if tool_name not in self._run_meta["capability_escalations"]:
                self._run_meta["capability_escalations"].append(tool_name)
        gaps = result.data.get("evidence_gaps")
        if isinstance(gaps, list):
            for gap in gaps:
                text = str(gap)
                if text and text not in self._run_meta["evidence_gaps_identified"]:
                    self._run_meta["evidence_gaps_identified"].append(text)
        for update in result.state_updates:
            if isinstance(update, ScientificEvidence):
                self._run_meta["evidence_created"] += 1
                source = update.source
                if source and source not in self._run_meta["evidence_sources"]:
                    self._run_meta["evidence_sources"].append(source)

    async def _deny_tool(
        self,
        tool_call: ToolCall,
        iteration: int,
        reason: str,
        *,
        bridge_tool: str | None = None,
        protocol_tool_name: str | None = None,
    ) -> AsyncIterator[RuntimeEvent]:
        yield await self._emit(
            ToolPermissionDenied(
                iteration=iteration,
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                reason=reason,
                bridge_tool=bridge_tool,
                underlying_tool=tool_call.name if bridge_tool else None,
            )
        )
        self._conversation.add(
            ToolResultMessage(
                tool_call_id=tool_call.id,
                tool_name=protocol_tool_name or tool_call.name,
                content=f"permission denied: {reason}",
                is_error=True,
            )
        )

    async def _emit_hypothesis_rejection(
        self,
        tool_call: ToolCall,
        reason_code: HypothesisRegistrationReasonCode,
    ) -> RuntimeEvent:
        request_arguments = tool_call.arguments
        nested_arguments = request_arguments.get("arguments")
        if (
            request_arguments.get("name") == _HYPOTHESIS_REGISTRATION_TOOL
            and isinstance(nested_arguments, dict)
        ):
            request_arguments = nested_arguments
        request_id = _bounded_event_text(
            request_arguments.get("request_id", ""), limit=128
        )
        return await self._emit(
            HypothesisRegistrationRejected(
                request_id=request_id,
                reason_code=reason_code,
            )
        )

    async def _record_tool_failure(
        self,
        tool_call: ToolCall,
        iteration: int,
        error: str,
        duration_ms: float = 0.0,
        error_type: str | None = None,
        *,
        bridge_tool: str | None = None,
        protocol_tool_name: str | None = None,
        observation: Any | None = None,
    ) -> AsyncIterator[RuntimeEvent]:
        yield await self._emit(
            ToolFailed(
                iteration=iteration,
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                error=error,
                duration_ms=duration_ms,
                error_type=error_type,
                bridge_tool=bridge_tool,
                underlying_tool=tool_call.name if bridge_tool else None,
                truncated=bool(observation and observation.truncated),
                original_chars=(observation.original_chars if observation else None),
                delivered_chars=(observation.delivered_chars if observation else None),
                redacted=bool(observation and observation.redacted),
            )
        )
        self._conversation.add(
            ToolResultMessage(
                tool_call_id=tool_call.id,
                tool_name=protocol_tool_name or tool_call.name,
                content=error,
                is_error=True,
            )
        )

    def _bridge_error(self, tool_call: ToolCall, detail: str) -> str:
        payload = {
            "error": "invalid_tool_call_bridge",
            "detail": detail,
            "required": ["name", "arguments"],
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _deferred_validation_error(
        self, name: str, arguments: dict[str, object], detail: str
    ) -> str:
        definition = self._tools.definition(name)
        raw_required = definition.input_schema.get("required", [])
        required = raw_required if isinstance(raw_required, list) else []
        missing = [item for item in required if item not in arguments]
        payload = {
            "error": "missing_required_arguments" if missing else "validation_error",
            "tool": name,
            "missing": missing,
            "detail": detail,
            "parameter_help": compact_parameter_help(definition),
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _apply_state_update(self, update: Any) -> None:
        if isinstance(update, (Evidence, ScientificEvidence)):
            self._scientific.add_evidence(update)
        elif isinstance(update, ScientificClaim):
            self._scientific.add_claim(update)
        elif isinstance(update, CalculationRecord):
            self._scientific.add_calculation(update)
        elif isinstance(update, ScientificTask):
            self._scientific.add_task(update)
        elif isinstance(update, ScientificReferenceUpdate):
            self._apply_scientific_reference(self._scientific, update)
        elif isinstance(update, ScientificHypothesis):
            self._scientific.add_material_hypothesis(update)
        elif isinstance(update, StructureDerivation):
            self._scientific.add_structure_derivation(update)
        elif isinstance(update, EvidenceAttestation):
            self._evidence_authority.attest(self._scientific, update)
        else:
            raise ToolError(f"unsupported scientific state update: {type(update).__name__}")

    @staticmethod
    def _apply_scientific_reference(
        state: ScientificState, update: ScientificReferenceUpdate
    ) -> None:
        """Apply resumable references without replacing authoritative ledgers.

        Field contract:

        * ``None`` means "not provided" and leaves the current pointer.
        * An explicit empty string/list clears that pointer (a completed
          workflow has no next stage and no outstanding gaps).
        * A change of ``workflow_id`` replaces the whole reference: earlier
          workflow A's pending stage/gaps must not survive into workflow B as
          if they belonged to it.  Explicitly provided B fields are then
          applied on top of the cleared reference.

        These pointers never assert that a stage ran; manifests and the
        ``JobRegistry`` remain the authoritative execution facts.
        """
        workflow_changed = (
            update.workflow_id is not None
            and update.workflow_id != state.workflow_id
        )
        if update.workflow_id is not None:
            state.workflow_id = update.workflow_id
        if workflow_changed:
            state.next_stage = ""
            state.evidence_gaps = []
        if update.next_stage is not None:
            state.next_stage = update.next_stage
        if update.evidence_gaps is not None:
            state.evidence_gaps = list(update.evidence_gaps)

    def _prepare_state_updates(
        self,
        updates: list[Any],
        *,
        tool_name: str,
        tool_call_id: str,
    ) -> list[Any]:
        """Validate a complete update batch against a shadow state before mutation."""

        shadow = self._scientific.model_copy(deep=True)
        self._evidence_authority.bind(shadow, replace=True)
        self._evidence_authority.copy_ledger(self._scientific, shadow)
        structure_updates = [
            update for update in updates if isinstance(update, StructureRegistration)
        ]
        trusted_structures: list[StructureDerivation] = []
        if structure_updates:
            if tool_name not in _STRUCTURE_REGISTRATION_TOOLS:
                raise ToolError(
                    "structure registration is only accepted from authoritative structure tools"
                )
            try:
                trusted_structures = verify_structure_derivations(
                    self._workspace,
                    [update.derivation for update in structure_updates],
                )
            except Exception as exc:
                raise ToolError(f"structure artifact verification failed: {exc}") from exc
        structure_cursor = 0
        prepared: list[Any] = []
        for update in updates:
            if isinstance(update, HypothesisRegistration):
                validate_proposal(update.proposal, shadow)
                if self._run_id is None:
                    raise ToolError("hypothesis registration requires an active runtime run")
                record = build_hypothesis(
                    update.proposal,
                    HypothesisOrigin(
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        session_id=self._session_id,
                        run_id=self._run_id,
                        provider=self._model.provider,
                        model=self._model.model,
                    ),
                )
                hypothesis_count = len(shadow.material_hypotheses)
                shadow.add_material_hypothesis(record)
                if len(shadow.material_hypotheses) > hypothesis_count:
                    prepared.append(record)
            elif isinstance(update, StructureRegistration):
                structure_record: StructureDerivation = self._prepare_structure_derivation(
                    update.derivation,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    trusted=trusted_structures[structure_cursor],
                )
                structure_cursor += 1
                existing_count = len(shadow.structure_derivations)
                shadow.add_structure_derivation(structure_record)
                if len(shadow.structure_derivations) > existing_count:
                    prepared.append(structure_record)
            elif isinstance(
                update,
                (
                    Evidence,
                    ScientificEvidence,
                    ScientificClaim,
                    CalculationRecord,
                    ScientificTask,
                    ScientificReferenceUpdate,
                ),
            ):
                self._apply_state_update_to(shadow, update)
                prepared.append(update)
                if isinstance(update, (Evidence, ScientificEvidence)):
                    attestation = self._evidence_attestation.attest(
                        tool=self._tools.get(tool_name),
                        evidence_id=update.id,
                        tool_call_id=tool_call_id,
                    )
                    self._apply_state_update_to(shadow, attestation)
                    prepared.append(attestation)
            else:
                raise ToolError(
                    f"unsupported scientific state update: {type(update).__name__}"
                )
        return prepared

    def _apply_state_update_to(self, state: ScientificState, update: Any) -> None:
        if isinstance(update, (Evidence, ScientificEvidence)):
            state.add_evidence(update)
        elif isinstance(update, ScientificClaim):
            state.add_claim(update)
        elif isinstance(update, CalculationRecord):
            state.add_calculation(update)
        elif isinstance(update, ScientificTask):
            state.add_task(update)
        elif isinstance(update, ScientificReferenceUpdate):
            self._apply_scientific_reference(state, update)
        elif isinstance(update, StructureDerivation):
            state.add_structure_derivation(update)
        elif isinstance(update, EvidenceAttestation):
            self._evidence_authority.attest(state, update)

    def _prepare_structure_derivation(
        self,
        record: StructureDerivation,
        *,
        tool_name: str,
        tool_call_id: str,
        trusted: StructureDerivation | None = None,
    ) -> StructureDerivation:
        """Revalidate tool data and replace its untrusted origin at runtime."""

        trusted = trusted or verify_structure_derivation(self._workspace, record)
        expected_candidate_id = f"cand_{trusted.structure_hash[:24]}"
        if trusted.candidate_id != expected_candidate_id:
            raise ToolError(
                "structure candidate_id must be derived from structure_hash"
            )
        parent_candidate_id = self._trusted_structure_parent(trusted)
        payload = trusted.model_dump(mode="python")
        payload["parent_candidate_id"] = parent_candidate_id
        lineage = dict(payload.get("lineage", {}))
        lineage["parent_candidate_id"] = parent_candidate_id
        payload["lineage"] = lineage
        output = self._workspace.resolve(trusted.output_path, must_exist=True)
        payload["origin"] = {
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "session_id": self._session_id,
            "run_id": self._run_id,
            "provider": self._model.provider,
            "model": self._model.model,
            "output_sha256": file_sha256(output),
        }
        return StructureDerivation.model_validate(payload)

    def _trusted_structure_parent(
        self, record: StructureDerivation
    ) -> str | None:
        """Resolve parentage only from bytes of a live registered artifact."""

        matches: set[str] = set()
        for previous in self._scientific.structure_derivations:
            if previous.output_path == record.output_path:
                continue
            try:
                path = self._workspace.resolve(previous.output_path, must_exist=True)
                if not path.is_file() or path.is_symlink():
                    continue
                if file_sha256(path) == record.input_sha256:
                    parsed, _ = load_structure_input(
                        self._workspace, previous.output_path
                    )
                    if canonical_structure_hash(parsed) != previous.structure_hash:
                        continue
                    matches.add(previous.candidate_id)
            except (OSError, ToolError, ValueError):
                continue
        matches.discard(record.candidate_id)
        if len(matches) > 1:
            raise ToolError(
                "structure parent is ambiguous across registered artifacts"
            )
        return next(iter(matches), None)

    async def _emit_budget(self, iteration: int) -> RuntimeEvent:
        return await self._emit(
            BudgetUpdated(
                model_calls=self._budget.model_calls,
                tool_calls=self._budget.tool_calls,
                iteration=iteration,
                input_tokens=self._budget.input_tokens,
                output_tokens=self._budget.output_tokens,
            )
        )

    async def _emit(self, event: RuntimeEvent) -> RuntimeEvent:
        event.session_id = self._session_id
        event.run_id = self._run_id
        for sink in self._event_sinks:
            result = sink(event)
            if inspect.isawaitable(result):
                await result
        return event
