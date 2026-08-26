"""Render initial/system context around a ContextEngine-selected message set.

Cache-friendly layout: the system message is intentionally static for the whole
session (base instructions + skill index + capability manifest). Everything that
changes between loop iterations -- the scientific state and the derived
investigation ledger -- is appended as a single trailing user message so the
provider's prompt-cache prefix (system + conversation history) stays stable and
only the final "latest state" line is re-processed.
"""

from __future__ import annotations

from photomatagent.models.types import ModelMessage, SystemMessage, UserMessage
from photomatagent.runtime.state import ConversationState
from photomatagent.scientific.evidence import Evidence
from photomatagent.scientific.state import ScientificState
from photomatagent.skills.loader import SkillLoader


SYSTEM_PROMPT = """You are PhotomatAgent, a scientific agent runtime for materials science research.
You help scientists investigate materials, especially for infrared photodetection.
You can call tools to inspect state and run mock scientific calculations.
Be concise, cite evidence from your scientific state, and mark uncertainty explicitly.
Before invoking another tool, check whether current observations already support a reliable answer.
Use another tool only to resolve a meaningful uncertainty, verify a material claim, or obtain
information required to complete the task. Do not gather evidence solely for completeness.

File organization (mandatory):
- Deliver every user-facing output (final reports, figures, data files, artifacts) under a
  newly created subfolder in user_output/ named after the task or deliverable, e.g.
  user_output/<task-name>/. Create the folder with your file tools before writing deliverables.
- Keep every intermediate/scratch file you must generate while working but the user does not
  need (temporary inputs, staging, downloads, transfer scratch, per-step logs) under tmp/.
- Never write intermediate or temporary files into user_output/, and never scatter
  user-facing deliverables next to source files or inside tmp/.
"""


def format_skill_index(loader: SkillLoader) -> str:
    entries = loader.load_index()
    if not entries:
        return "(none)"
    lines: list[str] = []
    for entry in entries:
        metadata = ", ".join(part for part in (entry.category, *entry.tags) if part)
        suffix = f" [{metadata}]" if metadata else ""
        lines.append(f"{entry.name}{suffix}\n  {entry.description}")
    return "\n".join(lines)


def format_scientific_state(state: ScientificState) -> str:
    """Render the scientific state as plain text for the model."""
    lines: list[str] = []
    lines.append(f"Goal: {state.goal or '(none yet)'}")
    if state.hypotheses:
        lines.append("Hypotheses:")
        lines += [f"- {h}" for h in state.hypotheses]
    if state.claims:
        lines.append("Claims:")
        for claim in state.claims:
            lines.append(
                f"- [{claim.status}] {claim.statement} (confidence={claim.confidence}, "
                f"evidence={claim.supporting_evidence})"
            )
    if state.evidence:
        lines.append("Evidence:")
        for ev in state.evidence:
            if isinstance(ev, Evidence):
                lines.append(
                    f"- ({ev.type} from {ev.source}) {ev.content} "
                    f"(confidence={ev.confidence})"
                )
            else:
                lines.append(
                    f"- ({getattr(ev, 'source_type', 'observation')} from "
                    f"{getattr(ev, 'source', 'unknown')}) "
                    f"{getattr(ev, 'summary', '')} ({getattr(ev, 'property', '')} "
                    f"= {getattr(ev, 'value', '')} {getattr(ev, 'unit', '')})"
                )
    if state.calculations:
        lines.append("Calculations:")
        for calc in state.calculations:
            lines.append(
                f"- [{calc.status}] {calc.task_type} on {calc.input_reference} "
                f"-> {calc.output_reference}"
            )
    if state.open_questions:
        lines.append("Open questions:")
        lines += [f"- {q}" for q in state.open_questions]
    if state.contradictions:
        lines.append("Contradictions:")
        lines += [f"- {c}" for c in state.contradictions]
    if state.pending_tasks:
        lines.append("Pending tasks:")
        for task in state.pending_tasks:
            lines.append(f"- [{task.status}] {task.task_id} ({task.backend})")
    return "\n".join(lines)


class ContextBuilder:
    """Build the model context from conversation + scientific state."""

    def __init__(self, skill_loader: SkillLoader | None = None) -> None:
        self.skill_loader = skill_loader or SkillLoader()

    def build(
        self,
        conversation: ConversationState,
        scientific: ScientificState,
        *,
        capability_manifest: str = "",
    ) -> list[ModelMessage]:
        return self.build_messages(
            conversation.messages,
            scientific,
            capability_manifest=capability_manifest,
        )

    def build_messages(
        self,
        messages: list[ModelMessage],
        scientific: ScientificState,
        *,
        capability_manifest: str = "",
        investigation_state: str = "",
        compaction_state: object | None = None,
    ) -> list[ModelMessage]:
        """Return [static system, compaction?, conversation..., latest state].

        The scientific state and bounded investigation ledger are appended as the
        final message instead of being baked into the system prompt. That keeps
        the system + conversation prefix byte-identical between loop iterations,
        preserving provider prompt-cache hits while only the trailing snapshot is
        updated.
        """
        scientific_section = format_scientific_state(scientific)
        skill_index = format_skill_index(self.skill_loader)
        capability_section = capability_manifest or "(no deferred capabilities)"
        system = SystemMessage(
            content=(
                f"{SYSTEM_PROMPT}\n\n--- Available skills (index only) ---\n{skill_index}"
                "\nUse skill_view(name[, path]) to load a skill or one reference when needed."
                f"\n\n--- Deferred capability manifest ---\n{capability_section}"
            ),
        )
        latest_state = UserMessage(
            content=(
                "--- Current scientific state (latest snapshot; supersedes any earlier "
                "state in this conversation) ---\n"
                f"{scientific_section}"
                "\n\n--- Investigation state (bounded, derived) ---\n"
                f"{investigation_state or '(none yet)'}"
            ),
        )
        compacted: list[ModelMessage] = []
        if compaction_state is not None:
            # Local import prevents ContextBuilder and ContextEngine from forming
            # a module-import cycle while keeping rendering in this one seam.
            from photomatagent.runtime.context_engine import (
                CompactionState,
                format_compaction_state,
            )

            state = CompactionState.model_validate(compaction_state)
            compacted.append(SystemMessage(content=format_compaction_state(state)))
        return [system, *compacted, *messages, latest_state]
