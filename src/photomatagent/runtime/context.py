"""ContextBuilder: fuses ConversationState + ScientificState into model context.

This is the single seam where future context engineering / compaction happens.
The first version injects the full scientific state as a system section.
"""

from __future__ import annotations

from photomatagent.runtime.state import ConversationState, Message
from photomatagent.scientific.state import ScientificState


SYSTEM_PROMPT = """You are PhotomatAgent, a scientific agent runtime for materials science research.
You help scientists investigate materials, especially for infrared photodetection.
You can call tools to inspect state and run mock scientific calculations.
Be concise, cite evidence from your scientific state, and mark uncertainty explicitly.
"""


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
            lines.append(
                f"- ({ev.type} from {ev.source}) {ev.content} (confidence={ev.confidence})"
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

    def build(self, conversation: ConversationState, scientific: ScientificState) -> list[Message]:
        scientific_section = format_scientific_state(scientific)
        system = Message(
            role="system",
            content=f"{SYSTEM_PROMPT}\n\n--- Current scientific state ---\n{scientific_section}",
        )
        return [system, *conversation.messages]
