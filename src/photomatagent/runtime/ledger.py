"""Small, derived investigation ledger for trajectory efficiency."""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from photomatagent.models.types import AssistantMessage, ModelMessage, ToolResultMessage
from photomatagent.redaction import redact_text


class WorkingLedger(BaseModel):
    """Bounded facts derived from the durable conversation, never persisted twice."""

    searched_queries: list[str] = Field(default_factory=list)
    inspected_paths: list[str] = Field(default_factory=list)
    executed_commands: list[str] = Field(default_factory=list)
    key_observations: list[str] = Field(default_factory=list)


def derive_working_ledger(
    messages: list[ModelMessage], *, max_chars: int = 1_200, max_items: int = 8
) -> WorkingLedger:
    calls: dict[str, tuple[str, dict[str, object]]] = {}
    searched: list[str] = []
    paths: list[str] = []
    commands: list[str] = []
    observations: list[str] = []

    for message in messages:
        if isinstance(message, AssistantMessage):
            for call in message.tool_calls:
                calls[call.id] = (call.name, call.arguments)
                if call.name in {"grep", "glob", "tool_search"}:
                    query = call.arguments.get("pattern") or call.arguments.get("query")
                    if query:
                        scope = call.arguments.get("path") or call.arguments.get("glob")
                        searched.append(
                            f'{query!s}' + (f" under {scope!s}" if scope else "")
                        )
                if call.name in {"read", "grep", "glob", "edit", "write"}:
                    path = call.arguments.get("path") or call.arguments.get("pattern")
                    if path:
                        paths.append(str(path))
                if call.name == "skill_view":
                    path = call.arguments.get("path") or call.arguments.get("name")
                    if path:
                        paths.append(f"skill:{path}")
                if call.name == "bash" and call.arguments.get("command"):
                    commands.append(redact_text(str(call.arguments["command"])))
        elif isinstance(message, ToolResultMessage):
            name, _ = calls.get(message.tool_call_id, (message.tool_name, {}))
            first_line = next(
                (line.strip() for line in message.content.splitlines() if line.strip()),
                "(empty output)",
            )
            status = "failed" if message.is_error else "succeeded"
            observations.append(f"{name} {status}: {first_line[:160]}")

    ledger = WorkingLedger(
        searched_queries=_bounded_unique(searched, max_items=max_items),
        inspected_paths=_bounded_unique(paths, max_items=max_items),
        executed_commands=_bounded_unique(commands, max_items=max_items, item_chars=180),
        key_observations=_bounded_unique(
            [redact_text(item) for item in observations],
            max_items=max_items,
            item_chars=200,
        ),
    )
    return _fit_ledger(ledger, max_chars=max_chars)


def format_working_ledger(ledger: WorkingLedger) -> str:
    sections: list[str] = []
    for label, values in (
        ("Already searched", ledger.searched_queries),
        ("Already inspected", ledger.inspected_paths),
        ("Already executed", ledger.executed_commands),
        ("Recent observations", ledger.key_observations),
    ):
        if values:
            sections.append(label + ":\n" + "\n".join(f"- {item}" for item in values))
    return "\n".join(sections) or "(none yet)"


def _bounded_unique(
    values: list[str], *, max_items: int, item_chars: int = 140
) -> list[str]:
    # Keep the latest occurrence of each normalized action.
    result: list[str] = []
    seen: set[str] = set()
    for value in reversed(values):
        normalized = " ".join(value.split())
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        result.append(normalized[:item_chars])
        if len(result) >= max_items:
            break
    return list(reversed(result))


def _fit_ledger(ledger: WorkingLedger, *, max_chars: int) -> WorkingLedger:
    while len(format_working_ledger(ledger)) > max_chars:
        candidates = [
            values
            for values in (
                ledger.key_observations,
                ledger.executed_commands,
                ledger.searched_queries,
                ledger.inspected_paths,
            )
            if values
        ]
        if not candidates:
            break
        max(candidates, key=lambda values: len(json.dumps(values))).pop(0)
    return ledger
