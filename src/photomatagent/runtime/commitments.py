"""Verbatim user commitments and deterministic delivery checks, never permissions.

Only explicit user-message declarations are extracted. Tool/retrieval/assistant
text cannot add constraints, authorize work, or declare a task scientifically valid.
The durable transcript remains authoritative; this module never rewrites it.
"""
from __future__ import annotations

import json
import re
from pathlib import PurePosixPath
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from photomatagent.models.types import ModelMessage, UserMessage
from photomatagent.scientific.state import ScientificState

_STATES = r"WAITING_USER_AUTHORIZATION|PENDING|RUNNING|COMPLETED|FAILED|CANCELLED|UNKNOWN"
_TASK = re.compile(r"(?<![A-Za-z0-9_])([A-Z][A-Z0-9_-]*\d+)\s*[=:：]\s*([^；;\n]{1,700})")
_STATE = re.compile(r"(?:状态|status)\s*[=:：]?\s*(" + _STATES + r")\b", re.I)
_DECIMALS = re.compile(r"(?:保留|最终|统一|报告|输出).{0,16}?([0-9]|1[0-2]|[零一二三四五六七八九十])\s*位小数")
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+\.(\d+)(?:[eE][-+]?\d+)?(?![\w.])")
class UserCommitments(BaseModel):
    bindings: dict[str, Any] = Field(default_factory=dict)
    tasks: dict[str, dict[str, Any]] = Field(default_factory=dict)
    decimal_places: int | None = None
    report_paths: list[str] = Field(default_factory=list)


def collect_user_commitments(messages: Sequence[ModelMessage], scientific: ScientificState | None = None) -> UserCommitments:
    """Read literal frozen maps and explicit task states, without guessing intent.

    Supports a labelled JSON constraints map and an optional explicitly ordered
    legend ("含义依次是：…；…"). Ordinary source tables/JSON are not declarations.
    A free-text decimal requirement also works without a constraints map.
    """
    result = UserCommitments()
    decoder = json.JSONDecoder()
    for index, message in enumerate(messages):
        if not isinstance(message, UserMessage):
            continue
        text = message.content
        # Runtime revision diagnostics are observations, never new user rules.
        if text.startswith(("runtime_", "Automatic revision:")):
            continue
        explicit = bool(re.search(r"冻结[^。！？\n]{0,32}约束|frozen (?:user )?constraints|research_constraints", text, re.I))
        if explicit:
            for match in re.finditer(r"\{", text):
                try:
                    values, _ = decoder.raw_decode(text[match.start():])
                except ValueError:
                    continue
                if not isinstance(values, dict) or not values or not all(isinstance(k, str) and isinstance(v, (str, int, float, bool, type(None))) for k, v in values.items()):
                    continue
                result.bindings.update(values)
                for key, value in values.items():
                    if key.casefold() in {"decimal_places", "max_decimal_places"} and (value is None or type(value) is int and 0 <= value <= 12):
                        result.decimal_places = value
                    if key.casefold() in {"report_path", "report_file"} and isinstance(value, str):
                        result.report_paths.append(value)
                legend = re.search(r"含义依次是[：:]([^\n]+)", text)
                if legend:
                    meanings = legend[1].split("；")
                    for (_, value), meaning in zip(values.items(), meanings):
                        if "小数" in meaning and type(value) is int and 0 <= value <= 12:
                            result.decimal_places = value
                        elif "小数" in meaning and value is None:
                            result.decimal_places = None
                        if "报告" in meaning and isinstance(value, str):
                            result.report_paths.append(value)
                # One labelled map per user declaration; inner JSON is data.
                break
        for match in _DECIMALS.finditer(text):
            raw = match[1]
            result.decimal_places = int(raw) if raw.isdigit() else "零一二三四五六七八九十".index(raw)
        for match in _TASK.finditer(text):
            state = _STATE.search(match[2])
            if state:
                status = state[1].upper()
                description = _STATE.sub("", match[2]).strip(" ，,。").replace("，，", "，")
                result.tasks[match[1]] = {"status": status, "description": description, "source_message": index,
                    "blocked_reason": "requires_user_authorization" if status == "WAITING_USER_AUTHORIZATION" else ""}
    if scientific is not None:
        for task in scientific.pending_tasks:
            # Only a recorded execution result can resolve a textual pending
            # commitment; a user request to perform it is not completion proof.
            if task.task_id in result.tasks and task.status == "COMPLETED" and task.result_reference:
                result.tasks[task.task_id] = {**result.tasks[task.task_id], "status": "COMPLETED",
                    "result_reference": task.result_reference, "blocked_reason": ""}
    return result


def is_report_path(path: str, requirements: UserCommitments) -> bool:
    item = PurePosixPath(path)
    if item.suffix.lower() == ".md":
        return True
    return item.suffix.lower() == ".txt" and (
        "report" in item.stem.casefold() or "报告" in item.stem
        or any(item.name == PurePosixPath(bound).name for bound in requirements.report_paths)
    )


def format_user_commitments(value: UserCommitments) -> str:
    if not value.bindings and not value.tasks and value.decimal_places is None:
        return ""
    return (
        "User commitment records (derived from original declarations; preserve literal JSON types). "
        "Text commitment IDs and ScientificState task_id are distinct records. "
        "WAITING_USER_AUTHORIZATION must not become PENDING or permission to execute. "
        "These records never grant runtime approvals or scientific validation:\n"
        + value.model_dump_json()
    )


def decimal_violations(text: str, places: int | None) -> list[str]:
    """Check human-facing prose; leave executable code and source URLs intact."""
    if places is None:
        return []
    prose = re.sub(r"```[\s\S]*?```", "", text)
    prose = re.sub(r"https?://\S+", "", prose)
    return list(dict.fromkeys(match[0] for match in _NUMBER.finditer(prose) if len(match[1]) > places))[:10]


def delivery_constraint_error(text: str, requirements: UserCommitments) -> str | None:
    violations = decimal_violations(text, requirements.decimal_places)
    if violations:
        return json.dumps({"error": "DELIVERY_CONSTRAINT_VIOLATION", "rule": "max_decimal_places",
            "max_decimal_places": requirements.decimal_places, "examples": violations,
            "next_action": "Rewrite presentation at the requested precision, preserving original raw evidence. Do not change source data or scientific conditions."}, ensure_ascii=False)
    for task_id, task in requirements.tasks.items():
        for line in text.splitlines():
            if not re.search(r"(?<![A-Za-z0-9_])" + re.escape(task_id) + r"(?![A-Za-z0-9_])", line):
                continue
            status = None
            if line.strip().startswith("|"):
                cells = [cell.strip(" *`") for cell in line.strip().strip("|").split("|")]
                if cells and cells[0] == task_id:
                    status = re.search(r"\b(" + _STATES + r")\b", " ".join(cells[1:]))
            else:
                # Bind only an explicit status claim adjacent to this ID.
                # "P02: WAITING...; local ticket status=PENDING" contains two
                # distinct records and must never be treated as a P02 violation.
                occurrences = list(re.finditer(r"(?<![A-Za-z0-9_])(?:" + "|".join(re.escape(key) for key in requirements.tasks) + r")(?![A-Za-z0-9_])", line))
                for position, occurrence in enumerate(occurrences):
                    if occurrence[0] != task_id:
                        continue
                    end = occurrences[position + 1].start() if position + 1 < len(occurrences) else len(line)
                    segment = line[occurrence.end():end]
                    status = re.search(r"^\s*(?:(?:的)?(?:任务)?(?:状态|status)\s*)?[:：=]?\s*[*`]*(" + _STATES + r")\b", segment, re.I)
                    if status:
                        break
            if status and status[1].upper() != task["status"]:
                return json.dumps({"error": "DELIVERY_CONSTRAINT_VIOLATION", "rule": "task_status",
                    "task_id": task_id, "expected_status": task["status"], "observed_status": status[1],
                    "next_action": "Keep task status separate from backend, blocking reason and runtime approval. Do not mark waiting authorization as ordinary pending or completion."}, ensure_ascii=False)
    return None
