"""Deterministic task expectations; no LLM-as-Judge."""

from __future__ import annotations

from photomatagent.experiments.models import (
    ExpectationCheck,
    Expectations,
    TaskEvaluation,
)
from photomatagent.observability.analyzer import SessionSummary


def evaluate_expectations(
    expectations: Expectations | None,
    *,
    answer: str,
    summary: SessionSummary,
) -> TaskEvaluation:
    if expectations is None or expectations.is_empty():
        return TaskEvaluation(status="UNEVALUATED")

    checks: list[ExpectationCheck] = []
    folded_answer = answer.casefold()
    used = set(summary.tools_used)
    for needle in expectations.answer_contains:
        passed = needle.casefold() in folded_answer
        checks.append(
            ExpectationCheck(
                name="answer_contains",
                passed=passed,
                detail=f"answer {'contains' if passed else 'does not contain'} {needle!r}",
            )
        )
    for needle in expectations.answer_not_contains:
        passed = needle.casefold() not in folded_answer
        checks.append(
            ExpectationCheck(
                name="answer_not_contains",
                passed=passed,
                detail=f"answer {'excludes' if passed else 'contains'} {needle!r}",
            )
        )
    for tool in expectations.tools_used:
        passed = tool in used
        checks.append(
            ExpectationCheck(
                name="tools_used",
                passed=passed,
                detail=f"tool {tool!r} {'was' if passed else 'was not'} used",
            )
        )
    for tool in expectations.tools_not_used:
        passed = tool not in used
        checks.append(
            ExpectationCheck(
                name="tools_not_used",
                passed=passed,
                detail=f"tool {tool!r} {'was not' if passed else 'was'} used",
            )
        )
    if expectations.max_tool_calls is not None:
        passed = summary.tool_calls <= expectations.max_tool_calls
        checks.append(
            ExpectationCheck(
                name="max_tool_calls",
                passed=passed,
                detail=f"{summary.tool_calls} <= {expectations.max_tool_calls}",
            )
        )
    if expectations.max_iterations is not None:
        passed = summary.iterations <= expectations.max_iterations
        checks.append(
            ExpectationCheck(
                name="max_iterations",
                passed=passed,
                detail=f"{summary.iterations} <= {expectations.max_iterations}",
            )
        )
    return TaskEvaluation(
        status="PASS" if all(check.passed for check in checks) else "FAIL",
        checks=checks,
    )
