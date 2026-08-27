"""``photomatagent loop``: run the Evidence-Guided Scientific Feedback Loop.

The scientific outer loop wraps the normal AgentRuntime maker with a
deterministic evaluator, feedback engine, convergence policy and stagnation
detection. This CLI prints the structured loop summary (section 44 of the
spec) after the loop terminates.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.table import Table

from photomatagent.cli.chat import build_runtime
from photomatagent.config import DotEnvConfig, LLMConfig, resolve_llm_config
from photomatagent.runtime.events import RuntimeEvent
from photomatagent.scientific.loop.controller import (
    ScientificLoopConfig,
    ScientificLoopController,
)
from photomatagent.scientific.loop.target import (
    TargetSpec,
    canonical_lwir_detector_target,
)

ApprovalMode = Literal["ask", "auto", "deny"]


def resolve_loop_target(
    *,
    goal: str | None,
    target_json: str | None,
    demo: bool,
) -> TargetSpec:
    """Resolve a machine-verifiable TargetSpec (mode A: explicit JSON or demo).

    A bare natural-language goal is not enough for the loop: without
    constraints the evaluator would have nothing deterministic to check, so
    the CLI requires an explicit target or the built-in demo.
    """
    if target_json is not None:
        try:
            target = TargetSpec.model_validate(json.loads(target_json))
        except Exception as exc:
            raise ValueError(f"invalid --target-json: {exc}") from exc
    elif demo:
        target = canonical_lwir_detector_target()
    else:
        raise ValueError(
            "loop requires a machine-verifiable target: pass --demo (built-in "
            "8-14 um LWIR photodetector target) or --target-json '<TargetSpec JSON>'; "
            "--goal may override the natural-language goal text"
        )
    if goal:
        target = target.model_copy(update={"goal": goal})
    return target


def _resolve_config(
    store: DotEnvConfig,
    provider: str | None,
    model: str | None,
) -> tuple[LLMConfig, bool]:
    if provider is None or provider == "fake":
        return LLMConfig(provider="fake", model=model or "fake"), False
    from photomatagent.cli.app import _prompt_config_value

    return resolve_llm_config(
        store, prompt=_prompt_config_value, provider=provider, model=model
    )


def run_scientific_loop_cli(
    *,
    goal: str | None,
    target_json: str | None,
    demo: bool,
    workspace: Path,
    approval: ApprovalMode,
    provider: str | None,
    model: str | None,
    max_rounds: int,
    patience: int,
    min_confidence: float,
    log_events: bool,
) -> int:
    console = Console()
    target = resolve_loop_target(goal=goal, target_json=target_json, demo=demo)
    config, created = _resolve_config(DotEnvConfig(workspace), provider, model)
    if created:
        console.print(f"[dim]created configuration file: {DotEnvConfig(workspace).path}[/]")
    console.print(f"[dim]loop target: {target.goal!r}[/]")

    runtime, logger = build_runtime(
        provider=config.provider,
        model=config.model,
        workspace_root=workspace,
        approval=approval,
        max_iterations=25,
        log_events=log_events,
    )
    sinks: list = []
    if logger is not None:
        sinks.append(logger.log)
    controller = ScientificLoopController(
        target=target,
        runtime=runtime,
        config=ScientificLoopConfig(
            max_rounds=max_rounds,
            patience=patience,
            min_confidence=min_confidence,
        ),
        event_sinks=sinks,
        session_id=logger.session_id if logger is not None else None,
    )
    try:
        asyncio.run(_drive(controller, console))
    except Exception as exc:
        console.print(f"[red]scientific loop failed: {type(exc).__name__}: {exc}[/]")
        return 1
    summary = controller.summary
    if summary is None:
        console.print("[red]loop terminated without a summary[/]")
        return 1
    _render_summary(console, summary)
    if logger is not None:
        console.print(f"[dim]events logged: {logger.events_path}[/]")
    return 0


async def _drive(controller: ScientificLoopController, console: Console) -> None:
    async for event in controller.run():
        _render_event(console, event)


def _render_event(console: Console, event: RuntimeEvent) -> None:
    kind = event.kind
    if kind == "candidate_proposed":
        console.print(
            f"[cyan]round {event.round}[/] proposed [bold]{event.label or event.candidate_id}[/]"
            f" ({event.generation_method or 'structured state'})"
        )
    elif kind == "candidate_evaluated":
        style = "green" if event.verdict == "PASS" else "yellow"
        console.print(
            f"[{style}]round {event.round}[/] {event.candidate_id}: "
            f"verdict={event.verdict} score={event.score:.3f}"
        )
    elif kind == "scientific_loop_decision_made":
        console.print(
            f"[dim]round {event.round} decision: {event.action}[/] " f"[dim]({event.reason})[/]"
        )


def _render_summary(console: Console, summary) -> None:
    status = summary.status
    style = {
        "SUCCESS": "green",
        "STALLED": "yellow",
        "INCONCLUSIVE": "yellow",
        "BUDGET_EXHAUSTED": "red",
    }.get(status, "yellow")
    console.print(f"\n[bold {style}]Scientific loop completed: {status}[/]")
    table = Table()
    table.add_column("Metric")
    table.add_column("Value")
    rows = [
        ("Rounds", str(summary.rounds)),
        ("Candidates evaluated", str(summary.candidate_count)),
        ("Best candidate", summary.best_candidate_id or "—"),
        ("Score", f"{summary.best_score:.3f}"),
        ("Termination reason", summary.termination_reason or "—"),
    ]
    for label, value in rows:
        table.add_row(label, value)
    console.print(table)

    evaluation = summary.final_evaluation
    if evaluation is not None:
        passed = [r.property for r in evaluation.constraint_results if r.result == "PASS"]
        if passed:
            console.print("[green]Hard constraints:[/]")
            for property_name in passed:
                console.print(f"  ✓ {property_name}")
        console.print(f"Evidence confidence: {evaluation.confidence:.3f}")

    if summary.unresolved_violations or summary.unresolved_evidence_gaps:
        console.print("[yellow]Unresolved:[/]")
        for violation in summary.unresolved_violations:
            console.print(f"  - {violation.short()}")
        for gap in summary.unresolved_evidence_gaps:
            console.print(f"  - {gap} evidence unavailable")
        if status != "SUCCESS":
            console.print("[dim]The runtime did not claim scientific success.[/]")