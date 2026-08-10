"""Typer command surface for PhotomatAgent."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from photomatagent import __version__
from photomatagent.cli.chat import run_chat
from photomatagent.cli.render import print_skill_list
from photomatagent.config import DotEnvConfig, read_preferred_config, resolve_llm_config
from photomatagent.experiments.compare import compare_summaries
from photomatagent.experiments.loader import (
    ExperimentConfigError,
    load_experiment_config,
)
from photomatagent.experiments.runner import run_experiment
from photomatagent.experiments.storage import (
    load_experiment_summary,
    save_experiment,
)
from photomatagent.logging.event_logger import default_sessions_dir
from photomatagent.logging.session_stats import (
    latest_session,
    list_sessions,
    read_session_stats,
)
from photomatagent.scientific.capabilities.status import (
    format_status_table,
    probe_all_capabilities,
)
from photomatagent.models.factory import api_key_status
from photomatagent.models.fake import FakeModelProvider, FakeResponse, scripted_tool_call
from photomatagent.runtime.budget import BudgetState
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import AllowAllPolicy
from photomatagent.observability.replay import build_replay
from photomatagent.observability.trace import TraceError, load_trace
from photomatagent.scientific.state import ScientificState
from photomatagent.skills.loader import SkillLoader
from photomatagent.tools.factory import create_default_registry
from photomatagent.tools.exposure import ToolExposure
from photomatagent.tools.surface import ToolSurfacePlanner
from photomatagent.workspace import Workspace

app = typer.Typer(
    help="PhotomatAgent: an explicit Scientific Agent Runtime for materials science.",
    no_args_is_help=False,
)
console = Console()


@app.callback(invoke_without_command=True)
def default_command(ctx: typer.Context) -> None:
    """Start configured chat when no subcommand is supplied."""
    if ctx.invoked_subcommand is None:
        _launch_chat(
            goal=None,
            provider=None,
            model=None,
            workspace=Path.cwd(),
            approval="ask",
            max_iterations=25,
            log_events=True,
        )


@app.command()
def chat(
    goal: str | None = typer.Option(None, "--goal", "-g"),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="Override .env preference: fake | openai | anthropic",
    ),
    model: str | None = typer.Option(None, "--model"),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False),
    approval: str = typer.Option("ask", "--approval", help="ask | auto | deny"),
    max_iterations: int = typer.Option(25, "--max-iterations", min=1),
    log_events: bool = typer.Option(True, "--log-events/--no-log-events"),
) -> None:
    """Start an interactive or one-goal scientific agent session."""
    _launch_chat(
        goal=goal,
        provider=provider,
        model=model,
        workspace=workspace,
        approval=approval,
        max_iterations=max_iterations,
        log_events=log_events,
    )


def _launch_chat(
    *,
    goal: str | None,
    provider: str | None,
    model: str | None,
    workspace: Path,
    approval: str,
    max_iterations: int,
    log_events: bool,
) -> None:
    if approval not in {"ask", "auto", "deny"}:
        raise typer.BadParameter("--approval must be ask | auto | deny")
    try:
        store = DotEnvConfig(workspace)
        config, created = resolve_llm_config(
            store,
            prompt=_prompt_config_value,
            provider=provider,
            model=model,
        )
        if created:
            console.print(f"[dim]已创建配置文件：{store.path}[/]")
        summary = f"{config.provider} / {config.model}"
        if config.base_url:
            summary += f" / {config.base_url}"
        console.print(f"[dim]LLM 配置：{summary}（API Key 已隐藏）[/]")
        asyncio.run(
            run_chat(
                provider=config.provider,
                model=config.model,
                workspace_root=workspace,
                approval=approval,  # type: ignore[arg-type]
                max_iterations=max_iterations,
                log_events=log_events,
                goal=goal,
            )
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _prompt_config_value(
    message: str, secret: bool, default: str | None
) -> str:
    return str(
        typer.prompt(
            message,
            hide_input=secret,
            default=default,
            show_default=default is not None,
        )
    ).strip()


@app.command("configure")
def configure_llm(
    provider: str | None = typer.Option(
        None, "--provider", help="openai | anthropic"
    ),
    model: str | None = typer.Option(None, "--model"),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False),
) -> None:
    """Create or complete workspace LLM configuration without starting a chat."""
    try:
        store = DotEnvConfig(workspace)
        config, created = resolve_llm_config(
            store,
            prompt=_prompt_config_value,
            provider=provider,
            model=model,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    action = "已创建" if created else "已更新"
    console.print(f"[green]{action}配置：{store.path}[/]")
    console.print(f"Provider: {config.provider}")
    console.print(f"Model: {config.model}")
    if config.base_url:
        console.print(f"Base URL: {config.base_url}")
    console.print("API Key: configured（不会显示明文）")


tools_app = typer.Typer(help="Inspect the local tool registry.")
app.add_typer(tools_app, name="tools")


@tools_app.command("list")
def tools_list(
    workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False),
) -> None:
    registry = create_default_registry(ScientificState(), Workspace(workspace))
    table = Table("name", "namespace", "exposure", "description")
    for tool in registry.list_tools():
        table.add_row(tool.name, tool.namespace, tool.exposure.value, tool.description)
    console.print(table)


@tools_app.command("surface")
def tools_surface(
    workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False),
) -> None:
    """Show the registered universe and current model-visible tool surface."""
    registry = create_default_registry(ScientificState(), Workspace(workspace))
    surface = ToolSurfacePlanner(registry).plan()
    console.print(Panel("Tool Surface", border_style="cyan"))
    for exposure in ToolExposure:
        console.print(f"\n[bold]{exposure.value.title()}[/]")
        tools = registry.tools_for_exposure(exposure)
        if tools:
            for tool in tools:
                console.print(f"- {tool.name}")
        else:
            console.print("[dim]- none[/]")
    stats = surface.stats
    table = Table("Estimated Context Cost", "Value")
    for label, value in [
        ("Registered tools", stats.registered_tools),
        ("Direct tools", stats.direct_tools),
        ("Deferred tools", stats.deferred_tools),
        ("Hidden tools", stats.hidden_tools),
        ("Model-visible schema", f"~{stats.estimated_visible_schema_tokens} tokens"),
        ("Direct schemas", f"~{stats.estimated_direct_schema_tokens} tokens"),
        ("Bridge schemas", f"~{stats.estimated_bridge_schema_tokens} tokens"),
        ("Manifest", f"~{stats.estimated_manifest_tokens} tokens"),
        (
            "Deferred schemas avoided / call",
            f"~{stats.estimated_avoided_tokens} tokens",
        ),
    ]:
        table.add_row(label, str(value))
    console.print(table)
    console.print("[dim]All token figures above are chars/4 estimates.[/]")


@tools_app.command("search")
def tools_search(
    query: str = typer.Argument(...),
    limit: int = typer.Option(5, "--limit", min=1, max=20),
    namespace: str | None = typer.Option(None, "--namespace"),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False),
) -> None:
    """Debug deferred-tool BM25 search without making a model request."""
    registry = create_default_registry(ScientificState(), Workspace(workspace))
    matches = ToolSurfacePlanner(registry).catalog.search(
        query, limit=limit, namespace=namespace
    )
    table = Table("name", "namespace", "score", "description", "required")
    for match in matches:
        table.add_row(
            match.entry.name,
            match.entry.namespace,
            f"{match.score:.3f}",
            match.entry.short_description,
            ", ".join(match.entry.required_parameters) or "—",
        )
    console.print(table)


skills_app = typer.Typer(help="Inspect available skills.")
app.add_typer(skills_app, name="skills")


@skills_app.command("list")
def skills_list(
    sources: bool = typer.Option(False, "--sources", help="Show source roots and licenses."),
) -> None:
    loader = SkillLoader()
    skills = loader.load_index()
    source_names = ", ".join(source.name for source in loader.sources)
    console.print(
        f"[dim]{len(skills)} skill(s) from sources: {source_names or '(none)'}[/]"
    )
    if sources:
        table = Table("name", "description", "source", "license", "priority")
        for skill in skills:
            table.add_row(
                skill.name,
                skill.description,
                skill.source,
                skill.license or "—",
                str(skill.priority),
            )
        console.print(table)
    else:
        print_skill_list(console, skills)
    for diagnostic in loader.diagnostics:
        console.print(f"[yellow]skill diagnostic [{diagnostic.code}]: {diagnostic.message}[/]")


scientific_app = typer.Typer(help="Inspect scientific capability packs.")
app.add_typer(scientific_app, name="scientific")


@scientific_app.command("status")
def scientific_status(
    workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False),
) -> None:
    """Show dependency-probe status of every scientific capability pack."""
    infos = probe_all_capabilities(workspace=Workspace(workspace))
    table = Table("capability", "status", "version", "detail", "tools")
    for info in sorted(infos, key=lambda item: item.name):
        status_style = {
            "AVAILABLE": "[green]",
            "MISSING_DEPENDENCY": "[yellow]",
            "UNCONFIGURED": "[yellow]",
            "ERROR": "[red]",
        }.get(info.status.value, "")
        table.add_row(
            info.name,
            f"{status_style}{info.status.value}[/]",
            info.version or "—",
            info.detail,
            ", ".join(info.tools) or "—",
        )
    console.print(table)


mcp_app = typer.Typer(
    help="Inspect and operate MCP scientific gateway servers.",
    no_args_is_help=False,
)
app.add_typer(mcp_app, name="mcp")


def _mcp_manager(workspace: Path) -> Any:
    from photomatagent.mcp.manager import MCPServerManager

    return MCPServerManager(workspace=workspace)


@mcp_app.command("list")
def mcp_list(
    workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False),
) -> None:
    """Show discovered MCP server configurations (no connection attempt)."""
    manager = _mcp_manager(workspace)
    configs = manager.discovered()
    if not configs:
        console.print(
            "[yellow]no MCP servers configured[/]\n"
            "[dim]create .photomatagent/mcp.json (see .photomatagent/mcp.json.example)[/]"
        )
        return
    table = Table("Server", "Transport", "Enabled", "Namespace", "Command/URL", "Trust", "Tools")
    for cfg in configs:
        target = cfg.command or cfg.url or "—"
        table.add_row(
            cfg.name,
            cfg.transport,
            "yes" if cfg.enabled else "no",
            cfg.effective_namespace,
            str(target),
            cfg.trust_level,
            "—",
        )
    console.print(table)


@mcp_app.command("status")
def mcp_status(
    workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False),
) -> None:
    """Live-probe every configured server (connect + ping)."""
    manager = _mcp_manager(workspace)
    if not manager.discovered():
        console.print("[yellow]no MCP servers configured[/]")
        return
    rows = manager.live_status()
    table = Table("Server", "Transport", "Status", "Namespace", "Tools", "Latency", "Error")
    for row in rows:
        style = {
            "READY": "[green]",
            "UNCONFIGURED": "[yellow]",
            "DISABLED": "[dim]",
            "MISSING_DEPENDENCY": "[yellow]",
            "START_FAILED": "[red]",
            "UNHEALTHY": "[red]",
            "STOPPED": "[dim]",
        }.get(row.state.value, "")
        table.add_row(
            row.name,
            row.transport,
            f"{style}{row.state.value}[/]",
            row.namespace,
            str(row.tools),
            f"{row.latency_ms:.0f} ms" if row.latency_ms is not None else "—",
            (row.error or row.detail or "—")[:80],
        )
    console.print(table)


@mcp_app.command("doctor")
def mcp_doctor(
    workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False),
) -> None:
    """Deep diagnostics: command resolution, env refs, connect, health."""
    manager = _mcp_manager(workspace)
    if not manager.discovered():
        console.print("[yellow]no MCP servers configured[/]")
        return
    reports = manager.doctor()
    for report in reports:
        from photomatagent.redaction import redact_secrets

        safe = redact_secrets(report)
        console.print(Panel(json.dumps(safe, indent=2, ensure_ascii=False), title=f"mcp doctor: {safe['server']}"))


@mcp_app.command("tools")
def mcp_tools(
    server: str = typer.Argument(..., help="Configured server name"),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False),
) -> None:
    """List tools advertised by one server (connects if needed)."""
    manager = _mcp_manager(workspace)
    handle = manager.handles.get(server)
    if handle is None:
        raise typer.BadParameter(
            f"unknown server {server!r}; configured: {sorted(manager.handles)}"
        )
    manager._run_async(handle.start())
    if handle.state.value != "READY":
        console.print(f"[red]{handle.state.value}: {handle.detail or handle.last_error}[/]")
        raise typer.Exit(code=1)
    table = Table("Name", "Namespace", "Required", "Description")
    for spec in handle.remote_tools:
        table.add_row(
            f"{handle.config.effective_namespace}.{spec.name}",
            handle.config.effective_namespace,
            ", ".join(spec.required_parameters) or "—",
            (spec.description or "")[:120],
        )
    console.print(table)


@mcp_app.command("test")
def mcp_test(
    server: str = typer.Argument(..., help="Configured server name"),
    tool: str | None = typer.Option(None, "--tool", help="Remote tool to invoke"),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False),
) -> None:
    """Connect to one server, list tools, ping, and optionally invoke a tool."""
    manager = _mcp_manager(workspace)
    handle = manager.handles.get(server)
    if handle is None:
        raise typer.BadParameter(
            f"unknown server {server!r}; configured: {sorted(manager.handles)}"
        )
    manager._run_async(handle.restart())
    manager._run_async(handle.healthcheck())
    console.print(
        f"server={server} state={handle.state.value} "
        f"tools={len(handle.remote_tools)} "
        f"latency={handle.latency_ms if handle.latency_ms is not None else '—'} ms"
    )
    if handle.state.value != "READY":
        console.print(f"[red]{handle.detail or handle.last_error}[/]")
        raise typer.Exit(code=1)
    for spec in handle.remote_tools:
        console.print(f"  {spec.name}  required={', '.join(spec.required_parameters) or '—'}")
    if tool:
        spec = next((s for s in handle.remote_tools if s.name == tool), None)
        if spec is None:
            raise typer.BadParameter(f"server {server!r} has no tool {tool!r}")
        manager._run_async(handle.invoke(tool, {}))
        console.print(f"invoked {tool}: {handle.last_error or 'ok'}")


sessions_app = typer.Typer(help="Inspect JSONL session traces.")
app.add_typer(sessions_app, name="sessions")


@sessions_app.command("list")
def sessions_list() -> None:
    sessions = list_sessions()
    table = Table("Session ID", "Model", "Iterations", "Tools", "Duration", "Stop")
    for session in sessions:
        stats = read_session_stats(session)
        table.add_row(
            session.name,
            f"{stats.provider}/{stats.model}",
            str(stats.iterations),
            str(stats.tool_calls),
            f"{stats.duration_seconds:.1f}s",
            stats.stop_reason or "—",
        )
    console.print(table)


@sessions_app.command("show")
def sessions_show(target: str = typer.Argument("latest")) -> None:
    try:
        trace = load_trace(target)
        stats = read_session_stats(trace.session_dir)
    except TraceError as exc:
        raise typer.BadParameter(str(exc)) from exc
    starts = [event for event in trace.events if event.kind == "loop_started"]
    table = Table("Session Metadata", "Value")
    rows = [
        ("Session ID", stats.session_id),
        ("Trace", str(trace.events_path)),
        ("Schema versions", ", ".join(sorted({event.schema_version for event in trace.events}))),
        ("Provider", stats.provider),
        ("Model", stats.model),
        ("Runs", len(stats.run_ids) or len(starts)),
        ("Events", stats.event_count),
        ("Started", stats.started_at.isoformat() if stats.started_at else "—"),
        ("Ended", stats.ended_at.isoformat() if stats.ended_at else "—"),
        ("Workspace", getattr(starts[0], "workspace", "—") if starts else "—"),
        ("Goal(s)", "\n".join(getattr(event, "goal", "") for event in starts)),
    ]
    for label, value in rows:
        table.add_row(str(label), str(value))
    console.print(table)


@sessions_app.command("stats")
def sessions_stats(target: str = typer.Argument("latest")) -> None:
    try:
        stats = read_session_stats(target)
    except TraceError as exc:
        raise typer.BadParameter(str(exc)) from exc
    table = Table("Session Statistics", "Value")
    rows = [
        ("Session", stats.session_id),
        ("Provider", stats.provider),
        ("Model", stats.model),
        ("Iterations", stats.iterations),
        ("Model calls", stats.model_calls),
        ("Tool calls", stats.tool_calls),
        ("Unique tools", stats.unique_tools),
        ("Tool failures", stats.tool_failures),
        ("Repeated calls", stats.repeated_tool_calls),
        ("Consecutive repeats", stats.consecutive_repeat_count),
        ("Permission denied", stats.permission_denials),
        ("Tool failure rate", f"{stats.tool_failure_rate:.1%}"),
        ("Tools / iteration", _format_number(stats.tools_per_iteration)),
        ("Input tokens", _format_optional(stats.input_tokens)),
        ("Output tokens", _format_optional(stats.output_tokens)),
        ("Total tokens", _format_optional(stats.total_tokens)),
        ("Duration", f"{stats.duration_seconds:.3f}s"),
        ("Model latency", f"{stats.model_latency_seconds:.3f}s"),
        ("Tool latency", f"{stats.tool_latency_seconds:.3f}s"),
        ("Stop reason", stats.stop_reason or "—"),
        ("Registered tools", stats.registered_tools or "—"),
        ("Direct tools", stats.direct_tools or "—"),
        ("Deferred tools", stats.deferred_tools or "—"),
        ("Hidden tools", stats.hidden_tools),
        (
            "Direct schema tokens (estimated)",
            _format_optional(stats.direct_schema_estimated_tokens),
        ),
        (
            "Manifest tokens / call (estimated)",
            _format_optional(stats.manifest_estimated_tokens_per_call),
        ),
        (
            "Deferred schemas avoided / call (estimated)",
            _format_optional(
                stats.deferred_schemas_avoided_estimated_tokens_per_call
            ),
        ),
        (
            "Deferred schemas avoided total (estimated)",
            stats.cumulative_deferred_schemas_avoided_estimated_tokens,
        ),
        (
            "Bridge schema tokens (estimated)",
            _format_optional(stats.bridge_schema_estimated_tokens),
        ),
        ("tool_search calls", stats.tool_search_calls),
        ("tool_describe calls", stats.tool_describe_calls),
        ("tool_call bridge calls", stats.tool_call_bridge_calls),
    ]
    for label, value in rows:
        table.add_row(str(label), str(value))
    console.print(table)
    console.print("\n[bold]Loop anomalies:[/]")
    if stats.anomalies:
        for anomaly in stats.anomalies:
            console.print(f"- [yellow]{anomaly.code}[/]: {anomaly.detail}")
    else:
        console.print("[dim]- none[/]")
    console.print("\n[bold]Scientific trace:[/]")
    rows = [
        ("Skills loaded", ", ".join(stats.skills_loaded) or "—"),
        ("Scientific tools used", ", ".join(stats.scientific_tools_used) or "—"),
        ("Evidence created", str(stats.evidence_created)),
        ("Evidence sources", ", ".join(stats.evidence_sources) or "—"),
        ("Evidence gaps identified", ", ".join(stats.evidence_gaps_identified) or "—"),
        ("Capability escalations", ", ".join(stats.capability_escalations) or "—"),
    ]
    for label, value in rows:
        console.print(f"- {label}: {value}")


@sessions_app.command("context")
def sessions_context(target: str = typer.Argument("latest")) -> None:
    """Show working-context lifecycle diagnostics from an immutable trace."""
    try:
        stats = read_session_stats(target)
    except TraceError as exc:
        raise typer.BadParameter(str(exc)) from exc
    table = Table("Context Lifecycle", "Value")
    rows = [
        ("Session", stats.session_id),
        ("Working tokens (last, estimated)", _format_optional(stats.last_working_context_tokens)),
        ("Working tokens (peak, estimated)", _format_optional(stats.peak_working_context_tokens)),
        ("Working chars (last)", _format_optional(stats.last_working_context_chars)),
        ("Durable JSONL transcript chars", stats.durable_transcript_chars),
        ("Pruned tool outputs", stats.pruned_tool_results),
        ("Compaction count", stats.compaction_count),
        ("Compaction failures", stats.compaction_failures),
        (
            "Last compaction tokens before / after",
            f"{_format_optional(stats.last_compaction_tokens_before)} / "
            f"{_format_optional(stats.last_compaction_tokens_after)}",
        ),
        (
            "Last compaction chars before / after",
            f"{_format_optional(stats.last_compaction_chars_before)} / "
            f"{_format_optional(stats.last_compaction_chars_after)}",
        ),
    ]
    for label, value in rows:
        table.add_row(str(label), str(value))
    console.print(table)
    console.print("[dim]Token figures are chars/4 estimates unless provider usage says otherwise.[/]")


@sessions_app.command("replay")
def sessions_replay(
    target: str = typer.Argument("latest"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    try:
        replay = build_replay(load_trace(target))
    except TraceError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(Panel(f"Offline Trace Replay · {replay.session_id}", border_style="cyan"))
    for item in replay.items:
        if item.kind == "goal":
            console.print(f"[bold cyan]USER[/] {item.content}")
        elif item.kind == "iteration":
            console.rule(f"[bold]Iteration {item.iteration}")
        elif item.kind == "model":
            console.print("[yellow]→ MODEL[/]")
        elif item.kind == "tool":
            arguments = item.metadata.get("arguments", {})
            console.print(f"[magenta]→ TOOL[/] [bold]{item.label}[/]")
            console.print(json.dumps(arguments, ensure_ascii=False, indent=2), markup=False)
        elif item.kind == "tool_result":
            console.print(f"[green]← RESULT[/] {item.label}")
            if item.content:
                console.print(item.content, markup=False)
        elif item.kind == "final_response":
            console.print("[bold green]✓ FINAL RESPONSE[/]")
            console.print(item.content, markup=False)
        elif item.kind == "model_text":
            console.print("[green]MODEL TEXT[/]")
            console.print(item.content, markup=False)
        elif item.kind == "stop":
            console.print(f"[dim]STOP: {item.content}[/]")
        else:
            console.print(f"[red]✗ {item.label}: {item.content}[/]")
        if verbose and item.metadata:
            console.print_json(data=item.metadata)


experiments_app = typer.Typer(help="Run and compare deterministic loop experiments.")
app.add_typer(experiments_app, name="experiments")


@experiments_app.command("run")
def experiments_run(
    config_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False),
) -> None:
    try:
        experiment = load_experiment_config(config_path)
        provider, model = _resolve_experiment_model(experiment.variant.provider, experiment.variant.model, workspace)
        result = asyncio.run(
            run_experiment(
                experiment,
                provider=provider,
                model=model,
                workspace_root=workspace,
            )
        )
        path = save_experiment(result)
    except (ExperimentConfigError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    summary = result.summary
    table = Table("Experiment Summary", "Value")
    for label, value in [
        ("Experiment ID", result.experiment_id),
        ("Name", summary.name),
        ("Provider / model", f"{provider} / {model}"),
        ("Tasks", summary.tasks_total),
        ("Runtime completed", summary.tasks_completed),
        ("Expectations passed", summary.expectations_passed),
        ("Expectations failed", summary.expectations_failed),
        ("Unevaluated", summary.tasks_unevaluated),
        ("Avg iterations", f"{summary.average_iterations:.2f}"),
        ("Avg tool calls", f"{summary.average_tool_calls:.2f}"),
        ("Tool failure rate", f"{summary.tool_failure_rate:.1%}"),
        ("Repeated calls", summary.repeated_tool_calls),
        (
            "Estimated tool-schema tokens / call",
            _format_optional(summary.estimated_tool_schema_tokens_per_call),
        ),
        ("tool_search calls", summary.tool_search_calls),
        ("tool_describe calls", summary.tool_describe_calls),
        ("tool_call bridge calls", summary.tool_call_bridge_calls),
        ("Peak working context (estimated)", _format_optional(summary.peak_working_context_tokens)),
        ("Pruned tool results", summary.pruned_tool_results),
        ("Compaction count", summary.compaction_count),
        ("Compaction failures", summary.compaction_failures),
        ("Duration", f"{summary.duration_seconds:.3f}s"),
        ("Stored at", str(path)),
    ]:
        table.add_row(str(label), str(value))
    console.print(table)


@experiments_app.command("compare")
def experiments_compare(
    experiment_a: str = typer.Argument(...),
    experiment_b: str = typer.Argument(...),
) -> None:
    try:
        summary_a = load_experiment_summary(experiment_a)
        summary_b = load_experiment_summary(experiment_b)
        rows = compare_summaries(summary_a, summary_b)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    table = Table("Metric", summary_a.experiment_id, summary_b.experiment_id, "Delta (B-A)")
    for row in rows:
        table.add_row(
            row.metric,
            _format_optional(row.a),
            _format_optional(row.b),
            _format_delta(row.delta),
        )
    console.print(table)


def _resolve_experiment_model(
    provider: str | None, model: str | None, workspace: Path
) -> tuple[str, str]:
    config = read_preferred_config(
        DotEnvConfig(workspace), provider=provider, model=model
    )
    if config is None:
        raise ValueError("experiment variant must select a provider or workspace .env must configure one")
    if not config.model:
        raise ValueError(f"experiment provider {config.provider} has no configured model")
    if config.provider != "fake" and config.api_key_env is None:
        raise ValueError(f"experiment provider {config.provider} has no configured API key")
    return config.provider, config.model


def _format_optional(value: object | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _format_number(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _format_delta(value: float | int | None) -> str:
    if value is None:
        return "—"
    return f"{value:+.3f}" if isinstance(value, float) else f"{value:+d}"


@app.command()
def doctor(
    provider: str | None = typer.Option(
        None, "--provider", help="Override .env preference: fake | openai | anthropic"
    ),
    model: str | None = typer.Option(None, "--model"),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False),
) -> None:
    """Check local runtime prerequisites without exposing credentials."""
    store = DotEnvConfig(workspace)
    config_error = ""
    try:
        config = read_preferred_config(store, provider=provider, model=model)
    except ValueError as exc:
        config = None
        config_error = str(exc)
    configured_provider = config.provider if config else "missing"
    configured_model = config.model if config else ""
    configured_base_url = config.base_url if config else None
    key_status = api_key_status(configured_provider)
    key_ok = configured_provider == "fake" or (
        configured_provider in {"openai", "anthropic"} and key_status == "configured"
    )
    checks: list[tuple[str, bool, str]] = [
        ("python >= 3.12", sys.version_info >= (3, 12), sys.version.split()[0]),
        ("uv available", shutil.which("uv") is not None, shutil.which("uv") or "not found"),
        ("working directory", Path.cwd().is_dir(), str(Path.cwd())),
        (".env configuration", store.path.is_file(), str(store.path)),
        (
            "provider",
            configured_provider in {"fake", "openai", "anthropic"},
            config_error or configured_provider,
        ),
        ("model", bool(configured_model), configured_model or "missing"),
        (
            "Base URL",
            bool(configured_base_url) if configured_provider == "openai" else config is not None,
            configured_base_url or (
                "not required" if configured_provider in {"fake", "anthropic"} else "missing"
            ),
        ),
        (
            "API key",
            key_ok,
            key_status if configured_provider != "missing" else "missing",
        ),
    ]
    loader = SkillLoader()
    checks.append(("skills dir found", bool(loader.load_all()), str(loader.skills_dir)))
    sessions_dir = default_sessions_dir()
    try:
        sessions_dir.mkdir(parents=True, exist_ok=True)
        writable = True
    except OSError:
        writable = False
    checks.append(("sessions dir writable", writable, str(sessions_dir)))
    loop_ok, note = _smoke_loop()
    checks.append(("agent loop smoke test", loop_ok, note))

    table = Table("check", "status", "detail")
    for name, ok, detail in checks:
        table.add_row(name, "[green]PASS[/]" if ok else "[red]FAIL[/]", detail)
    console.print(table)
    console.print(f"[dim]photomatagent v{__version__}[/]")


def _smoke_loop() -> tuple[bool, str]:
    try:
        scientific = ScientificState()
        workspace = Workspace(Path.cwd())
        runtime = AgentRuntime(
            model=FakeModelProvider(
                [
                    scripted_tool_call("glob", {"pattern": "pyproject.toml"}),
                    FakeResponse(text="ok"),
                ]
            ),
            tools=create_default_registry(scientific, workspace),
            workspace=workspace,
            scientific_state=scientific,
            permission_policy=AllowAllPolicy(),
            budget=BudgetState(max_iterations=5),
        )
        events = asyncio.run(_collect(runtime))
        ok = any(event.kind == "loop_completed" for event in events) and any(
            event.kind == "tool_completed" for event in events
        )
        return ok, "model -> tool -> model -> completed" if ok else "incomplete"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


async def _collect(runtime: AgentRuntime) -> list:
    return [event async for event in runtime.run("smoke test")]


def main() -> None:
    app()


if __name__ == "__main__":
    main()
