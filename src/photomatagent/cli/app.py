"""Typer command surface for PhotomatAgent."""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from photomatagent import __version__
from photomatagent.cli.chat import run_chat
from photomatagent.cli.render import print_skill_list
from photomatagent.config import DotEnvConfig, read_preferred_config, resolve_llm_config
from photomatagent.logging.event_logger import default_sessions_dir
from photomatagent.logging.session_stats import (
    latest_session,
    list_sessions,
    read_session_stats,
)
from photomatagent.models.factory import api_key_status
from photomatagent.models.fake import FakeModelProvider, FakeResponse, scripted_tool_call
from photomatagent.runtime.budget import BudgetState
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import AllowAllPolicy
from photomatagent.scientific.state import ScientificState
from photomatagent.skills.loader import SkillLoader
from photomatagent.tools.factory import create_default_registry
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
            max_iterations=10,
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
    max_iterations: int = typer.Option(10, "--max-iterations", min=1),
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
    table = Table("name", "description")
    for tool in registry.list_tools():
        table.add_row(tool.name, tool.description)
    console.print(table)


skills_app = typer.Typer(help="Inspect available skills.")
app.add_typer(skills_app, name="skills")


@skills_app.command("list")
def skills_list() -> None:
    loader = SkillLoader()
    skills = loader.load_all()
    console.print(f"[dim]{len(skills)} skill(s) from {loader.skills_dir}[/]")
    print_skill_list(console, skills)


sessions_app = typer.Typer(help="Inspect JSONL session traces.")
app.add_typer(sessions_app, name="sessions")


@sessions_app.command("list")
def sessions_list() -> None:
    sessions = list_sessions()
    table = Table("session", "provider", "model", "iterations", "duration")
    for session in sessions:
        stats = read_session_stats(session)
        table.add_row(
            session.name,
            stats.provider,
            stats.model,
            str(stats.iterations),
            f"{stats.duration_seconds:.2f}s",
        )
    console.print(table)


@sessions_app.command("stats")
def sessions_stats(target: str = typer.Argument("latest")) -> None:
    session = latest_session() if target == "latest" else default_sessions_dir() / target
    if session is None or not (session / "events.jsonl").is_file():
        raise typer.BadParameter(f"session not found: {target}")
    stats = read_session_stats(session)
    table = Table("Session Statistics", "Value")
    rows = [
        ("Session", session.name),
        ("Provider", stats.provider),
        ("Model", stats.model),
        ("Iterations", stats.iterations),
        ("Model calls", stats.model_calls),
        ("Tool calls", stats.tool_calls),
        ("Tool failures", stats.tool_failures),
        ("Permission denied", stats.permission_denials),
        ("Input tokens", stats.input_tokens),
        ("Output tokens", stats.output_tokens),
        ("Duration", f"{stats.duration_seconds:.3f}s"),
    ]
    for label, value in rows:
        table.add_row(str(label), str(value))
    console.print(table)


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
                [scripted_tool_call("echo", {"text": "hello"}), FakeResponse(text="ok")]
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
