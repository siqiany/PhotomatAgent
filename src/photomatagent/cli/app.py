"""photomatagent CLI entry point (Typer)."""

from __future__ import annotations

import asyncio
import shutil
import sys

import typer
from rich.console import Console
from rich.table import Table

from photomatagent import __version__
from photomatagent.cli.chat import build_runtime, run_chat
from photomatagent.cli.render import print_skill_list, print_tool_list
from photomatagent.logging.event_logger import default_sessions_dir
from photomatagent.models.fake import FakeModelProvider, FakeResponse, scripted_tool_call
from photomatagent.runtime.budget import BudgetState
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import AllowAllPolicy
from photomatagent.scientific.state import ScientificState
from photomatagent.skills.loader import SkillLoader
from photomatagent.tools.factory import create_default_registry

app = typer.Typer(
    help="PhotomatAgent: a minimal, extensible Scientific Agent Runtime for materials science.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def chat(
    goal: str | None = typer.Option(
        None, "--goal", "-g", help="Run a single goal non-interactively, then exit."
    ),
    approval: str = typer.Option(
        "ask", "--approval", help="Tool approval mode: ask | auto | deny."
    ),
    max_iterations: int = typer.Option(10, "--max-iterations", min=1),
    log_events: bool = typer.Option(True, "--log-events/--no-log-events"),
) -> None:
    """Interactive scientific chat session (event-stream consumer)."""
    if approval not in {"ask", "auto", "deny"}:
        raise typer.BadParameter("--approval must be ask | auto | deny")
    asyncio.run(
        run_chat(
            approval=approval,  # type: ignore[arg-type]
            max_iterations=max_iterations,
            log_events=log_events,
            goal=goal,
        )
    )


tools_app = typer.Typer(help="Inspect the tool registry.")
app.add_typer(tools_app, name="tools")


@tools_app.command("list")
def tools_list() -> None:
    """List tools registered in the default registry."""
    registry = create_default_registry(ScientificState())
    table = Table("name", "description")
    for tool in registry.list_tools():
        table.add_row(tool.name, tool.description)
    console.print(table)


skills_app = typer.Typer(help="Inspect available skills.")
app.add_typer(skills_app, name="skills")


@skills_app.command("list")
def skills_list() -> None:
    """List skills found in the skills directory."""
    loader = SkillLoader()
    skills = loader.load_all()
    console.print(f"[dim]{len(skills)} skill(s) from {loader.skills_dir}[/]")
    print_skill_list(console, skills)


@app.command()
def doctor() -> None:
    """Check the environment and run a quick loop smoke test."""
    checks: list[tuple[str, bool, str]] = []
    checks.append(("python >= 3.12", sys.version_info >= (3, 12), sys.version.split()[0]))
    checks.append(("uv available", shutil.which("uv") is not None, shutil.which("uv") or "not found"))

    loader = SkillLoader()
    skills = loader.load_all()
    checks.append(("skills dir found", len(skills) > 0, str(loader.skills_dir)))

    sessions_dir = default_sessions_dir()
    try:
        sessions_dir.mkdir(parents=True, exist_ok=True)
        writable = True
    except OSError:
        writable = False
    checks.append(("sessions dir writable", writable, str(sessions_dir)))

    registry = create_default_registry(ScientificState())
    checks.append(
        ("default tools registered", len(registry.list_tools()) > 0,
         ", ".join(t.name for t in registry.list_tools()))
    )

    loop_ok, loop_note = _smoke_loop()
    checks.append(("agent loop smoke test", loop_ok, loop_note))

    table = Table("check", "status", "detail")
    for name, ok, detail in checks:
        table.add_row(name, "[green]PASS[/]" if ok else "[red]FAIL[/]", detail)
    console.print(table)
    console.print(f"[dim]photomatagent v{__version__}[/]")


def _smoke_loop() -> tuple[bool, str]:
    """Run one scripted tool-calling loop; True if it completes."""
    try:
        scientific = ScientificState()
        registry = create_default_registry(scientific)
        model = FakeModelProvider(
            [
                scripted_tool_call("echo", {"text": "hello"}),
                FakeResponse(text="Smoke test passed."),
            ]
        )
        runtime = AgentRuntime(
            model=model,
            tools=registry,
            scientific_state=scientific,
            permission_policy=AllowAllPolicy(),
            budget=BudgetState(max_iterations=5),
        )
        events = asyncio.run(_collect(runtime))
        if not any(e.kind == "loop_completed" for e in events):
            return False, "loop did not complete"
        if not any(e.kind == "tool_completed" for e in events):
            return False, "tool did not complete"
        return True, "loop -> tool -> loop -> completed"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


async def _collect(runtime: AgentRuntime) -> list:
    return [e async for e in runtime.run("smoke test")]


def main() -> None:
    app()


if __name__ == "__main__":
    main()
