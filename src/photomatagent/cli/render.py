"""Rich event consumer. Runtime never imports this module."""

from __future__ import annotations

import json

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from photomatagent.runtime.events import (
    LoopCompleted,
    LoopFailed,
    LoopStarted,
    ModelRequestStarted,
    ProviderFailed,
    RuntimeEvent,
    ScientificStateUpdated,
    TextDelta,
    ToolApprovalRequired,
    ToolCompleted,
    ToolFailed,
    ToolPermissionDenied,
    ToolRequested,
)


class ChatRenderer:
    def __init__(self, console: Console) -> None:
        self._console = console
        self._streaming_text = False

    def handle(self, event: RuntimeEvent) -> None:
        if isinstance(event, LoopStarted):
            subtitle = f"{event.provider} / {event.model}"
            self._console.print(
                Panel("Scientific Agent Runtime", subtitle=subtitle, border_style="cyan")
            )
            self._console.print(f"[bold cyan]Goal:[/] {event.goal}")
        elif isinstance(event, ModelRequestStarted):
            self._finish_text()
            self._console.print("[yellow]● Model thinking...[/]")
        elif isinstance(event, TextDelta):
            if not self._streaming_text:
                self._console.print(Text("\nAgent:", style="bold green"))
                self._streaming_text = True
            self._console.print(event.text, end="", markup=False, soft_wrap=True)
        elif isinstance(event, ToolRequested):
            self._finish_text()
            args = json.dumps(event.arguments, ensure_ascii=False)
            self._console.print(
                f"[magenta]● Calling[/] [bold]{event.tool_name}[/] [dim]{args}[/]"
            )
        elif isinstance(event, ToolApprovalRequired):
            self._console.print("\n[bold yellow]◆ Permission required[/]")
            self._console.print(f"  [bold]{event.tool_name}[/]")
            self._console.print(
                f"  [dim]{json.dumps(event.arguments, ensure_ascii=False)}[/]"
            )
        elif isinstance(event, ToolPermissionDenied):
            self._console.print(
                f"[yellow]– {event.tool_name} denied: {event.reason}[/]"
            )
        elif isinstance(event, ToolCompleted):
            self._console.print(
                f"[green]✓ {event.tool_name} completed[/] [dim]({event.duration_ms:.1f} ms)[/]"
            )
            if event.output:
                self._console.print(event.output, markup=False)
        elif isinstance(event, ToolFailed):
            self._console.print(f"[red]✗ {event.tool_name} failed: {event.error}[/]")
        elif isinstance(event, ScientificStateUpdated):
            self._console.print("[cyan]● Scientific state updated[/]")
        elif isinstance(event, ProviderFailed):
            self._finish_text()
            self._console.print(f"[red]provider failed: {event.error}[/]")
        elif isinstance(event, LoopCompleted):
            self._finish_text()
            self._console.print(
                f"[dim]loop finished: {event.reason} ({event.iterations} iterations, "
                f"{event.duration_ms / 1000:.2f}s)[/]"
            )
        elif isinstance(event, LoopFailed):
            self._finish_text()
            self._console.print(f"[red]loop failed: {event.error}[/]")

    def _finish_text(self) -> None:
        if self._streaming_text:
            self._console.print()
            self._streaming_text = False

    def flush_agent_text(self) -> None:
        self._finish_text()


def print_skill_list(console: Console, skills: list) -> None:
    for skill in skills:
        console.print(f"[bold]{skill.name}[/] — {skill.description or '(no description)'}")
        console.print(f"  [dim]{skill.path}[/]")
