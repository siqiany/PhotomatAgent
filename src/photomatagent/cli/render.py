"""Render RuntimeEvents to a Rich console. Kept out of the runtime package."""

from __future__ import annotations

import json

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from photomatagent.runtime.events import (
    LoopCompleted,
    LoopFailed,
    LoopStarted,
    ModelRequestStarted,
    RuntimeEvent,
    ScientificStateUpdated,
    TextDelta,
    ToolCompleted,
    ToolFailed,
    ToolRequested,
)


class ChatRenderer:
    """Accumulates model text and prints tool activity as events arrive."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self._agent_buffer: list[str] = []

    def handle(self, event: RuntimeEvent) -> None:
        if isinstance(event, LoopStarted):
            self._console.print(Panel("Scientific Agent Runtime", subtitle="Model: fake", border_style="cyan"))
            self._console.print(f"[bold cyan]Goal:[/] {event.goal}")
        elif isinstance(event, ModelRequestStarted):
            self._console.print("[yellow]● Model thinking...[/]")
        elif isinstance(event, TextDelta):
            self._agent_buffer.append(event.text)
        elif isinstance(event, ToolRequested):
            args = json.dumps(event.arguments, ensure_ascii=False)
            self._console.print(f"[magenta]● Calling[/] [bold]{event.tool_name}[/] [dim]{args}[/]")
        elif isinstance(event, ToolCompleted):
            self._console.print(f"[green]✓ {event.tool_name} completed[/]")
            self._console.print(Syntax(event.output, "json", theme="ansi_dark"))
        elif isinstance(event, ToolFailed):
            self._console.print(f"[red]✗ {event.tool_name} failed: {event.error}[/]")
        elif isinstance(event, ScientificStateUpdated):
            self._console.print("[cyan]● Scientific state updated[/]")
        elif isinstance(event, LoopCompleted):
            self.flush_agent_text()
            self._console.print(f"[dim]loop finished: {event.reason} ({event.iterations} iterations)[/]")
        elif isinstance(event, LoopFailed):
            self.flush_agent_text()
            self._console.print(f"[red]loop failed: {event.error}[/]")

    def flush_agent_text(self) -> None:
        text = "".join(self._agent_buffer).strip()
        self._agent_buffer.clear()
        if text:
            self._console.print(Text("\nAgent:", style="bold green"))
            self._console.print(text)


def print_tool_list(console: Console, tools: list[dict]) -> None:
    for tool in tools:
        console.print(f"[bold]{tool['name']}[/] — {tool['description']}")


def print_skill_list(console: Console, skills: list) -> None:
    for skill in skills:
        console.print(f"[bold]{skill.name}[/] — {skill.description or '(no description)'}")
        console.print(f"  [dim]{skill.path}[/]")
