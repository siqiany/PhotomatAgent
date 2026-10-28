"""Rich event consumer. Runtime never imports this module."""

from __future__ import annotations

import json
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from photomatagent.runtime.events import (
    ContextCompactionCompleted,
    ContextCompactionFailed,
    ContextCompactionSkipped,
    ContextPruneCompleted,
    LoopCompleted,
    LoopFailed,
    LoopStarted,
    ModelRequestStarted,
    ProviderFailed,
    RuntimeEvent,
    ScientificStateUpdated,
    SensitiveAccessBlocked,
    TextDelta,
    ToolApprovalRequired,
    ToolCompleted,
    ToolFailed,
    ToolPermissionDenied,
    ToolRequested,
)


class ChatRenderer:
    def __init__(
        self,
        console: Console,
        *,
        pause_summary_provider: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self._console = console
        self._streaming_text = False
        # Deterministic, model-free wrap-up facts.  The runtime supplies them;
        # the renderer never invents a scientific conclusion.
        self._pause_summary_provider = pause_summary_provider

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
        elif isinstance(event, SensitiveAccessBlocked):
            self._console.print(
                f"[bold red]Blocked access to sensitive file: {event.path}[/]"
            )
        elif isinstance(event, ContextPruneCompleted):
            self._console.print(
                f"[dim]context pruned {event.tool_results_pruned} old tool result(s): "
                f"~{event.tokens_before} → ~{event.tokens_after} tokens[/]"
            )
        elif isinstance(event, ContextCompactionCompleted):
            self._console.print(
                f"[green]工作上下文已压缩：约 {event.tokens_before} → "
                f"{event.tokens_after} tokens；完整历史已保留[/]"
            )
        elif isinstance(event, ContextCompactionSkipped):
            reason = _compaction_skip_reason(event.reason)
            self._console.print(f"[dim]未压缩工作上下文：{reason}[/]")
        elif isinstance(event, ContextCompactionFailed):
            reason = _compaction_skip_reason(
                event.reason or "provider_error"
            )
            self._console.print(
                f"[yellow]工作上下文压缩失败；完整历史保持不变："
                f"{reason}。{event.error}[/]",
                markup=False,
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
            if event.reason == "max_iterations":
                self._console.print(
                    "[dim]提示：可调大上限，例如 photomatagent chat --max-iterations 50[/]"
                )
            if event.reason in _PAUSE_REASONS:
                self.render_pause_summary()
        elif isinstance(event, LoopFailed):
            self._finish_text()
            self._console.print(f"[red]loop failed: {event.error}[/]")

    def render_pause_summary(self) -> dict[str, Any] | None:
        """Render the deterministic stop summary, if the runtime supplied one.

        Truncated output previously left the user with no visible result at
        all; this prints the persisted facts (reason, stage/evidence, job
        identity, missing logs) without asking the model to fill anything in.
        """
        if self._pause_summary_provider is None:
            return None
        try:
            summary = self._pause_summary_provider()
        except Exception as exc:  # pragma: no cover - defensive
            self._console.print(
                f"[yellow]stop summary unavailable: {type(exc).__name__}: {exc}[/]",
                markup=False,
            )
            return None
        if not isinstance(summary, dict):
            return None
        lines = [
            f"[bold]stop reason[/]: {summary.get('stop_reason', 'UNKNOWN')}",
            f"[bold]goal[/]: {summary.get('goal', 'UNKNOWN')}",
            f"[bold]subtask[/]: {summary.get('active_subtask', 'UNKNOWN')}",
            f"[bold]workflow[/]: {summary.get('workflow_id', 'UNKNOWN')} "
            f"(next stage: {summary.get('next_stage', 'UNKNOWN')})",
        ]
        # Evidence present in ScientificState is reported neutrally with its
        # own classification.  The runtime never prints "validated": only the
        # deterministic checker validates a target, and an attested
        # observation is not a validated conclusion.
        observed = summary.get("observed_evidence") or []
        attested = list(summary.get("attested_evidence") or [])
        exploratory = list(summary.get("exploratory_evidence") or [])
        background = list(summary.get("background_evidence") or [])
        if observed:
            lines.append(
                "[bold]observed evidence[/]: "
                f"{len(observed)} item(s) "
                f"(attested={len(attested)}, "
                f"exploratory={len(exploratory)}, "
                f"background={len(background)})"
            )
            for item in observed[:8]:
                status = item.get("assessment_status")
                lines.append(
                    "  - "
                    f"[{item.get('classification', 'BACKGROUND')}] "
                    f"{item.get('property', '')}="
                    f"{item.get('value', '')} {item.get('unit', '')} "
                    f"(stage={item.get('stage') or 'UNKNOWN'}, "
                    f"job={item.get('job_id') or 'UNKNOWN'}"
                    + (f", status={status}" if status else "")
                    + ")"
                )
        else:
            lines.append("[bold]observed evidence[/]: UNKNOWN (none recorded)")
        gaps = list(summary.get("evidence_gaps") or []) + list(
            summary.get("identified_gaps") or []
        )
        if gaps:
            lines.append("[bold]evidence gaps[/]:")
            for gap in gaps[:8]:
                lines.append(f"  - {gap}")
        else:
            lines.append("[bold]evidence gaps[/]: UNKNOWN (none recorded)")
        lines.append(f"[bold]blocked[/]: {summary.get('blocked_reason', 'UNKNOWN')}")
        lines.append(f"[bold]next action[/]: {summary.get('next_action', 'UNKNOWN')}")
        if summary.get("active_wait_workflows"):
            lines.append(
                "[bold]active waits[/]: "
                + ", ".join(str(item) for item in summary["active_wait_workflows"])
            )
        self._console.print(
            Panel(
                "\n".join(lines),
                title="运行暂停摘要 / deterministic stop summary",
                border_style="yellow",
            )
        )
        return summary

    def _finish_text(self) -> None:
        if self._streaming_text:
            self._console.print()
            self._streaming_text = False

    def flush_agent_text(self) -> None:
        self._finish_text()


# Runs that ended without a usable final answer: the user must always get a
# deterministic, fact-based wrap-up instead of silence.
_PAUSE_REASONS = frozenset(
    {
        "output_truncated",
        "output_truncated_scientific",
        "diagnostic_budget_exhausted",
        "no_progress",
        "max_iterations",
        "empty_response",
    }
)

_COMPACTION_SKIP_REASONS = {    "no_eligible_history": "没有可压缩的旧上下文",
    "inflight_tool_transaction": "工具调用尚未结束，暂不能压缩",
    "no_summarizer": "当前运行时未配置摘要器",
    "no_reduction": "压缩不会减少工作上下文，已保留原历史",
    "retry_cooldown": "自动压缩失败后的冷却期内未重试；可手动 /compact",
    "protected_history": "受保护的历史无法安全压缩",
    "target_unreachable": "在保留近期上下文的前提下无法压到目标大小",
    "unsafe_history": "工具调用历史配对不安全，无法压缩",
    "summary_unit_too_large": "存在不可拆分的超大历史事务",
    "summary_call_limit": "摘要调用次数达到上限",
    "summary_timeout": "摘要请求超时",
    "invalid_summary": "摘要格式无效",
    "summary_too_large": "摘要超过输出预算",
    "empty_summary": "摘要为空",
    "provider_error": "摘要 provider 调用失败",
    "cancelled": "压缩被取消",
}


def _compaction_skip_reason(reason: str) -> str:
    return _COMPACTION_SKIP_REASONS.get(reason, reason)


def print_skill_list(console: Console, skills: list) -> None:
    for skill in skills:
        console.print(f"[bold]{skill.name}[/] — {skill.description or '(no description)'}")
        console.print(f"  [dim]{skill.path}[/]")
