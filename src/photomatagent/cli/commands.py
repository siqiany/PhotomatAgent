"""Interactive slash-command routing shared by every chat session."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rich.console import Console

from photomatagent.cli.render import ChatRenderer
from photomatagent.errors import ToolExecutionError
from photomatagent.logging.event_logger import EventLogger
from photomatagent.redaction import redact_text
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import ApprovalScope, SwitchablePermissionPolicy
from photomatagent.scientific.evolution.service import EvolutionServiceError
from photomatagent.scientific.evolution.store import EvolutionStoreError
from photomatagent.workspace import Workspace


# Matches ANSI/control escape sequences Rich may embed in CLI output when the
# command was captured while its console believed it was writing to a TTY:
#   CSI sequences like "\x1b[1;2;36m" (colors, styles, cursor moves)
#   OSC sequences like "\x1b]8;;url\x1b\\" (hyperlink wrappers)
# plus any leftover bare ESC byte. Keeping them would make the interactive
# console print literal "[1;2;36m…[0m" garbage, because Rich drops the ESC
# character and re-renders the remainder as plain text.
_ANSI_CSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_ANSI_OSC_RE = re.compile(r"\x1b\].*?(?:\x07|\x1b\\)", re.DOTALL)


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI/control escape sequences so only readable text remains."""
    text = _ANSI_CSI_RE.sub("", text)
    text = _ANSI_OSC_RE.sub("", text)
    return text.replace("\x1b", "")


def _print_cli_capture(console: Console, captured: str) -> None:
    """Render captured CLI stdout without re-mangling embedded ANSI escapes."""
    console.print(
        strip_ansi_codes(captured).rstrip("\n"),
        markup=False,
        highlight=False,
    )


@dataclass(frozen=True)
class CommandSpec:
    usage: str
    description: str


class PromptSessionLike(Protocol):
    async def prompt_async(self, message: str) -> str: ...


COMMANDS = (
    CommandSpec("/help", "显示所有聊天命令及功能"),
    CommandSpec("/approve -o", "本次聊天任务完全允许所有工具"),
    CommandSpec("/approve -a", "在当前工作区持久保持完全允许"),
    CommandSpec("/approve -b", "清除完全允许并恢复启动时的初始权限策略"),
    CommandSpec("/doctor", "运行本地环境、配置和智能体循环诊断"),
    CommandSpec("/tools [list]", "列出本地工具注册表"),
    CommandSpec("/tools surface", "显示当前模型可见的工具表面与上下文成本"),
    CommandSpec("/tools search <query>", "搜索延迟暴露工具"),
    CommandSpec("/skills [list]", "查看可用技能"),
    CommandSpec("/scientific [status]", "查看所有科学能力包及依赖状态"),
    CommandSpec("/scientific approve <decision-id>", "显示并确认批准 VASP 应用级决策（仅用户）"),
    CommandSpec("/scientific scnet-doctor", "诊断 SCNet、Slurm 与科学应用"),
    CommandSpec("/mcp [list]", "列出 MCP 服务配置"),
    CommandSpec("/mcp status", "连接并探测 MCP 服务状态"),
    CommandSpec("/mcp doctor", "深度诊断 MCP 命令、环境和连接"),
    CommandSpec("/mcp tools <server>", "列出指定 MCP 服务提供的工具"),
    CommandSpec("/mcp test <server> [--tool NAME]", "测试 MCP 服务及可选工具调用"),
    CommandSpec("/sessions [list]", "列出历史任务"),
    CommandSpec("/sessions show [latest|id]", "显示任务元数据"),
    CommandSpec("/sessions stats [latest|id]", "显示任务运行统计"),
    CommandSpec("/sessions context [latest|id]", "显示上下文生命周期统计"),
    CommandSpec("/sessions replay [latest|id]", "离线重放任务轨迹"),
    CommandSpec("/experiments run <config>", "运行确定性实验"),
    CommandSpec("/experiments compare <a> <b>", "比较两份实验结果"),
    CommandSpec(
        "/evolve [list|status|history|start|feedback|compile|iterate]",
        "管理专家反馈驱动的持久演化任务；feedback/compile 复用当前交互会话",
    ),
    CommandSpec("/configure [options]", "配置工作区 LLM（可能交互询问）"),
    CommandSpec("/compact", "压缩较早的工作上下文"),
    CommandSpec("/resume <id|目录|latest>", "回溯加载历史 session，并在其基础上继续追问"),
    CommandSpec("/exit 或 /quit", "退出当前聊天"),
)


class ChatCommandRouter:
    """Parse slash commands without sending them to the language model."""

    _CLI_GROUPS = {
        "tools",
        "skills",
        "scientific",
        "mcp",
        "sessions",
        "experiments",
        "evolve",
    }
    _DEFAULT_SUBCOMMAND = {
        "tools": "list",
        "skills": "list",
        "scientific": "status",
        "mcp": "list",
        "sessions": "list",
        "evolve": "list",
    }

    def __init__(
        self,
        console: Console,
        runtime: AgentRuntime,
        workspace: Workspace,
        *,
        logger: EventLogger | None = None,
        sessions_dir: Path | str | None = None,
        prompt_session: PromptSessionLike | None = None,
    ) -> None:
        self.console = console
        self.runtime = runtime
        self.workspace = workspace
        self.logger = logger
        self.sessions_dir = sessions_dir
        self.prompt_session = prompt_session

    async def execute(self, line: str) -> bool:
        try:
            parts = shlex.split(line)
        except ValueError as exc:
            self.console.print(f"[red]命令解析失败：{exc}[/]")
            return True
        if not parts:
            return True
        first_token = parts[0]
        command = first_token.lower()
        args = parts[1:]
        if command == "/help":
            self._help()
        elif command == "/approve":
            self._approve(args)
        elif command == "/compact":
            await self._compact()
        elif command == "/resume":
            await self._resume(args)
        elif command == "/doctor":
            await self._run_cli(["doctor", *args])
        elif command == "/configure":
            await self._run_cli(["configure", *args])
        elif first_token == "/evolve":
            await self._evolve(args)
        elif (
            command.removeprefix("/") in self._CLI_GROUPS
            and command.removeprefix("/") != "evolve"
        ):
            group = command.removeprefix("/")
            if not args and group in self._DEFAULT_SUBCOMMAND:
                args = [self._DEFAULT_SUBCOMMAND[group]]
            await self._run_cli([group, *args])
        else:
            self.console.print(f"[yellow]未知命令：{command}。输入 /help 查看可用命令。[/]")
        return True

    async def _evolve(self, args: list[str]) -> None:
        """Route interactive evolve forms without redirecting process stdin."""

        subcommand = args[0] if args else "list"
        if subcommand not in {"feedback", "compile"} or "--help" in args:
            await self._run_cli(["evolve", *(args or ["list"])])
            return
        if self.prompt_session is None:
            self.console.print(
                "[red]当前聊天未提供交互 PromptSession，无法填写反馈或确认修订。[/]"
            )
            return

        try:
            from photomatagent.cli.evolve import (
                run_compile_command,
                run_feedback_command,
            )

            if subcommand == "feedback":
                parsed = _parse_evolve_form_args(
                    args[1:],
                    options={"--version", "--file"},
                    usage="/evolve feedback <evolution-id> [--version VERSION] [--file PATH]",
                )
                await run_feedback_command(
                    session=self.prompt_session,
                    output=self.console,
                    workspace=self.workspace.root,
                    evolution_id=parsed.evolution_id,
                    version=parsed.options.get("--version"),
                    feedback_file=(
                        Path(parsed.options["--file"])
                        if "--file" in parsed.options
                        else None
                    ),
                )
            else:
                parsed = _parse_evolve_form_args(
                    args[1:],
                    options={"--version", "--provider", "--model"},
                    usage=(
                        "/evolve compile <evolution-id> [--version VERSION] "
                        "[--provider PROVIDER] [--model MODEL]"
                    ),
                )
                await run_compile_command(
                    session=self.prompt_session,
                    output=self.console,
                    workspace=self.workspace.root,
                    evolution_id=parsed.evolution_id,
                    version=parsed.options.get("--version"),
                    provider=parsed.options.get("--provider"),
                    model=parsed.options.get("--model"),
                )
        except KeyboardInterrupt:
            self.console.print("[dim]Evolution flow cancelled; no partial data was written.[/]")
        except (
            OSError,
            UnicodeError,
            ValueError,
            ToolExecutionError,
            EvolutionServiceError,
            EvolutionStoreError,
        ) as exc:
            self.console.print(f"[red]{redact_text(str(exc))}[/]")

    def _help(self) -> None:
        from rich.table import Table

        table = Table("命令", "功能")
        for spec in COMMANDS:
            table.add_row(spec.usage, spec.description)
        self.console.print(table)

    def _approve(self, args: list[str]) -> None:
        policy = self.runtime.permission_policy
        if not isinstance(policy, SwitchablePermissionPolicy):
            self.console.print("[red]当前运行时不支持动态权限切换。[/]")
            return
        if args == ["-o"]:
            policy.allow_for_session()
            self.console.print("[bold yellow]已在本次聊天任务中完全允许所有工具。[/]")
        elif args == ["-a"]:
            policy.allow_always()
            self.console.print(
                "[bold red]已为当前工作区持久启用完全允许；后续启动仍会生效。[/]"
            )
        elif args == ["-b"]:
            policy.reset()
            self.console.print("[green]已恢复启动时的初始权限策略。[/]")
        else:
            scope = policy.scope
            label = {
                ApprovalScope.DEFAULT: "初始策略",
                ApprovalScope.SESSION: "本次任务完全允许",
                ApprovalScope.ALWAYS: "持久完全允许",
            }[scope]
            self.console.print(f"当前权限状态：{label}。用法：/approve -o | -a | -b")

    async def _compact(self) -> None:
        events = await self.runtime.compact_working_context()
        renderer = ChatRenderer(self.console)
        for event in events:
            renderer.handle(event)
        if not events:
            self.console.print("[dim]No eligible old turns to compact.[/]")

    async def _resume(self, args: list[str]) -> None:
        """Load a historical session into the running chat and continue on it."""
        from photomatagent.observability.trace import TraceError, resolve_session_path
        from photomatagent.sessions.store import (
            load_session_snapshot,
            save_session_snapshot,
            session_is_resumable,
        )

        if not args:
            self.console.print("[yellow]用法：/resume <session-id | 会话目录 | latest>[/]")
            return
        try:
            session_dir = resolve_session_path(args[0], self.sessions_dir)
        except TraceError as exc:
            self.console.print(f"[red]{exc}[/]")
            return
        if not session_is_resumable(session_dir):
            self.console.print(
                f"[red]session {session_dir.name} 没有可恢复的状态（只有离线轨迹），"
                "只能 /sessions replay。[/]"
            )
            return
        if self.logger is not None:
            # Persist the current in-chat session state before switching away.
            save_session_snapshot(
                self.logger.session_dir,
                conversation=self.runtime.conversation_state,
                scientific=self.runtime.scientific_state,
                engine=self.runtime.context_engine.snapshot(),
            )
        snapshot = load_session_snapshot(session_dir)
        self.runtime.restore_session(snapshot)
        if self.logger is not None:
            # Follow-up turns continue in the resumed session's trace.
            self.logger.session_id = session_dir.name
            self.logger.session_dir = session_dir
            self.logger.events_path = session_dir / "events.jsonl"
        self.console.print(
            f"[green]已回溯到 session {session_dir.name}："
            f"{len(snapshot.conversation.messages)} 条消息，"
            f"{len(snapshot.scientific.evidence)} 条证据，"
            f"{len(snapshot.scientific.claims)} 条结论；可直接继续追问。[/]"
        )

    async def _run_cli(self, args: list[str]) -> None:
        """Invoke the existing Typer command surface in-process and capture output."""
        from typer.testing import CliRunner

        from photomatagent.cli.app import app

        if "--workspace" not in args and args[0] not in {"sessions", "skills"}:
            args.extend(["--workspace", str(self.workspace.root)])
        # CliRunner temporarily replaces process-global standard streams.  On
        # some Click/Python combinations that can lose the event-loop wakeup
        # used by ``asyncio.to_thread`` even though the worker has completed.
        # Polling the plain concurrent Future keeps async CLI implementations
        # off this running loop without depending on that cross-thread wakeup.
        with ThreadPoolExecutor(max_workers=1) as executor:
            invocation = executor.submit(CliRunner().invoke, app, args)
            while not invocation.done():
                await asyncio.sleep(0.01)
            result = invocation.result()
        if result.stdout:
            # The CLI module console may render ANSI styles into the captured
            # buffer (it fixes its color system at import time, when stdout is
            # still the interactive TTY). Re-printing raw escapes through
            # another Rich console corrupts them into literal "[1;2;36m…[0m"
            # text, so hand the interactive console only clean plain text.
            _print_cli_capture(self.console, result.stdout)
        if result.exception is not None and result.exit_code != 0:
            self.console.print(f"[red]命令失败（exit={result.exit_code}）：{result.exception}[/]")


@dataclass(frozen=True)
class _ParsedEvolveFormArgs:
    evolution_id: str
    options: dict[str, str]


def _parse_evolve_form_args(
    args: list[str],
    *,
    options: set[str],
    usage: str,
) -> _ParsedEvolveFormArgs:
    """Parse the small interactive-only surface without invoking Click stdin."""

    parsed_options: dict[str, str] = {}
    positionals: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token.startswith("--"):
            option, separator, inline_value = token.partition("=")
            if option not in options:
                raise ValueError(f"未知选项：{option}。用法：{usage}")
            if option in parsed_options:
                raise ValueError(f"选项重复：{option}。用法：{usage}")
            if separator:
                value = inline_value
            else:
                index += 1
                if index >= len(args) or args[index].startswith("--"):
                    raise ValueError(f"选项 {option} 缺少值。用法：{usage}")
                value = args[index]
            if not value:
                raise ValueError(f"选项 {option} 缺少值。用法：{usage}")
            parsed_options[option] = value
        else:
            positionals.append(token)
        index += 1
    if len(positionals) != 1:
        raise ValueError(f"用法：{usage}")
    return _ParsedEvolveFormArgs(positionals[0], parsed_options)
