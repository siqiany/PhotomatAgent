"""Interactive slash-command routing shared by every chat session."""

from __future__ import annotations

import asyncio
import shlex
from dataclasses import dataclass

from rich.console import Console

from photomatagent.cli.render import ChatRenderer
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import ApprovalScope, SwitchablePermissionPolicy
from photomatagent.workspace import Workspace


@dataclass(frozen=True)
class CommandSpec:
    usage: str
    description: str


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
    CommandSpec("/configure [options]", "配置工作区 LLM（可能交互询问）"),
    CommandSpec("/compact", "压缩较早的工作上下文"),
    CommandSpec("/exit 或 /quit", "退出当前聊天"),
)


class ChatCommandRouter:
    """Parse slash commands without sending them to the language model."""

    _CLI_GROUPS = {"tools", "skills", "scientific", "mcp", "sessions", "experiments"}
    _DEFAULT_SUBCOMMAND = {
        "tools": "list",
        "skills": "list",
        "scientific": "status",
        "mcp": "list",
        "sessions": "list",
    }

    def __init__(self, console: Console, runtime: AgentRuntime, workspace: Workspace) -> None:
        self.console = console
        self.runtime = runtime
        self.workspace = workspace

    async def execute(self, line: str) -> bool:
        try:
            parts = shlex.split(line)
        except ValueError as exc:
            self.console.print(f"[red]命令解析失败：{exc}[/]")
            return True
        if not parts:
            return True
        command = parts[0].lower()
        args = parts[1:]
        if command == "/help":
            self._help()
        elif command == "/approve":
            self._approve(args)
        elif command == "/compact":
            await self._compact()
        elif command == "/doctor":
            await self._run_cli(["doctor", *args])
        elif command == "/configure":
            await self._run_cli(["configure", *args])
        elif command.removeprefix("/") in self._CLI_GROUPS:
            group = command.removeprefix("/")
            if not args and group in self._DEFAULT_SUBCOMMAND:
                args = [self._DEFAULT_SUBCOMMAND[group]]
            await self._run_cli([group, *args])
        else:
            self.console.print(f"[yellow]未知命令：{command}。输入 /help 查看可用命令。[/]")
        return True

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

    async def _run_cli(self, args: list[str]) -> None:
        """Invoke the existing Typer command surface in-process and capture output."""
        from typer.testing import CliRunner

        from photomatagent.cli.app import app

        if "--workspace" not in args and args[0] not in {"sessions", "skills"}:
            args.extend(["--workspace", str(self.workspace.root)])
        result = await asyncio.to_thread(CliRunner().invoke, app, args)
        if result.stdout:
            self.console.print(result.stdout.rstrip(), markup=False)
        if result.exception is not None and result.exit_code != 0:
            self.console.print(f"[red]命令失败（exit={result.exit_code}）：{result.exception}[/]")
