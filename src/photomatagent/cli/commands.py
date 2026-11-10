"""Interactive slash-command routing shared by every chat session."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

from rich.console import Console

from photomatagent.cli.render import ChatRenderer
from photomatagent.cli.research_mode import ResearchMode
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


ExpertTaskKind = Literal["proposal", "validation"]


COMMANDS = (
    CommandSpec("/project list|create|use|show|rename|archive|unarchive", "管理研究课题"),
    CommandSpec("/task list|show|rename|move|state|tag|archive|unarchive|input|replace-input", "管理任务与关联输入"),
    CommandSpec("/search 关键词 [--workspace]", "检索当前课题或工作区历史"),
    CommandSpec("/data status|doctor|import|refresh|reindex|export|backup|restore", "数据管理与恢复说明"),
    CommandSpec("/research [--session ID|--task ID]", "进入自动研究；后续输入为专家评审。/back 返回普通聊天"),
    CommandSpec("/research methods|sources|documents|reproduce|process <需求>", "从聊天进入材料方法、来源、文档、复现或工艺流程"),
    CommandSpec("/image <工作区相对路径|附件 ID> <问题>", "在普通聊天或科研任务中发送图像"),
    CommandSpec("/make skill [task_ID|session_ID]", "从历史提炼待审阅技能，保留科学证据边界"),
    CommandSpec("/goal <新目标>", "明确更换当前目标；普通后续输入默认补充原目标"),
    CommandSpec("/stop", "科研模式中取消后台当前轮；普通运行使用 Ctrl+C"),
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
    CommandSpec("/rag [status]", "查看并操作 Qdrant 文献 RAG"),
    CommandSpec("/research [list|start|show|feedback|watch|events|compact|abilities]", "与 Web 共用科研任务、自然语言返修和能力库"),
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
        "/evolve [list|status|history|start|feedback|compile|iterate|qd]",
        "管理专家反馈驱动的持久演化任务；feedback/compile 复用当前交互会话",
    ),
    CommandSpec(
        "/expert [session-id|history] [--task-kind proposal|validation]",
        "启动内部专家评审向导；默认 validation，模式由命令参数明确选择",
    ),
    CommandSpec("/configure [--role main|judge|feedback]", "在聊天内配置模型和独立 key；/configure --help 查看选项"),
    CommandSpec("/compact", "立即压缩工作上下文，保留完整历史"),
    CommandSpec("/resume <id|目录|latest>", "回溯加载历史 session，并在其基础上继续追问"),
    CommandSpec("/exit 或 /quit", "退出当前聊天"),
)


_RESEARCH_ENTRY_POINTS = {
    "methods": "用 method_search 比较本地方法及已验证归档。必要时经授权用 method_search_external 查询外部来源；用 method_inspect 查看候选范围和证据。需要保存决策或依据明确的方法缺口时用 decision.record。",
    "sources": "按问题选择 web.search、literature.search 或已有本地来源。逐项报告来源状态。",
    "documents": "使用 literature.read_paper 或 literature.read_object 按来源锚点阅读。需要检索时可按授权使用 literature.index_document。",
    "reproduce": "使用 reproduction.prepare 检查复现条件。明确准备、执行、参考值检查及适用性状态。",
    "process": "从有条件的工艺证据提出方案。用 process.register_plan 保存提案并保留证据缺口。",
}


class ChatCommandRouter:
    """Parse slash commands without sending them to the language model."""

    _CLI_GROUPS = {
        "tools",
        "skills",
        "scientific",
        "rag",
        "research",
        "mcp",
        "sessions",
        "experiments",
        "evolve",
    }
    _DEFAULT_SUBCOMMAND = {
        "tools": "list",
        "skills": "list",
        "scientific": "status",
        "rag": "status",
        "research": "list",
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
        self.research_mode: ResearchMode | None = None
        from photomatagent.cli.information import InformationCommands
        self.information = InformationCommands(console)

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
        if command in {"/project", "/task", "/search", "/data"}:
            try:
                await self.information.execute(command, args)
                if self.research_mode is not None:
                    self.research_mode.project_id = self.information.project_id
            except ValueError as exc:
                self.console.print(str(exc), markup=False)
        elif command == "/help":
            self._help()
        elif command == "/approve":
            self._approve(args)
        elif command == "/compact":
            await self._compact(args)
        elif command == "/resume":
            await self._resume(args)
        elif command == "/sessions" and args[:1] == ["resume"]:
            await self._resume(args[1:] or ["latest"])
        elif command == "/image":
            if len(args) < 2:
                self.console.print("用法：/image <工作区相对路径或科研任务附件 ID> <问题>")
            elif self.research_mode is not None:
                await self.research_mode.submit_image(args[0], " ".join(args[1:]))
            else:
                from photomatagent.cli.chat import run_goal

                await run_goal(self.console, self.runtime, " ".join(args[1:]), image_paths=(args[0],),
                               session_dir=self.logger.session_dir if self.logger else None)
        elif command == "/research" and args and args[0] in _RESEARCH_ENTRY_POINTS:
            await self._research_entry(args)
        elif command in {"/research", "/resarch"} and (not args or args[0] in {"--session", "--task", "task", "chat"}):
            if self.prompt_session is None:
                self.console.print("科研交互需要聊天输入会话。")
                return True
            if self.research_mode is None:
                self.research_mode = ResearchMode(self.console, self.prompt_session)
                self.research_mode.project_id = self.information.project_id
            await self.research_mode.select(args[1:] if args[:1] == ["chat"] else args)
        elif command == "/back":
            self.research_mode = None
            self.console.print("已返回普通聊天；后台科研任务保留。")
        elif command == "/make" and args[:1] == ["skill"]:
            if self.prompt_session is None:
                return True
            mode = self.research_mode or ResearchMode(self.console, self.prompt_session)
            source = args[1] if len(args) > 1 else (None if mode.task_id else (self.logger.session_id if self.logger else None))
            await mode.make_skill(source)
        elif command == "/stop" and self.research_mode is not None:
            await self.research_mode.stop()
        elif command == "/events" and self.research_mode is not None and self.research_mode.task_id is not None:
            self.console.print(await asyncio.to_thread(self.research_mode.client.events, self.research_mode.task_id), markup=False)
        elif command == "/doctor":
            await self._run_cli(["doctor", *args])
        elif command == "/configure":
            from photomatagent.cli.configure import ConfigurationPrompt, configure_in_chat
            from photomatagent.cli.prompt import make_prompt_session
            await configure_in_chat(
                args=args, session=cast(ConfigurationPrompt, self.prompt_session or make_prompt_session()),
                console=self.console, workspace=self.workspace, runtime=self.runtime,
            )
        elif first_token == "/evolve":
            await self._evolve(args)
        elif command == "/expert":
            await self._expert(args)
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

    async def _research_entry(self, args: list[str]) -> None:
        from photomatagent.cli.chat import run_goal

        if len(args) < 2:
            self.console.print(f"用法：/research {args[0]} <需求>")
            return
        text = f"{_RESEARCH_ENTRY_POINTS[args[0]]}\n用户需求：{' '.join(args[1:])}"
        if self.research_mode is not None:
            await self.research_mode.submit(text)
        else:
            await run_goal(self.console, self.runtime, text, intent="supplement",
                           session_dir=self.logger.session_dir if self.logger else None)

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

    async def _expert(self, args: list[str]) -> None:
        """Intercept the complete expert wizard so entries never reach the model."""
        if "--help" in args:
            self.console.print(
                "用法：/expert [session-id|history] "
                "[--task-kind proposal|validation]（默认 validation）"
            )
            return
        try:
            parsed = _parse_expert_args(args)
        except ValueError as exc:
            self.console.print(f"[EXPERT MODE | ERROR] {exc}")
            return
        if self.prompt_session is None:
            self.console.print("[EXPERT MODE | ERROR] 当前聊天未提供交互 PromptSession。")
            return
        from photomatagent.cli.expert import run_expert_mode

        try:
            await run_expert_mode(
                session=self.prompt_session,
                output=self.console,
                workspace=self.workspace.root,
                source=parsed.source,
                sessions_dir=self.sessions_dir,
                logger=self.logger,
                runtime=self.runtime,
                iterate_callback=self._run_expert_iterate,
                task_kind=parsed.task_kind,
            )
        except (OSError, UnicodeError, ValueError, ToolExecutionError,
                EvolutionServiceError, EvolutionStoreError) as exc:
            self.console.print(f"[EXPERT MODE | ERROR] {redact_text(str(exc))}")

    async def _run_expert_iterate(self, evolution_id: str) -> None:
        await self._run_cli(["evolve", "iterate", evolution_id])

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

    async def _compact(self, args: list[str]) -> None:
        if args in (["--help"], ["-h"]):
            self.console.print(
                "用法：/compact（立即压缩工作上下文；不接受定制摘要指令）"
            )
            return
        if args:
            self.console.print("用法：/compact")
            return
        if self.research_mode is not None:
            await self.research_mode.compact()
            return
        if not self.runtime.conversation_state.messages:
            self.console.print("[dim]没有可压缩的旧上下文[/]")
            return
        from photomatagent.runtime.context_engine import (
            ContextCompactionCompleted,
            ContextLimitExceeded,
        )

        self.console.print("[dim]正在压缩上下文…[/]")
        renderer = ChatRenderer(self.console)
        try:
            events = await self.runtime.compact_working_context()
        except ContextLimitExceeded as exc:
            for event in getattr(exc, "events", []):
                renderer.handle(event)
            self.console.print(
                f"[red]上下文压缩未完成：{redact_text(str(exc))}[/]",
                markup=False,
            )
            return
        except Exception as exc:
            self.console.print(
                f"[red]上下文压缩失败：{redact_text(str(exc))}[/]",
                markup=False,
            )
            return

        for event in events:
            renderer.handle(event)
        completed = any(
            isinstance(event, ContextCompactionCompleted) for event in events
        )
        if not completed:
            if not events:
                self.console.print("[dim]没有可压缩的旧上下文[/]")
            return

        if self.logger is None:
            self.console.print(
                "[yellow]压缩仅当前进程生效，未持久保存。[/]"
            )
            return
        try:
            from photomatagent.sessions.store import save_session_snapshot

            save_session_snapshot(
                self.logger.session_dir,
                conversation=self.runtime.conversation_state,
                scientific=self.runtime.scientific_state,
                engine=self.runtime.context_engine.snapshot(),
            )
        except OSError as exc:
            self.console.print(
                "[yellow]压缩已在内存生效，但保存失败；重启后可能恢复旧状态。[/]",
                markup=False,
            )
            self.console.print(
                f"[dim]{redact_text(str(exc))}[/]", markup=False
            )
            return
        self.console.print("[green]压缩状态已保存[/]")

    async def _resume(self, args: list[str]) -> None:
        """Load a historical session into the running chat and continue on it."""
        from photomatagent.observability.trace import TraceError, resolve_session_path
        from photomatagent.sessions.store import (
            load_session_snapshot,
            migration_diagnostic_codes,
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
        from photomatagent.sessions.binding import managed_task, attach_session
        task_id = managed_task(session_dir)
        if task_id is not None:
            if self.prompt_session is None:
                self.console.print(f"此会话属于科研任务 {task_id}，请用 /research --task {task_id} 继续。")
                return
            mode = ResearchMode(self.console, self.prompt_session)
            mode.project_id = self.information.project_id
            await mode.select(["--task", task_id])
            self.research_mode = mode
            return
        if not session_is_resumable(session_dir):
            self.console.print(
                f"[red]session {session_dir.name} 没有可恢复的状态（只有离线轨迹），"
                "可用 /sessions replay 查看；若快照含未完成工具调用，需先核对实际副作用。[/]"
            )
            return
        snapshot = load_session_snapshot(session_dir)
        self.runtime._validate_engine_snapshot(snapshot.conversation.messages, snapshot.engine)
        if self.logger is not None:
            # Persist the current in-chat session state before switching away.
            if self.runtime.persistence is not None:
                await self.runtime.persistence.checkpoint(self.runtime, "session_switch")
            else:
                save_session_snapshot(
                    self.logger.session_dir,
                    conversation=self.runtime.conversation_state,
                    scientific=self.runtime.scientific_state,
                    engine=self.runtime.context_engine.snapshot(),
                )
        try:
            forked = False
            if self.logger is not None:
                forked = attach_session(self.runtime, self.logger, session_dir, snapshot)
            else:
                self.runtime.restore_session(snapshot)
        except (ValueError, RuntimeError, OSError) as exc:
            self.console.print(
                f"[red]无法恢复 session：{redact_text(str(exc))}[/]",
                markup=False,
            )
            return
        diagnostic_codes = migration_diagnostic_codes(snapshot)
        diagnostic_text = (
            f" 迁移诊断：{','.join(diagnostic_codes)}。"
            if diagnostic_codes
            else ""
        )
        self.research_mode = None
        if forked:
            self.console.print(f"原会话是只读历史来源；已创建续聊会话 {self.logger.session_id if self.logger else ''}，保留原始记录。")
        self.console.print(
            f"[green]已回溯到 session {session_dir.name}："
            f"{len(snapshot.conversation.messages)} 条消息，"
            f"{len(snapshot.scientific.evidence)} 条证据，"
            f"{len(snapshot.scientific.claims)} 条结论；可直接继续追问。"
            f"{diagnostic_text}[/]"
        )

    async def _run_cli(self, args: list[str]) -> None:
        """Invoke the existing Typer command surface in-process and capture output."""
        from typer.testing import CliRunner

        from photomatagent.cli.app import app

        if args[:2] == ["sessions", "list"] and self.sessions_dir is not None and "--sessions-dir" not in args:
            args.extend(["--sessions-dir", str(self.sessions_dir)])
        if "--workspace" not in args and args[0] not in {"sessions", "skills", "research"}:
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


@dataclass(frozen=True)
class _ParsedExpertArgs:
    source: str | None
    task_kind: ExpertTaskKind


def _parse_expert_args(args: list[str]) -> _ParsedExpertArgs:
    """Parse the explicit expert target mode without consulting model output."""

    usage = (
        "用法：/expert [session-id|history] "
        "[--task-kind proposal|validation]（默认 validation）"
    )
    source: str | None = None
    task_kind: ExpertTaskKind = "validation"
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--task-kind" or token.startswith("--task-kind="):
            if task_kind != "validation" or any(
                item == "--task-kind" or item.startswith("--task-kind=")
                for item in args[:index]
            ):
                raise ValueError(f"选项 --task-kind 重复。{usage}")
            if token == "--task-kind":
                index += 1
                if index >= len(args):
                    raise ValueError(f"选项 --task-kind 缺少值。{usage}")
                value = args[index]
            else:
                value = token.partition("=")[2]
            if value not in {"proposal", "validation"}:
                raise ValueError(f"--task-kind 必须是 proposal 或 validation。{usage}")
            task_kind = value  # type: ignore[assignment]
        elif token.startswith("--"):
            raise ValueError(f"未知选项：{token}。{usage}")
        elif source is None:
            source = token
        else:
            raise ValueError(usage)
        index += 1
    return _ParsedExpertArgs(source=source, task_kind=task_kind)


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
