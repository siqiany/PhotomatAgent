"""Interactive model configuration inside the active chat terminal."""
from __future__ import annotations

from typing import Any, Protocol

from prompt_toolkit.history import DummyHistory
from prompt_toolkit.shortcuts import PromptSession
from pydantic import SecretStr
from rich.console import Console

from photomatagent.config import normalize_provider
from photomatagent.model_settings import ROLES, ModelSettingsUpdate, public_settings, save_settings
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.workspace import Workspace

USAGE = "/configure [--role main|judge|feedback] [--provider openai|anthropic|fake] [--model NAME] [--base-url URL]"


class ConfigurationPrompt(Protocol):
    async def prompt_async(self, message: str, **kwargs: Any) -> str: ...


def _options(args: list[str]) -> dict[str, str]:
    options: dict[str, str] = {}
    allowed = {"--role", "--provider", "--model", "--base-url"}
    index = 0
    while index < len(args):
        option, equals, value = args[index].partition("=")
        if option not in allowed or option in options:
            raise ValueError("配置命令选项无效或重复；请使用 /configure --help。API key 只能通过隐藏输入提供。")
        if not equals:
            index += 1
            if index >= len(args) or args[index].startswith("--"):
                raise ValueError("配置选项缺少值；请使用 /configure --help。")
            value = args[index]
        options[option] = value
        index += 1
    return options


async def _ask(session: ConfigurationPrompt, message: str, *, default: str = "", secret: bool = False) -> str:
    if secret and isinstance(session, PromptSession):
        # Use the same terminal input/output, with no history for credential entry.
        password_session: PromptSession[str] = PromptSession(
            input=session.input, output=session.output, history=DummyHistory(),
        )
        return (await password_session.prompt_async(message, is_password=True)).strip()
    return (await session.prompt_async(message, default=default, is_password=secret)).strip()


async def configure_in_chat(
    *, args: list[str], session: ConfigurationPrompt, console: Console,
    workspace: Workspace, runtime: AgentRuntime,
) -> None:
    if args == ["--help"]:
        console.print(USAGE, markup=False)
        console.print("省略参数可逐项交互填写；key 隐藏输入，留空保留；Ctrl+C 取消。配置保存到当前工作区 .env。")
        return
    try:
        options = _options(args)
        role = options.get("--role", "main")
        if role not in ROLES:
            raise ValueError("role 必须为 main、judge 或 feedback。")
        current = public_settings(workspace.root)[role]
        provider = normalize_provider(options.get("--provider") or await _ask(
            session, "提供商 > ", default=str(current["provider"]) or "openai"))
        model = options.get("--model") or await _ask(session, "模型名称 > ", default=str(current["model"]))
        url = options.get("--base-url")
        if url is None:
            url = await _ask(session, "Base URL（留空使用官方服务） > ", default=str(current["base_url"]))
        key = "" if provider == "fake" else await _ask(session, "API key（隐藏输入，留空保留） > ", secret=True)
        update = ModelSettingsUpdate.model_validate({
            "provider": provider, "model": model, "base_url": url,
            "api_key": SecretStr(key) if key else None,
        })
        save_settings(workspace.root, role, update)  # type: ignore[arg-type]
    except (EOFError, KeyboardInterrupt):
        console.print("配置已取消，未保存。")
        return
    except ValueError:
        console.print("[red]配置无效或 key 与其他角色重复；检查角色、模型、URL 与独立 key，使用 /configure --help 查看用法。[/]")
        return
    console.print(f"已保存 {role} 配置到当前工作区 .env；API key 不会显示明文。")
    if role == "main":
        from photomatagent.models.factory import create_provider
        from photomatagent.application.runtime_factory import RuntimeResources
        try:
            provider_instance = create_provider(role="main", workspace=workspace.root)
        except Exception:
            console.print("[red]配置已保存，但当前模型未切换。请检查提供商依赖与配置；当前会话继续使用原模型。[/]")
            return
        previous = runtime.replace_model_provider(provider_instance)
        await RuntimeResources((previous,)).aclose()
        console.print("当前会话下一轮使用新模型；聊天记录、科研状态和权限保留。")
    else:
        console.print("后续审查或反馈编译使用新配置。")
