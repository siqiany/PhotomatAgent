"""Read-only preflight and local startup for the browser workbench."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import shutil
import socket
import sys
import time
from typing import Sequence
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser

from photomatagent.config import DotEnvConfig, normalize_provider

LOCAL_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
_MODEL_FIELDS = {
    "openai": ("OPENAI_MODEL", "OPENAI_API_KEY"),
    "anthropic": ("ANTHROPIC_MODEL", "ANTHROPIC_API_KEY"),
}


class PreflightError(RuntimeError):
    """A safe, actionable startup diagnostic."""


@dataclass(frozen=True)
class StartupConfig:
    workspace: Path
    port: int
    provider: str
    model: str
    webui_directory: Path
    demo: bool = False


def find_webui_directory() -> Path | None:
    """Find Vite output in a checkout or in the installed wheel."""
    source_tree = Path(__file__).resolve().parents[3] / "webui" / "dist"
    packaged = Path(__file__).resolve().parent / "static"
    for candidate in (source_tree, packaged):
        if (candidate / "index.html").is_file():
            return candidate
    return None


def preflight(
    *,
    workspace: Path | str,
    port: int,
    demo: bool = False,
    webui_directory: Path | str | None = None,
) -> StartupConfig:
    """Check runtime prerequisites without creating files or printing secrets."""
    root = Path(workspace).expanduser().resolve()
    if not root.is_dir():
        raise PreflightError(f"工作区目录不存在：{root}")
    if not 1 <= port <= 65535:
        raise PreflightError("端口必须在 1 到 65535 之间。")
    if sys.version_info[:2] != (3, 12):
        raise PreflightError(
            f"工作台需要 Python 3.12；当前为 {sys.version_info.major}.{sys.version_info.minor}。"
        )
    missing = [
        name
        for name in ("fastapi", "uvicorn", "multipart")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise PreflightError(
            f"缺少 Web 依赖：{', '.join(missing)}。请在项目目录运行 "
            "`uv sync --extra web --inexact`，再重新启动。"
        )

    ui = Path(webui_directory).expanduser().resolve() if webui_directory else find_webui_directory()
    if ui is None or not (ui / "index.html").is_file():
        raise PreflightError(
            "没有找到构建后的 Web 界面。请在项目目录运行 `npm --prefix webui ci` "
            "和 `npm --prefix webui run build`。"
        )

    config_path = root / ".env"
    if config_path.exists() and not config_path.is_file():
        raise PreflightError("工作区的 .env 路径不是普通文件；请检查配置路径。")
    values = DotEnvConfig(root).values(environ=dict(os.environ))
    if demo:
        provider, model = "fake", "fake"
    else:
        provider_value = values.get("PHOTOMATAGENT_PROVIDER", "").strip()
        if not provider_value:
            raise PreflightError(
                "尚未配置模型。请在现有 .env 或进程环境中设置 "
                "PHOTOMATAGENT_PROVIDER=openai/anthropic、对应的模型变量 "
                "OPENAI_MODEL/ANTHROPIC_MODEL 及 OPENAI_API_KEY/ANTHROPIC_API_KEY；"
                "启动器不会创建或修改 .env。"
            )
        try:
            provider = normalize_provider(provider_value)
        except ValueError as error:
            raise PreflightError(
                "PHOTOMATAGENT_PROVIDER 只能设置为 openai 或 anthropic。"
            ) from error
        if provider == "fake":
            raise PreflightError(
                "Fake provider 只能通过显式的 --demo 启动；演示模式不代表科研验证。"
            )
        model_field, key_field = _MODEL_FIELDS[provider]
        model = values.get(model_field, "").strip()
        if not model:
            raise PreflightError(
                f"已选择 {provider}，但未设置 {model_field}。请在现有 .env 或进程环境中补齐。"
            )
        if not values.get(key_field, "").strip():
            raise PreflightError(
                f"已选择 {provider}，但未设置 {key_field}。请在现有 .env 或进程环境中补齐。"
            )

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((LOCAL_HOST, port))
    except OSError as error:
        raise PreflightError(
            f"本地端口 {port} 不可用。请关闭占用该端口的程序，或使用 --port 指定其他端口。"
        ) from error

    return StartupConfig(root, port, provider, model, ui, demo)


def _load_runtime_environment(startup: StartupConfig) -> None:
    """Load just the selected provider settings into process memory."""
    if startup.demo:
        return
    values = DotEnvConfig(startup.workspace).values(environ=dict(os.environ))
    os.environ["PHOTOMATAGENT_PROVIDER"] = startup.provider
    model_field, key_field = _MODEL_FIELDS[startup.provider]
    os.environ[model_field] = values[model_field]
    os.environ[key_field] = values[key_field]
    if startup.provider == "openai" and values.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = values["OPENAI_BASE_URL"]


def _create_app(startup: StartupConfig):
    from photomatagent.application.runner import RunManager
    from photomatagent.application.service import WorkbenchService
    from photomatagent.application.store import WorkbenchStore
    from photomatagent.web.app import create_app
    from photomatagent.web.artifacts import ArtifactService

    metadata = startup.workspace / ".photomatagent"
    store = WorkbenchStore(metadata / "workbench.sqlite3")
    options: dict[str, object] = {
        "workspace_root": startup.workspace,
        "provider": startup.provider,
        "model": startup.model,
    }
    manager = RunManager(store, runtime_options=options)
    service = WorkbenchService(
        store=store,
        manager=manager,
        workspace_root=startup.workspace,
        sessions_dir=metadata / "sessions",
        artifacts=ArtifactService(store, startup.workspace),
    )
    return create_app(service=service, manager=manager, webui_directory=startup.webui_directory)


async def _serve(startup: StartupConfig, *, no_browser: bool) -> None:
    import uvicorn

    app = _create_app(startup)
    address = f"http://{LOCAL_HOST}:{startup.port}/"
    server = uvicorn.Server(
        uvicorn.Config(app, host=LOCAL_HOST, port=startup.port, workers=1, log_level="info")
    )
    task = asyncio.create_task(server.serve(), name="photomatagent-workbench-server")
    try:
        deadline = time.monotonic() + 20
        ready = False
        while time.monotonic() < deadline and not task.done():
            if server.started:
                try:
                    await asyncio.to_thread(_probe_ready, address)
                    ready = True
                    break
                except (OSError, URLError):
                    pass
            await asyncio.sleep(0.1)
        if not ready:
            if task.done():
                await task
            raise RuntimeError("本地服务未能在 20 秒内就绪。请查看上方服务日志。")
        print(f"科研工作台已就绪：{address}", flush=True)
        if startup.demo:
            print("演示模式：当前使用 Fake 模型，仅演示工程流程，不构成科研验证。", flush=True)
        if not no_browser:
            webbrowser.open(address)
        await task
    finally:
        if not task.done():
            server.should_exit = True
            await task


def _probe_ready(address: str) -> None:
    with urlopen(address, timeout=1) as response:
        if response.status != 200:
            raise URLError(f"unexpected status: {response.status}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动本机 PhotomatAgent 科研工作台")
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="科研工作区，默认当前目录")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="本地端口，默认 8765")
    parser.add_argument("--no-browser", action="store_true", help="服务就绪后不自动打开浏览器")
    parser.add_argument("--demo", action="store_true", help="显式使用 Fake 模型的工程演示模式")
    parser.add_argument("--check", action="store_true", help="只检查依赖、配置、界面和端口，不启动服务")
    parser.add_argument("--ui-dir", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if shutil.which("uv") is None:
            raise PreflightError("找不到 uv。请先安装 uv，并从项目目录重新启动工作台。")
        startup = preflight(
            workspace=args.workspace,
            port=args.port,
            demo=args.demo,
            webui_directory=args.ui_dir,
        )
    except PreflightError as error:
        print(f"启动检查未通过：{error}", file=sys.stderr)
        return 2

    print("启动检查通过：Python 3.12、Web 依赖、构建界面和本地端口。")
    if startup.demo:
        print("演示模式：Fake 模型（仅工程演示，不代表科研验证）。")
    else:
        print(f"模型配置已就绪：{startup.provider}（具体配置值不会显示）。")
    if args.check:
        return 0

    _load_runtime_environment(startup)
    try:
        asyncio.run(_serve(startup, no_browser=args.no_browser))
    except KeyboardInterrupt:
        print("工作台已停止。")
    except Exception as error:
        print(f"工作台启动失败：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
